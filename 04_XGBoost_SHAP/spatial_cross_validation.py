"""Five-fold random, spatial-block and buffered XGBoost validation.

Run ``python spatial_cross_validation.py --input city.csv --city bj --output output``.
Coordinates are analysis-unit centers in verified EPSG:32650 meters. Synthetic
examples use a separate, explicitly fictional Cartesian coordinate domain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold, KFold
from xgboost_regression import (FEATURES, SHORT, TARGETS, fit_regressor,
    predictor_frame, prepare_responses, preprocess_train_only, read_config, regression_metrics)
from xgboost_classification import fit_classifier, classification_metrics
from shap_analysis import explain_heldout, mean_absolute_importance, ranking_stability


def origin_and_blocks(xy, city_code, cfg, origin=None):
    """Freeze origin=floor(city coordinate minimum/1000)*1000 before fitting.

    Labels retain lexicographic ordering in GroupKFold. An existing fixed origin
    can be supplied for subsequent subsets; never move it using model scores.
    Analysis units are distinct from the 2000 m validation blocks.
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError('Finite n-by-2 center coordinates in meters are required.')
    step = cfg['origin_grid_m']
    origin = np.floor(xy.min(axis=0) / step) * step if origin is None else np.asarray(origin, dtype=float)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError('The fixed origin must contain two finite coordinates.')
    ab = np.floor((xy - origin) / cfg['block_m']).astype(int)
    groups = np.array([f'{city_code}_{a}_{b}_{int(cfg["block_m"])}' for a, b in ab])
    if len(set(groups)) < cfg['n_splits']:
        raise ValueError('At least five occupied spatial blocks are required; no random fallback.')
    return origin, groups


def cv_splits(xy, city_code, cfg, origin=None):
    xy = np.asarray(xy, dtype=float)
    _, groups = origin_and_blocks(xy, city_code, cfg, origin)
    for k, (train, test) in enumerate(KFold(cfg['n_splits'], shuffle=True,
            random_state=cfg['random_seed']).split(np.arange(len(xy))), 1):
        yield 'Random5', k, train, test, groups
    for k, (train, test) in enumerate(GroupKFold(cfg['n_splits']).split(xy, groups=groups), 1):
        if set(groups[train]) & set(groups[test]):
            raise AssertionError('A spatial block occurs in both training and test sets.')
        yield 'Spatial5', k, train, test, groups
        # Euclidean center-to-center distance. Equality at 1000 m is retained.
        distance = cKDTree(xy[test]).query(xy[train])[0]
        kept = train[distance >= cfg['buffer_m']]
        if len(kept) < 2:
            raise ValueError(f'Buffer leaves insufficient training rows in fold {k}; the fold cannot be dropped.')
        yield 'BufferedSpatial5', k, kept, test, groups


def evaluate_city(data, city_code, cfg, targets=TARGETS, classification=True):
    data = prepare_responses(data)
    required = ['x_m', 'y_m'] + list(targets) + (['class_label'] if classification else [])
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f'Missing required fields: {sorted(missing)}')
    xy, raw = data[['x_m', 'y_m']].to_numpy(float), predictor_frame(data)
    folds, pooled, importance, stability, pair_records = [], [], [], [], []
    split_records = list(cv_splits(xy, city_code, cfg, cfg.get('origins_m', {}).get(city_code)))
    tasks = [(t, False) for t in targets] + ([('class_label', True)] if classification else [])
    for target, is_class in tasks:
        y = data[target].to_numpy()
        if not np.isfinite(y).all():
            raise ValueError(f'Target {target} is invalid; targets are never imputed.')
        if is_class and (not np.equal(y, y.astype(int)).all() or not set(y).issubset({0, 1, 2, 3})):
            raise ValueError('Nominal labels must be integers 0 through 3.')
        collected = {s: {'y': [], 'prediction': [], 'importance': []}
                     for s in ['Random5', 'Spatial5', 'BufferedSpatial5']}
        for scheme, fold, train, test, groups in split_records:
            xtrain, xtest, _ = preprocess_train_only(raw, train, test)
            model = (fit_classifier(xtrain, y[train], cfg) if is_class
                     else fit_regressor(xtrain, y[train], target, cfg))
            prediction = model.predict_proba(xtest) if is_class else model.predict(xtest)
            metrics = classification_metrics(y[test], prediction) if is_class else regression_metrics(y[test], prediction)
            prefix = {'city': city_code, 'target': target, 'scheme': scheme}
            folds.append(dict(prefix, fold=fold, n_train=len(train), n_test=len(test),
                n_train_blocks=len(set(groups[train])), n_test_blocks=len(set(groups[test])), **metrics))
            col = collected[scheme]
            col['y'].append(y[test]); col['prediction'].append(prediction)
            if scheme != 'Random5':
                explanation, error = explain_heldout(model, xtrain, xtest, cfg)
                values = mean_absolute_importance(explanation)
                col['importance'].append(values)
                ranks = pd.Series(values).rank(ascending=False, method='average')
                for i, feature in enumerate(FEATURES):
                    importance.append(dict(prefix, fold=fold, feature=feature, abbreviation=SHORT[feature],
                        mean_abs_shap=float(values[i]), rank=float(ranks.iloc[i]), additivity_max_error=error))
        for scheme, col in collected.items():
            truth, prediction = np.concatenate(col['y']), np.concatenate(col['prediction'])
            metrics = classification_metrics(truth, prediction) if is_class else regression_metrics(truth, prediction)
            prefix = {'city': city_code, 'target': target, 'scheme': scheme}
            pooled.append(dict(prefix, n_rows=len(y), n_folds=cfg['n_splits'], **metrics))
            if col['importance']:
                summary, pairs = ranking_stability(col['importance'])
                stability.append(dict(prefix, **summary))
                pair_records.extend([dict(prefix, **pair) for pair in pairs])
    return {'fold_metrics': folds, 'pooled_metrics': pooled, 'fold_importance': importance,
            'shap_stability': stability, 'stability_pairs': pair_records}


def write_results(result, output, prefix=''):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in result.items():
        frame = pd.DataFrame(rows)
        if 'confusion_matrix' in frame:
            frame['confusion_matrix'] = frame.confusion_matrix.map(
                lambda x: json.dumps(x) if isinstance(x, list) else '')
        frame.to_csv(output / f'{prefix}{name}.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--city', choices=['bj', 'tj', 'sjz'], required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('model_config.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip-classification', action='store_true')
    args = parser.parse_args()
    cfg = read_config(args.config)
    data = pd.read_csv(args.input)
    result = evaluate_city(data, args.city, cfg, classification=not args.skip_classification)
    write_results(result, args.output)
    origin, groups = origin_and_blocks(data[['x_m', 'y_m']].to_numpy(), args.city, cfg,
        cfg.get('origins_m', {}).get(args.city))
    (args.output / 'run_config.json').write_text(json.dumps(dict(cfg,
        origin_m=origin.tolist(), occupied_blocks=len(set(groups))), indent=2), encoding='utf-8')
    print(f'Completed {args.city}: {len(result["fold_metrics"])} fold fits.')


if __name__ == '__main__':
    main()
