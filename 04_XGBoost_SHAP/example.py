"""Run synthetic examples for three city interfaces, seven regressions and four classes.

This tests software behavior, not the manuscript's empirical accuracy. By default
only the example uses 12 trees of depth 3 for speed; --full-parameters uses the
unchanged 400/700/800-tree configurations in model_config.yaml.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost_regression import (FEATURES, TARGETS, attenuation_targets,
    fit_regressor, preprocess_train_only, read_config)
from spatial_cross_validation import cv_splits, evaluate_city, write_results
from shap_analysis import explain_heldout, save_summary_plot

MODULE = Path(__file__).resolve().parent


def make_synthetic(city_index=0):
    rng = np.random.default_rng(20261023 + city_index)
    i, j = np.meshgrid(np.arange(12), np.arange(12))
    n = i.size
    # Fictional Cartesian meters, unrelated to real geographic locations.
    data = pd.DataFrame({'x_m': i.ravel() * 650.0, 'y_m': j.ravel() * 650.0})
    for feature in FEATURES:
        data[feature] = rng.uniform(0.05, 0.95, n)
    data['mean_height'] *= 40
    data['FAR_index'] *= 3
    proportions = rng.dirichlet(np.ones(8), n)
    for k in range(8):
        data[f'land_type_{k + 1}'] = proportions[:, k]
    signal = 8 * data.road_area_density + 3 * data.FAR_index - 2 * data.land_type_2
    data['Lave'] = 45 + city_index + signal + rng.normal(0, 0.6, n)
    data['L10'] = data.Lave + 4 + rng.uniform(0, 2, n)
    data['L50'] = data.Lave - 0.5 + rng.normal(0, 0.2, n)
    data['L90'] = data.Lave - 5 - rng.uniform(0, 1, n)
    data['Lmin'] = data.L90 - 4 - data.land_type_4
    data['L_break_low'] = data.L90 + 1 + data.build_density
    data['L_break_high'] = data.L50 + 1 + 2 * data.road_area_density
    data['Lmax'] = data.L10 + 6 + 2 * data.build_edge_density
    for name, value in attenuation_targets(*[data[c].values for c in
            ['Lmax', 'L_break_high', 'L_break_low', 'Lmin']]).items():
        data[name] = value
    # Artificial nominal labels exercise the classifier; no real cluster labels.
    data['class_label'] = ((i + j) % 4).ravel()
    data['is_synthetic'] = True
    for feature in FEATURES:
        data.loc[rng.choice(n, 2, replace=False), feature] = np.nan
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=MODULE / 'example_data' / 'output')
    parser.add_argument('--full-parameters', action='store_true')
    parser.add_argument('--generate-inputs', action='store_true',
                        help='Regenerate the explicitly synthetic example CSVs.')
    args = parser.parse_args()
    cfg = copy.deepcopy(read_config())
    cfg['coordinate_domain'] = 'synthetic_cartesian_m'
    cfg['purpose'] = 'synthetic_example_only'
    if not args.full_parameters:
        for parameters in cfg['models'].values():
            parameters.update(n_estimators=12, max_depth=3)
    args.output.mkdir(parents=True, exist_ok=True)
    totals = {}
    for index, city in enumerate(['bj', 'tj', 'sjz']):
        path = MODULE / 'example_data' / f'SYNTHETIC_{city}.csv'
        if args.generate_inputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            make_synthetic(index).to_csv(path, index=False)
        data = pd.read_csv(path)
        if not data.is_synthetic.eq(True).all():
            raise ValueError('The example accepts only explicitly synthetic observations.')
        result = evaluate_city(data, city, cfg)
        for name, rows in result.items():
            totals.setdefault(name, []).extend(rows)
    write_results(totals, args.output, prefix='SYNTHETIC_')
    regression = pd.DataFrame([r for r in totals['pooled_metrics'] if r['target'] in TARGETS])
    classification = pd.DataFrame([r for r in totals['pooled_metrics'] if r['target'] == 'class_label'])
    regression.to_csv(args.output / 'SYNTHETIC_regression_metrics.csv', index=False)
    classification.to_csv(args.output / 'SYNTHETIC_classification_metrics.csv', index=False)
    _, _, train, test, _ = next(split for split in cv_splits(
        data[['x_m', 'y_m']].values, 'sjz', cfg) if split[0] == 'Spatial5')
    xtrain, xtest, _ = preprocess_train_only(data[FEATURES], train, test)
    model = fit_regressor(xtrain, data.Lave.values[train], 'Lave', cfg)
    explanation, _ = explain_heldout(model, xtrain, xtest, cfg)
    save_summary_plot(explanation, args.output / 'SYNTHETIC_SHAP_summary',
                      title='Synthetic example: held-out SHAP (not study results)')
    report = {'synthetic_only': True, 'empirical_results_reproduced': False,
        'cities': ['bj', 'tj', 'sjz'], 'regression_tasks': 21, 'classification_tasks': 3,
        'cv_fold_model_count': len(totals['fold_metrics']), 'illustration_model_count': 1,
        'full_parameters': args.full_parameters, 'effective_config': cfg,
        'maximum_shap_additivity_error': max(r['additivity_max_error'] for r in totals['fold_importance'])}
    (args.output / 'SYNTHETIC_run_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'effective_config'}, indent=2))


if __name__ == '__main__':
    main()
