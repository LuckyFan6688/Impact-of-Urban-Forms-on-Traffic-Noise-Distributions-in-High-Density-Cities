"""Held-out SHAP attributions and descriptive fold-ranking stability."""
from __future__ import annotations

import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import shap
from xgboost_regression import FEATURES, SHORT


def explain_heldout(model, xtrain, xtest, cfg):
    """Use training-only background; explain held-out raw regression/margin output.

    Multiclass attributions explain raw margins, not class probabilities.
    Additivity is checked in the same output space. Attributions represent
    model associations and do not establish causal effects.
    """
    maximum = cfg['shap_background_max']
    background = shap.utils.sample(xtrain, nsamples=min(maximum, len(xtrain)),
                                   random_state=cfg['shap_background_seed'])
    masker = shap.maskers.Independent(background, max_samples=maximum)
    explanation = shap.TreeExplainer(model, data=masker, model_output='raw',
        feature_perturbation='interventional', feature_names=FEATURES)(xtest, check_additivity=True)
    expected = (model.predict(xtest, output_margin=True) if explanation.values.ndim == 3
                else model.predict(xtest))
    error = float(np.max(np.abs(np.asarray(explanation.base_values)
        + explanation.values.sum(axis=1) - expected)))
    if error >= 0.005:
        raise AssertionError(f'SHAP additivity error {error}')
    return explanation, error


def mean_absolute_importance(explanation):
    """Equal class weighting for multiclass raw-margin attributions."""
    return (np.abs(explanation.values).mean(axis=(0, 2))
            if explanation.values.ndim == 3 else np.abs(explanation.values).mean(axis=0))


def ranking_stability(fold_importances):
    """Mean of ten fold pairs: Spearman over all 19 features and intersection/5.

    These pairwise quantities are dependent descriptive summaries; the ten
    pairs are not ten independent observations for a significance test.
    Tied values receive average ranks. Top-5 ties use fixed input-feature order.
    """
    a = np.asarray(fold_importances, dtype=float)
    if a.shape != (5, len(FEATURES)) or not np.isfinite(a).all():
        raise ValueError('Five complete finite fold attribution vectors are required.')
    top = [set(pd.Series(x).nlargest(5).index) for x in a]
    pairs = []
    for i, j in itertools.combinations(range(5), 2):
        if np.ptp(a[i]) == 0 or np.ptp(a[j]) == 0:
            rho = None  # an all-tied ranking has no defined Spearman correlation
        else:
            rho = float(spearmanr(a[i], a[j]).statistic)
        pairs.append({'fold_a': i + 1, 'fold_b': j + 1,
                      'spearman_rho': rho, 'top5_overlap': len(top[i] & top[j]) / 5})
    rho_values = [p['spearman_rho'] for p in pairs]
    return {'pair_count': 10,
            'mean_spearman_rho': (float(np.mean(rho_values))
                if all(r is not None for r in rho_values) else None),
            'undefined_rho_pairs': sum(r is None for r in rho_values),
            'mean_top5_overlap': float(np.mean([p['top5_overlap'] for p in pairs]))}, pairs


def save_summary_plot(explanation, output_stem, title='Held-out SHAP summary'):
    """Save a regression beeswarm (PNG and SVG); caller supplies output location."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if explanation.values.ndim != 2:
        raise ValueError('For multiclass plots, select and document a single class first.')
    explanation.feature_names = [SHORT[f] for f in FEATURES]
    shap.plots.beeswarm(explanation, max_display=19, show=False, plot_size=(8, 8))
    plt.title(title, fontsize=12)
    plt.tight_layout()
    for extension in ['png', 'svg']:
        plt.savefig(Path(output_stem).with_suffix('.' + extension), dpi=300, bbox_inches='tight')
    plt.close()
