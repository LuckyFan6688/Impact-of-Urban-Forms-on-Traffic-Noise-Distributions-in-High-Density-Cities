"""Seven-response XGBoost regression with training-fold median imputation.

Predictors may use the original field names or the Table 1 abbreviations.
Attenuation endpoints use absolute maxima/minima; the shared implementation is
provided by Module 02. Spatial Ln denotes an exceedance percentile of spatial
noise values, rather than a fixed-point temporal sound-level statistic.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load the shared indicator algorithms from the adjacent repository module.
INDICATOR_MODULE = Path(__file__).resolve().parents[1] / '02_Indicator_Calculation'
_indicator_path_present = str(INDICATOR_MODULE) in sys.path
if not _indicator_path_present:
    sys.path.insert(0, str(INDICATOR_MODULE))
try:
    from attenuation_intervals import attenuation_targets, prepare_responses, N_HM, N_ML
finally:
    if not _indicator_path_present:
        sys.path.remove(str(INDICATOR_MODULE))

FEATURES = ['mean_height', 'build_density', 'FAR_index', 'build_edge_density',
    'build_lsi', 'road_length_index', 'road_area_density', 'road_lsi',
    'main_road_length_index', 'land_mix_entropy', 'land_mix_norm_0to1'] + [
        f'land_type_{i}' for i in range(1, 9)]
SHORT = dict(zip(FEATURES, ['ABH', 'GSI', 'FSI', 'BBD', 'ALSI', 'RLI', 'RAD',
    'RLSI', 'TRLI', 'DLM', 'DFM', 'PT', 'PG', 'PB', 'PC', 'PR', 'PI', 'PW', 'PO']))
TARGETS = ['Lave', 'L10', 'L50', 'L90', 'Delta_H', 'Delta_M', 'Delta_L']
ENDPOINT_COLUMNS = ['Lmax', 'L_break_high', 'L_break_low', 'Lmin']
BREAK_HIGH_EXCEEDANCE = N_HM
BREAK_LOW_EXCEEDANCE = N_ML


def read_config(path=Path(__file__).with_name('model_config.yaml')):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if cfg['n_splits'] != 5 or cfg['buffer_m'] != 1000:
        raise ValueError('The configured validation uses five folds and a 1000 m center buffer.')
    if cfg['block_m'] != 2000 or cfg['origin_grid_m'] != 1000:
        raise ValueError('Use uniform 2000 m blocks and the fixed 1000 m origin grid.')
    if cfg['coordinate_domain'] not in ['EPSG:32650', 'synthetic_cartesian_m']:
        raise ValueError('Supply verified UTM 50N coordinates in meters, or explicitly synthetic coordinates.')
    endpoint = cfg['response_endpoints']
    if (endpoint['mode'] != 'absolute_extrema'
        or endpoint['high_exceedance_percent'] != BREAK_HIGH_EXCEEDANCE
        or endpoint['low_exceedance_percent'] != BREAK_LOW_EXCEEDANCE):
        raise ValueError('Response endpoints require absolute extrema and full-precision internal breakpoints.')
    return cfg



def predictor_frame(data):
    """Return the fixed 19-column model order, accepting Table 1 abbreviations.

    Supplying both aliases is allowed only when their numeric contents match.
    Missing predictor values are handled later inside each training fold.
    """
    result = data.copy()
    for field, abbreviation in SHORT.items():
        if field not in result and abbreviation in result:
            result[field] = result[abbreviation]
        elif field in result and abbreviation in result:
            left = pd.to_numeric(result[field], errors='coerce').to_numpy(float)
            right = pd.to_numeric(result[abbreviation], errors='coerce').to_numpy(float)
            if not np.allclose(left, right, rtol=0, atol=1e-12, equal_nan=True):
                raise ValueError(f'Conflicting predictor aliases: {field} and {abbreviation}')
    missing = set(FEATURES) - set(result.columns)
    if missing:
        raise ValueError(f'Missing predictor fields: {sorted(missing)}')
    return result[FEATURES]


def fit_regressor(x, y, target, cfg):
    """Fit the unchanged noise-level or attenuation estimator configuration."""
    if target not in TARGETS:
        raise ValueError(f'Unknown regression response: {target}')
    kind = 'Attenuation' if target.startswith('Delta_') else 'NoiseLevel'
    params = dict(cfg['models'][kind], random_state=cfg['random_seed'],
                  n_jobs=cfg['n_jobs'], tree_method='hist')
    return XGBRegressor(**params, objective='reg:squarederror', missing=np.nan).fit(x, y)


def preprocess_train_only(raw, train, test):
    x = pd.DataFrame(raw).apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    if x.iloc[train].isna().all(axis=0).any():
        raise ValueError('A predictor has no valid training values; no silent column removal is allowed.')
    imputer = SimpleImputer(strategy='median')
    xtrain = imputer.fit_transform(x.iloc[train]).astype(np.float32)
    xtest = imputer.transform(x.iloc[test]).astype(np.float32)
    return xtrain, xtest, imputer


def regression_metrics(y, prediction):
    return {'R2': float(r2_score(y, prediction)),
            'RMSE': float(np.sqrt(mean_squared_error(y, prediction))),
            'MAE': float(mean_absolute_error(y, prediction))}
