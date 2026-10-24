"""Spatial attenuation intervals with absolute maximum/minimum endpoints."""
from __future__ import annotations
import numpy as np
from noise_indicators import spatial_ln, valid_values

N_HM = 42.18899750814644
N_ML = 80.41019743147403
BREAK_HIGH_EXCEEDANCE = N_HM
BREAK_LOW_EXCEEDANCE = N_ML
ENDPOINT_COLUMNS = ['Lmax', 'L_break_high', 'L_break_low', 'Lmin']
DIFF_COLUMNS = ['D1', 'D2', 'D3', 'A_range', 'r1', 'r2', 'r3']


def attenuation_targets(lmax, level_high, level_low, lmin):
    """Return max-L42.18899750814644, L42...-L80..., and L80...-min.

    All four levels must be supplied directly; missing levels are never
    interpolated. The outer endpoints are the absolute maximum and minimum of
    the analysis-unit noise pixels. They are not Q95/Q05.
    """
    arrays = np.broadcast_arrays(*[np.asarray(x, dtype=float)
        for x in (lmax, level_high, level_low, lmin)])
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError('Missing response endpoint; responses are not imputed.')
    upper, high, low, lower = arrays
    if np.any(upper < high) or np.any(high < low) or np.any(low < lower):
        raise ValueError('Require Lmax >= L_break_high >= L_break_low >= Lmin.')
    return {'Delta_H': upper - high, 'Delta_M': high - low, 'Delta_L': low - lower}


def prepare_responses(data):
    """Derive the attenuation responses and reject inconsistent supplied values."""
    missing = set(ENDPOINT_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f'Missing absolute-endpoint input columns: {sorted(missing)}')
    result = data.copy()
    derived = attenuation_targets(*[result[c].to_numpy(float) for c in ENDPOINT_COLUMNS])
    for name, value in derived.items():
        if name in result and not np.allclose(result[name].to_numpy(float), value,
                                              rtol=0, atol=1e-10, equal_nan=False):
            raise ValueError(f'Supplied {name} differs from the absolute-endpoint calculation.')
        result[name] = value
    return result


def interval_features(values, n_hm=N_HM, n_ml=N_ML):
    """Seven clustering features using absolute maximum/minimum endpoints.

    D1=Lmax-L(n_hm), D2=L(n_hm)-L(n_ml), D3=L(n_ml)-Lmin.
    A_range=Lmax-Lmin and r1..r3 are the corresponding shares of A_range.
    """
    if not 0 < n_hm < n_ml < 100:
        raise ValueError("Require 0 < n_hm < n_ml < 100")
    a = valid_values(values)
    if not len(a):
        raise ValueError("No valid pixels")
    upper, lower = spatial_ln(a, n_hm), spatial_ln(a, n_ml)
    hi, lo = float(a.max()), float(a.min())
    d1, d2, d3 = hi - upper, upper - lower, lower - lo
    arange = hi - lo
    denominator = arange if arange > 1e-9 else 1e-9
    return dict(D1=d1, D2=d2, D3=d3, A_range=arange,
                r1=d1 / denominator, r2=d2 / denominator, r3=d3 / denominator,
                Lmax=hi, Lmin=lo, L_nHM=upper, L_nML=lower)


