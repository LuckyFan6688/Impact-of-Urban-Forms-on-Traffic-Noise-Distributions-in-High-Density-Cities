"""Reconcile Appendix Table 4 against the raw XGBoost / SHAP result tables.

Purpose
-------
Rebuild every cell of Appendix Table 4 from the two raw output files
(``2km_cross-validation results.csv`` and ``shap_Stability Results.csv``),
then audit the reported table cell by cell.

Outputs (written to ``results/``)
---------------------------------
appendix_table_4_from_raw.csv        Appendix table rebuilt from raw outputs
appendix_table_4_audit.csv           Cell-level comparison: reported vs rebuilt
2km_cross-validation results_v2.csv  Same numbers, explicit scheme/target labels
shap_Stability Results_with_scheme.csv  Stability table with the missing scheme column

Metric provenance (verified against spatial_cross_validation.py)
---------------------------------------------------------------
Spatial R2          <- pooled_R2 of scheme UniformSpatial2km
Buffered R2         <- pooled_R2 of scheme BufferedUniform2km
Spatial RMSE dBA    <- pooled_RMSE of scheme UniformSpatial2km
Mean rho            <- spearman_mean of the Spatial5 stability row
Top-5 overlap       <- top5_overlap_mean of the Spatial5 stability row
Random R2           <- NOT PRESENT: the 2 km file contains no Random5 scheme

Target label mapping (verified arithmetically from attenuation_intervals.py:
N_HM = 42.189 m, N_ML = 80.410 m, Delta_H = Lmax - L_break_high, etc.)

Usage: python reconcile_appendix_table_4.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'

CV_FILE = RESULTS / '2km_cross-validation results.csv'
STAB_FILE = RESULTS / 'shap_Stability Results.csv'
REPORTED_FILE = RESULTS / 'appendix_table_4.csv'

CITY_ORDER = ['Beijing', 'Tianjin', 'Shijiazhuang']
TARGET_MAP = {                      # raw target -> manuscript label
    'Lave': 'Lave', 'L10': 'L10', 'L50': 'L50', 'L90': 'L90',
    'Delta_H': 'Lmax-L42', 'Delta_M': 'L42-L80', 'Delta_L': 'L81-Lmin',
}
TARGET_ORDER = list(TARGET_MAP.values())
SCHEME_MAP = {'UniformSpatial2km': 'Spatial', 'BufferedUniform2km': 'Buffered'}
SCHEME_ORDER = ['Spatial5', 'BufferedSpatial5']   # stability row order per target

# Reported values carry two decimals, so 0.005 is the硬 rounding floor. RMSE gets a
# wider tolerance because dBA values are quoted to 2 dp with ordinary rounding noise.
TOL = {'R2': (0.005, 0.015), 'RMSE': (0.005, 0.05)}
METRIC_KIND = {'Random R2': 'R2', 'Spatial R2': 'R2', 'Buffered R2': 'R2',
               'Spatial RMSE dBA': 'RMSE', 'Mean rho': 'R2', 'Top-5 overlap': 'R2'}


def load():
    cv = pd.read_csv(CV_FILE, encoding='utf-8-sig')
    stab = pd.read_csv(STAB_FILE, encoding='utf-8-sig')
    reported = pd.read_csv(REPORTED_FILE, encoding='utf-8-sig')
    return cv, stab, reported


def tag_stability_scheme(stab):
    """Assign the missing scheme column.

    evaluate_city() appends stability rows in dict order, i.e. Spatial5 first,
    BufferedSpatial5 second, for every (city, target). Row order is preserved.
    """
    stab = stab.copy()
    stab['scheme'] = [SCHEME_ORDER[i % 2] for i in range(len(stab))]
    ordered = ['city', 'analysis', 'target', 'scheme'] + [
        c for c in stab.columns if c not in ('city', 'analysis', 'target', 'scheme')]
    return stab[ordered]


def rebuild(cv, stab):
    rows = []
    for city in CITY_ORDER:
        for label in TARGET_ORDER:
            target = next(k for k, v in TARGET_MAP.items() if v == label)
            sub = cv[(cv.city == city) & (cv.target == target)]
            sp = sub[sub.scheme == 'UniformSpatial2km'].iloc[0]
            bf = sub[sub.scheme == 'BufferedUniform2km'].iloc[0]
            st = stab[(stab.city == city) & (stab.target == target)
                      & (stab.scheme == 'Spatial5')].iloc[0]
            rows.append({
                'City': city,
                'Response variable': label,
                # No Random5 scheme in the 2 km file: value stays empty by design.
                'Random R2': np.nan,
                'Spatial R2': round(float(sp.pooled_R2), 2),
                'Buffered R2': round(float(bf.pooled_R2), 2),
                'Spatial RMSE dBA': round(float(sp.pooled_RMSE), 2),
                'Mean rho': round(float(st.spearman_mean), 2),
                'Top-5 overlap': round(float(st.top5_overlap_mean), 2),
                'Random R2 status': 'absent from 2 km CV output (no Random5 scheme)',
            })
    return pd.DataFrame(rows)


def audit(reported, rebuilt):
    metrics = ['Random R2', 'Spatial R2', 'Buffered R2', 'Spatial RMSE dBA',
               'Mean rho', 'Top-5 overlap']
    out = []
    for _, rep in reported.iterrows():
        raw = rebuilt[(rebuilt.City == rep.City)
                      & (rebuilt['Response variable'] == rep['Response variable'])].iloc[0]
        for m in metrics:
            rv, iv = rep[m], raw[m]
            if pd.isna(iv) or pd.isna(rv):
                diff, status = np.nan, 'no source in raw outputs'
            else:
                diff = round(float(rv) - float(iv), 4)
                match_tol, near_tol = TOL[METRIC_KIND[m]]
                if abs(diff) <= match_tol:
                    status = 'match'
                elif abs(diff) <= near_tol:
                    status = 'borderline'
                else:
                    status = 'MATERIAL'
            out.append({'City': rep.City, 'Response variable': rep['Response variable'],
                        'Metric': m, 'Reported': rv, 'From raw outputs': iv,
                        'Difference': diff, 'Status': status})
    return pd.DataFrame(out)


def monotonicity(reported, cv):
    """Buffering must not improve R2: flag violations in the reported table."""
    viol = reported[reported['Buffered R2'] > reported['Spatial R2']]
    raw = []
    for city in CITY_ORDER:
        for label in TARGET_ORDER:
            target = next(k for k, v in TARGET_MAP.items() if v == label)
            sub = cv[(cv.city == city) & (cv.target == target)]
            sp = float(sub[sub.scheme == 'UniformSpatial2km'].pooled_R2.iloc[0])
            bf = float(sub[sub.scheme == 'BufferedUniform2km'].pooled_R2.iloc[0])
            raw.append((city, label, bf > sp))
    return viol[['City', 'Response variable', 'Spatial R2', 'Buffered R2']], sum(r[2] for r in raw)


def align_cv(cv):
    out = cv.copy()
    out.insert(out.columns.get_loc('target') + 1, 'target_label', out.target.map(TARGET_MAP))
    out.insert(out.columns.get_loc('scheme') + 1, 'scheme_label', out.scheme.map(SCHEME_MAP))
    return out


def main():
    cv, stab, reported = load()
    stab_tagged = tag_stability_scheme(stab)
    rebuilt = rebuild(cv, stab_tagged)
    audit_df = audit(reported, rebuilt)
    cv_v2 = align_cv(cv)

    rebuilt.to_csv(RESULTS / 'appendix_table_4_from_raw.csv', index=False)
    audit_df.to_csv(RESULTS / 'appendix_table_4_audit.csv', index=False)
    cv_v2.to_csv(RESULTS / '2km_cross-validation results_v2.csv', index=False)
    stab_tagged.to_csv(RESULTS / 'shap_Stability Results_with_scheme.csv', index=False)

    print('=== audit summary (21 rows x 6 metrics = 126 cells) ===')
    print(audit_df.Status.value_counts().to_string())
    print('\n=== per metric ===')
    print(audit_df.groupby(['Metric', 'Status']).size().unstack(fill_value=0).to_string())
    print('\n=== MATERIAL cells ===')
    mat = audit_df[audit_df.Status == 'MATERIAL']
    print(mat.to_string(index=False) if len(mat) else 'none')
    print('\n=== borderline cells ===')
    bor = audit_df[audit_df.Status == 'borderline']
    print(bor.to_string(index=False) if len(bor) else 'none')
    viol, n_raw_viol = monotonicity(reported, cv)
    print(f'\n=== reported Buffered R2 > Spatial R2: {len(viol)} row(s) ===')
    print(viol.to_string(index=False) if len(viol) else 'none')
    print(f'=== raw outputs Buffered R2 > Spatial R2: {n_raw_viol} / 21 rows ===')


if __name__ == '__main__':
    main()
