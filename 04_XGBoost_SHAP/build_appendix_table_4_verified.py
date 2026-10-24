"""Rebuild Appendix Table 4 from the authoritative run-level summary.

Source of truth
---------------
``spatial_cv_uniform_2km_review/data/comparison.csv``  (21 rows = 3 cities x 7 targets)
This file is produced by the run that generated the 2 km CV outputs and carries all
eight Appendix-Table-4 quantities in one place, including ``random_R2`` (Random5).
``data/uniform_summary.csv`` and ``oof_predictions.csv`` are the fold-level and
observation-level backing for the same numbers.

The 2 km cross-validation table only contains the two spatial schemes, so on its own
it cannot back the Random R2 column. comparison.csv closes that gap.

Outputs (written next to this script, in results/)
--------------------------------------------------
appendix_table_4_verified.csv   Eight-column table built only from recorded outputs
appendix_table_4_audit_full.csv Cell-level audit: reported vs recorded (126 cells)

Usage: python build_appendix_table_4_verified.py
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'

COMPARISON = Path()
CV_FILE = RESULTS / '2km_cross-validation results.csv'
STAB_FILE = RESULTS / 'shap_Stability Results.csv'
REPORTED = RESULTS / 'appendix_table_4.csv'

CITY_ORDER = ['Beijing', 'Tianjin', 'Shijiazhuang']
LABEL = {'Lave': 'Lave', 'L10': 'L10', 'L50': 'L50', 'L90': 'L90',
         'Delta_H': 'Lmax-L42', 'Delta_M': 'L42-L80', 'Delta_L': 'L81-Lmin'}
TARGET_OF = {v: k for k, v in LABEL.items()}
ORDER = list(LABEL.values())

# appendix column -> comparison.csv column
COLMAP = {
    'Random R2': 'random_R2',
    'Spatial R2': 'uniform_R2',
    'Buffered R2': 'uniform_buffered_R2',
    'Spatial RMSE dBA': 'uniform_RMSE',
    'Mean rho': 'uniform_spearman_mean',
    'Top-5 overlap': 'uniform_top5_overlap_mean',
}
TOL = {'Spatial RMSE dBA': 0.05}


def build(comp):
    rows = []
    for city in CITY_ORDER:
        for label in ORDER:
            c = comp[(comp.city == city) & (comp.target == TARGET_OF[label])].iloc[0]
            row = {'City': city, 'Response variable': label}
            for col, src in COLMAP.items():
                row[col] = round(float(c[src]), 2)
            rows.append(row)
    return pd.DataFrame(rows)


def cross_check(verified, cv, stab):
    """Prove that the rebuilt table agrees with the fold-level 2 km outputs."""
    problems = []
    for _, r in verified.iterrows():
        t = TARGET_OF[r['Response variable']]
        sub = cv[(cv.city == r.City) & (cv.target == t)]
        sp = sub[sub.scheme == 'UniformSpatial2km'].iloc[0]
        bf = sub[sub.scheme == 'BufferedUniform2km'].iloc[0]
        st = stab[(stab.city == r.City) & (stab.target == t) & (stab.scheme == 'Spatial5')].iloc[0]
        for got, want, name in [(r['Spatial R2'], sp.pooled_R2, 'Spatial R2'),
                                (r['Buffered R2'], bf.pooled_R2, 'Buffered R2'),
                                (r['Spatial RMSE dBA'], sp.pooled_RMSE, 'Spatial RMSE'),
                                (r['Mean rho'], st.spearman_mean, 'Mean rho'),
                                (r['Top-5 overlap'], st.top5_overlap_mean, 'Top-5 overlap')]:
            if abs(got - round(float(want), 2)) > 0.0051:
                problems.append((r.City, r['Response variable'], name, got, round(float(want), 4)))
    return problems


def audit(reported, verified):
    out = []
    for _, rep in reported.iterrows():
        ver = verified[(verified.City == rep.City)
                       & (verified['Response variable'] == rep['Response variable'])].iloc[0]
        for col in COLMAP:
            rv, sv = rep[col], ver[col]
            diff = round(float(rv) - float(sv), 4)
            tol = TOL.get(col, 0.005)
            status = 'match' if abs(diff) <= tol else (
                'borderline' if abs(diff) <= 0.015 else 'MATERIAL')
            out.append({'City': rep.City, 'Response variable': rep['Response variable'],
                        'Metric': col, 'Reported': rv, 'Recorded': sv,
                        'Difference': diff, 'Status': status})
    return pd.DataFrame(out)


def main():
    comp = pd.read_csv(COMPARISON, encoding='utf-8-sig')
    cv = pd.read_csv(CV_FILE, encoding='utf-8-sig')
    stab = pd.read_csv(STAB_FILE, encoding='utf-8-sig')
    stab['scheme'] = ['Spatial5', 'BufferedSpatial5'] * (len(stab) // 2)
    reported = pd.read_csv(REPORTED, encoding='utf-8-sig')

    verified = build(comp)
    # carry the reported Random column forward for comparison only
    verified.to_csv(RESULTS / 'appendix_table_4_verified.csv', index=False)

    problems = cross_check(verified, cv, stab)
    print('cross-check vs fold-level 2 km outputs: '
          + ('ALL CONSISTENT' if not problems else f'{len(problems)} mismatches'))
    for p in problems:
        print('  ', p)

    a = audit(reported, verified)
    a.to_csv(RESULTS / 'appendix_table_4_audit_full.csv', index=False)
    print('\n=== audit (21 x 6 = 126 cells) ===')
    print(a.Status.value_counts().to_string())
    print('\n=== per column ===')
    print(a.groupby(['Metric', 'Status']).size().unstack(fill_value=0).to_string())
    print('\n=== MATERIAL cells ===')
    print(a[a.Status == 'MATERIAL'].to_string(index=False))
    print('\n=== Random R2 residuals ===')
    rr = a[a.Metric == 'Random R2'][['City', 'Response variable', 'Reported', 'Recorded', 'Difference']]
    print(rr.to_string(index=False))


if __name__ == '__main__':
    main()
