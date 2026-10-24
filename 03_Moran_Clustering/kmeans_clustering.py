"""Two-stage standardized K-means and explicit interval/type constraints."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys

INDICATORS = Path(__file__).resolve().parents[1] / "02_Indicator_Calculation"
if str(INDICATORS) not in sys.path:
    sys.path.insert(0, str(INDICATORS))
from attenuation_intervals import DIFF_COLUMNS

TYPE_NAMES = {"T1": "High-value attenuation", "T2": "Medium-value attenuation",
              "T3": "Low-value attenuation", "T4": "Low total attenuation"}
QUANTILE_COLUMNS = [f"Q{q:02d}" for q in range(5, 100, 5)]


def diagnostics(x, k_min=2, k_max=6, random_state=42, n_init=10):
    x = np.asarray(x)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Finite 2D features required; no silent row exclusion")
    rows = []
    for k in range(k_min, min(k_max, len(x)-1)+1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init).fit(x)
        if len(np.unique(model.labels_)) < 2:
            raise ValueError("Degenerate clustering cannot have a silhouette score")
        rows.append(dict(k=k, SSE=float(model.inertia_),
                         silhouette=float(silhouette_score(x, model.labels_))))
    if not rows:
        raise ValueError("Not enough observations for diagnostics")
    return pd.DataFrame(rows)


def choose_interval_k(diag, prefer_k=3, minimum_k=3, tolerance=0.02):
    """Silhouette selection with an explicit preferred count and lower bound."""
    best = diag.loc[diag.silhouette.idxmax()]
    k = int(best.k)
    close = diag.loc[best.silhouette - diag.silhouette < tolerance]
    if len(close) > 1:
        if prefer_k in diag.k.values:
            k = int(close.iloc[np.argmin(abs(close.k.values - prefer_k))].k)
        elif len(diag) >= 3:
            s = diag.SSE.values
            normed = (s-s.min())/(s.max()-s.min()+1e-9)
            k = int(diag.iloc[1 + np.argmax(normed[:-2]-2*normed[1:-1]+normed[2:])].k)
    return max(minimum_k, k)


def _min_error_cut(a, b, n_grid):
    grid = np.linspace(float(min(a.min(), b.min())), float(max(a.max(), b.max())), n_grid)
    error = (a < grid[:, None]).mean(axis=1) + (b >= grid[:, None]).mean(axis=1)
    return float(grid[error.argmin()])


def interval_discovery(metrics, random_state=42, n_init=10,
                       k_min=2, k_max=6, preferred_k=3, minimum_k=3,
                       required_k=3, silhouette_tolerance=0.02):
    """Discover three distribution bands from 19 standardized quantiles.

    SSE/silhouette are diagnostic, with a preferred/minimum k=3 constraint.
    The threshold procedure requires three bands and fails if another count
    is selected. Record ranges use Q90-Q10; center ordering uses Q95-Q10.
    Candidate thresholds maximize in-sample agreement, not external accuracy.
    """
    x = metrics[QUANTILE_COLUMNS].to_numpy(float)
    if not np.isfinite(x).all() or len(x) < 7:
        raise ValueError("Need >=7 complete quantile vectors")
    scaler = StandardScaler()
    xn = scaler.fit_transform(x)
    diag = diagnostics(xn, k_min, k_max, random_state, n_init)
    k = choose_interval_k(diag, preferred_k, minimum_k, silhouette_tolerance)
    if required_k != 3 or k != required_k:
        raise ValueError("Three-band threshold derivation requires selected k=3")
    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init).fit(xn)
    centers = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=QUANTILE_COLUMNS)
    centers["center_ordering_range_Q95_Q10"] = centers.Q95 - centers.Q10
    order = centers.sort_values("center_ordering_range_Q95_Q10", ascending=False).index.tolist()
    mapping = dict(zip(order, ["High", "Mid", "Low"]))
    labels = pd.Series(model.labels_).map(mapping)
    arange = pd.Series(metrics.Q90.values - metrics.Q10.values)
    names = arange.groupby(labels).median().sort_values(ascending=False).index.tolist()
    groups = [arange[labels == name].values for name in names]
    midpoints = [float((np.median(groups[i])+np.median(groups[i+1]))/2) for i in range(2)]
    candidates = {"midpoint": dict(order=names, cuts=midpoints),
                  "min_error": dict(order=names, cuts=[_min_error_cut(groups[i], groups[i+1], 801)
                                                      for i in range(2)])}
    def apply(v, cuts, names):
        return np.where(v >= cuts[0], names[0], np.where(v >= cuts[1], names[1], names[2]))
    for entry in candidates.values():
        entry["agreement"] = float(np.mean(apply(arange, entry["cuts"], entry["order"]) == labels))
    lda = LinearDiscriminantAnalysis(n_components=1)
    z = lda.fit_transform(xn, labels).ravel()
    z_series = pd.Series(z)
    means = z_series.groupby(labels).mean().sort_values(ascending=False)
    lda_names = means.index.tolist()
    lda_cuts = [float((means.iloc[i]+means.iloc[i+1])/2) for i in range(2)]
    pred = apply(z, lda_cuts, lda_names)
    lda_entry = dict(order=lda_names, cuts=lda_cuts,
                     agreement=float(np.mean(pred == labels)))
    candidates["lda"] = lda_entry
    winner = max(candidates, key=lambda name: candidates[name]["agreement"])
    if winner == "lda":
        # Convert LDA-assigned classes into A_range cutpoints.
        names = arange.groupby(pred).median().sort_values(ascending=False).index.tolist()
        if len(names) != 3:
            raise ValueError("LDA conversion lost a class; cannot recover three intervals")
        groups = [arange[pred == name].values for name in names]
        cuts = [_min_error_cut(groups[i], groups[i+1], 1201) for i in range(2)]
    else:
        cuts, names = candidates[winner]["cuts"], candidates[winner]["order"]
    raw_n = [float(100*np.mean(arange >= cut)) for cut in cuts]
    return dict(method=winner, k=k, raw_cut_order=names, cuts_A_range=cuts,
                raw_equivalent_exceedance_percentages=raw_n,
                ordered_exceedance_for_second_stage=sorted(raw_n),
                selection_candidates=candidates), diag


def classify_four_types(features, random_state=42, n_init=10,
                        n_clusters=4, k_min=2, k_max=6):
    """Fit fixed four-type K-means; rank inverse-scaled centers by D1."""
    if n_clusters != 4:
        raise ValueError("The four-type classification requires n_clusters=4")
    x = features[DIFF_COLUMNS].to_numpy(np.float32)
    if not np.isfinite(x).all() or len(x) < 7:
        raise ValueError("Need >=7 complete seven-dimensional difference vectors")
    scaler = StandardScaler()
    xn = scaler.fit_transform(x)
    diag = diagnostics(xn, k_min, k_max, random_state, n_init)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init).fit(xn)
    centers = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=DIFF_COLUMNS)
    order = centers.sort_values("D1", ascending=False).index.tolist()
    names = {old: f"T{i+1}" for i, old in enumerate(order)}
    labels = pd.Series(model.labels_).map(names)
    centers["type"] = [names[i] for i in centers.index]
    centers["type_name"] = centers.type.map(TYPE_NAMES)
    return labels, centers.sort_values("type"), diag
