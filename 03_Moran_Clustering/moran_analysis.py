"""Queen-contiguity spatial weights and Global/Local Moran statistics.

Global z uses normal-theory variance. Permutation p-values use the smaller-tail
pseudo-p convention with the +1 correction. Local permutations condition on the
focal observation. These array-based routines provide reproducible analysis
without requiring a geographic-data service.
"""
from itertools import permutations as all_permutations
import numpy as np
from scipy.stats import norm


def queen_neighbors(polygon_vertices):
    """Queen contiguity = at least one identical vertex; no distance threshold.

    Accept one iterable of exterior/interior vertices per polygon. Coordinates
    must already share a CRS and vertex precision. Vertex contiguity includes
    diagonal contact; overlap alone does not define this adjacency.
    """
    owners = {}
    neighbors = {i: set() for i in range(len(polygon_vertices))}
    for i, vertices in enumerate(polygon_vertices):
        for point in set(tuple(p) for p in vertices):
            owners.setdefault(point, []).append(i)
    for indices in owners.values():
        for i in indices:
            neighbors[i].update(j for j in indices if j != i)
    return neighbors


def row_weights(neighbors, centers, island_k=2):
    """Repair initial Queen islands only, symmetrically adding k-nearest links.

    Non-island Queen links are preserved; every resulting binary row is divided
    by its row sum. Equal-distance ties are ordered
    by input row index for deterministic behavior.
    """
    points = np.asarray(centers, dtype=float)
    n = len(points)
    if n < 2 or points.shape != (n, 2) or not np.isfinite(points).all():
        raise ValueError("Need at least two finite 2D centers")
    if set(neighbors) != set(range(n)):
        raise ValueError("Adjacency keys must exactly match row indices")
    adjacency = {i: set(v) for i, v in neighbors.items()}
    islands = [i for i in range(n) if not adjacency[i]]
    k = max(1, min(int(island_k), n-1))
    for i in islands:
        squared_distance = ((points - points[i]) ** 2).sum(axis=1)
        nearest = np.lexsort((np.arange(n), squared_distance))
        for j in [j for j in nearest if j != i][:k]:
            adjacency[i].add(int(j))
            adjacency[int(j)].add(i)
    binary = np.zeros((n, n), dtype=float)
    for i, js in adjacency.items():
        if i in js or any(j < 0 or j >= n for j in js):
            raise ValueError("Self-links and out-of-range neighbors are forbidden")
        binary[i, list(js)] = 1
    weights = binary / binary.sum(axis=1, keepdims=True)
    return weights, islands


def _check(values, weights):
    y = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    n = len(y)
    if n < 3 or w.shape != (n, n) or not np.isfinite(y).all() or not np.isfinite(w).all():
        raise ValueError("Require >=3 finite aligned observations and a square matrix")
    if np.any(w < 0) or np.any(np.diag(w) != 0) or not np.allclose(w.sum(axis=1), 1):
        raise ValueError("Weights must be nonnegative row-standardized with no diagonal")
    z = y - y.mean()
    if z @ z == 0:
        raise ValueError("Moran statistics are undefined for a constant variable")
    return z, w


def _folded_p(observed, simulations):
    above = (np.asarray(simulations) >= observed).sum(axis=0)
    count = np.minimum(above, len(simulations) - above)
    return (count + 1) / (len(simulations) + 1)


def global_moran(values, weights, permutations=999, seed=42):
    if not isinstance(permutations, (int, np.integer)) or permutations < 1:
        raise ValueError("permutations must be a positive integer")
    z, w = _check(values, weights)
    n, s0 = len(z), w.sum()
    statistic = lambda v: n / s0 * (v @ w @ v) / (v @ v)
    observed = float(statistic(z))
    expected = -1 / (n - 1)
    s1 = 0.5 * ((w + w.T) ** 2).sum()
    s2 = ((w.sum(axis=0) + w.sum(axis=1)) ** 2).sum()
    variance = ((n*n*s1 - n*s2 + 3*s0*s0) / ((n*n - 1)*s0*s0)) - expected**2
    z_norm = (observed - expected) / np.sqrt(variance) if variance > 0 else float("nan")
    rng = np.random.default_rng(seed)
    sims = np.array([statistic(rng.permutation(z)) for _ in range(permutations)])
    return dict(Moran_I=observed, expected_I=expected, variance_normal=float(variance),
                z_norm=float(z_norm), p_norm_two_sided=float(2*norm.sf(abs(z_norm))),
                p_perm_smaller_tail=float(_folded_p(observed, sims)),
                permutations=int(permutations), seed=int(seed))


def local_moran(values, weights, permutations=999, seed=42, alpha=0.05,
                exhaustive=False):
    """Local Ii=(n-1) z_i sum_j(w_ij*z_j)/sum(z^2).

    q=1 HH, 2 LH, 3 LL, 4 HL. NS if smaller-tail pseudo-p > alpha.
    z_sim=(Ii-mean(sim))/populationSD(sim). No multiplicity correction is applied
    at the nominal alpha=0.05. Exhaustive mode is for tiny tests only.
    """
    if not 0 < alpha < 1 or not isinstance(permutations, (int, np.integer)) or permutations < 1:
        raise ValueError("Require 0 < alpha < 1 and positive integer permutations")
    z, w = _check(values, weights)
    factor = (len(z)-1) / (z @ z)
    lag = w @ z
    observed = factor * z * lag
    quadrant = np.where(z > 0, np.where(lag > 0, 1, 4), np.where(lag > 0, 2, 3))
    rng = np.random.default_rng(seed)
    result = []
    for i in range(len(z)):
        js = np.flatnonzero(w[i])
        others = np.delete(z, i)
        if exhaustive:
            if len(z) > 8:
                raise ValueError("Exhaustive local permutations limited to n<=8")
            draws = np.asarray(list(all_permutations(others, len(js))))
        else:
            draws = np.asarray([rng.choice(others, size=len(js), replace=False)
                                for _ in range(permutations)])
        sims = factor * z[i] * (draws @ w[i, js])
        p = float(_folded_p(observed[i], sims))
        sd = sims.std(ddof=0)
        q = int(quadrant[i])
        result.append(dict(Ii=float(observed[i]), z_sim=float((observed[i]-sims.mean())/sd)
                           if sd > 0 else None, p_sim=p, quadrant=q,
                           category={1:"HH", 2:"LH", 3:"LL", 4:"HL"}[q]
                           if p <= alpha else "NS"))
    return result
