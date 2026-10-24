"""Run spatial statistics and two-stage clustering on wholly synthetic data."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import yaml

from kmeans_clustering import TYPE_NAMES, classify_four_types, interval_discovery
from moran_analysis import global_moran, local_moran, queen_neighbors, row_weights
import sys

INDICATORS = Path(__file__).resolve().parents[1] / "02_Indicator_Calculation"
if str(INDICATORS) not in sys.path:
    sys.path.insert(0, str(INDICATORS))
from attenuation_intervals import N_HM, N_ML, interval_features
from noise_indicators import spatial_metrics

ROOT = Path(__file__).resolve().parent


def generate(seed=42, rows=12, columns=12, pixels_per_unit=128):
    """Fictional grid and artificial pixel levels, without a geographic CRS."""
    rng = np.random.default_rng(seed)
    profiles = np.array([[17., 5., 5.], [7., 15., 6.], [6., 5., 17.], [5., 3., 3.]])
    u = np.linspace(0, 1, pixels_per_unit)
    cells, pixels = [], []
    for row in range(rows):
        for col in range(columns):
            group = int(row >= rows // 2) * 2 + int(col >= columns // 2)
            d1, d2, d3 = profiles[group] * rng.uniform(.92, 1.08, size=3)
            base = 35 + .3 * row + .2 * col + rng.normal(0, .15)
            a = base + np.interp(u, [0, 1-N_ML/100, 1-N_HM/100, 1],
                                 [0, d3, d3+d2, d3+d2+d1])
            pixels.append(a.astype(np.float32))
            cells.append(dict(synthetic_index=len(cells), fictional_row=row,
                              fictional_col=col, abstract_x=col+.5, abstract_y=row+.5))
    # A three-population artificial set separately demonstrates interval discovery.
    interval_pixels = []
    for group in range(3):
        for _ in range(20):
            a = np.linspace(25+group*18, 31+group*22, pixels_per_unit)
            interval_pixels.append((a+rng.normal(0, .05, pixels_per_unit)).astype(np.float32))
    return pd.DataFrame(cells), np.asarray(pixels), np.asarray(interval_pixels)


def write_inputs(config, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cfg = config["example"]
    grid, pixels, intervals = generate(cfg["seed"], cfg["grid_rows"],
                                      cfg["grid_columns"], cfg["pixels_per_unit"])
    grid.to_csv(directory/"synthetic_grid.csv", index=False)
    np.save(directory/"synthetic_pixel_levels.npy", pixels, allow_pickle=False)
    np.save(directory/"synthetic_interval_pixels.npy", intervals, allow_pickle=False)


def save_plot(fig, stem):
    fig.savefig(str(stem)+".png", dpi=300, facecolor="white",
                metadata={"Software": "Synthetic example"})
    fig.savefig(str(stem)+".svg", facecolor="white",
                metadata={"Creator": "Synthetic example", "Date": None})
    plt.close(fig)


def plot_diagnostics(interval_table, type_table, output):
    with plt.rc_context({"font.size": 10, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
        for col, (table, name, selected) in enumerate([
            (interval_table, "Interval discovery", 3), (type_table, "Four-type clustering", 4)
        ]):
            axes[0, col].plot(table.k, table.SSE, "o-", color="#0072B2")
            axes[1, col].plot(table.k, table.silhouette, "s--", color="#D55E00")
            axes[0, col].set(title=f"{name}: SSE", ylabel="Within-cluster SSE", ylim=(0, None))
            axes[1, col].set(title=f"{name}: silhouette", ylabel="Mean silhouette", ylim=(-1, 1))
            for ax in axes[:, col]:
                ax.axvline(selected, color="#666666", ls=":", label=f"Configured k={selected}")
                ax.set(xlabel="Number of clusters", xticks=table.k)
                ax.legend(frameon=False, fontsize=8)
        fig.suptitle("Synthetic clustering diagnostics")
        save_plot(fig, output/"synthetic_clustering_diagnostics")


def plot_grid(metrics, labels, grid, output):
    rows = int(grid.fictional_row.max())+1
    columns = int(grid.fictional_col.max())+1
    with plt.rc_context({"font.size": 10, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        image = axes[0].imshow(metrics.Lave.to_numpy().reshape(rows, columns),
                               origin="lower", cmap="viridis", interpolation="nearest")
        fig.colorbar(image, ax=axes[0], label="Artificial spatial Lave (dB)")
        axes[0].set(title="Synthetic noise levels")
        values = labels.str[1:].astype(int).to_numpy().reshape(rows, columns)
        cmap = ListedColormap(["#0072B2", "#E69F00", "#009E73", "#CC79A7"])
        image = axes[1].imshow(values, origin="lower", cmap=cmap,
                               norm=BoundaryNorm([.5, 1.5, 2.5, 3.5, 4.5], 4),
                               interpolation="nearest")
        fig.colorbar(image, ax=axes[1], ticks=[1, 2, 3, 4], label="Synthetic type (T1-T4)")
        axes[1].set(title="Synthetic K-means types")
        for ax in axes:
            ax.set(xlabel="Fictional column", ylabel="Fictional row")
        save_plot(fig, output/"synthetic_grid_and_types")


def run(input_directory=None, output_directory=None, config_path=None, generate_inputs=False):
    config = yaml.safe_load(Path(config_path or ROOT/"clustering_config.yaml").read_text(encoding="utf-8"))
    input_directory = Path(input_directory or ROOT/"example_data")
    output = Path(output_directory or input_directory/"output")
    output.mkdir(parents=True, exist_ok=True)
    if generate_inputs:
        write_inputs(config, input_directory)
    cells = pd.read_csv(input_directory/"synthetic_grid.csv")
    pixels = np.load(input_directory/"synthetic_pixel_levels.npy", allow_pickle=False)
    interval_pixels = np.load(input_directory/"synthetic_interval_pixels.npy", allow_pickle=False)
    if pixels.ndim != 2 or len(pixels) != len(cells) or cells.synthetic_index.tolist() != list(range(len(cells))):
        raise ValueError("Pixel rows and ordered synthetic grid indices must match")
    if cells.duplicated(["fictional_row", "fictional_col"]).any():
        raise ValueError("Synthetic grid cells must be unique")
    noise_cfg, moran_cfg, cluster_cfg = config["noise"], config["moran"], config["clustering"]
    metrics = pd.DataFrame([spatial_metrics(a) for a in pixels])
    differences = pd.DataFrame([interval_features(a, noise_cfg["n_hm"], noise_cfg["n_ml"])
                                for a in pixels])
    metrics.to_csv(output/"synthetic_spatial_metrics.csv", index=False)
    differences.to_csv(output/"synthetic_interval_features.csv", index=False)
    polygons = [[(c, r), (c+1, r), (c+1, r+1), (c, r+1)]
                for r, c in cells[["fictional_row", "fictional_col"]].to_numpy()]
    weights, islands = row_weights(queen_neighbors(polygons),
                                    cells[["abstract_x", "abstract_y"]], moran_cfg["island_k"])
    seed = moran_cfg["synthetic_seed"]
    global_results = []
    for column in ["Lave"]+[f"L{n}" for n in range(10, 100, 10)]:
        result = global_moran(metrics[column], weights, moran_cfg["permutations"], seed)
        global_results.append(dict(variable=column, **result))
    pd.DataFrame(global_results).to_csv(output/"synthetic_global_moran.csv", index=False)
    local = pd.DataFrame(local_moran(metrics.Lave, weights, moran_cfg["permutations"],
                                    seed, moran_cfg["alpha"]))
    local.insert(0, "synthetic_index", cells.synthetic_index)
    local.to_csv(output/"synthetic_local_moran_Lave.csv", index=False)
    kwargs = dict(random_state=cluster_cfg["random_state"], n_init=cluster_cfg["n_init"],
                  k_min=cluster_cfg["diagnostic_k_min"], k_max=cluster_cfg["diagnostic_k_max"])
    labels, centers, type_diagnostics = classify_four_types(
        differences, n_clusters=cluster_cfg["four_type_k"], **kwargs)
    selection = cluster_cfg["interval_selection"]
    interval_metrics = pd.DataFrame([spatial_metrics(a) for a in interval_pixels])
    interval_fit, interval_diagnostics = interval_discovery(
        interval_metrics, preferred_k=selection["preferred_k"], minimum_k=selection["minimum_k"],
        required_k=selection["required_interval_k"],
        silhouette_tolerance=selection["silhouette_tolerance"], **kwargs)
    counts = labels.value_counts().sort_index().rename_axis("type").reset_index(name="count")
    counts["type_name"] = counts.type.map(TYPE_NAMES)
    counts["percentage"] = counts["count"]*100/len(cells)
    counts.to_csv(output/"synthetic_type_counts.csv", index=False)
    centers.to_csv(output/"synthetic_type_centers.csv", index=False)
    type_diagnostics.to_csv(output/"synthetic_type_diagnostics.csv", index=False)
    interval_diagnostics.to_csv(output/"synthetic_interval_diagnostics.csv", index=False)
    (output/"synthetic_interval_fit.json").write_text(json.dumps(interval_fit, indent=2), encoding="utf-8")
    plot_diagnostics(interval_diagnostics, type_diagnostics, output)
    plot_grid(metrics, labels, cells, output)
    summary = dict(purpose="Synthetic software demonstration; not manuscript observations or validation results",
                   coordinate_system=config["example"]["coordinate_system"],
                   synthetic_units=len(cells), pixels_per_unit=pixels.shape[1],
                   queen_islands=len(islands), random_seed=seed)
    print(f"Module 03 synthetic example completed: {len(cells)} fictional units")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Directory containing the three synthetic input files")
    parser.add_argument("--output", type=Path, help="Directory for synthetic CSV, JSON, PNG and SVG outputs")
    parser.add_argument("--config", type=Path, help="Configuration YAML; defaults to clustering_config.yaml")
    parser.add_argument("--generate-inputs", action="store_true", help="Regenerate artificial input files from the fixed seed")
    args = parser.parse_args()
    run(args.input, args.output, args.config, args.generate_inputs)
