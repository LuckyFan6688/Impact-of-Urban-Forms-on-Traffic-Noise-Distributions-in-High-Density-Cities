"""Generate the Appendix Table 4 layout from a supplied run-level summary.

This utility contains no study data. The run-level summary is an optional,
user-supplied input and is not included in the public repository. A synthetic
mode demonstrates the expected format without using empirical observations.

The output is a formatted summary, NOT independent evidence that manuscript
figures were reproduced or verified.

Examples (run from the repository root)::

    python 04_XGBoost_SHAP/build_appendix_table_4_verified.py --demo
    python 04_XGBoost_SHAP/build_appendix_table_4_verified.py \
        --comparison /path/to/private/run_summary.csv \
        --output /path/to/local/appendix_table_4_rebuilt.csv

Required input columns:
city, target, random_R2, uniform_R2, uniform_buffered_R2,
uniform_RMSE, uniform_spearman_mean, uniform_top5_overlap_mean.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CITIES = ("Beijing", "Tianjin", "Shijiazhuang")
TARGET_MAP = {
    "Lave": "Lave",
    "L10": "L10",
    "L50": "L50",
    "L90": "L90",
    "Delta_H": "Lmax-L42",
    "Delta_M": "L42-L80",
    "Delta_L": "L81-Lmin",
}
METRIC_MAP = {
    "Random R2": "random_R2",
    "Spatial R2": "uniform_R2",
    "Buffered R2": "uniform_buffered_R2",
    "Spatial RMSE dBA": "uniform_RMSE",
    "Mean rho": "uniform_spearman_mean",
    "Top-5 overlap": "uniform_top5_overlap_mean",
}
REQUIRED = ("city", "target", *METRIC_MAP.values())
PUBLIC_TABLE = HERE / "results" / "appendix_table_4.csv"


def validate_run_summary(frame: pd.DataFrame) -> None:
    """Check schema, one-to-one model coverage, and basic metric domains."""
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {', '.join(sorted(missing))}")
    if frame[list(REQUIRED)].isna().any().any():
        raise ValueError("Required run-summary fields contain missing values.")

    actual = list(zip(frame["city"], frame["target"]))
    expected = {(city, target) for city in CITIES for target in TARGET_MAP}
    if len(actual) != len(set(actual)):
        raise ValueError("Duplicate (city, target) pairs in the input.")
    if set(actual) != expected or len(frame) != len(expected):
        raise ValueError(
            f"Expected exactly {len(expected)} rows covering three cities "
            "and seven responses; check omitted or extra combinations."
        )
    numeric = frame[list(METRIC_MAP.values())].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("All summary metrics must be finite numeric values.")
    if (numeric["uniform_RMSE"] < 0).any():
        raise ValueError("RMSE must be non-negative.")
    if not numeric["uniform_spearman_mean"].between(-1, 1).all():
        raise ValueError("Spearman correlation must be between -1 and 1.")
    if not numeric["uniform_top5_overlap_mean"].between(0, 1).all():
        raise ValueError("Top-5 overlap must be between 0 and 1.")
    if (numeric[["random_R2", "uniform_R2", "uniform_buffered_R2"]] > 1).any().any():
        raise ValueError("R-squared values cannot exceed 1.")


def build_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Format the supplied 21-model summary in manuscript table order."""
    validate_run_summary(frame)
    indexed = frame.set_index(["city", "target"])
    rows = []
    for city in CITIES:
        for target, label in TARGET_MAP.items():
            source = indexed.loc[(city, target)]
            row = {"City": city, "Response variable": label}
            for output_column, source_column in METRIC_MAP.items():
                row[output_column] = round(float(source[source_column]), 2)
            rows.append(row)
    return pd.DataFrame(rows, columns=["City", "Response variable", *METRIC_MAP])


def synthetic_run_summary() -> pd.DataFrame:
    """Make artificial input data for a formatting demonstration ONLY."""
    rows = []
    for city_index, city in enumerate(CITIES):
        for target_index, target in enumerate(TARGET_MAP):
            # Deterministic invented numbers, not sampled from research outputs.
            base = 0.70 - 0.04 * city_index - 0.02 * target_index
            rows.append({
                "city": city,
                "target": target,
                "random_R2": base + 0.04,
                "uniform_R2": base,
                "uniform_buffered_R2": base - 0.03,
                "uniform_RMSE": 2.0 + 0.25 * target_index + 0.12 * city_index,
                "uniform_spearman_mean": 0.90 - 0.015 * target_index,
                "uniform_top5_overlap_mean": 0.90 - 0.02 * target_index,
            })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--comparison", type=Path, help="User-supplied run-level summary CSV")
    group.add_argument("--demo", action="store_true", help="Use clearly synthetic demonstration values")
    parser.add_argument("--output", type=Path, help="Local destination; never the published table")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing local output file")
    args = parser.parse_args()

    if args.demo:
        source = synthetic_run_summary()
    else:
        if not args.comparison.is_file():
            parser.error(f"Input file not found: {args.comparison}")
        source = pd.read_csv(args.comparison, encoding="utf-8-sig")

    try:
        table = build_table(source)
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))

    default_name = "synthetic_appendix_table_4_demo.csv" if args.demo else "appendix_table_4_rebuilt.csv"
    output = args.output or HERE / "outputs" / default_name
    if output.resolve() == PUBLIC_TABLE.resolve():
        parser.error("Refusing to replace results/appendix_table_4.csv. Review research results separately.")
    if output.exists() and not args.overwrite:
        parser.error(f"Output already exists: {output}. Use --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, encoding="utf-8")
    print(f"Wrote {len(table)} rows to {output}")
    if args.demo:
        print("SYNTHETIC DEMONSTRATION: these are not manuscript results.")
    else:
        print("Formatted from supplied run summary; numerical provenance is not independently verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
