"""Check the published Appendix Table 4 structure and optionally compare values.

The script works with the public appendix_table_4.csv alone. In that mode it
checks completeness, duplicate keys and metric domains, but does NOT establish
that the reported values were reproduced from the original study data.

For an independent numerical comparison, optionally pass a *separately derived*
summary table with the same eight-column layout. Private run-level inputs,
training data and model outputs are never required for the default check.

Examples (from the repository root)::

    python 04_XGBoost_SHAP/reconcile_appendix_table_4.py
    python 04_XGBoost_SHAP/reconcile_appendix_table_4.py \
        --reference /path/to/local/appendix_table_4_rebuilt.csv \
        --audit-out /path/to/local/table_comparison.csv

Do not use the published CSV itself as an independent reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_REPORTED = HERE / "results" / "appendix_table_4.csv"
CITIES = ("Beijing", "Tianjin", "Shijiazhuang")
TARGETS = ("Lave", "L10", "L50", "L90", "Lmax-L42", "L42-L80", "L81-Lmin")
KEYS = ("City", "Response variable")
METRICS = (
    "Random R2", "Spatial R2", "Buffered R2",
    "Spatial RMSE dBA", "Mean rho", "Top-5 overlap",
)
REQUIRED = (*KEYS, *METRICS)


def load_and_validate(path: Path) -> pd.DataFrame:
    """Validate the published table's schema and basic statistical domains."""
    if not path.is_file():
        raise ValueError(f"CSV not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing table columns: {', '.join(sorted(missing))}")
    frame = frame[list(REQUIRED)].copy()
    if frame[list(KEYS)].isna().any().any():
        raise ValueError("City and response labels cannot be missing.")
    if frame.duplicated(list(KEYS)).any():
        raise ValueError("Duplicate city/response pairs.")
    expected = {(city, target) for city in CITIES for target in TARGETS}
    observed = set(zip(frame["City"], frame["Response variable"]))
    if observed != expected or len(frame) != len(expected):
        raise ValueError("Expected exactly three cities x seven manuscript responses (21 rows).")
    for name in METRICS:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    vals = frame[list(METRICS)].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError("All 21 x 6 reported metrics must be finite numbers.")
    if (frame["Spatial RMSE dBA"] < 0).any():
        raise ValueError("RMSE cannot be negative.")
    if not frame["Mean rho"].between(-1, 1).all():
        raise ValueError("Mean rho is outside [-1, 1].")
    if not frame["Top-5 overlap"].between(0, 1).all():
        raise ValueError("Top-5 overlap is outside [0, 1].")
    if (frame[["Random R2", "Spatial R2", "Buffered R2"]] > 1).any().any():
        raise ValueError("R-squared cannot exceed 1.")
    return frame


def compare_tables(reported: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Compare corresponding displayed statistics without changing either source."""
    left = reported.set_index(list(KEYS))
    right = reference.set_index(list(KEYS))
    records = []
    for city in CITIES:
        for target in TARGETS:
            for metric in METRICS:
                pub = float(left.loc[(city, target), metric])
                ref = float(right.loc[(city, target), metric])
                # The published table reports two decimal places. Compare to
                # independently derived values rounded to that precision.
                delta = pub - round(ref, 2)
                records.append({
                    "City": city,
                    "Response variable": target,
                    "Metric": metric,
                    "Published": pub,
                    "Reference": ref,
                    "Difference_at_reported_precision": round(delta, 6),
                    "Status": "match" if abs(delta) < 1e-8 else "different",
                })
    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reported", type=Path, default=DEFAULT_REPORTED,
                        help="Public Appendix Table 4 CSV (default: results/appendix_table_4.csv)")
    parser.add_argument("--reference", type=Path,
                        help="Optional independently calculated eight-column summary CSV")
    parser.add_argument("--audit-out", type=Path,
                        help="Optional LOCAL destination for comparison report")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing local report")
    args = parser.parse_args()
    if args.audit_out and not args.reference:
        parser.error("--audit-out requires --reference.")
    if args.reference and args.reference.resolve() == args.reported.resolve():
        parser.error("Reported and reference must be different files.")

    try:
        published = load_and_validate(args.reported)
        print("Public table structure: PASS (21 unique city/response rows; six numeric metrics).")
        print("This structural check does NOT independently verify the published results.")
        if not args.reference:
            return 0
        reference = load_and_validate(args.reference)
    except (ValueError, pd.errors.ParserError) as exc:
        parser.error(str(exc))

    audit = compare_tables(published, reference)
    matches = (audit["Status"] == "match").sum()
    print(f"Comparison against supplied reference: {matches}/{len(audit)} matching values.")
    if matches != len(audit):
        print(audit.loc[audit["Status"] != "match"].to_string(index=False))
    if args.audit_out:
        out = args.audit_out
        if out.resolve() in (args.reported.resolve(), args.reference.resolve()):
            parser.error("Refusing to overwrite an input CSV with the comparison report.")
        if out.exists() and not args.overwrite:
            parser.error(f"Comparison report already exists: {out}; use --overwrite.")
        out.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(out, index=False)
        print(f"Comparison report saved locally: {out}")
    return 0 if matches == len(audit) else 1


if __name__ == "__main__":
    raise SystemExit(main())
