"""Validate standardized CSV outputs against canonical schema requirements."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

import pandas as pd

from pipelines.hospital_transparency.scripts.pipeline_config import (
    configure_logging,
    ensure_local_layout,
    load_config,
    metadata_file,
    resolve_repo_path,
)

try:
    import polars as pl

    HAS_POLARS = True
except Exception:  # pylint: disable=broad-except
    pl = None  # type: ignore
    HAS_POLARS = False

LOGGER = logging.getLogger(__name__)


def _read_schema(schema_path: Path) -> List[Dict[str, str]]:
    with schema_path.open("r", encoding="utf-8") as infile:
        return list(csv.DictReader(infile))


def _required_columns(schema: List[Dict[str, str]]) -> List[str]:
    required: List[str] = []
    for row in schema:
        if str(row.get("required", "")).strip().lower() in {"true", "1", "yes", "y"}:
            required.append(row["canonical_column"])
    return required


def _validate_with_polars(file_path: Path, required_cols: List[str], threshold: float) -> Tuple[str, List[str], Dict[str, float], List[str]]:
    lf = pl.scan_csv(str(file_path), ignore_errors=True, truncate_ragged_lines=True)
    columns = lf.collect_schema().names()

    missing = [column for column in required_cols if column not in columns]
    warnings: List[str] = []

    non_null_pct: Dict[str, float] = {}
    for column in required_cols:
        if column not in columns:
            continue
        pct_df = lf.select((1 - pl.col(column).is_null().mean()) * 100).collect(streaming=True)
        pct_value = pct_df.item() if pct_df.height else None
        pct = (
            float(pct_value)
            if pct_value is not None and not (isinstance(pct_value, float) and math.isnan(pct_value))
            else 0.0
        )
        non_null_pct[column] = round(pct, 4)
        if pct < threshold * 100:
            warnings.append(f"{column} completeness below threshold: {pct:.2f}%")

    status = "PASS"
    if missing or warnings:
        status = "FAIL"

    return status, missing, non_null_pct, warnings


def _validate_with_pandas(file_path: Path, required_cols: List[str], threshold: float) -> Tuple[str, List[str], Dict[str, float], List[str]]:
    dataframe = pd.read_csv(file_path, dtype="string")
    columns = list(dataframe.columns)

    missing = [column for column in required_cols if column not in columns]
    warnings: List[str] = []

    non_null_pct: Dict[str, float] = {}
    for column in required_cols:
        if column not in columns:
            continue
        pct = float((1 - dataframe[column].isna().mean()) * 100)
        non_null_pct[column] = round(pct, 4)
        if pct < threshold * 100:
            warnings.append(f"{column} completeness below threshold: {pct:.2f}%")

    status = "PASS"
    if missing or warnings:
        status = "FAIL"

    return status, missing, non_null_pct, warnings


def _write_validation_results(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file_name", "status", "missing_columns", "required_non_null_pct", "warnings"]
    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_processed_outputs(config: Dict[str, Any], threshold: float = 0.5) -> Tuple[List[Dict[str, Any]], bool]:
    ensure_local_layout(config)

    processed_root = resolve_repo_path(config["local_processed_root"])
    schema_path = resolve_repo_path(config["schema_path"])

    schema = _read_schema(schema_path)
    required_cols = _required_columns(schema)

    rows: List[Dict[str, Any]] = []
    all_pass = True

    standardized_files = sorted(path for path in processed_root.glob("*.csv") if path.is_file())
    for file_path in standardized_files:
        if HAS_POLARS:
            status, missing, non_null_pct, warnings = _validate_with_polars(file_path, required_cols, threshold)
        else:
            status, missing, non_null_pct, warnings = _validate_with_pandas(file_path, required_cols, threshold)

        if status != "PASS":
            all_pass = False

        rows.append(
            {
                "file_name": file_path.name,
                "status": status,
                "missing_columns": ";".join(missing),
                "required_non_null_pct": json.dumps(non_null_pct),
                "warnings": " | ".join(warnings),
            }
        )

    if not standardized_files:
        all_pass = False
        rows.append(
            {
                "file_name": "",
                "status": "FAIL",
                "missing_columns": "",
                "required_non_null_pct": json.dumps({}),
                "warnings": "No standardized CSV files found to validate",
            }
        )

    output_path = metadata_file(config, "validation_results.csv")
    _write_validation_results(rows, output_path)
    LOGGER.info("Wrote validation results: %s", output_path)

    return rows, all_pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate standardized CSV outputs")
    parser.add_argument("--config", default="pipelines/hospital_transparency/config.yml")
    parser.add_argument("--threshold", type=float, default=0.5, help="Required-column completeness threshold")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)

    _, all_pass = validate_processed_outputs(config, threshold=args.threshold)
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
