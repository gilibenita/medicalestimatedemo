"""Standardize raw hospital CSV files into canonical schema outputs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

import pandas as pd
from tqdm import tqdm

from pipelines.hospital_transparency.scripts.gdrive_client import GDriveClient, resolve_drive_folder_ids
from pipelines.hospital_transparency.scripts.pipeline_config import (
    configure_logging,
    ensure_local_layout,
    load_config,
    metadata_file,
    resolve_repo_path,
    sample_limit,
    should_run_sample_mode,
)

try:
    import polars as pl

    HAS_POLARS = True
except Exception:  # pylint: disable=broad-except
    pl = None  # type: ignore
    HAS_POLARS = False

LOGGER = logging.getLogger(__name__)


NUMERIC_TYPES = {"float", "double", "decimal", "numeric", "number", "int", "integer"}


def normalize_column_name(name: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def _read_schema(schema_path: Path) -> List[Dict[str, str]]:
    with schema_path.open("r", encoding="utf-8") as infile:
        return list(csv.DictReader(infile))


def _read_column_mapping(mapping_path: Path) -> Dict[str, List[Tuple[str, str]]]:
    grouped: Dict[str, List[Tuple[str, str]]] = {}
    with mapping_path.open("r", encoding="utf-8") as infile:
        for row in csv.DictReader(infile):
            source = normalize_column_name(row.get("source_column", ""))
            canonical = normalize_column_name(row.get("canonical_column", ""))
            rule = (row.get("transform_rule") or "").strip()
            if not source or not canonical:
                continue
            grouped.setdefault(canonical, []).append((source, rule))
    return grouped


def _split_rules(rule: str) -> List[str]:
    return [item.strip().lower() for item in rule.split("|") if item.strip()]


def _apply_polars_rules(expr: "pl.Expr", rule: str) -> "pl.Expr":
    for token in _split_rules(rule):
        if token in {"strip", "trim"}:
            expr = expr.cast(pl.Utf8, strict=False).str.strip_chars()
        elif token == "upper":
            expr = expr.cast(pl.Utf8, strict=False).str.to_uppercase()
        elif token == "lower":
            expr = expr.cast(pl.Utf8, strict=False).str.to_lowercase()
        elif token == "to_float":
            expr = (
                expr.cast(pl.Utf8, strict=False)
                .str.replace_all(r"[$,]", "")
                .cast(pl.Float64, strict=False)
            )
    return expr


def _apply_pandas_rules(series: pd.Series, rule: str) -> pd.Series:
    result = series
    for token in _split_rules(rule):
        if token in {"strip", "trim"}:
            result = result.astype("string").str.strip()
        elif token == "upper":
            result = result.astype("string").str.upper()
        elif token == "lower":
            result = result.astype("string").str.lower()
        elif token == "to_float":
            result = pd.to_numeric(result.astype("string").str.replace(r"[$,]", "", regex=True), errors="coerce")
    return result


def _cast_polars(expr: "pl.Expr", type_name: str) -> "pl.Expr":
    normalized = type_name.strip().lower()
    if normalized in NUMERIC_TYPES:
        return (
            expr.cast(pl.Utf8, strict=False)
            .str.replace_all(r"[$,]", "")
            .cast(pl.Float64, strict=False)
        )
    return expr.cast(pl.Utf8, strict=False)


def _infer_hospital_name(file_path: Path) -> str:
    stem = file_path.stem.replace("__standardized", "")
    return stem.replace("_", " ").strip() or "unknown"


def _collect_csv_inputs(raw_root: Path, max_files: int) -> List[Path]:
    base_csv = sorted(path for path in (raw_root / "csv").glob("*.csv") if path.is_file())
    converted_csv = sorted(
        path for path in (raw_root / "csv" / "converted_from_json").glob("*.csv") if path.is_file()
    )
    return (base_csv + converted_csv)[:max_files]


def _profile_with_polars(lf: "pl.LazyFrame", canonical_columns: Sequence[str]) -> Tuple[int, List[str], Dict[str, float]]:
    row_count_df = lf.select(pl.len().alias("rows")).collect(streaming=True)
    row_count = int(row_count_df[0, "rows"]) if row_count_df.height else 0

    code_values: List[str] = []
    if "code_type" in canonical_columns:
        code_df = lf.select(pl.col("code_type").drop_nulls().unique().sort()).collect(streaming=True)
        if code_df.width:
            code_values = [str(item) for item in code_df.to_series(0).to_list()]

    null_exprs = [
        (pl.col(column).is_null().mean() * 100).alias(column)
        for column in canonical_columns
    ]
    null_df = lf.select(null_exprs).collect(streaming=True)
    percent_null = {
        column: (
            round(float(null_df[0, column]), 4)
            if null_df.height
            and null_df[0, column] is not None
            and not (isinstance(null_df[0, column], float) and math.isnan(null_df[0, column]))
            else 0.0
        )
        for column in canonical_columns
    }

    return row_count, code_values, percent_null


def _standardize_with_polars(
    source_path: Path,
    output_path: Path,
    canonical_schema: List[Dict[str, str]],
    mapping_by_canonical: Dict[str, List[Tuple[str, str]]],
    sample_mode: bool,
    max_rows: int,
) -> Tuple[int, List[str], Dict[str, float], List[str]]:
    warnings: List[str] = []

    lf = pl.scan_csv(
        str(source_path),
        ignore_errors=True,
        truncate_ragged_lines=True,
        infer_schema_length=2000,
        null_values=["", "NA", "N/A", "null", "None"],
    )

    input_columns = lf.collect_schema().names()
    rename_map = {
        column: normalize_column_name(column)
        for column in input_columns
        if normalize_column_name(column) and normalize_column_name(column) != column
    }
    if rename_map:
        lf = lf.rename(rename_map)

    normalized_columns = lf.collect_schema().names()

    canonical_columns = [normalize_column_name(item["canonical_column"]) for item in canonical_schema]
    canonical_types = {
        normalize_column_name(item["canonical_column"]): item.get("type", "string")
        for item in canonical_schema
    }

    expressions: List["pl.Expr"] = []
    for canonical in canonical_columns:
        candidates: List["pl.Expr"] = []

        if canonical in normalized_columns:
            candidates.append(pl.col(canonical))

        for source_column, transform_rule in mapping_by_canonical.get(canonical, []):
            if source_column in normalized_columns and source_column != canonical:
                transformed = _apply_polars_rules(pl.col(source_column), transform_rule)
                candidates.append(transformed)

        if candidates:
            expr: "pl.Expr" = pl.coalesce(candidates)
        elif canonical == "hospital_name":
            expr = pl.lit(_infer_hospital_name(source_path))
            warnings.append("hospital_name was missing and inferred from filename")
        elif canonical == "source_file":
            expr = pl.lit(source_path.name)
        else:
            expr = pl.lit(None)

        expr = _cast_polars(expr, canonical_types.get(canonical, "string")).alias(canonical)
        expressions.append(expr)

    standardized = lf.select(expressions)
    if sample_mode:
        standardized = standardized.limit(max_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        standardized.sink_csv(str(output_path))
    except Exception:  # pylint: disable=broad-except
        standardized.collect(streaming=True).write_csv(str(output_path))

    row_count, distinct_code_types, percent_null = _profile_with_polars(standardized, canonical_columns)
    return row_count, distinct_code_types, percent_null, warnings


def _standardize_with_pandas(
    source_path: Path,
    output_path: Path,
    canonical_schema: List[Dict[str, str]],
    mapping_by_canonical: Dict[str, List[Tuple[str, str]]],
    sample_mode: bool,
    max_rows: int,
) -> Tuple[int, List[str], Dict[str, float], List[str]]:
    warnings: List[str] = ["Polars unavailable; using pandas fallback"]

    dataframe = pd.read_csv(source_path, dtype="string", nrows=max_rows if sample_mode else None)
    dataframe.columns = [normalize_column_name(column) for column in dataframe.columns]

    standardized = pd.DataFrame()

    for field in canonical_schema:
        canonical = normalize_column_name(field["canonical_column"])
        type_name = field.get("type", "string")

        candidates: List[pd.Series] = []
        if canonical in dataframe.columns:
            candidates.append(dataframe[canonical])

        for source_column, transform_rule in mapping_by_canonical.get(canonical, []):
            if source_column in dataframe.columns and source_column != canonical:
                candidates.append(_apply_pandas_rules(dataframe[source_column], transform_rule))

        if candidates:
            merged = candidates[0]
            for candidate in candidates[1:]:
                merged = merged.combine_first(candidate)
            standardized[canonical] = merged
        elif canonical == "hospital_name":
            standardized[canonical] = _infer_hospital_name(source_path)
            warnings.append("hospital_name was missing and inferred from filename")
        elif canonical == "source_file":
            standardized[canonical] = source_path.name
        else:
            standardized[canonical] = pd.NA

        if type_name.lower() in NUMERIC_TYPES:
            standardized[canonical] = pd.to_numeric(
                standardized[canonical].astype("string").str.replace(r"[$,]", "", regex=True),
                errors="coerce",
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    standardized.to_csv(output_path, index=False)

    row_count = int(len(standardized))
    distinct_code_types = []
    if "code_type" in standardized.columns:
        distinct_code_types = sorted(
            [str(value) for value in standardized["code_type"].dropna().unique().tolist()]
        )

    percent_null = {
        column: round(float(standardized[column].isna().mean() * 100), 4)
        for column in standardized.columns
    }

    return row_count, distinct_code_types, percent_null, warnings


def _write_profile_report(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_file",
        "row_count",
        "distinct_code_types",
        "percent_null_by_column",
        "warnings",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def standardize_csv_files(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    ensure_local_layout(config)

    raw_root = resolve_repo_path(config["download_cache_root"])
    processed_root = resolve_repo_path(config["local_processed_root"])
    schema_path = resolve_repo_path(config["schema_path"])
    mapping_path = resolve_repo_path(config["column_mapping_path"])

    sample_mode = should_run_sample_mode(config)
    max_rows = sample_limit(config)
    max_files = int(config.get("max_files", 100))

    canonical_schema = _read_schema(schema_path)
    mapping_by_canonical = _read_column_mapping(mapping_path)

    csv_files = _collect_csv_inputs(raw_root, max_files=max_files)

    upload_enabled = bool(config.get("upload_outputs_to_drive", True))
    drive_client: GDriveClient | None = None
    standardized_folder_id: str | None = None
    if upload_enabled:
        drive_client = GDriveClient()
        folder_ids = resolve_drive_folder_ids(drive_client, config)
        standardized_folder_id = folder_ids["standardized"]

    profile_rows: List[Dict[str, Any]] = []
    generated_outputs: List[Path] = []
    upload_ok = True

    for source_path in tqdm(csv_files, desc="Standardizing CSV", unit="file"):
        output_path = processed_root / f"{source_path.stem}__standardized.csv"

        if HAS_POLARS:
            row_count, distinct_code_types, percent_null, warnings = _standardize_with_polars(
                source_path=source_path,
                output_path=output_path,
                canonical_schema=canonical_schema,
                mapping_by_canonical=mapping_by_canonical,
                sample_mode=sample_mode,
                max_rows=max_rows,
            )
        else:
            row_count, distinct_code_types, percent_null, warnings = _standardize_with_pandas(
                source_path=source_path,
                output_path=output_path,
                canonical_schema=canonical_schema,
                mapping_by_canonical=mapping_by_canonical,
                sample_mode=sample_mode,
                max_rows=max_rows,
            )

        generated_outputs.append(output_path)

        if upload_enabled and drive_client and standardized_folder_id:
            try:
                drive_client.upload_file(
                    output_path,
                    standardized_folder_id,
                    mime_type="text/csv",
                    overwrite=True,
                )
            except Exception as exc:  # pylint: disable=broad-except
                upload_ok = False
                warning = f"upload failed for {output_path.name}: {exc}"
                LOGGER.exception(warning)
                warnings.append(warning)

        profile_rows.append(
            {
                "source_file": source_path.name,
                "row_count": row_count,
                "distinct_code_types": json.dumps(distinct_code_types),
                "percent_null_by_column": json.dumps(percent_null),
                "warnings": " | ".join(dict.fromkeys(warnings)),
            }
        )

    profile_report_path = metadata_file(config, "profile_report.csv")
    _write_profile_report(profile_rows, profile_report_path)
    LOGGER.info("Wrote profile report: %s", profile_report_path)

    if (
        generated_outputs
        and upload_enabled
        and upload_ok
        and bool(config.get("delete_local_cache_after_upload", False))
    ):
        for output_path in generated_outputs:
            if output_path.exists():
                output_path.unlink()

    return profile_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standardize cached CSV files into canonical schema")
    parser.add_argument("--config", default="pipelines/hospital_transparency/config.yml")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)
    if args.sample:
        config["sample_mode"] = True
    if args.max_rows is not None:
        config["sample_max_rows"] = args.max_rows

    standardize_csv_files(config)


if __name__ == "__main__":
    main()
