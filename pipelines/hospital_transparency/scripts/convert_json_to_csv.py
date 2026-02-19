"""Convert JSON hospital transparency files to CSV and optionally upload to Drive."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

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

LOGGER = logging.getLogger(__name__)

JSON_SUFFIXES = {".json", ".ndjson"}


def flatten_record(value: Any, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dict structures into dot-notation columns."""
    items: Dict[str, Any] = {}

    if isinstance(value, dict):
        for key, nested in value.items():
            next_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
            if isinstance(nested, dict):
                items.update(flatten_record(nested, next_key, sep=sep))
            elif isinstance(nested, list):
                if all(not isinstance(item, (dict, list)) for item in nested):
                    items[next_key] = "|".join("" if item is None else str(item) for item in nested)
                else:
                    items[next_key] = json.dumps(nested, ensure_ascii=False)
            else:
                items[next_key] = nested
    elif isinstance(value, list):
        items[parent_key or "value"] = json.dumps(value, ensure_ascii=False)
    else:
        items[parent_key or "value"] = value

    return items


def _safe_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in name).strip("_") or "table"


def _match_name(name: str, include_patterns: Sequence[str], exclude_patterns: Sequence[str]) -> bool:
    includes = [pattern for pattern in include_patterns if pattern]
    excludes = [pattern for pattern in exclude_patterns if pattern]

    if includes and not any(fnmatch.fnmatch(name, pattern) for pattern in includes):
        return False
    if excludes and any(fnmatch.fnmatch(name, pattern) for pattern in excludes):
        return False
    return True


def _detect_ndjson(json_path: Path) -> bool:
    if json_path.suffix.lower() == ".ndjson":
        return True

    with json_path.open("r", encoding="utf-8") as infile:
        non_empty: List[str] = []
        for line in infile:
            stripped = line.strip()
            if stripped:
                non_empty.append(stripped)
            if len(non_empty) >= 5:
                break

    if not non_empty:
        return False

    first = non_empty[0]
    if first.startswith("["):
        return False

    parsed_lines = 0
    for line in non_empty:
        try:
            json.loads(line)
            parsed_lines += 1
        except json.JSONDecodeError:
            return False

    return parsed_lines >= 2


def _iter_ndjson(path: Path, row_cap: int | None = None) -> Iterator[Tuple[int, Dict[str, Any], str | None]]:
    row_index = 0
    with path.open("r", encoding="utf-8") as infile:
        for raw_line in infile:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                yield row_index, {}, f"line {row_index + 1}: {exc}"
                continue

            if not isinstance(payload, dict):
                payload = {"value": payload}

            yield row_index, flatten_record(payload), None
            row_index += 1
            if row_cap is not None and row_index >= row_cap:
                break


def _write_dict_rows_csv(rows: List[Dict[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row.keys()})
    if not columns:
        columns = ["value"]

    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})

    return len(rows)


def _convert_ndjson_file(json_path: Path, output_path: Path, row_cap: int | None = None) -> Tuple[int, List[str]]:
    warnings: List[str] = []
    columns: set[str] = set()

    for _, flattened, warning in _iter_ndjson(json_path, row_cap=row_cap):
        if warning:
            warnings.append(warning)
            continue
        columns.update(flattened.keys())

    if not columns:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["value"])
        return 0, warnings

    ordered_columns = sorted(columns)
    row_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=ordered_columns)
        writer.writeheader()
        for _, flattened, warning in _iter_ndjson(json_path, row_cap=row_cap):
            if warning:
                warnings.append(warning)
                continue
            writer.writerow({column: flattened.get(column) for column in ordered_columns})
            row_count += 1

    return row_count, warnings


def _extract_tables(payload: Any) -> Dict[str, List[Any]]:
    if isinstance(payload, list):
        return {"records": payload}

    if isinstance(payload, dict):
        nested_tables = {
            key: value
            for key, value in payload.items()
            if isinstance(value, list)
        }
        if nested_tables:
            return nested_tables
        return {"records": [payload]}

    return {"records": [{"value": payload}]}


def _convert_standard_json_file(
    json_path: Path,
    output_dir: Path,
    row_cap: int | None = None,
) -> Tuple[List[Path], Dict[str, int], List[str]]:
    warnings: List[str] = []
    outputs: List[Path] = []
    row_counts: Dict[str, int] = {}

    if json_path.stat().st_size > 500 * 1024 * 1024:
        warnings.append(
            "Large non-NDJSON file detected. Standard JSON parsing may require significant memory."
        )

    with json_path.open("r", encoding="utf-8") as infile:
        payload = json.load(infile)

    tables = _extract_tables(payload)
    stem = json_path.stem

    for table_name, table_rows in tables.items():
        safe_table = _safe_name(table_name)
        if len(tables) == 1 and safe_table == "records":
            output_name = f"{stem}.csv"
        else:
            output_name = f"{stem}__{safe_table}.csv"

        output_path = output_dir / output_name
        limited_rows = table_rows[:row_cap] if row_cap is not None else table_rows

        flattened_rows: List[Dict[str, Any]] = []
        for row in limited_rows:
            if isinstance(row, dict):
                flattened_rows.append(flatten_record(row))
            else:
                flattened_rows.append({"value": row})

        row_count = _write_dict_rows_csv(flattened_rows, output_path)
        outputs.append(output_path)
        row_counts[output_name] = row_count

    return outputs, row_counts, warnings


def _write_conversion_log(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source_file", "output_files", "row_count", "warnings"]
    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def convert_json_to_csv(
    config: Dict[str, Any],
    include_patterns: Iterable[str] | None = None,
    exclude_patterns: Iterable[str] | None = None,
    max_files: int | None = None,
) -> List[Dict[str, Any]]:
    ensure_local_layout(config)

    include_patterns = list(include_patterns or [])
    exclude_patterns = list(exclude_patterns or [])

    raw_root = resolve_repo_path(config["download_cache_root"])
    json_dir = raw_root / "json"
    converted_dir = raw_root / "csv" / "converted_from_json"
    converted_dir.mkdir(parents=True, exist_ok=True)

    sample_mode = should_run_sample_mode(config)
    row_cap = sample_limit(config) if sample_mode else None
    file_cap = int(max_files if max_files is not None else config.get("max_files", 100))

    json_files = [
        file_path
        for file_path in sorted(json_dir.glob("*"))
        if file_path.is_file()
        and file_path.suffix.lower() in JSON_SUFFIXES
        and _match_name(file_path.name, include_patterns, exclude_patterns)
    ][:file_cap]

    upload_enabled = bool(config.get("upload_outputs_to_drive", True))
    drive_client: GDriveClient | None = None
    converted_folder_id: str | None = None
    if upload_enabled:
        drive_client = GDriveClient()
        folder_ids = resolve_drive_folder_ids(drive_client, config)
        converted_folder_id = folder_ids["converted_csv"]

    log_rows: List[Dict[str, Any]] = []

    for json_path in tqdm(json_files, desc="Converting JSON", unit="file"):
        warnings: List[str] = []
        output_paths: List[Path] = []
        total_rows = 0

        if _detect_ndjson(json_path):
            output_path = converted_dir / f"{json_path.stem}.csv"
            rows_written, ndjson_warnings = _convert_ndjson_file(json_path, output_path, row_cap=row_cap)
            total_rows += rows_written
            warnings.extend(ndjson_warnings)
            output_paths.append(output_path)
        else:
            outputs, row_counts, standard_warnings = _convert_standard_json_file(
                json_path,
                converted_dir,
                row_cap=row_cap,
            )
            output_paths.extend(outputs)
            total_rows += sum(row_counts.values())
            warnings.extend(standard_warnings)

        upload_ok = True
        if upload_enabled and drive_client and converted_folder_id:
            for output_path in output_paths:
                try:
                    drive_client.upload_file(
                        output_path,
                        converted_folder_id,
                        mime_type="text/csv",
                        overwrite=True,
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    upload_ok = False
                    warning = f"upload failed for {output_path.name}: {exc}"
                    LOGGER.exception(warning)
                    warnings.append(warning)

        if (
            output_paths
            and upload_ok
            and upload_enabled
            and bool(config.get("delete_local_cache_after_upload", False))
        ):
            for output_path in output_paths:
                if output_path.exists():
                    output_path.unlink()

        log_rows.append(
            {
                "source_file": json_path.name,
                "output_files": ";".join(path.name for path in output_paths),
                "row_count": total_rows,
                "warnings": " | ".join(dict.fromkeys(warnings)),
            }
        )

    conversion_log = metadata_file(config, "conversion_log.csv")
    _write_conversion_log(log_rows, conversion_log)
    LOGGER.info("Wrote conversion log: %s", conversion_log)

    return log_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert cached JSON files to CSV")
    parser.add_argument("--config", default="pipelines/hospital_transparency/config.yml")
    parser.add_argument("--include", action="append", default=[], help="Glob include pattern (repeatable)")
    parser.add_argument("--exclude", action="append", default=[], help="Glob exclude pattern (repeatable)")
    parser.add_argument("--max-files", type=int, default=None)
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

    convert_json_to_csv(
        config=config,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
