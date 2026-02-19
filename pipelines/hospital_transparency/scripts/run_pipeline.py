"""Orchestrate the full hospital transparency pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Iterable

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from pipelines.hospital_transparency.scripts.build_manifest import build_manifest
from pipelines.hospital_transparency.scripts.convert_json_to_csv import convert_json_to_csv
from pipelines.hospital_transparency.scripts.pipeline_config import (
    configure_logging,
    load_config,
    resolve_repo_path,
)
from pipelines.hospital_transparency.scripts.standardize_csv import standardize_csv_files
from pipelines.hospital_transparency.scripts.sync_from_drive import sync_from_drive
from pipelines.hospital_transparency.scripts.sync_reports_to_drive import sync_reports_to_drive
from pipelines.hospital_transparency.scripts.validate_processed import validate_processed_outputs

LOGGER = logging.getLogger(__name__)


def _cleanup_local_outputs(config: Dict[str, Any]) -> None:
    raw_root = resolve_repo_path(config["download_cache_root"])
    processed_root = resolve_repo_path(config["local_processed_root"])

    cleanup_dirs = [raw_root / "csv" / "converted_from_json", processed_root]
    for directory in cleanup_dirs:
        if not directory.exists():
            continue
        for file_path in directory.glob("*.csv"):
            file_path.unlink(missing_ok=True)
        LOGGER.info("Cleaned local output cache: %s", directory)


def _apply_runtime_overrides(
    config: Dict[str, Any],
    sample: bool,
    max_rows: int | None,
    max_files: int | None,
) -> Dict[str, Any]:
    runtime = dict(config)
    if sample:
        runtime["sample_mode"] = True
    if max_rows is not None:
        runtime["sample_max_rows"] = max_rows
    if max_files is not None:
        runtime["max_files"] = max_files
    return runtime


def run_pipeline(
    config: Dict[str, Any],
    only: str | None = None,
    include_patterns: Iterable[str] | None = None,
    exclude_patterns: Iterable[str] | None = None,
) -> None:
    include_patterns = list(include_patterns or [])
    exclude_patterns = list(exclude_patterns or [])

    if only == "sync":
        sync_from_drive(config, include_patterns=include_patterns, exclude_patterns=exclude_patterns)
        return
    if only == "manifest":
        build_manifest(config)
        return
    if only == "convert":
        convert_json_to_csv(config, include_patterns=include_patterns, exclude_patterns=exclude_patterns)
        return
    if only == "standardize":
        standardize_csv_files(config)
        return
    if only == "validate":
        _, is_valid = validate_processed_outputs(config)
        if not is_valid:
            raise SystemExit(1)
        return
    if only == "reports":
        sync_reports_to_drive(config)
        return

    delete_after_upload = bool(config.get("delete_local_cache_after_upload", False))
    full_run_config = dict(config)
    full_run_config["delete_local_cache_after_upload"] = False

    sync_from_drive(full_run_config, include_patterns=include_patterns, exclude_patterns=exclude_patterns)
    build_manifest(full_run_config)
    convert_json_to_csv(full_run_config)
    standardize_csv_files(full_run_config)

    _, is_valid = validate_processed_outputs(full_run_config)
    if not is_valid:
        raise SystemExit(1)

    sync_reports_to_drive(full_run_config)

    if delete_after_upload:
        _cleanup_local_outputs(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hospital transparency Drive pipeline")
    parser.add_argument("--config", default="pipelines/hospital_transparency/config.yml")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--only",
        choices=["sync", "convert", "standardize", "validate", "manifest", "reports"],
        default=None,
    )
    parser.add_argument("--include", action="append", default=[], help="Glob include pattern (repeatable)")
    parser.add_argument("--exclude", action="append", default=[], help="Glob exclude pattern (repeatable)")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    base_config = load_config(args.config)
    runtime_config = _apply_runtime_overrides(
        config=base_config,
        sample=args.sample,
        max_rows=args.max_rows,
        max_files=args.max_files,
    )

    run_pipeline(
        config=runtime_config,
        only=args.only,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
    )


if __name__ == "__main__":
    main()
