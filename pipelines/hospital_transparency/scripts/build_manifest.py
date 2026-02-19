"""Generate an auto manifest from local raw cache files."""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any, Dict, List

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from pipelines.hospital_transparency.scripts.pipeline_config import (
    configure_logging,
    ensure_local_layout,
    load_config,
    metadata_file,
    resolve_repo_path,
)

LOGGER = logging.getLogger(__name__)


SUPPORTED_SUFFIXES = {".json", ".ndjson", ".csv"}


def _scan_raw_files(raw_root: Path) -> List[Path]:
    files: List[Path] = []
    if not raw_root.exists():
        return files

    for file_path in raw_root.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(file_path)
    return sorted(files)


def build_manifest(config: Dict[str, Any]) -> Path:
    ensure_local_layout(config)

    raw_root = resolve_repo_path(config["download_cache_root"])
    output_path = metadata_file(config, "files_manifest_generated.csv")

    rows: List[Dict[str, Any]] = []
    for file_path in _scan_raw_files(raw_root):
        stat = file_path.stat()
        modified_time = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
        rows.append(
            {
                "hospital_system": "",
                "hospital_name": "",
                "source_url": "",
                "file_name": file_path.name,
                "file_type": file_path.suffix.lower().lstrip("."),
                "notes": "auto-generated",
                "size_bytes": stat.st_size,
                "modified_time": modified_time,
                "local_path": str(file_path),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "hospital_system",
        "hospital_name",
        "source_url",
        "file_name",
        "file_type",
        "notes",
        "size_bytes",
        "modified_time",
        "local_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    LOGGER.info("Generated file manifest: %s (%d records)", output_path, len(rows))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a generated manifest from local raw cache")
    parser.add_argument("--config", default="pipelines/hospital_transparency/config.yml")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)
    build_manifest(config)


if __name__ == "__main__":
    main()
