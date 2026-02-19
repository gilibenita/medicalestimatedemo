"""Download raw hospital transparency files from Google Drive into local cache."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

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
)

LOGGER = logging.getLogger(__name__)


def _matches_patterns(name: str, includes: Iterable[str], excludes: Iterable[str]) -> bool:
    include_patterns = [pattern for pattern in includes if pattern]
    exclude_patterns = [pattern for pattern in excludes if pattern]

    if include_patterns and not any(fnmatch.fnmatch(name, pattern) for pattern in include_patterns):
        return False
    if exclude_patterns and any(fnmatch.fnmatch(name, pattern) for pattern in exclude_patterns):
        return False
    return True


def _write_sync_log(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file_id",
        "name",
        "size",
        "modified_time",
        "local_path",
        "source_subfolder",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sync_from_drive(
    config: Dict[str, Any],
    include_patterns: Iterable[str] | None = None,
    exclude_patterns: Iterable[str] | None = None,
    max_files: int | None = None,
) -> List[Dict[str, Any]]:
    include_patterns = include_patterns or []
    exclude_patterns = exclude_patterns or []

    ensure_local_layout(config)
    raw_root = resolve_repo_path(config["download_cache_root"])

    effective_max_files = int(max_files if max_files is not None else config.get("max_files", 100))
    if effective_max_files <= 0:
        raise ValueError("max_files must be a positive integer")

    client = GDriveClient()
    folders = resolve_drive_folder_ids(client, config)

    json_files = [
        {**item, "source_subfolder": str(config.get("json_subdir_name", "json"))}
        for item in client.list_files(folders["json"])
    ]
    csv_files = [
        {**item, "source_subfolder": str(config.get("csv_subdir_name", "csv"))}
        for item in client.list_files(folders["csv"])
    ]

    all_files = sorted(json_files + csv_files, key=lambda item: item["name"].lower())

    filtered_files = [
        file_info
        for file_info in all_files
        if _matches_patterns(file_info["name"], include_patterns, exclude_patterns)
    ]
    filtered_files = filtered_files[:effective_max_files]

    download_rows: List[Dict[str, Any]] = []
    for file_info in tqdm(filtered_files, desc="Downloading from Drive", unit="file"):
        source_subfolder = file_info["source_subfolder"]
        local_dir = raw_root / source_subfolder
        local_path = local_dir / file_info["name"]

        LOGGER.info(
            "Downloading %s (%s) to %s",
            file_info["name"],
            file_info["id"],
            local_path,
        )
        downloaded_path = client.download_file(file_info["id"], local_path)

        download_rows.append(
            {
                "file_id": file_info["id"],
                "name": file_info.get("name", ""),
                "size": file_info.get("size", ""),
                "modified_time": file_info.get("modifiedTime", ""),
                "local_path": str(downloaded_path),
                "source_subfolder": source_subfolder,
            }
        )

    sync_log_path = metadata_file(config, "drive_sync_log.csv")
    _write_sync_log(download_rows, sync_log_path)
    LOGGER.info("Wrote sync log: %s", sync_log_path)

    return download_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync raw files from Google Drive to local cache")
    parser.add_argument("--config", default="pipelines/hospital_transparency/config.yml")
    parser.add_argument("--include", action="append", default=[], help="Glob include pattern (repeatable)")
    parser.add_argument("--exclude", action="append", default=[], help="Glob exclude pattern (repeatable)")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)
    sync_from_drive(
        config=config,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
