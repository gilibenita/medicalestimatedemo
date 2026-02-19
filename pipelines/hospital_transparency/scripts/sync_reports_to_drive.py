"""Upload pipeline metadata reports to Google Drive."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from pipelines.hospital_transparency.scripts.gdrive_client import GDriveClient, resolve_drive_folder_ids
from pipelines.hospital_transparency.scripts.pipeline_config import (
    configure_logging,
    load_config,
    resolve_repo_path,
)

LOGGER = logging.getLogger(__name__)

REPORT_FILES = [
    "files_manifest_generated.csv",
    "drive_sync_log.csv",
    "conversion_log.csv",
    "profile_report.csv",
    "validation_results.csv",
    "schema_v1.csv",
    "column_mapping.csv",
]


def sync_reports_to_drive(config: Dict[str, Any]) -> List[str]:
    if not bool(config.get("upload_outputs_to_drive", True)):
        LOGGER.info("upload_outputs_to_drive=false; skipping report uploads")
        return []

    metadata_root = resolve_repo_path(config["metadata_root"])

    client = GDriveClient()
    folder_ids = resolve_drive_folder_ids(client, config)
    reports_folder_id = folder_ids["reports"]

    uploaded: List[str] = []
    for filename in REPORT_FILES:
        local_path = metadata_root / filename
        if not local_path.exists():
            LOGGER.warning("Skipping missing report: %s", local_path)
            continue

        client.upload_file(local_path, reports_folder_id, mime_type="text/csv", overwrite=True)
        uploaded.append(filename)
        LOGGER.info("Uploaded report: %s", filename)

    return uploaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload metadata reports to Drive")
    parser.add_argument("--config", default="pipelines/hospital_transparency/config.yml")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)
    sync_reports_to_drive(config)


if __name__ == "__main__":
    main()
