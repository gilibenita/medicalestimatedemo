"""Configuration and path helpers for the hospital transparency pipeline."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PIPELINE_ROOT.parent.parent
DEFAULT_CONFIG_PATH = PIPELINE_ROOT / "config.yml"


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def resolve_repo_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load YAML config and apply environment overrides."""
    load_dotenv(PIPELINE_ROOT / ".env", override=False)

    config_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    config_file = config_file if config_file.is_absolute() else (REPO_ROOT / config_file).resolve()

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as infile:
        config = yaml.safe_load(infile) or {}

    if "GDRIVE_ROOT_FOLDER_ID" in os.environ:
        config["drive_root_folder_id"] = os.environ["GDRIVE_ROOT_FOLDER_ID"]

    config.setdefault("download_cache_root", "./pipelines/hospital_transparency/data/cache/raw")
    config.setdefault("local_processed_root", "./pipelines/hospital_transparency/data/processed")
    config.setdefault("json_subdir_name", "json")
    config.setdefault("csv_subdir_name", "csv")
    config.setdefault("converted_json_drive_subdir_name", "csv/converted_from_json")
    config.setdefault("standardized_drive_subdir_name", "processed/standardized")
    config.setdefault("reports_drive_subdir_name", "metadata/reports")
    config.setdefault("sample_mode", True)
    config.setdefault("sample_max_rows", 5000)
    config.setdefault("max_files", 100)
    config.setdefault("upload_outputs_to_drive", True)
    config.setdefault("delete_local_cache_after_upload", True)

    config["metadata_root"] = str(PIPELINE_ROOT / "data" / "metadata")
    config["schema_path"] = str(PIPELINE_ROOT / "data" / "metadata" / "schema_v1.csv")
    config["column_mapping_path"] = str(PIPELINE_ROOT / "data" / "metadata" / "column_mapping.csv")

    return config


def ensure_local_layout(config: Dict[str, Any]) -> None:
    raw_root = resolve_repo_path(config["download_cache_root"])
    processed_root = resolve_repo_path(config["local_processed_root"])
    metadata_root = resolve_repo_path(config["metadata_root"])

    (raw_root / "json").mkdir(parents=True, exist_ok=True)
    (raw_root / "csv").mkdir(parents=True, exist_ok=True)
    (raw_root / "csv" / "converted_from_json").mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)


def require_service_account_credentials() -> Path:
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not creds:
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS is not set. "
            "Set it to the service-account JSON path (for example via Codespaces secrets)."
        )

    creds_path = Path(creds).expanduser().resolve()
    if not creds_path.exists():
        raise RuntimeError(f"GOOGLE_APPLICATION_CREDENTIALS path does not exist: {creds_path}")

    try:
        creds_path.relative_to(REPO_ROOT)
        LOGGER.warning(
            "Service-account credential path resolves inside the git repository: %s. "
            "Prefer storing credentials in an external secrets directory (for example ~/.secrets) "
            "to reduce accidental commits.",
            creds_path,
        )
    except ValueError:
        # Expected case: credential file lives outside the repository tree.
        pass

    return creds_path


def metadata_file(config: Dict[str, Any], filename: str) -> Path:
    return resolve_repo_path(Path(config["metadata_root"]) / filename)


def should_run_sample_mode(config: Dict[str, Any]) -> bool:
    return bool(config.get("sample_mode", False))


def sample_limit(config: Dict[str, Any]) -> int:
    try:
        return int(config.get("sample_max_rows", 5000))
    except (TypeError, ValueError):
        LOGGER.warning("Invalid sample_max_rows in config. Falling back to 5000.")
        return 5000
