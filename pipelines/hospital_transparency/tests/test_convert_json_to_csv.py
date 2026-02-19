from __future__ import annotations

import csv
import shutil
from pathlib import Path

from pipelines.hospital_transparency.scripts.convert_json_to_csv import convert_json_to_csv


def test_convert_json_to_csv_sample(tmp_path: Path) -> None:
    raw_root = tmp_path / "cache" / "raw"
    json_dir = raw_root / "json"
    metadata_dir = tmp_path / "metadata"

    json_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    sample_json = Path("pipelines/hospital_transparency/data/samples/sample_prices.json")
    shutil.copy(sample_json, json_dir / sample_json.name)

    config = {
        "download_cache_root": str(raw_root),
        "local_processed_root": str(tmp_path / "processed"),
        "metadata_root": str(metadata_dir),
        "schema_path": "pipelines/hospital_transparency/data/metadata/schema_v1.csv",
        "column_mapping_path": "pipelines/hospital_transparency/data/metadata/column_mapping.csv",
        "sample_mode": False,
        "sample_max_rows": 5000,
        "max_files": 100,
        "upload_outputs_to_drive": False,
        "delete_local_cache_after_upload": False,
    }

    logs = convert_json_to_csv(config)

    output_csv = raw_root / "csv" / "converted_from_json" / "sample_prices.csv"
    assert output_csv.exists()
    assert len(logs) == 1
    assert logs[0]["source_file"] == "sample_prices.json"

    with output_csv.open("r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)
        assert len(rows) == 2
        assert "hospital.name" in reader.fieldnames
        assert "standard_charge" in reader.fieldnames
