from __future__ import annotations

import csv
import shutil
from pathlib import Path

from pipelines.hospital_transparency.scripts.standardize_csv import standardize_csv_files


def _canonical_columns(schema_path: Path) -> list[str]:
    with schema_path.open("r", encoding="utf-8") as infile:
        return [row["canonical_column"] for row in csv.DictReader(infile)]


def test_standardize_csv_outputs_canonical_columns(tmp_path: Path) -> None:
    raw_root = tmp_path / "cache" / "raw"
    csv_dir = raw_root / "csv"
    metadata_dir = tmp_path / "metadata"

    csv_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    sample_csv = Path("pipelines/hospital_transparency/data/samples/sample_prices.csv")
    shutil.copy(sample_csv, csv_dir / sample_csv.name)

    schema_path = Path("pipelines/hospital_transparency/data/metadata/schema_v1.csv")
    mapping_path = Path("pipelines/hospital_transparency/data/metadata/column_mapping.csv")

    config = {
        "download_cache_root": str(raw_root),
        "local_processed_root": str(tmp_path / "processed"),
        "metadata_root": str(metadata_dir),
        "schema_path": str(schema_path),
        "column_mapping_path": str(mapping_path),
        "sample_mode": False,
        "sample_max_rows": 5000,
        "max_files": 100,
        "upload_outputs_to_drive": False,
        "delete_local_cache_after_upload": False,
    }

    standardize_csv_files(config)

    output_csv = tmp_path / "processed" / "sample_prices__standardized.csv"
    assert output_csv.exists()

    expected_columns = _canonical_columns(schema_path)
    with output_csv.open("r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        assert reader.fieldnames is not None
        for column in expected_columns:
            assert column in reader.fieldnames

        rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["code"] == "99213"
        assert rows[0]["code_type"].lower() == "cpt"
