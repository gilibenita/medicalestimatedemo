# Hospital Transparency Google Drive Pipeline

This pipeline runs in GitHub Codespaces (or any Linux VM), pulls large NYC hospital transparency source files from Google Drive, standardizes them, and writes outputs and reports back to Google Drive.

## What It Does

1. Downloads raw files from Drive subfolders `json/` and `csv/`.
2. Converts JSON files to CSV and uploads conversions to `csv/converted_from_json`.
3. Standardizes all CSV inputs to a canonical schema and uploads outputs to `processed/standardized`.
4. Builds validation and profiling reports and uploads them to `metadata/reports`.

## Folder Structure

```text
pipelines/hospital_transparency/
  config.yml
  README.md
  scripts/
    gdrive_client.py
    sync_from_drive.py
    build_manifest.py
    convert_json_to_csv.py
    standardize_csv.py
    validate_processed.py
    sync_reports_to_drive.py
    run_pipeline.py
  notebooks/
  tests/
    test_convert_json_to_csv.py
    test_standardize_csv.py
  data/
    metadata/
      files_manifest.csv
      files_manifest_generated.csv
      schema_v1.csv
      column_mapping.csv
      drive_sync_log.csv
      conversion_log.csv
      profile_report.csv
      validation_results.csv
    samples/
      sample_prices.json
      sample_prices.csv
    cache/                 # gitignored
      raw/
        json/
        csv/
          converted_from_json/
    processed/             # gitignored
```

## Google Drive Folder ID

1. Open the Drive folder in browser.
2. Copy the ID from the URL:
   - `https://drive.google.com/drive/folders/<FOLDER_ID>`
3. Put this ID in `pipelines/hospital_transparency/config.yml` as `drive_root_folder_id`, or export `GDRIVE_ROOT_FOLDER_ID`.

## Google Cloud Service Account Setup

1. Create/select a GCP project.
2. Enable **Google Drive API**.
3. Create a service account.
4. Create and download a JSON key.
5. Share the target Drive root folder with the service account email (Editor access).

## Codespaces Credentials Setup (Secure Pattern)

The scripts use service-account auth only (no browser login flow by default).

1. **Do not commit the JSON key.** Keep it outside the repo checkout (for example `~/.secrets/service_account.json`).
2. Store the key in a secrets manager (GitHub Codespaces Secret, GitHub Actions Secret, 1Password, etc.).
3. In Codespaces, write the secret value to a locked-down file at runtime:
   ```bash
   mkdir -p ~/.secrets
   printf '%s' "$GCP_SERVICE_ACCOUNT_JSON" > ~/.secrets/service_account.json
   chmod 600 ~/.secrets/service_account.json
   ```
4. Set environment variables:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=~/.secrets/service_account.json
   export GDRIVE_ROOT_FOLDER_ID=your_drive_root_id  # optional override of config.yml
   ```

> Tip: if your secret store requires single-line values, store a base64-encoded JSON value and decode it before writing the file.

## Run

```bash
python pipelines/hospital_transparency/scripts/run_pipeline.py
```

Useful flags:

```bash
python pipelines/hospital_transparency/scripts/run_pipeline.py --sample --max-rows 5000 --max-files 20
python pipelines/hospital_transparency/scripts/run_pipeline.py --only sync --include "*.json"
python pipelines/hospital_transparency/scripts/run_pipeline.py --only validate
```

## Sample Mode

`sample_mode: true` limits processing to `sample_max_rows` per file, useful for fast iteration in Codespaces.

CLI overrides:

- `--sample`
- `--max-rows N`
- `--max-files N`

## Troubleshooting Large Files

- Use `--sample` first to validate mappings before full runs.
- CSV standardization uses Polars lazy scanning/streaming to reduce memory pressure.
- JSON NDJSON conversion uses streaming line-by-line passes.
- Process runs per file and uploads incrementally, so a single failure does not require restarting everything.
- If standard JSON files are extremely large, consider converting them to NDJSON upstream for better streaming behavior.
