"""Google Drive API helpers for cloud-native file sync."""

from __future__ import annotations

import io
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload, MediaIoBaseUpload

from pipelines.hospital_transparency.scripts.pipeline_config import require_service_account_credentials

LOGGER = logging.getLogger(__name__)

DRIVE_SCOPE = ["https://www.googleapis.com/auth/drive"]
FOLDER_MIME = "application/vnd.google-apps.folder"


class GDriveClient:
    def __init__(self) -> None:
        creds_path = require_service_account_credentials()
        credentials = service_account.Credentials.from_service_account_file(
            str(creds_path), scopes=DRIVE_SCOPE
        )
        self.service = build("drive", "v3", credentials=credentials, cache_discovery=False)

    @staticmethod
    def _escape(value: str) -> str:
        return value.replace("'", "\\'")

    def find_file_by_name(self, folder_id: str, name: str, mime_type: str | None = None) -> Optional[Dict[str, Any]]:
        query_parts = [
            f"'{folder_id}' in parents",
            f"name = '{self._escape(name)}'",
            "trashed = false",
        ]
        if mime_type:
            query_parts.append(f"mimeType = '{mime_type}'")

        response = (
            self.service.files()
            .list(
                q=" and ".join(query_parts),
                fields="files(id,name,mimeType,size,modifiedTime)",
                pageSize=10,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        files = response.get("files", [])
        return files[0] if files else None

    def get_subfolder_id(self, parent_id: str, name: str, create_if_missing: bool = True) -> str:
        """Return folder ID for `name` under `parent_id`, creating folders for nested paths if requested."""
        parts = [part for part in name.split("/") if part and part != "."]
        if not parts:
            return parent_id

        current_parent = parent_id
        for part in parts:
            existing = self.find_file_by_name(current_parent, part, mime_type=FOLDER_MIME)
            if existing:
                current_parent = existing["id"]
                continue

            if not create_if_missing:
                raise FileNotFoundError(
                    f"Drive folder '{part}' not found under parent '{current_parent}'."
                )

            metadata = {"name": part, "mimeType": FOLDER_MIME, "parents": [current_parent]}
            created = (
                self.service.files()
                .create(
                    body=metadata,
                    fields="id,name",
                    supportsAllDrives=True,
                )
                .execute()
            )
            current_parent = created["id"]
            LOGGER.info("Created Drive folder: %s (%s)", created["name"], created["id"])

        return current_parent

    def list_files(self, folder_id: str) -> List[Dict[str, Any]]:
        query = (
            f"'{folder_id}' in parents and trashed = false "
            f"and mimeType != '{FOLDER_MIME}'"
        )
        files: List[Dict[str, Any]] = []
        page_token: str | None = None

        while True:
            response = (
                self.service.files()
                .list(
                    q=query,
                    fields="nextPageToken, files(id,name,size,modifiedTime,mimeType)",
                    pageToken=page_token,
                    pageSize=1000,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            files.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                break

        return files

    def download_file(self, file_id: str, dest_path: str | Path) -> Path:
        destination = Path(dest_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        request = self.service.files().get_media(fileId=file_id, supportsAllDrives=True)
        with destination.open("wb") as outfile:
            downloader = MediaIoBaseDownload(outfile, request, chunksize=10 * 1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk()

        return destination

    def upload_file(
        self,
        local_path: str | Path,
        folder_id: str,
        mime_type: str | None = None,
        overwrite: bool = True,
    ) -> Dict[str, Any]:
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"Local file not found for upload: {path}")

        if overwrite:
            existing = self.find_file_by_name(folder_id, path.name)
            if existing:
                self.delete_file(existing["id"])

        media = MediaFileUpload(str(path), mimetype=mime_type, resumable=True)
        metadata = {"name": path.name, "parents": [folder_id]}
        uploaded = (
            self.service.files()
            .create(
                body=metadata,
                media_body=media,
                fields="id,name,mimeType,size,modifiedTime",
                supportsAllDrives=True,
            )
            .execute()
        )
        return uploaded

    def upload_bytes(
        self,
        filename: str,
        bytes_data: bytes,
        folder_id: str,
        mime_type: str = "application/octet-stream",
        overwrite: bool = True,
    ) -> Dict[str, Any]:
        if overwrite:
            existing = self.find_file_by_name(folder_id, filename)
            if existing:
                self.delete_file(existing["id"])

        stream = io.BytesIO(bytes_data)
        media = MediaIoBaseUpload(stream, mimetype=mime_type, resumable=True)
        metadata = {"name": filename, "parents": [folder_id]}
        uploaded = (
            self.service.files()
            .create(
                body=metadata,
                media_body=media,
                fields="id,name,mimeType,size,modifiedTime",
                supportsAllDrives=True,
            )
            .execute()
        )
        return uploaded

    def delete_file(self, file_id: str) -> None:
        self.service.files().delete(fileId=file_id, supportsAllDrives=True).execute()


def resolve_drive_folder_ids(client: GDriveClient, config: Dict[str, Any]) -> Dict[str, str]:
    root_id = config.get("drive_root_folder_id", "").strip()
    if not root_id or root_id == "PUT_FOLDER_ID_HERE":
        raise RuntimeError(
            "Drive root folder ID is not configured. Set drive_root_folder_id in config.yml "
            "or export GDRIVE_ROOT_FOLDER_ID."
        )

    folder_ids = {
        "root": root_id,
        "json": client.get_subfolder_id(root_id, str(config.get("json_subdir_name", "json")), create_if_missing=False),
        "csv": client.get_subfolder_id(root_id, str(config.get("csv_subdir_name", "csv")), create_if_missing=False),
        "converted_csv": client.get_subfolder_id(
            root_id,
            str(config.get("converted_json_drive_subdir_name", "csv/converted_from_json")),
            create_if_missing=True,
        ),
        "standardized": client.get_subfolder_id(
            root_id,
            str(config.get("standardized_drive_subdir_name", "processed/standardized")),
            create_if_missing=True,
        ),
        "reports": client.get_subfolder_id(
            root_id,
            str(config.get("reports_drive_subdir_name", "metadata/reports")),
            create_if_missing=True,
        ),
    }

    return folder_ids
