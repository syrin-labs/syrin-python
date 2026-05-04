"""Google Drive loader for public folders and files."""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

import httpx

from syrin.knowledge._document import Document
from syrin.knowledge._loader import DocumentLoader

# Google Workspace MIME types
MIME_GOOGLE_DOCUMENT = "application/vnd.google-apps.document"
MIME_GOOGLE_SHEET = "application/vnd.google-apps.spreadsheet"
MIME_GOOGLE_PRESENTATION = "application/vnd.google-apps.presentation"
MIME_GOOGLE_FOLDER = "application/vnd.google-apps.folder"

# Export format mapping for Google Workspace files
EXPORT_FORMAT: dict[str, str] = {
    MIME_GOOGLE_DOCUMENT: "text/plain",
    MIME_GOOGLE_SHEET: "text/csv",
    MIME_GOOGLE_PRESENTATION: "text/plain",  # Export as plain text (no native; falls back)
}


def _extract_id_from_url(url_or_id: str) -> str:
    """Extract folder or file ID from Google Drive URL.

    Supports:
    - https://drive.google.com/drive/folders/FOLDER_ID
    - https://drive.google.com/file/d/FILE_ID/view
    - https://drive.google.com/open?id=FILE_ID
    - Raw ID (alphanumeric + - _)
    """
    s = url_or_id.strip()
    if not s:
        return ""
    # Raw ID: typical format is alphanumeric, about 33-44 chars
    if re.fullmatch(r"[\w\-]{20,}", s):
        return s
    parsed = urlparse(s)
    path = parsed.path or ""
    # /drive/folders/ID or /file/d/ID
    m = re.search(r"/folders/([\w\-]+)", path) or re.search(r"/file/d/([\w\-]+)", path)
    if m:
        return m.group(1)
    # ?id=FILE_ID or &id=FILE_ID
    q = parsed.query or ""
    m = re.search(r"(?:^|[?&])id=([\w\-]+)", q)
    if m:
        return m.group(1)
    return s


def _is_file_url(url_or_id: str) -> bool:
    """Return True if URL explicitly points to a file (not folder)."""
    s = url_or_id.strip().lower()
    return "drive.google.com" in s and ("/file/d/" in s or "/open" in s or "/uc?" in s)


class GoogleDriveLoader:
    """Load documents from a public Google Drive folder.

    Uses Google Drive API v3 with an API key (no OAuth) for public folders.
    Folder must be shared as "Anyone with the link can view".

    Example:
        # Load from public folder
        loader = GoogleDriveLoader(
            folder="https://drive.google.com/drive/folders/1ABC123...",
            api_key="your-api-key",  # or set GOOGLE_API_KEY env
            recursive=True,
            pattern=r"\\.(txt|md|pdf)$",
        )
        docs = await loader.aload()

        # Single public file (folder param as file URL)
        loader = GoogleDriveLoader(
            folder="https://drive.google.com/file/d/FILE_ID/view",
            api_key="...",
        )
    """

    def __init__(
        self,
        folder: str,
        *,
        recursive: bool = True,
        pattern: str | None = None,
        allowed_folder: list[str] | None = None,
        excluded_folder: list[str] | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize GoogleDriveLoader.

        Args:
            folder: Google Drive folder URL, folder ID, or single file URL.
            recursive: If True, traverse subfolders. Default True.
            pattern: Regex pattern for file names to include (e.g. r"\\.(txt|md)$").
                     None = include all supported files.
            allowed_folder: If non-empty, only include files from these folder names or IDs.
            excluded_folder: Exclude these folder names or IDs from traversal.
            api_key: Google API key for Drive API. Falls back to GOOGLE_API_KEY env.
        """
        self._folder = folder.strip()
        self._recursive = recursive
        self._pattern = re.compile(pattern) if pattern else None
        self._allowed_folder = list(allowed_folder) if allowed_folder else []
        self._excluded_folder = list(excluded_folder) if excluded_folder else []
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")

    @property
    def folder(self) -> str:
        """Folder URL or ID."""
        return self._folder

    def load(self) -> list[Document]:
        """Load documents synchronously. Calls aload() via asyncio.run().

        For use in already-async contexts prefer aload() directly.

        Returns:
            List of Documents from matching files.
        """
        import asyncio

        return asyncio.run(self.aload())

    async def aload(self) -> list[Document]:
        """Load documents from the Google Drive folder.

        Returns:
            List of Documents from matching files.
        """
        if not self._api_key:
            raise ValueError(
                "api_key is required for Google Drive loader. "
                "Pass api_key= or set GOOGLE_API_KEY environment variable."
            )

        folder_id = _extract_id_from_url(self._folder)
        if not folder_id:
            raise ValueError("Could not extract folder/file ID from: " + self._folder)

        # Single file mode: folder param is actually a file URL
        if _is_file_url(self._folder):
            meta = await self._get_file_metadata(folder_id)
            if meta:
                doc = await self._load_single_file(
                    folder_id,
                    str(meta.get("mimeType", "application/octet-stream")),
                    str(meta.get("name", "")),
                    str(meta.get("name", "")),
                )
                return [doc] if doc else []
            return []

        files = await self._list_files(folder_id, recursive=self._recursive)
        docs: list[Document] = []

        for file_info in files:
            try:
                doc = await self._load_file(file_info)
                if doc:
                    docs.append(doc)
            except Exception as e:
                warnings.warn(
                    f"Failed to load {file_info.get('name', '?')}: {e}",
                    stacklevel=2,
                )

        return docs

    async def _list_files(
        self,
        folder_id: str,
        recursive: bool,
        parent_path: str = "",
    ) -> list[dict[str, object]]:
        """List files in folder via Drive API v3."""
        base = "https://www.googleapis.com/drive/v3/files"
        q = f"'{folder_id}' in parents and trashed = false"
        params: dict[str, str] = {
            "q": q,
            "key": self._api_key,
            "fields": "files(id,name,mimeType,size)",
        }

        result: list[dict[str, object]] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(base, params=params)
            resp.raise_for_status()
            data = resp.json()
            files = data.get("files", [])

            for f in files:
                mime = f.get("mimeType", "") or ""
                name = f.get("name", "") or ""
                fid = f.get("id", "") or ""
                path = f"{parent_path}/{name}".lstrip("/")

                if mime == MIME_GOOGLE_FOLDER:
                    if not recursive:
                        continue
                    # Check excluded
                    if self._excluded_folder and (
                        name in self._excluded_folder or fid in self._excluded_folder
                    ):
                        continue
                    # Check allowed (if specified)
                    if self._allowed_folder and (
                        name not in self._allowed_folder and fid not in self._allowed_folder
                    ):
                        continue
                    sub = await self._list_files(str(fid), recursive=True, parent_path=path)
                    result.extend(sub)
                else:
                    # Only include if parent folder is in allowed (root files always included)
                    if self._allowed_folder and parent_path:
                        parts = path.split("/")
                        parent_name = parts[-2] if len(parts) >= 2 else ""
                        if parent_name not in self._allowed_folder:
                            continue
                    if not self._matches_pattern(name):
                        continue
                    result.append(
                        {
                            "id": fid,
                            "name": name,
                            "mimeType": mime,
                            "path": path,
                        }
                    )

        return result

    async def _get_file_metadata(self, file_id: str) -> dict[str, object] | None:
        """Fetch file metadata from Drive API."""
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
        params = {"key": self._api_key, "fields": "id,name,mimeType"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                return None
            data: object = resp.json()
            return cast("dict[str, object]", data) if isinstance(data, dict) else None

    def _matches_pattern(self, name: str) -> bool:
        """Return True if filename matches pattern."""
        if self._pattern is None:
            return True
        return bool(self._pattern.search(name))

    async def _load_file(self, file_info: dict[str, object]) -> Document | None:
        """Load a single file and return a Document."""
        fid = str(file_info.get("id", ""))
        name = str(file_info.get("name", ""))
        mime = str(file_info.get("mimeType", ""))
        path = str(file_info.get("path", name))

        return await self._load_single_file(fid, mime, name, path)

    async def _load_single_file(
        self,
        file_id: str,
        mime_type: str,
        name: str = "",
        path: str = "",
    ) -> Document | None:
        """Load content of a file by ID and MIME type."""
        content: str

        if mime_type in (
            MIME_GOOGLE_DOCUMENT,
            MIME_GOOGLE_SHEET,
            MIME_GOOGLE_PRESENTATION,
        ):
            content = await self._export_google_workspace(file_id, mime_type)
        else:
            content = await self._download_blob(file_id, mime_type, name)

        if not content or not content.strip():
            return None

        source = path or name or f"gdrive/{file_id}"
        return Document(
            content=content,
            source=source,
            source_type="google_drive",
            metadata={"file_id": file_id, "name": name or path, "mime_type": mime_type},
        )

    async def _export_google_workspace(self, file_id: str, mime_type: str) -> str:
        """Export Google Docs/Sheets/Slides to text."""
        if mime_type == MIME_GOOGLE_DOCUMENT:
            url = f"https://docs.google.com/document/d/{file_id}/export?format=txt"
        elif mime_type == MIME_GOOGLE_SHEET:
            url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"
        elif mime_type == MIME_GOOGLE_PRESENTATION:
            # Slides don't have plain text export; use PDF and skip for now, or try txt
            url = f"https://docs.google.com/presentation/d/{file_id}/export?format=txt"
        else:
            url = f"https://docs.google.com/document/d/{file_id}/export?format=txt"

        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text

    async def _download_blob(self, file_id: str, mime_type: str, name: str) -> str:
        """Download blob file and extract text."""
        # Use Drive API alt=media for reliable download with API key
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
        params = {"key": self._api_key}

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.content

        # Try to decode as text for known text types
        if mime_type.startswith("text/") or mime_type in (
            "application/json",
            "application/x-yaml",
            "application/csv",
        ):
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                return data.decode("latin-1")

        # For PDF/DOCX, use existing loaders if available
        ext = Path(name).suffix.lower() if name else ""
        if ext == ".pdf":
            return await self._extract_text_via_loader(data, "pdf", ".pdf")
        if ext in (".docx", ".doc"):
            return await self._extract_text_via_loader(data, "docx", ".docx")
        if ext in (".xlsx", ".xls"):
            return await self._extract_text_via_loader(data, "excel", ".xlsx")
        if ext == ".csv":
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                return data.decode("latin-1")

        # Fallback: try UTF-8
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            warnings.warn(
                f"Cannot decode binary file {name} (mime={mime_type}) as text; skipping.",
                stacklevel=2,
            )
            return ""

    async def _extract_text_via_loader(
        self,
        data: bytes,
        loader_type: str,
        suffix: str,
    ) -> str:
        """Write bytes to temp file and use existing loader to extract text."""
        loader: DocumentLoader
        with tempfile.NamedTemporaryFile(suffix=suffix) as tf:
            tf.write(data)
            tf.flush()
            if loader_type == "pdf":
                from syrin.knowledge.loaders._pdf import PDFLoader

                loader = PDFLoader(tf.name)
            elif loader_type == "docx":
                from syrin.knowledge.loaders._docx import DOCXLoader

                loader = DOCXLoader(tf.name, use_docling=True)
            elif loader_type == "excel":
                from syrin.knowledge.loaders._excel import ExcelLoader

                loader = ExcelLoader(tf.name)
            else:
                try:
                    return data.decode("utf-8")
                except UnicodeDecodeError:
                    return ""

            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, loader.load)

        if not docs:
            return ""
        return "\n\n".join(d.content for d in docs)


__all__ = ["GoogleDriveLoader"]
