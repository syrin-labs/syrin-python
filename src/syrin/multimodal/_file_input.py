"""Internal module: file-to-message and PDF text extraction."""

from __future__ import annotations

import base64
import tempfile


def file_to_message(data: bytes, mimetype: str, role: str = "user") -> str:
    """Convert file bytes to a base64 data URL suitable as Message content.

    Args:
        data: Raw file bytes.
        mimetype: MIME type (e.g. image/png, application/pdf).
        role: Message role (user/system/assistant). Used by callers when
            building messages; does not affect the returned string.

    Returns:
        Data URL string: data:{mimetype};base64,{base64_data}

    Example:
        >>> file_to_message(b"hello", "text/plain", "user")
        'data:text/plain;base64,aGVsbG8='
    """
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mimetype};base64,{b64}"


def pdf_extract_text(data: bytes) -> str:
    """Extract text from PDF bytes using docling.

    Requires the docling package (pip install syrin[pdf]).

    Args:
        data: Raw PDF file bytes.

    Returns:
        Extracted text as string. Empty string for empty input.

    Raises:
        ImportError: When docling is not installed. Install with pip install syrin[pdf].
    """
    if not data:
        return ""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as e:
        raise ImportError("syrin[pdf] required for PDF extraction. pip install syrin[pdf]") from e

    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(data)
        tmp.flush()
        converter = DocumentConverter()
        result = converter.convert(str(tmp.name))
        parts: list[str] = []
        for page in result.document.pages:
            page_text = getattr(page, "text", "") or ""
            if page_text:
                parts.append(page_text)
        return "\n".join(parts)
