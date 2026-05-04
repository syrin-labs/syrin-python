"""URL loader for fetching web content."""

from __future__ import annotations

from typing import cast

import httpx

from syrin.knowledge._document import Document


class URLLoader:
    """Load content from URLs.

    Fetches web pages and extracts readable text content.
    Uses trafilatura if available for better extraction.

    Example:
        loader = URLLoader("https://example.com/article")
        docs = await loader.aload()
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize URLLoader.

        Args:
            url: URL to fetch.
            headers: Optional HTTP headers.
        """
        self._url = url
        self._headers = headers or {}

    @property
    def url(self) -> str:
        """URL being loaded."""
        return self._url

    async def aload(self) -> list[Document]:
        """Load the URL asynchronously.

        Returns:
            List containing one Document with the page content.
        """
        text = await self._fetch_content()
        return [
            Document(
                content=text,
                source=self._url,
                source_type="url",
            )
        ]

    async def _fetch_content(self) -> str:
        """Fetch content from URL.

        Returns:
            Extracted text content.
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(self._url, headers=self._headers)
            response.raise_for_status()
            html = response.text

            # Try trafilatura first for better extraction
            try:
                import trafilatura  # type: ignore[import-not-found]

                text = trafilatura.extract(html)
                if text:
                    return cast(str, text)
            except ImportError:
                pass

            # Fallback: simple HTML stripping
            return self._strip_html(html)

    def _strip_html(self, html: str) -> str:
        """Simple HTML tag stripping.

        Args:
            html: HTML content.

        Returns:
            Plain text content.
        """
        import html as html_module
        import re

        # Remove script and style elements
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)

        # Replace br/p with newlines
        html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
        html = re.sub(r"</p>", "\n\n", html, flags=re.IGNORECASE)

        # Remove all remaining tags
        text = re.sub(r"<[^>]+>", "", html)

        # Decode all HTML entities (handles &nbsp; &amp; &lt; &gt; &quot; &#39; &#x27; etc.)
        text = html_module.unescape(text)

        # Clean up whitespace
        text = re.sub(r"\n\n+", "\n\n", text)
        text = re.sub(r" +", " ", text)
        text = text.strip()

        return text


__all__ = ["URLLoader"]
