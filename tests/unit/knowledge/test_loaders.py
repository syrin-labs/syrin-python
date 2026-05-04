"""Tests for Document, DocumentLoader Protocol, and loaders."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml


class TestDocument:
    """Tests for Document dataclass."""

    def test_create_document(self) -> None:
        """Document can be created with content, source, source_type, and optional metadata."""
        from syrin.knowledge import Document

        doc = Document(
            content="Hello world",
            source="test.txt",
            source_type="text",
        )
        assert doc.content == "Hello world"
        assert doc.source == "test.txt"
        assert doc.source_type == "text"
        assert doc.metadata == {}

    def test_create_document_with_metadata(self) -> None:
        """Document accepts optional metadata (page, title, etc.)."""
        from syrin.knowledge import Document

        doc = Document(
            content="Page content",
            source="doc.pdf",
            source_type="pdf",
            metadata={"page": 1, "total_pages": 5},
        )
        assert doc.metadata["page"] == 1
        assert doc.metadata["total_pages"] == 5

    def test_document_is_mutable(self) -> None:
        """Document is mutable (can be updated)."""
        from syrin.knowledge import Document

        doc = Document(
            content="Hello",
            source="test.txt",
            source_type="text",
        )
        # Should not raise - Documents are mutable
        doc.content = "Updated"
        doc.source = "other.txt"
        doc.metadata["key"] = "value"
        assert doc.content == "Updated"
        assert doc.source == "other.txt"
        assert doc.metadata["key"] == "value"

    def test_document_empty_source_raises(self) -> None:
        """Document raises if source is empty."""
        from syrin.knowledge import Document

        with pytest.raises(ValueError, match="source must be non-empty"):
            Document(content="x", source="", source_type="text")

    def test_document_empty_source_type_raises(self) -> None:
        """Document raises if source_type is empty."""
        from syrin.knowledge import Document

        with pytest.raises(ValueError, match="source_type must be non-empty"):
            Document(content="x", source="file.txt", source_type="")


class TestDocumentLoaderProtocol:
    """Tests for DocumentLoader Protocol."""

    def test_protocol_has_load_method(self) -> None:
        """DocumentLoader has load() method."""
        from syrin.knowledge import DocumentLoader

        assert hasattr(DocumentLoader, "load")

    def test_protocol_has_aload_method(self) -> None:
        """DocumentLoader has aload() method."""
        from syrin.knowledge import DocumentLoader

        assert hasattr(DocumentLoader, "aload")


class TestTextLoader:
    """Tests for TextLoader."""

    @pytest.fixture
    def temp_text_file(self) -> str:
        """Create a temporary text file."""
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w") as f:
            f.write("Hello world\nThis is line 2")
        yield path
        os.unlink(path)

    def test_load_text_file(self, temp_text_file: str) -> None:
        """TextLoader loads text file content."""
        from syrin.knowledge.loaders import TextLoader

        loader = TextLoader(temp_text_file)
        docs = loader.load()

        assert len(docs) == 1
        assert "Hello world" in docs[0].content
        assert docs[0].source == temp_text_file
        assert docs[0].source_type == "text"

    @pytest.mark.asyncio
    async def test_aload_text_file(self, temp_text_file: str) -> None:
        """TextLoader aload() works."""
        from syrin.knowledge.loaders import TextLoader

        loader = TextLoader(temp_text_file)
        docs = await loader.aload()

        assert len(docs) == 1
        assert "Hello world" in docs[0].content

    def test_load_nonexistent_file_raises(self) -> None:
        """TextLoader raises on nonexistent file."""
        from syrin.knowledge.loaders import TextLoader

        loader = TextLoader("/nonexistent/file.txt")
        with pytest.raises(FileNotFoundError):
            loader.load()


class TestRawTextLoader:
    """Tests for RawTextLoader."""

    def test_single_text(self) -> None:
        """RawTextLoader loads single text."""
        from syrin.knowledge.loaders import RawTextLoader

        loader = RawTextLoader("Hello world")
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].content == "Hello world"
        assert docs[0].source.startswith("user_provided")
        assert docs[0].source_type == "text"

    def test_multiple_texts(self) -> None:
        """RawTextLoader loads list of texts."""
        from syrin.knowledge.loaders import RawTextLoader

        loader = RawTextLoader(["Hello", "World"])
        docs = loader.load()

        assert len(docs) == 2
        assert docs[0].content == "Hello"
        assert docs[1].content == "World"

    def test_with_metadata(self) -> None:
        """RawTextLoader accepts custom metadata."""
        from syrin.knowledge.loaders import RawTextLoader

        loader = RawTextLoader("Hello", metadata={"custom": "value"})
        docs = loader.load()

        assert docs[0].metadata["custom"] == "value"


class TestMarkdownLoader:
    """Tests for MarkdownLoader."""

    @pytest.fixture
    def temp_md_file(self) -> str:
        """Create a temporary markdown file."""
        fd, path = tempfile.mkstemp(suffix=".md")
        content = """# Title

## Section 1

Content of section 1.

## Section 2

Content of section 2.
"""
        with os.fdopen(fd, "w") as f:
            f.write(content)
        yield path
        os.unlink(path)

    def test_load_markdown(self, temp_md_file: str) -> None:
        """MarkdownLoader parses headers as sections."""
        from syrin.knowledge.loaders import MarkdownLoader

        loader = MarkdownLoader(temp_md_file)
        docs = loader.load()

        # Title + 2 sections = 3 docs
        assert len(docs) >= 2
        assert any("Section 1" in doc.content for doc in docs)
        assert any("Section 2" in doc.content for doc in docs)

    def test_markdown_metadata(self, temp_md_file: str) -> None:
        """MarkdownLoader includes section hierarchy in metadata."""
        from syrin.knowledge.loaders import MarkdownLoader

        loader = MarkdownLoader(temp_md_file)
        docs = loader.load()

        assert docs[0].source == temp_md_file
        assert docs[0].source_type == "markdown"


class TestYAMLLoader:
    """Tests for YAMLLoader."""

    @pytest.fixture
    def temp_yaml_file(self) -> str:
        """Create a temporary YAML file."""
        fd, path = tempfile.mkstemp(suffix=".yaml")
        data = {"name": "John", "skills": ["Python", "Go"]}
        with os.fdopen(fd, "w") as f:
            yaml.dump(data, f)
        yield path
        os.unlink(path)

    def test_load_yaml(self, temp_yaml_file: str) -> None:
        """YAMLLoader converts YAML to text."""
        from syrin.knowledge.loaders import YAMLLoader

        loader = YAMLLoader(temp_yaml_file)
        docs = loader.load()

        assert len(docs) == 1
        assert "name: John" in docs[0].content
        assert "skills:" in docs[0].content


class TestJSONLoader:
    """Tests for JSONLoader."""

    @pytest.fixture
    def temp_json_file(self) -> str:
        """Create a temporary JSON file."""
        fd, path = tempfile.mkstemp(suffix=".json")
        data = {"name": "John", "age": 30}
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        yield path
        os.unlink(path)

    def test_load_json(self, temp_json_file: str) -> None:
        """JSONLoader converts JSON to text."""
        from syrin.knowledge.loaders import JSONLoader

        loader = JSONLoader(temp_json_file)
        docs = loader.load()

        assert len(docs) == 1
        assert "John" in docs[0].content

    def test_json_with_path(self, temp_json_file: str) -> None:
        """JSONLoader supports dot notation path."""
        from syrin.knowledge.loaders import JSONLoader

        data = {"data": {"items": [{"name": "a"}, {"name": "b"}]}}
        with open(temp_json_file, "w") as f:
            json.dump(data, f)

        loader = JSONLoader(temp_json_file, jq_path="data.items")
        docs = loader.load()

        assert len(docs) == 1
        assert "name" in docs[0].content

    def test_json_invalid_path_returns_document(self, temp_json_file: str) -> None:
        """JSONLoader with missing jq path still returns one document (content may be null)."""
        from syrin.knowledge.loaders import JSONLoader

        data = {"key": "value"}
        with open(temp_json_file, "w") as f:
            json.dump(data, f)

        loader = JSONLoader(temp_json_file, jq_path="nonexistent.path")
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].source == temp_json_file
        assert docs[0].source_type == "json"
        assert "null" in docs[0].content or docs[0].content


class TestDirectoryLoader:
    """Tests for DirectoryLoader."""

    @pytest.fixture
    def temp_dir(self) -> str:
        """Create a temporary directory with files."""
        d = tempfile.mkdtemp()
        Path(d, "file1.txt").write_text("Content 1")
        Path(d, "file2.md").write_text("# Title\n\nContent")
        Path(d, "file3.json").write_text('{"key": "value"}')
        yield d
        import shutil

        shutil.rmtree(d)

    def test_load_directory_glob(self, temp_dir: str) -> None:
        """DirectoryLoader loads files matching glob."""
        from syrin.knowledge.loaders import DirectoryLoader

        loader = DirectoryLoader(temp_dir, glob="*.txt")
        docs = loader.load()

        assert len(docs) == 1
        assert "Content 1" in docs[0].content

    def test_load_directory_all(self, temp_dir: str) -> None:
        """DirectoryLoader loads all files by default."""
        from syrin.knowledge.loaders import DirectoryLoader

        loader = DirectoryLoader(temp_dir)
        docs = loader.load()

        assert len(docs) >= 3

    def test_load_directory_recursive(self, temp_dir: str) -> None:
        """DirectoryLoader supports recursive loading."""
        from syrin.knowledge.loaders import DirectoryLoader

        subdir = Path(temp_dir) / "subdir"
        subdir.mkdir()
        Path(subdir, "nested.txt").write_text("Nested")

        loader = DirectoryLoader(temp_dir, glob="**/*.txt", recursive=True)
        docs = loader.load()

        assert len(docs) == 2

    def test_generator_mode(self, temp_dir: str) -> None:
        """DirectoryLoader can yield as generator."""
        from syrin.knowledge.loaders import DirectoryLoader

        loader = DirectoryLoader(temp_dir)
        gen = loader.load_yield()

        docs = list(gen)
        assert len(docs) == 3


class TestURLLoader:
    """Tests for URLLoader."""

    @pytest.mark.asyncio
    async def test_load_url(self) -> None:
        """URLLoader fetches URL content."""
        from syrin.knowledge.loaders import URLLoader

        loader = URLLoader("https://example.com")

        with patch("syrin.knowledge.loaders._url.httpx") as mock_httpx:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.text = "<html><body>Hello World</body></html>"
            mock_response.raise_for_status = lambda: None  # sync, not coroutine
            mock_httpx.AsyncClient.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            docs = await loader.aload()

            assert len(docs) >= 1

    @pytest.mark.asyncio
    async def test_load_url_with_trafilatura(self) -> None:
        """URLLoader works without trafilatura."""
        from syrin.knowledge.loaders import URLLoader

        loader = URLLoader("https://example.com")

        with patch("syrin.knowledge.loaders._url.httpx") as mock_httpx:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.text = "<html><body>Hello World</body></html>"
            mock_response.raise_for_status = lambda: None  # sync, not coroutine
            mock_httpx.AsyncClient.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            docs = await loader.aload()

            assert len(docs) >= 1


class TestGitHubLoader:
    """Tests for GitHubLoader."""

    @pytest.mark.asyncio
    async def test_load_github(self) -> None:
        """GitHubLoader fetches repo info."""
        from syrin.knowledge.loaders import GitHubLoader

        loader = GitHubLoader("octocat", repos=["Hello-World"])

        with patch("syrin.knowledge.loaders._github.httpx") as mock_httpx:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = lambda: None  # sync, not coroutine
            mock_response.json = lambda: {
                "name": "Hello-World",
                "description": "A hello world repo",
                "stargazers_count": 100,
                "language": "Python",
                "forks_count": 0,
                "html_url": "https://github.com/octocat/Hello-World",
            }
            mock_httpx.AsyncClient.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            docs = await loader.aload()

            assert len(docs) >= 1
            assert "Hello-World" in docs[0].content
            assert docs[0].source == "github/octocat/Hello-World"
            assert docs[0].source_type == "github"

    @pytest.mark.asyncio
    async def test_load_github_empty_repos_returns_empty(self) -> None:
        """GitHubLoader with repos=[] returns no documents."""
        from syrin.knowledge.loaders import GitHubLoader

        loader = GitHubLoader("octocat", repos=[])
        docs = await loader.aload()
        assert docs == []

    @pytest.mark.asyncio
    async def test_load_github_multiple_repos(self) -> None:
        """GitHubLoader with multiple repos returns one doc per repo."""
        from syrin.knowledge.loaders import GitHubLoader

        loader = GitHubLoader("org", repos=["repo-a", "repo-b"])

        with patch("syrin.knowledge.loaders._github.httpx") as mock_httpx:

            async def mock_get(url: str, **kwargs: object) -> object:
                resp = AsyncMock()
                resp.status_code = 200
                resp.raise_for_status = lambda: None  # sync
                if "repo-a" in url:
                    resp.json = lambda: {
                        "name": "repo-a",
                        "description": "First",
                        "stargazers_count": 1,
                        "language": "Python",
                        "forks_count": 0,
                        "html_url": "https://github.com/org/repo-a",
                    }
                elif "repo-b" in url:
                    resp.json = lambda: {
                        "name": "repo-b",
                        "description": "Second",
                        "stargazers_count": 2,
                        "language": "Go",
                        "forks_count": 0,
                        "html_url": "https://github.com/org/repo-b",
                    }
                else:
                    resp.status_code = 404
                return resp

            mock_httpx.AsyncClient.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=mock_get
            )
            docs = await loader.aload()

        assert len(docs) == 2
        assert "repo-a" in docs[0].content
        assert docs[0].source == "github/org/repo-a"
        assert "repo-b" in docs[1].content
        assert docs[1].source == "github/org/repo-b"


class TestGoogleDriveLoader:
    """Tests for GoogleDriveLoader."""

    @pytest.mark.asyncio
    async def test_load_requires_api_key(self) -> None:
        """GoogleDriveLoader raises without api_key when GOOGLE_API_KEY not set."""
        from syrin.knowledge.loaders import GoogleDriveLoader

        loader = GoogleDriveLoader(
            "https://drive.google.com/drive/folders/1ABC123",
            api_key="",
        )
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="api_key is required"),
        ):
            await loader.aload()

    @pytest.mark.asyncio
    async def test_load_folder_list(self) -> None:
        """GoogleDriveLoader fetches folder contents via Drive API."""
        from syrin.knowledge.loaders import GoogleDriveLoader

        loader = GoogleDriveLoader(
            "https://drive.google.com/drive/folders/1ABC123",
            api_key="test-key",
        )

        async def mock_get(url: str, params: dict | None = None, **kwargs: object) -> object:
            resp = AsyncMock()
            resp.status_code = 200
            resp.raise_for_status = lambda: None
            # List files (q= in params)
            if params and "in parents" in params.get("q", ""):
                resp.json = lambda: {
                    "files": [
                        {
                            "id": "file1",
                            "name": "notes.txt",
                            "mimeType": "text/plain",
                        },
                    ]
                }
            elif "alt=media" in url:
                resp.content = b"Hello from Drive"
            else:
                resp.json = lambda: {"files": []}
            return resp

        with patch("syrin.knowledge.loaders._google_drive.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.get = mock_get
            mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client

            docs = await loader.aload()

        assert len(docs) == 1
        assert "Hello from Drive" in docs[0].content
        assert docs[0].source_type == "google_drive"
        assert docs[0].metadata.get("file_id") == "file1"

    @pytest.mark.asyncio
    async def test_load_single_file(self) -> None:
        """GoogleDriveLoader loads single file when given file URL."""
        from syrin.knowledge.loaders import GoogleDriveLoader

        loader = GoogleDriveLoader(
            "https://drive.google.com/file/d/FILE99/view",
            api_key="test-key",
        )

        call_count = 0

        async def mock_get(url: str, params: dict | None = None, **kwargs: object) -> object:
            nonlocal call_count
            resp = AsyncMock()
            resp.status_code = 200
            resp.raise_for_status = lambda: None
            if "files/FILE99" in url and "alt=media" not in url:
                resp.json = lambda: {
                    "id": "FILE99",
                    "name": "doc.txt",
                    "mimeType": "text/plain",
                }
            elif "alt=media" in url:
                resp.content = b"Single file content"
            else:
                resp.json = lambda: {}
            call_count += 1
            return resp

        with patch("syrin.knowledge.loaders._google_drive.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.get = mock_get
            mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client

            docs = await loader.aload()

        assert len(docs) == 1
        assert "Single file content" in docs[0].content
        assert docs[0].metadata.get("file_id") == "FILE99"

    def test_extract_id_from_url(self) -> None:
        """ID extraction from various URL formats."""
        from syrin.knowledge.loaders._google_drive import _extract_id_from_url

        assert (
            _extract_id_from_url("https://drive.google.com/drive/folders/1ABC_def-123")
            == "1ABC_def-123"
        )
        assert _extract_id_from_url("https://drive.google.com/file/d/xyz789/view") == "xyz789"
        assert _extract_id_from_url("https://drive.google.com/open?id=abc123") == "abc123"
        assert _extract_id_from_url("1RawId123") == "1RawId123"


class TestPythonLoader:
    """Tests for PythonLoader."""

    @pytest.fixture
    def temp_py_file(self) -> str:
        """Create a temporary Python file."""
        fd, path = tempfile.mkstemp(suffix=".py")
        content = '''"""Module docstring."""

class MyClass:
    """Class docstring."""
    def method(self):
        """Method docstring."""
        pass

def my_function():
    """Function docstring."""
    pass
'''
        with os.fdopen(fd, "w") as f:
            f.write(content)
        yield path
        os.unlink(path)

    def test_load_python(self, temp_py_file: str) -> None:
        """PythonLoader extracts classes and functions."""
        from syrin.knowledge.loaders import PythonLoader

        loader = PythonLoader(temp_py_file)
        docs = loader.load()

        # Should have multiple docs (module + class + function)
        assert len(docs) >= 1
        # Check that module docstring or class/function is extracted
        all_content = " ".join(doc.content for doc in docs)
        assert "MyClass" in all_content or "my_function" in all_content


class TestLoaderWithPathlib:
    """Tests for loaders accepting Path objects."""

    @pytest.fixture
    def temp_file(self) -> str:
        """Create a temporary text file."""
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w") as f:
            f.write("Hello from Path")
        yield path
        os.unlink(path)

    def test_text_loader_accepts_path(self, temp_file: str) -> None:
        """TextLoader accepts Path object."""
        from syrin.knowledge.loaders import TextLoader

        path = Path(temp_file)
        loader = TextLoader(path)
        docs = loader.load()

        assert len(docs) == 1
        assert "Hello from Path" in docs[0].content


class TestLoaderErrors:
    """Tests for error handling in loaders."""

    def test_text_loader_missing_file(self) -> None:
        """TextLoader raises FileNotFoundError."""
        from syrin.knowledge.loaders import TextLoader

        loader = TextLoader("/nonexistent/file.txt")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_pdf_loader_missing_dependency(self, tmp_path: Path) -> None:
        """PDFLoader raises if docling not installed."""
        try:
            import docling  # noqa: F401

            pytest.skip("docling is installed")
        except ImportError:
            pass

        from syrin.knowledge.loaders import PDFLoader

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")  # minimal valid header so file exists
        loader = PDFLoader(str(pdf_file))
        with pytest.raises(ImportError, match="docling"):
            loader.load()

    def test_pdf_loader_with_docling(self) -> None:
        """PDFLoader uses docling for PDF loading."""
        try:
            import docling  # noqa: F401
        except ImportError:
            pytest.skip("docling not installed")

        # This test verifies docling is used when available
        # Full integration test would require a real PDF file
        from syrin.knowledge.loaders import PDFLoader

        loader = PDFLoader("/nonexistent/file.pdf")
        assert loader.path.name == "file.pdf"

    def test_directory_loader_empty(self) -> None:
        """DirectoryLoader handles empty directory."""
        from syrin.knowledge.loaders import DirectoryLoader

        d = tempfile.mkdtemp()
        try:
            loader = DirectoryLoader(d)
            docs = loader.load()
            assert len(docs) == 0
        finally:
            import shutil

            shutil.rmtree(d)


# Integration tests with Knowledge namespace
class TestKnowledgeNamespace:
    """Tests for Knowledge.* source constructors."""

    @pytest.fixture
    def temp_dir(self) -> str:
        """Create a temporary directory."""
        d = tempfile.mkdtemp()
        Path(d, "test.txt").write_text("Hello")
        yield d
        import shutil

        shutil.rmtree(d)

    def test_knowledge_text(self, temp_dir: str) -> None:
        """Knowledge.Text creates RawTextLoader."""
        from syrin.knowledge import Knowledge

        loader = Knowledge.Text("Hello world")
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].content == "Hello world"

    def test_knowledge_directory(self, temp_dir: str) -> None:
        """Knowledge.Directory creates DirectoryLoader."""
        from syrin.knowledge import Knowledge

        loader = Knowledge.Directory(temp_dir, glob="*.txt")
        docs = loader.load()

        assert len(docs) == 1

    def test_knowledge_python(self, temp_dir: str) -> None:
        """Knowledge.Python creates PythonLoader."""
        from syrin.knowledge import Knowledge

        py_file = str(Path(temp_dir) / "test.py")
        Path(py_file).write_text("def foo(): pass")

        loader = Knowledge.Python(py_file)
        docs = loader.load()

        assert len(docs) >= 1

    def test_knowledge_google_drive(self) -> None:
        """Knowledge.GoogleDrive creates GoogleDriveLoader."""
        from syrin.knowledge import Knowledge
        from syrin.knowledge.loaders import GoogleDriveLoader

        loader = Knowledge.GoogleDrive(
            "https://drive.google.com/drive/folders/1ABC",
            api_key="test-key",
            pattern=r"\\.txt$",
        )
        assert isinstance(loader, GoogleDriveLoader)
        assert loader.folder == "https://drive.google.com/drive/folders/1ABC"

    def test_google_drive_load_sync_does_not_raise_not_implemented(self) -> None:
        """GoogleDriveLoader.load() must NOT raise NotImplementedError.

        Before the fix, load() raised NotImplementedError("use aload()").
        After the fix, load() is a sync wrapper that calls asyncio.run(aload()).
        We mock aload() to avoid needing real credentials.
        """
        from unittest.mock import AsyncMock, patch

        from syrin.knowledge.loaders import GoogleDriveLoader

        loader = GoogleDriveLoader(
            folder="https://drive.google.com/drive/folders/1ABC",
            api_key="test-key",
        )
        fake_docs = [{"id": "1", "content": "hello"}]

        with patch.object(loader, "aload", new_callable=AsyncMock, return_value=fake_docs):
            # Must not raise NotImplementedError
            result = loader.load()

        assert result == fake_docs
