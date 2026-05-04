---
title: Knowledge Loaders
description: Load documents from any source into Syrin's Knowledge Pool
weight: 191
---

## Where Does Your Data Come From?

You can't search what you haven't loaded. The first step in any RAG pipeline is getting your documents into the system. That's what **loaders** do.

Syrin provides loaders for virtually every document format and source:

**File Loaders**
- PDF, DOCX, Markdown, Text
- CSV, Excel

**Web Loaders**
- URL (fetch web pages)
- Google Drive (shared folders)
- GitHub (repositories)

**Code Loaders**
- Python, JavaScript, and other languages

**Structured Data Loaders**
- JSON, YAML
- Directory (batch load)

## The Loading Pipeline

Every loader follows the same pattern:

```python
from syrin.knowledge import Knowledge

loader = Knowledge.PDF("./document.pdf")
documents = loader.load()
```

Each loader returns a list of `Document` objects:

```python
class Document:
    content: str          # The text content
    source: str           # Identifier (file path, URL, etc.)
    source_type: str     # "pdf", "markdown", "url", etc.
    metadata: dict        # Additional info (page numbers, etc.)
```

## File-Based Loaders

### PDF

Extract text from PDF files:

```python
from syrin.knowledge import Knowledge

loader = Knowledge.PDF("./contract.pdf")
documents = loader.load()
```

For advanced PDF processing (tables, OCR), use **Docling**:

```python
loader = Knowledge.Docling(
    "./contract.pdf",
    extract_tables=True,  # Extract tables as separate documents
    ocr=True,            # Enable OCR for scanned documents
)
```

### DOCX

Extract text from Word documents:

```python
loader = Knowledge.DOCX("./report.docx")
documents = loader.load()
```

### Markdown

Extract Markdown content:

```python
loader = Knowledge.Markdown("./README.md")
documents = loader.load()
```

### Text Files

Load plain text:

```python
loader = Knowledge.TextFile("./notes.txt")
documents = loader.load()
```

### Raw Text

For programmatically created content:

```python
loader = Knowledge.Text("This is my product description.")
documents = loader.load()

# Multiple texts at once
loader = Knowledge.Texts([
    "First point about the product.",
    "Second point about features.",
    "Third point about pricing.",
])
documents = loader.load()
```

## Spreadsheet Loaders

### CSV

```python
loader = Knowledge.CSV(
    "./sales_data.csv",
    rows_per_document=100,  # Rows per document (None = entire file)
)
```

### Excel

```python
loader = Knowledge.Excel(
    "./report.xlsx",
    sheets=["Q1 Sales", "Q2 Sales"],  # Specific sheets, None = all
)
```

## Code Loaders

### Python

Preserves code structure:

```python
loader = Knowledge.Python("./main.py")
documents = loader.load()
```

## Structured Data Loaders

### JSON

```python
# Load entire JSON
loader = Knowledge.JSON("./config.json")

# Extract specific path with jq-style syntax
loader = Knowledge.JSON("./api_response.json", jq_path="results.items")
```

### YAML

```python
loader = Knowledge.YAML("./settings.yaml")
```

## Web Loaders

Web loaders (`URL`, `GitHub`, `GoogleDrive`) are async-only — they make network requests and
return results via `await loader.aload()`. Pass them directly to `Knowledge(sources=[...])` and
Syrin handles the async loading automatically.

### URL

Fetch content from websites:

```python
# Pass to Knowledge (recommended — handles async internally)
knowledge = Knowledge(
    sources=[Knowledge.URL("https://example.com/docs")],
    embedding=Embedding.OpenAI("text-embedding-3-small"),
)

# Direct async use
loader = Knowledge.URL("https://example.com/docs")
documents = await loader.aload()
```

### Google Drive

Load documents from shared Google Drive folders:

```python
loader = Knowledge.GoogleDrive(
    "https://drive.google.com/drive/folders/...",  # Folder URL or ID
    recursive=True,                    # Include subfolders
    pattern=r"\.pdf$",                  # Regex filter
    api_key="your-google-api-key",     # Optional API key
)
documents = await loader.aload()
```

Requirements:
- Google Cloud API key with Drive API enabled
- Documents shared as "Anyone with the link"

### GitHub

Load repositories:

```python
loader = Knowledge.GitHub(
    username="your-org",
    repos=["repo1", "repo2"],           # None = all repos
    include_readme=True,                # Include README files
    include_code=False,                 # Include code files
    token="ghp_...",                   # Optional GitHub token
)
documents = await loader.aload()
```

## Directory Loaders

### Directory

Load all matching files from a folder:

```python
loader = Knowledge.Directory(
    "./docs",
    glob="**/*.md",                    # Glob pattern (default: **/*)
    pattern=r"v\d+\.\d+",              # Regex alternative
    recursive=True,                    # Include subdirectories
)
```

This is useful for:
- Loading entire documentation sites
- Batch processing file collections
- Incremental updates

## Advanced: Docling Loader

**Docling** is a best-in-class document loader that handles:

- PDF (including scanned)
- DOCX
- PPTX
- XLSX
- HTML
- Images (with OCR)

```python
loader = Knowledge.Docling(
    "./presentation.pptx",
    extract_tables=True,       # Tables become separate documents
    table_format="markdown",    # "markdown", "csv", or "html"
    ocr=False,                 # Enable OCR for images
)
```

Tables are extracted with structured metadata:

```python
documents = loader.load()
for doc in documents:
    if doc.metadata.get("table_csv"):
        print(f"Table at {doc.source}:")
        print(doc.metadata["table_csv"])
```

## Custom Loaders

Create your own loader by implementing the `DocumentLoader` protocol:

```python
from syrin.knowledge import DocumentLoader
from syrin.knowledge._document import Document

class MyLoader(DocumentLoader):
    def load(self) -> list[Document]:
        """Load documents synchronously."""
        # Your loading logic here
        return [
            Document(
                content="Document content",
                source="my-source",
                source_type="custom",
                metadata={"key": "value"},
            )
        ]
    
    async def aload(self) -> list[Document]:
        """Load documents asynchronously."""
        # Async version
        return self.load()
```

Then use it with Knowledge:

```python
knowledge = Knowledge(
    sources=[MyLoader(), ...],
    embedding=embedding,
)
```

## Using Loaders with Knowledge

Pass loaders to the Knowledge constructor:

```python
from syrin import Knowledge
from syrin.embedding import Embedding

knowledge = Knowledge(
    sources=[
        Knowledge.PDF("./manual.pdf"),
        Knowledge.Markdown("./docs/"),
        Knowledge.URL("https://api.example.com/docs"),
    ],
    embedding=Embedding.OpenAI("text-embedding-3-small"),
)
```

The Knowledge pool handles:
- Loading all sources
- Chunking the content
- Embedding the chunks
- Storing in the vector database

## Loader Options Reference

Simple file loaders — `PDF`, `DOCX`, `Markdown`, `TextFile`, `Text`, and `Texts` — take just a path or string and output text content. `Markdown` preserves Markdown formatting; the others output plain text.

Spreadsheet loaders add options: `CSV` accepts `rows_per_document` and `encoding`; `Excel` accepts a `sheets` list to target specific tabs. Both output tabular text.

`Python` outputs code content with no additional options.

Structured data loaders: `JSON` accepts a `jq_path` for extracting nested values; `YAML` has no additional options. Both output structured text.

`URL` fetches a web page and returns HTML or text content with no additional options.

Multi-file loaders: `Directory` accepts `glob`, `pattern`, and `recursive`; `GitHub` accepts `repos`, `include_readme`, and `include_code`; `GoogleDrive` accepts `recursive`, `pattern`, and `api_key`. All three output mixed content types.

`Docling` is the most capable loader — it outputs text, tables, and images, controlled by `extract_tables`, `table_format`, and `ocr`.

## Performance Tips

### Async Loading

For multiple sources, loading can be slow:

```python
# Synchronous (sequential)
for loader in loaders:
    docs = loader.load()  # Waits for each

# Async (parallel) - Knowledge handles this internally
knowledge = Knowledge(sources=loaders, ...)
await knowledge.ingest()  # Parallel loading
```

### Large Documents

For very large PDFs:

```python
# Use Docling with page-based chunking
loader = Knowledge.Docling(
    "./large_book.pdf",
    extract_tables=True,
)
```

### Selective Loading

Don't load everything:

```python
# Only relevant folders
loader = Knowledge.Directory(
    "./docs",
    glob="**/api/*.md",  # Only API docs
)

# Only recent files (pre-filter with code)
from pathlib import Path
from datetime import datetime, timedelta

recent = [f for f in Path("./docs").glob("**/*.md") 
          if datetime.fromtimestamp(f.stat().st_mtime) > 
             datetime.now() - timedelta(days=7)]

loaders = [Knowledge.TextFile(str(f)) for f in recent]
```

## Loader Surface

The public loader API includes `DirectoryLoader`, `DoclingLoader`, `CSVLoader`, `DOCXLoader`, `ExcelLoader`, `GitHubLoader`, `GoogleDriveLoader`, `JSONLoader`, `MarkdownLoader`, `PDFLoader`, `PythonLoader`, `RawTextLoader`, `TextLoader`, `URLLoader`, and `YAMLLoader`. `DocumentMetadata` is also public when you need to annotate documents before ingestion.

## What's Next?

- [Chunking Strategies](/agent-kit/integrations/knowledge-chunking) — Split documents into optimal chunks
- [Knowledge Pool](/agent-kit/integrations/knowledge-pool) — Complete RAG overview
- [RAG Configuration](/agent-kit/integrations/knowledge-rag) — Configure retrieval

## See Also

- [Document Model](/agent-kit/core/models) — Understanding documents
- [Vector Storage](/agent-kit/integrations/knowledge-pool) — Storing embeddings
- [Embedding Providers](/agent-kit/core/models) — Vector embeddings
