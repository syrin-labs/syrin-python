"""Knowledge module: Document model and document loaders.

Shows the immutable Document (content, source, source_type, metadata),
and how to use loaders directly or via Knowledge.* factory methods.

Run:
    uv run python examples/19_knowledge/loaders_and_document.py
"""

from __future__ import annotations

from pathlib import Path

from syrin.knowledge import Document, Knowledge
from syrin.knowledge.loaders import (
    RawTextLoader,
    TextLoader,
)


def main() -> None:
    # Document is mutable; required: content, source, source_type
    doc = Document(
        content="I have 8 years of Python experience.",
        source="user_provided",
        source_type="text",
        metadata={"index": 0},
    )
    print("Document:", doc.content[:40], "...", "source=", doc.source)

    # Documents can be modified (mutable)
    doc.metadata["updated"] = True
    doc.content += " Specializing in ML."
    print("Updated document:", doc.content[:60], "...")

    # Raw text (no file): single or multiple
    raw = RawTextLoader("One fact.")
    docs = raw.load()
    assert len(docs) == 1
    assert docs[0].source.startswith("user_provided")  # user_provided_1, user_provided_2, etc.
    assert docs[0].source_type == "text"

    raw_multi = RawTextLoader(["Fact A", "Fact B"])
    docs_multi = raw_multi.load()
    assert len(docs_multi) == 2

    # Via Knowledge namespace
    k_text = Knowledge.Text("Inline fact for RAG.")
    k_docs = k_text.load()
    assert k_docs[0].source.startswith("user_provided")  # user_provided_N format

    k_texts = Knowledge.Texts(["Fact 1", "Fact 2"])
    assert len(k_texts.load()) == 2

    # Text file loader (if a file exists)
    tmp = Path("/tmp/syrin_knowledge_demo.txt")
    tmp.write_text("Hello from file.")
    try:
        file_loader = TextLoader(tmp)
        file_docs = file_loader.load()
        assert len(file_docs) == 1
        assert file_docs[0].source == str(tmp)
        assert file_docs[0].source_type == "text"
        print("TextLoader:", file_docs[0].content)
    finally:
        tmp.unlink(missing_ok=True)

    # Knowledge.Directory, .PDF, .Docling, .DOCX, .CSV, .Excel, .GitHub etc.:
    # loader = Knowledge.Directory("./docs/", glob="**/*.md")
    # loader = Knowledge.Docling("report.pdf")   # Best table extraction; requires syrin[docling]
    # loader = Knowledge.DOCX("board.docx")      # Requires syrin[docx] or syrin[docling]
    # loader = Knowledge.CSV("data.csv")         # No extra deps
    # loader = Knowledge.Excel("sheet.xlsx")     # Requires syrin[excel]
    # Async-only loaders (URL, GitHub, GoogleDrive) — use aload() or pass to Knowledge(sources=[])
    # loader = Knowledge.GitHub("org", repos=["repo1", "repo2"])
    # docs = await loader.aload()
    # loader = Knowledge.URL("https://example.com/docs")
    # docs = await loader.aload()

    print("OK: Document and loaders work.")


if __name__ == "__main__":
    main()
