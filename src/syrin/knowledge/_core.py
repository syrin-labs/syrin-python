"""Knowledge module for RAG - document loading, chunking, and retrieval.

This module provides the Knowledge class for declarative knowledge base management.

Example:
    from syrin import Knowledge, KnowledgeBackend
    from syrin.embedding import Embedding

    knowledge = Knowledge(
        sources=[
            Knowledge.PDF("./resume.pdf"),
            Knowledge.Markdown("./about.md"),
            Knowledge.Text("I have 5 years of Python experience."),
        ],
        backend=KnowledgeBackend.POSTGRES,
        embedding=Embedding.OpenAI("text-embedding-3-small"),
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

from syrin.enums import Hook, KnowledgeBackend
from syrin.knowledge._agentic import AgenticRAGConfig as _AgenticRAGConfig
from syrin.knowledge._chunker import Chunk, Chunker, ChunkMetadata, ChunkStrategy
from syrin.knowledge._chunker import ChunkConfig as _ChunkConfig
from syrin.knowledge._document import Document, DocumentMetadata
from syrin.knowledge._grounding import GroundedFact
from syrin.knowledge._grounding import GroundingConfig as _GroundingConfig
from syrin.knowledge._hybrid import CancellableIngestTask, CodeChunk, HybridSearchConfig
from syrin.knowledge._loader import DocumentLoader
from syrin.knowledge._store import MetadataFilter, SearchResult
from syrin.knowledge.chunkers import get_chunker
from syrin.knowledge.loaders import (
    CSVLoader,
    DirectoryLoader,
    DoclingLoader,
    DOCXLoader,
    ExcelLoader,
    GitHubLoader,
    GoogleDriveLoader,
    JSONLoader,
    MarkdownLoader,
    PDFLoader,
    PythonLoader,
    RawTextLoader,
    TextLoader,
    URLLoader,
    YAMLLoader,
)
from syrin.knowledge.stores import get_knowledge_store

if TYPE_CHECKING:
    from syrin.budget import BudgetTracker
    from syrin.embedding._protocol import EmbeddingProvider
    from syrin.events import EventContext
    from syrin.knowledge._store import KnowledgeStore

_log = logging.getLogger(__name__)

__all__ = [
    "GroundedFact",
    "Chunk",
    "ChunkMetadata",
    "ChunkStrategy",
    "Chunker",
    "CodeChunk",
    "Document",
    "DocumentMetadata",
    "DocumentLoader",
    "DirectoryLoader",
    "DoclingLoader",
    "CSVLoader",
    "CancellableIngestTask",
    "DOCXLoader",
    "ExcelLoader",
    "get_chunker",
    "GitHubLoader",
    "GoogleDriveLoader",
    "HybridSearchConfig",
    "JSONLoader",
    "Knowledge",
    "MarkdownLoader",
    "PDFLoader",
    "PythonLoader",
    "RawTextLoader",
    "TextLoader",
    "URLLoader",
    "YAMLLoader",
]


def _default_sqlite_path() -> str:
    """Default path for SQLite backend: ~/.syrin/knowledge.db."""
    return os.path.expanduser("~/.syrin/knowledge.db")


def _content_hash(content: str) -> str:
    """Stable hash of chunk content for deduplication."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _deduplicate_search_results(results: list[SearchResult]) -> list[SearchResult]:
    """Remove duplicate results by content hash, preserving order and re-ranking."""
    seen: set[str] = set()
    deduped: list[SearchResult] = []
    for r in results:
        h = _content_hash(r.chunk.content)
        if h in seen:
            continue
        seen.add(h)
        deduped.append(SearchResult(chunk=r.chunk, score=r.score, rank=len(deduped) + 1))
    return deduped


class Knowledge:
    """Declarative knowledge base for AI agents.

    Manages the full pipeline: load → chunk → embed → store → search.
    Attach to an Agent via `knowledge=` parameter.

    Example:
        knowledge = Knowledge(
            sources=[
                Knowledge.PDF("./resume.pdf"),
                Knowledge.Markdown("./about.md"),
                Knowledge.Text("I have 8 years of Python experience."),
            ],
            embedding=Embedding.OpenAI("text-embedding-3-small"),
        )

        class MyAgent(Agent):
            knowledge = knowledge
    """

    def __init__(
        self,
        sources: list[DocumentLoader],
        *,
        embedding: EmbeddingProvider | None = None,
        backend: KnowledgeBackend = KnowledgeBackend.SQLITE,
        connection_url: str | None = None,
        path: str | None = None,
        collection: str = "default",
        chunk_strategy: ChunkStrategy = ChunkStrategy.AUTO,
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        chunk_min_size: int = 50,
        chunk_preserve_tables: bool = True,
        chunk_preserve_code_blocks: bool = True,
        chunk_preserve_headers: bool = True,
        chunk_similarity_threshold: float = 0.5,
        chunk_language: str | None = None,
        top_k: int = 5,
        score_threshold: float = 0.3,
        auto_sync: bool = False,
        sync_interval: int = 0,
        agentic: bool = False,
        agentic_max_iterations: int = 3,
        agentic_decompose: bool = True,
        agentic_grade_results: bool = True,
        agentic_relevance_threshold: float = 0.5,
        agentic_web_fallback: bool = False,
        grounding_enabled: bool = False,
        grounding_confidence: float = 0.7,
        grounding_extract_facts: bool = True,
        grounding_cite_sources: bool = True,
        grounding_flag_missing: bool = True,
        grounding_verify: bool = True,
        grounding_max_facts: int | None = 15,
        grounding_max_chunk_preview: int = 800,
        inject_system_prompt: bool = True,
        emit: Callable[[str, EventContext], None] | None = None,
        get_budget_tracker: Callable[[], object | None] | None = None,
        get_model: Callable[[], object | None] | None = None,
    ) -> None:
        """Create a Knowledge orchestrator.

        Args:
            sources: Document loaders (Knowledge.PDF, Knowledge.Text, etc.).
            embedding: Embedding provider (required). Use Embedding.OpenAI or Embedding.Ollama.
            backend: Vector store backend. Default SQLITE.
            connection_url: Postgres connection URL (required for POSTGRES).
            path: File path for SQLite or Chroma. Default ~/.syrin/knowledge.db for SQLite.
            collection: Collection/table name for vector stores.
            chunk_strategy: Chunking strategy (AUTO, RECURSIVE, MARKDOWN, CODE, etc.).
            chunk_size: Target tokens per chunk. Default 512.
            chunk_overlap: Token overlap between consecutive chunks. Default 0.
            chunk_min_size: Minimum chunk size in tokens; smaller chunks are dropped. Default 50.
            chunk_preserve_tables: Prevent splitting markdown/HTML tables. Default True.
            chunk_preserve_code_blocks: Prevent splitting fenced code blocks. Default True.
            chunk_preserve_headers: Keep heading hierarchy in chunk metadata. Default True.
            chunk_similarity_threshold: Similarity threshold for SEMANTIC strategy. Default 0.5.
            chunk_language: Source language for CODE strategy (e.g. "python"). Default None.
            top_k: Max results per search. Default 5.
            score_threshold: Minimum similarity score for search results. Default 0.3.
            auto_sync: Enable file watching for automatic document reload.
            sync_interval: Seconds between sync checks (0 = file watcher).
            agentic: Enable agentic retrieval (decompose, grade, refine, verify tools).
            agentic_max_iterations: Max refinement loops in agentic retrieval. Default 3.
            agentic_decompose: Auto-decompose complex queries into sub-queries. Default True.
            agentic_grade_results: Use LLM to grade result relevance. Default True.
            agentic_relevance_threshold: Min score to accept results without refinement. Default 0.5.
            agentic_web_fallback: Suggest web search if KB fails. Default False.
            grounding_enabled: Enable grounding layer (fact extraction/verification). Default False.
            grounding_confidence: Min confidence to include a fact (0–1). Default 0.7.
            grounding_extract_facts: Extract structured facts from chunks. Default True.
            grounding_cite_sources: Attach source doc + page to each fact. Default True.
            grounding_flag_missing: Flag when expected data is missing. Default True.
            grounding_verify: Verify each fact before including. Default True.
            grounding_max_facts: Max facts per search (None = no limit). Default 15.
            grounding_max_chunk_preview: Max chars per chunk for fact extraction. Default 800.
            inject_system_prompt: Inject knowledge context into agent system prompt.
            emit: Hook emitter (set automatically when attached to agent).
            get_budget_tracker: Callable to get BudgetTracker for embedding cost tracking.
            get_model: Callable to get Model for decomposition/grading.
        """
        if embedding is None:
            raise ValueError(
                "embedding is required. Use Embedding.OpenAI(api_key=...) or "
                "Embedding.Ollama() for local embeddings."
            )
        if not sources:
            raise ValueError("sources must be a non-empty list of DocumentLoaders")
        if auto_sync:
            import warnings

            warnings.warn(
                "Knowledge(auto_sync=True) has no effect — file watching is not currently supported. "
                "Call knowledge.sync() manually to reload changed documents.",
                UserWarning,
                stacklevel=2,
            )

        self._agentic = agentic
        self._agentic_config = (
            _AgenticRAGConfig(
                max_search_iterations=agentic_max_iterations,
                decompose_complex=agentic_decompose,
                grade_results=agentic_grade_results,
                relevance_threshold=agentic_relevance_threshold,
                web_fallback=agentic_web_fallback,
            )
            if agentic
            else None
        )
        self._grounding_config = (
            _GroundingConfig(
                enabled=True,
                extract_facts=grounding_extract_facts,
                cite_sources=grounding_cite_sources,
                flag_missing=grounding_flag_missing,
                verify_before_use=grounding_verify,
                confidence_threshold=grounding_confidence,
                max_facts=grounding_max_facts,
                max_chunk_preview=grounding_max_chunk_preview,
            )
            if grounding_enabled
            else None
        )
        self._get_model = get_model

        self._embedding = embedding
        self._sources = list(sources)
        self._chunk_config = _ChunkConfig(
            strategy=chunk_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=chunk_min_size,
            preserve_tables=chunk_preserve_tables,
            preserve_code_blocks=chunk_preserve_code_blocks,
            preserve_headers=chunk_preserve_headers,
            similarity_threshold=chunk_similarity_threshold,
            language=chunk_language,
        )
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._inject_system_prompt = inject_system_prompt
        self._emit = emit
        self._get_budget_tracker = get_budget_tracker
        self._ingested = False
        self._document_ids: set[str] = set()

        if backend == KnowledgeBackend.SQLITE and path is None:
            path = _default_sqlite_path()
        self._path = path

        collection_for_store = "syrin_knowledge" if collection == "default" else collection
        store: KnowledgeStore = get_knowledge_store(
            backend,
            embedding_dimensions=embedding.dimensions,
            connection_url=connection_url,
            path=path,
            collection=collection_for_store,
        )
        self._store = store

    @property
    def embedding(self) -> EmbeddingProvider:
        """Embedding provider (read-only)."""
        return self._embedding

    async def ingest(self) -> None:
        """Load, chunk, embed, and store documents. Lazy-invoked on first search."""
        self._emit_hook(Hook.KNOWLEDGE_INGEST_START, {"source_count": len(self._sources)})
        try:
            documents: list[Document] = []
            for loader in self._sources:
                if hasattr(loader, "aload") and callable(loader.aload):
                    docs = await loader.aload()
                else:
                    loop = asyncio.get_event_loop()
                    docs = await loop.run_in_executor(None, loader.load)
                documents.extend(docs)

            if not documents:
                self._emit_hook(Hook.KNOWLEDGE_INGEST_END, {"chunk_count": 0})
                self._ingested = True
                return

            chunker = get_chunker(self._chunk_config)
            chunks = chunker.chunk(documents)

            if not chunks:
                self._emit_hook(Hook.KNOWLEDGE_INGEST_END, {"chunk_count": 0})
                self._ingested = True
                return

            texts = [c.content for c in chunks]
            bt = self._get_budget_tracker() if self._get_budget_tracker else None
            embeddings = await self._embedding.embed(
                texts,
                budget_tracker=cast("BudgetTracker | None", bt),
            )

            for c in chunks:
                self._document_ids.add(c.document_id)
            await self._store.upsert(chunks, embeddings)
            self._ingested = True
            self._emit_hook(
                Hook.KNOWLEDGE_INGEST_END,
                {"chunk_count": len(chunks), "document_count": len(documents)},
            )
        except Exception as e:
            self._emit_hook(Hook.KNOWLEDGE_INGEST_END, {"error": str(e)})
            raise

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filter: MetadataFilter | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Semantic search. Triggers lazy ingest on first call."""
        if not self._ingested:
            await self.ingest()

        self._emit_hook(Hook.KNOWLEDGE_SEARCH_START, {"query": query[:200]})
        k = top_k if top_k is not None else self._top_k
        thresh = score_threshold if score_threshold is not None else self._score_threshold

        bt = self._get_budget_tracker() if self._get_budget_tracker else None
        query_embeddings = await self._embedding.embed(
            [query],
            budget_tracker=cast("BudgetTracker | None", bt),
        )
        query_emb = query_embeddings[0]
        results = await self._store.search(
            query_emb,
            top_k=k,
            filter=filter,
            score_threshold=thresh,
        )
        # Deduplicate by content hash so overlapping chunks do not repeat
        results = _deduplicate_search_results(results)
        max_preview = 200
        results_preview = [
            {
                "rank": r.rank,
                "score": round(r.score, 3),
                "content": r.chunk.content[:max_preview]
                + ("..." if len(r.chunk.content) > max_preview else ""),
            }
            for r in results[:10]
        ]
        self._emit_hook(
            Hook.KNOWLEDGE_SEARCH_END,
            {"result_count": len(results), "results": results_preview},
        )
        return results

    def add_source(self, loader: DocumentLoader) -> None:
        """Add a source. Call ingest() to load it."""
        self._sources.append(loader)
        self._emit_hook(Hook.KNOWLEDGE_SOURCE_ADDED, {"source_count": len(self._sources)})

    async def remove_source(self, loader: DocumentLoader) -> None:
        """Remove a source and delete its chunks from the store."""
        if loader in self._sources:
            self._sources.remove(loader)
            self._emit_hook(Hook.KNOWLEDGE_SOURCE_REMOVED, {"source_count": len(self._sources)})
        if hasattr(loader, "aload") and callable(loader.aload):
            docs = await loader.aload()
        else:
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, loader.load)
        for doc in docs:
            await self._store.delete(document_id=doc.source)
            self._document_ids.discard(doc.source)

    async def clear(self) -> None:
        """Delete all chunks from the store."""
        for doc_id in list(self._document_ids):
            await self._store.delete(document_id=doc_id)
        self._document_ids.clear()

    async def stats(self) -> dict[str, int]:
        """Return chunk count and source count."""
        count = await self._store.count()
        return {"chunk_count": count, "source_count": len(self._sources)}

    def _attach_to_agent(
        self,
        emit: Callable[[str, EventContext], None],
        get_budget_tracker: Callable[[], object | None] | None = None,
        get_model: Callable[[], object | None] | None = None,
    ) -> None:
        """Wire emit, budget tracker, and model when attached to an agent. Called by Agent."""
        self._emit = emit
        self._get_budget_tracker = get_budget_tracker
        self._get_model = get_model

    def get_remote_config_schema(self, section_key: str) -> tuple[object, dict[str, object]]:
        """RemoteConfigurable: return (schema, current_values) for the knowledge section."""
        from syrin.remote._schema import extract_dataclass_schema
        from syrin.remote._types import ConfigSchema, FieldSchema

        if section_key != "knowledge":
            return (ConfigSchema(section="knowledge", class_name="Knowledge", fields=[]), {})
        prefix = "knowledge"
        # Top-level scalars + nested configs
        grounding_children = extract_dataclass_schema(_GroundingConfig, f"{prefix}.grounding")
        agentic_children = extract_dataclass_schema(_AgenticRAGConfig, f"{prefix}.agentic_config")
        agentic_children = [f for f in agentic_children if not f.remote_excluded]
        chunk_children = extract_dataclass_schema(_ChunkConfig, f"{prefix}.chunk_config")
        chunk_children = [f for f in chunk_children if not f.remote_excluded]
        fields: list[FieldSchema] = [
            FieldSchema(name="top_k", path=f"{prefix}.top_k", type="int", default=5),
            FieldSchema(
                name="score_threshold", path=f"{prefix}.score_threshold", type="float", default=0.3
            ),
            FieldSchema(
                name="grounding",
                path=f"{prefix}.grounding",
                type="object",
                default=None,
                children=grounding_children,
            ),
            FieldSchema(
                name="agentic_config",
                path=f"{prefix}.agentic_config",
                type="object",
                default=None,
                children=agentic_children,
            ),
            FieldSchema(
                name="chunk_config",
                path=f"{prefix}.chunk_config",
                type="object",
                default=None,
                children=chunk_children,
            ),
        ]
        schema = ConfigSchema(section="knowledge", class_name="Knowledge", fields=fields)
        # Build current_values from live state
        current: dict[str, object] = {
            f"{prefix}.top_k": self._top_k,
            f"{prefix}.score_threshold": self._score_threshold,
        }
        if self._grounding_config is not None:
            g = self._grounding_config
            for f in grounding_children:
                v = getattr(g, f.name, None)
                if v is not None or f.default is not None:
                    current[f.path] = v if v is not None else f.default
        if self._agentic_config is not None:
            a = self._agentic_config
            for f in agentic_children:
                v = getattr(a, f.name, None)
                if v is not None or f.default is not None:
                    current[f.path] = v if v is not None else f.default
        c = self._chunk_config
        for f in chunk_children:
            v = getattr(c, f.name, None)
            if v is not None or f.default is not None:
                val: object = (
                    v.value
                    if v is not None and hasattr(v, "value") and not hasattr(v, "model_dump")
                    else v
                )
                current[f.path] = val if val is not None else f.default
        return (schema, current)

    def apply_remote_overrides(
        self,
        agent: object,
        pairs: list[tuple[str, object]],
        section_schema: object,
    ) -> None:
        """RemoteConfigurable: apply knowledge overrides to this Knowledge instance."""
        import dataclasses

        from syrin.remote._resolver_helpers import build_nested_update
        from syrin.remote._types import ConfigSchema

        schema = cast("ConfigSchema", section_schema)
        section = getattr(schema, "section", None)
        if section != "knowledge":
            return
        update = build_nested_update(schema, pairs, "knowledge")
        if not update:
            return
        # Apply top-level scalars
        if "top_k" in update:
            object.__setattr__(self, "_top_k", int(cast("int | float | str", update["top_k"])))
        if "score_threshold" in update:
            object.__setattr__(
                self,
                "_score_threshold",
                float(cast("int | float | str", update["score_threshold"])),
            )
        # Merge grounding
        if "grounding" in update and isinstance(update["grounding"], dict):
            g_update = update["grounding"]
            base = self._grounding_config or _GroundingConfig()
            g_dict = dataclasses.asdict(base)
            g_dict.update({k: v for k, v in g_update.items() if v is not None})
            object.__setattr__(self, "_grounding_config", _GroundingConfig(**g_dict))
        # Merge agentic_config
        if "agentic_config" in update and isinstance(update["agentic_config"], dict):
            a_update = update["agentic_config"]
            agentic_base: _AgenticRAGConfig = (
                _AgenticRAGConfig() if self._agentic_config is None else self._agentic_config
            )
            a_dict = dataclasses.asdict(agentic_base)
            a_dict.update({k: v for k, v in a_update.items() if v is not None})
            object.__setattr__(self, "_agentic_config", _AgenticRAGConfig(**a_dict))
        # Merge chunk_config
        if "chunk_config" in update and isinstance(update["chunk_config"], dict):
            c_update = update["chunk_config"]
            c_dict = dataclasses.asdict(self._chunk_config)
            for k, v in c_update.items():
                if v is not None:
                    c_dict[k] = ChunkStrategy(v) if k == "strategy" and isinstance(v, str) else v
            object.__setattr__(self, "_chunk_config", _ChunkConfig(**c_dict))

    def _emit_hook(self, hook: Hook, ctx: dict[str, object]) -> None:
        """Emit hook if emitter configured. Pass Hook enum (not .value) so agent event system captures it."""
        if self._emit is not None:
            from syrin.events import EventContext

            self._emit(hook, EventContext(ctx))

    # -- File Sources --

    @staticmethod
    def Text(content: str, **metadata: object) -> RawTextLoader:
        """Create raw text source.

        Args:
            content: Text content.
            **metadata: Additional metadata (reserved for future use).

        Returns:
            RawTextLoader instance.
        """
        return RawTextLoader(
            content,
            metadata=cast(DocumentMetadata, dict(metadata)) if metadata else None,
        )

    @staticmethod
    def Texts(contents: list[str], **metadata: object) -> RawTextLoader:
        """Create multiple text sources.

        Args:
            contents: List of text contents.
            **metadata: Additional metadata (reserved for future use).

        Returns:
            RawTextLoader instance.
        """
        return RawTextLoader(
            contents,
            metadata=cast(DocumentMetadata, dict(metadata)) if metadata else None,
        )

    @staticmethod
    def TextFile(path: str | Path, **_metadata: object) -> TextLoader:
        """Create text file source.

        Args:
            path: Path to text file.
            **metadata: Additional metadata (reserved for future use).

        Returns:
            TextLoader instance.
        """
        return TextLoader(path)

    @staticmethod
    def Markdown(path: str | Path, **_metadata: object) -> MarkdownLoader:
        """Create Markdown file source.

        Args:
            path: Path to Markdown file.
            **metadata: Additional metadata (reserved for future use).

        Returns:
            MarkdownLoader instance.
        """
        return MarkdownLoader(path)

    @staticmethod
    def Docling(
        path: str | Path,
        *,
        extract_tables: bool = True,
        table_format: str = "markdown",
        ocr: bool = False,
        **_metadata: object,
    ) -> DoclingLoader:
        """Create Docling-powered source (PDF, DOCX, PPTX, XLSX, HTML, images).

        Best-in-class table extraction via IBM Docling. Tables are extracted
        as separate Documents with structured metadata (table_csv, table_html,
        table_markdown).

        Args:
            path: Path to the document file.
            extract_tables: If True, extract tables as separate Documents.
            table_format: Primary format for table content: "markdown", "csv", "html".
            ocr: Enable OCR for scanned documents.

        Returns:
            DoclingLoader instance.

        Requires:
            pip install syrin[docling]
        """
        return DoclingLoader(
            path,
            extract_tables=extract_tables,
            table_format=table_format,
            ocr=ocr,
        )

    @staticmethod
    def DOCX(path: str | Path, **_metadata: object) -> DOCXLoader:
        """Create DOCX file source.

        Uses Docling when available (best table extraction), otherwise
        python-docx. Extracts text and tables with structure preserved.

        Args:
            path: Path to DOCX file.
            **metadata: Additional metadata (reserved for future use).

        Returns:
            DOCXLoader instance.

        Requires:
            pip install syrin[docx] or syrin[docling]
        """
        return DOCXLoader(path, use_docling=True)

    @staticmethod
    def CSV(
        path: str | Path,
        *,
        rows_per_document: int | None = None,
        encoding: str = "utf-8",
        **_metadata: object,
    ) -> CSVLoader:
        """Create CSV file source. No extra deps.

        Args:
            path: Path to CSV file.
            rows_per_document: Rows per Document; None = entire file.
            encoding: File encoding.

        Returns:
            CSVLoader instance.
        """
        return CSVLoader(
            path,
            rows_per_document=rows_per_document,
            encoding=encoding,
        )

    @staticmethod
    def Excel(
        path: str | Path,
        *,
        sheets: list[str] | None = None,
        **_metadata: object,
    ) -> ExcelLoader:
        """Create Excel file source. Each sheet becomes Document(s).

        Args:
            path: Path to XLSX file.
            sheets: Sheet names to load; None = all.

        Returns:
            ExcelLoader instance.

        Requires:
            pip install syrin[excel]
        """
        return ExcelLoader(path, sheets=sheets)

    @staticmethod
    def PDF(path: str | Path, **_metadata: object) -> PDFLoader:
        """Create PDF file source.

        Args:
            path: Path to PDF file.
            **metadata: Additional metadata (reserved for future use).

        Returns:
            PDFLoader instance.
        """
        return PDFLoader(path)

    @staticmethod
    def Python(path: str | Path, **_metadata: object) -> PythonLoader:
        """Create Python source file source.

        Args:
            path: Path to Python file.
            **metadata: Additional metadata (reserved for future use).

        Returns:
            PythonLoader instance.
        """
        return PythonLoader(path)

    @staticmethod
    def YAML(path: str | Path, **_metadata: object) -> YAMLLoader:
        """Create YAML file source.

        Args:
            path: Path to YAML file.
            **metadata: Additional metadata (reserved for future use).

        Returns:
            YAMLLoader instance.
        """
        return YAMLLoader(path)

    @staticmethod
    def JSON(
        path: str | Path,
        jq_path: str | None = None,
        **_metadata: object,
    ) -> JSONLoader:
        """Create JSON file source.

        Args:
            path: Path to JSON file.
            jq_path: Optional dot notation path to extract.
            **metadata: Additional metadata (reserved for future use).

        Returns:
            JSONLoader instance.
        """
        return JSONLoader(path, jq_path=jq_path)

    # -- Directory Sources --

    @staticmethod
    def Directory(
        path: str | Path,
        glob: str = "**/*",
        pattern: str | None = None,
        recursive: bool = True,
        **_metadata: object,
    ) -> DirectoryLoader:
        """Create directory source.

        Args:
            path: Path to directory.
            glob: Glob pattern for file matching.
            pattern: Regex pattern as alternative to glob.
            recursive: Whether to search recursively.
            **metadata: Additional metadata (reserved for future use).

        Returns:
            DirectoryLoader instance.
        """
        return DirectoryLoader(
            path,
            glob=glob,
            pattern=pattern,
            recursive=recursive,
        )

    # -- Remote Sources --

    @staticmethod
    def URL(url: str, **_metadata: object) -> URLLoader:
        """Create URL source.

        Args:
            url: URL to fetch.
            **metadata: Additional metadata (reserved for future use).

        Returns:
            URLLoader instance.
        """
        return URLLoader(url)

    @staticmethod
    def GitHub(
        username: str,
        repos: list[str] | None = None,
        include_readme: bool = True,
        include_code: bool = False,
        token: str | None = None,
        **_metadata: object,
    ) -> GitHubLoader:
        """Create GitHub repository source.

        Args:
            username: GitHub username or organization.
            repos: List of repository names (None for all repos).
            include_readme: Include README content.
            include_code: Include code files.
            token: Optional GitHub API token.
            **metadata: Additional metadata (reserved for future use).

        Returns:
            GitHubLoader instance.
        """
        return GitHubLoader(
            username,
            repos=repos,
            include_readme=include_readme,
            include_code=include_code,
            token=token,
        )

    @staticmethod
    def GoogleDrive(
        folder: str,
        *,
        recursive: bool = True,
        pattern: str | None = None,
        allowed_folder: list[str] | None = None,
        excluded_folder: list[str] | None = None,
        api_key: str | None = None,
        **_metadata: object,
    ) -> GoogleDriveLoader:
        """Create Google Drive folder source (public links only).

        Loads documents from a public Google Drive folder. Folder must be shared
        as "Anyone with the link can view". Uses Google Drive API v3 with an
        API key (no OAuth).

        Args:
            folder: Google Drive folder URL, folder ID, or single file URL.
            recursive: If True, traverse subfolders. Default True.
            pattern: Regex pattern for file names (e.g. r"\\.(txt|md)$").
            allowed_folder: Only include files from these folder names or IDs.
            excluded_folder: Exclude these folder names or IDs from traversal.
            api_key: Google API key. Falls back to GOOGLE_API_KEY env.
            **metadata: Additional metadata (reserved for future use).

        Returns:
            GoogleDriveLoader instance.

        Requires:
            Google Cloud API key with Drive API enabled.
        """
        return GoogleDriveLoader(
            folder,
            recursive=recursive,
            pattern=pattern,
            allowed_folder=allowed_folder,
            excluded_folder=excluded_folder,
            api_key=api_key,
        )
