"""Chroma backend for persistent memory storage with vector embeddings."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from syrin.enums import MemoryScope, MemoryType
from syrin.memory.config import MemoryEntry
from syrin.memory.embedding import EmbeddingConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import chromadb
    from chromadb.config import Settings

try:
    import chromadb
    from chromadb.config import Settings

    chroma_available = True
except ImportError:
    chroma_available = False
    chromadb = None  # type: ignore[assignment]
    Settings = None  # type: ignore[assignment,misc]
_chroma = chromadb
_Settings = Settings


class ChromaBackend:
    """Chroma-based storage for memories with vector search.

    Requires: pip install chromadb

    Note: Default _get_embedding uses MD5 hash-based pseudo-embeddings.
    These do NOT provide semantic similarity. Override _get_embedding for
    real semantic search (e.g. sentence-transformers).

    Features:
    - Vector search (override _get_embedding for semantic similarity)
    - Persistent storage
    - Fast similarity search
    - Lightweight
    """

    def __init__(
        self,
        path: str | None = None,
        collection_name: str = "syrin_memory",
        namespace: str | None = None,
        embedding_config: EmbeddingConfig | None = None,
    ) -> None:
        """Initialize Chroma backend.

        Args:
            path: Directory for persistent storage; None for ephemeral.
            collection_name: Collection name for memories.
            namespace: Per-tenant isolation; all ops scoped to this namespace.
            embedding_config: Optional EmbeddingConfig with custom_fn for embeddings.
        """
        if not chroma_available or _chroma is None or _Settings is None:
            raise ImportError("chromadb is not installed. Install with: pip install chromadb")

        self._collection_name = collection_name
        self._namespace = namespace
        self._embedding_config: EmbeddingConfig | None = embedding_config

        # Initialize Chroma client. Use PersistentClient when path is set to avoid
        # "An instance of Chroma already exists for ephemeral" when multiple clients
        # are created in the same process (e.g. across tests).
        settings = _Settings(anonymized_telemetry=False)
        if path:
            self._client = _chroma.PersistentClient(path=path, settings=settings)
        else:
            self._client = _chroma.Client(settings)

        # Get or create collection
        try:
            self._collection = self._client.get_collection(name=collection_name)
        except Exception:
            self._collection = self._client.create_collection(
                name=collection_name,
                metadata={"description": "Syrin memory store"},
            )

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text. Uses EmbeddingConfig if set, else MD5 fallback."""
        if self._embedding_config is not None and self._embedding_config.custom_fn is not None:
            emb = self._embedding_config.embed(text)
            return list(emb) if not isinstance(emb, list) else emb
        import hashlib

        hash_val = hashlib.md5(text.encode()).digest()
        embedding = [float(b) / 255.0 for b in hash_val]
        embedding.extend([0.0] * (384 - len(embedding)))
        return embedding

    def _entry_to_metadata(self, entry: MemoryEntry) -> dict[str, object]:
        """Convert MemoryEntry to Chroma metadata."""
        metadata: dict[str, object] = {
            "id": entry.id,
            "content": entry.content,
            "type": entry.type.value,
            "importance": entry.importance,
            "scope": entry.scope.value,
            "source": entry.source,
            "created_at": entry.created_at.isoformat() if entry.created_at else None,
            "keywords": ",".join(entry.keywords),
        }
        if self._namespace is not None:
            metadata["namespace"] = self._namespace
        return metadata

    def add(self, memory: MemoryEntry) -> None:
        """Add a memory to Chroma."""
        embedding = self._get_embedding(memory.content)
        metadata = self._entry_to_metadata(memory)

        self._collection.upsert(
            ids=[memory.id],
            embeddings=[embedding],  # type: ignore[arg-type]
            documents=[memory.content],
            metadatas=[metadata],  # type: ignore[list-item]
        )

    def get(self, memory_id: str) -> MemoryEntry | None:
        """Get a memory by ID."""
        try:
            raw = self._collection.get(ids=[memory_id])
            result: dict[str, object] = dict(raw) if isinstance(raw, dict) else {}
            ids_list: list[str] = result.get("ids") or []  # type: ignore[assignment]
            metas: list[dict[str, object]] | None = result.get("metadatas")  # type: ignore[assignment]
            docs: list[str] | None = result.get("documents")  # type: ignore[assignment]
            if not ids_list or metas is None or docs is None or not metas or not docs:
                return None
            meta: dict[str, object] = dict(metas[0]) if metas[0] is not None else {}
            doc_str: str = str(docs[0]) if docs[0] is not None else ""
            return self._metadata_to_entry(meta, doc_str)
        except Exception:
            return None

    def _metadata_to_entry(self, metadata: dict[str, object], content: str) -> MemoryEntry:
        """Convert Chroma metadata to MemoryEntry."""
        raw_id = metadata.get("id", "")
        raw_type = metadata.get("type", "facts")
        raw_importance = metadata.get("importance", 1.0)
        raw_scope = metadata.get("scope", "user")
        raw_source = metadata.get("source")
        raw_created = metadata.get("created_at")
        raw_keywords = metadata.get("keywords", "")
        return MemoryEntry(
            id=str(raw_id),
            content=content,
            type=MemoryType(str(raw_type)),
            importance=float(raw_importance) if isinstance(raw_importance, (int, float)) else 1.0,
            scope=MemoryScope(str(raw_scope)),
            source=str(raw_source) if raw_source is not None else None,
            created_at=datetime.fromisoformat(str(raw_created)) if raw_created else datetime.now(),
            keywords=str(raw_keywords).split(",") if raw_keywords else [],
        )

    def _build_where(
        self,
        memory_type: MemoryType | None = None,
        scope: MemoryScope | None = None,
    ) -> dict[str, object] | None:
        """Build Chroma where filter for memory_type, scope, and namespace."""
        conditions: list[dict[str, object]] = []
        if memory_type is not None:
            conditions.append({"type": memory_type.value})
        if scope is not None:
            conditions.append({"scope": scope.value})
        if self._namespace is not None:
            conditions.append({"namespace": self._namespace})
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def search(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        top_k: int = 10,
        scope: MemoryScope | None = None,
    ) -> list[MemoryEntry]:
        """Search memories by semantic similarity."""
        query_embedding = self._get_embedding(query)

        where = self._build_where(memory_type=memory_type, scope=scope)

        raw_query = self._collection.query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
            n_results=top_k,
            where=where,  # type: ignore[arg-type]
        )
        query_result: dict[str, object] = dict(raw_query) if isinstance(raw_query, dict) else {}
        docs_list: list[list[str]] | None = query_result.get("documents")  # type: ignore[assignment]
        metas_list: list[list[dict[str, object]]] | None = query_result.get("metadatas")  # type: ignore[assignment]
        if not docs_list or not docs_list[0] or metas_list is None or not metas_list:
            return []

        entries: list[MemoryEntry] = []
        for i, doc in enumerate(docs_list[0]):
            meta_row = metas_list[0][i] if i < len(metas_list[0]) else {}
            meta_dict: dict[str, object] = dict(meta_row) if meta_row is not None else {}
            entries.append(self._metadata_to_entry(meta_dict, str(doc)))

        return entries

    def list(
        self,
        memory_type: MemoryType | None = None,
        scope: MemoryScope | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """List all memories."""
        where = self._build_where(memory_type=memory_type, scope=scope)

        try:
            raw = self._collection.get(where=where, limit=limit)  # type: ignore[arg-type]
            list_result: dict[str, object] = dict(raw) if isinstance(raw, dict) else {}
            ids_list: list[str] = list_result.get("ids") or []  # type: ignore[assignment]
            docs: list[str] = list_result.get("documents") or []  # type: ignore[assignment]
            metas: list[dict[str, object]] = list_result.get("metadatas") or []  # type: ignore[assignment]
            if not ids_list:
                return []

            entries: list[MemoryEntry] = []
            for i, doc in enumerate(docs):
                meta_row = metas[i] if i < len(metas) else {}
                meta_dict: dict[str, object] = dict(meta_row) if meta_row is not None else {}
                doc_str = str(doc) if doc is not None else ""
                entries.append(self._metadata_to_entry(meta_dict, doc_str))

            return entries
        except Exception:
            return []

    def update(self, memory: MemoryEntry) -> None:
        """Update a memory."""
        self.add(memory)

    def delete(self, memory_id: str) -> None:
        """Delete a memory by ID."""
        self._collection.delete(ids=[memory_id])

    def clear(self) -> None:
        """Clear all memories."""
        # Chroma where: $exists matches all (our memories have type). Chroma Where type is strict.
        self._collection.delete(where=cast(Any, {"type": {"$exists": True}}))  # type: ignore[explicit-any]

    def close(self) -> None:
        """Close the client connection. Chroma PersistentClient uses SQLite internally."""
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            close_fn()


__all__ = ["ChromaBackend"]
