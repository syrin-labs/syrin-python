"""Qdrant backend for persistent memory storage with vector embeddings."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from syrin.enums import MemoryScope, MemoryType

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, FilterSelector, PointStruct, VectorParams

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None  # type: ignore[assignment,misc]
    PointStruct = None  # type: ignore[assignment,misc]
    VectorParams = None  # type: ignore[assignment,misc]
    FilterSelector = None  # type: ignore[assignment,misc]

from syrin.memory.config import MemoryEntry


class QdrantBackend:
    """Qdrant-based storage for memories with vector search.

    Requires: pip install qdrant-client

    Features:
    - Vector search (override _get_embedding for semantic similarity)
    - Persistent storage
    - Fast similarity search

    Note: Default _get_embedding uses MD5 hash-based pseudo-embeddings.
    These do NOT provide semantic similarity — similar texts have unrelated vectors.
    For real semantic search, override _get_embedding with sentence-transformers
    or another embedding model.
    """

    def __init__(
        self,
        path: str | None = None,
        host: str = "localhost",
        port: int = 6333,
        url: str | None = None,
        api_key: str | None = None,
        collection: str = "syrin_memory",
        vector_size: int = 384,
        namespace: str | None = None,
        embedding_config: object = None,
    ) -> None:
        """Initialize Qdrant backend.

        Args:
            path: Local path for embedded Qdrant; if set, host/port/url ignored.
            host: Qdrant server host (when url/path not set).
            port: Qdrant server port (when url/path not set).
            url: Full URL for Qdrant Cloud or remote (e.g. https://xyz.qdrant.tech).
            api_key: API key for Qdrant Cloud auth (used with url).
            collection: Collection name for memories.
            vector_size: Embedding dimension (default 384).
            namespace: Per-tenant isolation; all ops scoped to this namespace.
            embedding_config: Optional EmbeddingConfig with custom_fn for embeddings.
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is not installed. Install with: pip install qdrant-client"
            )

        self._collection = collection
        self._vector_size = vector_size
        self._namespace = namespace
        self._embedding_config = embedding_config

        if path:
            self._client = QdrantClient(path=path)
        elif url:
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            self._client = QdrantClient(host=host, port=port)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        collections = self._client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self._collection not in collection_names:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text. Uses EmbeddingConfig if set, else MD5 fallback."""
        if self._embedding_config is not None and self._embedding_config.custom_fn is not None:  # type: ignore[attr-defined]
            emb = self._embedding_config.embed(text)  # type: ignore[attr-defined]
            return list(emb) if not isinstance(emb, list) else emb
        import hashlib

        hash_val = hashlib.md5(text.encode()).digest()
        embedding = [float(b) / 255.0 for b in hash_val]

        if len(embedding) < self._vector_size:
            padding_needed = self._vector_size - len(embedding)
            embedding.extend([0.0] * padding_needed)
        elif len(embedding) > self._vector_size:
            embedding = embedding[: self._vector_size]

        return embedding

    def _entry_to_payload(self, entry: MemoryEntry) -> dict[str, object]:
        """Convert MemoryEntry to Qdrant payload."""
        payload: dict[str, object] = {
            "id": entry.id,
            "content": entry.content,
            "type": entry.type.value,
            "importance": entry.importance,
            "scope": entry.scope.value,
            "source": entry.source,
            "created_at": entry.created_at.isoformat() if entry.created_at else None,
            "keywords": entry.keywords,
            "metadata": entry.metadata,
        }
        if self._namespace is not None:
            payload["namespace"] = self._namespace
        return payload

    def add(self, memory: MemoryEntry) -> None:
        """Add a memory to Qdrant."""
        import uuid

        vector = self._get_embedding(memory.content)

        point_id = memory.id
        try:
            uuid.UUID(point_id)
        except ValueError:
            point_id = str(uuid.uuid4())

        point = PointStruct(
            id=point_id,
            vector=vector,
            payload=self._entry_to_payload(memory),
        )

        self._client.upsert(
            collection_name=self._collection,
            points=[point],
        )

    def get(self, memory_id: str) -> MemoryEntry | None:
        """Get a memory by ID."""
        results = self._client.retrieve(
            collection_name=self._collection,
            ids=[memory_id],
        )

        if not results:
            return None

        payload = results[0].payload
        if payload is None:
            return None
        return self._payload_to_entry(dict(payload) if not isinstance(payload, dict) else payload)

    def _payload_to_entry(self, payload: dict[str, object]) -> MemoryEntry:
        """Convert Qdrant payload to MemoryEntry."""
        raw_id = payload["id"]
        raw_content = payload["content"]
        raw_type = payload["type"]
        raw_importance = payload.get("importance", 1.0)
        raw_scope = payload.get("scope", "user")
        raw_source = payload.get("source")
        raw_created_at = payload.get("created_at")
        raw_keywords = payload.get("keywords", [])
        raw_metadata = payload.get("metadata", {})
        return MemoryEntry(
            id=str(raw_id),
            content=str(raw_content),
            type=MemoryType(str(raw_type)),
            importance=float(raw_importance) if isinstance(raw_importance, (int, float)) else 1.0,
            scope=MemoryScope(str(raw_scope)),
            source=str(raw_source) if raw_source is not None else None,
            created_at=datetime.fromisoformat(str(raw_created_at))
            if raw_created_at
            else datetime.now(),
            keywords=list(raw_keywords) if isinstance(raw_keywords, list) else [],
            metadata=dict(raw_metadata) if isinstance(raw_metadata, dict) else {},
        )

    def _build_filter(
        self,
        memory_type: MemoryType | None = None,
        scope: MemoryScope | None = None,
    ) -> object:
        """Build Qdrant filter for memory_type, scope, and namespace."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        conditions: list[object] = []
        if memory_type is not None:
            conditions.append(FieldCondition(key="type", match=MatchValue(value=memory_type.value)))
        if scope is not None:
            conditions.append(FieldCondition(key="scope", match=MatchValue(value=scope.value)))
        if self._namespace is not None:
            conditions.append(
                FieldCondition(key="namespace", match=MatchValue(value=self._namespace))
            )
        return Filter(must=conditions) if conditions else None  # type: ignore[arg-type]

    def search(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        top_k: int = 10,
        scope: MemoryScope | None = None,
    ) -> list[MemoryEntry]:
        """Search memories by semantic similarity."""
        query_vector = self._get_embedding(query)
        filter_condition = self._build_filter(memory_type=memory_type, scope=scope)

        results = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
            query_filter=filter_condition,  # type: ignore[arg-type]
        )

        entries: list[MemoryEntry] = []
        for r in results.points:
            p = r.payload
            if p is not None:
                d = dict(p) if not isinstance(p, dict) else p
                entries.append(self._payload_to_entry(d))
        return entries

    def list(
        self,
        memory_type: MemoryType | None = None,
        scope: MemoryScope | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """List all memories."""
        filter_obj = self._build_filter(memory_type=memory_type, scope=scope)

        results = self._client.scroll(
            collection_name=self._collection,
            limit=limit,
            with_payload=True,
            scroll_filter=filter_obj,  # type: ignore[arg-type]
        )

        entries2: list[MemoryEntry] = []
        for p in results[0]:
            pl = p.payload
            if pl is not None:
                d = dict(pl) if not isinstance(pl, dict) else pl
                entries2.append(self._payload_to_entry(d))
        return entries2

    def update(self, memory: MemoryEntry) -> None:
        """Update a memory."""
        self.add(memory)

    def delete(self, memory_id: str) -> None:
        """Delete a memory by ID."""
        self._client.delete(
            collection_name=self._collection,
            points_selector=[memory_id],
        )

    def clear(self) -> None:
        """Clear all memories."""
        from qdrant_client.models import Filter, FilterSelector

        self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(filter=Filter()),
        )

    def close(self) -> None:
        """Close the client connection."""
        # Qdrant client doesn't need explicit close


__all__ = ["QdrantBackend"]
