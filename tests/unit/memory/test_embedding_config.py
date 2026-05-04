"""Tests for EmbeddingConfig - pluggable embeddings for vector backends."""

from __future__ import annotations

import os
import tempfile
import uuid
from unittest.mock import AsyncMock

import pytest

from syrin.enums import MemoryBackend, MemoryType, WriteMode
from syrin.memory import ChromaConfig, EmbeddingConfig, Memory, QdrantConfig


def _fake_embedding(text: str) -> list[float]:
    """Deterministic fake embedding for tests. Returns 384 dimensions."""
    vals = [float(ord(c) % 10) / 10.0 for c in (text + "\0" * 16)[:16]]
    return vals + [0.0] * (384 - len(vals))


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""

    def test_defaults(self) -> None:
        cfg = EmbeddingConfig()
        assert cfg.model == "text-embedding-3-small"
        assert cfg.provider == "openai"
        assert cfg.dimensions == 1536

    def test_custom_fn(self) -> None:
        cfg = EmbeddingConfig(custom_fn=_fake_embedding, dimensions=384)
        result = cfg.embed("hello")
        assert len(result) == 384
        assert result[0] == 0.4  # 'h' -> ord 104 % 10 / 10 = 0.4

    def test_custom_fn_dimensions_mismatch_raises(self) -> None:
        def wrong_size(text: str) -> list[float]:
            return [1.0, 2.0]  # Too short

        cfg = EmbeddingConfig(custom_fn=wrong_size, dimensions=384)
        with pytest.raises(ValueError, match="returned 2 dimensions"):
            cfg.embed("test")

    def test_embed_sync_with_provider_raises(self) -> None:
        """embed() raises when embedding_provider is set (must use embed_async)."""
        mock_provider = AsyncMock()
        cfg = EmbeddingConfig(embedding_provider=mock_provider, dimensions=384)

        with pytest.raises(ValueError, match="is sync"):
            cfg.embed("hello")

    @pytest.mark.asyncio
    async def test_embed_async_with_provider(self) -> None:
        """embed_async() uses embedding_provider."""
        mock_provider = AsyncMock()
        mock_provider.embed = AsyncMock(return_value=[[0.1] * 384])
        cfg = EmbeddingConfig(embedding_provider=mock_provider, dimensions=384)

        result = await cfg.embed_async("hello")

        assert result == [0.1] * 384
        mock_provider.embed.assert_called_once_with(["hello"])

    @pytest.mark.asyncio
    async def test_embed_async_without_provider_raises(self) -> None:
        """embed_async() raises when embedding_provider is not set."""
        cfg = EmbeddingConfig(dimensions=384)

        with pytest.raises(ValueError, match="requires embedding_provider"):
            await cfg.embed_async("hello")

    @pytest.mark.asyncio
    async def test_embed_async_empty_result_raises(self) -> None:
        """embed_async() raises when provider returns empty result."""
        mock_provider = AsyncMock()
        mock_provider.embed = AsyncMock(return_value=[])
        cfg = EmbeddingConfig(embedding_provider=mock_provider, dimensions=384)

        with pytest.raises(ValueError, match="empty result"):
            await cfg.embed_async("hello")


class TestMemoryWithEmbeddingConfig:
    """Memory with EmbeddingConfig in Qdrant/Chroma."""

    @pytest.fixture
    def temp_dir(self) -> str:
        d = tempfile.mkdtemp()
        yield d
        import shutil

        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)

    def test_qdrant_with_custom_embedding(self, temp_dir: str) -> None:
        """Qdrant with EmbeddingConfig.custom_fn uses custom embeddings."""
        pytest.importorskip(
            "qdrant_client", reason="qdrant_client not installed (pip install syrin[vector])"
        )
        mem = Memory(
            backend=MemoryBackend.QDRANT,
            write_mode=WriteMode.SYNC,
            qdrant=QdrantConfig(
                path=temp_dir,
                collection="test_emb",
                embedding_config=EmbeddingConfig(
                    custom_fn=_fake_embedding,
                    dimensions=384,
                ),
            ),
        )
        mem.remember("Custom embedding test", memory_type=MemoryType.FACTS)
        results = mem.recall(query="Custom", limit=5)
        assert len(results) >= 1

    def test_chroma_with_custom_embedding(self, temp_dir: str) -> None:
        """Chroma with EmbeddingConfig.custom_fn uses custom embeddings."""
        pytest.importorskip("chromadb", reason="chromadb not installed (pip install syrin[vector])")
        chroma_path = os.path.join(temp_dir, f"chroma_{uuid.uuid4().hex[:8]}")
        mem = Memory(
            backend=MemoryBackend.CHROMA,
            write_mode=WriteMode.SYNC,
            chroma=ChromaConfig(
                path=chroma_path,
                collection="test_emb",
                embedding_config=EmbeddingConfig(
                    custom_fn=_fake_embedding,
                    dimensions=384,
                ),
            ),
        )
        mem.remember("Chroma custom embedding", memory_type=MemoryType.HISTORY)
        results = mem.recall(query="Chroma", limit=5)
        assert len(results) >= 1
