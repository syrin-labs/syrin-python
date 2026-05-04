"""Tests for QdrantConfig and ChromaConfig."""

from __future__ import annotations

import os
import tempfile
import uuid

import pytest

from syrin.enums import MemoryBackend, MemoryType, WriteMode
from syrin.memory import ChromaConfig, Memory, QdrantConfig


class TestQdrantConfig:
    """Tests for QdrantConfig."""

    def test_defaults(self) -> None:
        cfg = QdrantConfig()
        assert cfg.collection == "syrin_memory"
        assert cfg.vector_size == 384
        assert cfg.url is None
        assert cfg.path is None
        assert cfg.host == "localhost"
        assert cfg.port == 6333

    def test_url_and_api_key(self) -> None:
        cfg = QdrantConfig(url="https://xyz.qdrant.tech", api_key="sk-xxx")
        assert cfg.url == "https://xyz.qdrant.tech"
        assert cfg.api_key == "sk-xxx"

    def test_namespace(self) -> None:
        cfg = QdrantConfig(namespace="tenant_123")
        assert cfg.namespace == "tenant_123"

    def test_frozen(self) -> None:
        cfg = QdrantConfig(collection="my_collection")
        with pytest.raises(AttributeError):
            cfg.collection = "other"  # type: ignore[misc]


class TestChromaConfig:
    """Tests for ChromaConfig."""

    def test_defaults(self) -> None:
        cfg = ChromaConfig()
        assert cfg.collection == "syrin_memory"
        assert cfg.path is None

    def test_path_and_collection(self) -> None:
        cfg = ChromaConfig(path="./data/chroma", collection="my_memories")
        assert cfg.path == "./data/chroma"
        assert cfg.collection == "my_memories"


class TestMemoryWithVectorConfigs:
    """Memory with QdrantConfig/ChromaConfig."""

    @pytest.fixture
    def temp_dir(self) -> str:
        d = tempfile.mkdtemp()
        yield d
        import shutil

        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)

    def test_memory_with_qdrant_config_path(self, temp_dir: str) -> None:
        """Memory with QdrantConfig path uses embedded Qdrant."""
        pytest.importorskip(
            "qdrant_client", reason="qdrant_client not installed (pip install syrin[vector])"
        )
        mem = Memory(
            backend=MemoryBackend.QDRANT,
            write_mode=WriteMode.SYNC,
            qdrant=QdrantConfig(path=temp_dir, collection="test_mem"),
        )
        mem.remember("Test memory", memory_type=MemoryType.FACTS)
        results = mem.recall(query="Test", limit=5)
        assert len(results) >= 1
        assert any("Test memory" in e.content for e in results)

    def test_memory_with_chroma_config(self, temp_dir: str) -> None:
        """Memory with ChromaConfig uses Chroma backend."""
        pytest.importorskip("chromadb", reason="chromadb not installed (pip install syrin[vector])")
        chroma_path = os.path.join(temp_dir, f"chroma_{uuid.uuid4().hex[:8]}")
        mem = Memory(
            backend=MemoryBackend.CHROMA,
            write_mode=WriteMode.SYNC,
            chroma=ChromaConfig(path=chroma_path, collection="test_chroma"),
        )
        mem.remember("Chroma test", memory_type=MemoryType.HISTORY)
        results = mem.recall(query="Chroma", limit=5)
        assert len(results) >= 1

    def test_memory_with_qdrant_config_namespace(self, temp_dir: str) -> None:
        """Memory with QdrantConfig namespace stores namespace in payload."""
        pytest.importorskip(
            "qdrant_client", reason="qdrant_client not installed (pip install syrin[vector])"
        )
        qdrant_path = os.path.join(temp_dir, "qdrant_ns")
        os.makedirs(qdrant_path, exist_ok=True)
        mem = Memory(
            backend=MemoryBackend.QDRANT,
            write_mode=WriteMode.SYNC,
            qdrant=QdrantConfig(path=qdrant_path, namespace="tenant_123"),
        )
        mem.remember("Namespaced memory", memory_type=MemoryType.FACTS)
        results = mem.recall(query="Namespaced", limit=10)
        assert len(results) >= 1
        assert any("Namespaced memory" in e.content for e in results)
