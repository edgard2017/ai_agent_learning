import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ocean_agent.config import Settings
from ocean_agent.manual_embedding_index import (
    build_manual_embedding_index,
    semantic_search_manual_chunks,
)


def _write_chunk_store(path: Path, count: int = 5) -> None:
    chunks = []
    for index in range(count):
        content = "RS-232 interface" if index == 0 else f"Other document {index}"
        chunk_id = f"sample-doc-{index + 1:03d}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "product_id": "sample-product",
                "title": "Sample manual",
                "section": "Interface" if index == 0 else "Other",
                "content": content,
                "keywords": [],
                "source": {
                    "title": "Official manual",
                    "url": "https://manufacturer.example/manual.pdf",
                    "source_type": "manufacturer_official",
                    "accessed_on": "2026-09-02",
                    "document_version": "A",
                    "verification_status": "verified",
                },
                "page_number": index + 1,
                "document_id": "sample-doc",
                "chunk_type": "text",
                "previous_chunk_id": (
                    f"sample-doc-{index:03d}" if index > 0 else None
                ),
                "next_chunk_id": (
                    f"sample-doc-{index + 2:03d}" if index + 1 < count else None
                ),
                "content_hash": hashlib.sha256(content.encode()).hexdigest(),
                "cleaning_actions": [],
                "review_status": "auto_cleaned",
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "summary": {"chunk_count": count},
                "chunks": chunks,
            }
        ),
        encoding="utf-8",
    )


class ManualEmbeddingIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(
            model_provider="ollama",
            ollama_embedding_model="test-embedding-model",
        )

    def test_builds_index_in_batches_and_reuses_cache(self) -> None:
        calls: list[int] = []

        def fake_embed(texts):
            calls.append(len(texts))
            return [[float(len(text)), 1.0] for text in texts]

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            chunk_store = root / "chunks.json"
            cache = root / "embeddings.json"
            index = root / "index.json"
            _write_chunk_store(chunk_store)

            first = build_manual_embedding_index(
                chunk_store_path=chunk_store,
                cache_path=cache,
                index_path=index,
                settings=self.settings,
                batch_size=2,
                embed_batch=fake_embed,
            )
            second = build_manual_embedding_index(
                chunk_store_path=chunk_store,
                cache_path=cache,
                index_path=index,
                settings=self.settings,
                batch_size=2,
                embed_batch=fake_embed,
            )

            self.assertEqual(calls, [2, 2, 1])
            self.assertEqual(first.cache_miss_count, 5)
            self.assertEqual(second.cache_hit_count, 5)
            self.assertEqual(first.vector_dimension, 2)
            self.assertTrue(index.is_file())

    def test_semantic_search_keeps_chunk_metadata(self) -> None:
        def fake_documents(texts):
            return [
                [1.0, 0.0] if "RS-232" in text else [0.0, 1.0]
                for text in texts
            ]

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            cache = root / "embeddings.json"
            chunk_store = root / "chunks.json"
            _write_chunk_store(chunk_store)
            matches = semantic_search_manual_chunks(
                "怎么连接电脑",
                product_id="sample-product",
                limit=20,
                chunk_store_path=chunk_store,
                cache_path=cache,
                settings=self.settings,
                query_embedder=lambda _: [1.0, 0.0],
                document_embedder=fake_documents,
            )

            self.assertEqual(matches[0].rank, 1)
            self.assertEqual(len(matches), 5)
            self.assertIn("RS-232", matches[0].chunk.content)
            self.assertIsNotNone(matches[0].chunk.page_number)


if __name__ == "__main__":
    unittest.main()
