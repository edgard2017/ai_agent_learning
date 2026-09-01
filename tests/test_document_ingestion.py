import json
import tempfile
import unittest
from pathlib import Path

from ocean_agent.document_chunker import chunk_document, chunk_documents
from ocean_agent.document_loader import load_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DocumentLoaderTests(unittest.TestCase):
    def test_loads_example_markdown_and_metadata(self) -> None:
        documents = load_documents(PROJECT_ROOT / "documents")

        self.assertEqual(len(documents), 1)
        document = documents[0]
        self.assertEqual(document.product_id, "seabird-sbe-19plus-v2")
        self.assertIn("通信与采样", document.content)
        self.assertEqual(document.source.verification_status.value, "verified")

    def test_rejects_unknown_product_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "sample.md").write_text("# 测试\n\n正文", encoding="utf-8")
            manifest = {
                "documents": [
                    {
                        "document_id": "unknown-product-doc",
                        "file": "sample.md",
                        "product_id": "unknown-product",
                    }
                ]
            }
            (root / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "未知 product_id"):
                load_documents(root)


class DocumentChunkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = load_documents(PROJECT_ROOT / "documents")[0]

    def test_chunks_keep_source_and_product_metadata(self) -> None:
        chunks = chunk_document(self.document, max_chars=160)

        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(len(chunk.content) <= 160 for chunk in chunks))
        self.assertTrue(
            all(chunk.product_id == self.document.product_id for chunk in chunks)
        )
        self.assertTrue(all(chunk.source == self.document.source for chunk in chunks))
        self.assertEqual(chunks[0].chunk_id, "sbe19plus-v2-project-summary-001")
        self.assertIn("通信与采样", {chunk.section for chunk in chunks})

    def test_batch_chunking_returns_search_ready_model(self) -> None:
        chunks = chunk_documents((self.document,), max_chars=200)

        communication = [chunk for chunk in chunks if chunk.section == "通信与采样"]
        self.assertTrue(communication)
        self.assertTrue(any("RS-232" in chunk.content for chunk in communication))

    def test_rejects_unreasonably_small_chunk_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "不能小于 50"):
            chunk_document(self.document, max_chars=20)


if __name__ == "__main__":
    unittest.main()
