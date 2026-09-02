import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ocean_agent.document_chunk_store import load_document_chunks


def _chunk_payload(content: str = "RS-232 interface") -> dict:
    return {
        "chunk_id": "sample-doc-001",
        "product_id": "sample-product",
        "title": "Sample manual",
        "section": "Interface",
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
        "page_number": 1,
        "document_id": "sample-doc",
        "chunk_type": "text",
        "previous_chunk_id": None,
        "next_chunk_id": None,
        "content_hash": hashlib.sha256(content.encode()).hexdigest(),
        "cleaning_actions": [],
        "review_status": "auto_cleaned",
    }


class DocumentChunkStoreTests(unittest.TestCase):
    def test_loads_valid_chunk_store(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "chunks.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "summary": {"chunk_count": 1},
                        "chunks": [_chunk_payload()],
                    }
                ),
                encoding="utf-8",
            )
            chunks = load_document_chunks(path)
            self.assertEqual(chunks[0].content, "RS-232 interface")

    def test_rejects_changed_content_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "chunks.json"
            item = _chunk_payload()
            item["content"] = "changed"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "summary": {"chunk_count": 1},
                        "chunks": [item],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "正文哈希不一致"):
                load_document_chunks(path)


if __name__ == "__main__":
    unittest.main()
