from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from ocean_agent.embedding_cache import get_or_create_document_embeddings


class DocumentEmbeddingCacheTests(unittest.TestCase):
    def test_second_call_reuses_all_document_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.json"
            create_missing = MagicMock(return_value=[[1.0, 0.0], [0.0, 1.0]])

            first = get_or_create_document_embeddings(
                ["document a", "document b"],
                model_name="qwen3-embedding:0.6b",
                create_missing=create_missing,
                cache_path=cache_path,
            )
            second = get_or_create_document_embeddings(
                ["document a", "document b"],
                model_name="qwen3-embedding:0.6b",
                create_missing=create_missing,
                cache_path=cache_path,
            )

        self.assertEqual((first.hit_count, first.miss_count), (0, 2))
        self.assertEqual((second.hit_count, second.miss_count), (2, 0))
        self.assertEqual(first.vectors, second.vectors)
        create_missing.assert_called_once_with(["document a", "document b"])

    def test_changed_document_only_rebuilds_changed_vector(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.json"
            create_missing = MagicMock(
                side_effect=[[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5]]]
            )
            get_or_create_document_embeddings(
                ["document a", "document b"],
                model_name="qwen3-embedding:0.6b",
                create_missing=create_missing,
                cache_path=cache_path,
            )

            result = get_or_create_document_embeddings(
                ["document a", "document b changed"],
                model_name="qwen3-embedding:0.6b",
                create_missing=create_missing,
                cache_path=cache_path,
            )

        self.assertEqual((result.hit_count, result.miss_count), (1, 1))
        self.assertEqual(result.vectors, [[1.0, 0.0], [0.5, 0.5]])
        self.assertEqual(
            create_missing.call_args_list[1].args[0],
            ["document b changed"],
        )

    def test_different_model_uses_separate_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.json"
            first_factory = MagicMock(return_value=[[1.0, 0.0]])
            second_factory = MagicMock(return_value=[[0.0, 1.0]])
            get_or_create_document_embeddings(
                ["same document"],
                model_name="model-a",
                create_missing=first_factory,
                cache_path=cache_path,
            )

            result = get_or_create_document_embeddings(
                ["same document"],
                model_name="model-b",
                create_missing=second_factory,
                cache_path=cache_path,
            )

        self.assertEqual((result.hit_count, result.miss_count), (0, 1))
        self.assertEqual(result.vectors, [[0.0, 1.0]])
        second_factory.assert_called_once()

    def test_changed_vector_dimension_rebuilds_old_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.json"
            get_or_create_document_embeddings(
                ["document"],
                model_name="mutable-model-tag",
                create_missing=lambda texts: [[1.0, 0.0]],
                cache_path=cache_path,
                expected_dimension=2,
            )
            create_missing = MagicMock(return_value=[[1.0, 0.0, 0.0]])

            result = get_or_create_document_embeddings(
                ["document"],
                model_name="mutable-model-tag",
                create_missing=create_missing,
                cache_path=cache_path,
                expected_dimension=3,
            )

        self.assertEqual((result.hit_count, result.miss_count), (0, 1))
        self.assertEqual(len(result.vectors[0]), 3)
        create_missing.assert_called_once()

    def test_corrupted_cache_is_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.json"
            cache_path.write_text("not-json", encoding="utf-8")
            create_missing = MagicMock(return_value=[[1.0, 0.0]])

            result = get_or_create_document_embeddings(
                ["document"],
                model_name="qwen3-embedding:0.6b",
                create_missing=create_missing,
                cache_path=cache_path,
            )

        self.assertEqual((result.hit_count, result.miss_count), (0, 1))
        self.assertEqual(result.vectors, [[1.0, 0.0]])

    def test_cache_file_contains_hashes_not_document_text(self) -> None:
        sensitive_text = "public document text that should still not be copied"
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.json"
            get_or_create_document_embeddings(
                [sensitive_text],
                model_name="qwen3-embedding:0.6b",
                create_missing=lambda texts: [[1.0, 0.0]],
                cache_path=cache_path,
            )
            cache_content = cache_path.read_text(encoding="utf-8")

        self.assertNotIn(sensitive_text, cache_content)


if __name__ == "__main__":
    unittest.main()
