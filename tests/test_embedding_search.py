import json
import unittest
from unittest.mock import MagicMock, patch

from ocean_agent.config import Settings
from ocean_agent.embedding_search import (
    cosine_similarity,
    create_embeddings,
    semantic_search,
)


class EmbeddingSearchTests(unittest.TestCase):
    def test_cosine_similarity_of_same_direction_is_one(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1, 2], [2, 4]), 1.0)

    def test_cosine_similarity_of_perpendicular_vectors_is_zero(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_cosine_similarity_rejects_different_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "维度必须相同"):
            cosine_similarity([1, 2], [1])

    @patch("ocean_agent.embedding_search.urlopen")
    def test_create_embeddings_formats_qwen_query_with_instruction(
        self, mock_urlopen: MagicMock
    ) -> None:
        response = MagicMock()
        response.read.return_value = json.dumps(
            {"embeddings": [[0.1, 0.2, 0.3]]}
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = response
        settings = Settings(_env_file=None)

        vectors = create_embeddings(
            ["设备如何连接电脑？"],
            input_type="query",
            settings=settings,
        )

        self.assertEqual(vectors, [[0.1, 0.2, 0.3]])
        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "qwen3-embedding:0.6b")
        self.assertEqual(
            payload["input"],
            [
                "Instruct: Given a marine equipment technical query, retrieve "
                "passages that answer the query.\nQuery:设备如何连接电脑？"
            ],
        )

    @patch("ocean_agent.embedding_search.urlopen")
    def test_create_embeddings_keeps_qwen_document_unchanged(
        self, mock_urlopen: MagicMock
    ) -> None:
        response = MagicMock()
        response.read.return_value = json.dumps(
            {"embeddings": [[0.1, 0.2, 0.3]]}
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = response

        create_embeddings(
            ["The instrument communicates through RS-232."],
            input_type="document",
            settings=Settings(_env_file=None),
        )

        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(
            payload["input"],
            ["The instrument communicates through RS-232."],
        )

    @patch("ocean_agent.embedding_search.urlopen")
    def test_nomic_model_still_uses_search_prefix(
        self, mock_urlopen: MagicMock
    ) -> None:
        response = MagicMock()
        response.read.return_value = json.dumps(
            {"embeddings": [[0.1, 0.2, 0.3]]}
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = response

        create_embeddings(
            ["设备如何连接电脑？"],
            input_type="query",
            settings=Settings(
                ollama_embedding_model="nomic-embed-text:latest",
                _env_file=None,
            ),
        )

        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["input"], ["search_query: 设备如何连接电脑？"])

    @patch("ocean_agent.embedding_search.create_embedding")
    @patch("ocean_agent.embedding_search.create_embeddings")
    def test_semantic_search_sorts_by_similarity(
        self,
        mock_create_embeddings: MagicMock,
        mock_create_embedding: MagicMock,
    ) -> None:
        mock_create_embeddings.return_value = [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
        ]
        mock_create_embedding.return_value = [1.0, 0.0]

        matches = semantic_search(
            "如何连接计算机？",
            ["RS-232连接电脑", "6000米水深", "串口通信"],
            limit=3,
            settings=Settings(_env_file=None),
        )

        self.assertEqual(
            [match.document for match in matches],
            ["RS-232连接电脑", "串口通信", "6000米水深"],
        )
        self.assertAlmostEqual(matches[0].similarity, 1.0)


if __name__ == "__main__":
    unittest.main()
