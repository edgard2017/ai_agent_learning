import unittest
from unittest.mock import patch

from ocean_agent.document_data import DOCUMENT_CHUNKS
from ocean_agent.embedding_search import EmbeddingServiceError, SemanticSearchMatch
from ocean_agent.hybrid_search import hybrid_search_documents


def _semantic_match(document_index: int, similarity: float) -> SemanticSearchMatch:
    return SemanticSearchMatch(
        document_index=document_index,
        document=DOCUMENT_CHUNKS[document_index].content,
        similarity=similarity,
    )


class HybridDocumentSearchTests(unittest.TestCase):
    @patch("ocean_agent.hybrid_search.semantic_search")
    def test_embedding_recovers_oral_query_missed_by_keywords(
        self, mock_semantic_search
    ) -> None:
        interface_index = next(
            index
            for index, chunk in enumerate(DOCUMENT_CHUNKS)
            if chunk.chunk_id == "sbe19-interface-sampling"
        )
        other_indexes = [
            index for index in range(len(DOCUMENT_CHUNKS)) if index != interface_index
        ]
        mock_semantic_search.return_value = [
            _semantic_match(interface_index, 0.55),
            *[
                _semantic_match(index, 0.40 - position * 0.01)
                for position, index in enumerate(other_indexes)
            ],
        ]

        matches = hybrid_search_documents("19plus怎么把数据传到电脑里？")

        self.assertEqual(matches[0].chunk.chunk_id, "sbe19-interface-sampling")
        self.assertEqual(matches[0].retrieval_methods, ("embedding",))

    @patch("ocean_agent.hybrid_search.semantic_search")
    def test_keyword_rank_corrects_misleading_embedding_rank(
        self, mock_semantic_search
    ) -> None:
        correct_index = next(
            index
            for index, chunk in enumerate(DOCUMENT_CHUNKS)
            if chunk.chunk_id == "sbe37-known-limits"
        )
        wrong_index = next(
            index
            for index, chunk in enumerate(DOCUMENT_CHUNKS)
            if chunk.chunk_id == "sbe16-deployment-sampling"
        )
        remaining = [
            index
            for index in range(len(DOCUMENT_CHUNKS))
            if index not in {correct_index, wrong_index}
        ]
        mock_semantic_search.return_value = [
            _semantic_match(wrong_index, 0.66),
            _semantic_match(correct_index, 0.65),
            *[
                _semantic_match(index, 0.40 - position * 0.01)
                for position, index in enumerate(remaining)
            ],
        ]

        matches = hybrid_search_documents("MicroCAT的压力是不是选配？")

        self.assertEqual(matches[0].chunk.chunk_id, "sbe37-known-limits")
        self.assertEqual(matches[0].keyword_rank, 1)
        self.assertEqual(matches[0].embedding_rank, 2)
        self.assertEqual(matches[0].retrieval_methods, ("keyword", "embedding"))

    @patch("ocean_agent.hybrid_search.semantic_search")
    def test_explicit_model_limits_embedding_candidates(
        self, mock_semantic_search
    ) -> None:
        mock_semantic_search.return_value = [
            SemanticSearchMatch(0, "first", 0.6),
            SemanticSearchMatch(1, "second", 0.5),
        ]

        matches = hybrid_search_documents(
            "怎么把数据传到电脑里？",
            model_or_id="SBE 19plus V2",
        )

        self.assertTrue(matches)
        self.assertTrue(
            all(match.chunk.product_id == "seabird-sbe-19plus-v2" for match in matches)
        )
        documents = mock_semantic_search.call_args.args[1]
        self.assertEqual(len(documents), 2)
        self.assertTrue(mock_semantic_search.call_args.kwargs["cache_documents"])

    @patch("ocean_agent.hybrid_search.semantic_search")
    def test_embedding_failure_falls_back_to_keyword_search(
        self, mock_semantic_search
    ) -> None:
        mock_semantic_search.side_effect = EmbeddingServiceError("offline")

        matches = hybrid_search_documents("RBRconcerto使用什么电源？")

        self.assertTrue(matches)
        self.assertEqual(
            matches[0].chunk.chunk_id,
            "rbr-ctd-interface-power-sampling",
        )
        self.assertEqual(matches[0].retrieval_methods, ("keyword",))
        self.assertIsNone(matches[0].embedding_similarity)

    def test_unknown_model_returns_no_match(self) -> None:
        self.assertEqual(
            hybrid_search_documents("如何连接？", model_or_id="不存在的CTD"),
            [],
        )


if __name__ == "__main__":
    unittest.main()
