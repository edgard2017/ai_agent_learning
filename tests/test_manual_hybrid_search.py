import hashlib
import unittest
from unittest.mock import patch

from ocean_agent.embedding_search import EmbeddingServiceError
from ocean_agent.manual_embedding_index import ManualSemanticSearchMatch
from ocean_agent.manual_hybrid_search import (
    _select_diverse_matches,
    hybrid_search_manual_chunks,
    keyword_search_manual_chunks,
    ManualHybridSearchMatch,
)
from ocean_agent.models import (
    SourceReference,
    SourceType,
    TechnicalDocumentChunk,
    VerificationStatus,
)


SOURCE = SourceReference(
    title="Official manual",
    url="https://manufacturer.example/manual.pdf",
    source_type=SourceType.MANUFACTURER_OFFICIAL,
    accessed_on="2026-09-02",
    verification_status=VerificationStatus.VERIFIED,
)


def _chunk(
    number: int,
    section: str,
    content: str,
    *,
    previous: str | None = None,
    next_: str | None = None,
) -> TechnicalDocumentChunk:
    return TechnicalDocumentChunk(
        chunk_id=f"manual-{number:03d}",
        product_id="sample-product",
        document_id="manual",
        title="Sample manual",
        section=section,
        content=content,
        source=SOURCE,
        page_number=number,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        previous_chunk_id=previous,
        next_chunk_id=next_,
        review_status="auto_cleaned",
    )


class ManualHybridSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = (
            _chunk(1, "Interface", "USB-C configuration and data download.", next_="manual-002"),
            _chunk(
                2,
                "Interface",
                "External MCBH connector pinout for RS-232 and RS-485.",
                previous="manual-001",
                next_="manual-003",
            ),
            _chunk(3, "Interface", "Sampling requires internal or external power.", previous="manual-002"),
            _chunk(
                4,
                "Maintenance",
                "Replace the damaged O-ring and lubricate the new O-ring before deployment.",
            ),
        )

    def test_chinese_aliases_find_english_pinout(self) -> None:
        matches = keyword_search_manual_chunks("外部接口针脚怎么接？", self.chunks)
        self.assertEqual(matches[0].chunk.chunk_id, "manual-002")

    @patch("ocean_agent.manual_hybrid_search.semantic_search_manual_chunks")
    @patch("ocean_agent.manual_hybrid_search.load_document_chunks")
    def test_hybrid_result_keeps_anchor_and_neighbors(
        self, mock_load, mock_semantic
    ) -> None:
        mock_load.return_value = self.chunks
        mock_semantic.return_value = [
            ManualSemanticSearchMatch(1, self.chunks[1], 0.8),
            ManualSemanticSearchMatch(2, self.chunks[0], 0.7),
            ManualSemanticSearchMatch(3, self.chunks[2], 0.6),
        ]

        groups = hybrid_search_manual_chunks(
            "MCBH接口针脚",
            product_id="sample-product",
            limit=1,
            candidate_limit=3,
        )

        self.assertEqual(groups[0].anchor.chunk.chunk_id, "manual-002")
        self.assertEqual(groups[0].previous_chunk.chunk_id, "manual-001")
        self.assertEqual(groups[0].next_chunk.chunk_id, "manual-003")
        self.assertEqual(
            groups[0].anchor.retrieval_methods, ("keyword", "embedding")
        )

    @patch("ocean_agent.manual_hybrid_search.semantic_search_manual_chunks")
    @patch("ocean_agent.manual_hybrid_search.load_document_chunks")
    def test_embedding_failure_falls_back_to_keyword(
        self, mock_load, mock_semantic
    ) -> None:
        mock_load.return_value = self.chunks
        mock_semantic.side_effect = EmbeddingServiceError("offline")

        groups = hybrid_search_manual_chunks(
            "更换O型圈",
            limit=1,
            candidate_limit=3,
        )

        self.assertEqual(groups[0].anchor.chunk.chunk_id, "manual-004")
        self.assertEqual(groups[0].anchor.retrieval_methods, ("keyword",))

    def test_diversity_skips_near_duplicate_and_limits_section(self) -> None:
        chunks = (
            _chunk(1, "A", "alpha beta gamma delta"),
            _chunk(2, "A", "alpha beta gamma delta epsilon"),
            _chunk(3, "A", "different evidence in same section"),
            _chunk(4, "B", "another topic"),
        )
        ranked = [
            ManualHybridSearchMatch(
                chunk=chunk,
                fused_score=1.0 - index * 0.1,
                keyword_score=None,
                keyword_rank=None,
                embedding_similarity=0.9 - index * 0.1,
                embedding_rank=index + 1,
                retrieval_methods=("embedding",),
            )
            for index, chunk in enumerate(chunks)
        ]
        selected = _select_diverse_matches(
            ranked,
            limit=3,
            max_per_section=1,
            duplicate_threshold=0.7,
        )

        self.assertEqual(
            [match.chunk.chunk_id for match in selected],
            ["manual-001", "manual-004"],
        )


if __name__ == "__main__":
    unittest.main()
