"""融合关键词与本地 Embedding 的技术资料检索。"""

from dataclasses import dataclass

from .config import Settings
from .document_data import DOCUMENT_CHUNKS
from .embedding_search import (
    EmbeddingServiceError,
    SemanticSearchMatch,
    semantic_search,
)
from .models import TechnicalDocumentChunk
from .tools import get_product_spec, search_documents


RRF_K = 60


@dataclass(frozen=True)
class HybridDocumentSearchMatch:
    """混合检索结果，保留两路排名以便解释和调试。"""

    chunk: TechnicalDocumentChunk
    fused_score: float
    keyword_score: int | None
    keyword_rank: int | None
    embedding_similarity: float | None
    embedding_rank: int | None
    retrieval_methods: tuple[str, ...]


def _candidate_chunks(
    model_or_id: str | None,
) -> tuple[TechnicalDocumentChunk, ...]:
    if not model_or_id or not model_or_id.strip():
        return DOCUMENT_CHUNKS

    product = get_product_spec(model_or_id)
    if product is None:
        return ()
    return tuple(
        chunk for chunk in DOCUMENT_CHUNKS if chunk.product_id == product.product_id
    )


def _embedding_text(chunk: TechnicalDocumentChunk) -> str:
    """把标题、章节和正文组合成一段可独立检索的文本。"""

    return f"{chunk.title}\n{chunk.section}\n{chunk.content}"


def hybrid_search_documents(
    query: str,
    *,
    model_or_id: str | None = None,
    limit: int = 3,
    settings: Settings | None = None,
) -> list[HybridDocumentSearchMatch]:
    """使用RRF融合关键词排名和Qwen Embedding排名。

    明确型号时只在该产品资料内检索。Embedding不可用时自动退回关键词结果。
    这里不使用相似度阈值判断有无答案，因为相关主题的片段不一定真正包含答案。
    """

    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("query 不能为空")
    if not 1 <= limit <= 10:
        raise ValueError("limit 必须在 1 到 10 之间")

    candidates = _candidate_chunks(model_or_id)
    if not candidates:
        return []

    keyword_matches = search_documents(
        cleaned_query,
        model_or_id=model_or_id,
        limit=min(10, len(candidates)),
    )
    keyword_by_id = {
        match.chunk.chunk_id: (rank, match.score)
        for rank, match in enumerate(keyword_matches, start=1)
    }

    try:
        embedding_matches = semantic_search(
            cleaned_query,
            [_embedding_text(chunk) for chunk in candidates],
            limit=len(candidates),
            settings=settings,
            cache_documents=True,
        )
    except EmbeddingServiceError:
        return [
            HybridDocumentSearchMatch(
                chunk=match.chunk,
                fused_score=1.0 / (RRF_K + rank),
                keyword_score=match.score,
                keyword_rank=rank,
                embedding_similarity=None,
                embedding_rank=None,
                retrieval_methods=("keyword",),
            )
            for rank, match in enumerate(keyword_matches[:limit], start=1)
        ]

    embedding_by_id: dict[str, tuple[int, SemanticSearchMatch]] = {}
    for rank, match in enumerate(embedding_matches, start=1):
        chunk = candidates[match.document_index]
        embedding_by_id[chunk.chunk_id] = (rank, match)

    results: list[HybridDocumentSearchMatch] = []
    for chunk in candidates:
        keyword_data = keyword_by_id.get(chunk.chunk_id)
        embedding_data = embedding_by_id.get(chunk.chunk_id)
        keyword_rank = keyword_data[0] if keyword_data else None
        keyword_score = keyword_data[1] if keyword_data else None
        embedding_rank = embedding_data[0] if embedding_data else None
        embedding_similarity = (
            embedding_data[1].similarity if embedding_data else None
        )

        fused_score = 0.0
        methods: list[str] = []
        if keyword_rank is not None:
            fused_score += 1.0 / (RRF_K + keyword_rank)
            methods.append("keyword")
        if embedding_rank is not None:
            fused_score += 1.0 / (RRF_K + embedding_rank)
            methods.append("embedding")

        if methods:
            results.append(
                HybridDocumentSearchMatch(
                    chunk=chunk,
                    fused_score=fused_score,
                    keyword_score=keyword_score,
                    keyword_rank=keyword_rank,
                    embedding_similarity=embedding_similarity,
                    embedding_rank=embedding_rank,
                    retrieval_methods=tuple(methods),
                )
            )

    results.sort(
        key=lambda item: (
            -item.fused_score,
            item.keyword_rank if item.keyword_rank is not None else 10_000,
            item.embedding_rank if item.embedding_rank is not None else 10_000,
            item.chunk.chunk_id,
        )
    )
    return results[:limit]
