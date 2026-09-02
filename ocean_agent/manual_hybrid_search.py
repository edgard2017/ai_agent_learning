"""真实厂家手册的关键词/Embedding融合、结果多样性和邻居扩展。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
import unicodedata
from pathlib import Path
from typing import Sequence

from .config import Settings
from .document_chunk_store import (
    DEFAULT_DOCUMENT_CHUNK_STORE,
    load_document_chunks,
)
from .embedding_cache import DEFAULT_DOCUMENT_EMBEDDING_CACHE
from .embedding_search import EmbeddingServiceError
from .manual_embedding_index import (
    ManualSemanticSearchMatch,
    semantic_search_manual_chunks,
)
from .models import TechnicalDocumentChunk


RRF_K = 60
QUERY_ALIASES: dict[str, tuple[str, ...]] = {
    "连接": ("connect", "connection", "interface"),
    "电脑": ("pc", "computer"),
    "通信": ("communication", "interface"),
    "接口": ("interface", "connector"),
    "针脚": ("pin", "pinout", "connector"),
    "接线": ("wiring", "pinout", "connector"),
    "上传": ("upload", "transmit", "data"),
    "下载": ("download", "upload", "data"),
    "更换": ("replace", "replacing"),
    "型圈": ("o-ring", "o-rings"),
    "电池": ("battery", "batteries"),
    "供电": ("power", "supply"),
    "采样": ("sample", "sampling"),
    "校准": ("calibrate", "calibration"),
    "维护": ("maintenance",),
}


@dataclass(frozen=True)
class ManualKeywordSearchMatch:
    chunk: TechnicalDocumentChunk
    score: int


@dataclass(frozen=True)
class ManualHybridSearchMatch:
    chunk: TechnicalDocumentChunk
    fused_score: float
    keyword_score: int | None
    keyword_rank: int | None
    embedding_similarity: float | None
    embedding_rank: int | None
    retrieval_methods: tuple[str, ...]


@dataclass(frozen=True)
class ManualEvidenceGroup:
    """一个主命中及其前后上下文；邻居不冒充直接命中。"""

    anchor: ManualHybridSearchMatch
    previous_chunk: TechnicalDocumentChunk | None
    next_chunk: TechnicalDocumentChunk | None


def _normalize_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _query_terms(query: str) -> set[str]:
    normalized = _normalize_text(query)
    terms = {
        term
        for term in re.findall(r"[a-z0-9]+(?:[.+-][a-z0-9]+)*", normalized)
        if len(term) >= 2 or term.isdigit()
    }
    chinese_groups = re.findall(r"[\u4e00-\u9fff]+", normalized)
    for group in chinese_groups:
        if len(group) == 1:
            terms.add(group)
        else:
            terms.update(group[index : index + 2] for index in range(len(group) - 1))
    for trigger, aliases in QUERY_ALIASES.items():
        if trigger in normalized:
            terms.update(aliases)
    return terms


def keyword_search_manual_chunks(
    query: str,
    chunks: Sequence[TechnicalDocumentChunk],
    *,
    limit: int = 20,
) -> list[ManualKeywordSearchMatch]:
    if not query.strip():
        raise ValueError("query不能为空")
    if limit < 1:
        raise ValueError("limit不能小于1")

    terms = _query_terms(query)
    matches: list[ManualKeywordSearchMatch] = []
    for chunk in chunks:
        title = _normalize_text(chunk.title)
        section = _normalize_text(chunk.section)
        content = _normalize_text(chunk.content)
        keywords = _normalize_text(" ".join(chunk.keywords))
        score = 0
        for term in terms:
            if term in content:
                score += 1
            if term in section:
                score += 3
            if term in title:
                score += 2
            if term in keywords:
                score += 2
        if score >= 2:
            matches.append(ManualKeywordSearchMatch(chunk=chunk, score=score))
    matches.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
    return matches[:limit]


def _content_tokens(text: str) -> set[str]:
    normalized = _normalize_text(text)
    words = set(re.findall(r"[a-z0-9]+", normalized))
    chinese = re.findall(r"[\u4e00-\u9fff]+", normalized)
    for group in chinese:
        words.update(group[index : index + 2] for index in range(len(group) - 1))
    return words


def _has_enough_information(chunk: TechnicalDocumentChunk) -> bool:
    """封面型号、单独网址等内容不作为主命中。"""

    return len(chunk.content.strip()) >= 40 and len(_content_tokens(chunk.content)) >= 5


def _jaccard_similarity(left: str, right: str) -> float:
    left_tokens = _content_tokens(left)
    right_tokens = _content_tokens(right)
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def _select_diverse_matches(
    ranked: Sequence[ManualHybridSearchMatch],
    *,
    limit: int,
    max_per_section: int,
    duplicate_threshold: float,
) -> list[ManualHybridSearchMatch]:
    selected: list[ManualHybridSearchMatch] = []
    section_counts: Counter[tuple[str | None, str]] = Counter()
    for match in ranked:
        section_key = (match.chunk.document_id, match.chunk.section)
        if section_counts[section_key] >= max_per_section:
            continue
        if any(
            _jaccard_similarity(match.chunk.content, item.chunk.content)
            >= duplicate_threshold
            for item in selected
        ):
            continue
        selected.append(match)
        section_counts[section_key] += 1
        if len(selected) >= limit:
            break
    return selected


def hybrid_search_manual_chunks(
    query: str,
    *,
    product_id: str | None = None,
    limit: int = 5,
    candidate_limit: int = 20,
    max_per_section: int = 1,
    duplicate_threshold: float = 0.82,
    chunk_store_path: Path = DEFAULT_DOCUMENT_CHUNK_STORE,
    cache_path: Path = DEFAULT_DOCUMENT_EMBEDDING_CACHE,
    settings: Settings | None = None,
) -> list[ManualEvidenceGroup]:
    """融合两路Top-N，选择不同证据主题，并附带前后Chunk。"""

    if not query.strip():
        raise ValueError("query不能为空")
    if limit < 1 or candidate_limit < limit:
        raise ValueError("candidate_limit必须不小于limit，且limit不能小于1")
    if max_per_section < 1:
        raise ValueError("max_per_section不能小于1")
    if not 0.0 <= duplicate_threshold <= 1.0:
        raise ValueError("duplicate_threshold必须在0到1之间")

    all_chunks = load_document_chunks(chunk_store_path)
    candidates = tuple(
        chunk
        for chunk in all_chunks
        if (product_id is None or chunk.product_id == product_id)
        and _has_enough_information(chunk)
    )
    if not candidates:
        return []

    keyword_matches = keyword_search_manual_chunks(
        query, candidates, limit=min(candidate_limit, len(candidates))
    )
    keyword_by_id = {
        match.chunk.chunk_id: (rank, match.score)
        for rank, match in enumerate(keyword_matches, start=1)
    }

    embedding_matches: list[ManualSemanticSearchMatch]
    try:
        embedding_matches = semantic_search_manual_chunks(
            query,
            product_id=product_id,
            limit=min(candidate_limit, len(candidates)),
            chunk_store_path=chunk_store_path,
            cache_path=cache_path,
            settings=settings,
        )
    except EmbeddingServiceError:
        embedding_matches = []
    embedding_by_id = {
        match.chunk.chunk_id: (rank, match.similarity)
        for rank, match in enumerate(embedding_matches, start=1)
    }

    ranked: list[ManualHybridSearchMatch] = []
    for chunk in candidates:
        keyword_data = keyword_by_id.get(chunk.chunk_id)
        embedding_data = embedding_by_id.get(chunk.chunk_id)
        if keyword_data is None and embedding_data is None:
            continue
        keyword_rank = keyword_data[0] if keyword_data else None
        keyword_score = keyword_data[1] if keyword_data else None
        embedding_rank = embedding_data[0] if embedding_data else None
        embedding_similarity = embedding_data[1] if embedding_data else None
        fused_score = 0.0
        methods: list[str] = []
        if keyword_rank is not None:
            fused_score += 1.0 / (RRF_K + keyword_rank)
            methods.append("keyword")
        if embedding_rank is not None:
            fused_score += 1.0 / (RRF_K + embedding_rank)
            methods.append("embedding")
        ranked.append(
            ManualHybridSearchMatch(
                chunk=chunk,
                fused_score=fused_score,
                keyword_score=keyword_score,
                keyword_rank=keyword_rank,
                embedding_similarity=embedding_similarity,
                embedding_rank=embedding_rank,
                retrieval_methods=tuple(methods),
            )
        )

    ranked.sort(
        key=lambda item: (
            -item.fused_score,
            item.keyword_rank if item.keyword_rank is not None else 10_000,
            item.embedding_rank if item.embedding_rank is not None else 10_000,
            item.chunk.chunk_id,
        )
    )
    selected = _select_diverse_matches(
        ranked,
        limit=min(limit, len(ranked)),
        max_per_section=max_per_section,
        duplicate_threshold=duplicate_threshold,
    )

    chunks_by_id = {chunk.chunk_id: chunk for chunk in all_chunks}

    def same_section_neighbor(
        match: ManualHybridSearchMatch,
        neighbor_id: str | None,
    ) -> TechnicalDocumentChunk | None:
        if not neighbor_id:
            return None
        neighbor = chunks_by_id.get(neighbor_id)
        if (
            neighbor is None
            or neighbor.document_id != match.chunk.document_id
            or neighbor.section != match.chunk.section
        ):
            return None
        return neighbor

    return [
        ManualEvidenceGroup(
            anchor=match,
            previous_chunk=same_section_neighbor(
                match, match.chunk.previous_chunk_id
            ),
            next_chunk=same_section_neighbor(match, match.chunk.next_chunk_id),
        )
        for match in selected
    ]
