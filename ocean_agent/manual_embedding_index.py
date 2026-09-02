"""为清洗后的真实厂家手册Chunk构建并查询Embedding索引。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

from .config import Settings, get_settings
from .document_chunk_store import (
    DEFAULT_DOCUMENT_CHUNK_STORE,
    embedding_text_for_chunk,
    load_document_chunks,
)
from .embedding_cache import (
    DEFAULT_DOCUMENT_EMBEDDING_CACHE,
    DocumentEmbeddingCacheResult,
    get_or_create_document_embeddings,
)
from .embedding_search import cosine_similarity, create_embedding, create_embeddings
from .models import TechnicalDocumentChunk


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANUAL_EMBEDDING_INDEX = (
    PROJECT_ROOT / ".agent_data" / "manual_embedding_index.json"
)
EmbeddingBatchFunction = Callable[[Sequence[str]], list[list[float]]]


@dataclass(frozen=True)
class ManualEmbeddingBuildReport:
    model_name: str
    chunk_count: int
    vector_dimension: int
    cache_hit_count: int
    cache_miss_count: int
    elapsed_seconds: float
    cache_path: str
    index_path: str


@dataclass(frozen=True)
class ManualSemanticSearchMatch:
    rank: int
    chunk: TechnicalDocumentChunk
    similarity: float


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _create_in_batches(
    texts: Sequence[str],
    *,
    batch_size: int,
    embed_batch: EmbeddingBatchFunction,
) -> list[list[float]]:
    if batch_size < 1:
        raise ValueError("batch_size不能小于1")
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(embed_batch(texts[start : start + batch_size]))
    return vectors


def get_manual_chunk_embeddings(
    chunks: Sequence[TechnicalDocumentChunk],
    *,
    settings: Settings | None = None,
    batch_size: int = 16,
    cache_path: Path = DEFAULT_DOCUMENT_EMBEDDING_CACHE,
    embed_batch: EmbeddingBatchFunction | None = None,
    expected_dimension: int | None = None,
) -> DocumentEmbeddingCacheResult:
    if not chunks:
        raise ValueError("chunks不能为空")
    settings = settings or get_settings()
    texts = [embedding_text_for_chunk(chunk) for chunk in chunks]
    batch_function = embed_batch or (
        lambda batch: create_embeddings(
            batch,
            input_type="document",
            settings=settings,
            timeout_seconds=120,
        )
    )
    return get_or_create_document_embeddings(
        texts,
        model_name=settings.ollama_embedding_model,
        create_missing=lambda missing: _create_in_batches(
            missing,
            batch_size=batch_size,
            embed_batch=batch_function,
        ),
        cache_path=cache_path,
        expected_dimension=expected_dimension,
    )


def build_manual_embedding_index(
    *,
    chunk_store_path: Path = DEFAULT_DOCUMENT_CHUNK_STORE,
    cache_path: Path = DEFAULT_DOCUMENT_EMBEDDING_CACHE,
    index_path: Path = DEFAULT_MANUAL_EMBEDDING_INDEX,
    settings: Settings | None = None,
    batch_size: int = 16,
    embed_batch: EmbeddingBatchFunction | None = None,
) -> ManualEmbeddingBuildReport:
    settings = settings or get_settings()
    chunks = load_document_chunks(chunk_store_path)
    started = perf_counter()
    cache_result = get_manual_chunk_embeddings(
        chunks,
        settings=settings,
        batch_size=batch_size,
        cache_path=cache_path,
        embed_batch=embed_batch,
    )
    elapsed = perf_counter() - started
    dimension = len(cache_result.vectors[0])
    index_payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": settings.ollama_embedding_model,
        "vector_dimension": dimension,
        "chunk_store_path": str(Path(chunk_store_path).resolve()),
        "chunk_store_sha256": hashlib.sha256(
            Path(chunk_store_path).read_bytes()
        ).hexdigest(),
        "embedding_cache_path": str(Path(cache_path).resolve()),
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "content_hash": chunk.content_hash,
                "embedding_text_hash": hashlib.sha256(
                    embedding_text_for_chunk(chunk).encode()
                ).hexdigest(),
            }
            for chunk in chunks
        ],
    }
    _write_json_atomic(index_path, index_payload)
    return ManualEmbeddingBuildReport(
        model_name=settings.ollama_embedding_model,
        chunk_count=len(chunks),
        vector_dimension=dimension,
        cache_hit_count=cache_result.hit_count,
        cache_miss_count=cache_result.miss_count,
        elapsed_seconds=elapsed,
        cache_path=str(cache_result.cache_path.resolve()),
        index_path=str(index_path.resolve()),
    )


def semantic_search_manual_chunks(
    query: str,
    *,
    product_id: str | None = None,
    limit: int = 20,
    chunk_store_path: Path = DEFAULT_DOCUMENT_CHUNK_STORE,
    cache_path: Path = DEFAULT_DOCUMENT_EMBEDDING_CACHE,
    settings: Settings | None = None,
    batch_size: int = 16,
    query_embedder: Callable[[str], list[float]] | None = None,
    document_embedder: EmbeddingBatchFunction | None = None,
) -> list[ManualSemanticSearchMatch]:
    if not query.strip():
        raise ValueError("query不能为空")
    all_chunks = load_document_chunks(chunk_store_path)
    chunks = tuple(
        chunk
        for chunk in all_chunks
        if product_id is None or chunk.product_id == product_id
    )
    if not chunks:
        return []
    if limit < 1:
        raise ValueError("limit不能小于1")
    effective_limit = min(limit, len(chunks))

    settings = settings or get_settings()
    make_query_vector = query_embedder or (
        lambda text: create_embedding(text, input_type="query", settings=settings)
    )
    query_vector = make_query_vector(query.strip())
    cache_result = get_manual_chunk_embeddings(
        chunks,
        settings=settings,
        batch_size=batch_size,
        cache_path=cache_path,
        embed_batch=document_embedder,
        expected_dimension=len(query_vector),
    )
    ranked = sorted(
        (
            (index, cosine_similarity(query_vector, vector))
            for index, vector in enumerate(cache_result.vectors)
        ),
        key=lambda item: (-item[1], chunks[item[0]].chunk_id),
    )[:effective_limit]
    return [
        ManualSemanticSearchMatch(
            rank=rank,
            chunk=chunks[index],
            similarity=similarity,
        )
        for rank, (index, similarity) in enumerate(ranked, start=1)
    ]


def main() -> None:
    report = build_manual_embedding_index()
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
