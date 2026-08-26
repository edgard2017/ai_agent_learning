"""使用本地小模型完成文本 Embedding 和语义相似度搜索。"""

from dataclasses import dataclass
import json
from math import fsum, sqrt
from pathlib import Path
from typing import Literal, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import Settings, get_settings
from .embedding_cache import (
    DEFAULT_DOCUMENT_EMBEDDING_CACHE,
    get_or_create_document_embeddings,
)


EmbeddingInputType = Literal["query", "document"]

NOMIC_INPUT_PREFIXES: dict[EmbeddingInputType, str] = {
    "query": "search_query: ",
    "document": "search_document: ",
}

QWEN_QUERY_INSTRUCTION = (
    "Given a marine equipment technical query, retrieve passages that answer "
    "the query."
)


class EmbeddingServiceError(RuntimeError):
    """本地 Embedding 服务不可用或返回格式不正确。"""


@dataclass(frozen=True)
class SemanticSearchMatch:
    """一条按语义相似度排序的文本搜索结果。"""

    document_index: int
    document: str
    similarity: float


def _format_embedding_input(
    text: str,
    *,
    input_type: EmbeddingInputType,
    model_name: str,
) -> str:
    """按模型要求组织查询和文档输入。"""

    if "qwen3-embedding" in model_name.casefold():
        if input_type == "query":
            return f"Instruct: {QWEN_QUERY_INSTRUCTION}\nQuery:{text}"
        return text
    return f"{NOMIC_INPUT_PREFIXES[input_type]}{text}"


def create_embeddings(
    texts: Sequence[str],
    *,
    input_type: EmbeddingInputType,
    settings: Settings | None = None,
    timeout_seconds: float = 30,
) -> list[list[float]]:
    """调用 Ollama，把多段文字批量转换成向量。"""

    if not texts:
        raise ValueError("texts 不能为空")
    cleaned_texts = [text.strip() for text in texts]
    if any(not text for text in cleaned_texts):
        raise ValueError("texts 不能包含空字符串")
    if input_type not in NOMIC_INPUT_PREFIXES:
        raise ValueError("input_type 必须是 query 或 document")

    settings = settings or get_settings()
    prefixed_texts = [
        _format_embedding_input(
            text,
            input_type=input_type,
            model_name=settings.ollama_embedding_model,
        )
        for text in cleaned_texts
    ]
    payload = json.dumps(
        {
            "model": settings.ollama_embedding_model,
            "input": prefixed_texts,
        }
    ).encode("utf-8")
    request = Request(
        f"{settings.ollama_embedding_base_url.rstrip('/')}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise EmbeddingServiceError(
            "无法调用本地 Embedding 服务，请检查 Ollama 地址和模型。"
        ) from error

    embeddings = response_data.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(cleaned_texts):
        raise EmbeddingServiceError("Embedding 服务返回的向量数量不正确")

    vectors: list[list[float]] = []
    for embedding in embeddings:
        if not isinstance(embedding, list) or not embedding:
            raise EmbeddingServiceError("Embedding 服务返回了空向量")
        try:
            vectors.append([float(value) for value in embedding])
        except (TypeError, ValueError) as error:
            raise EmbeddingServiceError("Embedding 向量包含非数字内容") from error
    return vectors


def create_embedding(
    text: str,
    *,
    input_type: EmbeddingInputType,
    settings: Settings | None = None,
) -> list[float]:
    """把一段文字转换成一个向量。"""

    return create_embeddings(
        [text],
        input_type=input_type,
        settings=settings,
    )[0]


def cosine_similarity(
    vector_a: Sequence[float],
    vector_b: Sequence[float],
) -> float:
    """计算两个向量的余弦相似度，结果范围为 -1 到 1。"""

    if not vector_a or not vector_b:
        raise ValueError("向量不能为空")
    if len(vector_a) != len(vector_b):
        raise ValueError("两个向量的维度必须相同")

    dot_product = fsum(a * b for a, b in zip(vector_a, vector_b, strict=True))
    norm_a = sqrt(fsum(value * value for value in vector_a))
    norm_b = sqrt(fsum(value * value for value in vector_b))
    if norm_a == 0 or norm_b == 0:
        raise ValueError("零向量不能计算余弦相似度")

    similarity = dot_product / (norm_a * norm_b)
    return max(-1.0, min(1.0, similarity))


def semantic_search(
    query: str,
    documents: Sequence[str],
    *,
    limit: int = 3,
    settings: Settings | None = None,
    cache_documents: bool = False,
    cache_path: Path = DEFAULT_DOCUMENT_EMBEDDING_CACHE,
) -> list[SemanticSearchMatch]:
    """使用本地 Embedding 模型搜索语义最接近的文本。"""

    if not query.strip():
        raise ValueError("query 不能为空")
    if not documents:
        raise ValueError("documents 不能为空")
    if not 1 <= limit <= len(documents):
        raise ValueError("limit 必须在 1 到文档数量之间")

    settings = settings or get_settings()
    query_vector = create_embedding(
        query,
        input_type="query",
        settings=settings,
    )
    if cache_documents:
        cache_result = get_or_create_document_embeddings(
            documents,
            model_name=settings.ollama_embedding_model,
            create_missing=lambda missing_texts: create_embeddings(
                missing_texts,
                input_type="document",
                settings=settings,
            ),
            cache_path=cache_path,
            expected_dimension=len(query_vector),
        )
        document_vectors = cache_result.vectors
    else:
        document_vectors = create_embeddings(
            documents,
            input_type="document",
            settings=settings,
        )
    matches = [
        SemanticSearchMatch(
            document_index=index,
            document=document,
            similarity=cosine_similarity(query_vector, document_vector),
        )
        for index, (document, document_vector) in enumerate(
            zip(documents, document_vectors, strict=True)
        )
    ]
    matches.sort(key=lambda item: (-item.similarity, item.document_index))
    return matches[:limit]
