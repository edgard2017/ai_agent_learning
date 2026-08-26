"""只缓存公开技术资料向量，不缓存用户查询。"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCUMENT_EMBEDDING_CACHE = (
    PROJECT_ROOT / ".agent_data" / "document_embeddings.json"
)
CACHE_VERSION = 1


@dataclass(frozen=True)
class DocumentEmbeddingCacheResult:
    """按原文顺序返回向量，并报告本次缓存命中情况。"""

    vectors: list[list[float]]
    hit_count: int
    miss_count: int
    cache_path: Path


def _text_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _empty_cache() -> dict[str, Any]:
    return {"version": CACHE_VERSION, "models": {}}


def _valid_vector(value: Any, expected_dimension: int | None = None) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and (expected_dimension is None or len(value) == expected_dimension)
        and all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in value
        )
    )


def _read_cache(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_cache()
    if (
        not isinstance(data, dict)
        or data.get("version") != CACHE_VERSION
        or not isinstance(data.get("models"), dict)
    ):
        return _empty_cache()
    return data


def _write_cache(path: Path, data: dict[str, Any]) -> None:
    """使用同目录临时文件原子替换；写失败不影响本次检索。"""

    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            json.dump(data, temporary_file, ensure_ascii=False)
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, path)
    except OSError:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def get_or_create_document_embeddings(
    texts: Sequence[str],
    *,
    model_name: str,
    create_missing: Callable[[Sequence[str]], list[list[float]]],
    cache_path: Path = DEFAULT_DOCUMENT_EMBEDDING_CACHE,
    expected_dimension: int | None = None,
) -> DocumentEmbeddingCacheResult:
    """读取文档向量缓存，只批量生成缺失或内容已变化的向量。"""

    if not texts:
        raise ValueError("texts 不能为空")
    cleaned_texts = [text.strip() for text in texts]
    if any(not text for text in cleaned_texts):
        raise ValueError("texts 不能包含空字符串")

    cache = _read_cache(cache_path)
    models = cache["models"]
    model_cache = models.get(model_name)
    if not isinstance(model_cache, dict):
        model_cache = {}
        models[model_name] = model_cache

    hashes = [_text_hash(text) for text in cleaned_texts]
    missing_indexes = [
        index
        for index, text_hash in enumerate(hashes)
        if not _valid_vector(model_cache.get(text_hash), expected_dimension)
    ]
    if missing_indexes:
        missing_vectors = create_missing(
            [cleaned_texts[index] for index in missing_indexes]
        )
        if len(missing_vectors) != len(missing_indexes):
            raise ValueError("新生成的文档向量数量不正确")
        for index, vector in zip(missing_indexes, missing_vectors, strict=True):
            if not _valid_vector(vector, expected_dimension):
                raise ValueError("新生成的文档向量格式不正确")
            model_cache[hashes[index]] = [float(value) for value in vector]
        _write_cache(cache_path, cache)

    vectors = [model_cache[text_hash] for text_hash in hashes]
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1 or (
        expected_dimension is not None and dimensions != {expected_dimension}
    ):
        raise ValueError("缓存中的文档向量维度不一致")
    return DocumentEmbeddingCacheResult(
        vectors=vectors,
        hit_count=len(cleaned_texts) - len(missing_indexes),
        miss_count=len(missing_indexes),
        cache_path=cache_path,
    )
