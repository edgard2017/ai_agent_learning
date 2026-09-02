"""读取并校验由构建器生成的正式Chunk JSON。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from .models import TechnicalDocumentChunk


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCUMENT_CHUNK_STORE = PROJECT_ROOT / ".agent_data" / "document_chunks.json"


def embedding_text_for_chunk(chunk: TechnicalDocumentChunk) -> str:
    """Embedding只编码可帮助语义检索的标题、章节和正文。"""

    return f"{chunk.title}\n{chunk.section}\n{chunk.content}"


def load_document_chunks(
    path: str | Path = DEFAULT_DOCUMENT_CHUNK_STORE,
) -> tuple[TechnicalDocumentChunk, ...]:
    store_path = Path(path)
    try:
        payload = json.loads(store_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"找不到Chunk JSON：{store_path}，请先运行 "
            "python -m ocean_agent.build_document_chunks"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Chunk JSON格式错误：{exc}") from exc

    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("chunks"), list
    ):
        raise ValueError("不支持的Chunk JSON结构")

    try:
        chunks = tuple(
            TechnicalDocumentChunk.model_validate(item) for item in payload["chunks"]
        )
    except ValidationError as exc:
        raise ValueError(f"Chunk字段校验失败：{exc}") from exc

    expected_count = payload.get("summary", {}).get("chunk_count")
    if expected_count != len(chunks):
        raise ValueError("Chunk数量与summary不一致")

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Chunk JSON包含重复chunk_id")
    known_ids = set(chunk_ids)
    for chunk in chunks:
        actual_hash = hashlib.sha256(chunk.content.encode()).hexdigest()
        if chunk.content_hash != actual_hash:
            raise ValueError(f"Chunk正文哈希不一致：{chunk.chunk_id}")
        for neighbor_id in (chunk.previous_chunk_id, chunk.next_chunk_id):
            if neighbor_id is not None and neighbor_id not in known_ids:
                raise ValueError(
                    f"Chunk邻居不存在：{chunk.chunk_id} -> {neighbor_id}"
                )
    return chunks
