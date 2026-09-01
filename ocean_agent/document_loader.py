"""读取项目中的 Markdown/TXT 技术资料及其元数据。"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from .models import LoadedDocument, SourceReference
from .product_data import PRODUCTS


SUPPORTED_EXTENSIONS = {".md", ".txt"}
KNOWN_PRODUCT_IDS = {product.product_id for product in PRODUCTS}


def load_documents(documents_dir: str | Path) -> tuple[LoadedDocument, ...]:
    """根据 documents/manifest.json 读取全部文档。

    manifest 保存产品ID、来源等结构化信息，Markdown/TXT 只保存正文，
    这样资料内容和元数据各司其职。
    """

    root = Path(documents_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到文档清单：{manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"文档清单不是有效 JSON：{exc}") from exc

    entries = manifest.get("documents")
    if not isinstance(entries, list):
        raise ValueError("manifest.json 必须包含 documents 数组")

    documents: list[LoadedDocument] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"documents 第 {index} 项必须是对象")

        document_id = entry.get("document_id")
        if document_id in seen_ids:
            raise ValueError(f"document_id 重复：{document_id}")
        seen_ids.add(document_id)

        product_id = entry.get("product_id")
        if product_id not in KNOWN_PRODUCT_IDS:
            raise ValueError(f"未知 product_id：{product_id}")

        relative_file = entry.get("file")
        if not isinstance(relative_file, str) or not relative_file.strip():
            raise ValueError(f"documents 第 {index} 项缺少 file")
        file_path = (root / relative_file).resolve()
        if not file_path.is_relative_to(root):
            raise ValueError(f"文档路径不能超出 documents 目录：{relative_file}")
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"暂不支持该文档格式：{file_path.suffix}")
        if not file_path.is_file():
            raise FileNotFoundError(f"找不到文档：{file_path}")

        content = file_path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"文档内容为空：{relative_file}")

        try:
            source = SourceReference.model_validate(entry.get("source"))
            document = LoadedDocument(
                document_id=document_id,
                product_id=product_id,
                title=entry.get("title", ""),
                content=content,
                keywords=tuple(entry.get("keywords", ())),
                source=source,
                file_path=str(file_path),
            )
        except ValidationError as exc:
            raise ValueError(f"文档元数据无效（第 {index} 项）：{exc}") from exc
        documents.append(document)

    return tuple(documents)
