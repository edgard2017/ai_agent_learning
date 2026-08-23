"""以后可以直接注册给 Agent 使用的普通 Python 查询函数。"""

from collections.abc import Iterable
from dataclasses import dataclass
import re
import unicodedata

from .document_data import DOCUMENT_CHUNKS
from .models import DeploymentType, Product, TechnicalDocumentChunk
from .product_data import PRODUCTS


PARAMETER_ALIASES = {
    "c": "conductivity",
    "conductivity": "conductivity",
    "电导率": "conductivity",
    "t": "temperature",
    "temperature": "temperature",
    "温度": "temperature",
    "p": "pressure",
    "pressure": "pressure",
    "压力": "pressure",
    "盐度": "salinity",
    "salinity": "salinity",
    "深度": "depth",
    "depth": "depth",
    "溶解氧": "dissolved_oxygen",
    "do": "dissolved_oxygen",
    "dissolved_oxygen": "dissolved_oxygen",
}


def _normalize_parameter(value: str) -> str:
    cleaned = value.strip().lower()
    return PARAMETER_ALIASES.get(cleaned, cleaned.replace(" ", "_"))


def search_products(
    *,
    minimum_depth_m: int | None = None,
    required_parameters: Iterable[str] = (),
    deployment_type: DeploymentType | str | None = None,
) -> list[Product]:
    """按水深、参数和部署方式筛选公开产品。

    选配参数也算“产品可以支持”，但调用者应阅读产品的 notes，确认是否需要选配。
    如果提出水深要求，深度未知的产品不会被返回，防止把“未知”误当成“满足”。
    """

    if minimum_depth_m is not None and minimum_depth_m < 0:
        raise ValueError("minimum_depth_m 不能小于 0")

    requested = {_normalize_parameter(item) for item in required_parameters if item.strip()}
    deployment = DeploymentType(deployment_type) if deployment_type else None

    matches: list[Product] = []
    for product in PRODUCTS:
        if not requested.issubset(product.supported_parameters):
            continue
        if deployment and deployment not in product.deployment_types:
            continue
        if minimum_depth_m is not None:
            documented_depth = product.maximum_documented_depth_m
            if documented_depth is None or documented_depth < minimum_depth_m:
                continue
        matches.append(product)

    return matches


def get_product_spec(model_or_id: str) -> Product | None:
    """按型号或 product_id 查询；简称只有唯一命中时才接受。"""

    query = _normalize_text(model_or_id.strip())
    for product in PRODUCTS:
        if query in {
            _normalize_text(product.model),
            _normalize_text(product.product_id),
        }:
            return product

    if len(query) >= 4:
        partial_matches = [
            product
            for product in PRODUCTS
            if query in _normalize_text(product.model)
            or query in _normalize_text(product.product_id)
        ]
        if len(partial_matches) == 1:
            return partial_matches[0]
    return None


@dataclass(frozen=True)
class DocumentSearchMatch:
    """技术资料检索结果；score 只用于排序，不属于厂家事实。"""

    chunk: TechnicalDocumentChunk
    score: int


def _normalize_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _query_terms(query: str) -> set[str]:
    """提取英文/数字词和中文二元片段，支持没有空格的中文问题。"""

    normalized = _normalize_text(query)
    terms = set(re.findall(r"[a-z0-9]+(?:[.+-][a-z0-9]+)*", normalized))
    chinese_groups = re.findall(r"[\u4e00-\u9fff]+", normalized)
    for group in chinese_groups:
        if len(group) == 1:
            terms.add(group)
        else:
            terms.update(group[index : index + 2] for index in range(len(group) - 1))
    return terms


def search_documents(
    query: str,
    *,
    model_or_id: str | None = None,
    limit: int = 3,
) -> list[DocumentSearchMatch]:
    """从本地技术资料片段中检索相关内容。

    这是第一版关键词检索：不调用模型、不联网，也不使用向量数据库。
    """

    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("query 不能为空")
    if not 1 <= limit <= 10:
        raise ValueError("limit 必须在 1 到 10 之间")

    product_id: str | None = None
    if model_or_id and model_or_id.strip():
        product = get_product_spec(model_or_id)
        if product is None:
            return []
        product_id = product.product_id

    normalized_query = _normalize_text(cleaned_query)
    query_terms = _query_terms(cleaned_query)
    matches: list[DocumentSearchMatch] = []

    for chunk in DOCUMENT_CHUNKS:
        if product_id and chunk.product_id != product_id:
            continue

        searchable = _normalize_text(
            " ".join(
                (
                    chunk.title,
                    chunk.section,
                    chunk.content,
                    *chunk.keywords,
                )
            )
        )
        score = sum(1 for term in query_terms if term in searchable)
        score += sum(
            4
            for keyword in chunk.keywords
            if _normalize_text(keyword) in normalized_query
        )
        if normalized_query in searchable:
            score += 8

        # 只有一个普通词重合时容易误命中，例如“维修步骤”碰到“接线步骤”。
        if score >= 2:
            matches.append(DocumentSearchMatch(chunk=chunk, score=score))

    matches.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
    return matches[:limit]
