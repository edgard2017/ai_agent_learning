"""以后可以直接注册给 Agent 使用的普通 Python 查询函数。"""

from collections.abc import Iterable

from .models import DeploymentType, Product
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
    """按完整型号或 product_id 查询；查不到就明确返回 None。"""

    query = model_or_id.strip().casefold()
    for product in PRODUCTS:
        if query in {product.model.casefold(), product.product_id.casefold()}:
            return product
    return None
