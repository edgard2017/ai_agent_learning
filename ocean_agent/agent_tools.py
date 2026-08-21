"""把确定性的产品查询函数包装成 OpenAI Agents SDK Tool。"""

import json

from agents.decorators import tool

from .models import DeploymentType, Product
from .tools import get_product_spec, search_products


def _parameter_role(product: Product, parameter: str) -> str:
    """说明某参数是标配、选配还是派生值。"""

    if parameter in product.standard_parameters:
        return "standard"
    if parameter in product.optional_parameters:
        return "optional"
    if parameter in product.derived_parameters:
        return "derived"
    return "unknown"


def search_ocean_products_data(
    minimum_depth_m: int | None = None,
    required_parameters: list[str] | None = None,
    deployment_type: str | None = None,
) -> str:
    """执行产品筛选并返回便于模型读取的 JSON。"""

    parameters = required_parameters or []
    normalized_deployment = deployment_type
    if deployment_type is not None and deployment_type.strip().lower() in {
        "",
        "null",
        "none",
    }:
        normalized_deployment = None
    matches = search_products(
        minimum_depth_m=minimum_depth_m,
        required_parameters=parameters,
        deployment_type=normalized_deployment,
    )

    products = []
    for product in matches:
        products.append(
            {
                "product_id": product.product_id,
                "manufacturer": product.manufacturer,
                "model": product.model,
                "maximum_documented_depth_m": product.maximum_documented_depth_m,
                "deployment_types": [item.value for item in product.deployment_types],
                "standard_parameters": list(product.standard_parameters),
                "optional_parameters": list(product.optional_parameters),
                "derived_parameters": list(product.derived_parameters),
                "configuration_notes": list(product.notes),
                "sources": [str(source.url) for source in product.sources],
            }
        )

    payload = {
        "count": len(products),
        "products": products,
        "message": (
            "没有满足全部条件的已收录产品；不要编造型号。"
            if not products
            else "候选只来自当前公开产品目录，仍需按具体配置向厂家核验。"
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


def get_ocean_product_details_data(model_or_id: str) -> str:
    """按产品型号或 ID 查询完整详情，并返回便于模型读取的 JSON。"""

    product = get_product_spec(model_or_id)
    if product is None:
        return json.dumps(
            {
                "found": False,
                "product": None,
                "message": "当前已核验目录中没有这个型号；不要根据模型记忆补全参数。",
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "found": True,
            "product": product.model_dump(mode="json"),
            "message": "数据来自当前公开产品目录，具体配置仍需向厂家核验。",
        },
        ensure_ascii=False,
    )


def compare_ocean_products_data(product_ids: list[str]) -> str:
    """对齐比较多个已收录产品；不生成主观推荐。"""

    unique_ids: list[str] = []
    seen: set[str] = set()
    for product_id in product_ids:
        cleaned = product_id.strip()
        normalized = cleaned.casefold()
        if cleaned and normalized not in seen:
            unique_ids.append(cleaned)
            seen.add(normalized)

    products: list[Product] = []
    missing_product_ids: list[str] = []
    for product_id in unique_ids:
        product = get_product_spec(product_id)
        if product is None:
            missing_product_ids.append(product_id)
        else:
            products.append(product)

    comparison_rows = [
        {
            "product_id": product.product_id,
            "manufacturer": product.manufacturer,
            "model": product.model,
            "maximum_documented_depth_m": product.maximum_documented_depth_m,
            "deployment_types": [item.value for item in product.deployment_types],
            "standard_parameters": list(product.standard_parameters),
            "optional_parameters": list(product.optional_parameters),
            "derived_parameters": list(product.derived_parameters),
            "depth_configurations": [
                item.model_dump(mode="json") for item in product.depth_configurations
            ],
            "sampling": [item.model_dump(mode="json") for item in product.sampling],
            "communications": list(product.communications),
            "housings": list(product.housings),
            "configuration_notes": list(product.notes),
            "sources": [str(source.url) for source in product.sources],
        }
        for product in products
    ]

    common_supported_parameters = (
        sorted(set.intersection(*(product.supported_parameters for product in products)))
        if products
        else []
    )
    can_compare = len(products) >= 2
    return json.dumps(
        {
            "can_compare": can_compare,
            "requested_product_ids": unique_ids,
            "missing_product_ids": missing_product_ids,
            "common_supported_parameters": common_supported_parameters,
            "products": comparison_rows,
            "message": (
                "比较数据已按统一字段对齐；推荐结论必须基于这些数据，并提示配置核验。"
                if can_compare
                else "至少需要两个已收录产品才能比较；不要编造缺失型号或参数。"
            ),
        },
        ensure_ascii=False,
    )


@tool
def search_ocean_products(
    minimum_depth_m: int | None = None,
    required_parameters: list[str] | None = None,
    deployment_type: str | None = None,
) -> str:
    """按水深、测量参数和部署方式查询已核验的公开海洋设备产品。

    Args:
        minimum_depth_m: 要求的最小工作水深（米）；用户未提出时传 null。
        required_parameters: 必须支持的参数，例如 temperature、salinity、pressure。
        deployment_type: 部署方式，只能是 profiling、moored、fixed_site 或 null。
    """

    if deployment_type is not None and deployment_type.strip().lower() not in {
        "",
        "null",
        "none",
    }:
        DeploymentType(deployment_type)
    return search_ocean_products_data(
        minimum_depth_m=minimum_depth_m,
        required_parameters=required_parameters,
        deployment_type=deployment_type,
    )


@tool
def get_ocean_product_details(model_or_id: str) -> str:
    """查询一个已收录海洋设备型号的完整公开参数。

    用户询问某个明确型号的参数、用途、接口、壳体、采样率或来源时使用。

    Args:
        model_or_id: 完整产品型号或产品 ID，例如 SBE 19plus V2 SeaCAT。
    """

    return get_ocean_product_details_data(model_or_id)


@tool
def compare_ocean_products(product_ids: list[str]) -> str:
    """按统一字段比较两个或多个已收录海洋设备产品。

    当候选来自 search_ocean_products 时，必须传入搜索结果返回的 product_id。
    此工具用于比较深度、部署方式、参数角色、采样、接口、壳体和配置风险。

    Args:
        product_ids: 两个或多个产品 ID，例如 seabird-sbe-19plus-v2。
    """

    return compare_ocean_products_data(product_ids)
