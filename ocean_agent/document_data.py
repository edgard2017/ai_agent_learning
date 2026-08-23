"""供本地 RAG 练习使用的已核验技术资料片段。

当前片段是对厂家公开产品页和数据表的中文整理，不是完整说明书原文。
它们只用于演示“检索后回答”，不代表杭州海询科技有限公司的服务承诺。
"""

from .models import TechnicalDocumentChunk
from .product_data import PRODUCTS


PRODUCT_BY_ID = {product.product_id: product for product in PRODUCTS}


def _chunk(
    *,
    chunk_id: str,
    product_id: str,
    section: str,
    content: str,
    keywords: tuple[str, ...],
    source_index: int = 0,
) -> TechnicalDocumentChunk:
    product = PRODUCT_BY_ID[product_id]
    return TechnicalDocumentChunk(
        chunk_id=chunk_id,
        product_id=product_id,
        title=f"{product.model} 技术资料",
        section=section,
        content=content,
        keywords=(product.model, product.product_id, *keywords),
        source=product.sources[source_index],
    )


DOCUMENT_CHUNKS: tuple[TechnicalDocumentChunk, ...] = (
    _chunk(
        chunk_id="sbe19-depth-housing",
        product_id="seabird-sbe-19plus-v2",
        section="工作深度与壳体",
        content=(
            "SBE 19plus V2 SeaCAT 的公开配置包括 600 m 共聚醋酸酯壳体、"
            "7000 m 钛壳和 10500 m 钛壳。深度能力取决于实际订购的壳体配置，"
            "不能把 10500 m 当成所有版本的统一能力。"
        ),
        keywords=("深度", "水深", "壳体", "耐压", "钛壳", "housing", "depth"),
    ),
    _chunk(
        chunk_id="sbe19-interface-sampling",
        product_id="seabird-sbe-19plus-v2",
        section="通信与采样",
        content=(
            "SBE 19plus V2 SeaCAT 的公开通信接口为 RS-232。应变式压力传感器配置"
            "对应 4 Hz 采样，石英压力传感器配置对应 2 Hz 采样；采样率与压力传感器"
            "选型有关。公开产品页信息不足以说明具体接线步骤。"
        ),
        keywords=("通信", "接口", "连接", "接线", "RS-232", "采样", "频率", "sampling"),
    ),
    _chunk(
        chunk_id="sbe16-deployment-sampling",
        product_id="seabird-sbe-16plus-v2",
        section="部署与采样",
        content=(
            "SBE 16plus V2 SeaCAT 面向长期锚系和固定站点监测。公开资料给出的基础"
            "电导率/温度采样间隔可在 10 秒至 4 小时之间编程设置；压力是选配参数。"
        ),
        keywords=("部署", "锚系", "固定站", "长期", "采样间隔", "压力选配", "moored"),
    ),
    _chunk(
        chunk_id="sbe16-expansion-interface",
        product_id="seabird-sbe-16plus-v2",
        section="通信与扩展",
        content=(
            "SBE 16plus V2 SeaCAT 使用 RS-232 通信，并公开列出 6 个 A/D 通道和"
            "1 个 RS-232 辅助通道用于扩展。辅助传感器兼容性仍需按具体配置向厂家核验。"
        ),
        keywords=("通信", "接口", "扩展", "辅助传感器", "A/D", "RS-232"),
    ),
    _chunk(
        chunk_id="sbe37-known-limits",
        product_id="seabird-sbe-37-microcat",
        section="已核验能力与资料边界",
        content=(
            "当前收录的 SBE 37 MicroCAT 厂家公开摘要确认电导率和温度为标配，压力和"
            "溶解氧为选配。当前资料没有确认统一工作深度，也没有确认细分串行接口，"
            "因此不能据此回答具体深度、连接方式或接线步骤。"
        ),
        keywords=("压力", "溶解氧", "深度", "通信", "接口", "未知", "选配"),
    ),
    _chunk(
        chunk_id="rbr-ctd-depth-housing",
        product_id="rbr-concerto3-ctd",
        section="压力范围与壳体",
        content=(
            "RBRconcerto³ C.T.D 的塑料壳压力选项包括 20、50、100、200、500 和"
            "750 dbar；钛壳压力选项包括 1000、2000、4000 和 6000 dbar，公开数据表"
            "标示钛壳配置最高可到 6000 m。深度和压力范围取决于具体配置。"
        ),
        keywords=("深度", "水深", "压力", "壳体", "塑料", "钛壳", "dbar", "depth"),
        source_index=1,
    ),
    _chunk(
        chunk_id="rbr-ctd-interface-power-sampling",
        product_id="rbr-concerto3-ctd",
        section="通信、电源与采样",
        content=(
            "RBRconcerto³ C.T.D 的公开资料列出 USB-C，以及 RS-232 或 RS-485 通信"
            "选项；供电包括 8 节 AA 电池或 4.5–30 V 外部电源。标准采样率为 2 Hz，"
            "选配配置最高可到 32 Hz。接口和采样能力均需结合订购配置确认。"
        ),
        keywords=("通信", "接口", "连接", "USB-C", "RS-232", "RS-485", "电源", "供电", "采样"),
        source_index=1,
    ),
)
