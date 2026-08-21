"""来自厂家官网公开资料的第一版学习产品目录。

注意：这些是第三方公开产品，不代表杭州海询科技有限公司自研、代理或在售。
采集日期统一为 2026-08-19；官网未明确的字段没有自行补写。
"""

from datetime import date

from .models import (
    DeploymentType,
    DepthConfiguration,
    MeasurementSpec,
    Product,
    SamplingConfiguration,
    SourceReference,
    SourceType,
    VerificationStatus,
)


COLLECTED_ON = date(2026, 8, 19)


def official_source(title: str, url: str, version: str | None = None) -> SourceReference:
    return SourceReference(
        title=title,
        url=url,
        source_type=SourceType.MANUFACTURER_OFFICIAL,
        accessed_on=COLLECTED_ON,
        document_version=version,
        verification_status=VerificationStatus.VERIFIED,
    )


PRODUCTS: tuple[Product, ...] = (
    Product(
        product_id="seabird-sbe-19plus-v2",
        manufacturer="Sea-Bird Scientific",
        model="SBE 19plus V2 SeaCAT",
        family="CTD",
        use_cases=("船载剖面观测", "自容式 CTD 数据采集"),
        deployment_types=(DeploymentType.PROFILING,),
        standard_parameters=("conductivity", "temperature", "pressure"),
        derived_parameters=("salinity", "depth", "density", "sound_speed"),
        depth_configurations=(
            DepthConfiguration(depth_rating_m=600, housing="acetal copolymer"),
            DepthConfiguration(depth_rating_m=7000, housing="titanium"),
            DepthConfiguration(depth_rating_m=10500, housing="titanium"),
        ),
        sampling=(
            SamplingConfiguration(rate_text="4 Hz", condition="strain-gauge pressure"),
            SamplingConfiguration(rate_text="2 Hz", condition="quartz pressure"),
        ),
        measurement_specs=(
            MeasurementSpec(
                parameter="conductivity",
                accuracy="±0.0005 S/m",
                resolution="0.00005 S/m typical",
            ),
            MeasurementSpec(
                parameter="temperature",
                accuracy="±0.005 °C",
                resolution="0.0001 °C",
            ),
            MeasurementSpec(
                parameter="pressure",
                accuracy="±0.1% FS strain-gauge; ±0.02% FS quartz",
                resolution="0.002% FS strain-gauge; 0.0025% FS quartz",
            ),
        ),
        power=("9 alkaline D cells", "approximately 60 hours profiling"),
        communications=("RS-232",),
        housings=("acetal copolymer", "titanium"),
        expansion=("auxiliary sensors", "optional biofouling protection"),
        notes=("深度、采样率和压力精度取决于所选配置。",),
        sources=(
            official_source(
                "SBE 19plus V2 SeaCAT CTD product page",
                "https://www.seabird.com/products/sbe-19plus-v2-seacat-ctd",
            ),
        ),
    ),
    Product(
        product_id="seabird-sbe-16plus-v2",
        manufacturer="Sea-Bird Scientific",
        model="SBE 16plus V2 SeaCAT",
        family="CTD",
        use_cases=("长期锚系观测", "固定站点监测"),
        deployment_types=(DeploymentType.MOORED, DeploymentType.FIXED_SITE),
        standard_parameters=("conductivity", "temperature"),
        optional_parameters=("pressure",),
        derived_parameters=("salinity", "depth", "density", "sound_speed"),
        depth_configurations=(
            DepthConfiguration(depth_rating_m=600, housing="plastic"),
            DepthConfiguration(depth_rating_m=10500, housing="titanium"),
        ),
        sampling=(
            SamplingConfiguration(
                rate_text="programmable from 10 seconds to 4 hours",
                condition="basic C/T sampling interval",
            ),
        ),
        measurement_specs=(
            MeasurementSpec(
                parameter="conductivity",
                accuracy="±0.0005 S/m",
                resolution="0.00005 S/m",
            ),
            MeasurementSpec(
                parameter="temperature",
                accuracy="±0.005 °C",
                resolution="0.0001 °C",
            ),
            MeasurementSpec(
                parameter="pressure",
                accuracy="±0.1% FS strain-gauge",
                resolution="0.002% FS strain-gauge",
                notes="Pressure is optional.",
            ),
        ),
        power=("9 alkaline D cells", "approximately 355,000 C/T samples"),
        communications=("RS-232",),
        housings=("plastic", "titanium"),
        expansion=("six A/D channels", "one RS-232 auxiliary channel", "biofouling protection"),
        notes=("压力为选配；深度能力取决于壳体配置。",),
        sources=(
            official_source(
                "SBE 16plus V2 SeaCAT CTD product page",
                "https://www.seabird.com/products/sbe-16plus-v2-seacat-ctd",
            ),
        ),
    ),
    Product(
        product_id="seabird-sbe-37-microcat",
        manufacturer="Sea-Bird Scientific",
        model="SBE 37 MicroCAT",
        family="MicroCAT CT(D)",
        use_cases=("锚系观测", "长期海洋监测"),
        deployment_types=(DeploymentType.MOORED, DeploymentType.FIXED_SITE),
        standard_parameters=("conductivity", "temperature"),
        optional_parameters=("pressure", "dissolved_oxygen"),
        derived_parameters=("salinity", "depth", "density", "sound_speed"),
        depth_configurations=(),
        sampling=(
            SamplingConfiguration(rate_text="2.4–3.2 seconds per sample"),
        ),
        measurement_specs=(
            MeasurementSpec(
                parameter="conductivity",
                accuracy="±0.0003 S/m",
                resolution="0.00001 S/m",
            ),
            MeasurementSpec(
                parameter="temperature",
                range_text="-5 to 45 °C",
                accuracy="±0.002 °C (-5 to 35 °C); ±0.01 °C (35 to 45 °C)",
                resolution="0.0001 °C",
            ),
            MeasurementSpec(
                parameter="pressure",
                accuracy="±0.1% FS",
                resolution="0.002% FS",
                notes="Pressure is optional.",
            ),
            MeasurementSpec(
                parameter="dissolved_oxygen",
                accuracy="larger of ±3 µmol/kg or ±2%",
                resolution="0.2 µmol/kg",
                notes="Dissolved oxygen is optional.",
            ),
        ),
        power=("internal battery",),
        communications=("serial interface; exact option depends on model configuration",),
        housings=("configuration dependent",),
        expansion=("optional pressure", "optional dissolved oxygen", "biofouling protection"),
        notes=(
            "本轮查阅的官网摘要未明确给出统一深度等级，因此深度字段保持为空。",
            "未核验的电池容量和接口细分未写入。",
        ),
        sources=(
            official_source(
                "SBE 37 MicroCAT product page",
                "https://www.seabird.com/products/sbe-37-microcat?id=54627892408",
            ),
        ),
    ),
    Product(
        product_id="rbr-concerto3-ctd",
        manufacturer="RBR",
        model="RBRconcerto³ C.T.D",
        family="Standard CTD logger",
        use_cases=("自容式 CTD 观测", "长期部署", "剖面或锚系数据记录"),
        deployment_types=(DeploymentType.PROFILING, DeploymentType.MOORED),
        standard_parameters=("conductivity", "temperature", "pressure"),
        derived_parameters=("salinity", "depth", "density", "sound_speed"),
        depth_configurations=(
            DepthConfiguration(
                pressure_range_dbar=750,
                housing="plastic",
                notes="Plastic pressure options: 20/50/100/200/500/750 dbar.",
            ),
            DepthConfiguration(
                depth_rating_m=6000,
                pressure_range_dbar=6000,
                housing="titanium",
                notes="Titanium pressure options: 1000/2000/4000/6000 dbar; up to 6000 m.",
            ),
        ),
        sampling=(
            SamplingConfiguration(rate_text="2 Hz standard"),
            SamplingConfiguration(rate_text="up to 32 Hz", condition="optional configuration"),
        ),
        measurement_specs=(
            MeasurementSpec(
                parameter="conductivity",
                range_text="0 to 85 mS/cm",
                accuracy="±0.003 mS/cm",
                resolution="0.0001 mS/cm",
                notes="Stability: 0.010 mS/cm/year.",
            ),
            MeasurementSpec(
                parameter="temperature",
                range_text="-5 to 35 °C",
                accuracy="±0.002 °C",
                resolution="<0.00005 °C",
                notes="Stability: 0.002 °C/year.",
            ),
            MeasurementSpec(
                parameter="pressure",
                accuracy="±0.05% FS",
                resolution="<0.001% FS",
            ),
        ),
        power=("8 AA cells", "external power 4.5–30 V"),
        communications=("USB-C", "RS-232 or RS-485"),
        housings=("plastic", "titanium"),
        expansion=("C.T.D+ and C.T.D++ variants", "auxiliary sensor integration"),
        notes=("水深、压力范围、采样率和通信接口均取决于具体配置。",),
        sources=(
            official_source(
                "RBRconcerto³ C.T.D product page",
                "https://rbr-global.com/products/standard-loggers/rbrduo-ct/",
            ),
            official_source(
                "RBR CT and CTD Instruments datasheet",
                "https://rbr-global.com/wp-content/uploads/2024/12/RBR-CT-and-CTD-Instruments-RIG-0013236revF.pdf",
                "RIG-0013236 rev F",
            ),
        ),
    ),
)
