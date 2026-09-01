"""产品目录的数据模型。

这里保存的是可核验的结构化事实，不保存厂家整本说明书的原文。
"""

from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class SourceType(str, Enum):
    MANUFACTURER_OFFICIAL = "manufacturer_official"
    SIMULATED = "simulated"
    INTERNAL = "internal"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    NEEDS_REVIEW = "needs_review"


class DeploymentType(str, Enum):
    PROFILING = "profiling"
    MOORED = "moored"
    FIXED_SITE = "fixed_site"


class SourceReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    url: HttpUrl
    source_type: SourceType
    accessed_on: date
    document_version: str | None = None
    verification_status: VerificationStatus


class MeasurementSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    parameter: str
    range_text: str | None = None
    accuracy: str | None = None
    resolution: str | None = None
    notes: str | None = None


class DepthConfiguration(BaseModel):
    """一个具体配置的耐压/工作深度，避免把不同壳体混成一个数字。"""

    model_config = ConfigDict(frozen=True)

    depth_rating_m: int | None = Field(default=None, ge=0)
    pressure_range_dbar: int | None = Field(default=None, ge=0)
    housing: str
    notes: str | None = None


class SamplingConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    rate_text: str
    condition: str | None = None


class Product(BaseModel):
    model_config = ConfigDict(frozen=True)

    product_id: str = Field(pattern=r"^[a-z0-9-]+$")
    manufacturer: str
    model: str
    family: str
    data_scope: str = "third_party_public_product"
    use_cases: tuple[str, ...]
    deployment_types: tuple[DeploymentType, ...]
    standard_parameters: tuple[str, ...]
    optional_parameters: tuple[str, ...] = ()
    derived_parameters: tuple[str, ...] = ()
    depth_configurations: tuple[DepthConfiguration, ...] = ()
    sampling: tuple[SamplingConfiguration, ...]
    measurement_specs: tuple[MeasurementSpec, ...]
    power: tuple[str, ...]
    communications: tuple[str, ...]
    housings: tuple[str, ...]
    expansion: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    sources: tuple[SourceReference, ...]

    @field_validator(
        "standard_parameters", "optional_parameters", "derived_parameters"
    )
    @classmethod
    def normalize_parameters(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(value.strip().lower() for value in values)

    @property
    def supported_parameters(self) -> set[str]:
        return set(
            self.standard_parameters
            + self.optional_parameters
            + self.derived_parameters
        )

    @property
    def maximum_documented_depth_m(self) -> int | None:
        depths = [
            item.depth_rating_m
            for item in self.depth_configurations
            if item.depth_rating_m is not None
        ]
        return max(depths) if depths else None


class TechnicalDocumentChunk(BaseModel):
    """一段可以独立检索和引用的技术资料。"""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(pattern=r"^[a-z0-9-]+$")
    product_id: str = Field(pattern=r"^[a-z0-9-]+$")
    title: str
    section: str
    content: str
    keywords: tuple[str, ...] = ()
    source: SourceReference
    page_number: int | None = Field(default=None, ge=1)


class LoadedDocument(BaseModel):
    """从文件读取、但尚未切块的一份技术资料。"""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(pattern=r"^[a-z0-9-]+$")
    product_id: str = Field(pattern=r"^[a-z0-9-]+$")
    title: str
    content: str
    keywords: tuple[str, ...] = ()
    source: SourceReference
    file_path: str
    pages: tuple[str, ...] = ()
