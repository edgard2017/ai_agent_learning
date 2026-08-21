"""集中读取项目配置，业务代码不直接接触 .env 文件。"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """从工程根目录的 .env 或系统环境变量读取配置。"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_provider: Literal["ollama", "openai"] = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11435/v1"
    ollama_model: str = "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.6-luna"

    @field_validator("ollama_base_url", "ollama_model", "openai_model")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("模型地址和模型名称不能为空")
        return text

    @model_validator(mode="after")
    def validate_provider_credentials(self) -> "Settings":
        if self.model_provider != "openai":
            return self

        secret = (
            self.openai_api_key.get_secret_value().strip()
            if self.openai_api_key is not None
            else ""
        )
        if not secret or secret == "your-api-key-here":
            raise ValueError(
                "MODEL_PROVIDER=openai 时，请在工程根目录的 .env 中填写 OPENAI_API_KEY"
            )
        self.openai_api_key = SecretStr(secret)
        return self


@lru_cache
def get_settings() -> Settings:
    """只加载一次配置，供 Agent 和其他模块统一使用。"""

    return Settings()
