import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from ocean_agent.config import Settings


class SettingsTests(unittest.TestCase):
    def test_reads_key_without_exposing_it_in_repr(self) -> None:
        raw_key = "sk-test-not-a-real-key"
        settings = Settings(
            model_provider="openai", openai_api_key=raw_key, _env_file=None
        )

        self.assertEqual(settings.openai_api_key.get_secret_value(), raw_key)
        self.assertNotIn(raw_key, repr(settings))

    def test_local_provider_does_not_require_openai_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        self.assertEqual(settings.model_provider, "ollama")
        self.assertIsNone(settings.openai_api_key)

    def test_openai_provider_requires_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValidationError, "MODEL_PROVIDER=openai"):
                Settings(model_provider="openai", _env_file=None)

    def test_placeholder_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "MODEL_PROVIDER=openai"):
            Settings(
                model_provider="openai",
                openai_api_key="your-api-key-here",
                _env_file=None,
            )

    def test_uses_local_qwen_by_default(self) -> None:
        settings = Settings(_env_file=None)

        self.assertEqual(settings.ollama_base_url, "http://127.0.0.1:11435/v1")
        self.assertIn("Qwen3.8-27B", settings.ollama_model)

    def test_uses_small_local_embedding_model_by_default(self) -> None:
        settings = Settings(_env_file=None)

        self.assertEqual(
            settings.ollama_embedding_base_url,
            "http://127.0.0.1:11435",
        )
        self.assertEqual(
            settings.ollama_embedding_model,
            "qwen3-embedding:0.6b",
        )


if __name__ == "__main__":
    unittest.main()
