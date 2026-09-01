import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ocean_agent.document_downloader import download_official_documents


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class DocumentDownloaderTests(unittest.TestCase):
    def test_downloads_and_then_reuses_verified_pdf(self) -> None:
        pdf = b"%PDF-1.4\nsmall test file"
        checksum = hashlib.sha256(pdf).hexdigest()
        manifest = {
            "documents": [
                {
                    "file": "raw/test.pdf",
                    "download_url": "https://manufacturer.example/test.pdf",
                    "sha256": checksum,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "official_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with patch(
                "ocean_agent.document_downloader.urlopen",
                return_value=FakeResponse(pdf),
            ) as mocked_urlopen:
                first = download_official_documents(root)
                second = download_official_documents(root)

            self.assertEqual(first[0]["status"], "downloaded")
            self.assertEqual(second[0]["status"], "cached")
            self.assertEqual((root / "raw/test.pdf").read_bytes(), pdf)
            mocked_urlopen.assert_called_once()

    def test_rejects_non_https_download(self) -> None:
        manifest = {
            "documents": [
                {
                    "file": "raw/test.pdf",
                    "download_url": "http://manufacturer.example/test.pdf",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "official_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "必须是 HTTPS"):
                download_official_documents(root)


if __name__ == "__main__":
    unittest.main()
