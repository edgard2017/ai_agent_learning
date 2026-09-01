"""从清单下载厂家公开资料到本地 raw 目录。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_official_documents(
    documents_dir: str | Path,
    *,
    manifest_name: str = "official_manifest.json",
    overwrite: bool = False,
) -> tuple[dict[str, str], ...]:
    """下载清单里的PDF；已存在且校验正确的文件不会重复下载。"""

    root = Path(documents_dir).resolve()
    manifest = json.loads((root / manifest_name).read_text(encoding="utf-8"))
    results: list[dict[str, str]] = []

    for entry in manifest.get("documents", []):
        url = entry.get("download_url")
        if not isinstance(url, str) or urlparse(url).scheme != "https":
            raise ValueError(f"download_url 必须是 HTTPS：{url}")

        target = (root / entry["file"]).resolve()
        if not target.is_relative_to(root / "raw"):
            raise ValueError(f"下载文件必须放在 documents/raw：{entry['file']}")
        target.parent.mkdir(parents=True, exist_ok=True)

        expected = entry.get("sha256")
        if target.exists() and not overwrite:
            actual = sha256_file(target)
            if expected and actual != expected:
                raise ValueError(f"本地文件校验失败：{target.name}")
            results.append({"file": str(target), "sha256": actual, "status": "cached"})
            continue

        request = Request(url, headers={"User-Agent": "ai-agent-learning/1.0"})
        download_temporary = target.with_suffix(target.suffix + ".download.part")
        output_temporary = target.with_suffix(target.suffix + ".part")
        try:
            with urlopen(request, timeout=60) as response, download_temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)

            archive_member = entry.get("archive_member")
            if archive_member:
                archive_expected = entry.get("archive_sha256")
                archive_actual = sha256_file(download_temporary)
                if archive_expected and archive_actual != archive_expected:
                    raise ValueError(f"下载ZIP校验失败：{target.name}")
                with ZipFile(download_temporary) as archive:
                    try:
                        pdf_bytes = archive.read(archive_member)
                    except KeyError as exc:
                        raise ValueError(f"ZIP中找不到文件：{archive_member}") from exc
                output_temporary.write_bytes(pdf_bytes)
            else:
                download_temporary.replace(output_temporary)

            if output_temporary.read_bytes()[:5] != b"%PDF-":
                raise ValueError(f"下载内容不是有效PDF：{url}")
            actual = sha256_file(output_temporary)
            if expected and actual != expected:
                raise ValueError(f"下载文件校验失败：{target.name}")
            output_temporary.replace(target)
        finally:
            download_temporary.unlink(missing_ok=True)
            output_temporary.unlink(missing_ok=True)

        results.append({"file": str(target), "sha256": actual, "status": "downloaded"})

    return tuple(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="下载厂家公开PDF资料")
    parser.add_argument("--documents-dir", default="documents")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for result in download_official_documents(
        args.documents_dir, overwrite=args.overwrite
    ):
        print(f"{result['status']}: {result['file']}")
        print(f"sha256: {result['sha256']}")


if __name__ == "__main__":
    main()
