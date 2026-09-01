"""清洗技术资料并持久化为可检查的Chunk JSON。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from collections import Counter

from .document_chunker import chunk_document
from .document_cleaner import CleanedDocument, clean_document, infer_chunk_type
from .document_downloader import sha256_file
from .document_loader import load_documents
from .models import TechnicalDocumentChunk


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _persist_cleaned_document(cleaned: CleanedDocument, output_dir: Path) -> None:
    pages = [
        {
            "page_number": page.page_number,
            "raw_text": page.raw_text,
            "raw_text_hash": hashlib.sha256(page.raw_text.encode()).hexdigest(),
            "cleaned_text": page.cleaned_text,
            "cleaning_actions": list(page.cleaning_actions),
            "review_status": page.review_status,
            "excluded_from_chunks": page.excluded_from_chunks,
        }
        for page in cleaned.pages
    ]
    _write_json_atomic(
        output_dir / f"{cleaned.document.document_id}.json",
        {
            "document_id": cleaned.document.document_id,
            "source_file": cleaned.document.file_path,
            "pages": pages,
        },
    )


def _enrich_chunks(
    chunks: tuple[TechnicalDocumentChunk, ...],
    cleaned: CleanedDocument,
) -> tuple[TechnicalDocumentChunk, ...]:
    page_actions = {
        page.page_number: page.cleaning_actions for page in cleaned.pages
    }
    classifications = [infer_chunk_type(chunk.content) for chunk in chunks]
    table_pages = {
        chunk.page_number
        for chunk, (chunk_type, _) in zip(chunks, classifications)
        if chunk_type == "table_or_pinout"
    }
    enriched: list[TechnicalDocumentChunk] = []
    for index, (chunk, (chunk_type, review_status)) in enumerate(
        zip(chunks, classifications)
    ):
        if chunk.page_number in table_pages and chunk_type != "table_or_pinout":
            chunk_type = "table_context"
            review_status = "needs_review"
        enriched.append(
            chunk.model_copy(
                update={
                    "previous_chunk_id": chunks[index - 1].chunk_id if index else None,
                    "next_chunk_id": (
                        chunks[index + 1].chunk_id
                        if index + 1 < len(chunks)
                        else None
                    ),
                    "content_hash": hashlib.sha256(chunk.content.encode()).hexdigest(),
                    "chunk_type": chunk_type,
                    "review_status": review_status,
                    "cleaning_actions": page_actions.get(chunk.page_number, ()),
                }
            )
        )
    return tuple(enriched)


def _remove_duplicate_chunks(
    chunks: tuple[TechnicalDocumentChunk, ...],
) -> tuple[tuple[TechnicalDocumentChunk, ...], int]:
    unique: list[TechnicalDocumentChunk] = []
    seen_hashes: set[str] = set()
    for chunk in chunks:
        content_hash = hashlib.sha256(chunk.content.encode()).hexdigest()
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)
        unique.append(chunk)
    return tuple(unique), len(chunks) - len(unique)


def build_document_chunks(
    documents_dir: str | Path,
    *,
    manifest_names: tuple[str, ...] = ("official_manifest.json",),
    output_path: str | Path = ".agent_data/document_chunks.json",
    cleaned_output_dir: str | Path = ".agent_data/cleaned_documents",
    max_chars: int = 800,
) -> dict[str, object]:
    root = Path(documents_dir)
    output = Path(output_path)
    cleaned_dir = Path(cleaned_output_dir)
    documents = tuple(
        document
        for manifest_name in manifest_names
        for document in load_documents(root, manifest_name=manifest_name)
    )

    all_chunks: list[TechnicalDocumentChunk] = []
    document_reports: list[dict[str, object]] = []
    total_removed_duplicates = 0
    for document in documents:
        cleaned = clean_document(document)
        _persist_cleaned_document(cleaned, cleaned_dir)
        raw_chunks = chunk_document(cleaned.as_loaded_document(), max_chars=max_chars)
        raw_chunks, removed_duplicates = _remove_duplicate_chunks(raw_chunks)
        total_removed_duplicates += removed_duplicates
        chunks = _enrich_chunks(raw_chunks, cleaned)
        all_chunks.extend(chunks)
        excluded_pages = sum(page.excluded_from_chunks for page in cleaned.pages)
        action_counts = Counter(
            action for page in cleaned.pages for action in page.cleaning_actions
        )
        document_reports.append(
            {
                "document_id": document.document_id,
                "title": document.title,
                "source_file": document.file_path,
                "source_file_sha256": sha256_file(Path(document.file_path)),
                "page_count": len(document.pages),
                "excluded_page_count": excluded_pages,
                "chunk_count": len(chunks),
                "needs_review_chunk_count": sum(
                    chunk.review_status == "needs_review" for chunk in chunks
                ),
                "removed_duplicate_chunk_count": removed_duplicates,
                "raw_character_count": sum(
                    len(page.raw_text) for page in cleaned.pages
                ),
                "cleaned_character_count": sum(
                    len(page.cleaned_text)
                    for page in cleaned.pages
                    if not page.excluded_from_chunks
                ),
                "cleaning_action_counts": dict(sorted(action_counts.items())),
            }
        )

    payload: dict[str, object] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "max_chars": max_chars,
        "summary": {
            "document_count": len(documents),
            "page_count": sum(len(document.pages) for document in documents),
            "excluded_page_count": sum(
                report["excluded_page_count"] for report in document_reports
            ),
            "chunk_count": len(all_chunks),
            "needs_review_chunk_count": sum(
                chunk.review_status == "needs_review" for chunk in all_chunks
            ),
            "removed_duplicate_chunk_count": total_removed_duplicates,
            "raw_character_count": sum(
                report["raw_character_count"] for report in document_reports
            ),
            "cleaned_character_count": sum(
                report["cleaned_character_count"] for report in document_reports
            ),
        },
        "documents": document_reports,
        "chunks": [chunk.model_dump(mode="json") for chunk in all_chunks],
    }
    _write_json_atomic(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗PDF并生成Chunk JSON")
    parser.add_argument("--documents-dir", default="documents")
    parser.add_argument("--output", default=".agent_data/document_chunks.json")
    parser.add_argument("--cleaned-dir", default=".agent_data/cleaned_documents")
    parser.add_argument("--max-chars", type=int, default=800)
    args = parser.parse_args()

    payload = build_document_chunks(
        args.documents_dir,
        output_path=args.output,
        cleaned_output_dir=args.cleaned_dir,
        max_chars=args.max_chars,
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Chunk JSON: {Path(args.output).resolve()}")
    print(f"清洗逐页JSON: {Path(args.cleaned_dir).resolve()}")


if __name__ == "__main__":
    main()
