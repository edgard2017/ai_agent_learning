"""把长文档切成适合检索的小段。"""

from __future__ import annotations

import re

from .models import LoadedDocument, TechnicalDocumentChunk


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？.!?；;])")


def _split_oversized_text(text: str, max_chars: int) -> list[str]:
    """优先按句子拆分；单句仍太长时才按字符硬切。"""

    sentences = [item.strip() for item in SENTENCE_BOUNDARY.split(text) if item.strip()]
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(
                sentence[start : start + max_chars]
                for start in range(0, len(sentence), max_chars)
            )
        elif not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current = f"{current} {sentence}"
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces


def _parse_markdown_sections(
    document: LoadedDocument,
    content: str | None = None,
) -> list[tuple[str, list[str]]]:
    """将 Markdown 按标题和空行解析为 section + paragraphs。"""

    sections: list[tuple[str, list[str]]] = []
    section_name = document.title
    paragraphs: list[str] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_lines:
            paragraphs.append(" ".join(paragraph_lines).strip())
            paragraph_lines.clear()

    def flush_section() -> None:
        flush_paragraph()
        if paragraphs:
            sections.append((section_name, paragraphs.copy()))
            paragraphs.clear()

    for raw_line in (content if content is not None else document.content).splitlines():
        line = raw_line.strip()
        heading = HEADING_PATTERN.match(line)
        if heading:
            flush_section()
            section_name = heading.group(2)
        elif not line:
            flush_paragraph()
        else:
            paragraph_lines.append(line)
    flush_section()
    return sections


def chunk_document(
    document: LoadedDocument,
    *,
    max_chars: int = 800,
) -> tuple[TechnicalDocumentChunk, ...]:
    """按章节和段落切块，保证每块不超过 max_chars。

    第一版使用字符数而非 Token 数，目的是不绑定某个 tokenizer；之后可以
    针对实际生成模型替换为 Token 计数。
    """

    if max_chars < 50:
        raise ValueError("max_chars 不能小于 50")

    pending: list[tuple[str, str, int | None]] = []
    is_pdf = document.file_path.lower().endswith(".pdf")
    page_inputs = (
        tuple((f"第 {index} 页", page, index) for index, page in enumerate(document.pages, 1))
        if is_pdf
        else ((None, document.content, None),)
    )
    for page_section, page_content, page_number in page_inputs:
        parsed_sections = (
            [(page_section or document.title, [page_content])]
            if is_pdf
            else _parse_markdown_sections(document, page_content)
        )
        for section, paragraphs in parsed_sections:
            current = ""
            for paragraph in paragraphs:
                for piece in _split_oversized_text(paragraph, max_chars):
                    if not current:
                        current = piece
                    elif len(current) + 2 + len(piece) <= max_chars:
                        current = f"{current}\n\n{piece}"
                    else:
                        pending.append((section, current, page_number))
                        current = piece
            if current:
                pending.append((section, current, page_number))

    return tuple(
        TechnicalDocumentChunk(
            chunk_id=f"{document.document_id}-{index:03d}",
            product_id=document.product_id,
            title=document.title,
            section=section,
            content=content,
            keywords=document.keywords,
            source=document.source,
            page_number=page_number,
        )
        for index, (section, content, page_number) in enumerate(pending, start=1)
    )


def chunk_documents(
    documents: tuple[LoadedDocument, ...],
    *,
    max_chars: int = 800,
) -> tuple[TechnicalDocumentChunk, ...]:
    """批量切块，返回现有检索系统认识的 TechnicalDocumentChunk。"""

    return tuple(
        chunk
        for document in documents
        for chunk in chunk_document(document, max_chars=max_chars)
    )
