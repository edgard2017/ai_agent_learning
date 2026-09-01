"""对PDF提取文字做可重复、不会改写技术事实的规则清洗。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
import unicodedata

from .models import LoadedDocument


STANDALONE_PAGE_NUMBER = re.compile(r"^(?:page\s+)?\d+(?:\s+of\s+\d+)?$", re.I)
DOT_LEADER = re.compile(r"(?:\.\s*){8,}\s*\d*\s*$")
INLINE_DOT_RUN = re.compile(r"(?:\.\s*){2,}")
TRAILING_PAGE_NUMBER = re.compile(r"\s+\d+\s*$")
NUMBERED_HEADING = re.compile(
    r"^(?P<number>\d{1,2}(?:\.\d{1,2})*)\s+(?P<title>[A-Za-z][^.!?]{2,100})$"
)
HYPHENATED_LINE_BREAK = re.compile(r"([A-Za-z]{2,})-\s*\n\s*([a-z]{2,})")
PROTECTED_LIST_SPACE = "\ue001"


@dataclass(frozen=True)
class CleanedPage:
    page_number: int
    raw_text: str
    cleaned_text: str
    cleaning_actions: tuple[str, ...]
    review_status: str
    excluded_from_chunks: bool


@dataclass(frozen=True)
class CleanedDocument:
    document: LoadedDocument
    pages: tuple[CleanedPage, ...]

    def as_loaded_document(self) -> LoadedDocument:
        page_texts = tuple(
            "" if page.excluded_from_chunks else page.cleaned_text for page in self.pages
        )
        return self.document.model_copy(
            update={
                "pages": page_texts,
                "content": "\n\n".join(text for text in page_texts if text),
            }
        )


def _line_signature(line: str) -> str:
    normalized = unicodedata.normalize("NFC", line)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    normalized = re.sub(r"\s+\d+\s*$", " #", normalized)
    return normalized


def find_repeated_margin_lines(pages: tuple[str, ...]) -> frozenset[str]:
    """寻找重复出现在页面顶部/底部的页眉页脚。"""

    counts: Counter[str] = Counter()
    for page in pages:
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        candidates = set(lines[:4] + lines[-4:])
        counts.update(
            signature
            for line in candidates
            if 3 <= len(signature := _line_signature(line)) <= 120
        )

    threshold = max(2, math.ceil(len(pages) * 0.18))
    return frozenset(
        signature for signature, count in counts.items() if count >= threshold
    )


def is_table_of_contents(text: str) -> bool:
    lower = text.lower()
    if "table of contents" in lower:
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    dot_leaders = sum(bool(DOT_LEADER.search(line)) for line in lines)
    trailing_numbers = sum(
        bool(TRAILING_PAGE_NUMBER.search(line)) and len(line) > 12 for line in lines
    )
    return dot_leaders >= 3 or (dot_leaders >= 1 and trailing_numbers >= 6)


SIMPLE_HEADING_ENDINGS = (
    "activation",
    "batteries",
    "calibration",
    "capsules",
    "cleaning",
    "clock",
    "connector",
    "connectors",
    "hardware",
    "instrument",
    "interface",
    "maintenance",
    "module",
    "o-ring",
    "o-rings",
    "parameters",
    "pinout",
    "power",
    "repair",
    "selection",
    "specifications",
)


def looks_like_pdf_heading(line: str) -> bool:
    numbered = NUMBERED_HEADING.fullmatch(line)
    if numbered:
        number = numbered.group("number")
        title = numbered.group("title")
        if any(symbol in title for symbol in ("/", "+", "=")):
            return False
        if "." in number:
            return True
        single_word_chapters = {"hardware", "maintenance", "repair"}
        return len(title.split()) >= 2 or title.lower() in single_word_chapters
    letters = [character for character in line if character.isalpha()]
    if (
        3 <= len(line) <= 80
        and len(letters) >= 4
        and all(character.isupper() for character in letters)
    ):
        return True
    words = line.split()
    if line.lower().count("o-ring") > 1:
        return False
    return (
        1 <= len(words) <= 8
        and len(line) <= 60
        and line[0].isupper()
        and not line.endswith((".", ",", ";", ":", "!", "?"))
        and line.lower().endswith(SIMPLE_HEADING_ENDINGS)
    )


def clean_page_text(
    raw_text: str,
    *,
    repeated_margin_lines: frozenset[str] = frozenset(),
) -> tuple[str, tuple[str, ...]]:
    """清理一页文字，返回清洗结果和实际执行的动作。"""

    actions: list[str] = []
    text = unicodedata.normalize("NFC", raw_text).replace("\u00ad", "")
    text, private_use_removed = re.subn(r"[\ue000-\uf8ff]", "", text)
    if private_use_removed:
        actions.append("removed_private_use_glyphs")

    dehyphenated, replacements = HYPHENATED_LINE_BREAK.subn(r"\1\2", text)
    if replacements:
        actions.append("joined_hyphenated_words")
    text = dehyphenated

    kept_lines: list[str] = []
    removed_margin = False
    removed_page_number = False
    removed_dot_leader = False
    for raw_line in text.splitlines():
        line = re.sub(r"[^\S\n]+", " ", raw_line).strip()
        if not line:
            if kept_lines and kept_lines[-1] != "":
                kept_lines.append("")
            continue
        if _line_signature(line) in repeated_margin_lines:
            removed_margin = True
            continue
        if STANDALONE_PAGE_NUMBER.fullmatch(line):
            removed_page_number = True
            continue
        if line in {".", "•", "-", "–"}:
            actions.append("removed_empty_marker")
            continue
        if DOT_LEADER.search(line):
            line = DOT_LEADER.sub("", line).strip()
            removed_dot_leader = True
            if not line:
                continue
        line = re.sub(r"(?<=\w)\.\s+(?=(?:com|org|net)\b)", ".", line, flags=re.I)
        kept_lines.append(line)

    if removed_margin:
        actions.append("removed_repeated_margin")
    if removed_page_number:
        actions.append("removed_standalone_page_number")
    if removed_dot_leader:
        actions.append("removed_dot_leaders")

    paragraphs: list[str] = []
    current: list[str] = []
    for line in kept_lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if looks_like_pdf_heading(line):
            if current:
                paragraphs.append(" ".join(current))
                current = []
            paragraphs.append(line)
        elif line.startswith(("•", "- ", "– ")) or re.match(r"^\d+[.)]\s", line):
            if current:
                paragraphs.append(" ".join(current))
                current = []
            paragraphs.append(line)
        else:
            current.append(line)
    if current:
        paragraphs.append(" ".join(current))

    cleaned = "\n\n".join(item.strip() for item in paragraphs if item.strip())
    cleaned = re.sub(
        r"(?:\b\d+\.\s+){2,}\d+\.",
        lambda match: re.sub(
            r"\.\s+", f".{PROTECTED_LIST_SPACE}", match.group()
        ),
        cleaned,
    )
    cleaned, decimal_fixes = re.subn(r"(?<=\d)\.\s+(?=\d)", ".", cleaned)
    cleaned, hyphen_space_fixes = re.subn(
        r"\b([A-Za-z]{1,20})-\s+([a-z]{2,})\b", r"\1-\2", cleaned
    )
    cleaned, domain_fixes = re.subn(
        r"(?<=\w)\.\s+(?=(?:com|org|net)\b)", ".", cleaned, flags=re.I
    )
    cleaned, inline_dot_fixes = INLINE_DOT_RUN.subn(" ", cleaned)
    cleaned = cleaned.replace(PROTECTED_LIST_SPACE, " ")
    if decimal_fixes:
        actions.append("rejoined_decimal_numbers")
    if hyphen_space_fixes:
        actions.append("rejoined_spaced_hyphenated_words")
    if domain_fixes:
        actions.append("rejoined_domain_names")
    if inline_dot_fixes:
        actions.append("removed_inline_dot_runs")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if cleaned != raw_text.strip():
        actions.append("normalized_whitespace")
    return cleaned, tuple(dict.fromkeys(actions))


def clean_document(document: LoadedDocument) -> CleanedDocument:
    """清洗完整文档；Markdown/TXT不做PDF页眉页脚检测。"""

    is_pdf = document.file_path.lower().endswith(".pdf")
    repeated = find_repeated_margin_lines(document.pages) if is_pdf else frozenset()
    cleaned_pages: list[CleanedPage] = []
    for page_number, raw_text in enumerate(document.pages, start=1):
        cleaned_text, actions = clean_page_text(
            raw_text, repeated_margin_lines=repeated
        )
        excluded = is_pdf and is_table_of_contents(raw_text)
        review_status = "excluded_toc" if excluded else "auto_cleaned"
        if excluded:
            actions = (*actions, "excluded_table_of_contents")
        if not cleaned_text:
            excluded = True
            review_status = "excluded_empty"
            actions = (*actions, "excluded_empty_page")
        cleaned_pages.append(
            CleanedPage(
                page_number=page_number,
                raw_text=raw_text,
                cleaned_text=cleaned_text,
                cleaning_actions=tuple(dict.fromkeys(actions)),
                review_status=review_status,
                excluded_from_chunks=excluded,
            )
        )
    return CleanedDocument(document=document, pages=tuple(cleaned_pages))


def infer_chunk_type(content: str) -> tuple[str, str]:
    """保守标记高风险结构；不重写表格、参数或接线事实。"""

    lower = content.lower()
    if len(re.findall(r"\b\d+\.\s+(?=\d+\.)", content)) >= 2:
        return "procedure_extraction_issue", "needs_review"
    if (
        "pin no" in lower
        or "pinout" in lower
        or ("rs-232" in lower and "rs-485" in lower and "ground" in lower)
    ):
        return "table_or_pinout", "needs_review"
    numbered_steps = len(re.findall(r"(?:^|\n)\s*\d+[.)]\s", content))
    bullets = len(re.findall(r"(?:^|\n)\s*[•-]\s", content))
    if numbered_steps + bullets >= 2:
        return "procedure_or_list", "auto_cleaned"
    return "text", "auto_cleaned"
