"""Clause chunking utilities with paragraph-first strategy."""

from __future__ import annotations

import re
from typing import List, Sequence

import nltk

_HEADING_PATTERN = re.compile(
    r"^(?:((section|clause)\s+\d+(?:\.\d+)*)|(\d+(?:\.\d+){0,3}))([).\s-])",
    flags=re.IGNORECASE,
)


def _ensure_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:  # pragma: no cover - downloads once
        try:
            nltk.download("punkt", quiet=True)
        except Exception:  # pragma: no cover - offline fallback
            pass


def _split_into_sentences(text: str) -> List[str]:
    _ensure_punkt()
    try:
        return nltk.sent_tokenize(text)
    except LookupError:
        return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]


def _paragraphs_from_text(text: str) -> List[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    paragraphs: List[str] = []
    current: List[str] = []

    def flush_current() -> None:
        if current:
            paragraph = "\n".join(current).strip()
            if paragraph:
                paragraphs.append(paragraph)
            current.clear()

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            flush_current()
            continue
        if _HEADING_PATTERN.match(stripped) and current:
            flush_current()
            current.append(line)
            continue
        current.append(line)
    flush_current()

    if not paragraphs and text.strip():
        paragraphs.append(text.strip())
    return paragraphs


def _token_length(text: str, tokenizer) -> int:
    tokens = tokenizer(text, add_special_tokens=False, padding=False, truncation=False)
    input_ids = tokens.get("input_ids")
    if isinstance(input_ids, list):
        if input_ids and isinstance(input_ids[0], list):
            return len(input_ids[0])
        return len(input_ids)
    return int(tokens.get("length", 0))


def _split_long_paragraph(paragraph: str, tokenizer, max_tokens: int) -> Sequence[str]:
    sentences = _split_into_sentences(paragraph)
    if not sentences:
        return [paragraph]
    chunks: List[str] = []
    current: List[str] = []
    for sentence in sentences:
        projected = " ".join(current + [sentence]).strip()
        if projected and _token_length(projected, tokenizer) > max_tokens and current:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            continue
        current.append(sentence)
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def chunk_contract_into_clauses(
    text: str,
    tokenizer,
    max_tokens: int = 450,
    paragraph_overlap: int = 0,
) -> List[str]:
    """Split a contract into clauses, keeping one clause per paragraph.

    Paragraphs that exceed ``max_tokens`` are split using sentence-level
    chunks, but paragraphs are never merged together. The ``paragraph_overlap``
    parameter is currently ignored to preserve strict paragraph boundaries.
    """

    paragraphs = _paragraphs_from_text(text)
    if not paragraphs:
        return []

    clauses: List[str] = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        para_tokens = _token_length(paragraph, tokenizer)
        if para_tokens <= max_tokens:
            clauses.append(paragraph)
            continue
        parts = _split_long_paragraph(paragraph, tokenizer, max_tokens)
        if len(parts) == 1 and _token_length(parts[0], tokenizer) > max_tokens:
            raise ValueError("Tokenizer max_tokens too small for even a single sentence")
        clauses.extend(part.strip() for part in parts if part.strip())

    return clauses
