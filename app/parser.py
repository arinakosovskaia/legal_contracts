from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pdfplumber

from .models import Paragraph
from .headings import is_heading, parse_heading

_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\(?\d{1,3}[\).\]]|[A-Za-z][\).\]])\s+")
_SENTENCE_END_RE = re.compile(r"[.!?]$")
_BULLET_LETTER_O_RE = re.compile(r"^\s*o\s+", re.IGNORECASE)


def _should_merge_continuation(prev_para: str, next_para: str) -> bool:
    prev = (prev_para or "").strip()
    nxt = (next_para or "").strip()
    if not prev or not nxt:
        return False
    # Never merge headings.
    if is_heading(prev) or is_heading(nxt):
        return False
    # If previous is a list block, do not merge forward.
    if "\n" in prev:
        return False
    # If next looks like a bullet/list item, don't merge (except ":" intro handled separately).
    if _BULLET_RE.match(nxt) or _BULLET_LETTER_O_RE.match(nxt):
        return False
    # If prev clearly ends a sentence, don't merge.
    if _SENTENCE_END_RE.search(prev):
        return False
    # Merge if next starts like a continuation (lowercase or punctuation).
    if re.match(r"^[a-z,;:\)\]]", nxt):
        return True
    # Also merge if prev is very short (common PDF artefact: broken line treated as paragraph).
    if len(prev) <= 80:
        return True
    return False


def _merge_continuation_paragraphs(paras: list[str]) -> list[str]:
    """
    Best-effort post-processing merge:
    - If a paragraph doesn't end a sentence and the next looks like continuation, merge them.
    - If a paragraph ends with ':' and the next is a list block, merge with newline.
    - If a heading is broken across 2–3 lines, merge the heading lines.
    """
    def is_heading_like(s: str) -> bool:
        return parse_heading(s, max_len=160) is not None

    def is_short_heading_fragment(s: str) -> bool:
        t = (s or "").strip()
        if not t:
            return False
        # short lines are most often broken headings
        if len(t) > 60:
            return False
        # Avoid merging if it already looks like a full sentence.
        if _SENTENCE_END_RE.search(t):
            return False
        return True

    def ends_with_connector(s: str) -> bool:
        t = (s or "").strip().upper()
        return bool(re.search(r"\b(OF|AND|OR|TO|FOR|IN|ON|WITH|WITHOUT|UNDER)\s*$", t))

    out: list[str] = []
    i = 0
    while i < len(paras):
        cur = paras[i]
        if i + 1 < len(paras):
            nxt = paras[i + 1]
            # Broken heading merge:
            # e.g. "3. LIMITATION OF" + "LIABILITY"  -> "3. LIMITATION OF LIABILITY"
            if is_heading_like(cur) and is_heading_like(nxt) and is_short_heading_fragment(cur):
                merged = (cur.rstrip() + " " + nxt.lstrip()).strip()
                # Optionally merge a 3rd line if it's also heading-like and we still look like a fragment.
                if i + 2 < len(paras):
                    nxt2 = paras[i + 2]
                    if (ends_with_connector(merged) or is_short_heading_fragment(merged)) and is_heading_like(nxt2):
                        merged2 = (merged.rstrip() + " " + nxt2.lstrip()).strip()
                        if is_heading_like(merged2):
                            out.append(merged2)
                            i += 3
                            continue
                if is_heading_like(merged):
                    out.append(merged)
                    i += 2
                    continue

            # Intro line that leads into a list block (we keep list newlines).
            if cur.strip().endswith(":") and "\n" in nxt:
                out.append(cur.rstrip() + "\n" + nxt.lstrip())
                i += 2
                continue
            if _should_merge_continuation(cur, nxt):
                out.append((cur.rstrip() + " " + nxt.lstrip()).strip())
                i += 2
                continue
        out.append(cur)
        i += 1
    return out


def _is_heading_like(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return is_heading(t, max_len=160)


def _normalize_text(text: str) -> str:
    # Join hyphenated line breaks: "termina-\n tion" -> "termination"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def _split_into_paragraphs(page_text: str) -> List[str]:
    """
    Heuristic paragraph splitting for PDFs.

    Goals:
    - Treat blank lines as hard paragraph boundaries.
    - Join soft line wraps inside a paragraph (single '\n' -> space).
    - Preserve list structure: consecutive bullet/numbered items become ONE paragraph
      with items separated by '\n' (so we don't lose list markers).
    - Treat headings (SECTION/ARTICLE/1.2 etc.) as standalone paragraphs.
    """
    page_text = _normalize_text(page_text)
    lines = page_text.split("\n")

    paragraphs: List[str] = []
    para_lines: List[str] = []
    list_items: List[str] = []
    current_item_lines: List[str] = []

    def flush_para() -> None:
        nonlocal para_lines
        text = " ".join(para_lines).strip()
        if text:
            paragraphs.append(text)
        para_lines = []

    def flush_list() -> None:
        nonlocal list_items, current_item_lines
        if current_item_lines:
            list_items.append(" ".join(current_item_lines).strip())
            current_item_lines = []
        if list_items:
            paragraphs.append("\n".join(list_items).strip())
        list_items = []

    for raw in lines:
        if raw is None:
            continue
        s = raw.strip()
        if not s:
            # blank line = paragraph boundary
            flush_para()
            flush_list()
            continue

        # headings: flush current context and keep heading as its own paragraph
        if _is_heading_like(s):
            flush_para()
            flush_list()
            paragraphs.append(s)
            continue

        # bullet / numbered list item
        if _BULLET_RE.match(raw):
            flush_para()
            if current_item_lines:
                list_items.append(" ".join(current_item_lines).strip())
                current_item_lines = []
            current_item_lines = [s]
            continue

        # continuation of a list item (if we are inside a list)
        if current_item_lines:
            current_item_lines.append(s)
            continue

        # normal paragraph line
        para_lines.append(s)

    flush_para()
    flush_list()
    paragraphs = [p for p in paragraphs if p and p.strip()]
    return _merge_continuation_paragraphs(paragraphs)


def _should_merge_across_pages(prev_para: str, next_para: str) -> bool:
    """
    Detect page-break continuation: last paragraph on a page continues on the next page.
    This is a best-effort heuristic (demo).
    """
    prev = (prev_para or "").strip()
    nxt = (next_para or "").strip()
    if not prev or not nxt:
        return False
    # Don't merge list blocks or headings.
    if "\n" in prev or "\n" in nxt:
        return False
    # Don't merge headings (use shared heading heuristics).
    if is_heading(prev, max_len=160) or is_heading(nxt, max_len=160):
        return False
    if _BULLET_RE.match(nxt):
        return False
    # If prev clearly ends a sentence, don't merge.
    if _SENTENCE_END_RE.search(prev):
        return False
    # If next starts like a continuation (lowercase or punctuation), likely merge.
    if re.match(r"^[a-z,;:\)\]]", nxt):
        return True
    # Otherwise be conservative.
    return False


def _extract_layout_lines(page) -> list[tuple[float, float, str]]:
    """
    Extract text lines with rough layout using word coordinates.
    Returns list of (y_top, x_left, text).
    """
    try:
        words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
    except Exception:
        words = []
    if not words:
        return []

    buckets: dict[float, list[dict]] = {}
    for w in words:
        top = float(w.get("top", 0.0))
        # bucket by ~2px to stabilize slight jitter
        y = round(top / 2.0) * 2.0
        buckets.setdefault(y, []).append(w)

    lines: list[tuple[float, float, str]] = []
    for y in sorted(buckets.keys()):
        ws = sorted(buckets[y], key=lambda d: float(d.get("x0", 0.0)))
        text = " ".join(str(d.get("text", "")).strip() for d in ws).strip()
        if not text:
            continue
        x_left = min(float(d.get("x0", 0.0)) for d in ws)
        lines.append((y, x_left, text))
    return lines


def _lines_to_paragraphs(lines: list[tuple[float, float, str]]) -> list[str]:
    """
    Convert (y, x, text) lines into paragraph blocks.

    Heuristics:
    - paragraph break on large vertical gap
    - list block when bullet markers detected; preserve bullets with '\n'
    - merge wrapped lines within paragraph
    """
    paras: list[str] = []
    cur_lines: list[str] = []
    prev_y: float | None = None
    prev_x: float | None = None

    list_items: list[str] = []
    list_indent: float | None = None

    def flush_para() -> None:
        nonlocal cur_lines
        text = " ".join(cur_lines).strip()
        if text:
            paras.append(text)
        cur_lines = []

    def flush_list() -> None:
        nonlocal list_items, list_indent
        if list_items:
            paras.append("\n".join(list_items).strip())
        list_items = []
        list_indent = None

    for (y, x, text) in lines:
        s = (text or "").strip()
        if not s:
            continue

        is_heading = _is_heading_like(s)
        is_bullet = bool(_BULLET_RE.match(s) or _BULLET_LETTER_O_RE.match(s))

        if is_heading:
            flush_para()
            flush_list()
            paras.append(s)
            prev_y, prev_x = y, x
            continue

        # list handling
        if is_bullet:
            flush_para()
            if list_indent is None:
                list_indent = x
            list_items.append(s)
            prev_y, prev_x = y, x
            continue

        if list_items:
            # continuation of last bullet item if indented or previous line didn't end a sentence
            indent_thresh = (list_indent or 0.0) + 10.0
            if x >= indent_thresh or (list_items and not _SENTENCE_END_RE.search(list_items[-1].strip())):
                list_items[-1] = (list_items[-1].rstrip() + " " + s.lstrip()).strip()
                prev_y, prev_x = y, x
                continue
            else:
                flush_list()

        # normal paragraph handling
        if prev_y is None:
            cur_lines.append(s)
            prev_y, prev_x = y, x
            continue

        gap = y - prev_y
        # tuned for typical PDF line spacing
        if gap >= 12.0:
            flush_para()
            cur_lines.append(s)
        else:
            # indent-based break (e.g. new paragraph starts flush-left after an indented wrap)
            if prev_x is not None and abs(x - prev_x) >= 18.0 and cur_lines and _SENTENCE_END_RE.search(cur_lines[-1].strip()):
                flush_para()
                cur_lines.append(s)
            else:
                cur_lines.append(s)

        prev_y, prev_x = y, x

    flush_para()
    flush_list()
    paras = [p for p in paras if p and p.strip()]
    return _merge_continuation_paragraphs(paras)


def parse_pdf_to_paragraphs(
    pdf_path: Path,
    *,
    max_pages: int,
    max_paragraphs: int,
) -> tuple[list[Paragraph], int]:
    paragraphs: List[Paragraph] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        page_count = len(pdf.pages)
        if page_count > max_pages:
            raise ValueError(f"PDF has {page_count} pages, exceeds MAX_PAGES={max_pages}.")
        para_idx = 0
        for i, page in enumerate(pdf.pages, start=1):
            # Prefer layout-aware extraction to avoid "everything becomes one paragraph".
            lines = _extract_layout_lines(page)
            if lines:
                page_paras = _lines_to_paragraphs(lines)
            else:
                text = page.extract_text() or ""
                page_paras = _split_into_paragraphs(text)
            # Merge page-break continuations (optional but enabled by default for better UX).
            if paragraphs and page_paras:
                if _should_merge_across_pages(paragraphs[-1].text, page_paras[0]):
                    paragraphs[-1].text = (paragraphs[-1].text.rstrip() + " " + page_paras[0].lstrip()).strip()
                    page_paras = page_paras[1:]
            for para in page_paras:
                paragraphs.append(Paragraph(text=para, page=i, paragraph_index=para_idx))
                para_idx += 1
                if max_paragraphs > 0 and len(paragraphs) >= max_paragraphs:
                    raise ValueError(
                        f"PDF produced > {max_paragraphs} paragraphs, exceeds MAX_PARAGRAPHS={max_paragraphs}."
                    )
    return paragraphs, page_count

