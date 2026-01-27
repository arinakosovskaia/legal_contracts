from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import pdfplumber

from .models import Paragraph
from .headings import is_heading, parse_heading

_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\(?\d{1,3}[\).\]]|[A-Za-z][\).\]])\s+")
_SENTENCE_END_RE = re.compile(r"[.!?]$")
_BULLET_LETTER_O_RE = re.compile(r"^\s*o\s+", re.IGNORECASE)
# Standalone "- N -" or "Page - N -" lines (PDF artefacts; skip or strip)
_PAGE_NUMBER_LINE = re.compile(r"^\s*(?:Page\s*)?-\s*\d+\s*-\s*$", re.IGNORECASE)
_PAGE_NUMBER_FRAGMENT = re.compile(r"^Page\s*-\s*\d+\s*-\s*$", re.IGNORECASE)


@dataclass
class LineInfo:
    top: float
    x0: float
    x1: float
    bottom: float
    text: str


@dataclass
class ParaInfo:
    text: str
    bbox: Optional[tuple[float, float, float, float]]


def _bullet_kind(line: str) -> str:
    """
    Coarse bullet style detector for page-break merging.
    """
    s = (line or "").lstrip()
    if re.match(r"^(?:[-*•])\s+", s):
        return "symbol"
    if re.match(r"^\(?\d{1,3}[\).\]]\s+", s):
        return "number"
    if re.match(r"^[A-Za-z][\).\]]\s+", s):
        return "letter"
    return ""


def _page_break_merge_separator(prev_para: str, next_para: str) -> str:
    """
    Returns the separator to merge with (" " or "\\n"), or "" if not merging.
    """
    prev = (prev_para or "").strip()
    nxt = (next_para or "").strip()
    if not prev or not nxt:
        return ""
    # Don't merge across page-number artefacts ("Page -N-").
    if re.match(r"^\s*Page\s*-\s*\d+\s*-\s*", nxt, re.IGNORECASE):
        return ""
    # Don't merge headings (use shared heading heuristics).
    if is_heading(prev, max_len=160) or is_heading(nxt, max_len=160):
        return ""
    # Don't merge list blocks that START on the next page (keep them separate).
    if "\n" in nxt:
        return ""

    # If prev is a list block, allow continuing the list across pages:
    # - either continuation text ("based on" -> "claims ...") => join with space
    # - or a new bullet item (A-D then E) => join with newline (if bullet style matches)
    if "\n" in prev:
        if _BULLET_RE.match(nxt):
            prev_first = (prev.splitlines()[0] or "").strip()
            k1 = _bullet_kind(prev_first)
            k2 = _bullet_kind(nxt)
            if k1 and k2 and k1 == k2:
                return "\n"
            return ""
        last_line = (prev.splitlines()[-1] or "").strip()
        if not last_line:
            return ""
        if _SENTENCE_END_RE.search(last_line):
            return ""
        if re.match(r"^[a-z,;:\)\]]", nxt):
            return " "
        return ""

    # Normal paragraph merge heuristics.
    if _BULLET_RE.match(nxt):
        return ""
    if _SENTENCE_END_RE.search(prev):
        return ""
    if re.match(r"^[a-z,;:\)\]]", nxt):
        return " "
    return ""


def _should_merge_continuation(prev_para: str, next_para: str) -> bool:
    prev = (prev_para or "").strip()
    nxt = (next_para or "").strip()
    if not prev or not nxt:
        return False
    # Special case: heading ending with connector word (In, To, For, etc.) + continuation
    # e.g. "1.6 Distributor's Terms and Minimum Expectations. In" + "order to maintain..."
    if is_heading(prev):
        t_upper = prev.upper()
        if re.search(r"[.\s]\b(OF|AND|OR|TO|FOR|IN|ON|WITH|WITHOUT|UNDER|FROM|BY|AT|OVER|THROUGH|DURING|BEFORE|AFTER)\s*$", t_upper):
            if re.match(r"^[a-z]", nxt):
                return True
    # Never merge headings (unless handled above).
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


def _merge_continuation_paragraphs(paras: list[ParaInfo] | list[str]) -> list[ParaInfo]:
    """
    Best-effort post-processing merge:
    - If a paragraph doesn't end a sentence and the next looks like continuation, merge them.
    - If a paragraph ends with ':' and the next is a list block, merge with newline.
    - If a heading is broken across 2–3 lines, merge the heading lines.
    """
    def is_heading_like(s: str) -> bool:
        return parse_heading(s, max_len=160) is not None

    def _merge_box(a: Optional[tuple[float, float, float, float]], b: Optional[tuple[float, float, float, float]]):
        if a is None or b is None:
            return None
        return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

    def split_inline_heading_body(p: ParaInfo) -> list[ParaInfo]:
        """
        Split cases where a numbered/section heading and its first sentence are on the same line, e.g.:
        - "6.1 Assignment. No assignment of this Agreement ..."
        into:
        - "6.1 Assignment"
        - "No assignment of this Agreement ..."
        Also split "5. TITLE 5.1 SUBTITLE. BODY" (section + subsection on one line) into separate paragraphs
        so "5. TITLE" and "5.1 SUBTITLE." become detectable headings and we avoid multiple sections in one chunk.
        """
        t = (p.text or "").strip()
        if not t:
            return [p]
        # "5. REPRESENTATIONS AND WARRANTIES 5.1 Representations of Company. (A) ..." -> "5. TITLE" + "5.1 SUBTITLE. BODY"
        midline = re.compile(
            r"^(\d+\. [A-Z][^.]+?) (\d+\.\d+(?:\.\d+)* [A-Z][^.]+\.) (.+)$"
        )
        m = midline.match(t)
        if m:
            top, sub, body = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            if parse_heading(top, max_len=160) is not None and len(body.split()) >= 3:
                return [
                    ParaInfo(text=top, bbox=p.bbox),
                    ParaInfo(text=sub + " " + body, bbox=p.bbox),
                ]
        patterns = [
            re.compile(r"^(?P<num>\d+(?:\.\d+){0,6})\s+(?P<title>[^.]{1,80})\.\s+(?P<body>.+)$"),
            re.compile(
                r"^(?P<prefix>(?:SECTION|Section|CLAUSE|Clause)\s+\d+(?:\.\d+){0,6})\s+(?P<title>[^.]{1,80})\.\s+(?P<body>.+)$"
            ),
            re.compile(r"^(?P<prefix>§\s*\d+(?:\.\d+){0,6})\s+(?P<title>[^.]{1,80})\.\s+(?P<body>.+)$"),
        ]
        for pat in patterns:
            m = pat.match(t)
            if not m:
                continue
            num = (m.group("num") if "num" in m.groupdict() else "").strip()
            prefix = (m.group("prefix") if "prefix" in m.groupdict() else "").strip()
            title = (m.group("title") or "").strip()
            body = (m.group("body") or "").strip()
            heading = f"{prefix} {title}".strip() if prefix else f"{num} {title}".strip()
            # Only split if the heading-part is actually heading-like.
            if not heading or not body:
                continue
            if parse_heading(heading, max_len=160) is None:
                continue
            # Avoid splitting if the "body" is too short (likely false positive).
            if len(body.split()) < 4:
                continue
            return [
                ParaInfo(text=heading, bbox=p.bbox),
                ParaInfo(text=body, bbox=p.bbox),
            ]
        return [p]

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
        """
        Check if string ends with a connector word (preposition/conjunction).
        Handles cases like "1.6 Title. In" where "In" is a connector.
        """
        t = (s or "").strip().upper()
        # Match connector words at the end, optionally after a period
        return bool(re.search(r"[.\s]\b(OF|AND|OR|TO|FOR|IN|ON|WITH|WITHOUT|UNDER|FROM|BY|AT|OVER|THROUGH|DURING|BEFORE|AFTER)\s*$", t))

    # Normalize input to ParaInfo in case callers pass raw strings.
    norm: list[ParaInfo] = []
    for p in paras:
        if isinstance(p, ParaInfo):
            norm.append(p)
        else:
            norm.append(ParaInfo(text=str(p), bbox=None))

    out: list[ParaInfo] = []
    # First, split "heading + body" lines into two paragraphs to keep body text.
    expanded: list[ParaInfo] = []
    for p in norm:
        expanded.extend(split_inline_heading_body(p))
    paras = expanded

    i = 0
    while i < len(paras):
        cur = paras[i]
        if i + 1 < len(paras):
            nxt = paras[i + 1]
            # Broken heading merge:
            # e.g. "3. LIMITATION OF" + "LIABILITY"  -> "3. LIMITATION OF LIABILITY"
            if is_heading_like(cur.text) and is_heading_like(nxt.text) and is_short_heading_fragment(cur.text):
                # Do NOT merge if the next line looks like a new numbered/Section heading (e.g. "6.1 Assignment").
                if re.match(r"^\s*(?:\d+(?:\.\d+)*|SECTION\s+\d|CLAUSE\s+\d|ARTICLE\s+|§\s*\d)\b", nxt.text, re.IGNORECASE):
                    pass
                else:
                    merged = (cur.text.rstrip() + " " + nxt.text.lstrip()).strip()
                    # Optionally merge a 3rd line if it's also heading-like and we still look like a fragment.
                    if i + 2 < len(paras):
                        nxt2 = paras[i + 2]
                        if (ends_with_connector(merged) or is_short_heading_fragment(merged)) and is_heading_like(nxt2.text):
                            merged2 = (merged.rstrip() + " " + nxt2.text.lstrip()).strip()
                            if is_heading_like(merged2):
                                out.append(ParaInfo(text=merged2, bbox=_merge_box(cur.bbox, _merge_box(nxt.bbox, nxt2.bbox))))
                                i += 3
                                continue
                    if is_heading_like(merged):
                        out.append(ParaInfo(text=merged, bbox=_merge_box(cur.bbox, nxt.bbox)))
                        i += 2
                        continue

            # Heading ending with connector word (In, To, For, etc.) + continuation starting with lowercase
            # e.g. "1.6 Distributor's Terms and Minimum Expectations. In" + "order to maintain..."
            if is_heading_like(cur.text) and ends_with_connector(cur.text):
                if re.match(r"^[a-z]", nxt.text.strip()):
                    merged = (cur.text.rstrip() + " " + nxt.text.lstrip()).strip()
                    merged_bbox = _merge_box(cur.bbox, nxt.bbox)
                    # After merging heading with continuation, check if we can merge further
                    # (e.g., if continuation ends with "from the" and next is "Company:")
                    if i + 2 < len(paras):
                        nxt2 = paras[i + 2]
                        nxt2_stripped = nxt2.text.strip()
                        # Check if merged text ends with preposition + "the" and next is single word with colon
                        if (re.search(r"\b(from|to|of|in|on|at|for|with|by|under|over|through|during|before|after)\s+the\s*$", merged, re.IGNORECASE) and
                            re.match(r"^[A-Z][a-z]+:\s*$", nxt2_stripped)):
                            merged = (merged.rstrip() + " " + nxt2.text.lstrip()).strip()
                            merged_bbox = _merge_box(merged_bbox, nxt2.bbox)
                            out.append(ParaInfo(text=merged, bbox=merged_bbox))
                            i += 3
                            continue
                    out.append(ParaInfo(text=merged, bbox=merged_bbox))
                    i += 2
                    continue
            
            # Sentence ending with preposition + single word with colon (e.g. "from the" + "Company:")
            # e.g. "...from the" + "Company:" -> "...from the Company:"
            cur_stripped = cur.text.strip()
            nxt_stripped = nxt.text.strip()
            if (cur_stripped and nxt_stripped and 
                re.search(r"\b(from|to|of|in|on|at|for|with|by|under|over|through|during|before|after)\s+the\s*$", cur_stripped, re.IGNORECASE) and
                re.match(r"^[A-Z][a-z]+:\s*$", nxt_stripped)):
                merged = (cur.text.rstrip() + " " + nxt.text.lstrip()).strip()
                out.append(ParaInfo(text=merged, bbox=_merge_box(cur.bbox, nxt.bbox)))
                i += 2
                continue
            
            # Intro line that leads into a list block (we keep list newlines).
            if cur.text.strip().endswith(":") and "\n" in nxt.text:
                out.append(ParaInfo(text=cur.text.rstrip() + "\n" + nxt.text.lstrip(), bbox=_merge_box(cur.bbox, nxt.bbox)))
                i += 2
                continue
            if _should_merge_continuation(cur.text, nxt.text):
                out.append(ParaInfo(text=(cur.text.rstrip() + " " + nxt.text.lstrip()).strip(), bbox=_merge_box(cur.bbox, nxt.bbox)))
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


def _strip_page_fragment(text: str) -> str:
    """Remove 'Page -N-' prefix from paragraph text (PDF artefact when merged across page break)."""
    if not text:
        return text
    return re.sub(r"^\s*Page\s*-\s*\d+\s*-\s*", "", text, flags=re.IGNORECASE).strip()


def _split_into_paragraphs(page_text: str) -> List[ParaInfo]:
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
        # Skip PDF page-number artefacts (standalone "- N -" or "Page -N-")
        if _PAGE_NUMBER_LINE.match(s) or _PAGE_NUMBER_FRAGMENT.match(s):
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
    para_infos = [ParaInfo(text=_strip_page_fragment(p), bbox=None) for p in paragraphs]
    return _merge_continuation_paragraphs(para_infos)


def _should_merge_across_pages(prev_para: str, next_para: str) -> bool:
    """
    Detect page-break continuation: last paragraph on a page continues on the next page.
    This is a best-effort heuristic (demo).
    """
    return bool(_page_break_merge_separator(prev_para, next_para))


def _extract_layout_lines(page) -> list[LineInfo]:
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

    lines: list[LineInfo] = []
    for y in sorted(buckets.keys()):
        ws = sorted(buckets[y], key=lambda d: float(d.get("x0", 0.0)))
        text = " ".join(str(d.get("text", "")).strip() for d in ws).strip()
        if not text:
            continue
        x_left = min(float(d.get("x0", 0.0)) for d in ws)
        x_right = max(float(d.get("x1", 0.0)) for d in ws)
        top = min(float(d.get("top", y)) for d in ws)
        bottom = max(float(d.get("bottom", top)) for d in ws)
        lines.append(LineInfo(top=top, x0=x_left, x1=x_right, bottom=bottom, text=text))
    return lines


def _lines_to_paragraphs(lines: list[LineInfo]) -> list[ParaInfo]:
    """
    Convert (y, x, text) lines into paragraph blocks.

    Heuristics:
    - paragraph break on large vertical gap
    - list block when bullet markers detected; preserve bullets with '\n'
    - merge wrapped lines within paragraph
    """
    paras: list[ParaInfo] = []
    cur_lines: list[str] = []
    prev_y: float | None = None
    prev_x: float | None = None
    cur_box: Optional[tuple[float, float, float, float]] = None

    list_items: list[str] = []
    list_indent: float | None = None
    list_box: Optional[tuple[float, float, float, float]] = None

    def _merge_box(box: Optional[tuple[float, float, float, float]], line: LineInfo):
        if box is None:
            return (line.x0, line.top, line.x1, line.bottom)
        return (min(box[0], line.x0), min(box[1], line.top), max(box[2], line.x1), max(box[3], line.bottom))

    def flush_para() -> None:
        nonlocal cur_lines, cur_box
        text = " ".join(cur_lines).strip()
        if text:
            paras.append(ParaInfo(text=text, bbox=cur_box))
        cur_lines = []
        cur_box = None

    def flush_list() -> None:
        nonlocal list_items, list_indent, list_box
        if list_items:
            paras.append(ParaInfo(text="\n".join(list_items).strip(), bbox=list_box))
        list_items = []
        list_indent = None
        list_box = None

    for line in lines:
        y = line.top
        x = line.x0
        s = (line.text or "").strip()
        if not s:
            continue
        # Skip PDF page-number artefacts
        if _PAGE_NUMBER_LINE.match(s) or _PAGE_NUMBER_FRAGMENT.match(s):
            continue

        is_heading = _is_heading_like(s)
        is_bullet = bool(_BULLET_RE.match(s) or _BULLET_LETTER_O_RE.match(s))

        if is_heading:
            flush_para()
            flush_list()
            paras.append(ParaInfo(text=s, bbox=(line.x0, line.top, line.x1, line.bottom)))
            prev_y, prev_x = y, x
            continue

        # list handling
        if is_bullet:
            flush_para()
            if list_indent is None:
                list_indent = x
            list_items.append(s)
            list_box = _merge_box(list_box, line)
            prev_y, prev_x = y, x
            continue

        if list_items:
            # continuation of last bullet item if indented or previous line didn't end a sentence
            indent_thresh = (list_indent or 0.0) + 10.0
            if x >= indent_thresh or (list_items and not _SENTENCE_END_RE.search(list_items[-1].strip())):
                list_items[-1] = (list_items[-1].rstrip() + " " + s.lstrip()).strip()
                list_box = _merge_box(list_box, line)
                prev_y, prev_x = y, x
                continue
            else:
                flush_list()

        # normal paragraph handling
        if prev_y is None:
            cur_lines.append(s)
            cur_box = _merge_box(cur_box, line)
            prev_y, prev_x = y, x
            continue

        gap = y - prev_y
        # tuned for typical PDF line spacing
        if gap >= 12.0:
            flush_para()
            cur_lines.append(s)
            cur_box = _merge_box(cur_box, line)
        else:
            # indent-based break (e.g. new paragraph starts flush-left after an indented wrap)
            if prev_x is not None and abs(x - prev_x) >= 18.0 and cur_lines and _SENTENCE_END_RE.search(cur_lines[-1].strip()):
                flush_para()
                cur_lines.append(s)
                cur_box = _merge_box(cur_box, line)
            else:
                cur_lines.append(s)
                cur_box = _merge_box(cur_box, line)

        prev_y, prev_x = y, x

    flush_para()
    flush_list()
    paras = [p for p in paras if p.text and p.text.strip()]
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
                sep = _page_break_merge_separator(paragraphs[-1].text, page_paras[0].text)
                if sep:
                    paragraphs[-1].text = (paragraphs[-1].text.rstrip() + sep + page_paras[0].text.lstrip()).strip()
                    paragraphs[-1].bbox = None
                    page_paras = page_paras[1:]
            for para in page_paras:
                bbox = None
                if para.bbox:
                    x0, top, x1, bottom = para.bbox
                    if page.width and page.height:
                        bbox = {
                            "x0": max(0.0, min(1.0, x0 / page.width)),
                            "y0": max(0.0, min(1.0, top / page.height)),
                            "x1": max(0.0, min(1.0, x1 / page.width)),
                            "y1": max(0.0, min(1.0, bottom / page.height)),
                        }
                paragraphs.append(Paragraph(text=para.text, page=i, paragraph_index=para_idx, bbox=bbox))
                para_idx += 1
                if max_paragraphs > 0 and len(paragraphs) >= max_paragraphs:
                    raise ValueError(
                        f"PDF produced > {max_paragraphs} paragraphs, exceeds MAX_PARAGRAPHS={max_paragraphs}."
                    )
    return paragraphs, page_count

