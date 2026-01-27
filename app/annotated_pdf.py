from __future__ import annotations

from pathlib import Path
from typing import Iterable

from reportlab.lib.colors import Color
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

from .models import Paragraph


def _wrap_with_positions(text: str, max_chars: int) -> list[tuple[str, int, int]]:
    """
    Wrap text to max_chars while keeping original index ranges.
    Returns list of (line_text, start_idx, end_idx).
    """
    if max_chars <= 5:
        max_chars = 5
    lines: list[tuple[str, int, int]] = []
    i = 0
    line_start = 0
    n = len(text)
    while i < n:
        if text[i] == "\n":
            lines.append((text[line_start:i], line_start, i))
            i += 1
            line_start = i
            continue
        if i - line_start >= max_chars:
            break_pos = text.rfind(" ", line_start, i)
            if break_pos <= line_start:
                break_pos = i
            lines.append((text[line_start:break_pos], line_start, break_pos))
            line_start = break_pos
            i = line_start
            continue
        i += 1
    if line_start < n:
        lines.append((text[line_start:n], line_start, n))
    return lines


def _find_quote_ranges(text: str, quotes: Iterable[str]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for q in quotes:
        q = (q or "").strip()
        if not q:
            continue
        start = 0
        while True:
            idx = text.find(q, start)
            if idx == -1:
                break
            ranges.append((idx, idx + len(q)))
            start = idx + max(1, len(q))
    return ranges


def build_annotated_text_pdf(
    *,
    paragraphs: list[Paragraph],
    findings: list[dict],
    out_path: Path,
    title: str = "Annotated PDF (text-only)",
) -> Path:
    page_w, page_h = A4
    margin = 36
    font = "Courier"
    font_size = 10
    line_height = 14
    char_width = pdfmetrics.stringWidth("M", font, font_size)
    max_chars = max(20, int((page_w - 2 * margin) / max(1.0, char_width)))

    quotes_by_para: dict[int, list[str]] = {}
    for f in findings:
        if not f.get("is_unfair", True):
            continue
        loc = f.get("location") or {}
        pidx = loc.get("paragraph_index")
        if pidx is None:
            continue
        quotes_by_para.setdefault(int(pidx), []).append(str(f.get("evidence_quote") or ""))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_path), pagesize=A4)
    c.setTitle(title)
    c.setFont(font, font_size)
    y = page_h - margin
    highlight = Color(1.0, 0.95, 0.6, alpha=1.0)

    # Build paragraph index map for quick lookup
    para_by_idx = {p.paragraph_index: p for p in paragraphs}
    
    for p in paragraphs:
        text = (p.text or "").strip()
        if not text:
            continue
        # Get quotes for this paragraph
        para_quotes = quotes_by_para.get(int(p.paragraph_index), [])
        ranges = _find_quote_ranges(text, para_quotes)
        
        # If quote not found in current paragraph, try searching in adjacent paragraphs
        # (for quotes that span across paragraph boundaries)
        if para_quotes and not ranges:
            # Try previous paragraph
            prev_idx = p.paragraph_index - 1
            if prev_idx in para_by_idx:
                prev_para = para_by_idx[prev_idx]
                prev_text = (prev_para.text or "").strip()
                # Check if quote starts in previous paragraph and continues here
                for q in para_quotes:
                    if len(q) > 20:  # Only for longer quotes
                        # Try to find quote end in current paragraph
                        quote_end = " ".join(q.split()[-5:]) if len(q.split()) >= 5 else q[-50:]
                        if quote_end.lower() in text.lower():
                            # Quote spans paragraphs - highlight the part in current paragraph
                            # Find where quote end appears
                            end_idx = text.lower().find(quote_end.lower())
                            if end_idx >= 0:
                                ranges.append((end_idx, end_idx + len(quote_end)))
            
            # Try next paragraph
            next_idx = p.paragraph_index + 1
            if next_idx in para_by_idx:
                next_para = para_by_idx[next_idx]
                next_text = (next_para.text or "").strip()
                # Check if quote starts here and continues in next paragraph
                for q in para_quotes:
                    if len(q) > 20:  # Only for longer quotes
                        # Try to find quote start in current paragraph
                        quote_start = " ".join(q.split()[:5]) if len(q.split()) >= 5 else q[:50]
                        if quote_start.lower() in text.lower():
                            # Quote spans paragraphs - highlight the part in current paragraph
                            start_idx = text.lower().find(quote_start.lower())
                            if start_idx >= 0:
                                ranges.append((start_idx, start_idx + len(quote_start)))
        
        lines = _wrap_with_positions(text, max_chars)
        for line_text, start_idx, end_idx in lines:
            if y - line_height < margin:
                c.showPage()
                c.setFont(font, font_size)
                y = page_h - margin
            for r_start, r_end in ranges:
                overlap_start = max(r_start, start_idx)
                overlap_end = min(r_end, end_idx)
                if overlap_start < overlap_end:
                    col_start = overlap_start - start_idx
                    col_end = overlap_end - start_idx
                    rect_x = margin + col_start * char_width
                    rect_w = max(1, (col_end - col_start) * char_width)
                    rect_y = y - font_size * 0.75
                    rect_h = line_height * 0.8
                    c.setFillColor(highlight)
                    c.rect(rect_x, rect_y, rect_w, rect_h, fill=1, stroke=0)
            c.setFillColorRGB(0, 0, 0)
            c.drawString(margin, y, line_text)
            y -= line_height
        y -= line_height * 0.4

    c.save()
    return out_path
