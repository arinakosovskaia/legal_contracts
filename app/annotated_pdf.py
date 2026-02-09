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


def _find_quote_ranges_with_ids(
    text: str, quotes_with_ids: Iterable[tuple[str, int]]
) -> list[tuple[int, int, int]]:
    ranges: list[tuple[int, int, int]] = []
    for q, qid in quotes_with_ids:
        q = (q or "").strip()
        if not q:
            continue
        start = 0
        while True:
            idx = text.find(q, start)
            if idx == -1:
                break
            ranges.append((idx, idx + len(q), qid))
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
    gutter_chars = 10
    gutter_width = char_width * gutter_chars
    max_chars = max(20, int((page_w - 2 * margin - gutter_width) / max(1.0, char_width)))

    quotes_by_para: dict[int, list[tuple[str, int]]] = {}
    footnotes: list[dict] = []
    next_id = 1
    for f in findings:
        if not f.get("is_unfair", True):
            continue
        loc = f.get("location") or {}
        pidx = loc.get("paragraph_index")
        if pidx is None:
            continue
        quote = str(f.get("evidence_quote") or "").strip()
        finding_id = next_id
        next_id += 1
        footnotes.append(
            {
                "id": finding_id,
                "quote": quote,
                "explanation": (f.get("explanation") or "").strip(),
                "legal_references": list(f.get("legal_references") or []),
                "possible_consequences": (f.get("possible_consequences") or "").strip(),
                "severity": str((f.get("risk_assessment") or {}).get("severity_of_consequences") or "").strip(),
                "consequences_category": (f.get("consequences_category") or "").strip(),
                "risk_category": (f.get("risk_category") or "").strip(),
            }
        )
        if quote:
            quotes_by_para.setdefault(int(pidx), []).append((quote, finding_id))

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
        ranges = _find_quote_ranges_with_ids(text, para_quotes)
        # If quote matching fails, do NOT highlight the full paragraph.
        # This prevents highlighting entire paragraphs when the specific quote cannot be found.
        # Instead, the finding will be shown in the table but without highlighting.
        # if para_quotes and not ranges:
        #     # OLD BEHAVIOR: highlight full paragraph - REMOVED to prevent false highlighting
        #     # for _q, qid in para_quotes:
        #     #     ranges.append((0, len(text), qid))
        
        # If quote not found in current paragraph, try searching in adjacent paragraphs
        # (for quotes that span across paragraph boundaries)
        if para_quotes and not ranges:
            # Try previous paragraph
            prev_idx = p.paragraph_index - 1
            if prev_idx in para_by_idx:
                prev_para = para_by_idx[prev_idx]
                prev_text = (prev_para.text or "").strip()
                # Check if quote starts in previous paragraph and continues here
                for q, qid in para_quotes:
                    if len(q) > 20:  # Only for longer quotes
                        # Try to find quote end in current paragraph
                        quote_end = " ".join(q.split()[-5:]) if len(q.split()) >= 5 else q[-50:]
                        if quote_end.lower() in text.lower():
                            # Quote spans paragraphs - highlight the part in current paragraph
                            # Find where quote end appears
                            end_idx = text.lower().find(quote_end.lower())
                            if end_idx >= 0:
                                ranges.append((end_idx, end_idx + len(quote_end), qid))
            
            # Try next paragraph
            next_idx = p.paragraph_index + 1
            if next_idx in para_by_idx:
                next_para = para_by_idx[next_idx]
                next_text = (next_para.text or "").strip()
                # Check if quote starts here and continues in next paragraph
                for q, qid in para_quotes:
                    if len(q) > 20:  # Only for longer quotes
                        # Try to find quote start in current paragraph
                        quote_start = " ".join(q.split()[:5]) if len(q.split()) >= 5 else q[:50]
                        if quote_start.lower() in text.lower():
                            # Quote spans paragraphs - highlight the part in current paragraph
                            start_idx = text.lower().find(quote_start.lower())
                            if start_idx >= 0:
                                ranges.append((start_idx, start_idx + len(quote_start), qid))
        
        lines = _wrap_with_positions(text, max_chars)
        for line_text, start_idx, end_idx in lines:
            if y - line_height < margin:
                c.showPage()
                c.setFont(font, font_size)
                y = page_h - margin
            markers: list[tuple[int, int]] = []
            for r_start, r_end, r_id in ranges:
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
                if r_end > start_idx and r_end <= end_idx:
                    markers.append((r_end - start_idx, r_id))
            c.setFillColorRGB(0, 0, 0)
            c.drawString(margin, y, line_text)
            if markers:
                c.setFont(font, max(7, font_size - 2))
                marker_base_x = margin + max_chars * char_width + (char_width * 0.5)
                for i, (_col_pos, r_id) in enumerate(markers):
                    marker_text = f"[{r_id}]"
                    marker_x = marker_base_x + (i * char_width * 3)
                    marker_y = y + 2
                    c.drawString(marker_x, marker_y, marker_text)
                    dest = f"note-{r_id}"
                    marker_w = pdfmetrics.stringWidth(marker_text, font, max(7, font_size - 2))
                    c.linkRect(
                        "",
                        dest,
                        (marker_x, marker_y - 1, marker_x + marker_w, marker_y + font_size),
                        relative=0,
                        thickness=0,
                    )
                c.setFont(font, font_size)
            y -= line_height
        y -= line_height * 0.4

    # Append footnotes at the end
    if footnotes:
        y -= line_height
        if y - line_height < margin:
            c.showPage()
            c.setFont(font, font_size)
            y = page_h - margin
        c.setFont("Courier-Bold", font_size)
        c.drawString(margin, y, "Notes:")
        c.setFont(font, font_size)
        y -= line_height
        for note in footnotes:
            fid = note.get("id")
            quote = note.get("quote") or "—"
            expl = note.get("explanation") or "—"
            legal_refs = note.get("legal_references") or []
            possible = note.get("possible_consequences") or "—"
            severity = note.get("severity") or "—"
            cons_cat = note.get("consequences_category") or "—"
            risk_cat = note.get("risk_category") or "—"
            dest = f"note-{fid}"
            c.bookmarkPage(dest)

            def _draw_label_and_value(label: str, value: str) -> None:
                nonlocal y
                if y - line_height < margin:
                    c.showPage()
                    c.setFont(font, font_size)
                    y = page_h - margin
                c.setFont("Courier-Bold", font_size)
                c.drawString(margin, y, label)
                c.setFont(font, font_size)
                y -= line_height
                for line_text, _s, _e in _wrap_with_positions(value, max_chars):
                    if y - line_height < margin:
                        c.showPage()
                        c.setFont(font, font_size)
                        y = page_h - margin
                    c.drawString(margin, y, line_text)
                    y -= line_height

            _draw_label_and_value(f"[{fid}] Quote", quote)
            _draw_label_and_value("Explanation (UK law)", expl)
            _draw_label_and_value("Legal references", ", ".join(legal_refs) if legal_refs else "—")
            _draw_label_and_value("Possible consequences", possible)
            _draw_label_and_value("Severity of consequences (0–3)", severity)
            _draw_label_and_value("Consequences category", cons_cat)
            _draw_label_and_value("Risk category", risk_cat)

            if y - line_height < margin:
                c.showPage()
                c.setFont(font, font_size)
                y = page_h - margin
            c.setLineWidth(0.5)
            c.line(margin, y - 2, page_w - margin, y - 2)
            y -= line_height * 0.8

    c.save()
    return out_path
