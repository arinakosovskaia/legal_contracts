from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Markers / numbering forms (by shape, not by meaning).
_RE_ARTICLE_ROMAN = re.compile(r"^\s*ARTICLE\s+(?P<roman>[IVXLCDM]+)\b", re.IGNORECASE)
_RE_ARTICLE_NUM = re.compile(r"^\s*ARTICLE\s+(?P<num>\d+)\b", re.IGNORECASE)
_RE_SECTION_NUM = re.compile(r"^\s*SECTION\s+(?P<num>\d+(?:\.\d+)*)\b", re.IGNORECASE)
_RE_CLAUSE_NUM = re.compile(r"^\s*CLAUSE\s+(?P<num>\d+(?:\.\d+)*)\b", re.IGNORECASE)
_RE_PARA_SIGN = re.compile(r"^\s*§\s*(?P<num>\d+(?:\.\d+)*)\b")
_RE_NUMBERED = re.compile(
    r"^\s*(?P<num>\d+(?:\.\d+){0,6})(?P<tail>\([a-z]\))?\s*(?:[.)]\s*)?(?P<title>.+)?$",
    re.IGNORECASE,
)

# Unnumbered title-like lines (ALL CAPS / Title Case) — evaluated via heuristics.
_RE_ALLCAPS_LINE = re.compile(r"^[A-Z0-9][A-Z0-9 .,:;()'\"/\\-]{2,}$")

_SENTENCE_END = re.compile(r"[.!?]$")
_BAD_PUNCT = re.compile(r"[;]")
_EXCESS_COMMAS = re.compile(r",")
_DOT_SPACE = re.compile(r"\.\s+\w+")
_MODAL_VERBS = re.compile(
    r"\b(shall|will|must|may|may not|shall not|will not|agrees?|warrants?|represents?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Heading:
    level: int  # 0 = article-like, 1 = section-like, 2+ = sub-sections (1.2, 4.1(a), ...)
    label: str  # normalized label for "path"
    title: Optional[str]  # optional heading title text
    raw: str


def _uppercase_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    upp = sum(1 for c in letters if c.isupper())
    return upp / len(letters)


def _titlecase_ratio(text: str) -> float:
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if not words:
        return 0.0
    def is_title(w: str) -> bool:
        return len(w) >= 2 and w[0].isupper() and any(ch.islower() for ch in w[1:])
    return sum(1 for w in words if is_title(w)) / len(words)


def _looks_like_sentence(text: str) -> bool:
    t = text.strip()
    if _BAD_PUNCT.search(t):
        return True
    if len(_EXCESS_COMMAS.findall(t)) >= 2:
        return True
    # If it contains modal verbs, it's usually clause text, not a heading.
    # (Headings rarely say "shall/will/must".)
    if _MODAL_VERBS.search(t):
        return True
    # Multiple sentence-like segments (e.g., "6.1 Assignment. No assignment ...").
    # We only flag it if there's enough text after the first period to look like a sentence,
    # to avoid false positives for abbreviations like "U.S.".
    if _DOT_SPACE.search(t) and ". " in t:
        after = t.split(". ", 1)[1]
        if len(after.split()) >= 3:
            return True
    # If it ends with sentence punctuation and is not just a numbering marker, likely not a heading.
    if _SENTENCE_END.search(t) and not re.match(r"^\s*\d+(?:\.\d+)*[.)]?\s*$", t):
        return True
    return False


def parse_heading(text: str, *, max_len: int = 120) -> Optional[Heading]:
    """
    Detect heading-like lines by FORM.

    Rules (condensed):
    - short
    - begins with a marker (numbering / roman / Article/Section/Clause / §)
    - OR looks like a title (ALL CAPS / Title Case), not sentence-like
    """
    raw = (text or "").strip()
    if not raw:
        return None
    if len(raw) > max_len:
        return None
    
    # Filter out contact/address/signature lines that look like headings but aren't
    # Patterns like "Attn: Name, Title", "If to Company to:", "With a copy to:", etc.
    contact_patterns = [
        re.compile(r"^\s*Attn:\s+", re.IGNORECASE),  # "Attn: Joseph Marino, President"
        re.compile(r"^\s*Attention:\s+", re.IGNORECASE),  # "Attention: Name"
        re.compile(r"^\s*If\s+to\s+", re.IGNORECASE),  # "If to Company to:"
        re.compile(r"^\s*With\s+a\s+copy\s+to:\s*", re.IGNORECASE),  # "With a copy to:"
        re.compile(r"^\s*Facsimile:\s*", re.IGNORECASE),  # "Facsimile: number"
        re.compile(r"^\s*Fax:\s*", re.IGNORECASE),  # "Fax: number"
        re.compile(r"^\s*Phone:\s*", re.IGNORECASE),  # "Phone: number"
        re.compile(r"^\s*Email:\s*", re.IGNORECASE),  # "Email: address"
        re.compile(r"^\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+[A-Z][a-z]+)?\s*$"),  # "Joseph Marino, President" or "John Smith, Esq." (name, title pattern)
        re.compile(r"^\s*\d+\s+[A-Z][a-z]+\s+(?:Street|Road|Avenue|Drive|Lane|Boulevard|Way|Court|Place|Circle|Terrace|Parkway|Highway|Trail|Blvd)\b", re.IGNORECASE),  # Address lines
        # Signature patterns
        re.compile(r"^\s*By:\s*", re.IGNORECASE),  # "By: /s/Joseph Marino"
        re.compile(r"^\s*By:\s+.*By:\s+", re.IGNORECASE),  # "By: /s/Joseph Marino By: Jim Stump"
        re.compile(r"^\s*/s/\s*", re.IGNORECASE),  # "/s/ Name" (signature marker)
        re.compile(r"^\s*Signed:\s*", re.IGNORECASE),  # "Signed: Name"
        re.compile(r"^\s*Signature:\s*", re.IGNORECASE),  # "Signature: Name"
        re.compile(r"^\s*[A-Z][a-z]+\s+[A-Z][a-z]+\s*$"),  # Just two capitalized words (likely name, e.g., "Joseph Marino")
    ]
    for pattern in contact_patterns:
        if pattern.match(raw):
            return None

    # Marker-based headings
    m = _RE_ARTICLE_ROMAN.match(raw) or _RE_ARTICLE_NUM.match(raw)
    if m:
        token = m.group(0).strip()
        if _looks_like_sentence(raw):
            return None
        rest = raw[len(m.group(0)) :].strip(" \t-–—:") or None
        return Heading(level=0, label=token.title(), title=rest, raw=raw)

    m = _RE_SECTION_NUM.match(raw)
    if m:
        if _looks_like_sentence(raw):
            return None
        num = m.group("num")
        # Do not treat subsection numbers like "5.1" / "5.2" as headings at all.
        if "." in (num or ""):
            return None
        rest_raw = raw[len(m.group(0)) :]
        # Reject mid-sentence references like "SECTION 10, OR THE RESPECTIVE..."
        if rest_raw and rest_raw.lstrip()[0:1] in (",", ";"):
            return None
        rest = rest_raw.strip(" \t-–—:") or None
        return Heading(level=1, label=f"Section {num}", title=rest, raw=raw)

    m = _RE_CLAUSE_NUM.match(raw)
    if m:
        if _looks_like_sentence(raw):
            return None
        num = m.group("num")
        # Do not treat subsection numbers like "5.1" / "5.2" as headings at all.
        if "." in (num or ""):
            return None
        rest = raw[len(m.group(0)) :].strip(" \t-–—:") or None
        return Heading(level=1, label=f"Clause {num}", title=rest, raw=raw)

    m = _RE_PARA_SIGN.match(raw)
    if m:
        if _looks_like_sentence(raw):
            return None
        num = m.group("num")
        # Do not treat subsection numbers like "5.1" / "5.2" as headings at all.
        if "." in (num or ""):
            return None
        # treat like section-level
        rest = raw[len(m.group(0)) :].strip(" \t-–—:") or None
        return Heading(level=1, label=f"§ {num}", title=rest, raw=raw)

    m = _RE_NUMBERED.match(raw)
    if m and m.group("num"):
        num = m.group("num")
        # Do not treat pure decimal numbers like "5.1" / "5.2" as headings at all.
        # We only want top-level numeric markers ("1.", "2)") to be headings.
        if "." in (num or ""):
            return None
        tail = m.group("tail") or ""
        title = (m.group("title") or "").strip()
        title = title.strip(" \t-–—:") or ""
        title = title or None
        # Level 2 means "top-level numeric heading" (e.g., "1.", "2)")
        level = 2
        if tail:
            level += 1
        # Bare numbers without title or period/bracket (e.g. "2", "15") are page numbers, not headings
        if not title and not tail and not re.match(r"^\s*\d+\s*[.)]\s*$", raw):
            return None
        if _looks_like_sentence(raw):
            # Allow bare markers like "1." even if sentence end is "."
            if re.match(r"^\s*\d+[.)]?\s*$", raw):
                return Heading(level=level, label=num + tail, title=None, raw=raw)
            return None
        # Reject when "title" starts with a lowercase letter — it's body text, not a heading
        # e.g. "24 hours' written notice" has num=24 but title="hours' written notice"
        if title and title[0].islower():
            return None
        # Reject when "title" is purely numeric — likely zip code or page reference (e.g. "98004 51054")
        if title and re.match(r"^[\d\s]+$", title):
            return None
        return Heading(level=level, label=num + tail, title=title, raw=raw)

    # Unnumbered headings: ALL CAPS / Title Case and not sentence-like
    if _looks_like_sentence(raw):
        return None
    # Headings never start with a lowercase letter
    if raw and raw[0].islower():
        return None
    words = raw.split()
    if not (1 <= len(words) <= 12):
        return None
    # Exclude UK postcodes (e.g. "AB1 2CD", "SW1A 1AA")
    if re.match(r"^\s*[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\s*$", raw):
        return None
    # Exclude "Label: Value" lines (e.g. "Account Holder: Ashwick Court")
    if ":" in raw and not raw.strip().endswith(":"):
        return None
    # ALL CAPS headings (RECITALS, RIGHT OF WAY, etc.) — keep broad
    # But reject if it contains commas — likely body text not a heading
    if _RE_ALLCAPS_LINE.match(raw) and _uppercase_ratio(raw) >= 0.85:
        if "," not in raw:
            return Heading(level=1, label=raw.title(), title=None, raw=raw)
    # Title Case headings — stricter: require 2+ words, exclude addresses and names
    if re.search(r"\b(?:Street|Road|Avenue|Drive|Lane|Boulevard|Way|Court|Place|Circle|Suite|Floor|Flat|Room|Building|Terrace|Parkway|Highway|Trail|Blvd)\b", raw, re.IGNORECASE):
        return None
    if re.match(r"^\s*(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s", raw, re.IGNORECASE):
        return None
    if len(words) >= 2 and _titlecase_ratio(raw) >= 0.7:
        return Heading(level=1, label=raw, title=None, raw=raw)

    return None


def is_heading(text: str, *, max_len: int = 120) -> bool:
    return parse_heading(text, max_len=max_len) is not None


