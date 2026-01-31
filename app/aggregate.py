from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any, Dict, List

from .models import Finding, ResultSummary


def _dedupe_key(category_id: str, evidence_quote: str, what_is_wrong: str = "") -> str:
    h = hashlib.sha256()
    cat = (category_id or "").strip().lower()
    quote = (evidence_quote or "").strip()
    # Normalize whitespace more aggressively
    quote = re.sub(r"\s+", " ", quote)
    # Remove control chars and black squares
    quote = re.sub(r"[\u0000-\u001f\u007f\u25A0-\u25FF]", " ", quote)
    quote = quote.strip()
    # If quote is empty or very short, use what_is_wrong as additional discriminator
    if len(quote) < 10:
        what = (what_is_wrong or "").strip()[:200]
        what = re.sub(r"\s+", " ", what)
        quote = quote + "|" + what
    h.update(cat.encode("utf-8"))
    h.update(b"\n")
    h.update(quote.encode("utf-8"))
    return h.hexdigest()


def dedupe_findings(findings: list[Finding]) -> list[Finding]:
    """
    Deduplicate findings by:
    1. Exact match on category_id + normalized quote + what_is_wrong
    2. Same paragraph + same category_id (merge all findings from same paragraph with same category)
    3. Adjacent paragraphs (within 3 indices) + same category_id + similar explanations
       (handles cases where one logical clause spans multiple paragraphs in the chunk)
    """
    seen = set()
    out: list[Finding] = []

    def score_finding(f: Finding) -> tuple[int, int]:
        quote_len = len(f.evidence_quote or "")
        expl_len = len(f.explanation or "")
        return (quote_len, expl_len)
    
    # First pass: exact deduplication
    for f in findings:
        k = _dedupe_key(f.category_id, f.evidence_quote, f.what_is_wrong)
        if k in seen:
            continue
        seen.add(k)
        out.append(f)
    
    # Second pass: merge findings from same paragraph with same category
    merged: list[Finding] = []
    by_para_cat: dict[tuple[int, str], list[Finding]] = {}
    
    for f in out:
        para_idx = f.location.get("paragraph_index", -1) if f.location else -1
        if para_idx >= 0:
            key = (para_idx, f.category_id)
            if key not in by_para_cat:
                by_para_cat[key] = []
            by_para_cat[key].append(f)
        else:
            # No paragraph info, keep as is
            merged.append(f)
    
    # For each paragraph+category group, if multiple findings exist, merge them
    for (para_idx, cat_id), group in by_para_cat.items():
        if len(group) == 1:
            merged.append(group[0])
            continue
        
        # Multiple findings from same paragraph with same category - merge them
        best = max(group, key=score_finding)
        merged.append(best)
    
    # Third pass: merge findings from adjacent paragraphs (within 3 indices) with same category
    # This handles cases where one logical clause spans multiple paragraphs in a chunk
    # or where LLM finds sub-clauses that get assigned to different paragraphs
    final: list[Finding] = []
    processed = set()
    
    # Sort by paragraph_index for easier adjacency checking
    merged_sorted = sorted(merged, key=lambda f: f.location.get("paragraph_index", 0) if f.location else 0)
    
    for i, f in enumerate(merged_sorted):
        if id(f) in processed:
            continue
        
        para_idx = f.location.get("paragraph_index", -1) if f.location else -1
        if para_idx < 0:
            final.append(f)
            continue
        
        # Find all findings from adjacent paragraphs (within 3) with same category
        # Merge them if they're from the same logical clause (same category is enough)
        adjacent_group = [f]
        processed.add(id(f))
        
        for other in merged_sorted[i+1:]:
            if id(other) in processed:
                continue
            other_para = other.location.get("paragraph_index", -1) if other.location else -1
            if other_para < 0:
                continue
            
            # If adjacent (within 3 indices) and same category, merge them
            # This handles cases where sub-clauses (A), (b), (c) get assigned to different paragraphs
            if abs(other_para - para_idx) <= 3 and other.category_id == f.category_id:
                adjacent_group.append(other)
                processed.add(id(other))
        
        # If we found adjacent findings, merge them (keep the one with longest quote)
        if len(adjacent_group) > 1:
            best = max(adjacent_group, key=score_finding)
            final.append(best)
        else:
            final.append(f)
    
    return final


def compute_summary(findings: list[Finding]) -> ResultSummary:
    counts = Counter(f.severity for f in findings)
    score = counts.get("high", 0) * 20 + counts.get("medium", 0) * 10 + counts.get("low", 0) * 5
    risk_score = min(100, int(score))
    cat_counts = Counter(f.category_id for f in findings)
    top_categories = [{"category_id": k, "count": v} for k, v in cat_counts.most_common(10)]
    return ResultSummary(
        risk_score=risk_score,
        high=int(counts.get("high", 0)),
        medium=int(counts.get("medium", 0)),
        low=int(counts.get("low", 0)),
        top_categories=top_categories,
    )

