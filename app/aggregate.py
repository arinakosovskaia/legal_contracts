from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Dict, List

from .models import Finding, ResultSummary


def _dedupe_key(category_id: str, evidence_quote: str) -> str:
    h = hashlib.sha256()
    h.update(category_id.encode("utf-8"))
    h.update(b"\n")
    h.update(evidence_quote.strip().encode("utf-8"))
    return h.hexdigest()


def dedupe_findings(findings: list[Finding]) -> list[Finding]:
    seen = set()
    out: list[Finding] = []
    for f in findings:
        k = _dedupe_key(f.category_id, f.evidence_quote)
        if k in seen:
            continue
        seen.add(k)
        out.append(f)
    return out


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

