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
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"dedupe_findings: Input {len(findings)} findings")
    if findings:
        logger.info(f"dedupe_findings: Input categories: {[f.category_name for f in findings]}")
    seen = set()
    out: list[Finding] = []

    def score_finding(f: Finding) -> tuple[int, int]:
        quote_len = len(f.evidence_quote or "")
        expl_len = len(f.explanation or "")
        return (quote_len, expl_len)
    
    def _normalize_for_match(text: str) -> str:
        """
        Aggressive normalization for quote matching: first apply normalize_llm_text to ensure
        consistency with the text that LLM sees, then collapse all whitespace to single space
        and lowercase for robust matching.
        
        This ensures we always compare against the same normalized base text, preventing
        mismatches from different normalization approaches.
        """
        from .llm import normalize_llm_text
        # First normalize using the same function that processes text for LLM
        normalized = normalize_llm_text(text or "")
        # Then apply aggressive normalization for matching: lowercase + collapse whitespace
        t = normalized.lower().strip()
        t = re.sub(r"\s+", " ", t)
        return t
    
    def quotes_intersect(quote1: str, quote2: str, *, strict_for_adjacent: bool = False) -> bool:
        """
        Check if two quotes have any overlap/intersection (normalized for comparison).
        
        Args:
            strict_for_adjacent: If True, use stricter checks for adjacent paragraphs.
                Only returns True if quotes are clearly parts of one continuous quote
                (one continues the other, or one contains the other).
                If False, also allows merging based on common words (for backward compatibility).
        
        Returns True if:
        - One quote contains the other (substring match), OR
        - One quote continues the other (end of one matches start of other), OR
        - (if not strict) They share at least 3 meaningful common words
        
        Examples:
        - "Company may terminate" and "Company may terminate this Agreement" -> True (substring match)
        - "Company may terminate" and "may terminate this Agreement" -> True (one continues other)
        - "Company may terminate" and "Either party may terminate" -> True (if not strict) / False (if strict)
        - "any price changes" and "Uncapped liability clause" -> False (no overlap)
        """
        if not quote1 or not quote2:
            return False
        # Normalize both quotes
        q1_norm = _normalize_for_match(quote1)
        q2_norm = _normalize_for_match(quote2)
        
        # Check if one contains the other (substring match)
        # This handles cases like "Company may terminate" and "Company may terminate this Agreement"
        if q1_norm in q2_norm or q2_norm in q1_norm:
            return True
        
        # Check if one quote continues the other (for adjacent paragraphs)
        # This handles cases where quote is split across paragraphs:
        # - "Company may terminate" and "this Agreement at any time" -> check if they can be joined
        # Check if end of one matches beginning of other (at least 3 words overlap)
        q1_words = q1_norm.split()
        q2_words = q2_norm.split()
        
        for overlap_len in range(min(5, len(q1_words), len(q2_words)), 2, -1):
            if len(q1_words) >= overlap_len and len(q2_words) >= overlap_len:
                q1_end = " ".join(q1_words[-overlap_len:])
                q2_start = " ".join(q2_words[:overlap_len])
                if q1_end == q2_start:
                    return True
        
        for overlap_len in range(min(5, len(q1_words), len(q2_words)), 2, -1):
            if len(q1_words) >= overlap_len and len(q2_words) >= overlap_len:
                q1_start = " ".join(q1_words[:overlap_len])
                q2_end = " ".join(q2_words[-overlap_len:])
                if q1_start == q2_end:
                    return True
        
        if strict_for_adjacent:
            return False
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "from", "as", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "should", "could", "may", "might", "must",
            "this", "that", "these", "those", "other", "than", "any", "all", "some", "each", "every"
        }
        
        words1 = set(q1_norm.split())
        words2 = set(q2_norm.split())
        common_words = words1 & words2
        meaningful_common = {w for w in common_words if len(w) > 3 and w not in stopwords}
        if len(meaningful_common) >= 3:
            return True
        
        shorter = min(len(q1_norm), len(q2_norm))
        if shorter > 0:
            shorter_words = q1_norm.split() if len(q1_norm) < len(q2_norm) else q2_norm.split()
            longer_words = q2_norm.split() if len(q1_norm) < len(q2_norm) else q1_norm.split()
            longer_set = set(longer_words)
            matching_words = [w for w in shorter_words if len(w) > 3 and w not in stopwords and w in longer_set]
            if len(matching_words) >= max(3, len([w for w in shorter_words if len(w) > 3 and w not in stopwords]) * 0.5):
                return True
        
        return False
    
    def _core_signature(f: Finding) -> tuple[Any, ...]:
        """
        Build a signature of the 'core' fields of a finding, excluding evidence_quote,
        location, bbox and IDs. If two findings have the same core signature, they are
        likely the same logical finding (e.g., seen in two different windows) even if
        their quotes differ slightly.
        """
        # Normalize helper
        def norm(s: str | None) -> str:
            return (s or "").strip()
        
        risk = f.risk_assessment
        legal_refs = tuple(sorted(str(r).strip() for r in (f.legal_references or [])))
        return (
            f.category_id,
            f.category_name,
            norm(f.issue_title),
            norm(f.what_is_wrong),
            norm(f.explanation),
            norm(f.possible_consequences),
            risk.severity_of_consequences if risk is not None else 0,
            risk.degree_of_legal_violation if risk is not None else 0,
            f.consequences_category,
            f.risk_category,
            norm(f.revised_clause),
            norm(f.revision_explanation),
            norm(f.suggested_follow_up),
            bool(f.is_unfair),
            legal_refs,
        )
    
    def core_equal(f1: Finding, f2: Finding) -> bool:
        """Return True if two findings are identical in all core fields except the quote/location."""
        return _core_signature(f1) == _core_signature(f2)
    
    # First pass: exact deduplication
    # Also check for identical quotes (normalized) with same category - these should always be merged
    # This handles cases where LLM returns the same quote with slightly different what_is_wrong/explanation
    quote_category_map: dict[tuple[str, str], Finding] = {}  # (normalized_quote, category_id) -> best finding
    
    for f in findings:
        k = _dedupe_key(f.category_id, f.evidence_quote, f.what_is_wrong)
        if k in seen:
            logger.debug(f"dedupe_findings: Skipping exact duplicate: {f.category_name} - {f.evidence_quote[:50]}")
            continue
        seen.add(k)
        
        # Check if we've seen this exact quote (normalized) with same category before
        # If quote is identical and category is same, it's the same finding regardless of other fields
        quote_norm = _normalize_for_match(f.evidence_quote or "")
        if len(quote_norm) > 20:  # Only check for substantial quotes
            quote_cat_key = (quote_norm, f.category_id)
            
            if quote_cat_key in quote_category_map:
                # Same quote + same category = same finding, merge by keeping the best one
                existing = quote_category_map[quote_cat_key]
                if score_finding(f) > score_finding(existing):
                    # Replace with better finding
                    quote_category_map[quote_cat_key] = f
                    # Remove old one from out if it was added
                    if existing in out:
                        out.remove(existing)
                        out.append(f)
                    logger.info(
                        f"dedupe_findings: Merging duplicate quote+category (replaced): {f.category_name} - "
                        f"kept better finding"
                    )
                else:
                    # Keep existing, skip this one
                    logger.info(
                        f"dedupe_findings: Merging duplicate quote+category (skipped): {f.category_name} - "
                        f"kept existing finding"
                    )
                continue  # Skip adding this finding
            else:
                quote_category_map[quote_cat_key] = f
        
        out.append(f)
    
    logger.info(f"dedupe_findings: After first pass (exact dedup + identical quotes): {len(out)} findings")
    
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
            merged.append(f)
    
    # For each paragraph+category group, if multiple findings exist, check if they should be merged
    # Merge if quotes intersect OR all core fields (reasoning, refs, risk, etc.) are identical.
    for (para_idx, cat_id), group in by_para_cat.items():
        if len(group) == 1:
            merged.append(group[0])
            continue
        
        merged_groups: list[list[Finding]] = []
        processed_in_group = set()
        
        for f in group:
            if id(f) in processed_in_group:
                continue
            f_quote = (f.evidence_quote or "").strip()
            current_group = [f]
            processed_in_group.add(id(f))
            
            for other in group:
                if id(other) in processed_in_group:
                    continue
                other_quote = (other.evidence_quote or "").strip()
                # Check if one quote is a substring of the other (normalized) - strongest signal
                f_quote_norm = _normalize_for_match(f_quote)
                other_quote_norm = _normalize_for_match(other_quote)
                is_substring = (f_quote_norm in other_quote_norm or other_quote_norm in f_quote_norm) and len(f_quote_norm) > 20 and len(other_quote_norm) > 20
                
                # Priority order:
                # 1. One quote is substring of another (strongest signal - same quote, different length)
                # 2. Quotes intersect (overlap/continuation)
                # 3. Core fields match (same logical finding, different quotes)
                if is_substring or quotes_intersect(f_quote, other_quote) or core_equal(f, other):
                    current_group.append(other)
                    processed_in_group.add(id(other))
            
            merged_groups.append(current_group)
        
        for merge_group in merged_groups:
            if len(merge_group) == 1:
                merged.append(merge_group[0])
            else:
                logger.info(
                    f"dedupe_findings: Merging {len(merge_group)} findings from same paragraph {para_idx}, "
                    f"category {cat_id} - quotes intersect"
                )
                best = max(merge_group, key=score_finding)
                merged.append(best)
    logger.info(f"dedupe_findings: After second pass (same paragraph merge): {len(merged)} findings")
    
    # Third pass: merge findings from adjacent paragraphs (within 3 indices) with same category.
    # Merge if their quotes overlap/intersect (strict) OR all core fields are identical
    # (e.g., same reasoning/refs but split across nearby paragraphs/windows).
    # 
    # Note: Findings from different windows should typically be in adjacent or overlapping paragraphs.
    # If they're far apart (> 3), they're likely different findings, even if quotes are similar.
    final: list[Finding] = []
    processed = set()
    
    merged_sorted = sorted(merged, key=lambda f: f.location.get("paragraph_index", 0) if f.location else 0)
    
    for i, f in enumerate(merged_sorted):
        if id(f) in processed:
            continue
        
        para_idx = f.location.get("paragraph_index", -1) if f.location else -1
        if para_idx < 0:
            final.append(f)
            continue
        
        adjacent_group = [f]
        processed.add(id(f))
        f_quote = (f.evidence_quote or "").strip()
        
        for other in merged_sorted[i+1:]:
            if id(other) in processed:
                continue
            other_para = other.location.get("paragraph_index", -1) if other.location else -1
            if other_para < 0:
                continue
            
            para_distance = abs(other_para - para_idx)
            other_quote = (other.evidence_quote or "").strip()
            
            if other.category_id != f.category_id:
                continue
            
            # Check if one quote is a substring of the other (normalized) - strongest signal
            f_quote_norm = _normalize_for_match(f_quote)
            other_quote_norm = _normalize_for_match(other_quote)
            is_substring = (f_quote_norm in other_quote_norm or other_quote_norm in f_quote_norm) and len(f_quote_norm) > 20 and len(other_quote_norm) > 20
            
            # Merge findings if they're from the same category and:
            # Priority order:
            # 1. One quote is substring of another (strongest - same quote, different length)
            # 2. Quotes intersect (overlap/continuation)
            # 3. Core fields match (same logical finding, different quotes)
            # 
            # For adjacent paragraphs (<= 3): merge if any of the above
            # For far apart (> 3): merge ONLY if quotes are related (substring or intersect) AND core fields match
            quote_match = quotes_intersect(f_quote, other_quote, strict_for_adjacent=(para_distance <= 3))
            
            should_merge = False
            if para_distance <= 3:
                # For adjacent paragraphs: merge if quotes are related (substring/intersect) OR core fields match
                should_merge = is_substring or quote_match or core_equal(f, other)
            else:
                # For far apart paragraphs: merge ONLY if quotes are related (substring/intersect) AND core fields match
                # This handles cases where one quote is split across multiple windows (e.g., 3 windows)
                # Example: Window 1 finds "Company may terminate", Window 2 finds "may terminate this Agreement",
                #          Window 3 finds "this Agreement at any time" - all are parts of the same finding
                should_merge = (is_substring or quote_match) and core_equal(f, other)
            
            if should_merge:
                reason = []
                if is_substring:
                    reason.append("quote_substring")
                if quote_match:
                    reason.append("quotes_intersect")
                if core_equal(f, other):
                    reason.append("core_equal")
                logger.info(
                    f"dedupe_findings: Merging findings: para {para_idx} ({f.category_name}) "
                    f"and para {other_para} ({other.category_name}) - distance={para_distance}, "
                    f"reasons={', '.join(reason)}"
                )
                adjacent_group.append(other)
                processed.add(id(other))
        
        if len(adjacent_group) > 1:
            # Sort adjacent findings by paragraph index to preserve document order
            adjacent_group_sorted = sorted(
                adjacent_group,
                key=lambda x: x.location.get("paragraph_index", 0) if x.location else 0,
            )
            logger.info(
                f"dedupe_findings: Merging {len(adjacent_group_sorted)} adjacent findings: "
                f"{[f.category_name for f in adjacent_group_sorted]}"
            )
            
            merged = adjacent_group_sorted[0]
            
            all_quotes = []
            seen_quote_parts = set()
            for finding in adjacent_group_sorted:
                quote = (finding.evidence_quote or "").strip()
                if quote:
                    quote_norm = _normalize_for_match(quote)
                    is_duplicate = False
                    for existing in all_quotes:
                        existing_norm = _normalize_for_match(existing)
                        if quote_norm in existing_norm or existing_norm in quote_norm:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        all_quotes.append(quote)
            
            all_explanations = []
            seen_expl = set()
            for finding in adjacent_group_sorted:
                expl = (finding.explanation or "").strip()
                if expl and expl not in seen_expl:
                    seen_expl.add(expl)
                    all_explanations.append(expl)
            
            all_legal_refs = []
            seen_refs = set()
            for finding in adjacent_group_sorted:
                refs = finding.legal_references or []
                for ref in refs:
                    ref_str = str(ref).strip()
                    if ref_str and ref_str not in seen_refs:
                        seen_refs.add(ref_str)
                        all_legal_refs.append(ref_str)
            
            # Choose the best quote: prefer the shortest quote that contains all others,
            # or the longest if they're not nested (to preserve most context)
            if len(all_quotes) > 1:
                # Check if quotes are nested (one contains another)
                normalized_quotes = [(q, _normalize_for_match(q)) for q in all_quotes]
                # Find the shortest quote that contains all others
                best_quote = None
                best_quote_len = float('inf')
                for quote, quote_norm in normalized_quotes:
                    contains_all = True
                    for other_quote, other_norm in normalized_quotes:
                        if quote_norm != other_norm and other_norm not in quote_norm:
                            contains_all = False
                            break
                    if contains_all and len(quote) < best_quote_len:
                        best_quote = quote
                        best_quote_len = len(quote)
                
                if best_quote:
                    combined_quote = best_quote
                else:
                    # If no quote contains all others, use the longest one (most context)
                    combined_quote = max(all_quotes, key=len)
            else:
                combined_quote = all_quotes[0] if all_quotes else merged.evidence_quote
            
            combined_explanation = "\n\n".join(all_explanations) if len(all_explanations) > 1 else (all_explanations[0] if all_explanations else merged.explanation)
            
            max_severity_order = {"high": 3, "medium": 2, "low": 1}
            best_severity = max(adjacent_group_sorted, key=lambda x: max_severity_order.get(x.severity, 0))
            best_confidence = max(adjacent_group_sorted, key=lambda x: x.confidence or 0.0)
            
            # Create merged finding
            merged_finding = Finding(
                finding_id=merged.finding_id,  # Keep first finding's ID
                category_id=merged.category_id,
                category_name=merged.category_name,
                severity=best_severity.severity,
                confidence=best_confidence.confidence,
                is_unfair=any(f.is_unfair for f in adjacent_group_sorted),
                issue_title=merged.issue_title,
                what_is_wrong=merged.what_is_wrong,  # Keep first one
                explanation=combined_explanation,
                recommendation=merged.recommendation,  # Keep first one
                evidence_quote=combined_quote,
                legal_references=all_legal_refs[:6],  # Limit to 6
                possible_consequences=merged.possible_consequences,  # Keep first one
                risk_assessment=merged.risk_assessment,  # Keep first one
                consequences_category=merged.consequences_category,  # Keep first one
                risk_category=merged.risk_category,  # Keep first one
                revised_clause=merged.revised_clause,  # Keep first one
                revision_explanation=merged.revision_explanation,  # Keep first one
                suggested_follow_up=merged.suggested_follow_up,  # Keep first one
                paragraph_bbox=merged.paragraph_bbox,  # Keep first one's bbox
                location=merged.location,  # Keep first one's location
            )
            final.append(merged_finding)
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

