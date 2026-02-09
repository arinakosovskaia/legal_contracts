#!/usr/bin/env python3
"""
Simplified test to check deduplication logic without full dependencies.
"""

import re
from typing import List, Dict, Any

class MockFinding:
    def __init__(self, finding_id, category_id, category_name, evidence_quote, location, explanation=''):
        self.finding_id = finding_id
        self.category_id = category_id
        self.category_name = category_name
        self.evidence_quote = evidence_quote
        self.location = location
        self.explanation = explanation
        self.what_is_wrong = ''
        self.severity = 'medium'
        self.confidence = 0.8
        self.is_unfair = True
        self.issue_title = category_name
        self.recommendation = ''
        self.legal_references = []
        self.possible_consequences = ''
        self.risk_assessment = {'severity_of_consequences': 1, 'degree_of_legal_violation': 1}
        self.consequences_category = 'Unenforceable'
        self.risk_category = 'Nothing'
        self.revised_clause = ''
        self.revision_explanation = ''
        self.suggested_follow_up = ''
        self.paragraph_bbox = None

def _normalize_for_match(text: str) -> str:
    t = (text or '').lower().strip()
    t = re.sub(r'\s+', ' ', t)
    return t

def quotes_intersect(quote1: str, quote2: str) -> bool:
    if not quote1 or not quote2:
        return False
    q1_norm = _normalize_for_match(quote1)
    q2_norm = _normalize_for_match(quote2)
    if q1_norm in q2_norm or q2_norm in q1_norm:
        return True
    words1 = set(q1_norm.split())
    words2 = set(q2_norm.split())
    common_words = words1 & words2
    meaningful_common = {w for w in common_words if len(w) > 2}
    return len(meaningful_common) >= 2

def _dedupe_key(category_id: str, evidence_quote: str, what_is_wrong: str = "") -> str:
    import hashlib
    quote_norm = _normalize_for_match(evidence_quote)
    h = hashlib.md5()
    h.update(category_id.encode("utf-8"))
    h.update(quote_norm.encode("utf-8"))
    h.update((what_is_wrong or "").encode("utf-8"))
    return h.hexdigest()

def dedupe_findings_simple(findings: List[MockFinding]) -> List[MockFinding]:
    """Simplified version of dedupe_findings logic"""
    seen = set()
    out: List[MockFinding] = []
    
    def score_finding(f: MockFinding) -> tuple[int, int]:
        quote_len = len(f.evidence_quote or "")
        expl_len = len(f.explanation or "")
        return (quote_len, expl_len)
    
    # First pass: exact deduplication
    print("\n=== First pass: Exact deduplication ===")
    for f in findings:
        k = _dedupe_key(f.category_id, f.evidence_quote, f.what_is_wrong)
        if k in seen:
            print(f"  ✗ Skipping exact duplicate: {f.category_name} - {f.evidence_quote[:50]}")
            continue
        seen.add(k)
        out.append(f)
        print(f"  ✓ Keeping: {f.category_name} - {f.evidence_quote[:50]}")
    
    print(f"\nAfter first pass: {len(out)} findings")
    
    # Second pass: merge findings from same paragraph with same category
    print("\n=== Second pass: Same paragraph merge ===")
    merged: List[MockFinding] = []
    by_para_cat: Dict[tuple[int, str], List[MockFinding]] = {}
    
    for f in out:
        para_idx = f.location.get("paragraph_index", -1) if f.location else -1
        if para_idx >= 0:
            key = (para_idx, f.category_id)
            if key not in by_para_cat:
                by_para_cat[key] = []
            by_para_cat[key].append(f)
        else:
            merged.append(f)
    
    for (para_idx, cat_id), group in by_para_cat.items():
        if len(group) == 1:
            merged.append(group[0])
            print(f"  ✓ Para {para_idx}, category {cat_id}: 1 finding, keeping")
        else:
            print(f"  ⚠️  Para {para_idx}, category {cat_id}: {len(group)} findings, merging (keeping best)")
            best = max(group, key=score_finding)
            merged.append(best)
    
    print(f"\nAfter second pass: {len(merged)} findings")
    
    # Third pass: merge findings from adjacent paragraphs
    print("\n=== Third pass: Adjacent paragraph merge ===")
    final: List[MockFinding] = []
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
            
            if (abs(other_para - para_idx) <= 3 and 
                other.category_id == f.category_id and
                quotes_intersect(f_quote, (other.evidence_quote or "").strip())):
                print(f"  ⚠️  Merging: para {para_idx} ({f.category_name}) and para {other_para} ({other.category_name})")
                adjacent_group.append(other)
                processed.add(id(other))
        
        if len(adjacent_group) > 1:
            print(f"  → Keeping merged finding (group of {len(adjacent_group)})")
            final.append(adjacent_group[0])  # Simplified - just keep first
        else:
            final.append(f)
    
    print(f"\nAfter third pass: {len(final)} findings")
    return final

# Test with 3 findings from example contract
findings = [
    MockFinding('f1', 'uncapped_liability', 'Uncapped Liability',
                'Company and Distributor agree to indemnify',
                {'paragraph_index': 5}),
    MockFinding('f2', 'termination_for_convenience', 'Termination For Convenience',
                'any price changes, other than those based',
                {'paragraph_index': 8}),
    MockFinding('f3', 'termination_for_convenience', 'Termination For Convenience',
                'If Company terminates the Agreement without cause and for reasons other than Distributor\'s failure to meet its minimum expectations',
                {'paragraph_index': 12}),
]

print("=" * 80)
print("TEST: Deduplication of 3 findings")
print("=" * 80)

print(f"\nInput: {len(findings)} findings")
for i, f in enumerate(findings, 1):
    print(f"  {i}. {f.category_name} (para {f.location.get('paragraph_index')}): {f.evidence_quote[:70]}...")

result = dedupe_findings_simple(findings)

print("\n" + "=" * 80)
print(f"RESULT: {len(result)} findings (expected: 3)")
print("=" * 80)

if len(result) != 3:
    print(f"⚠️  PROBLEM: Lost {3 - len(result)} finding(s)!")
    input_ids = {f.finding_id for f in findings}
    output_ids = {f.finding_id for f in result}
    lost_ids = input_ids - output_ids
    if lost_ids:
        print(f"\nLost finding IDs: {lost_ids}")
        for f in findings:
            if f.finding_id in lost_ids:
                print(f"  - {f.category_name}: {f.evidence_quote[:60]}...")
else:
    print("✓ All 3 findings preserved!")

print("\nOutput findings:")
for i, f in enumerate(result, 1):
    print(f"  {i}. {f.category_name} (para {f.location.get('paragraph_index')}): {f.evidence_quote[:70]}...")

# Check quote intersections
print("\n" + "=" * 80)
print("Quote intersection analysis:")
print("=" * 80)

f1, f2, f3 = findings
print(f"\nFinding 1 vs 2:")
print(f"  Para distance: {abs(5 - 8)} (merge if <= 3)")
print(f"  Same category: {f1.category_id == f2.category_id}")
print(f"  Quotes intersect: {quotes_intersect(f1.evidence_quote, f2.evidence_quote)}")

print(f"\nFinding 2 vs 3:")
print(f"  Para distance: {abs(8 - 12)} (merge if <= 3)")
print(f"  Same category: {f2.category_id == f3.category_id}")
print(f"  Quotes intersect: {quotes_intersect(f2.evidence_quote, f3.evidence_quote)}")

print(f"\nFinding 1 vs 3:")
print(f"  Para distance: {abs(5 - 12)} (merge if <= 3)")
print(f"  Same category: {f1.category_id == f3.category_id}")
print(f"  Quotes intersect: {quotes_intersect(f1.evidence_quote, f3.evidence_quote)}")
