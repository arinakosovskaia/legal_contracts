#!/usr/bin/env python3
"""
Test script to debug why only 2 findings are found instead of 3.
Simulates the full processing pipeline.
"""

import sys
import re
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.models import Finding, RiskAssessment
from app.aggregate import dedupe_findings

# Simulate 3 findings that should be found in the example contract
findings = [
    Finding(
        finding_id="window1_f1",
        category_id="uncapped_liability",
        category_name="Uncapped Liability",
        severity="high",
        confidence=0.9,
        is_unfair=True,
        issue_title="Uncapped Liability",
        what_is_wrong="Company and Distributor agree to indemnify",
        explanation="This clause creates unlimited liability",
        recommendation="Limit liability to reasonable amounts",
        evidence_quote="Company and Distributor agree to indemnify",
        legal_references=["Consumer Rights Act 2015"],
        possible_consequences="Unlimited financial exposure",
        risk_assessment=RiskAssessment(severity_of_consequences=2, degree_of_legal_violation=2),
        consequences_category="Unenforceable",
        risk_category="Penalty",
        revised_clause="",
        revision_explanation="",
        suggested_follow_up="",
        paragraph_bbox=None,
        location={"page": 1, "paragraph_index": 5},
    ),
    Finding(
        finding_id="window2_f1",
        category_id="termination_for_convenience",
        category_name="Termination For Convenience",
        severity="medium",
        confidence=0.85,
        is_unfair=True,
        issue_title="Termination For Convenience",
        what_is_wrong="any price changes, other than those based",
        explanation="Allows price changes without notice",
        recommendation="Require reasonable notice for price changes",
        evidence_quote="any price changes, other than those based",
        legal_references=["Consumer Rights Act 2015"],
        possible_consequences="Unexpected price increases",
        risk_assessment=RiskAssessment(severity_of_consequences=1, degree_of_legal_violation=1),
        consequences_category="Voidable",
        risk_category="Nothing",
        revised_clause="",
        revision_explanation="",
        suggested_follow_up="",
        paragraph_bbox=None,
        location={"page": 1, "paragraph_index": 8},
    ),
    Finding(
        finding_id="window3_f1",
        category_id="termination_for_convenience",
        category_name="Termination For Convenience",
        severity="high",
        confidence=0.9,
        is_unfair=True,
        issue_title="Termination For Convenience",
        what_is_wrong="If Company terminates the Agreement without cause and for reasons other than Distributor's failure to meet its minimum expectations",
        explanation="Allows termination without cause",
        recommendation="Require cause for termination",
        evidence_quote="If Company terminates the Agreement without cause and for reasons other than Distributor's failure to meet its minimum expectations",
        legal_references=["Consumer Rights Act 2015"],
        possible_consequences="Unfair termination",
        risk_assessment=RiskAssessment(severity_of_consequences=2, degree_of_legal_violation=2),
        consequences_category="Unenforceable",
        risk_category="Penalty",
        revised_clause="",
        revision_explanation="",
        suggested_follow_up="",
        paragraph_bbox=None,
        location={"page": 1, "paragraph_index": 12},
    ),
]

print("=" * 80)
print("TEST: Processing 3 findings through deduplication")
print("=" * 80)

print(f"\nInput: {len(findings)} findings")
for i, f in enumerate(findings, 1):
    print(f"\n{i}. {f.category_name}")
    print(f"   Para: {f.location.get('paragraph_index')}")
    print(f"   Quote: {f.evidence_quote[:80]}...")
    print(f"   Finding ID: {f.finding_id}")

print("\n" + "=" * 80)
print("Running dedupe_findings...")
print("=" * 80)

result = dedupe_findings(findings)

print(f"\nOutput: {len(result)} findings")
if len(result) != len(findings):
    print(f"⚠️  WARNING: Lost {len(findings) - len(result)} finding(s)!")
    
    # Find which findings were lost
    input_ids = {f.finding_id for f in findings}
    output_ids = {f.finding_id for f in result}
    lost_ids = input_ids - output_ids
    
    if lost_ids:
        print(f"\nLost finding IDs: {lost_ids}")
        for f in findings:
            if f.finding_id in lost_ids:
                print(f"  - {f.category_name}: {f.evidence_quote[:60]}...")
else:
    print("✓ All findings preserved!")

for i, f in enumerate(result, 1):
    print(f"\n{i}. {f.category_name}")
    print(f"   Para: {f.location.get('paragraph_index')}")
    print(f"   Quote: {f.evidence_quote[:80]}...")
    print(f"   Finding ID: {f.finding_id}")
    if len(f.evidence_quote) > 100:
        print(f"   (Quote appears to be merged/combined)")

print("\n" + "=" * 80)
print("Checking quote intersections...")
print("=" * 80)

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

# Check all pairs
for i, f1 in enumerate(findings):
    for j, f2 in enumerate(findings[i+1:], i+1):
        para_dist = abs(f1.location.get('paragraph_index', 0) - f2.location.get('paragraph_index', 0))
        same_cat = f1.category_id == f2.category_id
        intersect = quotes_intersect(f1.evidence_quote, f2.evidence_quote)
        
        print(f"\nFinding {i+1} vs {j+1}:")
        print(f"  Categories: {f1.category_name} vs {f2.category_name} (same: {same_cat})")
        print(f"  Para distance: {para_dist} (merge if <= 3)")
        print(f"  Quotes intersect: {intersect}")
        if para_dist <= 3 and same_cat and intersect:
            print(f"  ⚠️  WOULD BE MERGED!")
        elif para_dist <= 3 and same_cat:
            print(f"  → Same category, adjacent, but quotes DON'T intersect - should NOT merge")
        else:
            print(f"  ✓ Should NOT merge")

print("\n" + "=" * 80)
print("Test complete!")
print("=" * 80)
