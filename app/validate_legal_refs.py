"""
Validation of legal references to filter out non-existent or invalid law citations.

This module validates legal references against known UK statutes and common law.
Invalid references are filtered out to prevent misleading information.
"""

import re
from typing import List


# Whitelist of valid UK statutes (for tenant/consumer protection context)
# 
# Sources:
# - Explicitly mentioned in app/prompts.py (line 460) as typical UK framework references
# - legislation.gov.uk (official UK legislation database)
# - UK government housing guidance
# - Legal research databases
# - Example contract categories (Termination, MFN, Uncapped Liability) - app/prompts.py lines 88-91, 151-162
#
# NOTE: This list should be reviewed and updated by legal experts to ensure completeness
# and accuracy for your specific use case.
# To update: run scripts/update_uk_tenancy_statutes.py
VALID_UK_STATUTES = {
    # Consumer protection (used for example contract categories: Termination, MFN, Uncapped Liability)
    "Consumer Rights Act 2015",
    "Consumer Rights Act",
    "CRA 2015",
    "Consumer Rights Act 2015, sections 62-68",
    "Consumer Rights Act 2015, sections",
    "Consumer Rights Act 2015, section",
    
    # Housing Acts (major legislation)
    "Housing Act 1985",
    "Housing Act 1988",
    "Housing Act 1996",
    "Housing Act 2004",
    "Housing Act",  # generic form
    
    # Landlord and Tenant Acts
    "Landlord and Tenant Act 1927",
    "Landlord and Tenant Act 1954",
    "Landlord and Tenant Act 1985",
    "Landlord and Tenant Act 1987",
    "Landlord and Tenant Act",  # generic form
    
    # Protection and Eviction
    "Protection from Eviction Act 1977",
    "Protection from Eviction Act",
    
    # Tenant Fees and Rights
    "Tenant Fees Act 2019",
    "Tenant Fees Act",
    "Homes (Fitness for Human Habitation) Act 2018",
    "Homes Act 2018",
    "Homes (Fitness for Human Habitation) Act",
    
    # Unfair Terms
    "Unfair Contract Terms Act 1977",
    "UCTA 1977",
    
    # Rent Acts (historical but still referenced)
    "Rent Act 1977",
    "Rent Act",
    
    # Leasehold Reform
    "Leasehold Reform Act 1967",
    "Leasehold Reform, Housing and Urban Development Act 1993",
    "Leasehold Reform Act",
    
    # Common Law
    "UK common law",
    "English common law",
    "common law",
    
    # Additional relevant statutes
    "Equality Act 2010",  # Discrimination in housing
    "Data Protection Act 2018",  # Tenant data
    "Gas Safety (Installation and Use) Regulations 1998",
    "Electrical Safety Standards in the Private Rented Sector (England) Regulations 2020",
    "Smoke and Carbon Monoxide Alarm (England) Regulations 2015",
    "Energy Efficiency (Private Rented Property) (England and Wales) Regulations 2015",
}

# Patterns for valid legal reference formats
VALID_PATTERNS = [
    # "Act Name YYYY" or "Act Name YYYY, sections X-Y" (e.g., "Consumer Rights Act 2015, sections 62-68")
    # Also supports parenthetical notes like "(fairness test)"
    re.compile(r"^[A-Z][A-Za-z\s\(\)]+Act\s+\d{4}(?:\s*,\s*sections?\s+\d+(?:[-\s]+\d+)?(?:\s*\([^)]+\))?)?$", re.IGNORECASE),
    # "Act Name, sections X-Y" (e.g., "Consumer Rights Act, sections 62-68")
    # Also supports parenthetical notes like "(fairness test)"
    re.compile(r"^[A-Z][A-Za-z\s\(\)]+Act(?:\s+\d{4})?(?:\s*,\s*sections?\s+\d+(?:[-\s]+\d+)?(?:\s*\([^)]+\))?)?$", re.IGNORECASE),
    # "UK common law" or "English common law"
    re.compile(r"^(UK|English)\s+common\s+law", re.IGNORECASE),
    # "common law on [topic]"
    re.compile(r"^common\s+law\s+on\s+", re.IGNORECASE),
    # Abbreviations like "CRA 2015"
    re.compile(r"^[A-Z]{2,}\s+\d{4}$"),
]


def is_valid_legal_reference(ref: str) -> bool:
    """
    Check if a legal reference is valid.
    
    A reference is considered valid if:
    1. It matches a known UK statute name (case-insensitive, partial match)
    2. It matches a valid format pattern
    3. It contains "common law" (for UK common law references)
    
    Args:
        ref: Legal reference string (e.g., "Consumer Rights Act 2015, sections 62-68")
    
    Returns:
        True if the reference appears valid, False otherwise
    """
    if not ref or not ref.strip():
        return False
    
    ref_clean = ref.strip()
    
    # Check against whitelist (case-insensitive, partial match)
    ref_lower = ref_clean.lower()
    for valid_statute in VALID_UK_STATUTES:
        if valid_statute.lower() in ref_lower:
            return True
    
    # Check against format patterns
    for pattern in VALID_PATTERNS:
        if pattern.search(ref_clean):
            return True
    
    # If it contains "common law" and mentions UK/English context, likely valid
    # Used for example contract categories (e.g., "UK common law on penalty clauses")
    if "common law" in ref_lower and ("uk" in ref_lower or "english" in ref_lower or "penalty" in ref_lower or "unfair" in ref_lower):
        return True
    
    # Check for "Consumer Rights Act 2015, sections X-Y" pattern (used in example contract)
    # This pattern is already covered by VALID_PATTERNS, but adding explicit check for clarity
    if "consumer rights act 2015" in ref_lower and "section" in ref_lower:
        return True
    
    return False


def filter_valid_legal_references(legal_refs: List[str]) -> List[str]:
    """
    Filter legal references to keep only valid ones.
    
    Args:
        legal_refs: List of legal reference strings
    
    Returns:
        List of valid legal references
    """
    if not legal_refs:
        return []
    
    valid_refs = []
    for ref in legal_refs:
        if is_valid_legal_reference(ref):
            valid_refs.append(ref)
    
    return valid_refs
