"""
LLM Prompts for Legal Contract Analysis

This file contains all prompt templates and instructions for the LLM.
Edit this file to modify how the model analyzes contracts.

For legal experts: Focus on the sections marked with "EDIT HERE" comments.

QUICK REFERENCE - Where to Edit What:
=====================================

TWO-STAGE MODE (UK Red Flags - Production):
- Router system prompt: get_uk_router_system_prompt() - line ~301
- Router user prompt: get_uk_router_user_prompt_template() - line ~316
- Classifier system prompt: get_uk_redflag_system_prompt() - line ~399
- Classifier user prompt: get_uk_redflag_user_prompt_template() - line ~416
- Few-shot examples: data/fewshot_uk_redflags.txt (plain text file)
- Category definitions: data/category_definitions_new.csv

LEGACY MODE (Single-Stage - Debug):
- System prompt: get_redflag_system_prompt() - line ~34
- User prompt: get_redflag_user_prompt_template() - line ~42
- Few-shot examples: Built from CUAD dataset (see app/fewshot.py)
- Category definitions: data/category_descriptions.csv

See PROMPTS_README.md for detailed editing instructions.
"""

import json
from typing import Any


def get_redflag_system_prompt() -> str:
    """
    System prompt for the red flag classifier (legacy single-stage mode).
    This sets the role and behavior of the LLM.
    
    EDIT HERE: Modify the role description or behavior instructions.
    """
    return "You are a legal clause classifier for red flag clause types. Output STRICT JSON only."


def get_redflag_user_prompt_template() -> str:
    """
    Main user prompt template for red flag classification.
    
    This is the core instruction that tells the LLM:
    - What categories to look for
    - What to return (quote, reasoning, legal analysis)
    - How to format the output
    
    EDIT HERE: Modify the instructions, rules, or output schema as needed.
    """
    return """You are a legal clause classifier for "red flag" clause types.

Knowing only the provided TEXT CHUNK, identify whether it contains any of these categories:
{categories_list}

CURRENT DOCUMENT LOCATION (for the chunk you must classify):
CONTEXT PREFIX: {section_path}

You must return:
1) the category (or categories)
2) verbatim evidence quoted from the provided TEXT CHUNK
3) short reasoning grounded only in the evidence
4) an unfairness analysis under UK law (when relevant)

Rules
- Evidence MUST be copied verbatim from the provided TEXT CHUNK. Do not paraphrase.
- Return ALL findings that match any of the categories. Do not skip findings just because they seem less important - if a category matches, include it.
- If nothing matches, return {{"findings": []}}.
- If you suspect a category but cannot quote clear evidence from the chunk, do NOT guess; return no finding for that category.
- Keep reasoning brief (1–3 bullets) and tied directly to the evidence.
- Explain unfairness under UK law. Refer where relevant to the Consumer Rights Act 2015 and UK common law on penalty clauses.
- For fair clauses, return no finding rather than marking them as unfair.
- When you identify a clause that matches a red flag category, return a finding even if you are uncertain about its fairness - the analysis will help determine fairness.
- Provide a short suggested follow-up action for the tenant.
- IMPORTANT: For "Most Favored Nation" category, look for clauses that require uniform/equal application of terms (e.g., "uniformly applied", "equally applied", "applied to all") to all parties, not just explicit "most favored nation" language. This includes clauses requiring that price changes or other terms be applied uniformly to all distributors, customers, or parties.

Output JSON schema:
{{
  "findings": [
    {{
      "category": "<one of the categories above>",
      "quote": "<verbatim quote from TEXT CHUNK>",
      "reasoning": ["<bullet>"],
      "is_unfair": true,
      "explanation": "Explain why the clause is unfair under UK law. Refer where relevant to the Consumer Rights Act 2015 and UK common law on penalty clauses (genuine pre-estimate of loss, proportionality, legitimate interest).",
      "legal_references": [
        "Consumer Rights Act 2015, section(s) ...",
        "UK common law on penalty clauses"
      ],
      "possible_consequences": "Describe financial, legal, and enforceability consequences.",
      "risk_assessment": {{
        "severity_of_consequences": 0,
        "degree_of_legal_violation": 0
      }},
      "consequences_category": "Invalid | Unenforceable | Void | Voidable | Nothing",
      "risk_category": "Penalty | Criminal liability | Failure of the other party to perform obligations on time | Nothing",
      "recommended_revision": {{
        "revised_clause": "Provide a rewritten, compliant clause.",
        "revision_explanation": "Explain briefly how the revision resolves the unfairness."
      }},
      "suggested_follow_up": "Suggested follow-up steps for the tenant."
    }}
  ]
}}

========================
CATEGORY DEFINITIONS (from category_descriptions.csv)
========================
{category_definitions}

{fewshot_section}

========================
NOW CLASSIFY THIS INPUT
========================
{section_prefix}TEXT CHUNK:
{text_chunk}
OUTPUT JSON:
"""


def build_redflag_fewshot_example(
    category_name: str,
    clause_text: str,
    evidence: str,
    question: str,
) -> str:
    """
    Build a single few-shot example for a category.
    
    EDIT HERE: Modify how few-shot examples are formatted.
    This affects what examples the LLM sees and learns from.
    
    Args:
        category_name: Name of the red flag category
        clause_text: The clause/snippet text from CUAD
        evidence: The evidence span (answer from CUAD QA)
        question: The question from CUAD QA
    
    Returns:
        Formatted few-shot example string
    """
    # EDIT HERE: Customize the reasoning, explanation, and other fields for each category
    # These are template values - you can make them category-specific
    
    if category_name == "Termination For Convenience":
        reasoning = ["Evidence shows one party can terminate without cause.", "No reciprocal right for the other party."]
        explanation = "The clause allows termination without cause (for convenience) without a reciprocal right, which may be deemed unfair under the Consumer Rights Act 2015 as it creates an imbalance in the contract."
        legal_refs = ["Consumer Rights Act 2015, sections 62-68 (fairness test)", "UK common law on penalty clauses"]
        possible = "The clause may be challenged as unfair and lead to renegotiation, disputes, or non-enforcement."
        risk_category = "Penalty"
        revised = "Add a reciprocal termination right for both parties, or limit termination to specific circumstances with notice periods."
        revision_expl = "Ensures balance and fairness by giving both parties similar rights."
        follow_up = "Request the counterparty to justify the termination clause and propose a balanced revision."
    
    elif category_name == "Uncapped Liability":
        reasoning = ["Evidence contains 'any and all' language without a cap on indemnification.", "This creates uncapped exposure for the indemnifying party."]
        explanation = "The clause creates an uncapped indemnity obligation without a cap, which may be disproportionate to the risk and not a genuine pre-estimate of loss. Under the Consumer Rights Act 2015, such terms may be deemed unfair if they are not transparent and are one-sided."
        legal_refs = ["Consumer Rights Act 2015, sections 62-68 (fairness test)", "UK common law on penalty clauses (genuine pre-estimate of loss, proportionality, legitimate interest)"]
        possible = "May be challenged as unfair and lead to renegotiation, disputes, or non-enforcement."
        risk_category = "Penalty"
        revised = "Replace with a clause that limits liability to a reasonable cap, such as the total consideration paid under the agreement, or a specific monetary amount, and ensures the cap is a genuine pre-estimate of loss."
        revision_expl = "Adds a cap on the indemnity obligation to ensure it is proportional and reasonable."
        follow_up = "Request the counterparty to revise the indemnity clause to include a reasonable liability cap and provide justification for the cap amount."
    
    elif category_name == "Irrevocable Or Perpetual License":
        reasoning = ["Evidence shows license is irrevocable or perpetual.", "No termination mechanism or time limit specified."]
        explanation = "An irrevocable or perpetual license without termination rights may be unfair as it creates an indefinite obligation. Under UK law, such terms may be challenged if they are not transparent or create an imbalance."
        legal_refs = ["Consumer Rights Act 2015, sections 62-68 (fairness test)"]
        possible = "May be challenged as unfair and lead to renegotiation or non-enforcement."
        risk_category = "Nothing"
        revised = "Add a termination clause allowing either party to terminate with reasonable notice, or specify a fixed term with renewal options."
        revision_expl = "Provides flexibility and balance by allowing termination under reasonable conditions."
        follow_up = "Request the counterparty to add termination provisions or limit the license term."
    
    elif category_name == "Most Favored Nation":
        reasoning = [
            "Evidence contains 'most favored nation', 'uniformly applied', 'equally applied', or similar language requiring equal treatment.",
            "This creates ongoing obligations to match terms given to others or to apply changes uniformly to all parties."
        ]
        explanation = "Most favored nation clauses (including clauses requiring uniform application of price changes or terms to all parties) may be unfair if they create disproportionate ongoing obligations or prevent individualized negotiations. Under the Consumer Rights Act 2015, such terms may be challenged if they are not transparent or create an imbalance."
        legal_refs = ["Consumer Rights Act 2015, sections 62-68 (fairness test)"]
        possible = "May be challenged as unfair and lead to renegotiation."
        risk_category = "Nothing"
        revised = "Limit the MFN clause to specific terms (e.g., pricing only) and add reasonable exceptions or time limits. Alternatively, allow for individualized negotiations based on specific circumstances."
        revision_expl = "Narrows the scope and reduces the burden of ongoing obligations, or allows for fair individualized treatment."
        follow_up = "Request the counterparty to narrow the scope of the MFN clause or add exceptions for individualized treatment."
    
    elif category_name == "Audit Rights":
        reasoning = ["Evidence shows broad audit rights without reasonable limitations.", "May create disproportionate burden on the audited party."]
        explanation = "Unlimited or overly broad audit rights may be unfair if they create a disproportionate burden. Under the Consumer Rights Act 2015, such terms may be challenged if they are not transparent or create an imbalance."
        legal_refs = ["Consumer Rights Act 2015, sections 62-68 (fairness test)"]
        possible = "May be challenged as unfair and lead to renegotiation."
        risk_category = "Nothing"
        revised = "Limit audit rights to reasonable frequency (e.g., once per year), reasonable notice periods, and scope limitations."
        revision_expl = "Adds reasonable limitations to prevent abuse and ensure proportionality."
        follow_up = "Request the counterparty to add reasonable limitations to audit rights."
    
    elif category_name == "Ip Ownership Assignment":
        reasoning = ["Evidence shows assignment of IP ownership without fair consideration.", "May unfairly transfer IP rights."]
        explanation = "IP ownership assignment clauses may be unfair if they transfer IP without fair consideration or create an imbalance. Under the Consumer Rights Act 2015, such terms may be challenged if they are not transparent."
        legal_refs = ["Consumer Rights Act 2015, sections 62-68 (fairness test)"]
        possible = "May be challenged as unfair and lead to renegotiation or non-enforcement."
        risk_category = "Nothing"
        revised = "Limit IP assignment to IP created specifically for the contract, or add fair consideration and exceptions for pre-existing IP."
        revision_expl = "Ensures fair treatment and protects pre-existing IP rights."
        follow_up = "Request the counterparty to narrow the IP assignment scope or add exceptions."
    
    else:
        # Default template for unknown categories
        reasoning = ["Evidence indicates potential red flag.", "Review required for fairness assessment."]
        explanation = "This clause may be unfair under UK law. Review for compliance with the Consumer Rights Act 2015."
        legal_refs = ["Consumer Rights Act 2015, sections 62-68 (fairness test)"]
        possible = "May be challenged as unfair."
        risk_category = "Nothing"
        revised = "Revise to ensure fairness and compliance with UK law."
        revision_expl = "Ensures compliance with UK consumer protection law."
        follow_up = "Request the counterparty to revise the clause for fairness."
    
    return f"""
EXAMPLE: {category_name}
TEXT CHUNK:
{clause_text}
OUTPUT:
{{
  "findings": [
    {{
      "category": "{category_name}",
      "quote": "{evidence}",
      "reasoning": {json.dumps(reasoning, ensure_ascii=False)},
      "is_unfair": true,
      "explanation": "{explanation}",
      "legal_references": {json.dumps(legal_refs, ensure_ascii=False)},
      "possible_consequences": "{possible}",
      "risk_assessment": {{
        "severity_of_consequences": 2,
        "degree_of_legal_violation": 2
      }},
      "consequences_category": "Unenforceable",
      "risk_category": "{risk_category}",
      "recommended_revision": {{
        "revised_clause": "{revised}",
        "revision_explanation": "{revision_expl}"
      }},
      "suggested_follow_up": "{follow_up}"
    }}
  ]
}}
"""


def format_redflag_prompt(
    *,
    categories_list: str,
    section_path: str,
    category_definitions: str,
    fewshot_section: str,
    text_chunk: str,
) -> tuple[str, dict[str, Any]]:
    """
    Format the complete red flag classification prompt.
    
    This function combines all parts of the prompt:
    - Categories list
    - Instructions and rules
    - Category definitions
    - Few-shot examples
    - The actual text chunk to classify
    
    EDIT HERE: Modify the overall structure or add/remove sections.
    
    Returns:
        Tuple of (user_prompt, sections_dict) for debugging
    """
    section_prefix = f"CONTEXT PREFIX: {section_path}\n\n" if section_path else ""
    
    user_prompt = get_redflag_user_prompt_template().format(
        categories_list=categories_list,
        section_path=section_path or "(none)",
        category_definitions=category_definitions,
        fewshot_section=fewshot_section,
        section_prefix=section_prefix,
        text_chunk=text_chunk,
    )
    
    sections = {
        "context_prefix": section_path or "",
        "text_chunk": text_chunk,
        "category_definitions": category_definitions,
        "fewshot_examples": fewshot_section,
        "fewshot_available": bool(fewshot_section and fewshot_section.strip()),
    }
    
    return user_prompt, sections


def get_uk_router_system_prompt() -> str:
    """
    System prompt for the UK router (first stage of two-stage classification).
    This sets the role and behavior of the LLM for category routing.
    
    EDIT HERE: Modify the role description or behavior instructions.
    """
    return (
        'You are a high-recall clause router for UK residential tenancy "red flag" categories.\n'
        "Your goal is to select ALL categories that might apply, even weakly, based ONLY on the TEXT CHUNK.\n"
        "Be inclusive (high recall), but do not hallucinate: every selected category must be supported by at least one "
        "verbatim trigger quote from the chunk."
    )


def get_uk_router_user_prompt_template() -> str:
    """
    User prompt template for the UK router (first stage).
    
    This prompt tells the router LLM:
    - What categories to consider
    - How to select categories (high recall)
    - What to return (category, confidence, trigger quotes, rationale)
    
    EDIT HERE: Modify the instructions, rules, or output schema for the router.
    """
    return """CANDIDATE CATEGORIES (select any that might apply):
{categories_list}

CATEGORY DEFINITIONS:
{category_definitions}

CURRENT DOCUMENT LOCATION (context only; not evidence):
{section_prefix}CONTEXT PREFIX: {section_path}

TASK:
From the TEXT CHUNK only, select ALL categories that could plausibly match (even if uncertain).
For each selected category:
- provide confidence: "low" | "medium" | "high"
- provide 1-3 short trigger_quotes copied VERBATIM from the TEXT CHUNK that caused you to include the category
- provide a one-sentence rationale tied to the trigger_quotes (no legal analysis here)

RULES:
- High recall: include categories even if you are only mildly suspicious.
- No guessing: if you cannot quote any supporting trigger text verbatim, do NOT include the category.
- Trigger quotes must be short (max ~20 words each) and copied exactly from the TEXT CHUNK.
- If nothing matches at all, return {{"candidate_categories": []}}.
- Do not output anything except the JSON object.

OUTPUT JSON ONLY:
Return STRICT JSON only. No Markdown. No extra keys. Must match schema exactly.
{{
  "candidate_categories": [
    {{
      "category": "<one of the categories above>",
      "confidence": "low|medium|high",
      "trigger_quotes": ["<verbatim quote>", "..."],
      "rationale": "<one sentence>"
    }}
  ]
}}

TEXT CHUNK:
{text_chunk}

OUTPUT JSON:
"""


def format_uk_router_prompt(
    *,
    categories_list: str,
    section_path: str,
    category_definitions: str,
    text_chunk: str,
) -> tuple[str, dict[str, Any]]:
    """
    Format the complete UK router prompt (first stage).
    
    This function assembles the prompt from the template.
    To modify the prompt content, edit get_uk_router_user_prompt_template() above.
    """
    section_prefix = ""
    user_prompt = get_uk_router_user_prompt_template().format(
        categories_list=categories_list,
        category_definitions=category_definitions,
        section_prefix=section_prefix,
        section_path=section_path or "(none)",
        text_chunk=text_chunk,
    )
    sections = {
        "context_prefix": section_path or "",
        "text_chunk": text_chunk,
        "category_definitions": category_definitions,
    }
    return user_prompt, sections


def get_uk_redflag_system_prompt() -> str:
    """
    System prompt for the UK red flag classifier (second stage of two-stage classification).
    This sets the role and behavior of the LLM for detailed analysis.
    
    EDIT HERE: Modify the role description or behavior instructions.
    """
    return (
        'You are a UK residential tenancy contract "red flag" detector for tenants.\n'
        "Your job is to identify potentially unlawful or unfair terms based ONLY on the provided TEXT CHUNK.\n"
        "You must be conservative: do not guess, do not invent missing context, and do not use knowledge outside the "
        "chunk for evidence.\n"
        "When you flag a red flag, you must quote verbatim evidence from the chunk and explain why it is problematic "
        "under UK law principles."
    )


def get_uk_redflag_user_prompt_template() -> str:
    """
    User prompt template for the UK red flag classifier (second stage).
    
    This is the core instruction that tells the classifier LLM:
    - What categories to analyze (only those selected by the router)
    - What to return (quote, reasoning, legal analysis, risk assessment, etc.)
    - The rules for classification and false-positive defense
    - Risk scoring rubric
    - Output JSON schema
    
    EDIT HERE: Modify the instructions, rules, risk scoring, or output schema.
    """
    return """TASK
Knowing only the provided TEXT CHUNK, identify whether it contains any unfair/unlawful tenant-facing terms that match one or more of these categories:
{categories_list}

CURRENT DOCUMENT LOCATION (context only; NOT evidence):
{section_prefix}CONTEXT PREFIX: {section_path}

OUTPUT REQUIREMENTS
Return STRICT JSON ONLY. No Markdown. No extra keys. Must match schema exactly.
Return ALL findings that match any of the categories. Do not skip findings - if a category matches, include it.
If nothing matches, return {{"findings": []}}.

DEFINITIONS
Use the category definitions below to decide what qualifies as a match and to avoid false positives:
{category_definitions}

FEW-SHOT EXAMPLES
(Use these patterns to guide decisions and to avoid false positives.)
{fewshot_section}

DECISION RULES (VERY IMPORTANT)
1) Evidence MUST be copied verbatim from the provided TEXT CHUNK. Do not paraphrase evidence.
2) Only flag a category if the chunk contains clear evidence of an UNFAIR or UNLAWFUL tenant-facing term.
   - If the clause is clearly fair/standard-compliant, return no finding (even if it resembles a category).
   - If it's suspicious but you cannot quote clear evidence, do NOT guess; return no finding.
3) Treat CONTEXT PREFIX and category definitions as guidance only; they are NOT evidence.
4) Prefer fewer, higher-confidence findings over many weak ones.
5) False-positive defense:
   - Do NOT flag emergency-only landlord access clauses that require genuine emergencies and/or reasonable notice.
   - Do NOT flag lawful notice clauses that explicitly comply with written notice/minimum notice requirements.
   - Do NOT flag permitted payments/fees when the clause explicitly says "only as permitted by law" and does not impose extra fees.
   - Do NOT flag service charges when explicitly limited to "reasonable" / "in accordance with law" and challenge mechanisms are preserved.
6) Legal analysis must be grounded in the chunk + general UK framework (typical references include CRA 2015, Tenant Fees Act 2019, Housing Act 2004 deposit protection, Protection from Eviction Act 1977, Landlord and Tenant Act 1985/1987, Homes (Fitness for Human Habitation) Act 2018). Only cite statutes that are plausibly relevant to the clause type.

RISK SCORING RUBRIC (use integers)
severity_of_consequences:
0 = no meaningful tenant risk (should usually correspond to no finding)
1 = potentially unfair; mainly civil dispute; likely negotiable/voidable
2 = likely unenforceable/void OR could trigger civil penalties/serious tenant harm
3 = potential criminal liability / severe illegality (e.g., unlawful eviction/harassment, illegal entry framed as a right, safety/licensing evasion)

degree_of_legal_violation:
0 = not a violation / compliant
1 = ambiguous / depends on facts
2 = likely violation or unfair term
3 = clear violation (term purports to override mandatory law or creates plainly unlawful power)

ENUMS (must use exactly one value from each list)
consequences_category: "Invalid" | "Unenforceable" | "Void" | "Voidable" | "Nothing"
risk_category: "Penalty" | "Criminal liability" | "Failure of the other party to perform obligations on time" | "Nothing"

OUTPUT JSON SCHEMA
{{
  "findings": [
    {{
      "category": "<one of the categories above>",
      "quote": "<verbatim quote from TEXT CHUNK>",
      "reasoning": ["<1-3 short bullets tied directly to the quote>"],
      "is_unfair": true,
      "explanation": "Explain briefly why the quoted term may be unfair/unlawful under UK law principles relevant to tenants.",
      "legal_references": [
        "Name the most relevant UK statute/regime (sections only if confident)."
      ],
      "possible_consequences": "Practical impact on the tenant (money/rights/eviction risk/enforceability).",
      "risk_assessment": {{
        "severity_of_consequences": 0,
        "degree_of_legal_violation": 0
      }},
      "consequences_category": "Invalid | Unenforceable | Void | Voidable | Nothing",
      "risk_category": "Penalty | Criminal liability | Failure of the other party to perform obligations on time | Nothing",
      "recommended_revision": {{
        "revised_clause": "Rewrite the clause to be compliant and tenant-fair while preserving legitimate landlord needs.",
        "revision_explanation": "One short sentence explaining what changed and why it fixes the issue."
      }},
      "suggested_follow_up": "Concrete next step(s) for the tenant (ask to amend, request certificate/scheme details, contact council/Trading Standards, get legal advice)."
    }}
  ]
}}

========================
NOW CLASSIFY THIS INPUT
========================
TEXT CHUNK:
{text_chunk}

OUTPUT JSON:
"""


def format_uk_redflag_prompt(
    *,
    categories_list: str,
    section_path: str,
    category_definitions: str,
    fewshot_section: str,
    text_chunk: str,
) -> tuple[str, dict[str, Any]]:
    """
    Format the complete UK classifier prompt (second stage).
    
    This function assembles the prompt from the template.
    To modify the prompt content, edit get_uk_redflag_user_prompt_template() above.
    To modify few-shot examples, edit data/fewshot_uk_redflags.txt (plain text file).
    """
    section_prefix = ""
    user_prompt = get_uk_redflag_user_prompt_template().format(
        categories_list=categories_list,
        section_path=section_path or "(none)",
        category_definitions=category_definitions,
        fewshot_section=fewshot_section,
        section_prefix=section_prefix,
        text_chunk=text_chunk,
    )
    sections = {
        "context_prefix": section_path or "",
        "text_chunk": text_chunk,
        "category_definitions": category_definitions,
        "fewshot_examples": fewshot_section,
        "fewshot_available": bool(fewshot_section and fewshot_section.strip()),
    }
    return user_prompt, sections
