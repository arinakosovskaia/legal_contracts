from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Type, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from .models import (
    Paragraph,
    RedFlagChunkOutput,
    RedFlagFindingLLM,
    Stage1Category,
    Stage1Output,
    Stage1Prob,
    Stage2Output,
)
from .categories import load_categories_from_csv, write_categories_json

T = TypeVar("T", bound=BaseModel)


_JSON_START = re.compile(r"\{", re.MULTILINE)


_MANY_NEWLINES = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+\n")
# Standalone page number lines: "- 11 -" or "Page - 11 -"
_PAGE_NUMBER_LINE = re.compile(r"^\s*(?:Page\s*)?-\s*\d+\s*-\s*$", re.MULTILINE | re.IGNORECASE)
# "Page -9-" or "Page - 9 -" style fragments (often merged with continuation across page break)
_PAGE_NUMBER_FRAGMENT = re.compile(r"Page\s*-\s*\d+\s*-\s*", re.IGNORECASE)
# Control chars (except \n \t) and black squares → space (display-safe; no meaning change)
_CONTROL_AND_BLACK = re.compile(r"[\u0000-\u0008\u000b\u000e-\u001f\u007f\u25A0-\u25FF]")


def normalize_llm_text(text: str) -> str:
    """
    Normalize noisy PDF-ish text for LLM, display, and quote matching.

    Preserves meaning: only whitespace, line endings, and PDF artefacts are changed.
    Use this same normalized text everywhere (LLM input, PDF, frontend) so quote
    matching is exact and we avoid raw vs normalized mismatches.
    
    Removes/replaces:
    - Control characters (except \n, \t) → space
    - Black squares and geometric shapes (Unicode \u25A0-\u25FF) → space
    - Greek semicolon (\u037E) → regular semicolon
    - Page number artefacts ("- N -", "Page - N -")
    - Invalid/undisplayable characters → space (for safety)
    """
    s = (text or "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\x0c", "\n")
    s = s.replace("\u037E", ";")
    s = _PAGE_NUMBER_LINE.sub("", s)
    s = _PAGE_NUMBER_FRAGMENT.sub("", s)
    s = _TRAILING_WS.sub("\n", s)
    s = _MANY_NEWLINES.sub("\n\n", s)
    s = _CONTROL_AND_BLACK.sub(" ", s)
    # Additional safety: replace any remaining invalid/undisplayable characters with space
    # This includes surrogate pairs, private use characters, etc.
    # Keep printable Unicode characters (letters, numbers, punctuation, symbols, spaces)
    import unicodedata
    result = []
    for char in s:
        # Keep newlines and tabs
        if char in ("\n", "\t"):
            result.append(char)
        # Check if character is printable or is a valid Unicode character
        elif unicodedata.category(char)[0] in ("L", "N", "P", "S", "Z"):  # Letter, Number, Punctuation, Symbol, Separator
            result.append(char)
        else:
            # Replace invalid/undisplayable characters with space
            result.append(" ")
    s = "".join(result)
    return s.strip()


def extract_first_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    starts = [m.start() for m in _JSON_START.finditer(text)]
    for start in starts:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "\\" and not escape:
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
            if not in_string:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
            escape = False
    return None


def load_categories(config_path: Path) -> dict[str, Any]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return payload


def load_categories_payload(
    *,
    csv_path: Optional[Path] = None,
    json_path: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Prefer CUAD `category_descriptions.csv` if available (generates `configs/categories.json`),
    else fall back to an existing categories.json.
    """
    if csv_path and csv_path.exists():
        payload = load_categories_from_csv(csv_path)
        if json_path is not None:
            write_categories_json(payload, json_path)
        return payload
    if json_path and json_path.exists():
        return load_categories(json_path)
    raise FileNotFoundError("No categories source found (CSV or JSON).")


def _categories_block(payload: dict[str, Any], *, include_descriptions: bool) -> str:
    lines = []
    for c in payload.get("categories", []):
        if include_descriptions:
            lines.append(f'- {c["id"]}: {c["name"]} — {c.get("description","")}')
        else:
            lines.append(f'- {c["id"]}: {c["name"]}')
    return "\n".join(lines)


@dataclass
class LLMConfig:
    api_key: str
    model_stage1: str
    model_stage2: str
    max_concurrency: int = 8
    base_url: Optional[str] = None
    stage1_include_descriptions: bool = False
    message_content_mode: str = "string"  # "string" | "parts"
    temperature: float = 0.0


class LLMRunner:
    def __init__(
        self,
        cfg: LLMConfig,
        categories_payload: dict[str, Any],
        *,
        fewshot_by_category: Optional[dict[str, list[dict[str, str]]]] = None,
    ) -> None:
        self.client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        self.cfg = cfg
        self.sem = asyncio.Semaphore(cfg.max_concurrency)
        self.categories_payload = categories_payload
        self.fewshot_by_category = fewshot_by_category or {}
        self.categories_block = _categories_block(
            categories_payload, include_descriptions=cfg.stage1_include_descriptions
        )
        self.category_ids = [c["id"] for c in categories_payload.get("categories", [])]

        self.redflag_category_id_by_name: dict[str, str] = {
            "Termination For Convenience": "termination_for_convenience",
            "Uncapped Liability": "uncapped_liability",
            "Irrevocable Or Perpetual License": "irrevocable_or_perpetual_license",
            "Most Favored Nation": "most_favored_nation",
            "Audit Rights": "audit_rights",
            "Ip Ownership Assignment": "ip_ownership_assignment",
        }
        # Build once per runner (keeps prompts stable and avoids repeated work).
        self._redflag_fewshot_block: str = self._build_redflag_fewshot_examples()
        self._redflag_definitions_block: str = self._build_redflag_definitions_block()

    def _build_redflag_fewshot_examples(self) -> str:
        """
        Build few-shot examples for the single-classifier prompt from CUAD cache.
        Evidence is sourced from CUAD_v1.json answers and is guaranteed to be inside the snippet.
        
        NOTE: To modify few-shot example templates (reasoning, explanation, etc.),
        edit the if/elif blocks below (lines ~184-235).
        The structure and data come from CUAD, but the template values can be customized.
        """
        from .prompts import build_redflag_fewshot_example

        blocks: list[str] = []
        for name in [
            "Termination For Convenience",
            "Uncapped Liability",
            "Irrevocable Or Perpetual License",
            "Most Favored Nation",
            "Audit Rights",
            "Ip Ownership Assignment",
        ]:
            cid = self.redflag_category_id_by_name.get(name, "")
            examples = (self.fewshot_by_category.get(cid) or [])[:2]
            for ex_i, ex in enumerate(examples, start=1):
                snippet = ex.get("clause", "")
                evidence = ex.get("evidence", "")
                question = ex.get("question", "")
                if not snippet or not evidence or evidence not in snippet:
                    continue
                snippet_clean = normalize_llm_text(snippet)
                # Keep evidence check strict: if normalization breaks substring, fall back to original snippet.
                if evidence not in snippet_clean:
                    snippet_clean = snippet.strip()
                if name == "Termination For Convenience":
                    reasoning = [
                        "Evidence grants termination without cause / for convenience.",
                        "Termination is triggered by notice, not breach.",
                    ]
                elif name == "Uncapped Liability":
                    reasoning = [
                        "Evidence contains a carve-out where liability limits do not apply.",
                        "This may create uncapped exposure for specified claims.",
                    ]
                elif name == "Irrevocable Or Perpetual License":
                    reasoning = [
                        "Evidence states the license is perpetual and/or irrevocable.",
                        "Rights may survive indefinitely beyond typical term expectations.",
                    ]
                elif name == "Most Favored Nation":
                    reasoning = [
                        "Evidence contains 'most favored', 'uniformly applied', 'equally applied', 'at least as favorable', or similar parity language.",
                        "This commits the party to best-terms treatment compared to others, or requires uniform application of changes to all parties.",
                    ]
                elif name == "Audit Rights":
                    reasoning = [
                        "Evidence grants a right to examine/audit records or confirm compliance.",
                        "This is a classic audit/inspection right.",
                    ]
                else:  # Ip Ownership Assignment
                    reasoning = [
                        "Evidence transfers ownership via assignment / 'right, title and interest' language.",
                        "This reallocates IP ownership to the counterparty.",
                    ]

                if name == "Uncapped Liability":
                    explanation = (
                        "The clause removes or weakens liability limits, which can be unfair if it creates "
                        "disproportionate exposure without a legitimate interest."
                    )
                    legal_refs = [
                        "Consumer Rights Act 2015, sections 62–68 (fairness test)",
                        "UK common law on penalty clauses (genuine pre-estimate of loss, proportionality, legitimate interest)",
                    ]
                    risk_category = "Penalty"
                else:
                    explanation = (
                        "The clause shifts risk or control in a way that may be unfair to the weaker party "
                        "depending on context and bargaining position."
                    )
                    legal_refs = ["Consumer Rights Act 2015, sections 62–68 (fairness test)"]
                    risk_category = "Nothing"
                possible = "May be challenged as unfair and lead to renegotiation, disputes, or non-enforcement."
                revised = "Replace with a balanced clause that limits scope, duration, and caps exposure."
                revision_expl = "Adds proportional limits and clarifies mutual obligations."
                follow_up = "Ask the counterparty to justify the clause and propose a balanced revision."

                blocks.append(
                    f"EXAMPLE {ex_i} (CUAD)\n"
                    f"Category: {name}\n"
                    + (f"Question: {question}\n" if question else "")
                    + "TEXT CHUNK (excerpt; may start/end mid-sentence):\n"
                    + "…\n"
                    + f"{snippet_clean}\n"
                    + "…\n"
                    + "OUTPUT:\n"
                    + json.dumps(
                        {
                            "findings": [
                                {
                                    "category": name,
                                    "quote": evidence,
                                    "reasoning": reasoning[:2],
                                    "is_unfair": True,
                                    "explanation": explanation,
                                    "legal_references": legal_refs,
                                    "possible_consequences": possible,
                                    "risk_assessment": {
                                        "severity_of_consequences": 2,
                                        "degree_of_legal_violation": 2,
                                    },
                                    "consequences_category": "Unenforceable",
                                    "risk_category": risk_category,
                                    "recommended_revision": {
                                        "revised_clause": revised,
                                        "revision_explanation": revision_expl,
                                    },
                                    "suggested_follow_up": follow_up,
                                }
                            ]
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            # Additionally, inject one synthetic MFN-like example for price-change / uniformly-applied clauses
            if name == "Most Favored Nation":
                synthetic_clause = (
                    "2.4. Prices.\n\n"
                    "(B) Inflation Price Adjustment. The prices set forth in Section 2.4(a) shall be subject to "
                    "adjustment annually on the first day of each Product Year beginning in the calendar year 2000 "
                    "and on the first day of each succeeding Product Year for the remainder of the Term and all "
                    "renewals of this Agreement in proportion to the increase or decrease in the Consumer Price Index "
                    "(CPI) as compared to the CPI as it existed on the first day of the Term of this Agreement. "
                    "The Company also reserves the right to increase or decrease the price per unit based on Company "
                    "wide changes in unit prices to all distributors of the Company, provided however, that any price "
                    "changes, other than those based on the CPI, shall be uniformly applied to all distributors of the "
                    "Products and shall reasonably reflect Company's costs of manufacturing the Products and/or market "
                    "demand for the Products, provided further than any increase in price based upon market demand shall "
                    "not be so great as to deprive Distributor of its normal and customary profit margin."
                )
                synthetic_evidence = (
                    "any price changes, other than those based on the CPI, shall be uniformly applied to all "
                    "distributors of the Products and shall reasonably reflect Company's costs of manufacturing the "
                    "Products and/or market demand for the Products, provided further than any increase in price based "
                    "upon market demand shall not be so great as to deprive Distributor of its normal and customary "
                    "profit margin."
                )
                synthetic_question = (
                    'Highlight the parts (if any) of this contract related to "Most Favored Nation" that should be '
                    "reviewed by a lawyer. In particular, look for clauses that require uniform application of price "
                    "changes to all distributors or customers."
                )
                blocks.append(
                    build_redflag_fewshot_example(
                        "Most Favored Nation",
                        synthetic_clause,
                        synthetic_evidence,
                        synthetic_question,
                    )
                )
        return "\n".join(blocks).strip()

    def _build_redflag_definitions_block(self) -> str:
        """
        Build a short category definitions block from CUAD `category_descriptions.csv`
        (already loaded into `categories_payload`).
        """
        id_to_desc: dict[str, str] = {}
        for c in self.categories_payload.get("categories", []):
            cid = str(c.get("id", "")).strip()
            if not cid:
                continue
            id_to_desc[cid] = str(c.get("description", "") or "").strip()

        lines: list[str] = []
        for name, cid in self.redflag_category_id_by_name.items():
            desc = id_to_desc.get(cid, "")
            if desc:
                lines.append(f"- {name}: {desc}")
            else:
                lines.append(f"- {name}: (no description found in category_descriptions.csv)")
        return "\n".join(lines).strip()

    def _build_redflag_prompt(
        self,
        *,
        text_chunk: str,
        section_path: str,
        max_findings: int,
    ) -> tuple[str, str, dict[str, Any]]:
        """
        Build the red flag classification prompt.
        
        NOTE: To modify the prompt instructions, edit app/prompts.py
        This method just assembles the prompt from templates.
        """
        from .prompts import get_redflag_system_prompt, format_redflag_prompt
        
        categories = list(self.redflag_category_id_by_name.keys())
        fewshot = self._redflag_fewshot_block
        defs = self._redflag_definitions_block
        
        header = (
            "========================\n"
            "FEW-SHOT EXAMPLES (from CUAD_v1.json; examples only, NOT the current document)\n"
            "========================\n"
        )
        if fewshot:
            fewshot_section = header + f"{fewshot}\n"
        else:
            fewshot_section = header + "(none available — few-shot cache is empty or disabled)\n"
        
        system = get_redflag_system_prompt()
        user, sections_dict = format_redflag_prompt(
            categories_list="\n".join("- " + c for c in categories),
            section_path=section_path or "(none)",
            max_findings=max_findings,
            category_definitions=defs,
            fewshot_section=fewshot_section,
            text_chunk=text_chunk,
        )
        sections: dict[str, Any] = {
            **sections_dict,
            "context_prefix": section_path or "",
        }
        return system, user, sections
    async def classify_redflags_for_chunk(
        self,
        *,
        text_chunk: str,
        section_path: str = "",
        max_findings: int = 2,
        return_prompt: bool = False,
    ) -> tuple[RedFlagChunkOutput, str, Optional[dict[str, str]]]:
        """
        Single-classifier mode:
        classify one TEXT CHUNK and return up to 2 findings with verbatim evidence.
        
        Note: text_chunk should already be normalized before calling this function.
        LLM will see normalized text and return quote from it.
        """
        # Don't normalize again - text_chunk is already normalized in main.py
        text_chunk_clean = text_chunk
        system, user, sections = self._build_redflag_prompt(
            text_chunk=text_chunk_clean,
            section_path=section_path,
            max_findings=max_findings,
        )

        out, raw = await self._call_json_and_raw(self.cfg.model_stage2, system, user, RedFlagChunkOutput)
        
        # Log what we got from parsing
        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            f"After _call_json_and_raw: out.findings type={type(out.findings)}, "
            f"len={len(out.findings) if isinstance(out.findings, list) else 'N/A'}, "
            f"value={out.findings if isinstance(out.findings, list) and len(out.findings) <= 2 else 'too long'}"
        )
        
        # Ensure findings is a list (not None)
        if out.findings is None:
            logger.warning("out.findings is None, setting to empty list")
            out.findings = []
        elif not isinstance(out.findings, list):
            logger.error(f"out.findings is not a list: {type(out.findings)}, value: {out.findings}")
            out.findings = []
        else:
            original_len = len(out.findings)
            out.findings = out.findings[:max_findings]
            if original_len > max_findings:
                logger.warning(f"Truncated findings from {original_len} to {max_findings}")
        
        # Log findings (only raw output, no text_chunk) for debugging
        # (logger already imported above)
        if not out.findings:
            logger.warning(
                f"LLM returned empty findings. Temperature={self.cfg.temperature}, "
                f"Raw output length={len(raw)}"
            )
            if raw:
                logger.warning(f"LLM raw output: {raw}")
        else:
            logger.info(
                f"LLM returned {len(out.findings)} findings. Raw output length={len(raw)}"
            )
            if raw:
                logger.info(f"LLM raw output: {raw}")
            # Log each finding's raw data
            for i, f in enumerate(out.findings):
                logger.debug(
                    f"Finding {i+1}: category={f.category}, quote_length={len(f.quote or '')}, "
                    f"quote_preview={f.quote[:100] if f.quote else 'EMPTY'}"
                )
        prompt: Optional[dict[str, str]] = None
        if return_prompt:
            prompt = {"system": system, "user": user, "sections": sections}  # type: ignore[assignment]
        return out, raw, prompt

    def _build_messages(self, *, system: str, user: str) -> list[dict[str, Any]]:
        if self.cfg.message_content_mode == "parts":
            # Nebius TokenFactory (and some OpenAI-compatible providers) accept content as a list of parts.
            return [
                {"role": "system", "content": system},
                {"role": "user", "content": [{"type": "text", "text": user}]},
            ]
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    async def _call_json_and_raw(
        self,
        model: str,
        system: str,
        user: str,
        schema: Type[T],
        *,
        retries: int = 2,
        extra_validator: Optional[Callable[[T], None]] = None,
    ) -> tuple[T, str]:
        last_err: Optional[Exception] = None
        last_text: Optional[str] = None
        use_response_format = True
        for attempt in range(retries + 1):
            try:
                async with self.sem:
                    kwargs: dict[str, Any] = {}
                    if use_response_format:
                        kwargs["response_format"] = {"type": "json_object"}
                    # Create the LLM call as a task so it can be cancelled
                    resp = await self.client.chat.completions.create(
                        model=model,
                        messages=self._build_messages(system=system, user=user),
                        temperature=self.cfg.temperature,
                        **kwargs,
                    )
                text = (resp.choices[0].message.content or "").strip()
                last_text = text
                blob = extract_first_json_object(text) or ""
                if not blob:
                    if schema == RedFlagChunkOutput:
                        return RedFlagChunkOutput(findings=[]), (last_text or "")
                    raise ValueError("No JSON object found in LLM response")
                data = json.loads(blob)
                try:
                    parsed = schema.model_validate(data)
                    # Log successful parsing for debugging
                    if schema == RedFlagChunkOutput:
                        import logging
                        logger = logging.getLogger(__name__)
                        findings_count = len(parsed.findings) if parsed.findings else 0
                        data_findings_count = len(data.get('findings', [])) if isinstance(data, dict) else 0
                        logger.info(
                            f"Successfully parsed RedFlagChunkOutput: {findings_count} findings. "
                            f"Data has {data_findings_count} findings. "
                            f"Data findings type: {type(data.get('findings')) if isinstance(data, dict) else 'N/A'}, "
                            f"Parsed findings type: {type(parsed.findings)}"
                        )
                        # If data has findings but parsed doesn't, this is a problem - trigger salvage
                        if data_findings_count > 0 and findings_count == 0:
                            logger.error(
                                f"CRITICAL: Data has {data_findings_count} findings but parsed has 0! "
                                f"Data findings: {data.get('findings', [])[:2] if isinstance(data, dict) else 'N/A'}. "
                                f"Triggering salvage logic..."
                            )
                            # Force ValidationError to trigger salvage logic
                            findings_data = data.get('findings', []) if isinstance(data, dict) else []
                            if findings_data:
                                # Manually trigger salvage by raising a ValidationError
                                # ValidationError is already imported at top of file
                                raise ValidationError.from_exception_data(
                                    "RedFlagChunkOutput",
                                    [{"type": "value_error", "loc": ("findings",), "msg": "Findings lost during validation", "input": findings_data}]
                                )
                except ValidationError as ve:
                    # Log validation error for debugging
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Validation error parsing LLM response: {ve}. "
                        f"Data keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}. "
                        f"Data findings count: {len(data.get('findings', [])) if isinstance(data, dict) else 0}. "
                        f"Attempting to salvage findings..."
                    )
                    if schema == RedFlagChunkOutput and "findings" in data:
                        findings_data = data.get("findings", [])
                        salvaged_findings = []
                        for f_data in findings_data:
                            try:
                                # Normalize consequences_category to match Literal type exactly
                                consequences_cat_raw = f_data.get("consequences_category", "Nothing")
                                consequences_cat = "Nothing"  # Default
                                
                                if isinstance(consequences_cat_raw, str):
                                    cat_lower = consequences_cat_raw.lower().strip()
                                    # Map to exact Literal values (case-sensitive matching)
                                    if cat_lower == "unenforceable":
                                        consequences_cat = "Unenforceable"
                                    elif cat_lower == "invalid":
                                        consequences_cat = "Invalid"
                                    elif cat_lower == "void":
                                        consequences_cat = "Void"
                                    elif cat_lower == "voidable":
                                        consequences_cat = "Voidable"
                                    elif cat_lower == "nothing":
                                        consequences_cat = "Nothing"
                                    elif consequences_cat_raw == "Unenforceable":  # Already correct
                                        consequences_cat = "Unenforceable"
                                    elif consequences_cat_raw == "Invalid":
                                        consequences_cat = "Invalid"
                                    elif consequences_cat_raw == "Void":
                                        consequences_cat = "Void"
                                    elif consequences_cat_raw == "Voidable":
                                        consequences_cat = "Voidable"
                                    else:
                                        # Default to Nothing if unknown
                                        import logging
                                        logger = logging.getLogger(__name__)
                                        logger.warning(f"Unknown consequences_category: '{consequences_cat_raw}', defaulting to 'Nothing'")
                                        consequences_cat = "Nothing"
                                
                                # Normalize risk_category to match Literal type exactly
                                risk_cat = f_data.get("risk_category", "Nothing")
                                if isinstance(risk_cat, str):
                                    cat_lower = risk_cat.lower()
                                    # Map to exact Literal values
                                    if cat_lower == "penalty":
                                        risk_cat = "Penalty"
                                    elif cat_lower == "criminal liability":
                                        risk_cat = "Criminal liability"
                                    elif "failure" in cat_lower and "obligations" in cat_lower:
                                        risk_cat = "Failure of the other party to perform obligations on time"
                                    elif cat_lower == "nothing":
                                        risk_cat = "Nothing"
                                    else:
                                        # Default to Nothing if unknown
                                        risk_cat = "Nothing"
                                
                                # Normalize risk_assessment values (must be 0-3)
                                risk_assessment = f_data.get("risk_assessment", {})
                                if isinstance(risk_assessment, dict):
                                    severity = risk_assessment.get("severity_of_consequences", 0)
                                    violation = risk_assessment.get("degree_of_legal_violation", 0)
                                    # Clamp to valid range [0, 3]
                                    severity = max(0, min(3, int(severity) if isinstance(severity, (int, float)) else 0))
                                    violation = max(0, min(3, int(violation) if isinstance(violation, (int, float)) else 0))
                                    risk_assessment = {"severity_of_consequences": severity, "degree_of_legal_violation": violation}
                                else:
                                    risk_assessment = {"severity_of_consequences": 0, "degree_of_legal_violation": 0}
                                
                                # Double-check that consequences_cat is correct before creating
                                if consequences_cat not in ("Invalid", "Unenforceable", "Void", "Voidable", "Nothing"):
                                    import logging
                                    logger = logging.getLogger(__name__)
                                    logger.error(f"INVALID consequences_cat value: '{consequences_cat}', defaulting to 'Nothing'")
                                    consequences_cat = "Nothing"
                                
                                # Log what we're trying to create
                                import logging
                                logger = logging.getLogger(__name__)
                                logger.info(
                                    f"Salvaging finding: category={f_data.get('category')}, "
                                    f"consequences_cat='{consequences_cat}' (type: {type(consequences_cat)}, "
                                    f"in valid list: {consequences_cat in ('Invalid', 'Unenforceable', 'Void', 'Voidable', 'Nothing')}), "
                                    f"risk_cat='{risk_cat}', risk_assessment={risk_assessment}"
                                )
                                
                                f_data_fixed = {
                                    "category": f_data.get("category", "Termination For Convenience"),
                                    "quote": f_data.get("quote") or f_data.get("evidence") or "",
                                    "reasoning": f_data.get("reasoning", []),
                                    "is_unfair": f_data.get("is_unfair", True),
                                    "explanation": f_data.get("explanation", ""),
                                    "legal_references": f_data.get("legal_references", []),
                                    "possible_consequences": f_data.get("possible_consequences", ""),
                                    "risk_assessment": risk_assessment,
                                    "consequences_category": consequences_cat,  # Use normalized value
                                    "risk_category": risk_cat,
                                    "recommended_revision": f_data.get("recommended_revision", {"revised_clause": "", "revision_explanation": ""}),
                                    "suggested_follow_up": f_data.get("suggested_follow_up", ""),
                                }
                                
                                # Verify the value one more time
                                if f_data_fixed["consequences_category"] not in ("Invalid", "Unenforceable", "Void", "Voidable", "Nothing"):
                                    logger.error(f"CRITICAL: f_data_fixed has invalid consequences_category: '{f_data_fixed['consequences_category']}'")
                                    f_data_fixed["consequences_category"] = "Nothing"
                                
                                salvaged_findings.append(RedFlagFindingLLM(**f_data_fixed))
                            except Exception as salvage_err:
                                import logging
                                logger = logging.getLogger(__name__)
                                logger.warning(
                                    f"Failed to salvage finding: {salvage_err}. "
                                    f"consequences_cat was set to: '{consequences_cat}', "
                                    f"risk_assessment: {risk_assessment}, "
                                    f"f_data consequences_category: '{f_data.get('consequences_category')}'"
                                )
                                import traceback
                                logger.debug(f"Salvage error traceback: {traceback.format_exc()}")
                                continue
                        if salvaged_findings:
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.info(f"Salvaged {len(salvaged_findings)} findings from validation error")
                        parsed = RedFlagChunkOutput(findings=salvaged_findings)
                    else:
                        # No findings to salvage, but validation failed - log the error
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.error(f"Validation error but no findings to salvage: {ve}")
                        raise
                if extra_validator is not None:
                    extra_validator(parsed)
                return parsed, (last_text or "")
            except Exception as e:
                last_err = e
                # If this looks like an HTTP status error (OpenAI-compatible SDK),
                # don't blindly retry on auth/permission errors.
                status = getattr(e, "status_code", None)
                if status in (401, 403):
                    break
                if use_response_format and "response_format" in str(e).lower():
                    use_response_format = False
                continue
        status = getattr(last_err, "status_code", None)
        msg = str(last_err)
        raw = ""
        if last_text:
            # Keep logs readable and avoid huge payloads.
            raw = last_text[:1200]
            if len(last_text) > 1200:
                raw += "\n... (truncated)"
        if status is not None:
            raise RuntimeError(
                f"LLM request failed (HTTP {status}). Check API key, base_url, and model access. "
                f"Raw error: {msg}"
            ) from last_err
        if raw:
            raise RuntimeError(
                "LLM JSON validation failed after retries: "
                f"{msg}\n--- raw_llm_output (truncated) ---\n{raw}"
            ) from last_err
        raise RuntimeError(f"LLM JSON validation failed after retries: {msg}") from last_err

    async def _call_json(
        self,
        model: str,
        system: str,
        user: str,
        schema: Type[T],
        *,
        retries: int = 2,
        extra_validator: Optional[Callable[[T], None]] = None,
    ) -> T:
        parsed, _raw = await self._call_json_and_raw(
            model, system, user, schema, retries=retries, extra_validator=extra_validator
        )
        return parsed

    def _normalize_stage1_category_id(self, cid: str) -> str:
        cid = (cid or "").strip()
        if cid.isdigit():
            idx = int(cid)
            if 1 <= idx <= len(self.category_ids):
                return self.category_ids[idx - 1]
        return cid

    def _validate_and_clean_stage1(
        self,
        out: Stage1Output,
        *,
        max_categories: int,
        allowed_evidence_indices: set[int],
        t_router: float,
        require_evidence: bool = True,
    ) -> tuple[Stage1Output, list[str]]:
        """
        Normalize numeric IDs, drop invalid IDs, enforce max_categories.
        Returns (cleaned_out, invalid_ids).
        """
        invalid: list[str] = []

        # Normalize and validate probs
        probs: dict[str, float] = {}
        cleaned_probs: list[Stage1Prob] = []
        for p in out.category_probs or []:
            cid = self._normalize_stage1_category_id(p.category_id)
            if cid not in self.category_ids:
                invalid.append(cid)
                continue
            val = float(p.prob)
            probs[cid] = max(0.0, min(1.0, val))
            cleaned_probs.append(Stage1Prob(category_id=cid, prob=probs[cid]))
        out.category_probs = cleaned_probs

        cleaned_cats: list[Stage1Category] = []
        for item in out.categories or []:
            cid = self._normalize_stage1_category_id(item.category_id)
            if cid not in self.category_ids:
                invalid.append(cid)
                continue
            # Prefer router prob if present; otherwise keep provided confidence.
            conf = probs.get(cid, float(item.confidence))
            if conf < t_router:
                continue
            item.category_id = cid
            item.confidence = max(0.0, min(1.0, conf))
            item.evidence_paragraph_indices = [
                int(i) for i in (item.evidence_paragraph_indices or []) if int(i) in allowed_evidence_indices
            ]
            if require_evidence and not item.evidence_paragraph_indices:
                continue
            cleaned_cats.append(item)
        out.categories = cleaned_cats[:max_categories]
        return out, invalid

    def _build_stage1_prompt(
        self,
        *,
        main_paragraphs: list[Paragraph],
        context_paragraphs: list[Paragraph],
        section_path: str = "",
        t_router: float,
        topk_probs: int,
        max_categories: int,
        strict: bool,
    ) -> tuple[str, str]:
        system = "You are a contract analyst. Output STRICT JSON only."
        allowed_ids = self.category_ids
        allowed_ids_json = json.dumps(allowed_ids, ensure_ascii=False)
        example_id = allowed_ids[0] if allowed_ids else ""
        main_indices = [p.paragraph_index for p in main_paragraphs]
        main_indices_json = json.dumps(main_indices)

        def fmt(ps: list[Paragraph], *, label: str) -> str:
            lines = []
            for p in ps:
                lines.append(f"[{label} ¶{p.paragraph_index} p.{p.page}] {p.text}")
            return "\n\n".join(lines).strip()

        context_block = fmt(context_paragraphs, label="CTX") if context_paragraphs else ""
        main_block = fmt(main_paragraphs, label="MAIN")
        section_prefix = f"Section path: {section_path}\n\n" if section_path else ""

        if strict:
            user = f"""
Task: multi-label screening of a contract SECTION (group of paragraphs).

You MUST return STRICT JSON only.

Allowed category_id values (MUST choose only from this list, copy exactly):
{allowed_ids_json}

Return STRICT JSON with:
- category_probs: list of objects (top candidates), each with:
  - category_id: string from the allowed list
  - prob: float 0..1
- unclear_prob: float 0..1 (how unsure/ambiguous this section is)
- categories: list of objects (possibly empty), ONLY for categories with prob >= {t_router}, each with:
  - category_id: string from the allowed list (never "unknown", never a number)
  - severity: one of ["low","medium","high"]
  - confidence: float 0..1 (router prob)
  - rationale_short: <=220 chars
  - evidence_paragraph_indices: list[int] (MUST be chosen from these paragraph indices: {main_indices_json}; max 5)

Rules:
- If nothing is relevant / the section is "safe" (e.g., operational details, product/deliverables description, generic background, contact info), return: {{"categories": []}}
- category_probs: include at least the top {topk_probs} candidates, plus any category with prob >= {t_router}.
- categories: include ALL categories with prob >= {t_router}; if too many, return at most {max_categories}.
- Do NOT invent category ids (no "unknown"/"other"/etc).
- For every returned category, you MUST point to at least 1 evidence paragraph index from the MAIN block.
- If ALL probs < {t_router}, set categories=[] (abstain) and set unclear_prob high (e.g. 0.6+).

Example:
{{"category_probs": [{{"category_id": "{example_id}", "prob": 0.72}}], "unclear_prob": 0.1, "categories": [{{"category_id": "{example_id}", "severity": "medium", "confidence": 0.72, "rationale_short": "Short rationale.", "evidence_paragraph_indices": [{main_indices[0] if main_indices else 0}]}}]}}

{section_prefix}Context (previous paragraphs, may help; do NOT use as evidence indices):
{context_block}

MAIN section paragraphs (use these for evidence indices):
{main_block}
""".strip()
        else:
            user = f"""
Your task: quick multi-label screening of a contract SECTION (group of paragraphs) for potential red-flag clause types.

Available categories (category_id -> name -> description):
{self.categories_block}

Return STRICT JSON with:
- category_probs: list of objects (top candidates), each with:
  - category_id: string from the allowed list
  - prob: float 0..1
- unclear_prob: float 0..1
- categories: list of objects, ONLY for categories with prob >= {t_router}, each with:
  - category_id: MUST be a STRING and one of {allowed_ids} (do NOT output a number)
  - severity: one of ["low","medium","high"]
  - confidence: float 0..1 (router prob)
  - rationale_short: <=220 chars
  - evidence_paragraph_indices: list[int] chosen from {main_indices_json} (max 5)

Rules:
- If no categories are relevant / the section is "safe" (e.g., operational details, product/deliverables description, generic background, contact info), return: {{"categories": []}}
- category_probs: include at least the top {topk_probs} candidates, plus any category with prob >= {t_router}.
- categories: include ALL categories with prob >= {t_router}; if too many, return at most {max_categories}.
- Do not output anything except JSON.
- For every returned category, you MUST point to at least 1 evidence paragraph index from the MAIN block.
- If ALL probs < {t_router}, set categories=[] and set unclear_prob high (e.g. 0.6+).

Example:
{{"category_probs": [{{"category_id": "{example_id}", "prob": 0.72}}], "unclear_prob": 0.1, "categories": [{{"category_id": "{example_id}", "severity": "medium", "confidence": 0.72, "rationale_short": "Short rationale.", "evidence_paragraph_indices": [{main_indices[0] if main_indices else 0}]}}]}}

{section_prefix}Context (previous paragraphs, may help; do NOT use as evidence indices):
{context_block}

MAIN section paragraphs (use these for evidence indices):
{main_block}
""".strip()
        return system, user

    async def stage1_for_block(
        self,
        *,
        main_paragraphs: list[Paragraph],
        context_paragraphs: list[Paragraph],
        section_path: str = "",
        t_router: float = 0.3,
        topk_probs: int = 15,
        max_categories: int = 10,
    ) -> Stage1Output:
        allowed_indices = {p.paragraph_index for p in main_paragraphs}
        # Pass 1 (fast)
        system, user = self._build_stage1_prompt(
            main_paragraphs=main_paragraphs,
            context_paragraphs=context_paragraphs,
            section_path=section_path,
            t_router=t_router,
            topk_probs=topk_probs,
            max_categories=max_categories,
            strict=False,
        )
        out = await self._call_json(self.cfg.model_stage1, system, user, Stage1Output)
        out, invalid = self._validate_and_clean_stage1(
            out,
            max_categories=max_categories,
            allowed_evidence_indices=allowed_indices,
            t_router=t_router,
        )
        if invalid:
            system2, user2 = self._build_stage1_prompt(
                main_paragraphs=main_paragraphs,
                context_paragraphs=context_paragraphs,
                section_path=section_path,
                t_router=t_router,
                topk_probs=topk_probs,
                max_categories=max_categories,
                strict=True,
            )
            out2 = await self._call_json(self.cfg.model_stage1, system2, user2, Stage1Output)
            out2, invalid2 = self._validate_and_clean_stage1(
                out2,
                max_categories=max_categories,
                allowed_evidence_indices=allowed_indices,
                t_router=t_router,
            )
            if not invalid2:
                return out2
        return out

    async def stage1_for_block_with_raw(
        self,
        *,
        main_paragraphs: list[Paragraph],
        context_paragraphs: list[Paragraph],
        section_path: str = "",
        t_router: float = 0.3,
        topk_probs: int = 15,
        max_categories: int = 10,
    ) -> tuple[Stage1Output, str]:
        allowed_indices = {p.paragraph_index for p in main_paragraphs}
        system, user = self._build_stage1_prompt(
            main_paragraphs=main_paragraphs,
            context_paragraphs=context_paragraphs,
            section_path=section_path,
            t_router=t_router,
            topk_probs=topk_probs,
            max_categories=max_categories,
            strict=False,
        )
        out, raw = await self._call_json_and_raw(self.cfg.model_stage1, system, user, Stage1Output)
        out, invalid = self._validate_and_clean_stage1(
            out,
            max_categories=max_categories,
            allowed_evidence_indices=allowed_indices,
            t_router=t_router,
        )
        if invalid:
            system2, user2 = self._build_stage1_prompt(
                main_paragraphs=main_paragraphs,
                context_paragraphs=context_paragraphs,
                section_path=section_path,
                t_router=t_router,
                topk_probs=topk_probs,
                max_categories=max_categories,
                strict=True,
            )
            out2, raw2 = await self._call_json_and_raw(self.cfg.model_stage1, system2, user2, Stage1Output)
            out2, invalid2 = self._validate_and_clean_stage1(
                out2,
                max_categories=max_categories,
                allowed_evidence_indices=allowed_indices,
                t_router=t_router,
            )
            if not invalid2:
                return out2, raw2
        return out, raw

    async def stage2(
        self,
        paragraph: Paragraph,
        stage1: Stage1Category,
        *,
        context_text: str = "",
        section_path: str = "",
    ) -> Stage2Output:
        system = "You are a senior contract lawyer. Output STRICT JSON only."
        fewshot_block = ""
        examples = self.fewshot_by_category.get(stage1.category_id) or []
        if examples:
            ex_lines = []
            for i, ex in enumerate(examples[:3], start=1):
                ex_lines.append(
                    f"Reference {i} (CUAD snippet for this category)\n"
                    f"Snippet:\n\"\"\"{ex.get('clause','')}\"\"\"\n"
                    f"Typical evidence span (from CUAD QA): \"{ex.get('evidence','')}\"\n"
                )
            fewshot_block = "\n".join(ex_lines).strip()
        fewshot_section = ""
        if fewshot_block:
            fewshot_section = (
                "Few-shot references (for pattern recognition only; do NOT copy evidence_quote from these):\n"
                + fewshot_block
                + "\n"
            )
        context_section = ""
        if context_text.strip():
            context_section = (
                "Context (neighbor paragraphs in the same section; use to interpret, but quote evidence from the EVIDENCE paragraph only):\n"
                + context_text.strip()
                + "\n"
            )
        section_path_section = f"Section path: {section_path}\n" if section_path else ""
        user = f"""
Your task: produce a detailed red-flag finding for the paragraph, based on the stage-1 category.

Category (id): {stage1.category_id}
Stage-1 severity: {stage1.severity}
Stage-1 rationale: {stage1.rationale_short}

{fewshot_section}
{section_path_section}
{context_section}

Return STRICT JSON with:
- issue_title: <= 120 chars
- what_is_wrong: <= 600 chars
- recommendation: <= 400 chars
- evidence_quote: <= 400 chars (a direct quote from the paragraph)
- severity: one of ["low","medium","high"]
- confidence: float 0..1
- tags: optional list[str], max 6

Rules:
- evidence_quote MUST be copied from the EVIDENCE paragraph below (verbatim).
- You MAY use the Context and Few-shot references to understand meaning, but evidence_quote MUST NOT be copied from them.
- If you cannot find a good direct quote inside the EVIDENCE paragraph, pick the most relevant sentence fragment from the EVIDENCE paragraph anyway.
- Do not output anything except JSON.

Example output (format only):
{{"issue_title":"...", "what_is_wrong":"...", "recommendation":"...", "evidence_quote":"...", "severity":"medium", "confidence":0.7, "tags":["..."]}}

EVIDENCE paragraph:
\"\"\"{paragraph.text}\"\"\"
""".strip()
        return await self._call_json(self.cfg.model_stage2, system, user, Stage2Output)

    async def stage2_with_raw(
        self,
        paragraph: Paragraph,
        stage1: Stage1Category,
        *,
        context_text: str = "",
        section_path: str = "",
    ) -> tuple[Stage2Output, str]:
        system = "You are a senior contract lawyer. Output STRICT JSON only."
        fewshot_block = ""
        examples = self.fewshot_by_category.get(stage1.category_id) or []
        if examples:
            ex_lines = []
            for i, ex in enumerate(examples[:3], start=1):
                ex_lines.append(
                    f"Reference {i} (CUAD snippet for this category)\n"
                    f"Snippet:\n\"\"\"{ex.get('clause','')}\"\"\"\n"
                    f"Typical evidence span (from CUAD QA): \"{ex.get('evidence','')}\"\n"
                )
            fewshot_block = "\n".join(ex_lines).strip()
        fewshot_section = ""
        if fewshot_block:
            fewshot_section = (
                "Few-shot references (for pattern recognition only; do NOT copy evidence_quote from these):\n"
                + fewshot_block
                + "\n"
            )
        context_section = ""
        if context_text.strip():
            context_section = (
                "Context (neighbor paragraphs in the same section; use to interpret, but quote evidence from the EVIDENCE paragraph only):\n"
                + context_text.strip()
                + "\n"
            )
        section_path_section = f"Section path: {section_path}\n" if section_path else ""
        user = f"""
Your task: produce a detailed red-flag finding for the paragraph, based on the stage-1 category.

Category (id): {stage1.category_id}
Stage-1 severity: {stage1.severity}
Stage-1 rationale: {stage1.rationale_short}

{fewshot_section}
{section_path_section}
{context_section}

Return STRICT JSON with:
- issue_title: <= 120 chars
- what_is_wrong: <= 600 chars
- recommendation: <= 400 chars
- evidence_quote: <= 400 chars (a direct quote from the paragraph)
- severity: one of ["low","medium","high"]
- confidence: float 0..1
- tags: optional list[str], max 6

Rules:
- evidence_quote MUST be copied from the EVIDENCE paragraph below (verbatim).
- You MAY use the Context and Few-shot references to understand meaning, but evidence_quote MUST NOT be copied from them.
- If you cannot find a good direct quote inside the EVIDENCE paragraph, pick the most relevant sentence fragment from the EVIDENCE paragraph anyway.
- Do not output anything except JSON.

Example output (format only):
{{"issue_title":"...", "what_is_wrong":"...", "recommendation":"...", "evidence_quote":"...", "severity":"medium", "confidence":0.7, "tags":["..."]}}

EVIDENCE paragraph:
\"\"\"{paragraph.text}\"\"\"
""".strip()
        return await self._call_json_and_raw(self.cfg.model_stage2, system, user, Stage2Output)

