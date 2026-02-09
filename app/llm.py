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
    RouterCategoriesOutput,
)
from .validate_legal_refs import filter_valid_legal_references
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
    version: Optional[str] = None,
) -> dict[str, Any]:
    """
    Prefer CUAD `category_descriptions.csv` if available (generates `configs/categories.json`),
    else fall back to an existing categories.json.
    """
    if csv_path and csv_path.exists():
        payload = load_categories_from_csv(csv_path, version=version or "cuad_v1_41_from_csv")
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
    seed: Optional[int] = None  # For reproducible outputs (OpenAI and compatible APIs)


class QuoteRepairOutput(BaseModel):
    quote: str = ""


class LLMRunner:
    def __init__(
        self,
        cfg: LLMConfig,
        categories_payload: dict[str, Any],
        *,
        fewshot_by_category: Optional[dict[str, list[dict[str, str]]]] = None,
        uk_fewshot_text: Optional[str] = None,
    ) -> None:
        self.client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        self.cfg = cfg
        self.sem = asyncio.Semaphore(cfg.max_concurrency)
        self.categories_payload = categories_payload
        self.fewshot_by_category = fewshot_by_category or {}
        self.uk_fewshot_text = (uk_fewshot_text or "").strip()
        self.categories_block = _categories_block(
            categories_payload, include_descriptions=cfg.stage1_include_descriptions
        )
        self.category_ids = [c["id"] for c in categories_payload.get("categories", [])]
        self.category_name_by_id = {
            str(c.get("id", "")).strip(): str(c.get("name", "")).strip()
            for c in categories_payload.get("categories", [])
        }
        self.category_id_by_name = {
            str(c.get("name", "")).strip().lower(): str(c.get("id", "")).strip()
            for c in categories_payload.get("categories", [])
        }
        self._canonical_name_by_norm: dict[str, str] = {}
        for c in categories_payload.get("categories", []):
            name = str(c.get("name", "")).strip()
            if not name:
                continue
            self._canonical_name_by_norm[self._normalize_category_name(name)] = name

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

    def _build_quote_repair_prompt(self, *, paragraph_text: str, target_quote: str) -> tuple[str, str]:
        system = (
            "You are a precise text extractor. "
            "Given a PARAGRAPH and a TARGET QUOTE (which may have minor differences like punctuation, spacing, or case), "
            "find the exact verbatim substring from the PARAGRAPH that matches the target quote in meaning. "
            "The quote should be semantically identical but may have minor formatting differences (punctuation, spacing, capitalization). "
            "Return the EXACT text from the PARAGRAPH, not a paraphrase. "
            "If you cannot find a clear match that is semantically identical, return an empty string."
        )
        user = (
            "Return JSON only in the form: {\"quote\": \"...\"}\n\n"
            f"PARAGRAPH:\n{paragraph_text}\n\n"
            f"TARGET QUOTE (find the semantically identical text in the paragraph above):\n{target_quote}\n\n"
            f"Find the exact substring from the PARAGRAPH that matches the TARGET QUOTE in meaning. "
            f"Minor differences in punctuation, spacing, or capitalization are acceptable, but the meaning must be identical."
        )
        return system, user

    async def repair_quote_from_paragraph(self, *, paragraph_text: str, target_quote: str) -> str:
        if not paragraph_text or not target_quote:
            return ""
        system, user = self._build_quote_repair_prompt(
            paragraph_text=paragraph_text, target_quote=target_quote
        )
        out, _ = await self._call_json_and_raw(
            self.cfg.model_stage2, system, user, QuoteRepairOutput
        )
        return (out.quote or "").strip()

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

    def _build_uk_category_blocks(self) -> tuple[str, str, list[str]]:
        """
        Build (categories_list, category_definitions, category_ids) blocks for UK tenancy categories.
        Uses canonical category IDS in the prompt (matches legacy id-based handling).
        """
        list_lines: list[str] = []
        def_lines: list[str] = []
        ids: list[str] = []
        for c in self.categories_payload.get("categories", []):
            cid = str(c.get("id", "")).strip()
            name = str(c.get("name", "")).strip()
            desc = str(c.get("description", "") or "").strip()
            if not cid:
                continue
            ids.append(cid)
            list_lines.append(f"- {cid}")
            if desc:
                def_lines.append(f"- {cid} ({name}): {desc}")
            else:
                def_lines.append(f"- {cid} ({name}): (no description)")
        return "\n".join(list_lines).strip(), "\n".join(def_lines).strip(), ids

    def resolve_category_id(self, value: str) -> str:
        v = (value or "").strip()
        if not v:
            return v
        if v in self.category_ids:
            return v
        v_lower = v.lower()
        if v_lower in self.category_id_by_name:
            return self.category_id_by_name[v_lower]
        if v in self.redflag_category_id_by_name:
            return self.redflag_category_id_by_name[v]
        for k, cid in self.redflag_category_id_by_name.items():
            if k.lower() == v_lower:
                return cid
        return v_lower.replace(" ", "_")

    def _normalize_category_name(self, name: str) -> str:
        return " ".join((name or "").split()).strip().casefold()

    def normalize_category_name(self, name: str) -> str:
        key = self._normalize_category_name(name)
        return self._canonical_name_by_norm.get(key, (name or "").strip())

    def _build_redflag_prompt(
        self,
        *,
        text_chunk: str,
        section_path: str,
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
            category_definitions=defs,
            fewshot_section=fewshot_section,
            text_chunk=text_chunk,
        )
        sections: dict[str, Any] = {
            **sections_dict,
            "context_prefix": section_path or "",
        }
        return system, user, sections

    def _build_uk_router_prompt(
        self,
        *,
        text_chunk: str,
        section_path: str,
    ) -> tuple[str, str, dict[str, Any]]:
        """
        Build the UK router prompt (first stage of two-stage classification).
        
        NOTE: To modify the prompt instructions, edit app/prompts.py:
        - System prompt: get_uk_router_system_prompt()
        - User prompt: get_uk_router_user_prompt_template()
        This method just assembles the prompt from templates.
        """
        from .prompts import get_uk_router_system_prompt, format_uk_router_prompt

        categories_list, defs, _names = self._build_uk_category_blocks()
        system = get_uk_router_system_prompt()
        user, sections_dict = format_uk_router_prompt(
            categories_list=categories_list,
            section_path=section_path or "(none)",
            category_definitions=defs,
            text_chunk=text_chunk,
        )
        sections: dict[str, Any] = {
            **sections_dict,
            "context_prefix": section_path or "",
        }
        return system, user, sections

    def _build_uk_redflag_prompt(
        self,
        *,
        text_chunk: str,
        section_path: str,
        allowed_category_ids: list[str],
    ) -> tuple[str, str, dict[str, Any]]:
        """
        Build the UK classifier prompt (second stage of two-stage classification).
        
        NOTE: To modify the prompt instructions, edit app/prompts.py:
        - System prompt: get_uk_redflag_system_prompt()
        - User prompt: get_uk_redflag_user_prompt_template()
        - Few-shot examples: data/fewshot_uk_redflags.txt (plain text file)
        This method just assembles the prompt from templates.
        """
        from .prompts import get_uk_redflag_system_prompt, format_uk_redflag_prompt

        allowed_set = {n.strip() for n in allowed_category_ids if n and n.strip()}
        list_lines: list[str] = []
        def_lines: list[str] = []
        for c in self.categories_payload.get("categories", []):
            name = str(c.get("name", "")).strip()
            cid = str(c.get("id", "")).strip()
            if cid not in allowed_set:
                continue
            desc = str(c.get("description", "") or "").strip()
            list_lines.append(f"- {cid}")
            def_lines.append(f"- {cid} ({name}): {desc}" if desc else f"- {cid} ({name}): (no description)")
        categories_list = "\n".join(list_lines).strip()
        category_definitions = "\n".join(def_lines).strip()
        fewshot_section = self.uk_fewshot_text or "(none provided)"
        system = get_uk_redflag_system_prompt()
        user, sections_dict = format_uk_redflag_prompt(
            categories_list=categories_list,
            section_path=section_path or "(none)",
            category_definitions=category_definitions,
            fewshot_section=fewshot_section,
            text_chunk=text_chunk,
        )
        sections: dict[str, Any] = {
            **sections_dict,
            "context_prefix": section_path or "",
        }
        return system, user, sections

    async def classify_redflags_for_chunk_v2(
        self,
        *,
        text_chunk: str,
        section_path: str = "",
        return_prompt: bool = False,
    ) -> tuple[RedFlagChunkOutput, str, Optional[dict[str, Any]]]:
        text_chunk_clean = text_chunk
        router_system, router_user, router_sections = self._build_uk_router_prompt(
            text_chunk=text_chunk_clean,
            section_path=section_path,
        )
        router_out, router_raw = await self._call_json_and_raw(
            self.cfg.model_stage1, router_system, router_user, RouterCategoriesOutput
        )
        _list, _defs, all_ids = self._build_uk_category_blocks()
        allowed_ids = []
        for item in router_out.candidate_categories or []:
            name_raw = (item.category or "").strip()
            if not name_raw:
                continue
            cid = self.resolve_category_id(name_raw)
            if cid in all_ids and cid not in allowed_ids:
                allowed_ids.append(cid)
        if not allowed_ids:
            out = RedFlagChunkOutput(findings=[])
            prompt: Optional[dict[str, Any]] = None
            if return_prompt:
                prompt = {
                    "system": "SKIPPED: no router categories",
                    "user": "SKIPPED: no router categories",
                    "sections": {
                        "context_prefix": section_path or "",
                        "text_chunk": text_chunk_clean,
                        "category_definitions": "",
                        "fewshot_examples": "",
                    },
                    "router": {
                        "system": router_system,
                        "user": router_user,
                        "sections": router_sections,
                        "raw": router_raw,
                        "parsed": router_out.model_dump(),
                    },
                }
            return out, "", prompt
        system, user, sections = self._build_uk_redflag_prompt(
            text_chunk=text_chunk_clean,
            section_path=section_path,
            allowed_category_ids=allowed_ids,
        )
        out, raw = await self._call_json_and_raw(
            self.cfg.model_stage2, system, user, RedFlagChunkOutput
        )
        prompt = None
        if return_prompt:
            prompt = {
                "system": system,
                "user": user,
                "sections": sections,
                "router": {
                    "system": router_system,
                    "user": router_user,
                    "sections": router_sections,
                    "raw": router_raw,
                    "parsed": router_out.model_dump(),
                },
            }
        return out, raw, prompt
    async def classify_redflags_for_chunk(
        self,
        *,
        text_chunk: str,
        section_path: str = "",
        return_prompt: bool = False,
    ) -> tuple[RedFlagChunkOutput, str, Optional[dict[str, str]]]:
        """
        Single-classifier mode:
        classify one TEXT CHUNK and return all findings with verbatim evidence (no limit).
        
        Note: text_chunk should already be normalized before calling this function.
        LLM will see normalized text and return quote from it.
        """
        text_chunk_clean = text_chunk
        system, user, sections = self._build_redflag_prompt(
            text_chunk=text_chunk_clean,
            section_path=section_path,
        )

        best_of = int(os.environ.get("LLM_BEST_OF", "2"))
        best_of = max(1, best_of)
        best_out: Optional[RedFlagChunkOutput] = None
        best_raw = ""
        best_count = -1
        all_attempts: list[tuple[RedFlagChunkOutput, str]] = []
        # Try multiple times to get the best result (LLM can be non-deterministic)
        # Always do all attempts to ensure we don't miss any findings
        import logging
        logger = logging.getLogger(__name__)
        for attempt_num in range(best_of):
            out_i, raw_i = await self._call_json_and_raw(
                self.cfg.model_stage2, system, user, RedFlagChunkOutput
            )
            count_i = len(out_i.findings or [])
            all_attempts.append((out_i, raw_i))
            logger.info(
                f"LLM attempt {attempt_num + 1}/{best_of}: returned {count_i} findings. "
                f"Categories: {[f.category for f in (out_i.findings or [])]}. "
                f"Quotes: {[f.quote[:80] + '...' if f.quote and len(f.quote) > 80 else (f.quote or 'EMPTY') for f in (out_i.findings or [])]}"
            )
            if count_i > best_count:
                logger.info(
                    f"Attempt {attempt_num + 1} is better (count {count_i} > {best_count}). "
                    f"Using this result."
                )
                best_out = out_i
                best_raw = raw_i
                best_count = count_i
            else:
                logger.info(
                    f"Attempt {attempt_num + 1} has fewer findings (count {count_i} <= {best_count}). "
                    f"Keeping previous best."
                )
        
        # If we have multiple attempts, try to merge findings from all attempts
        # to avoid losing findings that appear in different attempts
        if len(all_attempts) > 1 and best_out:
            all_findings = []
            seen_quotes = set()
            # Collect unique findings from all attempts (dedupe by quote)
            for out_i, _ in all_attempts:
                for f in (out_i.findings or []):
                    quote_key = (f.category, (f.quote or "").strip().lower()[:100])
                    if quote_key not in seen_quotes:
                        seen_quotes.add(quote_key)
                        all_findings.append(f)
            
            if len(all_findings) > best_count:
                logger.info(
                    f"Merging findings from all {best_of} attempts: "
                    f"best attempt had {best_count}, merged has {len(all_findings)} unique findings. "
                    f"Categories: {[f.category for f in all_findings]}"
                )
                best_out.findings = all_findings
                best_count = len(all_findings)
        
        out = best_out or RedFlagChunkOutput(findings=[])
        raw = best_raw
        logger.info(
            f"Final result after {best_of} attempts: {len(out.findings or [])} findings. "
            f"Categories: {[f.category for f in (out.findings or [])]}"
        )
        
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
        # Keep all findings (no truncation)
        
        # Log findings (only raw output, no text_chunk) for debugging
        if not out.findings:
            logger.warning(
                f"LLM returned empty findings. Temperature={self.cfg.temperature}, "
                f"Raw output length={len(raw)}"
            )
            if raw:
                logger.warning(f"LLM FULL raw output (empty findings): {raw}")
        else:
            logger.info(
                f"LLM returned {len(out.findings)} findings. Raw output length={len(raw)}"
            )
            if raw:
                logger.info(f"LLM FULL raw output: {raw}")
            # Log each finding's raw data with FULL quotes
            for i, f in enumerate(out.findings):
                full_quote = f.quote or ""
                logger.info(
                    f"Finding {i+1}: category={f.category}, quote_length={len(full_quote)}, "
                    f"quote_FULL={full_quote[:300] if full_quote else 'EMPTY'}..."
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
                    if self.cfg.seed is not None:
                        kwargs["seed"] = self.cfg.seed
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
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.error(
                            f"CRITICAL: No JSON object found in LLM response! "
                            f"Raw text length: {len(text)}, "
                            f"First 500 chars: {text[:500]}"
                        )
                        return RedFlagChunkOutput(findings=[]), (last_text or "")
                    raise ValueError("No JSON object found in LLM response")
                
                # Log raw JSON blob before parsing
                if schema == RedFlagChunkOutput:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"Raw JSON blob extracted (length={len(blob)}): {blob[:1000]}")
                
                data = json.loads(blob)
                
                # Log parsed data before validation
                if schema == RedFlagChunkOutput:
                    import logging
                    logger = logging.getLogger(__name__)
                    findings_in_data = data.get('findings', []) if isinstance(data, dict) else []
                    logger.info(
                        f"JSON parsed successfully. Data has {len(findings_in_data)} findings. "
                        f"Data keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}. "
                        f"Findings preview: {[f.get('category', 'NO_CATEGORY') + ': ' + (f.get('quote', '')[:50] or 'NO_QUOTE') for f in findings_in_data[:3]] if findings_in_data else 'EMPTY'}"
                    )
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
                        # If data has findings but parsed has fewer, this is a problem - trigger salvage
                        if data_findings_count > findings_count:
                            logger.error(
                                f"CRITICAL: Data has {data_findings_count} findings but parsed has only {findings_count}! "
                                f"Lost {data_findings_count - findings_count} finding(s) during validation. "
                                f"Data findings: {data.get('findings', [])[:3] if isinstance(data, dict) else 'N/A'}. "
                                f"Triggering salvage logic..."
                            )
                            # Log each finding from data to see why validation failed
                            findings_data = data.get('findings', []) if isinstance(data, dict) else []
                            for i, f_data in enumerate(findings_data):
                                logger.error(
                                    f"Data finding {i+1} that may be lost: "
                                    f"category={f_data.get('category', 'MISSING')}, "
                                    f"quote_preview={f_data.get('quote', '')[:100] if f_data.get('quote') else 'MISSING'}, "
                                    f"keys={list(f_data.keys())}"
                                )
                            if findings_data:
                                # Manually trigger salvage by raising a ValidationError
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
                        f"Validation errors: {ve.errors() if hasattr(ve, 'errors') else 'N/A'}. "
                        f"Attempting to salvage findings..."
                    )
                    # Log detailed validation errors
                    if hasattr(ve, 'errors'):
                        for err in ve.errors():
                            logger.warning(
                                f"Validation error detail: loc={err.get('loc')}, "
                                f"type={err.get('type')}, msg={err.get('msg')}, "
                                f"input_type={type(err.get('input'))}"
                            )
                    if schema == RedFlagChunkOutput and "findings" in data:
                        findings_data = data.get("findings", [])
                        salvaged_findings = []
                        logger.info(
                            f"Attempting to salvage {len(findings_data)} findings from validation error. "
                            f"Findings data: {[f.get('category', 'NO_CATEGORY') + ': ' + (f.get('quote', '')[:50] or 'NO_QUOTE') for f in findings_data[:3]] if findings_data else 'EMPTY'}"
                        )
                        for f_idx, f_data in enumerate(findings_data):
                            try:
                                logger.info(
                                    f"Salvaging finding {f_idx+1}/{len(findings_data)}: "
                                    f"category={f_data.get('category', 'MISSING')}, "
                                    f"quote_preview={f_data.get('quote', '')[:100] if f_data.get('quote') else 'MISSING'}"
                                )
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
                                logger.error(
                                    f"Failed to salvage finding {f_idx+1}: {salvage_err}. "
                                    f"Category: {f_data.get('category', 'MISSING')}, "
                                    f"Quote preview: {f_data.get('quote', '')[:100] if f_data.get('quote') else 'MISSING'}, "
                                    f"consequences_cat was set to: '{consequences_cat}', "
                                    f"risk_assessment: {risk_assessment}, "
                                    f"f_data consequences_category: '{f_data.get('consequences_category')}', "
                                    f"f_data keys: {list(f_data.keys())}"
                                )
                                import traceback
                                logger.error(f"Salvage error traceback: {traceback.format_exc()}")
                                continue
                        if salvaged_findings:
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.info(
                                f"Salvaged {len(salvaged_findings)} findings from validation error. "
                                f"Categories: {[f.category for f in salvaged_findings]}. "
                                f"Quotes: {[f.quote[:100] if f.quote else 'EMPTY' for f in salvaged_findings]}"
                            )
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

    async def regenerate_legal_references(
        self,
        finding: RedFlagFindingLLM,
        text_chunk: str,
        section_path: str = "",
    ) -> list[str]:
        """
        Regenerate legal references for a finding that has invalid references.
        
        This is called when ALL legal_references are invalid, indicating LLM hallucination.
        We ask the LLM to provide correct legal references based on the finding's category and quote.
        
        Args:
            finding: The finding with invalid legal references
            text_chunk: The original text chunk where the finding was found
            section_path: Section path for context
        
        Returns:
            List of regenerated legal references (may still be empty if LLM can't provide valid ones)
        """
        system = "You are a legal expert assistant. Provide ONLY valid UK legal references in JSON format."
        
        user = f"""A legal finding was identified in a contract, but the legal references provided were invalid/non-existent.

FINDING DETAILS:
- Category: {finding.category}
- Quote: {finding.quote[:500]}
- Explanation: {finding.explanation[:300]}

INVALID REFERENCES (DO NOT USE THESE):
{chr(10).join(f"- {ref}" for ref in (finding.legal_references or []))}

TEXT CHUNK (for context):
{text_chunk[:2000]}

TASK:
Provide ONLY valid UK legal references that are relevant to this finding. 
- Use real UK statutes (e.g., Consumer Rights Act 2015, Housing Act 2004, Tenant Fees Act 2019)
- Use "UK common law" or "English common law" if applicable
- Do NOT make up or invent statutes
- If no valid references are relevant, return an empty list

Return JSON format:
{{
  "legal_references": ["Consumer Rights Act 2015", "UK common law on penalty clauses"]
}}

OUTPUT JSON:"""
        
        try:
            # Use a simple schema for just legal references
            class LegalRefsOutput(BaseModel):
                legal_references: list[str] = []
            
            output, _ = await self._call_json_and_raw(
                self.cfg.model_stage2, system, user, LegalRefsOutput, retries=1
            )
            
            # Validate the regenerated references
            regenerated = output.legal_references or []
            valid_regenerated = filter_valid_legal_references(regenerated)
            
            if len(regenerated) > len(valid_regenerated):
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Regenerated legal references still contain invalid ones. "
                    f"Invalid: {[r for r in regenerated if r not in valid_regenerated]}"
                )
            
            return valid_regenerated
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to regenerate legal references: {e}")
            return []
