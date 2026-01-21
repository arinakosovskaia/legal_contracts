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

        # Mapping for the single-classifier mode (subset of CUAD categories)
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

    def _build_redflag_fewshot_examples(self) -> str:
        """
        Build few-shot examples for the single-classifier prompt from CUAD cache.
        Evidence is sourced from CUAD_v1.json answers and is guaranteed to be inside the snippet.
        """
        blocks: list[str] = []
        # Deterministic order by category name
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
            for ex in examples:
                snippet = ex.get("clause", "")
                evidence = ex.get("evidence", "")
                if not snippet or not evidence or evidence not in snippet:
                    continue
                # Minimal, category-grounded reasoning templates (few-shot only).
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
                        "Evidence contains 'most favored' / 'at least as favorable' parity language.",
                        "This commits the party to best-terms treatment compared to others.",
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

                blocks.append(
                    "TEXT CHUNK:\n"
                    f"{snippet}\n"
                    "OUTPUT:\n"
                    + json.dumps(
                        {
                            "findings": [
                                {"category": name, "evidence": evidence, "reasoning": reasoning[:2]}
                            ]
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return "\n".join(blocks).strip()

    def _build_redflag_prompt(
        self,
        *,
        text_chunk: str,
        section_path: str,
        max_findings: int,
    ) -> tuple[str, str]:
        categories = list(self.redflag_category_id_by_name.keys())
        fewshot = self._redflag_fewshot_block
        section_prefix = f"CONTEXT PREFIX: {section_path}\n\n" if section_path else ""

        system = "You are a legal clause classifier for red flag clause types. Output STRICT JSON only."
        user = f"""
You are a legal clause classifier for “red flag” clause types.

Knowing only the provided TEXT CHUNK, identify whether it contains any of these categories:
{chr(10).join("- " + c for c in categories)}

You must return BOTH:
1) the category (or categories) AND
2) verbatim evidence quoted from the provided TEXT CHUNK
3) short reasoning grounded only in the evidence

Rules
- Evidence MUST be copied verbatim from the provided TEXT CHUNK. Do not paraphrase.
- Return up to {max_findings} findings per chunk (only the most salient).
- If nothing matches, return {{"findings": []}}.
- If you suspect a category but cannot quote clear evidence from the chunk, do NOT guess; return no finding for that category.
- Keep reasoning brief (1–3 bullets) and tied directly to the evidence.

Output JSON schema:
{{"findings":[{{"category":"<one of the categories above>","evidence":"<verbatim quote from TEXT CHUNK>","reasoning":["<bullet>"]}}]}}

========================
FEW-SHOT EXAMPLES (from CUAD_v1.json)
========================
{fewshot}

========================
NOW CLASSIFY THIS INPUT
========================
{section_prefix}TEXT CHUNK:
{text_chunk}
OUTPUT JSON:
""".strip()
        return system, user

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
        """
        system, user = self._build_redflag_prompt(
            text_chunk=text_chunk,
            section_path=section_path,
            max_findings=max_findings,
        )

        out, raw = await self._call_json_and_raw(self.cfg.model_stage2, system, user, RedFlagChunkOutput)
        # Enforce constraints server-side:
        cleaned: list[RedFlagFindingLLM] = []
        for f in out.findings[:max_findings]:
            # Evidence must be verbatim substring of the chunk.
            if not f.evidence or f.evidence not in text_chunk:
                continue
            cleaned.append(f)
        out.findings = cleaned[:max_findings]
        prompt = {"system": system, "user": user} if return_prompt else None
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
                    resp = await self.client.chat.completions.create(
                        model=model,
                        messages=self._build_messages(system=system, user=user),
                        temperature=0.0 if attempt > 0 else 0.2,
                        **kwargs,
                    )
                text = (resp.choices[0].message.content or "").strip()
                last_text = text
                blob = extract_first_json_object(text) or ""
                data = json.loads(blob)
                parsed = schema.model_validate(data)
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

        # Normalize and validate selected categories (must pass threshold)
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
            # Pass 2 (strict): try once more if the model invented category ids.
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

