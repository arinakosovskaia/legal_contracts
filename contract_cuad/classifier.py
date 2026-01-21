"""Prompt construction and classification pipelines."""

from __future__ import annotations

import json
import re
import logging
import difflib
from typing import Dict, Iterable, List, Optional

from tqdm import tqdm

from .categories import Category, build_category_lookup
from .chunking import chunk_contract_into_clauses
from .config import FewShotConfig
from .llm_client import LLMClient

LOGGER = logging.getLogger(__name__)
_JSON_PATTERN = re.compile(r"\{", re.MULTILINE)
FALLBACK_NONE_REASON = "Model did not return any valid categories."


def build_categories_block(categories: Iterable[Category]) -> str:
    """Format CUAD categories into prompt-friendly text."""

    parts = [f"{cat.index}. {cat.name}: {cat.description}" for cat in categories]
    return "\n".join(parts)


def build_fewshot_block(
    fewshot_examples: Optional[Dict[str, List]],
    max_examples_per_category: int = 1,
) -> str:
    if not fewshot_examples:
        return ""
    lines: List[str] = []
    counter = 1
    for category, snippets in fewshot_examples.items():
        used_for_category = 0
        for snippet in snippets:
            if used_for_category >= max_examples_per_category:
                break
            clause_text = snippet.get("clause") if isinstance(snippet, dict) else str(snippet)
            question = snippet.get("question") if isinstance(snippet, dict) else None
            example_lines = [f"Example {counter}:", f"Category: {category}"]
            if question:
                example_lines.append(f"Prompt: {question}")
            example_lines.append("Clause:\n\"\"\"" + clause_text.strip() + "\"\"\"")
            lines.append("\n".join(example_lines))
            counter += 1
            used_for_category += 1
    return "\n\n".join(lines)


def _extract_first_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    start_positions = [m.start() for m in _JSON_PATTERN.finditer(text)]
    for start in start_positions:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            char = text[idx]
            if char == "\\" and not escape:
                escape = True
                continue
            if char == '"' and not escape:
                in_string = not in_string
            if not in_string:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : idx + 1]
            escape = False
    return None


def parse_llm_json(output_text: str) -> Dict:
    """Attempt to parse JSON from model output, returning diagnostics if needed."""

    candidate = _extract_first_json_object(output_text)
    if not candidate:
        return {"categories": [], "error": "PARSING_ERROR", "raw_output": output_text}
    try:
        data = json.loads(candidate)
        if isinstance(data, dict) and "categories" in data and isinstance(data["categories"], list):
            return data
        if isinstance(data, dict) and "category" in data:
            return {"categories": [data]}
        return {
            "categories": [],
            "error": "PARSING_ERROR",
            "raw_output": output_text,
            "malformed_json": candidate,
        }
    except json.JSONDecodeError:
        LOGGER.debug("Malformed JSON from LLM: %r", candidate)
        return {
            "categories": [],
            "error": "PARSING_ERROR",
            "raw_output": output_text,
            "malformed_json": candidate,
        }


def _normalize_category_predictions(
    predictions: Optional[List[Dict]], category_lookup: Dict[str, Category]
) -> List[Dict[str, object]]:
    """Ensure we always return 1-3 well-formed category entries."""

    normalized: List[Dict[str, object]] = []
    collected: List[Dict[str, object]] = []
    none_added = False
    if isinstance(predictions, list):
        for entry in predictions:
            if not isinstance(entry, dict):
                continue
            raw_name = str(entry.get("category", "")).strip()
            if not raw_name:
                continue
            cleaned_name = _clean_category_label(raw_name)
            if cleaned_name.lower() == "none":
                if none_added:
                    continue
                collected.append(
                    {
                        "category": "NONE",
                        "category_index": 0,
                        "confidence": 0.0,
                        "reason": entry.get("reason", "") or "",
                    }
                )
                none_added = True
                continue
            category = _resolve_category(raw_name, category_lookup)
            if not category:
                continue
            confidence_value = entry.get("confidence", 0.0)
            try:
                confidence = float(confidence_value)
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
            reason = str(entry.get("reason", "")).strip()
            collected.append(
                {
                    "category": category.name,
                    "category_index": category.index,
                    "confidence": confidence,
                    "reason": reason,
                }
            )
    normalized = sorted(collected, key=lambda item: item.get("confidence", 0.0), reverse=True)[:3]
    if not normalized:
        LOGGER.warning("No valid category predictions after normalization; falling back to NONE.")
        normalized.append(
            {
                "category": "NONE",
                "category_index": 0,
                "confidence": 0.0,
                "reason": FALLBACK_NONE_REASON,
            }
        )
    return normalized


def _resolve_category(name: str, category_lookup: Dict[str, Category]) -> Optional[Category]:
    cleaned = _clean_category_label(name)
    cleaned_lower = cleaned.lower()
    if cleaned_lower in category_lookup:
        return category_lookup[cleaned_lower]
    for cat_name, cat in category_lookup.items():
        if cat_name in cleaned_lower or cleaned_lower in cat_name:
            return cat
    candidates = list(category_lookup.keys())
    match = difflib.get_close_matches(cleaned_lower, candidates, n=1, cutoff=0.7)
    if match:
        return category_lookup[match[0]]
    LOGGER.warning("Unable to map category name '%s' to CUAD categories", name)
    return None


def _clean_category_label(name: str) -> str:
    cleaned = name.strip().strip("\"'")
    if ":" in cleaned:
        cleaned = cleaned.split(":", 1)[-1].strip()
    return cleaned


def classify_clause(
    clause: str,
    categories: List[Category],
    llm: LLMClient,
    fewshot_examples: Optional[Dict[str, List]] = None,
    fewshot_config: Optional[FewShotConfig] = None,
    *,
    cat_block: Optional[str] = None,
    fewshot_block: Optional[str] = None,
    category_lookup: Optional[Dict[str, Category]] = None,
) -> Dict:
    """Classify a clause and return normalized top-3 CUAD predictions.

    Returns:
        Dict with keys:
            - categories: list of 1–3 predictions sorted by confidence.
            - raw_output: raw string returned by the LLM.
            - error: optional parsing error indicator (e.g., "PARSING_ERROR" or None).
    """

    fewshot_cfg = fewshot_config or FewShotConfig()
    cat_block = cat_block or build_categories_block(categories)
    if fewshot_block is None:
        fewshot_block = build_fewshot_block(
            fewshot_examples,
            max_examples_per_category=fewshot_cfg.max_examples_per_category,
        )
    lookup = category_lookup or build_category_lookup(categories)

    def _run_pass(strict: bool) -> Dict:
        instruction = _build_instruction(strict)
        parts: List[str] = []
        if fewshot_block:
            parts.append("Labeled examples:\n" + fewshot_block)
        parts.append("Available CUAD categories:\n" + cat_block)
        parts.append("Clause to label:\n\"\"\"" + clause.strip() + "\"\"\"")
        parts.append(instruction)
        messages = [
            {
                "role": "system",
                "content": "You are a contract analyst specialized in CUAD contract clause categories.",
            },
            {"role": "user", "content": "\n\n".join(parts)},
        ]
        raw_output = llm.generate_chat(messages)
        parsed = parse_llm_json(raw_output)
        predictions = parsed.get("categories") if isinstance(parsed, dict) else None
        normalized = _normalize_category_predictions(predictions, lookup)
        error = parsed.get("error") if isinstance(parsed, dict) else None
        return {"categories": normalized, "raw_output": raw_output, "error": error}

    result = _run_pass(strict=False)
    if _needs_retry(result["categories"], result.get("error")):
        LOGGER.info("Retrying clause classification with strict prompt.")
        strict_result = _run_pass(strict=True)
        if not _needs_retry(strict_result["categories"], strict_result.get("error")):
            result = strict_result
        else:
            result = strict_result
    return result


def classify_contract_text(
    text: str,
    categories: List[Category],
    llm: LLMClient,
    *,
    fewshot_examples: Optional[Dict[str, List]] = None,
    fewshot_config: Optional[FewShotConfig] = None,
    max_clause_tokens: int = 450,
    max_clauses: Optional[int] = None,
    paragraph_overlap: int = 0,
    show_progress: bool = True,
) -> List[Dict]:
    """Run clause classification across an entire contract string."""

    tokenizer = llm.tokenizer
    clauses = chunk_contract_into_clauses(
        text,
        tokenizer=tokenizer,
        max_tokens=max_clause_tokens,
        paragraph_overlap=paragraph_overlap,
    )
    if max_clauses is not None:
        clauses = clauses[:max_clauses]
    LOGGER.info("Split contract into %s clauses (max_clauses=%s)", len(clauses), max_clauses)
    fewshot_cfg = fewshot_config or FewShotConfig()
    cat_block = build_categories_block(categories)
    fewshot_block = build_fewshot_block(
        fewshot_examples,
        max_examples_per_category=fewshot_cfg.max_examples_per_category,
    )
    category_lookup = build_category_lookup(categories)
    iterator: Iterable = enumerate(clauses)
    if show_progress:
        iterator = tqdm(iterator, total=len(clauses), desc="Classifying clauses")
    results: List[Dict] = []
    for idx, clause in iterator:
        clause_result = classify_clause(
            clause,
            categories,
            llm,
            fewshot_examples=fewshot_examples,
            fewshot_config=fewshot_cfg,
            cat_block=cat_block,
            fewshot_block=fewshot_block,
            category_lookup=category_lookup,
        )
        categories_payload = clause_result["categories"]
        primary = categories_payload[0] if categories_payload else {}
        results.append(
            {
                "clause_id": idx,
                "clause_text": clause,
                "categories": categories_payload,
                "primary_category": primary.get("category"),
                "primary_category_index": primary.get("category_index"),
                "primary_confidence": primary.get("confidence"),
                "raw_output": clause_result.get("raw_output", ""),
                "error": clause_result.get("error"),
            }
        )
    LOGGER.info("Completed classification for %s clauses", len(results))
    return results


def _build_instruction(strict: bool) -> str:
    base = (
        "Your task is to identify up to the top 3 most relevant CUAD categories for the clause. "
        "Do not list categories that are clearly irrelevant. "
        "If only one or two categories apply, return fewer than three. "
        "Order the categories by relevance (most relevant first). "
        "The ONLY valid category names are EXACTLY those listed above under 'Available CUAD categories', plus the special label NONE. "
        "Do NOT invent new category names. For each prediction, either choose the single closest category from the CUAD list or use NONE if no CUAD category reasonably applies. "
        "If no CUAD category fits at all, return exactly one entry with category=\"NONE\" and category_index=0. "
        'Return a single JSON object with a field "categories" that is an array of 1 to 3 objects, '
        'each containing: "category" (exact CUAD name or "NONE"), "category_index" (1-based CUAD index or 0 for NONE), '
        '"confidence" (0-1 float), and "reason" (short explanation). '
        "Do not output anything before or after the JSON object."
    )
    if strict:
        base += (
            " This is a STRICT validation pass. You MUST choose labels only from the CUAD list or NONE. "
            "If you are unsure, choose NONE rather than inventing a new label."
        )
    return base


def _needs_retry(categories: List[Dict], error: Optional[str]) -> bool:
    if error:
        return False
    if not categories:
        return True
    if len(categories) == 1:
        entry = categories[0]
        if (
            entry.get("category") == "NONE"
            and entry.get("reason") == FALLBACK_NONE_REASON
        ):
            return True
    return False
