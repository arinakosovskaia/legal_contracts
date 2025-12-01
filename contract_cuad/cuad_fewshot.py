"""Few-shot support built directly from the CUAD QA dataset."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import difflib
from typing import Dict, Iterable, List, Optional

from datasets import Dataset

from .categories import Category

LOGGER = logging.getLogger(__name__)

PACKAGE_DATA_PATH = Path(__file__).resolve().parent / "data" / "cuad_v1.json"
REPO_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cuad_v1.json"


def _read_json_resource(source: str | os.PathLike) -> Dict:
    source_path = os.fspath(source)
    with open(source_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_path_case_insensitive(candidate: Path) -> Optional[Path]:
    if candidate.exists():
        return candidate
    parent = candidate.parent
    if not parent.exists():
        return None
    target_name = candidate.name.lower()
    for child in parent.iterdir():
        if child.name.lower() == target_name:
            return child
    return None


def load_cuad_qa(split: str = "train", source: str | os.PathLike | None = None) -> Dataset:
    """Load the CUAD QA dataset by flattening the local CUAD JSON file."""

    if split != "train":
        raise ValueError("Only the 'train' split is available for the CUAD QA loader.")

    candidate_paths: List[Path] = []
    if source:
        candidate_paths.append(Path(source))
    env_override = os.environ.get("CUAD_QA_JSON")
    if env_override:
        candidate_paths.append(Path(env_override))
    candidate_paths.extend([REPO_DATA_PATH, PACKAGE_DATA_PATH])

    resolved_path: Optional[Path] = None
    for path in candidate_paths:
        if not path:
            continue
        resolved = _resolve_path_case_insensitive(path)
        if resolved:
            resolved_path = resolved
            break
    if resolved_path is None:
        raise FileNotFoundError(
            "cuad_v1.json not found. Place it under the repository-level 'data/' directory, "
            "inside the installed package at 'contract_cuad/data/', or point CUAD_QA_JSON "
            "to its location."
        )

    LOGGER.info("Loading CUAD QA JSON from %s", resolved_path)
    payload = _read_json_resource(resolved_path)
    records: List[Dict[str, object]] = []
    for document in payload.get("data", []):
        for paragraph in document.get("paragraphs", []):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                answers = qa.get("answers", []) or []
                records.append(
                    {
                        "question": qa.get("question", ""),
                        "context": context,
                        "answers": {
                            "text": [ans.get("text", "") for ans in answers],
                            "answer_start": [ans.get("answer_start", 0) for ans in answers],
                        },
                    }
                )
    if not records:
        raise RuntimeError(
            "CUAD QA JSON appears empty. Verify the file matches the CUAD data schema."
        )
    LOGGER.info("Loaded %s QA records from CUAD JSON", len(records))
    return Dataset.from_list(records)


def _normalize_label(text: str) -> str:
    if not text:
        return ""
    s = text.strip()
    if s.lower().startswith("category:"):
        s = s.split(":", 1)[1].strip()
    for sep in ("–", "—", "-"):
        if sep in s:
            s = s.split(sep, 1)[0].strip()
            break
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _match_question_to_category(question: str, categories: Iterable[Category]) -> Optional[Category]:
    q_norm = _normalize_label(question)
    if not q_norm:
        return None
    normalized_map: Dict[str, Category] = {}
    for category in categories:
        cat_norm = _normalize_label(category.name)
        if not cat_norm:
            continue
        normalized_map[cat_norm] = category
    if q_norm in normalized_map:
        return normalized_map[q_norm]
    for cat_norm, cat in normalized_map.items():
        if cat_norm in q_norm or q_norm in cat_norm:
            return cat
    candidates = list(normalized_map.keys())
    match = difflib.get_close_matches(q_norm, candidates, n=1, cutoff=0.8)
    if match:
        return normalized_map[match[0]]
    LOGGER.debug("Unable to match question '%s' (normalized '%s') to CUAD categories", question, q_norm)
    return None


def _extract_snippet(context: str, answer_start: int, answer_text: str, window: int = 240) -> str:
    if not context:
        return answer_text
    start = max(0, answer_start - window)
    end = min(len(context), answer_start + len(answer_text) + window)
    snippet = context[start:end].strip()
    return snippet or answer_text


def build_fewshot_examples(
    categories: List[Category],
    *,
    dataset: Optional[Dataset] = None,
    max_examples_per_category: int = 1,
    max_total_examples: int = 40,
    context_window: int = 240,
) -> Dict[str, List[Dict[str, str]]]:
    """Construct labeled clause snippets keyed by category."""

    LOGGER.info(
        "Building few-shot examples (max %s per category, %s total)",
        max_examples_per_category,
        max_total_examples,
    )
    ds = dataset or load_cuad_qa()
    fewshot: Dict[str, List[Dict[str, str]]] = {cat.name: [] for cat in categories}
    total = 0
    for record in ds:
        question = record.get("question", "")
        context = record.get("context", "")
        answers = record.get("answers", {}) or {}
        answer_texts = answers.get("text") or []
        answer_starts = answers.get("answer_start") or []
        if not question or not context or not answer_texts:
            continue
        matched_category = _match_question_to_category(question, categories)
        if not matched_category:
            continue
        bucket = fewshot.get(matched_category.name)
        if bucket is None or len(bucket) >= max_examples_per_category:
            continue
        snippet = _extract_snippet(context, int(answer_starts[0]), answer_texts[0], window=context_window)
        bucket.append({"clause": snippet, "question": question})
        total += 1
        if total >= max_total_examples:
            break
    result = {category: examples for category, examples in fewshot.items() if examples}
    LOGGER.info(
        "Prepared %s total few-shot examples across %s categories",
        sum(len(v) for v in result.values()),
        len(result),
    )
    return result
