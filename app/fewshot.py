from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

LOGGER = logging.getLogger("demo.fewshot")


def _normalize_label(text: str) -> str:
    """
    Normalize a label/question fragment to a canonical form for matching.
    - strip whitespace
    - lowercase
    - drop leading "Category:" prefix if present
    - drop anything after dash/emdash (used in 'Category: X – Does the contract...')
    - collapse non-alphanumeric sequences to a single space
    - collapse multiple spaces
    """
    if not text:
        return ""
    s = text.strip()
    if s.lower().startswith("category:"):
        s = s.split(":", 1)[1].strip()
    for sep in ["–", "—", "-"]:
        if sep in s:
            s = s.split(sep, 1)[0].strip()
            break
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = " ".join(s.split())
    return s


def _resolve_cuad_json_path(data_dir: Path) -> Path:
    """
    Resolve CUAD QA JSON path.
    We accept either:
    - data/CUAD_v1/CUAD_v1.json (official bundle)
    - data/cuad_v1.json (convenience copy)
    """
    env = os.environ.get("CUAD_QA_JSON")
    if env:
        p = Path(env)
        if p.exists():
            return p
    candidates = [
        data_dir / "CUAD_v1" / "CUAD_v1.json",
        data_dir / "CUAD_v1" / "cuad_v1.json",
        data_dir / "cuad_v1.json",
        data_dir / "CUAD_v1.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "CUAD QA JSON not found. Put it at data/CUAD_v1/CUAD_v1.json or data/cuad_v1.json, "
        "or set CUAD_QA_JSON=/path/to/CUAD_v1.json"
    )


def resolve_cuad_json_path(data_dir: Path) -> Path:
    """Public alias for resolving the CUAD_v1.json location."""
    return _resolve_cuad_json_path(data_dir)


def cuad_file_meta(path: Path) -> dict[str, Any]:
    """
    Lightweight provenance metadata for CUAD_v1.json.
    Avoids hashing/reading the full file (CUAD can be large).
    """
    st = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(st.st_size),
        "mtime": float(st.st_mtime),
    }


def _match_question_to_category_id(question: str, categories_payload: dict[str, Any]) -> Optional[str]:
    q_norm = _normalize_label(question)
    if not q_norm:
        return None
    # normalized category name -> id
    norm_to_id: dict[str, str] = {}
    for c in categories_payload.get("categories", []):
        name = str(c.get("name", ""))
        cid = str(c.get("id", ""))
        cn = _normalize_label(name)
        if cn and cid:
            norm_to_id[cn] = cid
    if q_norm in norm_to_id:
        return norm_to_id[q_norm]
    # substring match
    for cn, cid in norm_to_id.items():
        if cn in q_norm or q_norm in cn:
            return cid
    # small fuzzy match (difflib)
    try:
        import difflib

        candidates = list(norm_to_id.keys())
        m = difflib.get_close_matches(q_norm, candidates, n=1, cutoff=0.8)
        if m:
            return norm_to_id[m[0]]
    except Exception:
        pass
    return None


def _snippet_around(context: str, answer_start: int, answer_text: str, *, window: int) -> str:
    if not context:
        return ""
    ctx = (context or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n")
    start = max(0, int(answer_start) - window)
    end = min(len(ctx), int(answer_start) + max(len(answer_text), 1) + window)
    # Expand to nearest whitespace boundaries to avoid starting mid-word ("mage" instead of "damage").
    while start > 0 and start < len(ctx) and ctx[start].isalnum() and ctx[start - 1].isalnum():
        start -= 1
    while end < len(ctx) and end > 0 and ctx[end - 1].isalnum() and ctx[end].isalnum():
        end += 1
    snippet = ctx[start:end]
    # Remove standalone page-number lines like "- 11 -" and collapse huge blank areas.
    snippet = re.sub(r"^\s*-\s*\d+\s*-\s*$", "", snippet, flags=re.MULTILINE)
    snippet = re.sub(r"[ \t]+\n", "\n", snippet)
    snippet = re.sub(r"\n{3,}", "\n\n", snippet)
    return snippet.strip()


def build_fewshot_by_category(
    *,
    categories_payload: dict[str, Any],
    cuad_json_path: Path,
    max_examples_per_category: int = 2,
    context_window: int = 240,
    only_category_ids: Optional[set[str]] = None,
) -> dict[str, list[dict[str, str]]]:
    """
    Build few-shot reference snippets per category_id from CUAD SQuAD-style JSON.
    Output schema mirrors `contract_cuad` conceptually:
      {category_id: [{"clause": "...", "question": "...", "evidence": "..."}]}
    """
    payload = json.loads(cuad_json_path.read_text(encoding="utf-8"))
    out: dict[str, list[dict[str, str]]] = {}
    for c in categories_payload.get("categories", []):
        cid = str(c.get("id", "")).strip()
        if not cid:
            continue
        if only_category_ids is not None and cid not in only_category_ids:
            continue
        out[cid] = []
    if only_category_ids is not None:
        want = sorted(list(only_category_ids))
        have = sorted(list(out.keys()))
        if not have:
            LOGGER.warning(
                "Few-shot: none of requested category_ids matched categories payload. requested=%s",
                want,
            )
        else:
            missing = sorted([x for x in want if x not in set(have)])
            if missing:
                LOGGER.info("Few-shot: some requested category_ids missing from categories payload: %s", missing)

    total_added = 0
    def done() -> bool:
        if not out:
            return True
        return all(len(v) >= max_examples_per_category for v in out.values())

    for doc in payload.get("data", []):
        if done():
            break
        for para in doc.get("paragraphs", []):
            if done():
                break
            context = para.get("context", "") or ""
            for qa in para.get("qas", []):
                if done():
                    break
                question = qa.get("question", "") or ""
                cid = _match_question_to_category_id(question, categories_payload)
                if not cid or cid not in out:
                    continue
                if len(out[cid]) >= max_examples_per_category:
                    continue
                answers = qa.get("answers", []) or []
                if not answers:
                    continue
                a0 = answers[0] or {}
                ans_text = (a0.get("text") or "").strip()
                ans_start = a0.get("answer_start", 0) or 0
                if not ans_text:
                    continue
                clause = _snippet_around(context, ans_start, ans_text, window=context_window)
                if not clause:
                    continue
                # Ensure the evidence span is actually present in the snippet (verbatim).
                if ans_text not in clause:
                    continue
                out[cid].append({"clause": clause, "question": question, "evidence": ans_text})
                total_added += 1

    non_empty = sum(1 for v in out.values() if v)
    LOGGER.info("Few-shot built: %s examples across %s categories", total_added, non_empty)
    return out


def save_fewshot_by_category(
    fewshot: dict[str, list[dict[str, str]]],
    out_dir: Path,
    *,
    source_meta: Optional[dict[str, Any]] = None,
    only_category_ids: Optional[list[str]] = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"version": "fewshot_v2", "categories": {}}
    if source_meta is not None:
        manifest["source"] = source_meta
    if only_category_ids is not None:
        manifest["only_category_ids"] = list(only_category_ids)
    for cid, items in fewshot.items():
        fname = f"{cid}.json"
        (out_dir / fname).write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest["categories"][cid] = {"file": fname, "count": len(items)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir / "manifest.json"


def load_fewshot_by_category(out_dir: Path) -> Optional[dict[str, list[dict[str, str]]]]:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("version") or "") != "fewshot_v2":
        return None
    out: dict[str, list[dict[str, str]]] = {}
    for cid, meta in (manifest.get("categories") or {}).items():
        f = out_dir / meta.get("file", "")
        if f.exists():
            out[cid] = json.loads(f.read_text(encoding="utf-8"))
    return out


def load_or_build_fewshot(
    *,
    data_dir: Path,
    categories_payload: dict[str, Any],
    enabled: bool,
    max_examples_per_category: int,
    context_window: int,
    out_dir: Path,
    only_category_ids: Optional[set[str]] = None,
) -> Optional[dict[str, list[dict[str, str]]]]:
    if not enabled:
        return None
    cached = load_fewshot_by_category(out_dir)
    if cached is not None:
        non_empty = sum(1 for v in cached.values() if v)
        total = sum(len(v) for v in cached.values())
        if total > 0:
            LOGGER.info("Few-shot loaded from cache: %s examples across %s categories", total, non_empty)
            return cached
        # If cache exists but is empty, treat as cache-miss and rebuild.
        LOGGER.warning("Few-shot cache found but empty. Rebuilding from CUAD_v1.json ...")

    cuad_path = resolve_cuad_json_path(data_dir)
    LOGGER.info("Few-shot cache not found. Building from %s ...", cuad_path)
    fewshot = build_fewshot_by_category(
        categories_payload=categories_payload,
        cuad_json_path=cuad_path,
        max_examples_per_category=max_examples_per_category,
        context_window=context_window,
        only_category_ids=only_category_ids,
    )
    save_fewshot_by_category(
        fewshot,
        out_dir,
        source_meta=cuad_file_meta(cuad_path),
        only_category_ids=sorted(list(only_category_ids)) if only_category_ids is not None else None,
    )
    return fewshot

