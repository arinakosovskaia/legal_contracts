from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _clean_category_name(raw: str) -> str:
    s = (raw or "").strip()
    if s.lower().startswith("category:"):
        s = s.split(":", 1)[1].strip()
    return s


def _clean_description(raw: str) -> str:
    s = (raw or "").strip()
    if s.lower().startswith("description:"):
        s = s.split(":", 1)[1].strip()
    return s


def _slug_id(name: str) -> str:
    s = name.strip().lower()
    s = s.replace("&", " and ")
    s = _NON_ALNUM.sub("_", s)
    s = s.strip("_")
    s = re.sub(r"_+", "_", s)
    return s or "unknown"


def load_categories_from_csv(csv_path: Path, *, version: str = "cuad_v1_41_from_csv") -> dict[str, Any]:
    """
    Build a categories payload compatible with `configs/categories.json`.
    Supports both CUAD's `category_descriptions.csv` and a simple `id,name,description` CSV.
    """
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # Handle UTF-8 BOM and stray whitespace in headers (common when CSV is saved from Excel).
        if reader.fieldnames:
            reader.fieldnames = [(fn or "").lstrip("\ufeff").strip() for fn in reader.fieldnames]
        for r in reader:
            rows.append({k: (v or "") for k, v in r.items()})

    cats: list[dict[str, str]] = []
    used: set[str] = set()
    headers = {h.lower() for h in (rows[0].keys() if rows else [])}
    simple_schema = "id" in headers and "name" in headers
    for r in rows:
        if simple_schema:
            name = (r.get("name", "") or "").strip()
            desc_raw = r.get("description", "")
            cid = (r.get("id", "") or "").strip()
            if not name and not cid:
                continue
            if not cid:
                cid = _slug_id(name)
            # name and cid are already set above
        else:
            name_raw = r.get("Category (incl. context and answer)", "") or r.get("Category", "")
            desc_raw = r.get("Description", "")
            name = _clean_category_name(name_raw)
            if not name:
                continue
            cid = _slug_id(name)
        
        # Common validation (should not be needed for simple_schema, but keep for safety)
        if not name:
            continue
        if not cid:
            cid = _slug_id(name)
        base = cid
        i = 2
        while cid in used:
            cid = f"{base}_{i}"
            i += 1
        used.add(cid)
        cats.append({"id": cid, "name": name or cid, "description": _clean_description(desc_raw)})

    if not cats:
        raise ValueError(
            "Parsed 0 categories from category_descriptions.csv. "
            "Check file encoding (UTF-8 BOM) and headers."
        )
    return {"version": version, "categories": cats, "source": str(csv_path)}


def write_categories_json(payload: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

