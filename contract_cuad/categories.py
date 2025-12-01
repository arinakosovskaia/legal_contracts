"""Utilities for loading CUAD categories and metadata."""

from __future__ import annotations

import os
import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)

CATEGORY_CSV_URL = (
    "https://raw.githubusercontent.com/TheAtticusProject/cuad/main/data/category_descriptions.csv"
)
LOCAL_CATEGORY_CSV = Path(__file__).resolve().parent / "data" / "category_descriptions.csv"


_DATACLASS_KW = {"slots": True} if sys.version_info >= (3, 10) else {}


@dataclass(frozen=True, **_DATACLASS_KW)
class Category:
    """Simple representation of a CUAD category."""

    index: int
    name: str
    description: str


def _resolve_column(df: pd.DataFrame, candidates: Sequence[str], fallback: str | None = None) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    if fallback is not None:
        return fallback
    raise ValueError(f"None of the candidate columns {candidates} found in DataFrame")


def load_cuad_categories(source: str | os.PathLike | None = None) -> List[Category]:
    """Load CUAD category descriptions with smart source ordering.

    Priority:
    1. Explicit ``source`` argument.
    2. ``CUAD_CATEGORY_CSV`` environment variable.
    3. Packaged ``contract_cuad/data/category_descriptions.csv`` (if present).
    4. Remote GitHub CSV.
    """

    for candidate in _candidate_sources(source):
        try:
            df = pd.read_csv(candidate)
            break
        except Exception as exc:
            last_exc = exc
    else:  # pragma: no cover - triggered when all sources fail
        raise RuntimeError(
            "Failed to load CUAD category descriptions. Provide a local CSV via the "
            "CUAD_CATEGORY_CSV environment variable or place the file at "
            f"{LOCAL_CATEGORY_CSV}."
        ) from last_exc
    name_col = _resolve_column(df, ["Category", "Category Name", "Question"], df.columns[0])
    desc_col = _resolve_column(df, ["Description", "Category Description", "Answer"], df.columns[1])
    index_col = None
    if any(c in df.columns for c in ("Index", "Id", "Category #")):
        index_col = _resolve_column(df, ["Index", "Id", "Category #"], None)

    LOGGER.info("Loaded CUAD categories from %s", candidate)
    categories: List[Category] = []
    for idx, row in df.iterrows():
        category_index = int(row[index_col]) if index_col else int(idx + 1)
        name = str(row[name_col]).strip()
        description = str(row[desc_col]).strip()
        if not name:
            continue
        categories.append(Category(index=category_index, name=name, description=description))
    LOGGER.info("Parsed %s CUAD categories", len(categories))
    return categories


def build_category_lookup(categories: Sequence[Category]) -> Dict[str, Category]:
    """Create a case-insensitive mapping from name to Category."""

    return {cat.name.lower(): cat for cat in categories}


def index_to_category(categories: Sequence[Category]) -> Dict[int, Category]:
    """Return a mapping of category indices to Category instances."""

    return {cat.index: cat for cat in categories}


def ensure_category_name(categories: Sequence[Category], name: str) -> Category:
    """Retrieve a category by (case-insensitive) name, raising if missing."""

    lookup = build_category_lookup(categories)
    key = name.lower()
    if key not in lookup:
        raise KeyError(f"Category '{name}' not found in CUAD descriptions")
    return lookup[key]


def _candidate_sources(source: str | os.PathLike | None) -> List[str | os.PathLike]:
    env_override = os.environ.get("CUAD_CATEGORY_CSV")
    candidates: List[str | os.PathLike] = []
    if source:
        candidates.append(source)
    if env_override:
        candidates.append(env_override)
    if LOCAL_CATEGORY_CSV.exists():
        candidates.append(LOCAL_CATEGORY_CSV)
    candidates.append(CATEGORY_CSV_URL)
    # Remove duplicates while preserving order.
    seen = set()
    deduped: List[str | os.PathLike] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped
