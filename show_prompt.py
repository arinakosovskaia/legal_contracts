#!/usr/bin/env python3
"""
Preview the LLM prompt without running the full application.

Usage:
    python3 show_prompt.py

This script shows what prompt will be sent to the LLM for a sample text chunk.
Useful for testing prompt changes before deploying.
"""

import json
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.llm import LLMRunner, LLMConfig, normalize_llm_text, load_categories_payload
from app.fewshot import load_or_build_fewshot, resolve_cuad_json_path

# Sample text chunk for preview
SAMPLE_TEXT_CHUNK = """
The Company may terminate this Agreement at any time, with or without cause, 
by providing thirty (30) days written notice to the Distributor. Upon termination, 
the Distributor shall immediately cease all distribution activities and return 
all Products to the Company at the Distributor's expense.
"""


def main():
    """Show the prompt that would be sent to the LLM."""
    print("=" * 80)
    print("LLM Prompt Preview")
    print("=" * 80)
    print()
    
    # Load categories
    data_dir = Path(__file__).parent / "data"
    configs_dir = Path(__file__).parent / "configs"
    csv_path = data_dir / "CUAD_v1" / "category_descriptions.csv"
    json_path = configs_dir / "categories.json"
    
    try:
        categories_payload = load_categories_payload(csv_path=csv_path if csv_path.exists() else None, json_path=json_path)
        print(f"✓ Loaded categories from: {json_path if json_path.exists() else csv_path}")
    except Exception as e:
        print(f"✗ Failed to load categories: {e}")
        print("  Make sure configs/categories.json exists or data/CUAD_v1/category_descriptions.csv is available")
        return 1
    
    # Try to load few-shot examples (optional - will work without them)
    fewshot = None
    fewshot_dir = data_dir / "fewshot_redflags"
    try:
        cuad_path = resolve_cuad_json_path(data_dir)
        fewshot = load_or_build_fewshot(
            data_dir=data_dir,
            categories_payload=categories_payload,
            enabled=True,
            max_examples_per_category=2,
            context_window=240,
            out_dir=fewshot_dir,
            only_category_ids={
                "termination_for_convenience",
                "uncapped_liability",
                "irrevocable_or_perpetual_license",
                "most_favored_nation",
                "audit_rights",
                "ip_ownership_assignment",
            },
        )
        if fewshot:
            total = sum(len(v) for v in fewshot.values())
            print(f"✓ Loaded {total} few-shot examples")
        else:
            print("⚠ Few-shot examples not available (will show prompt without examples)")
    except Exception as e:
        print(f"⚠ Few-shot examples not available: {e}")
        print("  Prompt will be shown without few-shot examples")
    
    # Create LLM runner (we don't need actual API key for preview)
    try:
        llm = LLMRunner(
            LLMConfig(
                api_key="dummy-key-for-preview",
                model_stage1="dummy",
                model_stage2="dummy",
            ),
            categories_payload,
            fewshot_by_category=fewshot or {},
        )
        print("✓ Initialized LLM runner")
    except Exception as e:
        print(f"✗ Failed to initialize LLM runner: {e}")
        return 1
    
    # Build prompt
    text_chunk_clean = normalize_llm_text(SAMPLE_TEXT_CHUNK)
    system, user, sections = llm._build_redflag_prompt(
        text_chunk=text_chunk_clean,
        section_path="Sample Section / Termination Clause",
    )
    
    print()
    print("=" * 80)
    print("SYSTEM PROMPT")
    print("=" * 80)
    print(system)
    print()
    print("=" * 80)
    print("USER PROMPT")
    print("=" * 80)
    print(user)
    print()
    print("=" * 80)
    print("PROMPT SECTIONS (for debugging)")
    print("=" * 80)
    print(json.dumps({
        "context_prefix": sections.get("context_prefix"),
        "text_chunk_length": len(sections.get("text_chunk", "")),
        "category_definitions_length": len(sections.get("category_definitions", "")),
        "fewshot_available": sections.get("fewshot_available"),
        "fewshot_length": len(sections.get("fewshot_examples", "")),
    }, indent=2))
    print()
    print("=" * 80)
    print("SAMPLE TEXT CHUNK (normalized)")
    print("=" * 80)
    print(text_chunk_clean)
    print()
    print("=" * 80)
    print("✓ Prompt preview complete!")
    print("=" * 80)
    print()
    print("To modify the prompt:")
    print("  1. Edit app/prompts.py (main instructions)")
    print("  2. Edit app/llm.py method _build_redflag_fewshot_examples() (few-shot templates)")
    print("  3. Run this script again to see the updated prompt")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
