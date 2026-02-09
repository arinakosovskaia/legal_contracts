#!/usr/bin/env python3
"""
Preview the LLM prompt without running the full application.

Usage:
    python3 show_prompt.py [--mode two-stage|single-stage]

This script shows what prompt will be sent to the LLM for a sample text chunk.
Useful for testing prompt changes before deploying.

By default, shows two-stage prompts (router + classifier).
Use --mode single-stage to see legacy single-stage prompt.
"""

import json
import sys
import argparse
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.llm import LLMRunner, LLMConfig, normalize_llm_text, load_categories_payload
from app.fewshot import load_or_build_fewshot, resolve_cuad_json_path

# Sample text chunk for preview (UK tenancy example for two-stage)
SAMPLE_TEXT_CHUNK_UK = """
The Tenant agrees to pay a non-refundable deposit of £500 upon signing this agreement. 
The deposit will be returned at the landlord's convenience. The landlord may enter the 
property at any time without notice for inspection purposes.
"""

# Sample text chunk for legacy mode
SAMPLE_TEXT_CHUNK_LEGACY = """
The Company may terminate this Agreement at any time, with or without cause, 
by providing thirty (30) days written notice to the Distributor. Upon termination, 
the Distributor shall immediately cease all distribution activities and return 
all Products to the Company at the Distributor's expense.
"""


def show_two_stage_prompts(data_dir: Path, configs_dir: Path):
    """Show two-stage prompts (router + classifier)."""
    print("=" * 80)
    print("TWO-STAGE CLASSIFICATION PROMPTS (UK Red Flags)")
    print("=" * 80)
    print()
    
    # Load UK categories
    csv_path = data_dir / "category_definitions_new.csv"
    json_path = configs_dir / "categories.json"
    
    try:
        categories_payload = load_categories_payload(
            csv_path=csv_path if csv_path.exists() else None,
            json_path=json_path,
            version="uk_tenancy_v1"
        )
        print(f"✓ Loaded UK categories from: {csv_path if csv_path.exists() else json_path}")
    except Exception as e:
        print(f"✗ Failed to load categories: {e}")
        print(f"  Make sure {csv_path} exists or {json_path} is available")
        return 1
    
    # Load UK few-shot examples
    uk_fewshot_path = data_dir / "fewshot_uk_redflags.txt"
    uk_fewshot_text = ""
    if uk_fewshot_path.exists():
        uk_fewshot_text = uk_fewshot_path.read_text(encoding="utf-8").strip()
        print(f"✓ Loaded UK few-shot examples from: {uk_fewshot_path}")
        print(f"  Few-shot text length: {len(uk_fewshot_text)} characters")
    else:
        print(f"⚠ UK few-shot examples not found at: {uk_fewshot_path}")
        print("  Prompt will be shown without few-shot examples")
    
    # Create LLM runner
    try:
        llm = LLMRunner(
            LLMConfig(
                api_key="dummy-key-for-preview",
                model_stage1="dummy",
                model_stage2="dummy",
            ),
            categories_payload,
            uk_fewshot_text=uk_fewshot_text or None,
        )
        print("✓ Initialized LLM runner")
    except Exception as e:
        print(f"✗ Failed to initialize LLM runner: {e}")
        return 1
    
    # Build prompts
    text_chunk_clean = normalize_llm_text(SAMPLE_TEXT_CHUNK_UK)
    
    # Router prompt
    print()
    print("=" * 80)
    print("STAGE 1: ROUTER PROMPT")
    print("=" * 80)
    router_system, router_user, router_sections = llm._build_uk_router_prompt(
        text_chunk=text_chunk_clean,
        section_path="Sample Section / Deposit Clause",
    )
    print("\n[ROUTER SYSTEM PROMPT]")
    print("-" * 80)
    print(router_system)
    print("\n[ROUTER USER PROMPT]")
    print("-" * 80)
    print(router_user)
    print("\n[ROUTER SECTIONS]")
    print("-" * 80)
    print(json.dumps({
        "context_prefix": router_sections.get("context_prefix"),
        "text_chunk_length": len(router_sections.get("text_chunk", "")),
        "category_definitions_length": len(router_sections.get("category_definitions", "")),
    }, indent=2))
    
    # Classifier prompt (simulate router selecting some categories)
    print()
    print("=" * 80)
    print("STAGE 2: CLASSIFIER PROMPT")
    print("=" * 80)
    print("(Assuming router selected some categories)")
    print()
    
    # Get first few category IDs for preview
    all_categories = categories_payload.get("categories", [])
    sample_category_ids = [c.get("id", "") for c in all_categories[:3] if c.get("id")]
    
    classifier_system, classifier_user, classifier_sections = llm._build_uk_redflag_prompt(
        text_chunk=text_chunk_clean,
        section_path="Sample Section / Deposit Clause",
        allowed_category_ids=sample_category_ids,
    )
    print("\n[CLASSIFIER SYSTEM PROMPT]")
    print("-" * 80)
    print(classifier_system)
    print("\n[CLASSIFIER USER PROMPT]")
    print("-" * 80)
    print(classifier_user)
    print("\n[CLASSIFIER SECTIONS]")
    print("-" * 80)
    print(json.dumps({
        "context_prefix": classifier_sections.get("context_prefix"),
        "text_chunk_length": len(classifier_sections.get("text_chunk", "")),
        "category_definitions_length": len(classifier_sections.get("category_definitions", "")),
        "fewshot_available": classifier_sections.get("fewshot_available"),
        "fewshot_length": len(classifier_sections.get("fewshot_examples", "")),
    }, indent=2))
    
    print()
    print("=" * 80)
    print("SAMPLE TEXT CHUNK (normalized)")
    print("=" * 80)
    print(text_chunk_clean)
    print()
    print("=" * 80)
    print("✓ Two-stage prompt preview complete!")
    print("=" * 80)
    print()
    print("To modify the prompts:")
    print("  1. Router: Edit app/prompts.py functions get_uk_router_system_prompt() and get_uk_router_user_prompt_template()")
    print("  2. Classifier: Edit app/prompts.py functions get_uk_redflag_system_prompt() and get_uk_redflag_user_prompt_template()")
    print("  3. Few-shot examples: Edit data/fewshot_uk_redflags.txt")
    print("  4. Run this script again to see the updated prompts")
    
    return 0


def show_single_stage_prompts(data_dir: Path, configs_dir: Path):
    """Show single-stage (legacy) prompts."""
    print("=" * 80)
    print("SINGLE-STAGE CLASSIFICATION PROMPTS (Legacy Mode)")
    print("=" * 80)
    print()
    
    # Load categories
    csv_path = data_dir / "CUAD_v1" / "category_descriptions.csv"
    json_path = configs_dir / "categories.json"
    
    try:
        categories_payload = load_categories_payload(
            csv_path=csv_path if csv_path.exists() else None,
            json_path=json_path
        )
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
    
    # Create LLM runner
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
    text_chunk_clean = normalize_llm_text(SAMPLE_TEXT_CHUNK_LEGACY)
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
    print("✓ Single-stage prompt preview complete!")
    print("=" * 80)
    print()
    print("To modify the prompt:")
    print("  1. Edit app/prompts.py (main instructions)")
    print("  2. Edit app/prompts.py function build_redflag_fewshot_example() (few-shot templates)")
    print("  3. Run this script again to see the updated prompt")
    
    return 0


def main():
    """Show the prompt that would be sent to the LLM."""
    parser = argparse.ArgumentParser(description="Preview LLM prompts")
    parser.add_argument(
        "--mode",
        choices=["two-stage", "single-stage"],
        default="two-stage",
        help="Which mode to preview (default: two-stage)"
    )
    args = parser.parse_args()
    
    data_dir = Path(__file__).parent / "data"
    configs_dir = Path(__file__).parent / "configs"
    
    if args.mode == "two-stage":
        return show_two_stage_prompts(data_dir, configs_dir)
    else:
        return show_single_stage_prompts(data_dir, configs_dir)


if __name__ == "__main__":
    sys.exit(main())
