#!/usr/bin/env python3
"""
Debug script to test LLM with specific quotes from LIMEENERGYCO contract.
Tests why certain quotes are not being found as red flags.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.llm import LLMRunner, LLMConfig, normalize_llm_text
from app.parser import parse_pdf_to_paragraphs
from app.categories import load_categories_from_csv
from app.fewshot import load_or_build_fewshot, resolve_cuad_json_path

# Target quotes to test
TARGET_QUOTES = [
    "If Company terminates the Agreement without cause and for reasons other than Distributor's failure to meet its minimum expectations; it shall repurchase from Distributor any unopened Product, and shall bear all shipping, handling and related costs notwithstanding any other remedies to which Distributor may be entitled.",
    "any price changes, other than those based on the CPI, shall be uniformly applied to all distributors of the Products and shall reasonably applied to all distributors of the Products and shall reasonably reflect Company's costs of manufacturing the Products and/or market demand for the Products, provided further than any increase in price based upon market demand shall not be so great as to deprive Distributor of its normal and customary profit margin.",
    "Company and Distributor agree to indemnify, defend and hold each other harmless from any and all suits, claims, obligations, liabilities, damages, losses and the like (including attorneys' fees and costs) relating to or arising out of:",
    "any and all suits, claims, obligations, liabilities, damages, losses and the like (including attorneys' fees and costs) relating to or arising out of:",
]

async def main():
    # Load PDF
    pdf_path = Path(__file__).parent / "static" / "example_contract.pdf"
    if not pdf_path.exists():
        print(f"ERROR: PDF not found at {pdf_path}")
        return
    
    print(f"Loading PDF: {pdf_path}")
    paragraphs, page_count = parse_pdf_to_paragraphs(
        pdf_path,
        max_pages=20,
        max_paragraphs=200,
    )
    print(f"Parsed {len(paragraphs)} paragraphs from {page_count} pages\n")
    
    # Find paragraphs containing target quotes
    print("=" * 80)
    print("SEARCHING FOR TARGET QUOTES IN PARSED PARAGRAPHS")
    print("=" * 80)
    
    for i, target_quote in enumerate(TARGET_QUOTES, 1):
        print(f"\n--- Quote {i} ---")
        print(f"Target: {target_quote[:100]}...")
        
        target_norm = normalize_llm_text(target_quote)
        print(f"Normalized: {target_norm[:100]}...")
        
        found_paras = []
        for p_idx, para in enumerate(paragraphs):
            para_norm = normalize_llm_text(para.text)
            if target_norm.lower() in para_norm.lower():
                found_paras.append((p_idx, para))
                print(f"\n  Found in paragraph {p_idx} (page {para.page}):")
                print(f"  Original text: {para.text[:200]}...")
                print(f"  Normalized: {para_norm[:200]}...")
        
        if not found_paras:
            print("  ❌ NOT FOUND in any paragraph!")
            # Try partial match
            target_words = target_norm.lower().split()[:10]  # First 10 words
            for p_idx, para in enumerate(paragraphs):
                para_norm = normalize_llm_text(para.text)
                para_words = para_norm.lower().split()
                if all(word in para_words for word in target_words if len(word) > 3):
                    print(f"  ⚠️  Partial match in paragraph {p_idx} (page {para.page}):")
                    print(f"     {para.text[:200]}...")
        else:
            print(f"  ✓ Found in {len(found_paras)} paragraph(s)")
    
    # Now test LLM on paragraphs containing quotes
    print("\n" + "=" * 80)
    print("TESTING LLM ON PARAGRAPHS WITH TARGET QUOTES")
    print("=" * 80)
    
    # Load categories
    data_dir = Path(__file__).parent / "data"
    cats_payload = load_categories_from_csv(data_dir / "category_descriptions.csv")
    
    # Load fewshot
    cuad_path = resolve_cuad_json_path(data_dir)
    fewshot = load_or_build_fewshot(
        data_dir=data_dir,
        categories_payload=cats_payload,
        enabled=True,
        max_examples_per_category=2,
        context_window=240,
        out_dir=data_dir / "fewshot_cache",
    )
    
    # Setup LLM
    api_key = os.environ.get("NEBIUS_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    if not api_key:
        print("ERROR: No API key found in NEBIUS_API_KEY or OPENAI_API_KEY")
        return
    
    base_url = os.environ.get("NEBIUS_BASE_URL", "https://api.nebius.com/v1/tokenfactory")
    model = os.environ.get("OPENAI_MODEL_STAGE2", "Qwen/Qwen2.5-32B-Instruct-AWQ")
    # Try to detect if it's a thinking model
    if "thinking" in model.lower() or "30b" in model.lower():
        print("  Detected thinking model - may need special handling")
    
    print(f"\nLLM Config:")
    print(f"  Model: {model}")
    print(f"  Base URL: {base_url}")
    print(f"  Temperature: 0.0")
    
    llm = LLMRunner(
        LLMConfig(
            api_key=api_key,
            base_url=base_url,
            model_stage1="gpt-4o-mini",  # Not used in single-classifier mode
            model_stage2=model,
            max_concurrency=1,
            temperature=0.0,
        ),
        cats_payload,
        fewshot_by_category=fewshot,
    )
    
    # Test each target quote
    for i, target_quote in enumerate(TARGET_QUOTES, 1):
        print(f"\n{'=' * 80}")
        print(f"TESTING QUOTE {i}")
        print(f"{'=' * 80}")
        print(f"Target quote: {target_quote[:150]}...")
        
        # Find paragraph containing this quote
        target_norm = normalize_llm_text(target_quote)
        test_para = None
        for para in paragraphs:
            para_norm = normalize_llm_text(para.text)
            if target_norm.lower() in para_norm.lower():
                test_para = para
                break
        
        if not test_para:
            print("❌ Quote not found in any paragraph")
            # For Quote 1, try paragraph 48 anyway (known location)
            if i == 1:
                para_48 = next((p for p in paragraphs if p.paragraph_index == 48), None)
                if para_48:
                    print("⚠️  Using paragraph 48 as fallback (known to contain this quote)")
                    test_para = para_48
                else:
                    print("Skipping LLM test")
                    continue
            else:
                print("Skipping LLM test")
                continue
        
        print(f"\nTesting on paragraph {test_para.paragraph_index} (page {test_para.page}):")
        print(f"Original text length: {len(test_para.text)}")
        print(f"Full original text:")
        print("-" * 80)
        print(test_para.text)
        print("-" * 80)
        
        # Normalize for LLM
        text_chunk = normalize_llm_text(test_para.text)
        print(f"Normalized text length: {len(text_chunk)}")
        print(f"Normalized preview: {text_chunk[:300]}...")
        
        # Call LLM
        print(f"\nCalling LLM...")
        try:
            out, raw, _ = await llm.classify_redflags_for_chunk(
                text_chunk=text_chunk,
                section_path="",
                max_findings=5,
                return_prompt=False,
            )
            
            print(f"\nLLM Response:")
            print(f"  Findings count: {len(out.findings)}")
            print(f"  Raw output length: {len(raw)}")
            print(f"\nRaw output:")
            print("-" * 80)
            print(raw)
            print("-" * 80)
            
            if out.findings:
                print(f"\nParsed findings:")
                for j, f in enumerate(out.findings, 1):
                    print(f"  Finding {j}:")
                    print(f"    Category: {f.category}")
                    print(f"    Quote: {f.quote[:150] if f.quote else 'EMPTY'}...")
                    print(f"    Is unfair: {f.is_unfair}")
                    print(f"    Explanation: {f.explanation[:200] if f.explanation else 'EMPTY'}...")
            else:
                print("\n❌ NO FINDINGS RETURNED!")
                print("This is the problem - LLM returned empty findings array")
                
        except Exception as e:
            print(f"\n❌ ERROR calling LLM: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
