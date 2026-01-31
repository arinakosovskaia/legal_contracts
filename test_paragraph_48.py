#!/usr/bin/env python3
"""
Test script to check how paragraph 48 is processed in the real pipeline.
Simulates the window/chunk creation logic.
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.llm import LLMRunner, LLMConfig, normalize_llm_text
from app.parser import parse_pdf_to_paragraphs
from app.categories import load_categories_from_csv
from app.fewshot import load_or_build_fewshot, resolve_cuad_json_path
from app.headings import parse_heading, is_heading

async def main():
    pdf_path = Path(__file__).parent / "static" / "example_contract.pdf"
    paragraphs, _ = parse_pdf_to_paragraphs(pdf_path, max_pages=20, max_paragraphs=200)
    
    # Find paragraph 48
    para_48 = next((p for p in paragraphs if p.paragraph_index == 48), None)
    if not para_48:
        print("Paragraph 48 not found")
        return
    
    print(f"Paragraph 48: page {para_48.page}")
    print(f"Text length: {len(para_48.text)}")
    print(f"Text: {para_48.text[:500]}...")
    print()
    
    # Simulate how sections are built in main.py
    sections = []
    current_section = []
    
    for p in paragraphs:
        if is_heading(p.text):
            if current_section:
                sections.append(current_section)
            current_section = [p]
        else:
            current_section.append(p)
    if current_section:
        sections.append(current_section)
    
    # Find which section contains paragraph 48
    section_with_48 = None
    for i, sec in enumerate(sections):
        if any(p.paragraph_index == 48 for p in sec):
            section_with_48 = sec
            print(f"Paragraph 48 is in section {i} with {len(sec)} paragraphs")
            print(f"Section paragraphs: {[p.paragraph_index for p in sec[:10]]}...")
            break
    
    if not section_with_48:
        print("Paragraph 48 not found in any section")
        return
    
    # Find position of para 48 in section
    para_48_idx = next((i for i, p in enumerate(section_with_48) if p.paragraph_index == 48), None)
    if para_48_idx is None:
        print("Could not find para 48 index in section")
        return
    
    print(f"Paragraph 48 is at index {para_48_idx} in its section")
    print()
    
    # Simulate window creation - check what windows would include para 48
    # In single-classifier mode, windows are created with stride
    stride_tokens = 300
    window_size = 5  # Approximate
    
    # Check windows that would include para 48
    print("Windows that would include paragraph 48:")
    for start in range(max(0, para_48_idx - 3), min(len(section_with_48), para_48_idx + 3)):
        end = min(start + window_size, len(section_with_48))
        window_paras = section_with_48[start:end]
        window_ids = [p.paragraph_index for p in window_paras]
        
        if 48 in window_ids:
            print(f"  Window [{start}:{end}]: paragraphs {window_ids}")
            text_chunk = normalize_llm_text("\n\n".join(p.text for p in window_paras))
            print(f"    Chunk length: {len(text_chunk)}")
            print(f"    Contains target quote? {'If Company terminates' in text_chunk}")
            print()
    
    # Test LLM on the actual window that would be used
    # In real pipeline, window is created around para 48
    start_i = max(0, para_48_idx - 2)
    end_i = min(len(section_with_48), para_48_idx + 3)
    window_paras = section_with_48[start_i:end_i]
    
    print(f"Testing LLM on window [{start_i}:{end_i}] (paragraphs {[p.paragraph_index for p in window_paras]}):")
    text_chunk = normalize_llm_text("\n\n".join(p.text for p in window_paras))
    print(f"Chunk length: {len(text_chunk)}")
    print(f"First 500 chars: {text_chunk[:500]}...")
    print()
    
    # Load LLM
    data_dir = Path(__file__).parent / "data"
    cats_payload = load_categories_from_csv(data_dir / "category_descriptions.csv")
    cuad_path = resolve_cuad_json_path(data_dir)
    fewshot = load_or_build_fewshot(
        data_dir=data_dir,
        categories_payload=cats_payload,
        enabled=True,
        max_examples_per_category=2,
        context_window=240,
        out_dir=data_dir / "fewshot_cache",
    )
    
    api_key = os.environ.get("NEBIUS_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    base_url = os.environ.get("NEBIUS_BASE_URL", "https://api.nebius.com/v1/tokenfactory")
    model = os.environ.get("OPENAI_MODEL_STAGE2", "Qwen/Qwen2.5-32B-Instruct-AWQ")
    
    llm = LLMRunner(
        LLMConfig(
            api_key=api_key,
            base_url=base_url,
            model_stage1="gpt-4o-mini",
            model_stage2=model,
            max_concurrency=1,
            temperature=0.0,
        ),
        cats_payload,
        fewshot_by_category=fewshot,
    )
    
    # Get section path (simulate how it's built in main.py)
    # Build heading stack
    stack = []
    for p in paragraphs[:para_48.paragraph_index + 1]:
        h = parse_heading(p.text, max_len=160)
        if h is not None:
            while stack and stack[-1].level >= h.level:
                stack.pop()
            stack.append(h)
    
    # Build path from stack
    def _heading_stack_to_path(stack):
        if not stack:
            return ""
        parts = []
        for h in stack:
            label = h.label or ""
            title = h.title or ""
            if label and title:
                parts.append(f"{label}: {title}")
            elif label:
                parts.append(label)
            elif title:
                parts.append(title)
        return " > ".join(parts)
    
    section_path = _heading_stack_to_path(stack)
    
    print(f"Section path: {section_path}")
    print("Calling LLM...")
    
    out, raw, _ = await llm.classify_redflags_for_chunk(
        text_chunk=text_chunk,
        section_path=section_path,
        return_prompt=False,
    )
    
    print(f"\nFindings: {len(out.findings)}")
    print(f"Raw output: {raw}")
    if out.findings:
        for f in out.findings:
            print(f"  - {f.category}: {f.quote[:150]}...")
    else:
        print("  NO FINDINGS - this is the problem!")

if __name__ == "__main__":
    asyncio.run(main())
