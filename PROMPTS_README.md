# Editing LLM Prompts

This guide is for legal experts who want to modify how the AI analyzes contracts.

## Quick Start

**All prompts are in `app/prompts.py`** - edit this file to change how the model works.

## What You Can Edit

### 1. Main Instructions (`get_redflag_user_prompt_template()`)

This is the core prompt that tells the LLM:
- What categories to look for
- What to return (quote, reasoning, legal analysis)
- The rules for classification

**Location:** `app/prompts.py`, function `get_redflag_user_prompt_template()`

**What to change:**
- Instructions in the "Rules" section
- Output JSON schema (what fields the model should return)
- Legal references and explanations

### 2. Few-Shot Examples (`_build_redflag_fewshot_examples()`)

Few-shot examples show the model how to classify clauses. They come from CUAD dataset.

**Location:** `app/llm.py`, method `_build_redflag_fewshot_examples()` (lines ~156-275)

**What to change:**
- Reasoning templates for each category
- Explanation text for each category
- Legal references
- Risk categories
- Revision suggestions

**Note:** Few-shot examples are built from CUAD data, but the template values (reasoning, explanation, etc.) can be customized per category.

### 3. Category Definitions

Category definitions come from `category_descriptions.csv` and are automatically included in the prompt.

**Location:** `data/CUAD_v1/category_descriptions.csv` or `configs/categories.json`

## How to Preview the Prompt (Without Docker)

**Quick preview script:**

```bash
python3 show_prompt.py
```

This will:
- Load categories and few-shot examples
- Build the prompt for a sample text chunk
- Display the system and user prompts that would be sent to the LLM

**Workflow for legal experts:**
1. Edit `app/prompts.py` (or `app/llm.py` for few-shot templates)
2. Run `python3 show_prompt.py` to preview the prompt
3. Verify the prompt looks correct
4. Developer will integrate the changes

## How to Test Changes (Full Testing)

1. Edit `app/prompts.py` (or `app/llm.py` for few-shot templates)
2. Restart the Docker container: `docker compose restart`
3. Upload a test contract
4. Check the "Full LLM debug" checkbox to see the actual prompt sent to the model
5. Review the results and adjust as needed

## File Structure

```
app/
├── prompts.py          ← EDIT HERE: Main prompt templates
├── llm.py             ← EDIT HERE: Few-shot example templates (lines ~156-275)
└── fewshot.py          ← Few-shot data loading (usually don't need to edit)
```

## Example: Changing Instructions

To modify the rules, edit `app/prompts.py`:

```python
def get_redflag_user_prompt_template() -> str:
    return """...
Rules
- Evidence MUST be copied verbatim from the provided TEXT CHUNK. Do not paraphrase.
- [ADD YOUR NEW RULE HERE]
- Explain unfairness under UK law...
"""
```

## Example: Changing Few-Shot Examples

To modify how few-shot examples are formatted, edit `app/llm.py` around line 184:

```python
if name == "Termination For Convenience":
    reasoning = [
        "YOUR CUSTOM REASONING HERE",
        "Second bullet point",
    ]
    explanation = "YOUR CUSTOM EXPLANATION HERE"
    # ... etc
```

## Need Help?

- Check `app/prompts.py` for the main prompt structure
- Check `app/llm.py` method `_build_redflag_fewshot_examples()` for few-shot templates
- Enable "Full LLM debug" in the UI to see the exact prompt sent to the model
