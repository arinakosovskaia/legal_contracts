# How to Edit Prompts

This guide explains how to customize how the AI analyzes contracts. You don't need to be a programmer - just basic text editing skills.

## Quick Overview

The AI uses **prompts** (instructions) to understand what to look for in contracts. You can edit these prompts to:
- Change what the AI looks for
- Adjust how strict or lenient it is
- Add new rules or criteria
- Modify the examples it learns from

## Where Everything Is

| What You Want to Edit | File to Open |
|----------------------|--------------|
| **Main instructions** (two-stage mode) | `app/prompts.py` |
| **Main instructions** (single-stage mode) | `app/prompts.py` |
| **Example clauses** (two-stage mode) | `data/fewshot_uk_redflags.txt` |
| **Category definitions** (two-stage mode) | `data/category_definitions_new.csv` |
| **Category definitions** (single-stage mode) | `data/category_descriptions.csv` |

## Two Ways to Analyze Contracts

The system has two modes:

1. **Two-Stage Mode** (default) - For UK residential tenancy agreements
   - First, quickly finds potential issues
   - Then, does detailed analysis only on those issues
   - More efficient and accurate

2. **Single-Stage Mode** (legacy) - For general contracts
   - One step that finds and analyzes everything at once
   - Good for testing and general use

You choose the mode in the web interface with the "Use legacy single-step mode" checkbox.

## Editing Prompts (Two-Stage Mode)

### Step 1: Find the Right Function

Open `app/prompts.py` in any text editor. Use "Find" (Ctrl+F or Cmd+F) to search for:

- `get_uk_router_system_prompt` - How the router should behave
- `get_uk_router_user_prompt_template` - What the router should look for
- `get_uk_redflag_system_prompt` - How the classifier should behave
- `get_uk_redflag_user_prompt_template` - What the classifier should analyze

### Step 2: Look for "EDIT HERE" Comments

Each function has comments that say "EDIT HERE" - these mark exactly what you can change.

### Step 3: Edit the Text

Just edit the text inside the `return """..."""` block. Save the file when done.

### Step 4: Restart Docker

After editing, restart the server to see your changes:

```bash
docker compose restart
```

### What Each Prompt Does

**Router Prompts** (first stage):
- `get_uk_router_system_prompt()` - Sets the AI's role (e.g., "You are a clause router...")
- `get_uk_router_user_prompt_template()` - Instructions for finding potential issues

**Classifier Prompts** (second stage):
- `get_uk_redflag_system_prompt()` - Sets the AI's role for detailed analysis
- `get_uk_redflag_user_prompt_template()` - Detailed instructions for analysis

The classifier prompt is the most important one - it controls what gets flagged and how.

### Example: Changing Rules

Let's say you want to make the AI more strict. Find `get_uk_redflag_user_prompt_template()` and look for the "DECISION RULES" section:

```python
DECISION RULES (VERY IMPORTANT)
1) Evidence MUST be copied verbatim from the provided TEXT CHUNK...
2) Only flag a category if the chunk contains clear evidence...
3) [ADD YOUR CUSTOM RULE HERE]  ← Add your rule here
```

Just add your new rule at line 3, save, and restart Docker.

## Editing Few-Shot Examples

Few-shot examples are example clauses that teach the AI what to look for. They're stored in a plain text file - no programming needed!

### How to Edit Examples

1. Open `data/fewshot_uk_redflags.txt` in any text editor
2. Find the category you want to edit (search for the category name)
3. Edit the examples, helpful cues, or notes
4. Save the file
5. Restart Docker: `docker compose restart`

### File Structure

The file is organized like this:

```
GLOBAL RULES TO REDUCE FALSE POSITIVES (DO NOT FLAG)
[Rules that apply to all categories]

------------------------------------------------------------
CATEGORY: Deposits & Deposit Protection
------------------------------------------------------------
HELPFUL CUES (non-exhaustive; not mandatory)
- "returned at landlord's convenience"
- "deposit may be transferred to landlord's personal account"

[TRUE POSITIVE]
TEXT CHUNK:
"Deposit will be returned at landlord's convenience. No fixed timeframe..."

EXPECTED BEHAVIOR:
- Flag "Deposits & Deposit Protection"
- Quote: "returned at landlord's convenience" (verbatim)
- Core point: open-ended landlord discretion...

[HARD NEGATIVE #1]
TEXT CHUNK:
"The deposit will be returned within 10 working days..."

EXPECTED OUTPUT:
{"findings": []}

BORDERLINE NOTES
- If a clause mentions deductions, only flag if it allows arbitrary deductions...
```

### What You Can Change

- **Add new examples**: Copy an existing category section and modify it
- **Modify examples**: Edit the TEXT CHUNK or EXPECTED BEHAVIOR sections
- **Add categories**: Add a new section with the same format
- **Update helpful cues**: Add or remove keywords
- **Update borderline notes**: Modify guidance for edge cases

**Important**: Keep the section headers (`------------------------------------------------------------`) and format consistent.

## Editing Category Definitions

Categories define what types of issues to look for. They're stored in a CSV file.

### Two-Stage Mode Categories

File: `data/category_definitions_new.csv`

Format: `id,name,description`

Example:
```csv
id,name,description
deposits_deposit_protection,Deposits & Deposit Protection,Clauses related to deposit handling and protection requirements
```

### Single-Stage Mode Categories

File: `data/category_descriptions.csv`

Same format - just edit the CSV file, save, and restart Docker.

## Editing Prompts (Single-Stage Mode)

For single-stage mode, edit these functions in `app/prompts.py`:

- `get_redflag_system_prompt()` - Sets the AI's role
- `get_redflag_user_prompt_template()` - Main instructions

Same process: find the function, look for "EDIT HERE", edit the text, save, restart Docker.

## Previewing Your Changes

Before restarting Docker, you can preview what the prompts will look like:

```bash
python3 show_prompt.py
```

This shows you exactly what will be sent to the AI, so you can verify your changes look correct.

## Testing Your Changes

1. **Edit the prompt or examples** (as described above)
2. **Restart Docker**: `docker compose restart`
3. **Test in the web interface**:
   - Upload a test contract
   - Enable "Full LLM debug" checkbox
   - Process the contract
   - Scroll to "Debug traces" to see the actual prompts
4. **Review results** and adjust as needed

## Common Customizations

### Making the AI More Strict

In `get_uk_redflag_user_prompt_template()`, find "DECISION RULES" and add rules like:
- "Only flag if there is clear, unambiguous evidence"
- "Do not flag borderline cases"

### Making the AI More Lenient

Add rules like:
- "Flag if there is any reasonable suspicion"
- "Include cases where the clause might be problematic"

### Changing Risk Scoring

Find "RISK SCORING RUBRIC" and modify the definitions:
```python
severity_of_consequences:
0 = no meaningful tenant risk
1 = potentially unfair; mainly civil dispute
2 = likely unenforceable/void OR could trigger civil penalties
3 = potential criminal liability / severe illegality
```

### Adding New Categories

1. Add a row to `data/category_definitions_new.csv`:
   ```csv
   new_category_id,New Category Name,Description of what to look for
   ```

2. Add examples to `data/fewshot_uk_redflags.txt` (see format above)

3. Restart Docker

## Tips

- **Start small**: Make one change at a time so you can see what each change does
- **Use preview**: Run `python3 show_prompt.py` before restarting Docker
- **Enable debug**: Always enable "Full LLM debug" when testing to see what's happening
- **Keep backups**: Save a copy of files before making big changes
- **Test with real contracts**: Try your changes on actual contracts, not just examples

## Need Help?

- Check the function comments in `app/prompts.py` - they explain what each part does
- Look at existing examples in `data/fewshot_uk_redflags.txt` for formatting
- Enable debug mode in the UI to see exactly what prompts are being used
- The code has helpful comments - read them if you're curious about how things work

## File Locations Summary

```
app/
└── prompts.py                    ← Main prompt templates (edit here)

data/
├── category_definitions_new.csv  ← Two-stage categories (edit here)
├── category_descriptions.csv     ← Single-stage categories (edit here)
└── fewshot_uk_redflags.txt       ← Two-stage examples (edit here)
```

That's it! You don't need to be a programmer - just edit text files and restart Docker. Happy customizing! 🎉
