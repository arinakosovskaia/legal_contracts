# Contract CUAD Classifier + Web Demo

This repo contains:

- `contract_cuad/`: a research-oriented CUAD classifier (Hugging Face / local open-source LLMs).
- `app/`: a minimal **web demo** (FastAPI + OpenAI-compatible API) for uploading PDFs and showing “red flags”.

The original tool ingests a contract text file, splits it into paragraph-level clauses, and classifies each clause into CUAD categories with either zero-shot or few-shot prompting of open-source Hugging Face LLMs (default: `Qwen/Qwen2-7B-Instruct`).

## Web demo (FastAPI + PDF upload + 2-stage LLM)

### Configure env

Create a `.env` file in the repo root (copy from `env.example`):

```bash
cp env.example .env
```

Fill at least:

- `OPENAI_API_KEY` (OpenAI) **or** `NEBIUS_API_KEY` (Nebius)
- `BASIC_AUTH_USER` / `BASIC_AUTH_PASS`

Nebius TokenFactory note: set `NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1/` (default in `env.example`).

#### Environment variables

The demo reads configuration from environment variables (recommended: put them into `.env` in the repo root; Docker Compose will pass them into the container).

| Variable | Purpose | Default |
|---|---|---|
| `NEBIUS_API_KEY` | Nebius TokenFactory API key (preferred if set) | (none) |
| `NEBIUS_BASE_URL` | Nebius base URL (TokenFactory) | `https://api.tokenfactory.nebius.com/v1/` |
| `OPENAI_API_KEY` | OpenAI API key (used if `NEBIUS_API_KEY` is not set) | (none) |
| `OPENAI_MODEL_STAGE1` | Stage 1 model (screening) | `Qwen/Qwen3-30B-A3B-Thinking-2507` |
| `OPENAI_MODEL_STAGE2` | Stage 2 model (detailed analysis) | `Qwen/Qwen3-30B-A3B-Thinking-2507` |
| `BASIC_AUTH_USER` | Basic Auth username for demo | `demo` |
| `BASIC_AUTH_PASS` | Basic Auth password for demo | `demo` |
| `ENABLE_AUTH` | Enable Basic Auth (`0`/`1`) | `1` |
| `MAX_PAGES` | Max PDF pages | `20` |
| `MAX_PARAGRAPHS` | Max parsed paragraphs | `200` |
| `MAX_FILE_MB` | Max upload size (MB) | `15` |
| `TTL_HOURS` | TTL cleanup window (hours) | `24` |
| `MAX_LLM_CONCURRENCY` | Max parallel LLM requests | `8` |
| `SINGLE_CLASSIFIER` | Use single red-flag classifier (`0`/`1`) | `1` |
| `STAGE1_INCLUDE_DESCRIPTIONS` | Include category descriptions in Stage 1 prompt (`0`/`1`) | `0` |
| `STAGE1_MAX_CATEGORIES` | Stage 1: max categories per paragraph | `10` |
| `T_ROUTER` | Router threshold: select all categories with prob >= t | `0.3` |
| `ROUTER_TOPK_PROBS` | Stage 1: how many category probs to output | `15` |
| `STAGE1_CONTEXT_PARAGRAPHS` | Stage 1 (sections): previous paragraphs as context | `5` |
| `STAGE1_MAX_INPUT_CHARS` | Stage 1: approximate max input size (chars) | `60000` |
| `STAGE1_OVERLAP_FRAC` | Stage 1: overlap fraction inside a section | `0.15` |
| `STAGE1_OVERLAP_MIN` | Stage 1: min overlap paragraphs | `1` |
| `STAGE1_OVERLAP_MAX` | Stage 1: max overlap paragraphs | `8` |
| `STAGE1_WINDOW_TOKENS` | Stage 1: window size (approx tokens) | `1100` |
| `STAGE1_STRIDE_TOKENS` | Stage 1: stride/overlap (approx tokens) | `300` |
| `STAGE1_TRIGGER_EDGE_FRAC` | Stage 1: exception trigger edge fraction | `0.15` |
| `STAGE2_CONTEXT_PARAGRAPHS` | Stage 2: neighbor paragraphs (same section) | `3` |
| `STAGE2_MAX_CONTEXT_CHARS` | Stage 2: max context size (chars) | `6000` |
| `ENABLE_FEWSHOT` | Enable CUAD few-shot reference snippets in Stage 2 (`0`/`1`) | `0` |
| `FEWSHOT_MAX_PER_CATEGORY` | Few-shot: max snippets per category | `2` |
| `FEWSHOT_CONTEXT_WINDOW` | Few-shot: chars around answer span | `240` |
| `FEWSHOT_DIR` | Few-shot cache directory | `./data/fewshot_redflags` (single classifier) |
| `ENABLE_DEBUG_TRACES` | Include per-paragraph LLM debug traces in result (`0`/`1`) | `0` |

#### Setting values in `.env`

- **Option A (edit file)**: open `.env` and set `NEBIUS_API_KEY=...`
- **Option B (terminal)**:

```bash
printf "\nNEBIUS_API_KEY=%s\n" "PASTE_KEY_HERE" >> .env
```

### Run locally (Docker)

```bash
docker compose up -d --build
```

Open `http://localhost:8000/` (Basic Auth prompt will appear).

### API quick test (curl)

```bash
curl -u "$BASIC_AUTH_USER:$BASIC_AUTH_PASS" -F "file=@/path/to/contract.pdf" http://localhost:8000/upload
```

Then poll:

```bash
curl -u "$BASIC_AUTH_USER:$BASIC_AUTH_PASS" http://localhost:8000/job/<job_id>
curl -u "$BASIC_AUTH_USER:$BASIC_AUTH_PASS" http://localhost:8000/job/<job_id>/result
```

### Categories source (demo)

The demo loads CUAD categories from `data/category_descriptions.csv` and auto-generates `configs/categories.json` at startup (so the LLM prompt uses stable `category_id`s).

## Project Structure

```
contract-cuad/
├── contract_cuad/
│   ├── __init__.py
│   ├── categories.py
│   ├── chunking.py
│   ├── classifier.py
│   ├── config.py
│   ├── cuad_fewshot.py
│   ├── cli.py
│   └── llm_client.py
├── notebooks/
│   └── demo_contract_classification.ipynb
├── tests/
│   ├── test_categories.py
│   ├── test_chunking.py
│   └── test_json_parsing.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Installation

### Local environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m nltk.downloader punkt
```

### Google Colab

```python
!git clone https://github.com/<your-account>/contract-cuad.git
%cd contract-cuad
!pip install -r requirements.txt
import nltk
nltk.download("punkt")
```

(Optional) Enable GPU under `Runtime → Change runtime type → GPU` for faster inference.

### Offline CUAD assets

The loader automatically looks for `contract_cuad/data/category_descriptions.csv`
first, so dropping the file there avoids any network calls. Alternatively, set
`CUAD_CATEGORY_CSV=/path/to/category_descriptions.csv` to point at a custom copy.
Few-shot generation **requires** a `cuad_v1.json` file. Place it in the top-level
`data/` directory (preferred) or inside `contract_cuad/data/`. You can override
both locations via the environment variable `CUAD_QA_JSON=/path/to/cuad_v1.json`.

## Usage

### 1. Prepare few-shot cache

```bash
contract-cuad prepare-fewshot \
  --output-json data/fewshot.json \
  --output-dir data/fewshot_by_category \
  --max-examples-per-category 2
```

> **Tip:** Run `prepare-fewshot` once and reuse the saved JSON(s). This avoids repeatedly
> scanning `cuad_v1.json` and makes notebook runs much faster and more reproducible.

### 2. Classify a contract file

```bash
contract-cuad classify-file \
  --contract-path sample_contract.txt \
  --output-csv results.csv \
  --fewshot-json data/fewshot.json \
  --model-name Qwen/Qwen2-7B-Instruct \
  --temperature 0.1 \
  --max-clause-tokens 450 \
  --paragraph-overlap 1 \
  --device-map cuda
```

The resulting CSV contains one row per clause, the primary (best) category, and a `top_categories_json`
column capturing the full ranked list (with confidences and reasons) for easier downstream analysis.

### 3. Explore the demo notebook

`notebooks/demo_contract_classification.ipynb` now includes:

- A Colab-ready setup flow that enables INFO-level logging so you can monitor chunking, few-shot loading, and per-clause classification progress.
- A realistic multi-paragraph sample contract (termination, change-of-control, anti-assignment, liability cap, damages waiver, non-solicit) that demonstrates paragraph-level chunking and per-clause top-3 predictions.
- A helper for classifying full `.txt` contracts in the repo-level `data/` folder, returning a pandas DataFrame with `primary_category`, confidence, error flags, and serialized `top_categories_json` for each clause.

## Testing

Run the lint-sized pytest suite locally:

```bash
pytest
```

## Notebook Demo

`notebooks/demo_contract_classification.ipynb` sketches an interactive Colab-friendly workflow: configuring the pipeline, preparing few-shot examples, and classifying a toy contract snippet.

## Notes & Next Steps

- The first time you run the CLI it will download the CUAD datasets and the target LLM from Hugging Face, so plan for sufficient disk space.
- Adjust the tokenizer/model configuration via CLI flags or by instantiating `ModelConfig`/`FewShotConfig` directly in Python.
- Extend the tests with golden inputs or integrate with Weights & Biases for experiment tracking as your research progresses.
