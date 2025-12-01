# Contract CUAD Classifier

The tool ingests a contract text file, splits it into paragraph-level clauses, and classifies each clause into CUAD categories with either zero-shot or few-shot prompting of open-source Hugging Face LLMs (default: `Qwen/Qwen2-7B-Instruct`).

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
  --max-examples-per-category 2 \
  --max-total-examples 40
```

> **Data source note:** Few-shot preparation looks for `cuad_v1.json` in the repo-level
> `data/` folder first, then inside `contract_cuad/data/`, and finally in the path
> referenced by `CUAD_QA_JSON`. Make sure the file is available in one of those places
> before running this command.

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
