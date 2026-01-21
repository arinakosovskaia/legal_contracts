"""Typer CLI entrypoint for CUAD-based clause classification."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .categories import load_cuad_categories
from .classifier import classify_contract_text
from .config import FewShotConfig, ModelConfig
from .cuad_fewshot import build_fewshot_examples, load_cuad_qa, save_fewshot_by_category
from .llm_client import LLMClient

logging.basicConfig(level=logging.INFO)

app = typer.Typer(help="CLI for clause classification against CUAD categories")


@app.command("prepare-fewshot")
def prepare_fewshot(
    output_json: Optional[Path] = typer.Option(None, help="Optional path to save a single JSON cache."),
    output_dir: Optional[Path] = typer.Option(None, help="Optional directory to save per-category JSON files."),
    max_examples_per_category: int = typer.Option(1, help="Max snippets per CUAD category."),
    context_window: int = typer.Option(240, help="Characters to keep before/after each answer."),
) -> None:
    """Construct few-shot examples straight from CUAD QA and save as JSON."""

    if output_json is None and output_dir is None:
        raise typer.BadParameter("Provide at least one of --output-json or --output-dir.")

    categories = load_cuad_categories()
    dataset = load_cuad_qa()
    examples = build_fewshot_examples(
        categories,
        dataset=dataset,
        max_examples_per_category=max_examples_per_category,
        context_window=context_window,
    )
    total_examples = sum(len(v) for v in examples.values())

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(examples, indent=2, ensure_ascii=False))
        typer.echo(f"Saved {total_examples} examples to {output_json}")

    if output_dir is not None:
        manifest = save_fewshot_by_category(examples, output_dir)
        typer.echo(f"Saved per-category few-shot files (manifest: {manifest})")


@app.command("classify-file")
def classify_file(
    contract_path: Path = typer.Option(..., help="Path to the contract .txt file."),
    output_csv: Path = typer.Option(..., help="Destination CSV path."),
    model_name: str = typer.Option("Qwen/Qwen2-7B-Instruct", help="HF model to use."),
    fewshot_json: Optional[Path] = typer.Option(None, help="Optional few-shot JSON cache."),
    max_new_tokens: int = typer.Option(256, help="Generation max_new_tokens."),
    temperature: float = typer.Option(0.2, help="Sampling temperature."),
    top_p: float = typer.Option(0.95, help="Nucleus sampling top_p."),
    repetition_penalty: float = typer.Option(1.0, help="Generation repetition penalty."),
    max_clause_tokens: int = typer.Option(450, help="Token budget per clause."),
    paragraph_overlap: int = typer.Option(0, help="Paragraph overlap between clauses."),
    max_clauses: Optional[int] = typer.Option(None, help="Optional limit on clauses to classify."),
    disable_progress: bool = typer.Option(False, help="Disable tqdm progress bars."),
    fewshot_prompt_max_per_category: int = typer.Option(1, help="Max examples per category in the prompt."),
    device_map: str = typer.Option(
        "auto",
        help="Device placement for the model (e.g. 'cuda', 'cpu', 'auto').",
    ),
) -> None:
    """Classify every clause in the supplied contract file."""

    categories = load_cuad_categories()
    fewshot_examples = None
    if fewshot_json and fewshot_json.exists():
        fewshot_examples = json.loads(fewshot_json.read_text())
    model_config = ModelConfig(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        device_map=device_map,
    )
    llm = LLMClient(config=model_config)
    contract_text = contract_path.read_text()
    fewshot_config = FewShotConfig(
        max_examples_per_category=fewshot_prompt_max_per_category,
    )
    results = classify_contract_text(
        contract_text,
        categories,
        llm,
        fewshot_examples=fewshot_examples,
        fewshot_config=fewshot_config,
        max_clause_tokens=max_clause_tokens,
        max_clauses=max_clauses,
        paragraph_overlap=paragraph_overlap,
        show_progress=not disable_progress,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df["top_categories_json"] = df["categories"].apply(lambda cats: json.dumps(cats, ensure_ascii=False))
    df.to_csv(output_csv, index=False)
    typer.echo(f"Wrote {len(results)} clause predictions to {output_csv}")
