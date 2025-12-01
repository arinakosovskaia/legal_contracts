"""Configuration dataclasses for the contract CUAD pipeline."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

_DATACLASS_KW = {"slots": True} if sys.version_info >= (3, 10) else {}

@dataclass(**_DATACLASS_KW)
class ModelConfig:
    """Configuration for loading and generating with an LLM."""

    model_name: str = "Qwen/Qwen2-7B-Instruct"
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    device_map: Optional[str] = "auto"
    extra_generation_kwargs: Dict[str, float] = field(default_factory=dict)


@dataclass(**_DATACLASS_KW)
class ChunkingConfig:
    """Controls clause chunking behavior."""

    max_clause_tokens: int = 450
    paragraph_overlap: int = 0


@dataclass(**_DATACLASS_KW)
class FewShotConfig:
    """Controls few-shot example selection."""

    max_examples_per_category: int = 1
    max_total_examples: int = 40


@dataclass(**_DATACLASS_KW)
class PipelineConfig:
    """Top-level configuration bundle for the CLI pipeline."""

    model: ModelConfig = field(default_factory=ModelConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    fewshot: FewShotConfig = field(default_factory=FewShotConfig)
