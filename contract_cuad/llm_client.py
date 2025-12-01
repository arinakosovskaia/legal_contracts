"""Wrapper around Hugging Face causal LLMs for clause classification."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .config import ModelConfig

_DATACLASS_KW = {"slots": True} if sys.version_info >= (3, 10) else {}


@dataclass(**_DATACLASS_KW)
class ChatMessage:
    """Lightweight chat message container."""

    role: str
    content: str


class LLMClient:
    """Utility class that encapsulates tokenizer/model loading and generation."""

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        *,
        model_name: Optional[str] = None,
        tokenizer_kwargs: Optional[Dict] = None,
        model_kwargs: Optional[Dict] = None,
    ) -> None:
        torch_mod = self._require_torch()
        AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer = self._require_transformers()
        self.config = config or ModelConfig()
        self.model_name = model_name or self.config.model_name
        tokenizer_kwargs = tokenizer_kwargs or {}
        model_kwargs = model_kwargs or {}

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_fast=True, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        resolved_device_map = self.config.device_map
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=resolved_device_map,
            torch_dtype="auto",
            **model_kwargs,
        )
        self._torch = torch_mod
        self.device = torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
        if resolved_device_map not in {"auto", None}:
            self.model.to(resolved_device_map)  # type: ignore[arg-type]
        elif resolved_device_map is None:
            self.model.to(self.device)

        self.generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "repetition_penalty": self.config.repetition_penalty,
            **self.config.extra_generation_kwargs,
        }

    def generate_chat(self, messages: List[Dict[str, str]], **overrides) -> str:
        """Generate a chat completion style response."""

        prompt = self._build_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        target_device = self._resolve_target_device()
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
        generation_kwargs = {**self.generation_kwargs, **overrides}
        with self._torch.no_grad():
            output = self.model.generate(**inputs, **generation_kwargs)
        generated = output[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        parts: List[str] = []
        role_map = {"system": "[SYSTEM]", "user": "[USER]", "assistant": "[ASSISTANT]"}
        for message in messages:
            role = role_map.get(message.get("role", "user"), "[USER]")
            parts.append(f"{role} {message.get('content', '')}")
        parts.append("[ASSISTANT]")
        return "\n".join(parts)

    def _resolve_target_device(self):
        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            return model_device
        if hasattr(self.model, "hf_device_map"):
            device_val = next(iter(self.model.hf_device_map.values()))
            if isinstance(device_val, str):
                return self._torch.device(device_val)
            if isinstance(device_val, int):
                return self._torch.device(f"cuda:{device_val}")
            if isinstance(device_val, self._torch.device):
                return device_val
        return self.device

    @staticmethod
    def _require_torch():
        try:  # pragma: no cover - importability depends on environment
            import torch as torch_mod
        except (ImportError, OSError) as exc:
            raise ImportError(
                "PyTorch is required to instantiate LLMClient. "
                "Install torch or ensure it's available in this environment."
            ) from exc
        return torch_mod

    @staticmethod
    def _require_transformers() -> Tuple[Any, Any, Any]:
        try:  # pragma: no cover - importability depends on environment
            from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
        except ImportError as exc:
            raise ImportError(
                "Hugging Face transformers is required to instantiate LLMClient."
            ) from exc
        return AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
