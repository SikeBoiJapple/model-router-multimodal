from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


class ModelMappingError(ValueError):
    pass


@dataclass(frozen=True)
class ModelTarget:
    provider: str
    model: str


MODEL_MAP: dict[str, ModelTarget] = {
    "reasoning-high": ModelTarget(provider="openai", model="gpt-5"),
    "cost-fast": ModelTarget(provider="openai", model="gpt-4.1-mini"),
    "long-context": ModelTarget(provider="claude", model="claude-sonnet-4-5"),
    "multimodal": ModelTarget(provider="gemini", model="gemini-2.5-flash"),
}

AUTO_FILE_KIND_MODEL_PRIORITY: dict[str, list[str]] = {
    "image": ["multimodal", "cost-fast", "long-context"],
    "pdf": ["multimodal", "long-context", "reasoning-high"],
    "docx": ["multimodal", "reasoning-high", "long-context"],
    "xlsx": ["reasoning-high", "long-context", "multimodal"],
    "pptx": ["multimodal", "reasoning-high", "long-context"],
}

AUTO_TEXT_ONLY_MODEL_PRIORITY: list[str] = ["reasoning-high", "long-context", "cost-fast"]


def resolve_model(abstract_model_key: str) -> ModelTarget:
    target = MODEL_MAP.get(abstract_model_key)
    if target is None:
        valid = ", ".join(sorted(MODEL_MAP.keys()))
        raise ModelMappingError(
            f"Unknown model '{abstract_model_key}'. Valid models: {valid}"
        )
    return target


def resolve_auto_candidates(file_kinds: Iterable[str]) -> list[ModelTarget]:
    file_kinds = list(file_kinds)
    if not file_kinds:
        return [resolve_model(key) for key in AUTO_TEXT_ONLY_MODEL_PRIORITY]

    scores: dict[str, int] = {key: 0 for key in MODEL_MAP.keys()}
    for kind in file_kinds:
        order = AUTO_FILE_KIND_MODEL_PRIORITY.get(kind)
        if not order:
            continue
        for rank, key in enumerate(order):
            scores[key] += (len(order) - rank) * 10

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ordered_keys = [key for key, score in ranked if score > 0]
    if not ordered_keys:
        ordered_keys = AUTO_TEXT_ONLY_MODEL_PRIORITY
    return [resolve_model(key) for key in ordered_keys]
