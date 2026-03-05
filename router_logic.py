from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class ModelMappingError(ValueError):
    pass


@dataclass(frozen=True)
class ModelTarget:
    provider: str
    model: str


@dataclass(frozen=True)
class ModelScoreBreakdown:
    model_key: str
    final_score: float
    quality_score: float
    task_fit_score: float
    image_score: float | None
    latency_score: float
    cost_score: float
    ratings_used: dict[str, float]


MODEL_MAP: dict[str, ModelTarget] = {
    "gpt-5.3-codex": ModelTarget(provider="openai", model="gpt-5.3-codex"),
    "gpt-5.2": ModelTarget(provider="openai", model="gpt-5.2"),
    "gpt-5-mini": ModelTarget(provider="openai", model="gpt-5-mini"),
    "claude-opus-4-6": ModelTarget(provider="claude", model="claude-opus-4-6"),
    "claude-sonnet-4-6": ModelTarget(provider="claude", model="claude-sonnet-4-6"),
    "claude-haiku-4-5": ModelTarget(provider="claude", model="claude-haiku-4-5"),
    "gemini-3.1-pro-preview": ModelTarget(provider="gemini", model="gemini-3.1-pro-preview"),
    "gemini-3-flash-preview": ModelTarget(provider="gemini", model="gemini-3-flash-preview"),
}

ROUTING_OBJECTIVE = "balanced"
MODEL_RATINGS_FILE = Path(__file__).with_name("model_ratings.json")
PERFORMANCE_FIELDS: tuple[str, ...] = (
    "language",
    "reasoning",
    "coding",
    "mathematics",
    "data_analysis",
)
DEFAULT_QUERY_REQUIREMENTS: dict[str, float] = {
    "language": 0.35,
    "reasoning": 0.30,
    "coding": 0.20,
    "mathematics": 0.05,
    "data_analysis": 0.10,
}
OBJECTIVE_WEIGHTS: dict[str, dict[str, float]] = {
    "quality-first": {"quality": 0.80, "latency": 0.10, "cost": 0.10},
    "latency-first": {"quality": 0.30, "latency": 0.55, "cost": 0.15},
    "cost-first": {"quality": 0.25, "latency": 0.20, "cost": 0.55},
    "balanced": {"quality": 0.55, "latency": 0.25, "cost": 0.20},
}
IMAGE_ALPHA = 0.40


def _load_model_ratings() -> dict[str, dict[str, float]]:
    try:
        payload = json.loads(MODEL_RATINGS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return {}

    raw_models = payload.get("models", {})
    if not isinstance(raw_models, dict):
        return {}

    normalized: dict[str, dict[str, float]] = {}
    for key, values in raw_models.items():
        if not isinstance(key, str) or not isinstance(values, dict):
            continue
        entry: dict[str, float] = {}
        for metric, raw_value in values.items():
            try:
                entry[str(metric)] = float(raw_value)
            except (TypeError, ValueError):
                continue
        normalized[key] = entry
    return normalized


MODEL_RATINGS = _load_model_ratings()


def resolve_model(abstract_model_key: str) -> ModelTarget:
    target = MODEL_MAP.get(abstract_model_key)
    if target is None:
        valid = ", ".join(sorted(MODEL_MAP.keys()))
        raise ModelMappingError(
            f"Unknown model '{abstract_model_key}'. Valid models: {valid}"
        )
    return target


def resolve_auto_candidates(
    file_kinds: Iterable[str],
    query_requirements: dict[str, float] | None = None,
    has_image: bool = False,
    objective: str = ROUTING_OBJECTIVE,
) -> list[ModelTarget]:
    scored = score_auto_candidates(
        file_kinds=file_kinds,
        query_requirements=query_requirements,
        has_image=has_image,
        objective=objective,
    )
    return [resolve_model(item.model_key) for item in scored]


def score_auto_candidates(
    file_kinds: Iterable[str],
    query_requirements: dict[str, float] | None = None,
    has_image: bool = False,
    objective: str = ROUTING_OBJECTIVE,
) -> list[ModelScoreBreakdown]:
    _ = list(file_kinds)
    requirements = _normalize_requirements(query_requirements or DEFAULT_QUERY_REQUIREMENTS)
    weights = OBJECTIVE_WEIGHTS.get(objective, OBJECTIVE_WEIGHTS[ROUTING_OBJECTIVE])
    scored: list[ModelScoreBreakdown] = []
    for key in MODEL_MAP:
        scored.append(
            _score_with_ratings(
            model_key=key,
            requirements=requirements,
            has_image=has_image,
            weights=weights,
        )
        )
    return sorted(scored, key=lambda item: item.final_score, reverse=True)


def _score_with_ratings(
    model_key: str,
    requirements: dict[str, float],
    has_image: bool,
    weights: dict[str, float],
) -> ModelScoreBreakdown:
    metrics = MODEL_RATINGS.get(model_key, {})

    task_fit = 0.0
    for field in PERFORMANCE_FIELDS:
        task_fit += metrics.get(field, 5.0) * requirements.get(field, 0.0)

    quality_score = task_fit
    image_score: float | None = None
    if has_image:
        image_score = metrics.get("image", task_fit)
        quality_score = (1.0 - IMAGE_ALPHA) * task_fit + IMAGE_ALPHA * image_score

    latency = metrics.get("latency", 5.0)
    cost = metrics.get("cost", 5.0)

    final_score = (
        quality_score * weights["quality"]
        + latency * weights["latency"]
        + cost * weights["cost"]
    )
    ratings_used = {
        "language": metrics.get("language", 5.0),
        "reasoning": metrics.get("reasoning", 5.0),
        "coding": metrics.get("coding", 5.0),
        "mathematics": metrics.get("mathematics", 5.0),
        "data_analysis": metrics.get("data_analysis", 5.0),
        "image": metrics.get("image", task_fit),
        "latency": latency,
        "cost": cost,
    }
    return ModelScoreBreakdown(
        model_key=model_key,
        final_score=final_score,
        quality_score=quality_score,
        task_fit_score=task_fit,
        image_score=image_score,
        latency_score=latency,
        cost_score=cost,
        ratings_used=ratings_used,
    )


def _normalize_requirements(requirements: dict[str, float]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for field in PERFORMANCE_FIELDS:
        raw = requirements.get(field, 0.0)
        try:
            value = max(0.0, float(raw))
        except (TypeError, ValueError):
            value = 0.0
        cleaned[field] = value

    total = sum(cleaned.values())
    if total <= 0:
        return DEFAULT_QUERY_REQUIREMENTS.copy()
    return {field: value / total for field, value in cleaned.items()}
