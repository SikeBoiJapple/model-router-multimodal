from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from apim_client import APIMClient, APIMClientError
from config_loader import load_settings
from file_pipeline import FilePipelineError, FileStore, ProcessedFile, process_uploaded_file
from models import (
    FileInfoResponse,
    FileUploadItem,
    FileUploadResponse,
    GenerateAutoRequest,
    GenerateAutoResponse,
)
from router_logic import (
    DEFAULT_QUERY_REQUIREMENTS,
    IMAGE_ALPHA,
    MODEL_MAP,
    OBJECTIVE_WEIGHTS,
    ModelMappingError,
    score_auto_candidates,
    resolve_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("model-router")

settings = load_settings()
apim_client = APIMClient(settings=settings)
file_store = FileStore()
app = FastAPI(title="Centralized Model Router", version="1.1.0")
UI_FILE = Path(__file__).with_name("ui.html")
QUERY_SCORER_MODEL_KEY = "gpt-5-mini"
QUERY_SCORER_REASONING_EFFORT = "medium"
DEFAULT_OPENAI_REASONING_EFFORT = "low"
DEFAULT_ROUTER_OBJECTIVE = "balanced"
MODEL_PRICING_USD_PER_MILLION: dict[str, dict[str, float]] = {
    "gpt-5.3-codex": {"input": 1.75, "output": 14.00},
    "gpt-5.2": {"input": 1.75, "output": 14.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "claude-opus-4-6": {"input": 5.00, "output": 25.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
}


@app.get("/ui", response_class=HTMLResponse)
async def ui() -> HTMLResponse:
    if not UI_FILE.exists():
        raise HTTPException(status_code=404, detail="UI file not found")
    return HTMLResponse(UI_FILE.read_text(encoding="utf-8"))


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    return HTMLResponse(
        """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model Router</title>
</head>
<body style="font-family:Segoe UI,Tahoma,sans-serif;padding:24px;">
  <h1>Model Router</h1>
  <p>Quick links:</p>
  <ul>
    <li><a href="/ui">Playground UI</a></li>
    <li><a href="/docs">Swagger Docs</a></li>
    <li><a href="/redoc">ReDoc</a></li>
  </ul>
</body>
</html>
        """.strip()
    )


@app.post("/files/upload", response_model=FileUploadResponse)
async def upload_files(files: list[UploadFile] = File(...)) -> FileUploadResponse:
    uploaded: list[FileUploadItem] = []
    for upload in files:
        try:
            data = await upload.read()
            item = process_uploaded_file(upload.filename or "", upload.content_type, data)
            file_store.put(item)
            uploaded.append(
                FileUploadItem(
                    file_id=item.file_id,
                    filename=item.filename,
                    kind=item.kind,
                    mime_type=item.mime_type,
                    size_bytes=item.size_bytes,
                )
            )
        except FilePipelineError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileUploadResponse(files=uploaded)


@app.get("/files/{file_id}", response_model=FileInfoResponse)
async def get_file(file_id: str) -> FileInfoResponse:
    try:
        item = file_store.get(file_id)
    except FilePipelineError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return FileInfoResponse(
        file_id=item.file_id,
        filename=item.filename,
        kind=item.kind,
        mime_type=item.mime_type,
        size_bytes=item.size_bytes,
    )


@app.delete("/files/{file_id}")
async def delete_file(file_id: str) -> dict[str, str]:
    try:
        file_store.delete(file_id)
    except FilePipelineError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"deleted_file_id": file_id}


@app.post("/generate", response_model=GenerateAutoResponse)
async def generate(request: GenerateAutoRequest) -> GenerateAutoResponse:
    try:
        files = file_store.get_many(request.file_ids)
        file_kinds = [item.kind for item in files]
        objective = request.routing_objective or DEFAULT_ROUTER_OBJECTIVE

        if request.mode == "manual":
            if not request.manual_model:
                raise HTTPException(
                    status_code=400,
                    detail="manual_model is required when mode='manual'",
                )
            targets = [resolve_model(request.manual_model)]
            requirements = {}
            query_requirement_metrics: dict[str, Any] = {}
            objective_weights = {}
            routing_scores: list[dict[str, Any]] = []
        else:
            scorer_model_key = request.query_evaluator_model or QUERY_SCORER_MODEL_KEY
            requirements, query_requirement_metrics = await _infer_query_requirements(
                request.input,
                files,
                scorer_model_key,
            )
            objective_weights = OBJECTIVE_WEIGHTS.get(
                objective, OBJECTIVE_WEIGHTS[DEFAULT_ROUTER_OBJECTIVE]
            )
            scored = score_auto_candidates(
                file_kinds,
                query_requirements=requirements,
                has_image=("image" in file_kinds),
                objective=objective,
            )
            targets = [resolve_model(item.model_key) for item in scored]
            routing_scores = [
                {
                    "model_key": item.model_key,
                    "final_score": round(item.final_score, 4),
                    "quality_score": round(item.quality_score, 4),
                    "task_fit_score": round(item.task_fit_score, 4),
                    "image_score": round(item.image_score, 4) if item.image_score is not None else None,
                    "latency_score": round(item.latency_score, 4),
                    "cost_score": round(item.cost_score, 4),
                    "ratings_used": item.ratings_used,
                }
                for item in scored
            ]

        route_candidates = [_model_key_for_target(target.provider, target.model) for target in targets]
        errors: list[str] = []

        for target in targets:
            try:
                merged_options = _build_options_for_target(
                    provider=target.provider,
                    prompt=request.input,
                    files=files,
                    base_options=request.options,
                )
                merged_options = _apply_openai_reasoning_effort(
                    provider=target.provider,
                    model=target.model,
                    options=merged_options,
                    effort=DEFAULT_OPENAI_REASONING_EFFORT,
                )
                merged_options = _apply_gemini_thinking_level(
                    provider=target.provider,
                    model=target.model,
                    options=merged_options,
                )
                started = perf_counter()
                normalized = await apim_client.generate(
                    provider=target.provider,
                    model=target.model,
                    prompt=request.input,
                    options=merged_options,
                )
                latency_ms = (perf_counter() - started) * 1000.0
                token_usage = _extract_token_usage(target.provider, normalized["raw"])
                estimated_cost = _estimate_cost_usd(target.model, token_usage)
                return GenerateAutoResponse(
                    provider=target.provider,
                    model=target.model,
                    output=normalized["output"],
                    raw=normalized["raw"],
                    latency_ms=round(latency_ms, 2),
                    token_usage=token_usage,
                    estimated_cost_usd=estimated_cost,
                    route_mode=request.mode,
                    route_candidates=route_candidates,
                    used_file_ids=request.file_ids,
                    routing_objective=objective if request.mode == "auto" else None,
                    query_requirements=requirements,
                    query_requirement_metrics=query_requirement_metrics,
                    objective_weights=objective_weights,
                    image_alpha=IMAGE_ALPHA if ("image" in file_kinds and request.mode == "auto") else None,
                    routing_scores=routing_scores,
                )
            except APIMClientError as exc:
                logger.warning(
                    "Candidate failed model=%s provider=%s error=%s",
                    target.model,
                    target.provider,
                    exc,
                )
                errors.append(f"{target.provider}/{target.model}: {exc}")
                continue

        raise HTTPException(
            status_code=502,
            detail="All candidate routes failed: " + " | ".join(errors),
        )
    except ModelMappingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FilePipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": exc.detail, "status": exc.status_code}},
    )


def _model_key_for_target(provider: str, model: str) -> str:
    for key, target in MODEL_MAP.items():
        if target.provider == provider and target.model == model:
            return key
    return f"{provider}:{model}"


async def _infer_query_requirements(
    prompt: str,
    files: list[ProcessedFile],
    scorer_model_key: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    scorer_target = None
    try:
        scorer_target = resolve_model(scorer_model_key)
        scorer_prompt = (
            "You are a routing feature scorer.\n"
            "Score requirements using both the user text and any attached files/images.\n"
            "Score task requirements across these fields:\n"
            "- language\n"
            "- reasoning\n"
            "- coding\n"
            "- mathematics\n"
            "- data_analysis\n\n"
            "Return JSON only with this exact shape:\n"
            '{"language":0.0,"reasoning":0.0,"coding":0.0,"mathematics":0.0,"data_analysis":0.0}\n'
            "Rules:\n"
            "- Values must be non-negative numbers.\n"
            "- Values should sum to 1.0.\n"
            "- No explanation text.\n\n"
            f"User prompt:\n{prompt}"
        )
        scorer_options = _build_options_for_target(
            provider=scorer_target.provider,
            prompt=scorer_prompt,
            files=files,
            base_options={},
        )
        scorer_options = _apply_openai_reasoning_effort(
            provider=scorer_target.provider,
            model=scorer_target.model,
            options=scorer_options,
            effort=QUERY_SCORER_REASONING_EFFORT,
        )
        scorer_options = _apply_gemini_thinking_level(
            provider=scorer_target.provider,
            model=scorer_target.model,
            options=scorer_options,
        )
        started = perf_counter()
        normalized = await apim_client.generate(
            provider=scorer_target.provider,
            model=scorer_target.model,
            prompt=scorer_prompt,
            options=scorer_options,
        )
        latency_ms = (perf_counter() - started) * 1000.0
        token_usage = _extract_token_usage(scorer_target.provider, normalized["raw"])
        estimated_cost = _estimate_cost_usd(scorer_target.model, token_usage)
        metrics: dict[str, Any] = {
            "provider": scorer_target.provider,
            "model": scorer_target.model,
            "latency_ms": round(latency_ms, 2),
            "token_usage": token_usage,
            "estimated_cost_usd": estimated_cost,
        }
        parsed = _parse_query_requirements(normalized["output"])
        if parsed:
            return parsed, metrics
        logger.warning("Query requirement scorer returned invalid output; using defaults")
        metrics["error"] = "invalid_scorer_output"
        return DEFAULT_QUERY_REQUIREMENTS.copy(), metrics
    except (APIMClientError, ModelMappingError) as exc:
        logger.warning("Query requirement scoring failed (%s); using defaults", exc)
        metrics: dict[str, Any] = {
            "provider": scorer_target.provider if scorer_target else "openai",
            "model": scorer_target.model if scorer_target else scorer_model_key,
            "error": str(exc),
        }
        return DEFAULT_QUERY_REQUIREMENTS.copy(), metrics


def _parse_query_requirements(raw_output: str) -> dict[str, float] | None:
    try:
        parsed = json.loads(raw_output.strip())
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    keys = ("language", "reasoning", "coding", "mathematics", "data_analysis")
    values: dict[str, float] = {}
    for key in keys:
        raw = parsed.get(key)
        try:
            value = max(0.0, float(raw))
        except (TypeError, ValueError):
            return None
        values[key] = value

    total = sum(values.values())
    if total <= 0:
        return None
    return {key: value / total for key, value in values.items()}


def _build_options_for_target(
    provider: str,
    prompt: str,
    files: list[ProcessedFile],
    base_options: dict[str, Any],
) -> dict[str, Any]:
    options = dict(base_options)
    if not files:
        return options

    prompt = prompt.strip()
    if provider == "gemini":
        parts: list[dict[str, Any]] = []
        if prompt:
            parts.append({"text": prompt})
        for item in files:
            if item.kind in {"image", "pdf"} and item.b64_data:
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": item.mime_type,
                            "data": item.b64_data,
                        }
                    }
                )
            else:
                parts.append({"text": _tool_text_block(item)})
        options["contents"] = [{"role": "user", "parts": parts}]
        return options

    if provider == "openai":
        content: list[dict[str, Any]] = []
        if prompt:
            content.append({"type": "input_text", "text": prompt})
        for item in files:
            if item.kind == "image" and item.b64_data:
                content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{item.mime_type};base64,{item.b64_data}",
                    }
                )
            else:
                content.append({"type": "input_text", "text": _tool_text_block(item)})
        options["input"] = [{"role": "user", "content": content}]
        return options

    if provider == "claude":
        content: list[dict[str, Any]] = []
        if prompt:
            content.append({"type": "text", "text": prompt})
        for item in files:
            if item.kind == "image" and item.b64_data:
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": item.mime_type,
                            "data": item.b64_data,
                        },
                    }
                )
            else:
                content.append({"type": "text", "text": _tool_text_block(item)})
        options["messages"] = [{"role": "user", "content": content}]
        return options

    return options


def _tool_text_block(item: ProcessedFile) -> str:
    snippet = (item.extracted_text or "").strip()
    if not snippet:
        snippet = f"No text could be extracted from {item.filename}."
    snippet = snippet[:12000]
    return (
        f"[Tool Extracted {item.kind.upper()} File: {item.filename}]\n"
        f"{snippet}\n"
        "[End Extracted File]"
    )


def _apply_openai_reasoning_effort(
    provider: str,
    model: str,
    options: dict[str, Any],
    effort: str,
) -> dict[str, Any]:
    if provider != "openai":
        return dict(options)
    if not model.startswith("gpt-5"):
        return dict(options)

    merged = dict(options)
    merged["reasoning"] = {"effort": effort}
    return merged


def _apply_gemini_thinking_level(
    provider: str,
    model: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    if provider != "gemini":
        return dict(options)

    model_name = model.lower()
    level: str | None = None
    if model_name == "gemini-3.1-pro-preview":
        level = "low"
    elif model_name == "gemini-3-flash-preview":
        level = "minimal"

    if level is None:
        return dict(options)

    merged = dict(options)
    generation_config = merged.get("generationConfig")
    if not isinstance(generation_config, dict):
        generation_config = {}
    else:
        generation_config = dict(generation_config)

    thinking_config = generation_config.get("thinkingConfig")
    if not isinstance(thinking_config, dict):
        thinking_config = {}
    else:
        thinking_config = dict(thinking_config)

    # Gemini 3 docs: do not mix thinkingLevel with legacy thinkingBudget.
    thinking_config.pop("thinkingBudget", None)
    thinking_config["thinkingLevel"] = level
    generation_config["thinkingConfig"] = thinking_config
    merged["generationConfig"] = generation_config
    return merged


def _extract_token_usage(provider: str, raw: dict[str, Any]) -> dict[str, int]:
    usage: dict[str, Any] = {}
    if provider == "openai":
        candidate = raw.get("usage", {})
        usage = candidate if isinstance(candidate, dict) else {}
        input_tokens = _to_int(usage.get("input_tokens")) or _to_int(usage.get("prompt_tokens"))
        output_tokens = _to_int(usage.get("output_tokens")) or _to_int(usage.get("completion_tokens"))
        total_tokens = _to_int(usage.get("total_tokens"))
    elif provider == "claude":
        candidate = raw.get("usage", {})
        usage = candidate if isinstance(candidate, dict) else {}
        input_tokens = _to_int(usage.get("input_tokens"))
        output_tokens = _to_int(usage.get("output_tokens"))
        total_tokens = _to_int(usage.get("total_tokens"))
    elif provider == "gemini":
        candidate = raw.get("usageMetadata", {})
        usage = candidate if isinstance(candidate, dict) else {}
        input_tokens = _to_int(usage.get("promptTokenCount"))
        output_tokens = _to_int(usage.get("candidatesTokenCount"))
        total_tokens = _to_int(usage.get("totalTokenCount"))
    else:
        input_tokens = None
        output_tokens = None
        total_tokens = None

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    payload: dict[str, int] = {}
    if input_tokens is not None:
        payload["input_tokens"] = input_tokens
    if output_tokens is not None:
        payload["output_tokens"] = output_tokens
    if total_tokens is not None:
        payload["total_tokens"] = total_tokens
    return payload


def _estimate_cost_usd(model: str, token_usage: dict[str, int]) -> float | None:
    prices = MODEL_PRICING_USD_PER_MILLION.get(model)
    if prices is None:
        return None

    input_tokens = token_usage.get("input_tokens")
    output_tokens = token_usage.get("output_tokens")
    if input_tokens is None and output_tokens is None:
        return None

    cost = 0.0
    if input_tokens is not None:
        cost += (input_tokens / 1_000_000.0) * prices["input"]
    if output_tokens is not None:
        cost += (output_tokens / 1_000_000.0) * prices["output"]
    return round(cost, 6)


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
