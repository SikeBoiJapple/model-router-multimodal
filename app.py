from __future__ import annotations

import logging
from pathlib import Path
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
    GenerateRequest,
    GenerateResponse,
)
from router_logic import MODEL_MAP, ModelMappingError, resolve_auto_candidates, resolve_model

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


@app.get("/ui", response_class=HTMLResponse)
async def ui() -> HTMLResponse:
    if not UI_FILE.exists():
        raise HTTPException(status_code=404, detail="UI file not found")
    return HTMLResponse(UI_FILE.read_text(encoding="utf-8"))


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


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    try:
        target = resolve_model(request.model)
        logger.info(
            "Routing request model_key=%s provider=%s concrete_model=%s",
            request.model,
            target.provider,
            target.model,
        )
        normalized = await apim_client.generate(
            provider=target.provider,
            model=target.model,
            prompt=request.input,
            options=request.options,
        )
        return GenerateResponse(
            provider=target.provider,
            model=target.model,
            output=normalized["output"],
            raw=normalized["raw"],
        )
    except ModelMappingError as exc:
        logger.warning("Invalid abstract model key: %s", request.model)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except APIMClientError as exc:
        logger.error("APIM request failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@app.post("/generate/auto", response_model=GenerateAutoResponse)
async def generate_auto(request: GenerateAutoRequest) -> GenerateAutoResponse:
    try:
        files = file_store.get_many(request.file_ids)
        file_kinds = [item.kind for item in files]

        if request.mode == "manual":
            if not request.manual_model:
                raise HTTPException(
                    status_code=400,
                    detail="manual_model is required when mode='manual'",
                )
            targets = [resolve_model(request.manual_model)]
        else:
            targets = resolve_auto_candidates(file_kinds)

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
                normalized = await apim_client.generate(
                    provider=target.provider,
                    model=target.model,
                    prompt=request.input,
                    options=merged_options,
                )
                return GenerateAutoResponse(
                    provider=target.provider,
                    model=target.model,
                    output=normalized["output"],
                    raw=normalized["raw"],
                    route_mode=request.mode,
                    route_candidates=route_candidates,
                    used_file_ids=request.file_ids,
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
