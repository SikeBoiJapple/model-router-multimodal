from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    model: str = Field(..., description="Abstract model key, e.g. reasoning-high")
    input: str = Field(..., min_length=1, description="Prompt input text")
    options: dict[str, Any] = Field(default_factory=dict)


class GenerateAutoRequest(BaseModel):
    input: str = Field(default="", description="Prompt input text")
    mode: Literal["auto", "manual"] = "auto"
    manual_model: str | None = None
    file_ids: list[str] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)


class GenerateResponse(BaseModel):
    provider: str
    model: str
    output: str
    raw: dict[str, Any]


class FileUploadItem(BaseModel):
    file_id: str
    filename: str
    kind: str
    mime_type: str
    size_bytes: int


class FileUploadResponse(BaseModel):
    files: list[FileUploadItem]


class FileInfoResponse(BaseModel):
    file_id: str
    filename: str
    kind: str
    mime_type: str
    size_bytes: int


class GenerateAutoResponse(GenerateResponse):
    route_mode: Literal["auto", "manual"]
    route_candidates: list[str] = Field(default_factory=list)
    used_file_ids: list[str] = Field(default_factory=list)
