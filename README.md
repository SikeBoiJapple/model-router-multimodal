# Centralized Model Router (FastAPI)

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

UI:

- `GET /ui` provides prompt + file upload with `auto` and `manual` routing modes.

## API

`POST /generate`

Single generation endpoint for all routing behavior.

- `mode="auto"`: scored routing + fallback chain
- `mode="manual"`: uses `manual_model`
- `routing_objective`: `quality-first` | `latency-first` | `cost-first` | `balanced`
- Supports optional `file_ids` from `/files/upload`

Example request (auto):

```json
{
  "input": "Explain CAP theorem in 3 bullets.",
  "mode": "auto",
  "routing_objective": "balanced",
  "manual_model": null,
  "file_ids": [],
  "options": {}
}
```

`POST /files/upload` (multipart):

- Accepts: `image/*`, `.pdf`, `.docx`, `.xlsx`, `.pptx`
- Returns `file_id` values for generation calls

`GET /files/{file_id}`:

- Returns stored file metadata

`DELETE /files/{file_id}`:

- Removes an uploaded file from in-memory storage

Routing details:

- Auto routing uses `model_ratings.json` and scores each model with:
  - `final_score = w_quality * quality_score + w_latency * latency + w_cost * cost`
  - `quality_score` comes from a dot product of model performance fields and query requirement scores inferred by `gpt-5-mini`
  - when an image is uploaded, `quality_score` is blended with model `image` score using a fixed alpha
- UI shows routing breakdown (query requirements, objective weights, per-model scores) in the `Routing Scores` panel.
- `manual_model` supports:
  - `gpt-5.3-codex`
  - `gpt-5.2`
  - `gpt-5-mini`
  - `claude-opus-4-6`
  - `claude-sonnet-4-6`
  - `claude-haiku-4-5`
  - `gemini-3.1-pro-preview`
  - `gemini-3-flash-preview`
- Prompt-only (text) usage is supported without files.
- UI supports uploading multiple files and removing files before generation.
