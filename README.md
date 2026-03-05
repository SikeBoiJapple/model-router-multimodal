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

Example request:

```json
{
  "model": "gpt-5.2",
  "input": "Explain CAP theorem in 3 bullets.",
  "options": {
    "temperature": 0.2
  }
}
```

`POST /files/upload` (multipart):

- Accepts: `image/*`, `.pdf`, `.docx`, `.xlsx`, `.pptx`
- Returns `file_id` values for generation calls

`GET /files/{file_id}`:

- Returns stored file metadata

`DELETE /files/{file_id}`:

- Removes an uploaded file from in-memory storage

`POST /generate/auto`:

- `mode="auto"` routes by file type with fallback chain
- `routing_objective` supports: `quality-first`, `latency-first`, `cost-first`, `balanced`
- Auto routing uses `model_ratings.json` and scores each model with:
  - `final_score = w_quality * quality_score + w_latency * latency + w_cost * cost`
  - `quality_score` comes from a dot product of model performance fields and query requirement scores inferred by `gpt-5-mini`
  - when an image is uploaded, `quality_score` is blended with model `image` score using a fixed alpha
- UI now shows a routing score breakdown (query requirements, objective weights, and per-model scores) in the `Routing Scores` panel.
- `mode="manual"` uses `manual_model` and supports:
  - `gpt-5.3-codex`
  - `gpt-5.2`
  - `gpt-5-mini`
  - `claude-opus-4-6`
  - `claude-sonnet-4-6`
  - `claude-haiku-4-5`
  - `gemini-3.1-pro-preview`
  - `gemini-3-flash-preview`
- Supports prompt-only (text) usage without file uploads
- UI supports uploading multiple files and removing uploaded files before generation
