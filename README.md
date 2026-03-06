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
- `query_evaluator_model` (auto mode): which model infers query requirements; defaults to `gpt-5-mini`
- Supports optional `file_ids` from `/files/upload`
- Reasoning effort behavior for OpenAI GPT-5 family:
  - Normal generation requests use `reasoning.effort="low"`
  - Internal query-requirement scoring (`gpt-5-mini`) uses `reasoning.effort="medium"`
- Gemini thinking defaults:
  - `gemini-3.1-pro-preview` uses `generationConfig.thinkingConfig.thinkingLevel="low"`
  - `gemini-3-flash-preview` uses `generationConfig.thinkingConfig.thinkingLevel="minimal"`

Example request (auto):

```json
{
  "input": "Explain CAP theorem in 3 bullets.",
  "mode": "auto",
  "routing_objective": "balanced",
  "manual_model": null,
  "query_evaluator_model": "gpt-5-mini",
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
- Auto response also includes `query_requirement_metrics` (latency, token usage, estimated cost for the gpt-5-mini requirement-scorer call).

## Example Prompts (Text-Only)

1. `Write a joke about d/dx e^x = e^x.`
2. `Fix this Python bug and return code only: if n % 2 = 0`
3. `Write a Java quickSort implementation for an int[] and include a short main method example.`
4. `Prove the Central Limit Theorem for i.i.d. variables with finite mean and variance using characteristic functions; give a rigorous step-by-step proof and state every theorem used.`
5. `Explain the Central Limit Theorem intuitively to a middle school student, without heavy math notation.`
