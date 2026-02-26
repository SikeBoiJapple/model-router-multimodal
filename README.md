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
  "model": "reasoning-high",
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

`POST /generate/auto`:

- `mode="auto"` routes by file type with fallback chain
- `mode="manual"` uses `manual_model` (`reasoning-high`, `cost-fast`, `long-context`, `multimodal`)
- Supports prompt-only (text) usage without file uploads
