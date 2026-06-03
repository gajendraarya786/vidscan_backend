---
title: VidScan Backend
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# VidScan Backend

FastAPI service that accepts a video upload and returns a PDF containing one **unique** frame per second.

## How it works

1. The client `POST`s a video file (multipart form-data) to `/convert`.
2. The server samples one frame per second using **OpenCV**.
3. Consecutive frames whose mean absolute pixel difference is **< 30** are discarded as "similar".
4. The remaining unique frames are compiled into a PDF via **img2pdf**.
5. The PDF is streamed back as a file download.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/convert` | Upload video → download PDF |
| `GET`  | `/health`  | Health check (returns `{"status":"ok"}`) |
| `GET`  | `/docs`    | Swagger UI (auto-generated) |

### `POST /convert`

**Request** – `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | binary (video) | ✅ | mp4 / mov / avi / mkv / webm |

**Response headers**

| Header | Example | Description |
|--------|---------|-------------|
| `Content-Disposition` | `attachment; filename="myvideo_frames.pdf"` | Triggers browser download |
| `X-Frame-Count` | `42` | Number of unique frames in the PDF |
| `X-Processing-Time` | `3.14s` | Server-side processing duration |

---

## Local development

```bash
# 1. Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dev server (auto-reload)
uvicorn main:app --reload --port 8000
```

Then open <http://localhost:8000/docs> for the interactive Swagger UI.

### Quick curl test

```bash
curl -X POST http://localhost:8000/convert \
     -F "file=@/path/to/your/video.mp4" \
     --output output.pdf
```

---

## Configuration

| Env variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Port the server listens on (set automatically by Railway) |

Tunables live as module-level constants in `main.py`:

| Constant | Default | Description |
|---|---|---|
| `PIXEL_DIFF_THRESHOLD` | `30` | Mean-abs pixel diff below which a frame is skipped |
| `MAX_FILE_SIZE_MB` | `500` | Maximum accepted upload size |

---

## Deployment on Railway

1. Push this directory to a GitHub repository.
2. In Railway, click **"New Project → Deploy from GitHub repo"** and select your repo.
3. Railway auto-detects the `Procfile` and `requirements.txt`.
4. Set any environment variables you need in Railway's **Variables** tab.
5. Railway injects `$PORT` automatically – no changes needed.

---

## Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | Web framework |
| `uvicorn` | ASGI server |
| `python-multipart` | Multipart form / file uploads |
| `opencv-python-headless` | Video decoding & frame extraction |
| `img2pdf` | Lossless JPEG-to-PDF conversion |
| `numpy` | Pixel-diff computation |
