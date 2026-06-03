"""
main.py – VidScan Document Scanner API
========================================
Routes
------
  POST /convert         – upload video → download scanned PDF (old flow)
  POST /preview-frames  – upload video → JSON array of base64-JPEG scanned pages
  POST /apply-crop      – base64 image + crop % coords → cropped base64 image
  POST /generate-pdf    – array of edited pages → downloadable PDF
  POST /preview         – base64 image → base64 processed image (single-frame preview)
  GET  /health          – liveness probe
"""

import base64
import io
import logging
import os
import tempfile
import time
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.document_scanner import scan_document
from services.frame_extractor import extract_frames
from services.pdf_generator import frames_to_pdf
from services.supabase_client import get_supabase_client, download_file, upload_file, update_job


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config  (overridable via environment variables)
# ---------------------------------------------------------------------------
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS: list[str] = (
    ["*"] if _raw_origins.strip() == "*" else [o.strip() for o in _raw_origins.split(",")]
)

SUPPORTED_MIME_TYPES: frozenset[str] = frozenset({
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
    "video/webm",
    "video/mpeg",
    "video/ogg",
    "application/octet-stream",   # some browsers send this for any binary
})

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="VidScan – Document Scanner API",
    description=(
        "Upload a video of document pages and receive a clean, perspective-corrected PDF. "
        "Frames are sampled every 0.5 s; blurry and near-duplicate frames are discarded; "
        "each unique page is edge-detected, perspective-warped, and enhanced with "
        "adaptive thresholding before being assembled into a PDF."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Page-Count", "X-Processing-Time"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CropCoordinates(BaseModel):
    """
    Manual crop region expressed as percentages (0–100) of the image dimensions.
    All four fields must be provided together to trigger the manual override.
    """
    x: float   # left edge  as % of width
    y: float   # top  edge  as % of height
    w: float   # crop width as % of width
    h: float   # crop height as % of height


class PreviewRequest(BaseModel):
    """
    Base64-encoded image (JPEG or PNG) to process.

    Optionally, provide *crop* to bypass automatic boundary detection and
    apply an exact manual crop. Coordinates are percentages (0–100) of the
    true image dimensions, matching the values reported by the frontend
    canvas overlay.
    """
    image_b64: str
    crop: CropCoordinates | None = None


class PreviewResponse(BaseModel):
    """Base64-encoded JPEG of the processed image."""
    image_b64: str
    mode: str = "auto"   # 'manual' | 'auto' | 'fallback'


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    summary="Health / liveness check",
    tags=["Utility"],
)
async def health():
    """Returns ``{status: ok}`` – used by Railway and other orchestrators."""
    return {"status": "ok"}


def process_video_job(job_id: str):
    """
    Background worker task to download video from Supabase, extract frames,
    enhance them, compile to PDF, upload, and update job status.
    """
    import tempfile
    import os
    
    logger.info(f"Background process started for job {job_id}")
    
    # 1. Set status to processing
    try:
        update_job(job_id, {"status": "processing"})
    except Exception as e:
        logger.error(f"Failed to update status to processing for job {job_id}: {e}")
        return

    tmp_video_path = None
    tmp_pdf_path = None
    
    try:
        # Retrieve the job row to get the video path/URL
        supabase = get_supabase_client()
        job_data = supabase.table("jobs").select("video_url").eq("id", job_id).execute().data
        if not job_data:
            raise ValueError(f"Job {job_id} not found in database.")
        
        video_url = job_data[0].get("video_url")
        if not video_url:
            raise ValueError("No video_url found for this job.")

        # Extract bucket path key from URL (if it is a full HTTP URL)
        storage_path = video_url
        if video_url.startswith("http"):
            # Extract path after the public bucket name 'vidscan/'
            storage_path = video_url.split("/public/vidscan/")[-1]

        # Create local temp files for processing
        fd_v, tmp_video_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd_v)

        # Download video from Supabase Storage 'vidscan' bucket
        download_file("vidscan", storage_path, tmp_video_path)

        # Process the video to get frames
        raw_frames = extract_frames(tmp_video_path)
        if not raw_frames:
            raise ValueError("No usable frames could be extracted from the video.")

        # Enhance each frame
        scanned_frames = []
        for idx, raw_frame in enumerate(raw_frames):
            try:
                # Downscale raw frame to PREVIEW_MAX_DIM (1000px) before scanning to optimize memory & CPU
                h, w = raw_frame.shape[:2]
                longest = max(h, w)
                PREVIEW_MAX_DIM = 1000
                if longest > PREVIEW_MAX_DIM:
                    scale = PREVIEW_MAX_DIM / longest
                    resized_raw = cv2.resize(raw_frame, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
                else:
                    resized_raw = raw_frame
                
                scanned_frames.append(scan_document(resized_raw))
            except Exception:
                logger.exception("scan_document failed for frame %d – skipping", idx)

        if not scanned_frames:
            raise ValueError("Every frame failed to process during document scanning.")

        # Generate the PDF bytes
        pdf_bytes = frames_to_pdf(scanned_frames)

        # Write PDF to a temp file
        fd_p, tmp_pdf_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd_p)
        with open(tmp_pdf_path, "wb") as f:
            f.write(pdf_bytes)

        # Upload final PDF to Supabase Storage 'vidscan' bucket
        pdf_storage_path = f"results/{job_id}.pdf"
        pdf_url = upload_file("vidscan", pdf_storage_path, tmp_pdf_path, "application/pdf")

        # Mark job as completed
        update_job(job_id, {
            "status": "completed",
            "pdf_url": pdf_url
        })
        logger.info(f"Background process succeeded for job {job_id}")

    except Exception as e:
        logger.exception(f"Error processing job {job_id}")
        try:
            update_job(job_id, {
                "status": "failed",
                "error_message": str(e)
            })
        except Exception:
            logger.error("Could not write failure status to Supabase")
    finally:
        # Clean up temp files
        for path in (tmp_video_path, tmp_pdf_path):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass


@app.post(
    "/jobs/{job_id}/process",
    summary="Enqueue video processing background job in Supabase",
    tags=["Jobs"],
)
async def process_job_endpoint(job_id: str, background_tasks: BackgroundTasks):
    """
    Endpoint that registers a background task to process the video for the given job_id.
    """
    background_tasks.add_task(process_video_job, job_id)
    return {"status": "enqueued", "job_id": job_id}



# ---------------------------------------------------------------------------
# POST /convert
# ---------------------------------------------------------------------------

@app.post(
    "/convert",
    summary="Convert a document video to a clean scanned PDF",
    tags=["Conversion"],
    responses={
        200: {"content": {"application/pdf": {}}, "description": "Downloadable scanned PDF"},
        400: {"description": "Invalid input (unsupported MIME type, corrupt video, etc.)"},
        413: {"description": "File too large – maximum upload size is 50 MB"},
        422: {"description": "No usable frames found in the video"},
        500: {"description": "Server-side processing error"},
    },
)
async def convert_video_to_pdf(
    file: UploadFile = File(
        ...,
        description="Video of document pages (mp4 / mov / avi / mkv / webm).",
    ),
):
    """
    Full document scanning pipeline:

    1. Validate MIME type and file size (≤ 50 MB).
    2. Save to ``/tmp`` via a named temporary file.
    3. Extract frames every 0.5 s, dropping blurry and near-duplicate frames.
    4. Detect document edges, apply perspective correction, and enhance each frame.
    5. Assemble all scanned pages into a PDF and stream it back.

    The temp file is always deleted after the response, even on error.
    """
    t_start = time.perf_counter()
    tmp_path: str | None = None

    # ── 1. MIME validation ──────────────────────────────────────────────────
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{content_type}'. "
                "Please upload a video file (mp4, mov, avi, mkv, or webm)."
            ),
        )

    # ── 2. Read & size-check ────────────────────────────────────────────────
    video_bytes = await file.read()
    size_mb = len(video_bytes) / (1024 ** 2)
    logger.info("Upload  name=%r  size=%.1f MB  type=%r", file.filename, size_mb, content_type)

    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large: your upload is {size_mb:.1f} MB. "
                f"The maximum allowed size is {MAX_FILE_SIZE_MB} MB. "
                "Please trim or compress the video before uploading."
            ),
        )

    # ── 3. Write to /tmp ────────────────────────────────────────────────────
    original_name = file.filename or "upload.mp4"
    _, ext = os.path.splitext(original_name)
    ext = ext.lower() if ext else ".mp4"

    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False, dir="/tmp") as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        # ── 4. Extract frames ────────────────────────────────────────────────
        try:
            raw_frames = extract_frames(tmp_path)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        if not raw_frames:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No usable frames could be extracted. "
                    "The video may be too short, uniformly blurry, or all frames are duplicates."
                ),
            )

        # ── 5. Scan each frame ───────────────────────────────────────────────
        scanned_frames = []
        for frame in raw_frames:
            try:
                scanned_frames.append(scan_document(frame))
            except Exception:
                logger.exception("scan_document failed for a frame – skipping")

        if not scanned_frames:
            raise HTTPException(
                status_code=500,
                detail="Every frame failed during document scanning.",
            )

        # ── 6. Build PDF ─────────────────────────────────────────────────────
        try:
            pdf_bytes = frames_to_pdf(scanned_frames)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception:
            logger.exception("PDF generation failed")
            raise HTTPException(status_code=500, detail="PDF generation failed.")

    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error in /convert")
        raise HTTPException(status_code=500, detail="An unexpected server error occurred.")
    finally:
        # Always clean up the temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.debug("Temp file removed: %s", tmp_path)
            except Exception:
                logger.warning("Could not remove temp file: %s", tmp_path)

    elapsed = time.perf_counter() - t_start
    pdf_filename = os.path.splitext(original_name)[0] + "_scanned.pdf"
    logger.info(
        "PDF ready  pages=%d  size=%.1f KB  elapsed=%.2fs",
        len(scanned_frames), len(pdf_bytes) / 1024, elapsed,
    )

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{pdf_filename}"',
            "X-Page-Count": str(len(scanned_frames)),
            "X-Processing-Time": f"{elapsed:.2f}s",
        },
    )


# ---------------------------------------------------------------------------
# POST /preview
# ---------------------------------------------------------------------------

@app.post(
    "/preview",
    summary="Perspective-correct and enhance a single image (live preview)",
    response_model=PreviewResponse,
    tags=["Preview"],
    responses={
        400: {"description": "Invalid or undecodable base64 image data"},
        500: {"description": "Image processing error"},
    },
)
async def preview_frame(body: PreviewRequest):
    """
    Accepts a base64-encoded image (JPEG or PNG), runs the full document
    scanning pipeline, and returns the result as a base64-encoded JPEG.

    **Manual crop override**: if the optional *crop* field is provided the
    automatic boundary detection is bypassed entirely; the exact percentage
    coordinates are converted to absolute pixels and that region is returned.

    Intended for live frontend preview before the user commits to uploading
    a full video.
    """
    # ── Decode base64 → cv2 image ───────────────────────────────────────────
    try:
        img_bytes = base64.b64decode(body.image_b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("cv2.imdecode returned None – data is not a valid image.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}")

    # ── Extract optional manual crop coords ─────────────────────────────────
    crop_kwargs: dict = {}
    mode = "auto"

    if body.crop is not None:
        c = body.crop
        # Basic validation – reject obviously out-of-range values
        if not (0 <= c.x <= 100 and 0 <= c.y <= 100
                and 0 < c.w <= 100 and 0 < c.h <= 100
                and c.x + c.w <= 100 and c.y + c.h <= 100):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Crop coordinates out of range. "
                    "x, y, w, h must be 0–100 with x+w ≤ 100 and y+h ≤ 100."
                ),
            )
        crop_kwargs = {
            "crop_x_pct": c.x,
            "crop_y_pct": c.y,
            "crop_w_pct": c.w,
            "crop_h_pct": c.h,
        }
        mode = "manual"
        logger.info(
            "Preview: manual crop requested x=%.1f y=%.1f w=%.1f h=%.1f",
            c.x, c.y, c.w, c.h,
        )
    else:
        logger.info("Preview: auto boundary detection mode")

    # ── Scan ────────────────────────────────────────────────────────────────
    try:
        h_before, w_before = frame.shape[:2]
        processed = scan_document(frame, **crop_kwargs)
        h_after, w_after = processed.shape[:2]

        # Detect whether the fallback path was taken (same dimensions = no warp)
        if mode == "auto" and processed.shape == frame.shape and np.array_equal(processed, frame):
            mode = "fallback"
            logger.info("Preview: fallback – returning original frame")

    except Exception:
        logger.exception("Preview scan_document failed")
        raise HTTPException(status_code=500, detail="Image processing failed.")

    # ── Encode back to base64 ───────────────────────────────────────────────
    ok, buf = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode processed image.")

    return PreviewResponse(
        image_b64=base64.b64encode(buf.tobytes()).decode("utf-8"),
        mode=mode,
    )


# ---------------------------------------------------------------------------
# POST /preview-frames   (main scan & preview flow)
# ---------------------------------------------------------------------------

class ScannedPageItem(BaseModel):
    """One scanned page returned by /preview-frames."""
    page_number: int
    image: str        # base64-encoded JPEG
    width: int
    height: int


@app.post(
    "/preview-frames",
    summary="Extract, scan, and return all pages from a video as base64 JPEGs",
    response_model=List[ScannedPageItem],
    tags=["Scan"],
    responses={
        400: {"description": "Invalid video or unsupported MIME type"},
        413: {"description": "File too large"},
        422: {"description": "No usable frames extracted"},
        500: {"description": "Server-side processing error"},
    },
)
async def preview_frames(
    file: UploadFile = File(..., description="Video of document pages (mp4 / mov / avi / mkv / webm)."),
):
    """
    Upload a video, run the full Adobe-style scan pipeline, and receive every
    unique scanned page as a base64-encoded JPEG in a JSON array.

    The frontend stores these in sessionStorage and redirects to /preview for
    further editing.
    """
    t_start = time.perf_counter()
    tmp_path: str | None = None

    # ── 1. MIME validation ─────────────────────────────────────────────────────
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{content_type}'. "
                "Please upload a video file (mp4, mov, avi, mkv, or webm)."
            ),
        )

    # ── 2. Read & size-check ────────────────────────────────────────────────
    video_bytes = await file.read()
    size_mb = len(video_bytes) / (1024 ** 2)
    logger.info(
        "[preview-frames] upload name=%r  size=%.1f MB  type=%r",
        file.filename, size_mb, content_type,
    )

    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large: {size_mb:.1f} MB. "
                f"Maximum allowed size is {MAX_FILE_SIZE_MB} MB."
            ),
        )

    # ── 3. Write to /tmp ───────────────────────────────────────────────────
    original_name = file.filename or "upload.mp4"
    _, ext = os.path.splitext(original_name)
    ext = ext.lower() if ext else ".mp4"

    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False, dir="/tmp") as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        # ── 4. Extract frames ───────────────────────────────────────────────
        try:
            raw_frames = extract_frames(tmp_path)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        if not raw_frames:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No usable frames could be extracted. "
                    "The video may be too short, uniformly blurry, or all frames are duplicates."
                ),
            )

        # Cap at 30 pages to prevent excessive payload size
        MAX_PREVIEW_PAGES = 30
        if len(raw_frames) > MAX_PREVIEW_PAGES:
            logger.warning(
                "[preview-frames] %d pages detected – capping at %d",
                len(raw_frames), MAX_PREVIEW_PAGES,
            )
            raw_frames = raw_frames[:MAX_PREVIEW_PAGES]


        # ── 5. Encode each frame as base64 JPEG (resized for preview) ───────
        PREVIEW_MAX_DIM = 1000  # slightly smaller for faster transfer and processing
        PREVIEW_JPEG_QUALITY = 80

        result: List[ScannedPageItem] = []
        for idx, raw_frame in enumerate(raw_frames):
            try:
                h, w = raw_frame.shape[:2]
                longest = max(h, w)
                if longest > PREVIEW_MAX_DIM:
                    scale = PREVIEW_MAX_DIM / longest
                    new_w = max(1, int(round(w * scale)))
                    new_h = max(1, int(round(h * scale)))
                    resized_raw = cv2.resize(raw_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    resized_raw = raw_frame
                    new_w, new_h = w, h

                frame = scan_document(resized_raw)
            except Exception:
                logger.exception("[preview-frames] scan_document failed for frame %d – using raw", idx)
                frame = raw_frame

            h_f, w_f = frame.shape[:2]
            preview = frame

            ok, buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY])
            if not ok:
                logger.warning("[preview-frames] frame %d failed to encode – skipping", idx)
                continue
            result.append(
                ScannedPageItem(
                    page_number=idx + 1,
                    image=base64.b64encode(buf.tobytes()).decode("utf-8"),
                    width=w_f,
                    height=h_f,
                )
            )

        if not result:
            raise HTTPException(
                status_code=500,
                detail="Every frame failed to encode.",
            )

    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error in /preview-frames")
        raise HTTPException(status_code=500, detail="An unexpected server error occurred.")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                logger.warning("Could not remove temp file: %s", tmp_path)

    elapsed = time.perf_counter() - t_start
    logger.info(
        "[preview-frames] done  pages=%d  elapsed=%.2fs",
        len(result), elapsed,
    )
    return result


# ---------------------------------------------------------------------------
# POST /apply-crop
# ---------------------------------------------------------------------------

class ApplyCropRequest(BaseModel):
    """Crop a single base64 image by percentage coordinates."""
    image: str      # base64-encoded JPEG/PNG
    crop_x: float   # left edge  as % of width  (0–100)
    crop_y: float   # top  edge  as % of height (0–100)
    crop_width: float   # crop width  as % of image width
    crop_height: float  # crop height as % of image height


class ApplyCropResponse(BaseModel):
    image: str  # base64-encoded JPEG of the cropped result


@app.post(
    "/apply-crop",
    summary="Crop a single scanned page image using percentage coordinates",
    response_model=ApplyCropResponse,
    tags=["Edit"],
    responses={
        400: {"description": "Invalid image data or crop coordinates"},
        500: {"description": "Image processing error"},
    },
)
async def apply_crop(body: ApplyCropRequest):
    """
    Accepts a base64-encoded JPEG/PNG plus crop coordinates expressed as
    percentages (0–100) of the image dimensions, applies the exact crop,
    runs CLAHE + adaptive enhancement, and returns the result as base64 JPEG.
    """
    # ── Decode ────────────────────────────────────────────────────────────────
    try:
        img_bytes = base64.b64decode(body.image)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("cv2.imdecode returned None")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}")

    # ── Validate coords ──────────────────────────────────────────────────
    cx, cy, cw, ch = body.crop_x, body.crop_y, body.crop_width, body.crop_height
    if not (0 <= cx <= 100 and 0 <= cy <= 100
            and 0 < cw <= 100 and 0 < ch <= 100
            and cx + cw <= 100 and cy + ch <= 100):
        raise HTTPException(
            status_code=400,
            detail="Crop coordinates out of range (0–100, x+w≤100, y+h≤100).",
        )

    # ── Apply crop via scan_document manual override ──────────────────────
    try:
        cropped = scan_document(
            frame,
            crop_x_pct=cx,
            crop_y_pct=cy,
            crop_w_pct=cw,
            crop_h_pct=ch,
            enhance=False,
        )
    except Exception:
        logger.exception("/apply-crop scan_document failed")
        raise HTTPException(status_code=500, detail="Crop failed.")

    ok, buf = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode cropped image.")

    return ApplyCropResponse(
        image=base64.b64encode(buf.tobytes()).decode("utf-8")
    )


# ---------------------------------------------------------------------------
# POST /generate-pdf
# ---------------------------------------------------------------------------

class PageInput(BaseModel):
    """One edited page sent from the frontend."""
    image: str          # base64-encoded JPEG
    rotation: int = 0   # 0, 90, 180, 270
    brightness: int = 0 # -50 to +50
    contrast: int = 0   # -50 to +50


class GeneratePdfRequest(BaseModel):
    pages: List[PageInput]


@app.post(
    "/generate-pdf",
    summary="Compile edited scanned pages into a downloadable PDF",
    tags=["PDF"],
    responses={
        200: {"content": {"application/pdf": {}}, "description": "Downloadable PDF"},
        400: {"description": "No valid pages provided"},
        500: {"description": "PDF generation error"},
    },
)
async def generate_pdf(body: GeneratePdfRequest):
    """
    Accept an ordered list of base64-encoded JPEG pages (with optional
    rotation and brightness/contrast adjustments applied client-side via CSS
    filter), decode each, apply server-side rotation and brightness/contrast,
    and compile them into a single downloadable PDF.
    """
    if not body.pages:
        raise HTTPException(status_code=400, detail="No pages provided.")

    frames: list[np.ndarray] = []
    for idx, p in enumerate(body.pages):
        # ── Decode base64 image ────────────────────────────────────────────
        try:
            img_bytes = base64.b64decode(p.image)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("imdecode returned None")
        except Exception as exc:
            logger.warning("/generate-pdf page %d decode failed: %s – skipping", idx, exc)
            continue

        # ── Apply rotation ─────────────────────────────────────────────────
        rot = p.rotation % 360
        if rot == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rot == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # ── Apply brightness & contrast: out = clip(alpha*in + beta) ──────
        # brightness: -50..+50 → additive shift of -50..+50 on 0–255 scale
        # contrast:   -50..+50 → alpha multiplier 0.5–1.5
        alpha = 1.0 + p.contrast / 100.0   # contrast: 0–50 → 1.0–1.5
        beta  = float(p.brightness * 2)    # brightness: -50..+50 → -100..+100
        if alpha != 1.0 or beta != 0.0:
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        frames.append(frame)

    if not frames:
        raise HTTPException(
            status_code=400,
            detail="No valid pages could be decoded.",
        )

    try:
        pdf_bytes = frames_to_pdf(frames)
    except Exception:
        logger.exception("/generate-pdf PDF compilation failed")
        raise HTTPException(status_code=500, detail="PDF generation failed.")

    logger.info("/generate-pdf  pages=%d  size=%.1f KB", len(frames), len(pdf_bytes) / 1024)
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'attachment; filename="vidscan_document.pdf"',
            "X-Page-Count": str(len(frames)),
        },
    )
