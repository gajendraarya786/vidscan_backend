"""
services/yolo_detector.py
~~~~~~~~~~~~~~~~~~~~~~~~~
YOLOv8-based document detector.

Usage
-----
  from services.yolo_detector import detect_document

  corners, confidence = detect_document(frame)
  # corners: np.ndarray shape (4, 2) in pixel coords, or None if not detected
  # confidence: float 0.0-1.0

The model is lazy-loaded on first call. If the model file does not exist,
returns (None, 0.0) so the caller can fall back to OpenCV detection.

Model resolution
----------------
Looks for the model in this priority order:
  1. $YOLO_MODEL_PATH environment variable
  2. models/document.pt   (custom trained model — best accuracy)
  3. models/yolov8n.pt    (pretrained COCO nano — used as bootstrap)

Training
--------
See scripts/train_yolov8_colab.py for the full training pipeline.
Once trained, drop `document.pt` into the `models/` directory and restart.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Model path resolution ─────────────────────────────────────────────────────

_BASE_DIR = Path(__file__).parent.parent
_MODEL_CANDIDATES = [
    os.getenv("YOLO_MODEL_PATH", ""),          # env override
    str(_BASE_DIR / "models" / "document.pt"), # custom trained
    str(_BASE_DIR / "models" / "yolov8n.pt"),  # COCO pretrained fallback
]

# ── Singleton state ───────────────────────────────────────────────────────────

_model = None           # type: ignore
_model_loaded: bool = False
_model_failed: bool = False  # set True on first load failure to stop retrying

# ── Confidence thresholds ─────────────────────────────────────────────────────

YOLO_CONF_THRESHOLD: float = 0.45   # minimum detection confidence to trust
YOLO_IOU_THRESHOLD:  float = 0.45   # NMS IoU threshold


def _load_model():
    """Try to load YOLOv8 model. Sets global _model and _model_loaded."""
    global _model, _model_loaded, _model_failed

    if _model_loaded or _model_failed:
        return

    # Find which model file exists
    model_path: str | None = None
    for candidate in _MODEL_CANDIDATES:
        if candidate and Path(candidate).exists():
            model_path = candidate
            break

    if model_path is None:
        logger.warning(
            "YOLOv8: no model file found. Checked: %s. "
            "Using OpenCV-only detection. "
            "To enable YOLO, add models/document.pt",
            [c for c in _MODEL_CANDIDATES if c]
        )
        _model_failed = True
        return

    try:
        # Import here so server starts even if ultralytics is not installed
        from ultralytics import YOLO  # type: ignore
        logger.info("YOLOv8: loading model from %s …", model_path)
        _model = YOLO(model_path)
        # Warm up with a tiny dummy frame to pre-compile ops
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        _model.predict(dummy, verbose=False, conf=YOLO_CONF_THRESHOLD)
        _model_loaded = True
        logger.info("YOLOv8: model loaded and warmed up ✓  path=%s", model_path)
    except ImportError:
        logger.warning(
            "YOLOv8: 'ultralytics' package not installed. "
            "Run: pip install ultralytics==8.2.18"
        )
        _model_failed = True
    except Exception:
        logger.exception("YOLOv8: failed to load model from %s", model_path)
        _model_failed = True


def is_yolo_available() -> bool:
    """Return True if YOLO model is loaded and ready."""
    _load_model()
    return _model_loaded


def detect_document(
    frame: np.ndarray,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Detect a document in *frame* using YOLOv8.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (H × W × 3).

    Returns
    -------
    corners : np.ndarray | None
        4×2 array of (x, y) corner pixel coordinates in the original frame,
        ordered [top-left, top-right, bottom-right, bottom-left].
        None if no document detected above confidence threshold.
    confidence : float
        Detection confidence 0.0–1.0. 0.0 if no detection.
    """
    _load_model()

    if not _model_loaded:
        return None, 0.0

    try:
        h, w = frame.shape[:2]

        # Run inference — imgsz=640 matches training config
        results = _model.predict(
            frame,
            imgsz=640,
            conf=YOLO_CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            verbose=False,
            device="cpu",  # safe for all deployments; GPU auto-used if available
        )

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None, 0.0

        # Pick highest-confidence detection
        boxes = results[0].boxes
        confs = boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confs))
        best_conf = float(confs[best_idx])

        if best_conf < YOLO_CONF_THRESHOLD:
            return None, best_conf

        # Get bounding box in pixel coords (xyxy format)
        xyxy = boxes.xyxy[best_idx].cpu().numpy()
        x1, y1, x2, y2 = xyxy

        # Clamp to frame bounds
        x1 = max(0.0, float(x1)); y1 = max(0.0, float(y1))
        x2 = min(float(w), float(x2)); y2 = min(float(h), float(y2))

        if (x2 - x1) < 20 or (y2 - y1) < 20:
            # Degenerate box
            return None, 0.0

        # Check if OBB (oriented bounding box) is available for precise corners
        if hasattr(results[0], "obb") and results[0].obb is not None:
            try:
                obb = results[0].obb
                if len(obb) > 0:
                    # xy: shape (N, 4, 2) — polygon corners
                    poly = obb.xy[best_idx].cpu().numpy()  # (4, 2)
                    corners = _order_corners(poly)
                    return corners, best_conf
            except Exception:
                pass  # fall through to axis-aligned bbox

        # Standard bbox → 4 corners
        corners = np.array([
            [x1, y1],  # top-left
            [x2, y1],  # top-right
            [x2, y2],  # bottom-right
            [x1, y2],  # bottom-left
        ], dtype=np.float32)

        return corners, best_conf

    except Exception:
        logger.exception("YOLOv8: inference failed")
        return None, 0.0


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 corner points as [TL, TR, BR, BL].
    Compatible with the geometry helpers in frame_extractor.py.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]    # top-left  (smallest x+y)
    rect[1] = pts[np.argmin(diff)] # top-right (smallest x-y)
    rect[2] = pts[np.argmax(s)]    # bottom-right (largest x+y)
    rect[3] = pts[np.argmax(diff)] # bottom-left (largest x-y)
    return rect
