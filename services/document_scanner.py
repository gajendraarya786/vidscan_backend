"""
services/document_scanner.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adobe-Scan-quality document scanning pipeline:

Priority 1 – Manual Crop Override
    If percentage-based crop coordinates (x, y, w, h) are provided, convert
    them to absolute pixels and return the exact crop. DocTR / edge detection
    is completely skipped.

Priority 2 – Tuned OpenCV Boundary Detection (no DocTR dependency)
    Uses a multi-stage OpenCV pipeline designed to mimic Adobe Scan:
      • Softer adaptive Canny thresholds
      • Heavier dilation (7x7, 2 iters) to close edge gaps at page margins
      • Multi-epsilon contour approximation (won't miss slightly curved edges)
      • 5% dynamic padding added around the detected bounding box
      • Perspective warp to A4 (1240×1754)
      • CLAHE illumination correction (shadow removal)
      • Adaptive thresholding only when image is dark (preserves colour scans)

Priority 3 – Safety Fallback
    If detection fails or yields a degenerate / impossibly small area, the
    original, uncropped frame is returned untouched.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Tunables ─────────────────────────────────────────────────────────────────
MIN_DOC_AREA_RATIO: float    = 0.10   # Minimum contour area ratio to be a document
FULL_FRAME_THRESHOLD: float  = 0.20   # If quad < 20 % of frame → skip warp, use full frame
PADDING_RATIO: float         = 0.15   # 15 % outward padding around the detected quad
ADAPTIVE_BLOCK_SIZE: int     = 21
ADAPTIVE_C: int              = 10
BRIGHTNESS_BW_THRESHOLD: float = 160.0  # Only binarize frames darker than this


# ── Corner ordering ───────────────────────────────────────────────────────────

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Sort 4 corner points → (top-left, top-right, bottom-right, bottom-left)."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]    # top-left  – smallest x+y
    rect[2] = pts[np.argmax(s)]    # bottom-right – largest x+y
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right  – smallest y-x
    rect[3] = pts[np.argmax(diff)] # bottom-left – largest y-x
    return rect


# ── Document boundary detection ───────────────────────────────────────────────

def _find_document_quad(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Locate a document quadrilateral in *image* using a robust threshold-based
    contour search (designed to work even with partial hand occlusion).
    """
    h, w = image.shape[:2]
    min_area = h * w * MIN_DOC_AREA_RATIO

    # 1. Downscale for speed and structural consistency
    scale = 500.0 / max(h, w)
    small = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    sh, sw = small.shape[:2]

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # 2. Otsu thresholding to segment paper from desk
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Morphological close to bridge internal lines/text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    best_quad = None
    max_quad_area = 0.0

    # 4. Search contours on both foreground and background to support dark-on-light & light-on-dark
    for t_img in [thresh, cv2.bitwise_not(thresh)]:
        contours, _ = cv2.findContours(t_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
            
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            area = cv2.contourArea(c)
            if area < min_area * (scale ** 2):
                continue
            
            # Skip if the region is the entire canvas (likely pure background)
            if area > 0.95 * sh * sw:
                continue

            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            
            quad = None
            # Multi-epsilon DP approximation
            for eps in [0.02, 0.03, 0.04, 0.05, 0.06]:
                approx = cv2.approxPolyDP(hull, eps * peri, True)
                if len(approx) == 4:
                    quad = approx.reshape(4, 2)
                    break
            
            # Fallback to extreme convex hull corners if 4-point approx fails
            if quad is None:
                pts = hull.reshape(-1, 2)
                if len(pts) >= 4:
                    s = pts.sum(axis=1)
                    diff = np.diff(pts, axis=1)
                    quad = np.array([
                        pts[np.argmin(s)],
                        pts[np.argmin(diff)],
                        pts[np.argmax(s)],
                        pts[np.argmax(diff)],
                    ], dtype=np.float32)

            if quad is not None:
                qa = cv2.contourArea(cv2.convexHull(quad.astype(np.int32)))
                if qa > max_quad_area:
                    max_quad_area = qa
                    best_quad = quad

    if best_quad is not None:
        # Scale back to original coordinates
        return (best_quad / scale).astype(np.float32)
        
    return None


# ── Dynamic padding ────────────────────────────────────────────────────────────

def _pad_quad(quad: np.ndarray, img_h: int, img_w: int,
              ratio: float = PADDING_RATIO) -> np.ndarray:
    """
    Expand each corner of *quad* outward by *ratio* of the image dimensions
    so that page margins captured by the camera are not clipped.
    """
    pad_x = img_w * ratio
    pad_y = img_h * ratio
    cx    = float(np.mean(quad[:, 0]))
    cy    = float(np.mean(quad[:, 1]))
    padded = quad.copy()
    for i, (px, py) in enumerate(quad):
        dx = px - cx
        dy = py - cy
        length = max(1e-6, np.hypot(dx, dy))
        padded[i, 0] = px + (dx / length) * pad_x
        padded[i, 1] = py + (dy / length) * pad_y
    padded[:, 0] = np.clip(padded[:, 0], 0, img_w - 1)
    padded[:, 1] = np.clip(padded[:, 1], 0, img_h - 1)
    return padded


# ── Perspective warp (NATURAL dimensions) ─────────────────────────────────────

def _perspective_warp(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Warp *image* so *corners* become a top-down rectangle at the NATURAL
    width/height of the detected quad (not forced A4).
    This prevents content from being clipped by aspect-ratio distortion.
    """
    rect = _order_points(corners)
    tl, tr, br, bl = rect

    width  = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    if width < 20 or height < 20:
        logger.warning("_perspective_warp: degenerate quad – returning full frame")
        return image

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (width, height))


# ── Illumination correction ───────────────────────────────────────────────────

def _clahe_correction(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE on the L-channel of LAB to remove shadows / uneven lighting."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ── Enhancement ───────────────────────────────────────────────────────────────

def _enhance(image: np.ndarray) -> np.ndarray:
    """
    Adobe-style adaptive enhancement:
      • Dark images  → denoise + adaptive threshold → clean B&W scan
      • Bright images → mild sharpening (better readability for coloured text)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    mean_brightness = float(np.mean(gray))

    if mean_brightness < BRIGHTNESS_BW_THRESHOLD:
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        bw = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=ADAPTIVE_BLOCK_SIZE,
            C=ADAPTIVE_C,
        )
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    else:
        kernel_sharpen = np.array([[ 0, -1,  0],
                                   [-1,  5, -1],
                                   [ 0, -1,  0]], dtype=np.float32)
        return cv2.filter2D(image, -1, kernel_sharpen)


# ── Manual crop helper ────────────────────────────────────────────────────────

def _apply_manual_crop(
    frame: np.ndarray,
    x_pct: float,
    y_pct: float,
    w_pct: float,
    h_pct: float,
) -> np.ndarray:
    """
    Convert percentage-based crop coordinates to absolute pixels and crop.

    Parameters
    ----------
    frame   : BGR image
    x_pct   : left edge as % of image width  (0–100)
    y_pct   : top  edge as % of image height (0–100)
    w_pct   : crop width  as % of image width
    h_pct   : crop height as % of image height

    Returns the cropped region, or the full frame if the crop is degenerate.
    """
    img_h, img_w = frame.shape[:2]

    x1 = int(round(x_pct / 100.0 * img_w))
    y1 = int(round(y_pct / 100.0 * img_h))
    x2 = int(round((x_pct + w_pct) / 100.0 * img_w))
    y2 = int(round((y_pct + h_pct) / 100.0 * img_h))

    # Clamp to image boundaries
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))

    if x2 <= x1 or y2 <= y1:
        logger.warning(
            "_apply_manual_crop: degenerate crop box (%d,%d,%d,%d) – returning full frame",
            x1, y1, x2, y2,
        )
        return frame

    cropped = frame[y1:y2, x1:x2]
    logger.info(
        "Manual crop applied: pct=(%.1f,%.1f,%.1f,%.1f) → px=(%d,%d,%d,%d)",
        x_pct, y_pct, w_pct, h_pct, x1, y1, x2, y2,
    )
    return cropped


# ── Public API ────────────────────────────────────────────────────────────────

def scan_document(
    frame: np.ndarray,
    *,
    crop_x_pct: Optional[float] = None,
    crop_y_pct: Optional[float] = None,
    crop_w_pct: Optional[float] = None,
    crop_h_pct: Optional[float] = None,
    enhance: bool = True,
) -> np.ndarray:
    """
    Full Adobe-Scan-quality document scanning pipeline.

    Parameters
    ----------
    frame        : Raw BGR video frame.
    crop_x_pct   : Manual left-edge override as % of image width  (0–100).
    crop_y_pct   : Manual top-edge override  as % of image height (0–100).
    crop_w_pct   : Manual crop width  as % of image width.
    crop_h_pct   : Manual crop height as % of image height.
    enhance      : Whether to apply illumination correction + adaptive
                   thresholding. Defaults to True.

    Returns
    -------
    Processed BGR image.
    """

    # ── Priority 1: Manual Crop Override ──────────────────────────────────────
    manual_coords_provided = all(
        v is not None for v in (crop_x_pct, crop_y_pct, crop_w_pct, crop_h_pct)
    )
    if manual_coords_provided:
        logger.info("scan_document: manual crop override active")
        cropped = _apply_manual_crop(
            frame,
            float(crop_x_pct),
            float(crop_y_pct),
            float(crop_w_pct),
            float(crop_h_pct),
        )
        if enhance:
            corrected = _clahe_correction(cropped)
            return _enhance(corrected)
        return cropped

    # ── Priority 2: Tuned OpenCV Boundary Detection ───────────────────────────
    img_h, img_w = frame.shape[:2]
    frame_area   = img_h * img_w

    try:
        quad = _find_document_quad(frame)

        use_full_frame = False
        if quad is not None:
            hull_area = cv2.contourArea(cv2.convexHull(quad.astype(np.int32)))
            coverage  = hull_area / frame_area

            if coverage < 0.05:
                raise ValueError(
                    f"Detected quad area is only {coverage*100:.1f}% of frame – unreliable"
                )

            # If the detected quad is small relative to the frame, the page
            # probably IS most of the frame – skip the warp to avoid clipping.
            if coverage < FULL_FRAME_THRESHOLD:
                logger.info(
                    "scan_document: quad covers %.1f%% < %.0f%% threshold – using full frame",
                    coverage * 100, FULL_FRAME_THRESHOLD * 100,
                )
                use_full_frame = True
        else:
            use_full_frame = True

        if use_full_frame:
            warped = frame.copy()
        else:
            padded_quad = _pad_quad(quad, img_h, img_w)
            warped = _perspective_warp(frame, padded_quad)
            if warped is frame:
                raise ValueError("Perspective warp returned the original frame (degenerate quad)")

        if enhance:
            corrected = _clahe_correction(warped)
            return _enhance(corrected)
        return warped

    except Exception as exc:
        # ── Priority 3: Safety Fallback ───────────────────────────────────────
        logger.warning(
            "scan_document: boundary detection failed (%s) – returning original frame", exc
        )
        return frame
