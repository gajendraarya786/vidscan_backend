"""
services/frame_extractor.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One clean page per physical flip.

ALGORITHM (per sampled frame at 2 fps)
───────────────────────────────────────
  1. Compute motion MAD between current and previous RAW sample (256×256).
     → If MAD > FLIP_THRESHOLD (50): page is being flipped → discard.
       This threshold is deliberately HIGH so normal hand-shake never triggers it.
  2. Blur check on the document ROI.
  3. Cooldown: after keeping any page skip the next COOLDOWN_SAMPLES frames
     regardless of content, giving the user time to move to the next page.
  4. Dedup vs last KEPT page (ROI thumbnail MAD + histogram correlation):
     → If ROI looks the same as last kept → still on same page → skip and
       reset the cooldown counter so we don't keep re-checking the same page.
  5. If all checks pass → keep the frame → warp → correct → enhance.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Tunables ─────────────────────────────────────────────────────────────────

# Flip detection (vs previous RAW sample on 256×256 downscale)
FLIP_MAD_THRESHOLD: float   = 40.0
MIN_STABLE_FRAMES: int      = 1      # Need 1 stable frame before capture

# Dedup vs last KEPT page
THUMB_SAME_PAGE_MAD: float  = 12.0   # ROI thumb MAD below this → same page

# After each kept page, skip this many samples (~N × 0.5 s)
COOLDOWN_SAMPLES: int       = 3      # ≈ 1.5 s cooldown

# Document detection
BLUR_THRESHOLD: float       = 90.0   # Laplacian variance; rejects blurry/motion frames
MIN_DOC_AREA_RATIO: float   = 0.10
FULL_FRAME_THRESHOLD: float = 0.20
PADDING_RATIO: float        = 0.15
BRIGHTNESS_BW_THRESHOLD: float = 160.0

THUMB_SIZE:  tuple = (128, 128)
MOTION_SIZE: tuple = (256, 256)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1); diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)];    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(s)];    rect[3] = pts[np.argmax(diff)]
    return rect


def _pad_quad(q: np.ndarray, h: int, w: int) -> np.ndarray:
    px = w * PADDING_RATIO; py = h * PADDING_RATIO
    cx = float(np.mean(q[:, 0])); cy = float(np.mean(q[:, 1]))
    p  = q.copy()
    for i, (x, y) in enumerate(q):
        dx = x - cx; dy = y - cy; ln = max(1e-6, np.hypot(dx, dy))
        p[i, 0] = x + dx / ln * px; p[i, 1] = y + dy / ln * py
    p[:, 0] = np.clip(p[:, 0], 0, w - 1)
    p[:, 1] = np.clip(p[:, 1], 0, h - 1)
    return p


def _warp(img: np.ndarray, q: np.ndarray) -> np.ndarray:
    rect = _order_points(q)
    tl, tr, br, bl = rect
    W = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    H = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if W < 20 or H < 20:
        return img
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(rect, dst), (W, H))


# ── Image processing ──────────────────────────────────────────────────────────

def _clahe(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def _enhance(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.mean(g) < BRIGHTNESS_BW_THRESHOLD:
        d  = cv2.GaussianBlur(g, (3, 3), 0)
        bw = cv2.adaptiveThreshold(d, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 10)
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], np.float32)
    return cv2.filter2D(img, -1, k)


# ── Document detection ────────────────────────────────────────────────────────

def _find_quad(image_gray: np.ndarray, min_area: float):
    h, w = image_gray.shape[:2]

    # 1. Downscale for speed and structural consistency
    scale = 500.0 / max(h, w)
    small = cv2.resize(image_gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    sh, sw = small.shape[:2]

    blurred = cv2.GaussianBlur(small, (9, 9), 0)
    
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


# ── Similarity metrics ────────────────────────────────────────────────────────

def _motion_mad(a: np.ndarray, b: np.ndarray) -> float:
    """Frame-to-frame MAD on 256×256 downscale (flip detection)."""
    # a and b are already MOTION_SIZE (256, 256)
    diff = cv2.absdiff(a, b)
    return float(cv2.mean(diff)[0])


def _thumb_mad(a: np.ndarray, b: np.ndarray) -> float:
    """Thumbnail MAD for same-page dedup."""
    ta = cv2.resize(a, THUMB_SIZE, interpolation=cv2.INTER_NEAREST)
    tb = cv2.resize(b, THUMB_SIZE, interpolation=cv2.INTER_NEAREST)
    diff = cv2.absdiff(ta, tb)
    return float(cv2.mean(diff)[0])


def _hist_correl(a: np.ndarray, b: np.ndarray) -> float:
    ha = cv2.calcHist([a], [0], None, [256], [0, 256])
    hb = cv2.calcHist([b], [0], None, [256], [0, 256])
    cv2.normalize(ha, ha, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hb, hb, 0, 1, cv2.NORM_MINMAX)
    return float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))


# ── Public API ────────────────────────────────────────────────────────────────

def extract_frames(video_path: str) -> list[np.ndarray]:
    """Return one scanned image per unique document page in *video_path*."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path!r}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError("Cannot determine FPS.")

    skip = max(1, int(round(fps)))    # sample at 1 fps for speed
    logger.info("fps=%.2f  sample_every=%d  (1fps)", fps, skip)

    pages:          list[np.ndarray] = []
    prev_small:     np.ndarray | None = None   # 256×256 gray of previous sample
    last_roi_gray:  np.ndarray | None = None   # ROI gray of last KEPT page
    cooldown:       int               = 0
    stable_count:   int               = 0

    n_sampled = n_flip = n_blur = n_cool = n_dup = 0
    frame_idx = 0

    while True:
        if frame_idx % skip != 0:
            ret = cap.grab()
            if not ret:
                break
            frame_idx += 1
            continue

        ret = cap.grab()
        if not ret:
            break
        ret, frame = cap.retrieve()
        if not ret or frame is None:
            break

        n_sampled += 1
        h, w = frame.shape[:2]
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, MOTION_SIZE)

        # ── 1. Flip/Motion detection ──────────────────────────────────────────
        flip_mad = _motion_mad(small, prev_small) if prev_small is not None else 0.0
        prev_small = small

        if flip_mad > FLIP_MAD_THRESHOLD:
            # Large inter-sample change → page is in motion or hand is moving → discard
            cooldown = 0
            stable_count = 0
            n_flip  += 1
            logger.info("Motion detected (MAD=%.1f > %.1f) – resetting stability", flip_mad, FLIP_MAD_THRESHOLD)
            frame_idx += 1
            continue

        # ── 2. Blur check ─────────────────────────────────────────────────────
        if cv2.Laplacian(gray, cv2.CV_16S).var() < BLUR_THRESHOLD:
            n_blur += 1
            stable_count = 0  # Blurry frame is not stable document representation
            frame_idx += 1
            continue

        # ── 3. Cooldown ───────────────────────────────────────────────────────
        if cooldown > 0:
            cooldown -= 1
            stable_count = 0  # Reset stability during cooldown period
            n_cool   += 1
            frame_idx += 1
            continue

        # If it passes motion, blur, and cooldown checks, it counts as a stable frame
        stable_count += 1
        logger.info("Stable frame (MAD=%.1f) – stability count = %d", flip_mad, stable_count)

        if stable_count < MIN_STABLE_FRAMES:
            frame_idx += 1
            continue

        # ── 4. Document boundary detection ───────────────────────────────────
        quad     = _find_quad(gray, min_area=h * w * MIN_DOC_AREA_RATIO)
        use_full = True
        if quad is not None:
            qa = cv2.contourArea(cv2.convexHull(quad.astype(np.int32)))
            if qa / (h * w) >= FULL_FRAME_THRESHOLD:
                use_full = False

        # Build ROI gray for dedup comparison
        if use_full:
            roi_gray = gray
        else:
            xs = quad[:, 0]; ys = quad[:, 1]
            x0, x1 = max(0, int(np.min(xs))), min(w, int(np.max(xs)))
            y0, y1 = max(0, int(np.min(ys))), min(h, int(np.max(ys)))
            roi     = frame[y0:y1, x0:x1]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.size > 0 else gray

        # ── 5. Same-page dedup vs last KEPT frame ─────────────────────────────
        if last_roi_gray is not None:
            mad  = _thumb_mad(roi_gray, last_roi_gray)
            corr = _hist_correl(roi_gray, last_roi_gray)
            logger.info("dedup  flip_mad=%.1f  mad=%.1f  corr=%.3f", flip_mad, mad, corr)

            if mad < THUMB_SAME_PAGE_MAD:
                # Still on the same page – start a short cooldown and move on
                cooldown  = COOLDOWN_SAMPLES
                stable_count = 0
                n_dup    += 1
                frame_idx += 1
                continue

        # ── 6. Keep the raw frame ─────────────────────────────────────────────
        pages.append(frame.copy())
        last_roi_gray = roi_gray
        cooldown      = COOLDOWN_SAMPLES
        stable_count  = 0

        logger.info(
            "KEPT page #%d  flip=%.1f  %s",
            len(pages), flip_mad,
            "full" if use_full else "detected-document",
        )

        frame_idx += 1

    cap.release()
    logger.info(
        "Done. sampled=%d  flip=%d  blur=%d  cooldown=%d  dup=%d  kept=%d",
        n_sampled, n_flip, n_blur, n_cool, n_dup, len(pages),
    )
    return pages
