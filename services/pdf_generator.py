"""
services/pdf_generator.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Converts a list of OpenCV BGR frames to a single PDF byte string using
img2pdf.  Each frame is JPEG-encoded at high quality before being passed to
img2pdf, which embeds them losslessly without re-compressing.
"""

import logging

import cv2
import img2pdf
import numpy as np

logger = logging.getLogger(__name__)

JPEG_QUALITY: int = 95  # 1-100; higher = larger file but better fidelity


def frames_to_pdf(frames: list[np.ndarray]) -> bytes:
    """
    Encode *frames* as JPEG and combine them into a single PDF.

    Parameters
    ----------
    frames : list[np.ndarray]
        BGR images (as returned by ``scan_document``).

    Returns
    -------
    bytes
        Raw PDF data, ready to stream to the client.

    Raises
    ------
    ValueError
        If *frames* is empty or every frame fails to encode.
    """
    if not frames:
        raise ValueError("No frames provided for PDF generation.")

    jpeg_pages: list[bytes] = []
    for idx, frame in enumerate(frames):
        success, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        if not success:
            logger.warning("Frame %d failed to JPEG-encode – skipping", idx)
            continue
        jpeg_pages.append(buf.tobytes())

    if not jpeg_pages:
        raise ValueError("Every frame failed to encode – cannot build PDF.")

    logger.info("Building PDF from %d pages", len(jpeg_pages))
    return img2pdf.convert(jpeg_pages)
