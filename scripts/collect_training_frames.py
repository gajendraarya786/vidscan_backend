"""
scripts/collect_training_frames.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Collect labeled training frames from your own videos for YOLOv8 training.

Usage
-----
  python scripts/collect_training_frames.py --video path/to/video.mp4 --output frames/
  python scripts/collect_training_frames.py --camera 0 --output frames/  # live webcam

What it does
------------
  1. Opens a video file (or camera).
  2. Samples every N frames.
  3. Shows each frame in a window so you can:
       SPACE or ENTER → save this frame
       s              → save this frame (alias)
       d or DELETE    → skip / discard
       q              → quit
  4. Saves accepted frames as JPEG to the --output folder.
  5. Prints a summary at the end.

After collecting frames
-----------------------
  Upload the output folder to Roboflow:
  https://roboflow.com → New Project → Object Detection → Upload Images

  In Roboflow, draw bounding boxes around the document in each image.
  Label: "document"

  Then export with 5x augmentation and train (see train_yolov8_colab.py).

Condition checklist (aim for 200 total)
---------------------------------------
  [ ] Good lighting, flat camera           40 frames
  [ ] Bright light / overexposed           20 frames
  [ ] Dim light / underexposed             20 frames
  [ ] Side lamp (shadows on page)          20 frames
  [ ] Hand partially visible               25 frames
  [ ] Tilted camera angle                  20 frames
  [ ] Close-up shot                        15 frames
  [ ] Far away shot                        15 frames
  [ ] Dark desk surface                    15 frames
  [ ] White/bright desk surface            10 frames
  TOTAL                                   200 frames
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Collect training frames for YOLOv8 document detection"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video",  help="Path to video file(s)", nargs="+")
    src.add_argument("--camera", type=int, default=None, help="Camera index (0 = default webcam)")

    p.add_argument(
        "--output", "-o",
        default="training_frames",
        help="Output directory for saved frames (default: training_frames/)",
    )
    p.add_argument(
        "--sample-every", type=int, default=15,
        help="Sample every N frames from video file (default: 15 = ~0.5s at 30fps)",
    )
    p.add_argument(
        "--auto", action="store_true",
        help="Auto-save non-blurry frames without prompting (for bulk collection). "
             "Review and delete bad ones afterward.",
    )
    p.add_argument(
        "--blur-threshold", type=float, default=100.0,
        help="Laplacian blur threshold; frames below this are auto-skipped (default: 100)",
    )
    p.add_argument(
        "--skip-blurry", action="store_true", default=True,
        help="In interactive mode, auto-skip blurry frames without showing them (default: True)",
    )
    p.add_argument(
        "--show-blurry", action="store_true",
        help="Override --skip-blurry: show blurry frames so you can decide manually",
    )
    p.add_argument(
        "--prefix", default="frame",
        help="Filename prefix for saved frames (default: frame)",
    )
    p.add_argument(
        "--resize", type=int, default=None,
        help="Resize saved frames to this max dimension (default: original size)",
    )
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_blurry(frame, threshold: float) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def save_frame(frame, out_dir: Path, prefix: str, idx: int, resize_max: int | None) -> str:
    if resize_max:
        h, w = frame.shape[:2]
        longest = max(h, w)
        if longest > resize_max:
            scale = resize_max / longest
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    filename = out_dir / f"{prefix}_{idx:05d}.jpg"
    cv2.imwrite(str(filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return str(filename)


def draw_overlay(frame, count: int, frame_num: int, total_frames: int, total_target: int = 200):
    """Draw HUD overlay on preview window."""
    h, w = frame.shape[:2]
    display = frame.copy()

    # Scale font based on resolution so text is always readable
    font_scale = max(0.6, w / 1200)
    thickness  = max(1, int(font_scale * 2))

    # Dark bar at top (taller for high-res frames)
    bar_h = max(70, int(h * 0.06))
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

    # Progress bar (green fill)
    progress = min(count / total_target, 1.0)
    bar_w = int(w * progress)
    cv2.rectangle(display, (0, bar_h - 8), (bar_w, bar_h), (0, 200, 100), -1)

    # Text
    cv2.putText(display, f"Saved: {count}/{total_target}",
                (10, int(bar_h * 0.5)), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness)
    cv2.putText(display, f"Frame {frame_num}/{total_frames}   SPACE=save  D=skip  Q=quit",
                (10, bar_h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.55, (200, 200, 200), 1)
    return display


# ── Main collection loop ──────────────────────────────────────────────────────

def collect_from_source(source, out_dir: Path, args, saved_count_start: int = 0) -> int:
    """Run the collection loop on a single video source. Returns number of frames saved."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open source: {source}")
        return 0

    fps           = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_camera     = isinstance(source, int)
    skip_blurry   = args.skip_blurry and not args.show_blurry

    # Number of candidate frames the user will actually see
    n_candidates  = total_frames // args.sample_every if not is_camera else 0

    print(f"\n{'Camera' if is_camera else 'Video'}: {source}")
    if not is_camera:
        print(f"  Resolution  : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"  FPS         : {fps:.1f}")
        print(f"  Duration    : {total_frames/fps:.1f}s")
        print(f"  Sample every: every {args.sample_every} frames  "
              f"→ ~{n_candidates} candidate frames")
        print(f"  Blur filter : {'ON (blurry frames auto-skipped)' if skip_blurry else 'OFF'}")
    print()
    print("  Controls:")
    print("    SPACE / ENTER / S  →  ✅ Save this frame")
    print("    D / DELETE         →  ❌ Skip this frame")
    print("    Q                  →  🏁 Done — quit")
    print()

    frame_idx  = 0
    shown_idx  = 0       # count of frames shown to user (for display)
    n_skipped_blur = 0
    saved      = saved_count_start

    # Resize display window to fit screen (iPhone videos are portrait 1080x1920)
    DISPLAY_MAX = 800    # max height of preview window in pixels
    cv2.namedWindow("VidScan Frame Collector", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample at desired rate for video files
        if not is_camera and frame_idx % args.sample_every != 0:
            frame_idx += 1
            continue

        shown_idx += 1

        # Blur check
        blurry = is_blurry(frame, args.blur_threshold)

        if args.auto:
            # ── Auto mode: save non-blurry, skip blurry ──────────────────────
            if not blurry:
                path = save_frame(frame, out_dir, args.prefix, saved + 1, args.resize)
                saved += 1
                print(f"  AUTO saved [{saved:3d}] {Path(path).name}")
            else:
                n_skipped_blur += 1
        else:
            # ── Interactive mode ──────────────────────────────────────────────
            # In interactive mode, silently skip blurry frames (saves time)
            if skip_blurry and blurry:
                n_skipped_blur += 1
                frame_idx += 1
                continue

            # Rotate portrait iPhone videos for easier viewing
            h, w = frame.shape[:2]
            if h > w:   # portrait — rotate 90° so it fits on screen landscape
                display_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            else:
                display_frame = frame

            dh, dw = display_frame.shape[:2]
            # Resize window to fit screen
            if dh > DISPLAY_MAX:
                scale     = DISPLAY_MAX / dh
                disp_w    = int(dw * scale)
                disp_h    = DISPLAY_MAX
                cv2.resizeWindow("VidScan Frame Collector", disp_w, disp_h)
            else:
                cv2.resizeWindow("VidScan Frame Collector", dw, dh)

            display = draw_overlay(
                display_frame, saved, shown_idx, n_candidates or total_frames
            )

            cv2.imshow("VidScan Frame Collector", display)

            # ── KEY: waitKey(0) = pause FOREVER until a key is pressed ────────
            key = cv2.waitKey(0) & 0xFF

            if key in (ord(" "), ord("\r"), ord("\n"), ord("s"), 13):
                path = save_frame(frame, out_dir, args.prefix, saved + 1, args.resize)
                saved += 1
                print(f"  SAVED [{saved:3d}] {Path(path).name}")

            elif key in (ord("d"), 127, 8):  # d, delete, backspace
                pass  # silently skip

            elif key == ord("q"):
                print("  Quit.")
                break

        frame_idx += 1

    cap.release()
    if n_skipped_blur > 0:
        print(f"  (auto-skipped {n_skipped_blur} blurry frames)")
    return saved


def _count_existing(out_dir: Path, prefix: str) -> int:
    """Count frames already saved with this prefix so we don't overwrite them."""
    existing = list(out_dir.glob(f"{prefix}_*.jpg"))
    return len(existing)


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" VidScan Training Frame Collector")
    print("=" * 60)
    print(f"Output directory: {out_dir.absolute()}")
    print(f"Mode: {'AUTO' if args.auto else 'INTERACTIVE'}")
    if args.auto:
        print(f"Blur threshold: {args.blur_threshold}")

    # ── Auto-detect existing frames so we continue numbering, not overwrite ──
    existing = _count_existing(out_dir, args.prefix)
    if existing > 0:
        print(f"\n  ⚠️  Found {existing} existing '{args.prefix}_*.jpg' frames.")
        print(f"     New frames will be numbered from {args.prefix}_{existing+1:05d}.jpg")
        print(f"     (existing frames are safe — not overwritten)")
    print()

    # Start counter from existing frame count so files don't collide
    total_saved = existing

    if args.camera is not None:
        total_saved = collect_from_source(args.camera, out_dir, args, existing)
    else:
        for video_path in args.video:
            if not Path(video_path).exists():
                print(f"ERROR: Video not found: {video_path}")
                continue
            total_saved = collect_from_source(video_path, out_dir, args, total_saved)

    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print(f" DONE  —  {total_saved} frames saved to: {out_dir.absolute()}")
    print("=" * 60)
    print()
    print("NEXT STEPS:")
    print("  1. Upload frames to Roboflow: https://roboflow.com")
    print("  2. Label each image: draw box around document, label = 'document'")
    print("  3. Apply 5x augmentation (flip, rotation, brightness, blur, noise)")
    print("  4. Export dataset (YOLOv8 format)")
    print("  5. Run training: scripts/train_yolov8_colab.py on Google Colab")
    print()
    print("Target breakdown for 200 frames:")
    conditions = [
        ("Good lighting, flat camera",     40),
        ("Bright light / overexposed",     20),
        ("Dim light / underexposed",       20),
        ("Side lamp (shadows on page)",    20),
        ("Hand partially visible",         25),
        ("Tilted camera angle",            20),
        ("Close-up shot",                  15),
        ("Far away shot",                  15),
        ("Dark desk surface",              15),
        ("White/bright desk surface",      10),
    ]
    for cond, n in conditions:
        print(f"  {n:3d}  {cond}")
    print(f"  {'─'*36}")
    print(f"  200  TOTAL")


if __name__ == "__main__":
    main()
