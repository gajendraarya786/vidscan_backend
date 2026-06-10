#!/bin/bash
# scripts/run_collector.sh
# ─────────────────────────────────────────────────────────────────────────────
# Guided frame collection for YOLOv8 training.
# Run: bash scripts/run_collector.sh
#
# The script will:
#   1. Tell you exactly how to record each video on your phone
#   2. Wait while you record it
#   3. Ask you to drag-and-drop (or type) the video path
#   4. Open the review window — press SPACE to save, D to skip, Q to finish
#   5. Move to the next condition
#
# Controls in the preview window:
#   SPACE / ENTER / S  →  ✅ save frame
#   D / DELETE         →  ❌ skip frame
#   Q                  →  done with this video, move to next
# ─────────────────────────────────────────────────────────────────────────────

VIDEO_DIR="$HOME/Desktop/vidscan dataset"
VENV=".venv/bin/python"
SCRIPT="scripts/collect_training_frames.py"
OUTPUT="training_frames"

mkdir -p "$OUTPUT"

# ── Colour helpers ────────────────────────────────────────────────────────────
BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
CYAN="\033[0;36m"
RED="\033[0;31m"
RESET="\033[0m"

header() {
  echo ""
  echo -e "${BOLD}============================================================${RESET}"
  echo -e "${BOLD} $1${RESET}"
  echo -e "${BOLD}============================================================${RESET}"
}

step() {
  echo ""
  echo -e "${CYAN}▶ $1${RESET}"
}

tip() {
  echo -e "${YELLOW}  💡 $1${RESET}"
}

ok() {
  echo -e "${GREEN}  ✅ $1${RESET}"
}

saved_count() {
  ls "$OUTPUT"/*.jpg 2>/dev/null | wc -l | tr -d ' '
}

run_batch() {
  local BATCH_NUM="$1"
  local CONDITION="$2"
  local TARGET="$3"
  local PREFIX="$4"
  local SAMPLE_EVERY="$5"
  shift 5
  local TIPS=("$@")   # remaining args are tips

  header "[${BATCH_NUM}/10] ${CONDITION} — target: ${TARGET} frames"

  echo ""
  echo -e "${BOLD}HOW TO RECORD THIS VIDEO:${RESET}"
  for t in "${TIPS[@]}"; do
    tip "$t"
  done
  echo ""
  echo "  Duration: 30-60 seconds is enough"
  echo "  Keep the document stable in frame (don't flip pages in this clip)"
  echo ""
  echo -e "${YELLOW}→ Record the video on your phone NOW, then AirDrop it to your Mac.${RESET}"
  echo ""
  echo -n "  Once the video is on your Mac, drag it into this terminal window"
  echo    " (or type the full path):"
  echo -n "  Video path: "
  read -r VIDEO_PATH

  # Strip surrounding quotes that Finder drag-drop adds
  VIDEO_PATH="${VIDEO_PATH%\'}"
  VIDEO_PATH="${VIDEO_PATH#\'}"
  VIDEO_PATH="${VIDEO_PATH%\"}"
  VIDEO_PATH="${VIDEO_PATH#\"}"
  # Expand ~ if present
  VIDEO_PATH="${VIDEO_PATH/#\~/$HOME}"

  if [[ ! -f "$VIDEO_PATH" ]]; then
    echo -e "${RED}  ❌ File not found: $VIDEO_PATH${RESET}"
    echo "  Skipping this batch. You can re-run the script to redo skipped batches."
    return
  fi

  BEFORE=$(saved_count)

  echo ""
  echo -e "${BOLD}  Opening preview window...${RESET}"
  echo "  Press SPACE/ENTER to save  |  D to skip  |  Q when done"
  echo ""

  $VENV $SCRIPT \
    --video "$VIDEO_PATH" \
    --output "$OUTPUT" \
    --sample-every "$SAMPLE_EVERY" \
    --prefix "$PREFIX"

  AFTER=$(saved_count)
  SAVED_THIS=$((AFTER - BEFORE))
  TOTAL_NOW=$AFTER

  echo ""
  if [[ $SAVED_THIS -ge $TARGET ]]; then
    ok "Saved ${SAVED_THIS} frames for '${CONDITION}' (target was ${TARGET}) ✓"
  else
    echo -e "${YELLOW}  ⚠️  Saved ${SAVED_THIS}/${TARGET} frames for '${CONDITION}'.${RESET}"
    echo "     You can re-run this batch later if needed."
  fi
  echo -e "${CYAN}  Running total: ${TOTAL_NOW}/200 frames${RESET}"
}

# ── Welcome ───────────────────────────────────────────────────────────────────

clear
header "VidScan — Guided Frame Collection"
echo ""
echo "  This script guides you through recording and collecting"
echo "  200 training frames for your YOLOv8 document detector."
echo ""
echo "  You will be recording 10 short videos (~30-60 sec each)"
echo "  on your phone — one per lighting/condition type."
echo ""
echo -e "${BOLD}  What you need:${RESET}"
echo "    • Your phone (for recording)"
echo "    • AirDrop enabled on both phone and Mac"
echo "    • A notebook/document to scan"
echo "    • ~2 hours total time"
echo ""
echo -e "${BOLD}  Keyboard controls in the preview window:${RESET}"
echo -e "    ${GREEN}SPACE / ENTER / S${RESET}  →  ✅ Save this frame"
echo -e "    ${RED}D / DELETE${RESET}         →  ❌ Skip this frame"
echo -e "    ${YELLOW}Q${RESET}                  →  Done with this video"
echo ""
echo "  Output folder: $(pwd)/$OUTPUT"
echo ""
echo "  Press ENTER to start the first batch..."
read -r

# ─────────────────────────────────────────────────────────────────────────────
# BATCH 1: GOOD LIGHTING
# ─────────────────────────────────────────────────────────────────────────────
run_batch 1 "GOOD LIGHTING — FLAT CAMERA" 40 "good" 15 \
  "Hold phone directly above notebook, pointing straight down" \
  "Normal room ceiling light or desk lamp pointing at the page" \
  "Page should fill 60-80% of the frame" \
  "Keep phone steady — don't move it, just let it sit for 30 sec" \
  "Record 3-4 different pages of the notebook" \
  "Try with lined paper, blank paper, and printed text"

# ─────────────────────────────────────────────────────────────────────────────
# BATCH 2: BRIGHT / OVEREXPOSED
# ─────────────────────────────────────────────────────────────────────────────
run_batch 2 "BRIGHT / OVEREXPOSED" 20 "bright" 15 \
  "Sit near a sunny window and hold page toward the light" \
  "OR point a bright lamp directly at the page from above" \
  "The page should look very white/washed out on your phone screen" \
  "Phone camera should struggle with the brightness" \
  "Some frames can have slight glare on the page — that's good!"

# ─────────────────────────────────────────────────────────────────────────────
# BATCH 3: DIM / UNDEREXPOSED
# ─────────────────────────────────────────────────────────────────────────────
run_batch 3 "DIM / UNDEREXPOSED" 20 "dim" 15 \
  "Record in the evening or in a dark room" \
  "Turn off the main room light, use only a small lamp far away" \
  "The page should look yellowish/dark on your phone screen" \
  "You can still read the text but it looks low-contrast" \
  "Don't use your phone torch — that's too bright"

# ─────────────────────────────────────────────────────────────────────────────
# BATCH 4: SIDE LAMP / SHADOWS
# ─────────────────────────────────────────────────────────────────────────────
run_batch 4 "SIDE LAMP — DIAGONAL SHADOW" 20 "shadow" 15 \
  "Place a lamp to the LEFT or RIGHT of the notebook (not above)" \
  "The shadow should fall diagonally across the page" \
  "One half of page is bright, the other half is in shadow" \
  "This is the hardest condition for OpenCV — very important to cover" \
  "Move the notebook slightly during recording to get shadow at different angles"

# ─────────────────────────────────────────────────────────────────────────────
# BATCH 5: HAND IN FRAME
# ─────────────────────────────────────────────────────────────────────────────
run_batch 5 "HAND PARTIALLY IN FRAME" 25 "hand" 15 \
  "Use one hand to hold down a curling page corner while recording" \
  "Have a finger or thumb visible at the edge of the frame" \
  "Try: one finger at bottom-left corner, covering ~10% of page" \
  "Try: holding the notebook spine open with thumb visible" \
  "Try: index finger pressing middle of page flat" \
  "The document should still be clearly visible with partial hand"

# ─────────────────────────────────────────────────────────────────────────────
# BATCH 6: TILTED ANGLE
# ─────────────────────────────────────────────────────────────────────────────
run_batch 6 "TILTED CAMERA ANGLE" 20 "tilt" 15 \
  "Hold phone at 30-45 degrees angle (not directly above)" \
  "The page should show perspective distortion (trapezoid shape)" \
  "Try from front-left, front-right, and front-center angles" \
  "The far edge of the page should be narrower than the near edge" \
  "This tests perspective correction — very important for your use case"

# ─────────────────────────────────────────────────────────────────────────────
# BATCH 7: CLOSE-UP
# ─────────────────────────────────────────────────────────────────────────────
run_batch 7 "CLOSE-UP SHOT" 15 10 \
  "Hold phone very close to the page — 15-20 cm away" \
  "The page corners should be slightly cut off or just at the edge" \
  "Text should be large and very readable on screen" \
  "Good for capturing fine text detail" \
  "Try a few pages"

# ─────────────────────────────────────────────────────────────────────────────
# BATCH 8: FAR AWAY
# ─────────────────────────────────────────────────────────────────────────────
run_batch 8 "FAR AWAY SHOT" 15 10 \
  "Hold phone 50-80 cm above the notebook" \
  "The notebook should be small in the frame, surrounded by lots of desk" \
  "You should clearly see the desk surface around the notebook" \
  "The text on the page will be tiny but the page outline is clear" \
  "Try different zoom levels if your phone has them"

# ─────────────────────────────────────────────────────────────────────────────
# BATCH 9: DARK DESK SURFACE
# ─────────────────────────────────────────────────────────────────────────────
run_batch 9 "DARK DESK SURFACE" 15 10 \
  "Put the notebook on the darkest surface you can find" \
  "Dark wood, black desk mat, dark tablecloth all work great" \
  "Use a white or light-coloured notebook for best contrast" \
  "This tests dark background separation — OpenCV struggles here" \
  "Good room lighting for this batch (contrast comes from desk, not light)"

# ─────────────────────────────────────────────────────────────────────────────
# BATCH 10: WHITE / BRIGHT DESK
# ─────────────────────────────────────────────────────────────────────────────
run_batch 10 "WHITE / BRIGHT DESK SURFACE" 10 "white" 10 \
  "Put the notebook on a white desk, white paper, or white tablecloth" \
  "Use a dark-coloured notebook if you have one (better contrast)" \
  "This is the hardest: low contrast between page and surface" \
  "Try with both white notebook AND dark notebook on white desk" \
  "Good lighting — the challenge here is desk vs page colour, not light"

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

header "COLLECTION COMPLETE 🎉"

TOTAL=$(saved_count)
echo ""
echo -e "  ${BOLD}Total frames saved: ${GREEN}${TOTAL}${RESET}${BOLD} / 200 target${RESET}"
echo "  Location: $(pwd)/$OUTPUT"
echo ""
echo "  Breakdown by condition:"
declare -A TARGETS=( [good]=40 [bright]=20 [dim]=20 [shadow]=20 [hand]=25 [tilt]=20 [close]=15 [far]=15 [dark]=15 [white]=10 )
for prefix in good bright dim shadow hand tilt close far dark white; do
  count=$(ls "$OUTPUT/${prefix}_"*.jpg 2>/dev/null | wc -l | tr -d ' ')
  target=${TARGETS[$prefix]}
  if [[ $count -ge $target ]]; then
    status="${GREEN}✅${RESET}"
  elif [[ $count -gt 0 ]]; then
    status="${YELLOW}⚠️ ${RESET}"
  else
    status="${RED}❌${RESET}"
  fi
  printf "  %b  %-10s %3d / %d frames\n" "$status" "$prefix" "$count" "$target"
done

echo ""
if [[ $TOTAL -ge 150 ]]; then
  echo -e "${GREEN}  ✅ Enough frames collected! Ready for Roboflow labeling.${RESET}"
elif [[ $TOTAL -ge 100 ]]; then
  echo -e "${YELLOW}  ⚠️  Good start. Consider collecting more for the missing conditions.${RESET}"
else
  echo -e "${RED}  ❌ Too few frames. Please re-run for the missing conditions.${RESET}"
fi

echo ""
echo -e "${BOLD}  NEXT STEP: Upload to Roboflow${RESET}"
echo "  1. Go to https://roboflow.com → New Project → Object Detection"
echo "  2. Project name: vidscan-document"
echo "  3. Upload the entire  $(pwd)/$OUTPUT  folder"
echo "  4. Label each image: draw box around document → label = 'document'"
echo "  5. Generate dataset with 5x augmentation"
echo "  6. Export in YOLOv8 format"
echo ""
echo "  Open the frames folder now:"
echo "  open $(pwd)/$OUTPUT"
echo ""
