"""
scripts/train_yolov8_colab.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Complete YOLOv8 training script for document detection.
Run this on Google Colab (free T4 GPU) — NOT locally.

HOW TO USE
──────────
1. Open Google Colab: https://colab.research.google.com
2. New notebook → Runtime → Change runtime type → T4 GPU
3. Upload this file or paste the contents into a Code cell
4. Fill in your Roboflow API key and dataset version below
5. Run all cells
6. Download the trained model: runs/detect/document_detector/weights/best.pt
7. Rename it to document.pt and drop it in your backend's models/ directory

EXPECTED TIMELINE (Free T4 GPU)
────────────────────────────────
  200 images × 100 epochs ≈ 45-60 minutes
  After training, mAP@0.5 should be 88-93%

EXPECTED ACCURACY (with 200 real + 5x augmentation)
─────────────────────────────────────────────────────
  Good lighting:     95%+   ✅
  Dim lighting:      88-92% ✅
  Side shadows:      85-90% ✅
  Hand in frame:     82-88% ✅
  Extreme angles:    78-85% ⚠
  Very dark:         70-78% ⚠
  Overall mAP:       88-93% → production ready
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1: Install dependencies
# ═══════════════════════════════════════════════════════════════════════════════
# Run this cell first

install_commands = """
!pip install ultralytics==8.2.18 roboflow --quiet
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2: Configuration — EDIT THESE VALUES
# ═══════════════════════════════════════════════════════════════════════════════

# ┌──────────────────────────────────────────────────────────────────┐
# │  FILL IN YOUR ROBOFLOW DETAILS                                   │
# │  Get your API key from: https://app.roboflow.com/settings/api    │
# └──────────────────────────────────────────────────────────────────┘
ROBOFLOW_API_KEY    = "YOUR_API_KEY_HERE"   # ← paste your key
ROBOFLOW_WORKSPACE  = "YOUR_WORKSPACE"      # ← your workspace slug
ROBOFLOW_PROJECT    = "vidscan-document"    # ← your project name
ROBOFLOW_VERSION    = 1                     # ← dataset version number

# Training settings (optimised for 200 images → 90%+ accuracy)
EPOCHS      = 100
IMG_SIZE    = 640
BATCH_SIZE  = 16        # reduce to 8 if you get OOM on Colab free tier
PATIENCE    = 20        # early stopping: stop if no improvement for 20 epochs
PROJECT_NAME = "document_detector"

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3: Download dataset from Roboflow
# ═══════════════════════════════════════════════════════════════════════════════

colab_cell_3 = '''
from roboflow import Roboflow
import os

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
dataset = project.version(ROBOFLOW_VERSION).download("yolov8")

dataset_yaml = os.path.join(dataset.location, "data.yaml")
print(f"Dataset downloaded to: {dataset.location}")
print(f"YAML config: {dataset_yaml}")
'''

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4: Train YOLOv8
# ═══════════════════════════════════════════════════════════════════════════════

colab_cell_4 = '''
from ultralytics import YOLO
import torch

print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Load YOLOv8 nano — best accuracy/speed trade-off for document detection
# yolov8n.pt = 3.2M params, fast, good accuracy for single-class detection
model = YOLO("yolov8n.pt")

results = model.train(
    data=dataset_yaml,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    patience=PATIENCE,
    project="runs/detect",
    name=PROJECT_NAME,
    exist_ok=True,

    # ── Augmentation (matches your Roboflow 5x settings) ──
    augment=True,
    mosaic=1.0,        # YOLOv8 mosaic: combines 4 images — great for varied backgrounds
    mixup=0.1,         # Mixup augmentation: blends 2 images
    degrees=20.0,      # Random rotation ±20°
    translate=0.1,     # Random translation ±10%
    scale=0.5,         # Random scale ±50%
    fliplr=0.5,        # Horizontal flip 50% chance
    flipud=0.0,        # No vertical flip (documents are right-side up)
    hsv_h=0.015,       # Hue jitter
    hsv_s=0.7,         # Saturation jitter (handles lighting variation)
    hsv_v=0.4,         # Value/brightness jitter (handles dim/bright scenes)
    erasing=0.4,       # Random erasing (simulates hand occluding document)
    copy_paste=0.0,

    # ── Optimizer ──
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,

    # ── Validation ──
    val=True,
    plots=True,

    # ── Output ──
    save=True,
    save_period=10,    # save checkpoint every 10 epochs
    verbose=True,
)

print("\\n✅ Training complete!")
print(f"Best model saved at: runs/detect/{PROJECT_NAME}/weights/best.pt")
'''

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5: Evaluate the model
# ═══════════════════════════════════════════════════════════════════════════════

colab_cell_5 = '''
from ultralytics import YOLO

# Load the best trained model
best_model = YOLO(f"runs/detect/{PROJECT_NAME}/weights/best.pt")

# Run validation on the test set
metrics = best_model.val(
    data=dataset_yaml,
    split="test",
    conf=0.45,
    iou=0.45,
    plots=True,
    verbose=True,
)

print("\\n📊 Test Set Metrics:")
print(f"  mAP@0.5:       {metrics.box.map50:.3f}  (target: ≥ 0.88)")
print(f"  mAP@0.5:0.95:  {metrics.box.map:.3f}")
print(f"  Precision:     {metrics.box.mp:.3f}")
print(f"  Recall:        {metrics.box.mr:.3f}")

if metrics.box.map50 >= 0.88:
    print("\\n✅ Model is production ready! mAP@0.5 ≥ 88%")
elif metrics.box.map50 >= 0.80:
    print("\\n⚠️  Model is decent (80-88%). Collect 50-100 more images of failing conditions.")
else:
    print("\\n❌ Model needs more data. Review training curves and add more diverse images.")
'''

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6: Download the trained model
# ═══════════════════════════════════════════════════════════════════════════════

colab_cell_6 = '''
from google.colab import files
import shutil

# Copy best model with deployment name
best_path = f"runs/detect/{PROJECT_NAME}/weights/best.pt"
deploy_path = "document.pt"
shutil.copy(best_path, deploy_path)

# Download to your local machine
files.download(deploy_path)

print("\\n📦 Downloaded document.pt")
print("\\nNEXT STEPS:")
print("  1. Place document.pt in your backend\'s models/ directory")
print("  2. Restart your backend server")
print("  3. Check logs for: \'YOLOv8: model loaded and warmed up ✓\'")
print("  4. Test with a real video — you should see \'YOLO\' in the kept frame logs")
'''

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7: Inspect failures (run this to understand what to collect next)
# ═══════════════════════════════════════════════════════════════════════════════

colab_cell_7 = '''
import os
from pathlib import Path
from ultralytics import YOLO
import cv2
from IPython.display import display as ipy_display, Image as IPImage

best_model = YOLO(f"runs/detect/{PROJECT_NAME}/weights/best.pt")

# Run predictions on the validation set
val_images_dir = f"{dataset.location}/valid/images"
images = list(Path(val_images_dir).glob("*.jpg"))[:20]  # first 20

print(f"Showing predictions on {len(images)} validation images...")
for img_path in images:
    result = best_model.predict(str(img_path), conf=0.45, verbose=False)[0]
    plotted = result.plot()
    
    # Resize for display
    h, w = plotted.shape[:2]
    scale = min(400/w, 300/h)
    plotted_small = cv2.resize(plotted, (int(w*scale), int(h*scale)))
    
    cv2.imwrite("/tmp/preview.jpg", plotted_small)
    ipy_display(IPImage("/tmp/preview.jpg"))
    print(f"  {img_path.name}: {len(result.boxes)} detections")
'''


# ═══════════════════════════════════════════════════════════════════════════════
# PRINTABLE COLAB INSTRUCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           YOLOv8 Document Detector — Training Guide             ║
╚══════════════════════════════════════════════════════════════════╝

STEP 1: Data Collection (local machine)
───────────────────────────────────────
  python scripts/collect_training_frames.py \\
      --video your_video.mp4 \\
      --output training_frames/ \\
      --sample-every 15

  Collect 200 frames across these conditions:
    40 = Good lighting, flat camera
    20 = Bright/overexposed
    20 = Dim/underexposed
    20 = Side lamp (shadows on page)
    25 = Hand partially visible
    20 = Tilted angle
    15 = Close-up
    15 = Far away
    15 = Dark desk surface
    10 = White/bright desk

STEP 2: Upload to Roboflow
──────────────────────────
  1. Go to https://roboflow.com and create an account (free)
  2. New Project → Object Detection → name it "vidscan-document"
  3. Upload your 200 frames
  4. Annotate: draw box around document in EVERY image → label = "document"
  5. Generate dataset with:
       Train/Val/Test split: 80/15/5
       Augmentation: 5x with these settings:
         Flip: horizontal ✅
         Rotation: -20° to +20° ✅
         Brightness: -35% to +35% ✅
         Blur: 0 to 3px ✅
         Noise: up to 4% ✅
         Shadow: enabled ✅
         Crop: 0-15% ✅
  6. Export → Format = YOLOv8 → get API key

STEP 3: Train on Google Colab
──────────────────────────────
  1. Open https://colab.research.google.com
  2. Runtime → Change runtime type → T4 GPU
  3. Create new cells and paste CELL 1 through CELL 7 from this file
  4. Fill in ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT
  5. Run all cells (~45-60 min)
  6. Download document.pt when Cell 6 runs

STEP 4: Deploy
──────────────
  cp ~/Downloads/document.pt models/document.pt
  # Restart your backend
  # Check logs for: "YOLOv8: model loaded and warmed up ✓"

STEP 5: Test & Iterate
──────────────────────
  - Test with real videos in all conditions
  - If a condition fails: collect 50 more images of THAT condition only
  - Retrain with the expanded dataset (one round usually gets to 95%+)
""")
