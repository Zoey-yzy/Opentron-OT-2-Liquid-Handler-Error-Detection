"""
Auto-Label Dataset from Captured Images
=========================================
Generates YOLO bounding box label files for images in
captured_samples-selected/ using:

  1. Fixed pipette bbox  — averaged from targeted_pipette_data.csv
     (camera is fixed so pipette position is nearly constant, std < 0.002)

  2. Liquid bbox height/y — fitted linearly from CSV:
       liquid_h = 0.00116430 * volume_ul + 0.157529
       liquid_y = -0.00060237 * volume_ul + 0.814316
     Volume is read from the filename, e.g. Row_A_Well_1_500.0uL.jpg → 500 µL

Output YOLO label format (one line per object, normalized 0-1):
    0  pipette_x  pipette_y  pipette_w  pipette_h    ← Tip
    1  liquid_x   liquid_y   liquid_w   liquid_h     ← Liquid

Then splits images + labels into dataset/train/valid/test.

Usage:
    python prepare_labels.py
"""

import re
import random
import shutil
from pathlib import Path
from collections import defaultdict

random.seed(42)

SRC_DIR = Path(__file__).parent / "captured_samples-selected"
DST_DIR = Path(__file__).parent / "dataset"

# ── Fixed parameters from CSV statistics ─────────────────────────────────────
# Computed from targeted_pipette_data.csv (1000 rows, fixed camera setup)

PIPETTE_X = 0.505919   # x center  (std 0.001 — essentially constant)
PIPETTE_Y = 0.579167   # y center  (std 0.002)
PIPETTE_W = 0.062049   # width     (std 0.001)
PIPETTE_H = 0.625046   # height    (std 0.002)

LIQUID_X  = 0.501986   # x center  (std 0.001 — follows pipette)
LIQUID_W  = 0.044993   # width     (std 0.002)

# Linear fit: volume_ul → liquid_h and liquid_y
# liquid_h = LH_A * volume + LH_B
# liquid_y = LY_A * volume + LY_B
LH_A =  0.00116430
LH_B =  0.157529
LY_A = -0.00060237
LY_B =  0.814316

def liquid_bbox(volume_ul: float) -> tuple:
    """Return (liquid_x, liquid_y, liquid_w, liquid_h) for a given volume."""
    lh = LH_A * volume_ul + LH_B
    ly = LY_A * volume_ul + LY_B
    # Clamp to valid [0, 1] range
    lh = max(0.01, min(1.0, lh))
    ly = max(0.0,  min(1.0, ly))
    return LIQUID_X, ly, LIQUID_W, lh

# ── Find images and extract volumes ───────────────────────────────────────────

images = sorted(SRC_DIR.glob("*.jpg")) + sorted(SRC_DIR.glob("*.png"))

parsed = []
for img in images:
    m = re.search(r"_([\d.]+)uL", img.name)
    if not m:
        print(f"  WARNING: cannot parse volume from {img.name}, skipping")
        continue
    vol = float(m.group(1))
    parsed.append((img, vol))

print(f"Found {len(parsed)} images in {SRC_DIR}\n")

# ── Show generated labels for a few examples ─────────────────────────────────

print("Sample auto-generated labels:")
print(f"  {'Image':<40} {'Vol':>6}  liquid_h   liquid_y")
print("  " + "-" * 70)
for img, vol in parsed[:5]:
    _, ly, _, lh = liquid_bbox(vol)
    print(f"  {img.name:<40} {vol:>6.1f}  {lh:.6f}   {ly:.6f}")
print()

# ── Stratified split by volume level ─────────────────────────────────────────

by_vol = defaultdict(list)
for img, vol in parsed:
    by_vol[vol].append((img, vol))

train_data, val_data, test_data = [], [], []

for vol, items in sorted(by_vol.items()):
    random.shuffle(items)
    n       = len(items)
    n_val   = max(1, round(n * 0.15))
    n_test  = max(1, round(n * 0.10))
    n_train = n - n_val - n_test

    if n == 1:
        train_data += items
        continue

    train_data += items[:n_train]
    val_data   += items[n_train:n_train + n_val]
    test_data  += items[n_train + n_val:]

print(f"Split (stratified by volume):")
print(f"  Train : {len(train_data)} images")
print(f"  Valid : {len(val_data)} images")
print(f"  Test  : {len(test_data)} images\n")

# ── Write dataset folders ─────────────────────────────────────────────────────

for split, data in [("train", train_data), ("valid", val_data), ("test", test_data)]:
    img_dir = DST_DIR / split / "images"
    lbl_dir = DST_DIR / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path, vol in data:
        # Copy image
        shutil.copy2(img_path, img_dir / img_path.name)

        # Generate YOLO label
        lx, ly, lw, lh = liquid_bbox(vol)
        label_path = lbl_dir / (img_path.stem + ".txt")
        with open(label_path, "w") as f:
            # Class 0: Tip (pipette) — fixed position
            f.write(f"0 {PIPETTE_X:.10f} {PIPETTE_Y:.10f} "
                    f"{PIPETTE_W:.10f} {PIPETTE_H:.10f}\n")
            # Class 1: Liquid — height/y computed from volume
            f.write(f"1 {lx:.10f} {ly:.10f} "
                    f"{lw:.10f} {lh:.10f}\n")

    print(f"  {split}: {len(data)} images + labels written → dataset/{split}/")

# ── data.yaml ─────────────────────────────────────────────────────────────────

yaml_path = DST_DIR / "data.yaml"
with open(yaml_path, "w") as f:
    f.write(f"train: {DST_DIR / 'train' / 'images'}\n")
    f.write(f"val:   {DST_DIR / 'valid' / 'images'}\n")
    f.write(f"test:  {DST_DIR / 'test'  / 'images'}\n")
    f.write(f"nc: 2\n")
    f.write(f"names: ['Tip', 'Liquid']\n")

print(f"\ndata.yaml → {yaml_path}")
print("\nNext: python train_yolo.py")
