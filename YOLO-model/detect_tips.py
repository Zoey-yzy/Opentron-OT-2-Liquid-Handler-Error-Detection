"""
Single-Channel Tip Detection & Volume Estimation
=================================================
Runs YOLO-NAS inference on an input image and outputs:

    pipette_x, pipette_y, pipette_w, pipette_h   ← Tip bounding box (normalized)
    liquid_x,  liquid_y,  liquid_w,  liquid_h    ← Liquid bounding box (normalized)
    volume_ul                                     ← Estimated liquid volume (µL)

Volume is derived from the liquid/tip bounding box height ratio,
matching the approach in Helper.py from OT2-Computer-Vision:
    volume_ul = (liquid_h / tip_h) * max_volume_ul

Usage:
    python detect_tips.py --image test.jpg --model ckpt_best.pth
    python detect_tips.py --image test.jpg --model ckpt_best.pth --max_volume 300
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Arguments ────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument(
    "--image",
    default=str(Path(__file__).parent / "test_img.png"),
    help="Input image path",
)
parser.add_argument(
    "--model",
    default=str(Path(__file__).parent / "ckpt_best.pth"),
    help="Path to YOLO-NAS .pth weights",
)
parser.add_argument("--conf",       type=float, default=0.7,
                    help="Confidence threshold (default: 0.7)")
parser.add_argument("--max_volume", type=float, default=300.0,
                    help="Max pipette volume in µL for volume estimation (default: 300)")
parser.add_argument(
    "--output",
    default=str(Path(__file__).parent / "detected_tips.png"),
    help="Path to save annotated output image",
)
args = parser.parse_args()

# ── Load model ────────────────────────────────────────────────────────────────

try:
    from super_gradients.training import models
except ImportError:
    print("ERROR: super_gradients not installed.  pip install super-gradients")
    sys.exit(1)

image_path = Path(args.image)
model_path = Path(args.model)

if not image_path.exists():
    print(f"ERROR: Image not found: {image_path}")
    sys.exit(1)
if not model_path.exists():
    print(f"ERROR: Model not found: {model_path}")
    sys.exit(1)

print(f"Loading model : {model_path}")
best_model = models.get(
    "yolo_nas_l",
    num_classes=2,
    checkpoint_path=str(model_path),
)
print("Model loaded.\n")

# ── Inference ─────────────────────────────────────────────────────────────────

image_bgr = cv2.imread(str(image_path))
if image_bgr is None:
    print(f"ERROR: Cannot read image: {image_path}")
    sys.exit(1)

img_h, img_w = image_bgr.shape[:2]
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

print(f"Running inference on : {image_path}  ({img_w}×{img_h})")
predictions = best_model.predict(image_rgb, conf=args.conf)

# ── Extract bounding boxes ────────────────────────────────────────────────────
#
# Mirrors Helper.py process_predictions() from OT2-Computer-Vision.
# Coordinates are normalized (0-1) relative to image size.

prediction  = predictions.prediction
labels      = prediction.labels
bboxes_xyxy = prediction.bboxes_xyxy      # pixel coords: x1 y1 x2 y2
class_names = predictions.class_names     # {0: 'Tip', 1: 'Liquid'}

tip_boxes    = []   # each: {"x","y","w","h","cx","cy","conf", "bbox_px"}
liquid_boxes = []

confidences = prediction.confidence

for label, bbox, conf in zip(labels, bboxes_xyxy, confidences):
    class_name = class_names[int(label)].lower()

    x1, y1, x2, y2 = bbox

    # Normalized center + size (YOLO format, 0-1)
    x_n = ((x1 + x2) / 2) / img_w
    y_n = ((y1 + y2) / 2) / img_h
    w_n = (x2 - x1) / img_w
    h_n = (y2 - y1) / img_h

    entry = {
        "x": float(x_n), "y": float(y_n),
        "w": float(w_n), "h": float(h_n),
        "cx_px": float((x1 + x2) / 2),
        "cy_px": float((y1 + y2) / 2),
        "conf": float(conf),
        "bbox_px": (float(x1), float(y1), float(x2), float(y2)),
    }

    if "tip" in class_name:
        tip_boxes.append(entry)
    elif "liquid" in class_name:
        liquid_boxes.append(entry)

# Sort left-to-right (matches Helper.py sort by x center)
tip_boxes.sort(   key=lambda d: d["x"])
liquid_boxes.sort(key=lambda d: d["x"])

# For 1-channel: take the highest-confidence detection of each class
tip    = max(tip_boxes,    key=lambda d: d["conf"]) if tip_boxes    else None
liquid = max(liquid_boxes, key=lambda d: d["conf"]) if liquid_boxes else None

# ── Volume estimation (mirrors Helper.py liquid_height_percentage) ────────────
#
# volume_ul = (liquid_h / tip_h) * max_volume_ul
# This is the same ratio used in the original OT2-Computer-Vision repo.

volume_ul = None
if tip and liquid:
    ratio     = liquid["h"] / tip["h"] if tip["h"] > 0 else 0.0
    volume_ul = ratio * args.max_volume

# ── Print results ─────────────────────────────────────────────────────────────

print("=" * 65)
print("  DETECTION RESULTS")
print("=" * 65)
print(f"  {'Parameter':<20} {'Value':>15}  {'Note'}")
print("  " + "-" * 60)

if tip:
    print(f"  {'pipette_x':<20} {tip['x']:>15.10f}  normalized x center")
    print(f"  {'pipette_y':<20} {tip['y']:>15.10f}  normalized y center")
    print(f"  {'pipette_w':<20} {tip['w']:>15.10f}  normalized width")
    print(f"  {'pipette_h':<20} {tip['h']:>15.10f}  normalized height")
    print(f"  {'pipette conf':<20} {tip['conf']:>15.4f}")
else:
    print("  [No Tip detected]")

print("  " + "-" * 60)

if liquid:
    print(f"  {'liquid_x':<20} {liquid['x']:>15.10f}  normalized x center")
    print(f"  {'liquid_y':<20} {liquid['y']:>15.10f}  normalized y center")
    print(f"  {'liquid_w':<20} {liquid['w']:>15.10f}  normalized width")
    print(f"  {'liquid_h':<20} {liquid['h']:>15.10f}  normalized height")
    print(f"  {'liquid conf':<20} {liquid['conf']:>15.4f}")
else:
    print("  [No Liquid detected]")

print("  " + "-" * 60)

if volume_ul is not None:
    print(f"  {'volume_ul':<20} {volume_ul:>15.2f}  µL  (liquid_h/tip_h × {args.max_volume})")
else:
    print("  volume_ul             —  (need both Tip and Liquid detected)")

print("=" * 65)

# ── Structured output dict (importable by other scripts) ─────────────────────

result = {
    "volume_ul"  : round(volume_ul, 4) if volume_ul is not None else None,
    "pipette_x"  : tip["x"]    if tip    else None,
    "pipette_y"  : tip["y"]    if tip    else None,
    "pipette_w"  : tip["w"]    if tip    else None,
    "pipette_h"  : tip["h"]    if tip    else None,
    "liquid_x"   : liquid["x"] if liquid else None,
    "liquid_y"   : liquid["y"] if liquid else None,
    "liquid_w"   : liquid["w"] if liquid else None,
    "liquid_h"   : liquid["h"] if liquid else None,
}

# ── Annotated output image ────────────────────────────────────────────────────

if tip:
    x1, y1, x2, y2 = [int(v) for v in tip["bbox_px"]]
    cx, cy = int(tip["cx_px"]), int(tip["cy_px"])
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
    cv2.circle(image_bgr, (cx, cy), 5, (0, 0, 255), -1)
    cv2.putText(image_bgr, f"Tip ({tip['x']:.3f},{tip['y']:.3f})",
                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

if liquid:
    x1, y1, x2, y2 = [int(v) for v in liquid["bbox_px"]]
    cx, cy = int(liquid["cx_px"]), int(liquid["cy_px"])
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (255, 140, 0), 2)
    cv2.circle(image_bgr, (cx, cy), 5, (255, 140, 0), -1)
    cv2.putText(image_bgr, f"Liquid ({liquid['x']:.3f},{liquid['y']:.3f})",
                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 1)

if volume_ul is not None:
    cv2.putText(image_bgr, f"Vol: {volume_ul:.1f} uL",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

cv2.imwrite(args.output, image_bgr)
print(f"\nAnnotated image → {args.output}")
