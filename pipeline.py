"""
Integrated OT2 Vision Pipeline
================================
Connects three components in sequence for each well:

  1. ot2_protocol.py  — robot aspirates, moves to camera, captures image
  2. detect_tips.py   — YOLO-NAS reads the image → pipette/liquid coordinates + volume
  3. error_model.py   — RandomForest checks if volume is within tolerance

Decision after each well:
  ErrorModel = 0 → no error → dispense and continue
  ErrorModel = 1 → error    → drop tip, stop protocol, print report

Usage:
    python pipeline.py --robot-ip 169.254.122.0 --model checkpoints/single_channel_tips/RUN_.../ckpt_best.pth
    python pipeline.py --robot-ip 169.254.122.0 --model ckpt_best.pth --conf 0.7 --max-volume 600
"""

import argparse
import logging
import sys
import csv
from datetime import datetime
from pathlib import Path

import time
import cv2
import numpy as np

# ── Arguments ────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="OT2 vision pipeline", allow_abbrev=False)
parser.add_argument("--robot-ip",   default="169.254.122.0",
                    help="OT2 robot IP address (default: 169.254.122.0)")
parser.add_argument("--max-volume", type=float, default=400.0,
                    help="Max pipette volume µL for volume estimation (default: 400)")
parser.add_argument("--tolerance",  type=float, default=15.0,
                    help="ErrorModel volume tolerance %% (default: 15)")
parser.add_argument("--cam-idx",    type=int,   default=0,
                    help="Camera index (default: 0)")
parser.add_argument("--save-dir",   default="/Users/zoey/Documents/2026spring/Autolab/project/captured_images",
                    help="Folder to save captured images")
parser.add_argument("--log",        default="pipeline_log.csv",
                    help="CSV file to log results for every well")
# ── HSV range for auto-annotation (mirrors auto_label.py defaults) ────────────
parser.add_argument("--h_low",  type=int, default=60,  help="HSV hue lower  (0-179)")
parser.add_argument("--h_high", type=int, default=100, help="HSV hue upper  (0-179)")
parser.add_argument("--s_low",  type=int, default=30,  help="HSV sat lower  (0-255)")
parser.add_argument("--s_high", type=int, default=255, help="HSV sat upper  (0-255)")
parser.add_argument("--v_low",  type=int, default=30,  help="HSV val lower  (0-255)")
parser.add_argument("--v_high", type=int, default=255, help="HSV val upper  (0-255)")
parser.add_argument("--tip_expand_top", type=float, default=0.10,
                    help="Fraction of image height to extend tip box above liquid")
args = parser.parse_args()

HSV_LOWER = np.array([args.h_low,  args.s_low,  args.v_low])
HSV_UPPER = np.array([args.h_high, args.s_high, args.v_high])

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# ── Load OT2 protocol components ─────────────────────────────────────────────

from ot2_protocol import OT2Client
from state_bridge import write_live_state

# ── Detection (HSV color detection) ──────────────────────────────────────────

def detect(image_path: str, volume_str: str = "?") -> dict:
    """
    Detect tip and liquid using HSV color thresholding (same logic as auto_label.py).
    Saves an annotated copy with suffix _auto_annotated.
    Returns normalized coordinates and estimated volume.
    """
    img = cv2.imread(image_path)
    if img is None:
        log.error(f"Cannot read image: {image_path}")
        return {}

    h, w = img.shape[:2]
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    preview = img.copy()

    if not contours:
        log.warning(f"No liquid detected in {Path(image_path).name} — check HSV range or framing")
        cv2.putText(preview, "NO LIQUID DETECTED", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        p = Path(image_path)
        ann_path = p.parent / (p.stem + "_auto_annotated" + p.suffix)
        cv2.imwrite(str(ann_path), preview)
        log.info(f"  Annotated image saved: {ann_path}")
        return {"annotated_image": str(ann_path)}

    liquid_cnt = max(contours, key=cv2.contourArea)
    lx, ly, lw, lh = cv2.boundingRect(liquid_cnt)

    tip_margin_x = int(w * 0.005)
    tip_x = max(0, lx - tip_margin_x)
    tip_w = min(w, lw + 2 * tip_margin_x)
    tip_y = max(0, ly - int(h * args.tip_expand_top))
    tip_h = (ly + lh) - tip_y

    volume_ul = round((lh / tip_h) * args.max_volume, 2) if tip_h > 0 else None

    fill_ratio = (lh / tip_h) if tip_h > 0 else 0.0

    # Tip box — green
    cv2.rectangle(preview, (tip_x, tip_y), (tip_x + tip_w, tip_y + tip_h), (0, 200, 0), 2)
    cv2.putText(preview, "Tip", (tip_x, tip_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 1)
    # Liquid box — orange
    cv2.rectangle(preview, (lx, ly), (lx + lw, ly + lh), (0, 140, 255), 2)
    cv2.putText(preview, "Liquid", (lx, ly - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 1)
    # Volume labels
    cv2.putText(preview, f"Protocol: {volume_str} uL", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    detected_label = f"Detected: {volume_ul} uL" if volume_ul is not None else "No detection"
    cv2.putText(preview, detected_label, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    p = Path(image_path)
    ann_path = p.parent / (p.stem + "_auto_annotated" + p.suffix)
    cv2.imwrite(str(ann_path), preview)
    log.info(f"  Annotated image saved: {ann_path}  "
             f"liq_h={lh/h:.3f}  tip_h={tip_h/h:.3f}  fill={fill_ratio:.2f}")

    return {
        "pipette_x"       : (tip_x + tip_w / 2) / w,
        "pipette_y"       : (tip_y + tip_h / 2) / h,
        "pipette_w"       : tip_w / w,
        "pipette_h"       : tip_h / h,
        "liquid_x"        : (lx + lw / 2) / w,
        "liquid_y"        : (ly + lh / 2) / h,
        "liquid_w"        : lw / w,
        "liquid_h"        : lh / h,
        "volume_ul"       : volume_ul,
        "annotated_image" : str(ann_path),
    }

# ── Per-well check ────────────────────────────────────────────────────────────

def check_well(image_path: str, expected_volume: float) -> tuple[dict, int]:
    """
    Run HSV detection on a captured image and compare to the expected volume.

    Returns:
        (detection_result, error_flag)
        error_flag: 0 = OK (continue), 1 = error (stop)
    """
    result = detect(image_path, volume_str=str(expected_volume))

    if not result or result["volume_ul"] is None:
        log.warning("Detection failed — treating as error.")
        return result, 1

    detected = result["volume_ul"]
    deviation_pct = abs(detected - expected_volume) / expected_volume * 100
    error_flag = 1 if deviation_pct > args.tolerance else 0

    log.info(f"  Volume check: expected={expected_volume} µL  "
             f"detected={detected} µL  "
             f"deviation={deviation_pct:.1f}%  "
             f"tolerance={args.tolerance}%  "
             f"→ {'ERROR' if error_flag else 'OK'}")

    return result, error_flag

# ── Log writer ────────────────────────────────────────────────────────────────

log_path = Path(args.log)
log_fields = [
    "timestamp", "well", "expected_vol",
    "pipette_x", "pipette_y", "pipette_w", "pipette_h",
    "liquid_x",  "liquid_y",  "liquid_w",  "liquid_h",
    "volume_ul", "error_flag", "decision", "annotated_image",
]

with open(log_path, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=log_fields).writeheader()

def write_log(row: dict):
    with open(log_path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writerow(row)

# ── Main protocol ─────────────────────────────────────────────────────────────

def run():
    # Open camera once and warm it up before the protocol starts
    log.info(f"Opening camera index {args.cam_idx} ...")
    cap = cv2.VideoCapture(args.cam_idx)
    if not cap.isOpened():
        log.error(f"Cannot open camera index {args.cam_idx}. Aborting.")
        return
    time.sleep(2.0)                     # let the USB camera hardware initialise
    for _ in range(60):                 # flush stale/black frames
        cap.read()
    ret, test_frame = cap.read()
    if not ret or test_frame.mean() < 2:
        log.error("Camera opened but returns black frames. Check USB connection and macOS camera permissions.")
        cap.release()
        return
    log.info(f"Camera ready (brightness={test_frame.mean():.1f}).")

    robot = OT2Client(args.robot_ip)
    robot.set_lights(True)

    # ── Step 0: Safe reset ────────────────────────────────────────────────────
    # If a previous run was interrupted the pipette may still hold liquid.
    # Load a cleanup run, try to return liquid to reservoir and drop the tip,
    # then home all axes so the robot starts from a known position.
    log.info("Running safe reset before protocol ...")
    robot.create_run()
    robot.load_pipette("p1000_single_gen2", "right")
    _res_cleanup = robot.load_labware("nest_1_reservoir_290ml", "4")
    try:
        robot.dispense(args.max_volume, _res_cleanup, "A1")
        log.info("Safe reset: dispensed any remaining liquid back to reservoir.")
    except Exception:
        log.info("Safe reset: nothing to dispense (pipette was empty).")
    try:
        robot.drop_tip()
        log.info("Safe reset: tip dropped.")
    except Exception:
        log.info("Safe reset: no tip to drop.")
    robot.home()
    log.info("Safe reset complete — robot is homed.")

    # ── Protocol run ──────────────────────────────────────────────────────────
    robot.create_run()
    robot.load_pipette("p1000_single_gen2", "right")
    tips  = robot.load_labware("opentrons_96_tiprack_1000ul", "2")
    res   = robot.load_labware("nest_1_reservoir_290ml", "4")
    plate = robot.load_labware("nest_96_wellplate_200ul_flat", "5", version=5)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    rows_to_process = [
        ("A", 300.0), ("B", 325.0), ("C", 350.0), ("D", 375.0),
        ("E", 400.0), ("F", 275.0), ("G", 250.0), ("H", 225.0),
    ]

    report = []   # collect summary for final output

    print("\n" + "=" * 65)
    print("  OT2 VISION PIPELINE STARTED")
    print(f"  Robot IP   : {args.robot_ip}")
    print(f"  Detection  : HSV color thresholding")
    print(f"  Tolerance  : ±{args.tolerance}%")
    print("=" * 65 + "\n")

    for row_letter, volume in rows_to_process:
        log.info(f"── Row {row_letter}  ({volume} µL) ──────────────────────")
        robot.pick_up_tip(tips, f"{row_letter}1")

        for col in range(1, 13):
            well = f"{row_letter}{col}"

            # ── Step 1: Aspirate ──────────────────────────────────────
            log.info(f"  [{well}] Aspirating {volume} µL ...")
            robot.aspirate(volume, res, "A1")

            # ── Step 2: Move to camera + capture ─────────────────────
            robot.pose_for_camera(res, "A1")
            img_name = f"Row_{well[0]}_Well_{well[1:]}_{volume}uL.jpg"
            img_path = save_dir / img_name

            time.sleep(2.0)         # let auto-exposure settle after robot moves
            for _ in range(30):     # flush stale buffered frames
                cap.read()
            ret, frame = cap.read()

            if not ret:
                log.error(f"  [{well}] Camera capture failed — stopping.")
                robot.drop_tip()
                robot.set_lights(False)
                cap.release()
                return

            cv2.imwrite(str(img_path), frame)
            log.info(f"  [{well}] Image saved: {img_path}")

            # ── Step 3: HSV detection + volume check ──────────────────
            result, error_flag = check_well(str(img_path), volume)

            write_live_state(
                status="running",
                current_well=well,
                expected_vol=volume,
                result=result,
                error_flag=error_flag,
                wells_completed=len([r for r in report if r["error"] == 0]),
                wells_total=96,
                tolerance_pct=args.tolerance,
                camera_connected=True,
            )

            decision = "CONTINUE" if error_flag == 0 else "STOP"
            vol_detected = result.get("volume_ul", "N/A")
            if isinstance(vol_detected, (int, float)):
                deviation_pct = abs(vol_detected - volume) / volume * 100
                yolo_str      = f"{vol_detected:.1f}"
                dev_str       = f"{deviation_pct:.1f}%"
            else:
                deviation_pct = float("nan")
                yolo_str      = "N/A"
                dev_str       = "N/A"

            print(f"  [{well}]  Protocol: {volume:.1f} µL  |  "
                  f"HSV: {yolo_str} µL  |  "
                  f"Δ {dev_str}  |  {decision}")

            # ── Log to CSV ────────────────────────────────────────────
            write_log({
                "timestamp"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "well"           : well,
                "expected_vol"   : volume,
                "pipette_x"      : result.get("pipette_x"),
                "pipette_y"      : result.get("pipette_y"),
                "pipette_w"      : result.get("pipette_w"),
                "pipette_h"      : result.get("pipette_h"),
                "liquid_x"       : result.get("liquid_x"),
                "liquid_y"       : result.get("liquid_y"),
                "liquid_w"       : result.get("liquid_w"),
                "liquid_h"       : result.get("liquid_h"),
                "volume_ul"      : vol_detected,
                "error_flag"     : error_flag,
                "decision"       : decision,
                "annotated_image": result.get("annotated_image"),
            })

            report.append({
                "well": well, "expected": volume,
                "detected": vol_detected, "deviation": deviation_pct, "error": error_flag,
            })

            # ── Step 4: Decide ────────────────────────────────────────
            if error_flag == 1:
                log.error(f"  [{well}] ERROR DETECTED — dispensing back to reservoir.")
                robot.dispense(volume, res, "A1")
                log.info(f"  [{well}] Liquid returned to reservoir. Dropping tip.")
                robot.drop_tip()
                robot.set_lights(False)
                cap.release()
                write_live_state(
                    status="stopped",
                    current_well=well,
                    expected_vol=volume,
                    result=result,
                    error_flag=1,
                    wells_completed=len([r for r in report if r["error"] == 0]),
                    wells_total=96,
                    tolerance_pct=args.tolerance,
                    camera_connected=False,
                )
                _print_report(report, stopped_at=well)
                return

            # ── Step 5: Dispense ──────────────────────────────────────
            robot.dispense(volume, plate, well)
            log.info(f"  [{well}] Dispensed OK.")

        robot.drop_tip()
        log.info(f"Row {row_letter} complete.\n")

    robot.set_lights(False)
    cap.release()
    write_live_state(
        status="complete",
        current_well="—",
        expected_vol=0,
        result={},
        error_flag=0,
        wells_completed=96,
        wells_total=96,
        tolerance_pct=args.tolerance,
        camera_connected=False,
    )
    _print_report(report, stopped_at=None)

# ── Final report ──────────────────────────────────────────────────────────────

def _print_report(report: list, stopped_at):
    print("\n" + "=" * 72)
    if stopped_at:
        print(f"  PROTOCOL STOPPED at well {stopped_at}")
    else:
        print("  PROTOCOL COMPLETED SUCCESSFULLY")
    print("=" * 72)
    print(f"  {'Well':<6} {'Protocol (µL)':>13} {'HSV (µL)':>10} {'Δ%':>7} {'Error':>6}  Decision")
    print("  " + "-" * 62)
    for r in report:
        decision = "STOP" if r["error"] else "OK"
        det = r["detected"]
        dev = r["deviation"]
        det_str = f"{det:.1f}" if isinstance(det, (int, float)) else "N/A"
        dev_str = f"{dev:.1f}" if isinstance(dev, float) and dev == dev else "N/A"
        print(f"  {r['well']:<6} {r['expected']:>13.1f} {det_str:>10}  {dev_str:>6}%  "
              f"{'YES' if r['error'] else 'NO':>5}   {decision}")
    print("=" * 72)
    print(f"  Full log saved → {args.log}\n")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run()
