"""
Error Protocol 1: No Tip
========================
Robot skips pick_up_tip entirely, then tries to aspirate.
Without a tip, no liquid enters the pipette — the camera sees
nothing green, HSV detection finds no contours, and the pipeline
flags an error and stops on the very first well.

Expected behaviour:
  [A1]  Protocol: 300.0 µL  |  HSV: N/A µL  |  Δ N/A  |  STOP

Usage:
    python pipeline_error_no_tip.py --robot-ip 169.254.122.0
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

parser = argparse.ArgumentParser(description="Error protocol: no tip", allow_abbrev=False)
parser.add_argument("--robot-ip",    default="169.254.122.0")
parser.add_argument("--max-volume",  type=float, default=400.0)
parser.add_argument("--tolerance",   type=float, default=15.0)
parser.add_argument("--cam-idx",     type=int,   default=0)
parser.add_argument("--save-dir",    default="/Users/zoey/Documents/2026spring/Autolab/project/captured_images")
parser.add_argument("--log",         default="pipeline_error_no_tip_log.csv")
parser.add_argument("--h_low",       type=int,   default=60)
parser.add_argument("--h_high",      type=int,   default=100)
parser.add_argument("--s_low",       type=int,   default=30)
parser.add_argument("--s_high",      type=int,   default=255)
parser.add_argument("--v_low",       type=int,   default=30)
parser.add_argument("--v_high",      type=int,   default=255)
parser.add_argument("--tip_expand_top", type=float, default=0.10)
args = parser.parse_args()

HSV_LOWER = np.array([args.h_low,  args.s_low,  args.v_low])
HSV_UPPER = np.array([args.h_high, args.s_high, args.v_high])

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s - %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pipeline_no_tip")

from ot2_protocol import OT2Client

# ── Detection (HSV) ───────────────────────────────────────────────────────────

def detect(image_path: str, volume_str: str = "?") -> dict:
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
    p = Path(image_path)
    ann_path = p.parent / (p.stem + "_auto_annotated" + p.suffix)

    if not contours:
        log.warning(f"No liquid detected in {p.name}")
        cv2.putText(preview, "NO LIQUID DETECTED", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(preview, f"Protocol: {volume_str} uL", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
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
    volume_ul   = round((lh / tip_h) * args.max_volume, 2) if tip_h > 0 else None
    fill_ratio  = (lh / tip_h) if tip_h > 0 else 0.0

    cv2.rectangle(preview, (tip_x, tip_y), (tip_x + tip_w, tip_y + tip_h), (0, 200, 0), 2)
    cv2.putText(preview, "Tip", (tip_x, tip_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 1)
    cv2.rectangle(preview, (lx, ly), (lx + lw, ly + lh), (0, 140, 255), 2)
    cv2.putText(preview, "Liquid", (lx, ly - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 1)
    cv2.putText(preview, f"Protocol: {volume_str} uL", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(preview, f"Detected: {volume_ul} uL", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(ann_path), preview)
    log.info(f"  Annotated image saved: {ann_path}  fill={fill_ratio:.2f}")

    return {
        "pipette_x": (tip_x + tip_w / 2) / w, "pipette_y": (tip_y + tip_h / 2) / h,
        "pipette_w": tip_w / w,                "pipette_h": tip_h / h,
        "liquid_x":  (lx + lw / 2) / w,       "liquid_y":  (ly + lh / 2) / h,
        "liquid_w":  lw / w,                   "liquid_h":  lh / h,
        "volume_ul": volume_ul, "annotated_image": str(ann_path),
    }

def check_well(image_path: str, expected_volume: float) -> tuple[dict, int]:
    result = detect(image_path, volume_str=str(expected_volume))
    if not result or result.get("volume_ul") is None:
        log.warning("Detection failed (no liquid) — error flagged.")
        return result, 1
    detected = result["volume_ul"]
    deviation_pct = abs(detected - expected_volume) / expected_volume * 100
    error_flag = 1 if deviation_pct > args.tolerance else 0
    log.info(f"  Volume check: expected={expected_volume} µL  detected={detected} µL  "
             f"deviation={deviation_pct:.1f}%  → {'ERROR' if error_flag else 'OK'}")
    return result, error_flag

# ── Log writer ────────────────────────────────────────────────────────────────

log_path   = Path(args.log)
log_fields = ["timestamp", "well", "expected_vol",
              "pipette_x", "pipette_y", "pipette_w", "pipette_h",
              "liquid_x",  "liquid_y",  "liquid_w",  "liquid_h",
              "volume_ul", "error_flag", "decision", "annotated_image"]

with open(log_path, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=log_fields).writeheader()

def write_log(row: dict):
    with open(log_path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writerow(row)

# ── Main protocol ─────────────────────────────────────────────────────────────

def run():
    log.info(f"Opening camera index {args.cam_idx} ...")
    cap = cv2.VideoCapture(args.cam_idx)
    if not cap.isOpened():
        log.error("Cannot open camera. Aborting.")
        return
    time.sleep(2.0)
    for _ in range(60):
        cap.read()
    ret, test_frame = cap.read()
    if not ret or test_frame.mean() < 2:
        log.error("Camera returns black frames.")
        cap.release()
        return
    log.info(f"Camera ready (brightness={test_frame.mean():.1f}).")

    robot = OT2Client(args.robot_ip)
    robot.set_lights(True)

    # ── Step 0: Safe reset ────────────────────────────────────────────────────
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

    # Protocol run
    robot.create_run()
    robot.load_pipette("p1000_single_gen2", "right")
    robot.load_labware("opentrons_96_tiprack_1000ul", "2")   # load but do NOT pick up
    res   = robot.load_labware("nest_1_reservoir_290ml", "4")
    plate = robot.load_labware("nest_96_wellplate_200ul_flat", "5", version=5)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    volume = 300.0   # expected protocol volume
    report = []

    print("\n" + "=" * 65)
    print("  ERROR PROTOCOL: NO TIP")
    print(f"  Robot IP   : {args.robot_ip}")
    print(f"  Expected   : {volume} µL  (tip not picked up — should fail)")
    print(f"  Tolerance  : ±{args.tolerance}%")
    print("=" * 65 + "\n")

    # ── Intentional error: skip pick_up_tip ───────────────────────────────────
    log.warning("NO TIP PICKED UP — this is intentional for error testing.")

    for col in range(1, 13):
        well = f"A{col}"

        log.info(f"  [{well}] Aspirating {volume} µL (no tip attached) ...")
        robot.aspirate(volume, res, "A1")

        robot.pose_for_camera(res, "A1")
        img_name = f"NoTip_Well_{col}_{volume}uL.jpg"
        img_path = save_dir / img_name

        time.sleep(2.0)
        for _ in range(30):
            cap.read()
        ret, frame = cap.read()

        if not ret:
            log.error(f"  [{well}] Camera capture failed — stopping.")
            robot.set_lights(False)
            cap.release()
            return

        cv2.imwrite(str(img_path), frame)
        log.info(f"  [{well}] Image saved: {img_path}")

        result, error_flag = check_well(str(img_path), volume)

        decision    = "CONTINUE" if error_flag == 0 else "STOP"
        vol_detected = result.get("volume_ul")
        if isinstance(vol_detected, (int, float)):
            hsv_str = f"{vol_detected:.1f}"
            dev_str = f"{abs(vol_detected - volume) / volume * 100:.1f}%"
            deviation_pct = abs(vol_detected - volume) / volume * 100
        else:
            hsv_str = "N/A"
            dev_str = "N/A"
            deviation_pct = float("nan")

        print(f"  [{well}]  Protocol: {volume:.1f} µL  |  HSV: {hsv_str} µL  |  Δ {dev_str}  |  {decision}")

        write_log({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "well": well, "expected_vol": volume,
            "pipette_x": result.get("pipette_x"), "pipette_y": result.get("pipette_y"),
            "pipette_w": result.get("pipette_w"), "pipette_h": result.get("pipette_h"),
            "liquid_x":  result.get("liquid_x"),  "liquid_y":  result.get("liquid_y"),
            "liquid_w":  result.get("liquid_w"),   "liquid_h":  result.get("liquid_h"),
            "volume_ul": vol_detected, "error_flag": error_flag,
            "decision": decision, "annotated_image": result.get("annotated_image"),
        })

        report.append({"well": well, "expected": volume,
                       "detected": vol_detected, "deviation": deviation_pct, "error": error_flag})

        if error_flag == 1:
            log.error(f"  [{well}] ERROR DETECTED — stopping protocol.")
            robot.set_lights(False)
            cap.release()
            _print_report(report, stopped_at=well)
            return

        # Would dispense if no error, but we expect to stop immediately
        robot.dispense(volume, plate, well)

    robot.set_lights(False)
    cap.release()
    _print_report(report, stopped_at=None)

def _print_report(report: list, stopped_at: str | None):
    print("\n" + "=" * 72)
    print(f"  PROTOCOL STOPPED at well {stopped_at}" if stopped_at else "  PROTOCOL COMPLETED")
    print("=" * 72)
    print(f"  {'Well':<6} {'Protocol (µL)':>13} {'HSV (µL)':>10} {'Δ%':>7} {'Error':>6}  Decision")
    print("  " + "-" * 62)
    for r in report:
        det = r["detected"]
        dev = r["deviation"]
        det_str = f"{det:.1f}" if isinstance(det, (int, float)) else "N/A"
        dev_str = f"{dev:.1f}" if isinstance(dev, float) and dev == dev else "N/A"
        decision = "STOP" if r["error"] else "OK"
        print(f"  {r['well']:<6} {r['expected']:>13.1f} {det_str:>10}  {dev_str:>6}%  "
              f"{'YES' if r['error'] else 'NO':>5}   {decision}")
    print("=" * 72)
    print(f"  Full log saved → {args.log}\n")

if __name__ == "__main__":
    run()
