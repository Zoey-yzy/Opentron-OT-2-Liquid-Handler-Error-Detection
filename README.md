# Opentron-OT-2-Liquid-Handler-Error-Detection

Real-time vision-based error detection for the Opentrons OT-2 liquid handler. A USB camera captures the pipette tip after each aspiration; HSV color thresholding estimates the aspirated volume and flags errors before any liquid reaches the plate.

---

## How It Works

```
Robot aspirates → Camera captures image → HSV detection estimates volume → Error check → Dispense or Stop
```

For each well the pipeline:
1. Commands the OT-2 to aspirate and move the pipette to the camera position
2. Captures a frame and runs HSV color thresholding to locate the liquid column
3. Estimates volume from the liquid/tip height ratio
4. Compares detected volume to the expected protocol volume (±15% tolerance)
5. **No error** → dispenses into the plate and continues
6. **Error detected** → returns liquid to reservoir, drops tip, stops protocol

Two error scenarios are included alongside the normal run:

| Script | Error injected | Expected outcome |
|---|---|---|
| `pipeline.py` | None | All wells complete normally |
| `pipeline_error_no_tip.py` | Tip never picked up | No liquid detected → stops at well A1 |
| `pipeline_error_wrong_volume.py` | Aspirates 500 µL, expects 100 µL | 400% deviation → stops at well A1 |

---

## Requirements

**Hardware**
- Opentrons OT-2 robot (IP default: `169.254.122.0`)
- USB camera (index 0 by default)
- p1000 single-channel pipette (right mount)
- Slot 2: `opentrons_96_tiprack_1000ul`
- Slot 4: `nest_1_reservoir_290ml`
- Slot 5: `nest_96_wellplate_200ul_flat`

**Software**
```
pip install opencv-python numpy requests urllib3
```

---

## Usage

**Normal pipeline** — processes all 8 rows × 12 wells:
```bash
python pipeline.py --robot-ip 169.254.122.0
```

**Error demo 1: No tip**
```bash
python pipeline_error_no_tip.py --robot-ip 169.254.122.0
```

**Error demo 2: Wrong volume**
```bash
python pipeline_error_wrong_volume.py --robot-ip 169.254.122.0
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--robot-ip` | `169.254.122.0` | OT-2 IP address |
| `--cam-idx` | `0` | USB camera index |
| `--max-volume` | `400.0` | Pipette max volume µL (scales the HSV ratio) |
| `--tolerance` | `15.0` | Allowed volume deviation % before error |
| `--save-dir` | `captured_images/` | Folder for captured + annotated images |
| `--log` | `pipeline_log.csv` | CSV log of every well result |

### Tuning the HSV color range

If your liquid color differs from the default (green dye, hue 60–100), adjust:
```bash
python pipeline.py --robot-ip 169.254.122.0 --h_low 40 --h_high 80 --s_low 50
```

---

## Output

Each run produces:
- **Annotated images** (`*_auto_annotated.jpg`) with bounding boxes for tip and liquid, and volume labels
- **CSV log** with per-well timestamp, bounding box coordinates, detected volume, error flag, and decision

Console output per well:
```
  [A1]  Protocol: 300.0 µL  |  HSV: 297.3 µL  |  Δ 0.9%  |  CONTINUE
  [A2]  Protocol: 300.0 µL  |  HSV: 301.1 µL  |  Δ 0.4%  |  CONTINUE
```

---

## Files

| File | Description |
|---|---|
| `ot2_protocol.py` | OT-2 HTTP client — wraps the robot REST API |
| `pipeline.py` | Normal run with per-well error detection |
| `pipeline_error_no_tip.py` | Error demo: skips tip pick-up |
| `pipeline_error_wrong_volume.py` | Error demo: aspirates wrong volume |
