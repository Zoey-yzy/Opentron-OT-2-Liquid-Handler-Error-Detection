"""
Add to ot2_vision_pipeline.py — shows the two additions needed.

1. Import this at the top of ot2_vision_pipeline.py:
       from state_bridge import write_live_state, LIVE_STATE_FILE

2. Call write_live_state(...) after check_well() and after the decision in run().
"""

import json
from datetime import datetime
from pathlib import Path

LIVE_STATE_FILE = Path("live_state.json")

# Global counters — set these at the start of run()
_total_wells = 96
_wells_done  = 0


def write_live_state(
    *,
    status: str,          # "idle" | "running" | "complete" | "error" | "stopped"
    current_well: str,
    expected_vol: float,
    result: dict,         # the dict returned by detect()
    error_flag: int,
    wells_completed: int,
    wells_total: int,
    tolerance_pct: float,
    camera_connected: bool = True,
    z_height: float = 0.0,
    mae: str = "2.44",
    r2: str = "0.997",
    sample_count: int = 430,
):
    detected_vol  = result.get("volume_ul")
    deviation_pct = None
    if detected_vol is not None and expected_vol > 0:
        deviation_pct = abs(detected_vol - expected_vol) / expected_vol * 100

    # Pixel-space width from normalised fraction * assumed image width 640
    pipette_w_norm = result.get("pipette_w", 0.0)
    pipette_w_px   = pipette_w_norm * 640

    pipette_y_norm = result.get("pipette_y", 0.0)
    liquid_y_norm  = result.get("liquid_y", 0.0)
    # Scale to pixel space for the dashboard threshold checks
    pipette_y_px   = pipette_y_norm * 480
    liquid_y_px    = liquid_y_norm  * 480

    state = {
        "status"          : status,
        "current_well"    : current_well,
        "expected_vol"    : expected_vol,
        "detected_vol"    : detected_vol,
        "deviation_pct"   : round(deviation_pct, 2) if deviation_pct is not None else None,
        "error_flag"      : error_flag,
        "decision"        : "STOP" if error_flag else "CONTINUE",
        "wells_completed" : wells_completed,
        "wells_total"     : wells_total,
        "tolerance_pct"   : tolerance_pct,
        "camera_connected": camera_connected,
        "mpi"             : _estimate_mpi(result),
        "pipette_w_px"    : round(pipette_w_px, 1),
        "pipette_y"       : round(pipette_y_px, 1),
        "liquid_y"        : round(liquid_y_px, 1),
        "z_height"        : z_height,
        "annotated_image" : result.get("annotated_image"),
        "mae"             : mae,
        "r2"              : r2,
        "sample_count"    : sample_count,
        "timestamp"       : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Atomic write: write to .tmp then rename so the dashboard never reads partial JSON
    tmp = LIVE_STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(LIVE_STATE_FILE)


def _estimate_mpi(result: dict) -> float:
    """
    Mean pixel intensity proxy from the liquid bounding-box area.
    In practice you'd compute this from the actual frame in pipeline.py.
    Here we return a synthetic value so the dashboard QC check has something to show.
    """
    if result.get("volume_ul") is not None:
        return 142.0   # replace with cv2.mean(frame[ly:ly+lh, lx:lx+lw])[0] in pipeline.py
    return 0.0
