# OT-2 Vision Pipeline — How to Run

This guide walks you through setting up and running the automated pipette QC system on your laptop.
The system has two parts that run simultaneously:

- **`ot2_vision_pipeline.py`** — controls the OT-2 robot, captures images, detects liquid volume, and logs results
- **`qc_live_monitor.py`** — Streamlit web app that reads those results and shows a live QC display

---

## What You Need

| Item | Requirement |
|---|---|
| Computer | macOS or Linux (Windows works but USB camera drivers vary) |
| Python | 3.10 or newer (`python3 --version`) |
| USB Camera | Plugged in before running |
| OT-2 Robot | Connected via USB-to-Ethernet adapter, IP `169.254.122.0` |
| Network | Laptop and robot on the same local network (or direct USB link) |

---

## 1. Get the Code

If your team uses a shared folder or repo, clone/download it. Then open a terminal and go to the project folder:

```bash
cd /path/to/liverunmonitor
```

You should see these files:
```
liverunmonitor/
├── ot2_vision_pipeline.py
├── state_bridge.py
├── qc_live_monitor.py
├── ot2_protocol.py
├── SETUP_AND_RUN_GUIDE.md
└── requirements.txt
```

---

## 2. Create a Python Environment

Do this once per laptop. It keeps the project's packages separate from everything else on your system.

```bash
python3 -m venv .venv
```

Activate it — **you must do this every time you open a new terminal**:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Your terminal prompt will show `(.venv)` when it is active.

---

## 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If there is no `requirements.txt` yet, run this instead:

```bash
pip install \
  streamlit \
  streamlit-autorefresh \
  plotly \
  pandas \
  numpy \
  opencv-python \
  opentrons
```

This takes a few minutes the first time. Subsequent installs are instant.

---

## 4. Check Your Camera

Plug in the USB camera **before** running anything. To confirm Python can see it:

```bash
python3 - <<'EOF'
import cv2
cap = cv2.VideoCapture(0)
print("Camera OK" if cap.isOpened() else "Camera NOT found")
cap.release()
EOF
```

If you see `Camera NOT found`:
- Try a different USB port
- On macOS: System Settings → Privacy & Security → Camera → allow Terminal
- Change `--cam-idx 1` (or 2) when running the pipeline (see Step 6)

---

## 5. Check the Robot Connection

The robot must be reachable at `169.254.122.0`. Test with:

```bash
ping -c 3 169.254.122.0
```

If that times out:
- Make sure the USB-to-Ethernet adapter is plugged in
- macOS: System Settings → Network → the adapter should show as "Connected"
- The adapter needs to be set to manual IP `169.254.1.x`, subnet `255.255.0.0` (ask your lab admin if unsure)

---

## 6. Set the Image Save Folder

Open `ot2_vision_pipeline.py` and find this line near the top:

```python
parser.add_argument("--save-dir", default="/Users/zoey/Documents/2026spring/Autolab/project/captured_images", ...)
```

Either:
- Pass your own path on the command line (`--save-dir /your/path`), or
- Edit the `default=` to a folder that exists on your machine

The folder will be created automatically if it does not exist.

---

## 7. Running the System

You need **two terminal windows open at the same time**.

### Terminal 1 — Start the Dashboard First

```bash
cd /path/to/liverunmonitor
source .venv/bin/activate
streamlit run qc_live_monitor.py
```

Streamlit will print something like:

```
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open `http://localhost:8501` in your browser. You will see the dashboard in **WAITING** state — this is correct. It is waiting for the pipeline to start.

### Terminal 2 — Start the Pipeline

```bash
cd /path/to/liverunmonitor
source .venv/bin/activate
python ot2_vision_pipeline.py --robot-ip 169.254.122.0
```

Common optional flags:

| Flag | Default | When to use |
|---|---|---|
| `--robot-ip` | `169.254.122.0` | Change if your robot has a different IP |
| `--cam-idx` | `0` | Change to `1` or `2` if camera not found |
| `--tolerance` | `15.0` | % deviation allowed before error is flagged |
| `--max-volume` | `400` | Max pipette volume for fill-ratio calculation |
| `--save-dir` | (hardcoded path) | Where to save captured images |
| `--log` | `pipeline_log.csv` | CSV output file name |

Example with custom settings:
```bash
python ot2_vision_pipeline.py \
  --robot-ip 169.254.122.0 \
  --cam-idx 1 \
  --tolerance 10 \
  --save-dir ./images \
  --log my_run.csv
```

---

## 8. What Happens When It Runs

```
Terminal 2 starts                Terminal 1 (browser)
─────────────────                ──────────────────────────────
Robot homes (safe reset)     →   Dashboard shows WAITING

Row A, Well A1:
  Robot aspirates 300 µL
  Moves to camera position
  Image captured               →  Image appears on dashboard
  HSV detection runs           →  Detected volume updates
  QC check passes              →  Status turns GREEN, NOMINAL
  Robot dispenses              →  Wells Completed counter ticks up

Row A, Well A2 ... A12:
  Repeats above                →  Trend chart builds up

All 96 wells done              →  Status turns GREEN, PROTOCOL COMPLETE
```

If a volume error is detected (deviation > tolerance):

```
  Pipeline:                       Dashboard:
  Dispenses back to reservoir  →  Status turns RED, HALTED
  Drops tip                    →  Error well highlighted in audit log
  Stops, prints report         →  Trend chart shows red dot at that well
```

---

## 9. Reading the Dashboard

```
┌──────────────────────────────────────────────────────────┐
│ Status bar    Camera online/offline | Current well        │
│               Detected volume      | Deviation %          │
├──────────────────────────────────────────────────────────┤
│ QC Checks (left)    │  Verdict (right)                    │
│  ✅ Camera          │  🟢 NOMINAL  — all checks pass       │
│  ✅ Image intensity │  — or —                              │
│  ✅ Coordinate logic│  🔴 QC FAIL  — see warnings below   │
│  ✅ Tip width       │                                      │
│  ✅ Volume in range │                                      │
├──────────────────────────────────────────────────────────┤
│ Volume gauge   │  Model stats  │  Annotated image          │
├──────────────────────────────────────────────────────────┤
│ Volume trend chart (last 50 wells)                        │
├──────────────────────────────────────────────────────────┤
│ Audit log table (last 20 rows)                            │
└──────────────────────────────────────────────────────────┘
```

**Sidebar checkboxes** — before starting, an operator should tick all four boxes to confirm the physical setup is correct. The verdict panel will show PENDING until these are checked.

**Emergency Halt button** — does not stop the robot automatically (the pipeline process is separate). If you click it, immediately go to Terminal 2 and press `Ctrl+C`.

---

## 10. After the Run

Results are saved to two files in the project folder:

| File | Contents |
|---|---|
| `pipeline_log.csv` | One row per well: timestamp, well ID, expected volume, detected volume, deviation, error flag |
| `live_state.json` | Last known state of the pipeline (used by the dashboard) |
| `./images/*.jpg` | Raw captured images |
| `./images/*_auto_annotated.jpg` | Same images with tip + liquid boxes drawn on |

To open the CSV in Excel or Numbers, just double-click it. Each run appends to the same file unless you delete it or change `--log`.

---

## 11. Stopping Everything

1. Press `Ctrl+C` in Terminal 2 (pipeline) first
2. Press `Ctrl+C` in Terminal 1 (dashboard) when done reviewing
3. If the robot is mid-run when you stop, it will stay in place. Power-cycle it or run the pipeline again — it runs a safe reset at startup that homes the robot and drops any held tip.

---

## Troubleshooting

**Dashboard shows "No image yet" the whole time**
- Check that `--save-dir` in the pipeline matches a real path your user can write to
- Check that `state_bridge.py` is in the same folder as `ot2_vision_pipeline.py`

**"Cannot open camera index 0"**
- Try `--cam-idx 1`
- On macOS Ventura+: grant Terminal camera permission in System Settings

**"Connection refused" to robot IP**
- Run `ping 169.254.122.0` — if it fails, the network adapter is not configured
- Make sure only one network adapter is active (disable Wi-Fi if needed)

**Dashboard does not auto-refresh**
- Install the refresh package: `pip install streamlit-autorefresh`
- Or click the Manual Refresh button that appears as a fallback

**`ModuleNotFoundError: No module named 'opentrons'`**
- The `opentrons` package is large. Install it specifically: `pip install opentrons`
- On Apple Silicon (M1/M2/M3): `pip install opentrons --no-binary :all:` may be needed

**Streamlit port already in use**
- Another instance is running. Kill it: `pkill -f streamlit`
- Or run on a different port: `streamlit run qc_live_monitor.py --server.port 8502`

---

## Quick-Start Cheat Sheet

```bash
# One-time setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Every run — open two terminals, activate .venv in both
# Terminal 1:
streamlit run qc_live_monitor.py

# Terminal 2:
python ot2_vision_pipeline.py --robot-ip 169.254.122.0

# Open browser → http://localhost:8501
```
