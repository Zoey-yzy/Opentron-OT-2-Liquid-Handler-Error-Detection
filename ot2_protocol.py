from __future__ import annotations
import argparse
import logging
import time
from pathlib import Path
import cv2
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("ot2_protocol")
logging.basicConfig(level=logging.INFO)

HEADERS = {"Opentrons-Version": "*"} 

class OT2Client:
    def __init__(self, robot_ip: str, port: int = 31950, timeout: float = 120.0):
        self._base_url = f"http://{robot_ip}:{port}"
        self._timeout = timeout
        self._session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[502, 503, 504])
        self._session.mount("http://", HTTPAdapter(max_retries=retry))
        self._session.headers.update(HEADERS)
        self._run_id = None
        self._commands_url = None
        self._pipette_id = None

    def set_lights(self, on: bool):
        """Turn the OT-2 rail lights on or off."""
        r = self._session.post(
            f"{self._base_url}/robot/lights",
            json={"on": on},
            timeout=self._timeout,
        )
        r.raise_for_status()
        logger.info(f"Lights {'ON' if on else 'OFF'}.")

    def home(self):
        """Move all axes to their home position. Call this before starting a run
        to recover from any interrupted or unknown robot state."""
        r = self._session.post(
            f"{self._base_url}/robot/home",
            json={"target": "robot"},
            timeout=self._timeout,
        )
        r.raise_for_status()
        logger.info("Robot homed successfully.")

    def create_run(self) -> str:
        r = self._session.post(f"{self._base_url}/runs", json={"data": {}}, timeout=self._timeout)
        r.raise_for_status()
        self._run_id = r.json()["data"]["id"]
        self._commands_url = f"{self._base_url}/runs/{self._run_id}/commands"
        logger.info(f"Run created: {self._run_id}")
        return self._run_id

    def _post_command(self, cmd: dict) -> dict:
        if not self._commands_url:
            raise RuntimeError("No active run.")
        if "data" in cmd:
            cmd["data"]["intent"] = "setup" 
        r = self._session.post(self._commands_url, json=cmd, params={"waitUntilComplete": True}, timeout=self._timeout)
        r.raise_for_status()
        return r.json().get("data", {})

    def load_pipette(self, pipette_name: str, mount: str) -> str:
        cmd = {"data": {"commandType": "loadPipette", "params": {"pipetteName": pipette_name, "mount": mount}}}
        data = self._post_command(cmd)
        self._pipette_id = data["result"]["pipetteId"]
        return self._pipette_id

    def load_labware(self, load_name: str, slot: str, version: int = 1) -> str:
        cmd = {"data": {"commandType": "loadLabware", "params": {"location": {"slotName": slot}, "loadName": load_name, "namespace": "opentrons", "version": version}}}
        data = self._post_command(cmd)
        return data["result"]["labwareId"]

    def pick_up_tip(self, labware_id: str, well: str):
        self._post_command({"data": {"commandType": "pickUpTip", "params": {"pipetteId": self._pipette_id, "labwareId": labware_id, "wellName": well}}})

    def aspirate(self, volume: float, labware_id: str, well: str):
        self._post_command({
            "data": {
                "commandType": "aspirate", 
                "params": {
                    "pipetteId": self._pipette_id, 
                    "labwareId": labware_id, 
                    "wellName": well, 
                    "volume": volume, 
                    "flowRate": 46.4, 
                    "wellLocation": {"origin": "bottom", "offset": {"z": 2.5}}
                }
            }
        })
        time.sleep(3.0) # Settle time for accuracy

    def dispense(self, volume: float, labware_id: str, well: str):
        self._post_command({
            "data": {
                "commandType": "dispense", 
                "params": {
                    "pipetteId": self._pipette_id, 
                    "labwareId": labware_id, 
                    "wellName": well, 
                    "volume": volume, 
                    "flowRate": 46.4, 
                    "wellLocation": {"origin": "bottom", "offset": {"z": 1.5}}
                }
            }
        })

    def drop_tip(self):
        self._post_command({"data": {"commandType": "moveToAddressableAreaForDropTip", "params": {"pipetteId": self._pipette_id, "addressableAreaName": "fixedTrash"}}})
        self._post_command({"data": {"commandType": "dropTipInPlace", "params": {"pipetteId": self._pipette_id}}})

    def pose_for_camera(self, labware_id: str, well: str):
        self._post_command({
            "data": {
                "commandType": "moveToWell",
                "params": {
                    "pipetteId": self._pipette_id,
                    "labwareId": labware_id,
                    "wellName": well,
                    "wellLocation": {"origin": "top", "offset": {"z": 60}}
                }
            }
        })

def take_liquid_picture(well: str, vol: float, cam_idx: int = 1):
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return
    for _ in range(20):
        cap.read()
    ret, frame = cap.read()
    if ret:
        Path("captured_samples").mkdir(exist_ok=True)
        # Naming: captured_samples/Row_A_Well_1_500.0uL.jpg
        filename = f"captured_samples/Row_{well[0]}_Well_{well[1:]}_{vol}uL.jpg"
        cv2.imwrite(filename, frame)
    cap.release()

def run_protocol(robot_ip: str):
    robot = OT2Client(robot_ip)
    robot.create_run()

    robot.load_pipette("p1000_single_gen2", "right")
    tips = robot.load_labware("opentrons_96_tiprack_1000ul", "2")
    res = robot.load_labware("nest_1_reservoir_290ml", "4")
    # Make sure this is a deep-well plate if you are doing 600uL!
    plate = robot.load_labware("nest_96_wellplate_200ul_flat", "5", version=5)

    # Volumes mapped to Row letters
    rows_to_process = [
        ("A", 300.0), ("B", 325.0), ("C", 350.0), ("D", 375.0),
        ("E", 400.0), ("F", 275.0), ("G", 250.0), ("H", 225.0),
    ]

    for row_letter, volume in rows_to_process:
        logger.info(f"--- Processing Full Row {row_letter} at {volume} uL ---")
        
        # Pick up a fresh tip for the start of the row
        robot.pick_up_tip(tips, f"{row_letter}1")

        for col in range(1, 13):
            well = f"{row_letter}{col}"
            
            # 1. Aspirate from Reservoir
            robot.aspirate(volume, res, "A1")
            
            # 2. Photo
            robot.pose_for_camera(res, "A1")
            take_liquid_picture(well, volume, cam_idx=1)
            
            # 3. Dispense into Plate
            robot.dispense(volume, plate, well)
            
        # Drop tip after finishing all 12 wells in the row
        robot.drop_tip()

    logger.info("Protocol finished: All rows processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-ip", required=True)
    args = parser.parse_args()
    run_protocol(args.robot_ip)