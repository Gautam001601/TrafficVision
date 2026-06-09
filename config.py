import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Roboflow project info
ROBOFLOW_WORKSPACE = "sumans-workspace-7z4u0"
ROBOFLOW_PROJECT = "vehicles-q0x2v-2eo9h"
ROBOFLOW_VERSION = 1

# Use YOLOv8 Nano
MODEL_NAME = "yolov8n.pt"
RUN_NAME = "vehicle_detection_nano-3"

# CPU training
DEVICE = "0"
EPOCHS = 50
IMGSZ = 640
BATCH = 8
WORKERS = 0

# Paths
ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

DATASET_PATH_FILE = ARTIFACTS_DIR / "dataset_path.txt"

API_KEY = os.getenv("ROBOFLOW_API_KEY")


def save_dataset_path(path: str):
    DATASET_PATH_FILE.write_text(str(path))


def load_dataset_path():
    if DATASET_PATH_FILE.exists():
        return DATASET_PATH_FILE.read_text().strip()
    return None


def get_dataset_path():
    path = load_dataset_path()
    if not path:
        raise FileNotFoundError(
            "Dataset path not found. Please run download_dataset.py first."
        )
    return Path(path)


def get_data_yaml_path():
    return get_dataset_path() / "data.yaml"