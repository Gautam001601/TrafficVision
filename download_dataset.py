from roboflow import Roboflow
from config import API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, ROBOFLOW_VERSION, save_dataset_path

if not API_KEY:
    raise ValueError(
        "ROBOFLOW_API_KEY not found. Please add it to your .env file."
    )

print("Connecting to Roboflow...")
rf = Roboflow(api_key=API_KEY)

print("Loading project...")
project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)

print("Downloading dataset version...")
dataset = project.version(ROBOFLOW_VERSION).download("yolov8")

save_dataset_path(dataset.location)

print("\nDownload complete!")
print("Dataset location:", dataset.location)
print("data.yaml path:", f"{dataset.location}/data.yaml")