# CORRECTED debug_dataset.py - PASTE THIS IF THE TEST WORKS

from pathlib import Path
import yaml
import glob

# The data.yaml is inside the 'vehicles-1' folder!
data_yaml_path = Path(r"C:\Users\HP\Desktop\Project\vehicles-1\data.yaml")

# --- From here the code is the same, but it will now work ---

if not data_yaml_path.exists():
    print(f"ERROR: The file was not found at {data_yaml_path}")
    print("Please check the folder name (e.g., 'vehicles-1') is correct.")
else:
    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    print("✅ Successfully loaded YAML file.")
    print("\n--- YAML Content ---")
    print("Train path in YAML:", data.get("train"))
    print("Val path in YAML:", data.get("val") or data.get("valid"))
    print("Test path in YAML:", data.get("test"))
    print("Class names:", data.get("names"))
    print("--------------------")

    dataset_root = data_yaml_path.parent

    for split in ["train", "val", "test"]:
        # Roboflow sometimes uses 'valid' instead of 'val'
        path_in_yaml = data.get(split) or (data.get("valid") if split == "val" else None)

        if path_in_yaml:
            # Create the full, absolute path
            full_path = (dataset_root / path_in_yaml).resolve()
            
            image_count = 0
            if full_path.exists():
                image_count = len(glob.glob(str(full_path / "*.jpg")))

            print(f"\nChecking split: '{split}'")
            print(f"Path: {full_path}")
            print(f"Exists: {full_path.exists()}")
            print(f"Image Count: {image_count}")