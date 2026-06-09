from pathlib import Path
from ultralytics import YOLO

from config import RUN_NAME, DEVICE, get_data_yaml_path, ARTIFACTS_DIR

def find_best_model():
    # 1. Check saved path from previous runs
    saved_path = ARTIFACTS_DIR / "best_model_path.txt"
    if saved_path.exists():
        p = Path(saved_path.read_text().strip())
        if p.exists():
            return p

    # 2. Standard YOLO path
    standard_path = Path("runs") / "detect" / RUN_NAME / "weights" / "best.pt"
    if standard_path.exists():
        return standard_path

    # 3. Nested path (matches your actual file structure)
    nested_path = Path("runs") / "detect" / "runs" / "detect" / RUN_NAME / "weights" / "best.pt"
    if nested_path.exists():
        return nested_path

    # 4. Clear error if nothing is found
    raise FileNotFoundError(
        f"best.pt not found. Checked:\n"
        f"  • {saved_path}\n"
        f"  • {standard_path}\n"
        f"  • {nested_path}\n"
        f"Train the model first or verify RUN_NAME in config.py."
    )

def main():
    data_yaml = get_data_yaml_path()
    best_model_path = find_best_model()

    print("Loading model:", best_model_path)
    model = YOLO(str(best_model_path))

    print("\nRunning validation on validation set...")
    val_metrics = model.val(data=str(data_yaml), split="val", device=DEVICE)
    print("\nValidation results:")
    print(val_metrics)

    print("\nRunning validation on test set...")
    try:
        test_metrics = model.val(data=str(data_yaml), split="test", device=DEVICE)
        print("\nTest results:")
        print(test_metrics)
    except Exception as e:
        print("\nTest split validation failed or not available.")
        print("Error:", e)


if __name__ == "__main__":
    main()