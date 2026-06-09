import argparse
import glob
from pathlib import Path

from ultralytics import YOLO
from config import RUN_NAME, DEVICE, get_dataset_path, ARTIFACTS_DIR

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Image file, folder, or video path"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="test_predictions",
        help="Output folder name"
    )
    args = parser.parse_args()

    dataset_path = get_dataset_path()
    default_source = str(dataset_path / "test" / "images")

    source = args.source if args.source else default_source

    best_model_path = find_best_model()
    print("Loading model:", best_model_path)

    model = YOLO(str(best_model_path))

    results = model.predict(
        source=source,
        conf=args.conf,
        device=DEVICE,
        save=True,
        project="runs/detect",
        name=args.name,
        verbose=False
    )

    print("\nPrediction complete.")
    if results:
        print("Saved to:", results[0].save_dir)


if __name__ == "__main__":
    main()