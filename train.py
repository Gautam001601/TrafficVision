from pathlib import Path
from ultralytics import YOLO

from config import (
    MODEL_NAME,
    RUN_NAME,
    DEVICE,
    EPOCHS,
    IMGSZ,
    BATCH,
    WORKERS,
    get_data_yaml_path,
    ARTIFACTS_DIR,
)

def main():
    data_yaml = get_data_yaml_path()

    print("Training with:")
    print("Model:", MODEL_NAME)
    print("Data:", data_yaml)
    print("Device:", DEVICE)
    print("Epochs:", EPOCHS)
    print("Image size:", IMGSZ)
    print("Batch size:", BATCH)

    # YOLOv8 automatically handles preprocessing and feature extraction internally
    model = YOLO(MODEL_NAME)

    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        name=RUN_NAME,
        project="runs/detect",
        patience=10,
        save=True,
        plots=True
    )

    best_model_path = Path("runs") / "detect" / RUN_NAME / "weights" / "best.pt"

    print("\nTraining finished.")
    print("Best model path:", best_model_path)

    if best_model_path.exists():
        (ARTIFACTS_DIR / "best_model_path.txt").write_text(str(best_model_path))
        print("Best model path saved to artifacts/best_model_path.txt")
    else:
        print("Warning: best.pt not found.")

    return results


if __name__ == "__main__":
    main()