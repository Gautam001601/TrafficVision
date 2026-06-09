import random
from pathlib import Path

import cv2
import yaml
from config import get_data_yaml_path, ARTIFACTS_DIR

PREVIEW_DIR = ARTIFACTS_DIR / "previews"
PREVIEW_DIR.mkdir(exist_ok=True)


def resolve_path(base_dir: Path, p):
    p = Path(p)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def count_images(folder: Path):
    if not folder.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in exts)


def image_to_label_path(image_path: Path):
    # Roboflow YOLO format: images/... -> labels/...
    label_path = str(image_path).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
    return Path(label_path).with_suffix(".txt")


def draw_boxes(image_path: Path, label_path: Path, class_names):
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    h, w = img.shape[:2]

    if not label_path.exists():
        return img

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id, x_center, y_center, box_w, box_h = map(float, parts)
        class_id = int(class_id)

        x1 = int((x_center - box_w / 2) * w)
        y1 = int((y_center - box_h / 2) * h)
        x2 = int((x_center + box_w / 2) * w)
        y2 = int((y_center + box_h / 2) * h)

        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = class_names.get(class_id, str(class_id))
        cv2.putText(
            img,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return img


def main():
    data_yaml_path = get_data_yaml_path()
    base_dir = data_yaml_path.parent

    data = load_yaml(data_yaml_path)
    class_names = data.get("names", {})
    print("\nClass names:")
    print(class_names)

    splits = {
        "train": data.get("train"),
        "val": data.get("val") or data.get("valid"),
        "test": data.get("test"),
    }

    print("\nDataset summary:")
    for split_name, split_path in splits.items():
        if split_path is None:
            print(f"{split_name}: not found in data.yaml")
            continue

        split_dir = resolve_path(base_dir, split_path)
        image_count = count_images(split_dir)
        print(f"{split_name}: {image_count} images -> {split_dir}")

    # Random sample from train set for preview
    train_path = splits["train"]
    if train_path:
        train_dir = resolve_path(base_dir, train_path)
        images = [p for p in train_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

        if images:
            sample_img = random.choice(images)
            sample_lbl = image_to_label_path(sample_img)

            preview = draw_boxes(sample_img, sample_lbl, class_names)
            if preview is not None:
                out_path = PREVIEW_DIR / "sample_preview.jpg"
                cv2.imwrite(str(out_path), preview)
                print(f"\nPreview saved to: {out_path}")
        else:
            print("\nNo train images found for preview.")

    print("\nPreprocessing/inspection completed.")


if __name__ == "__main__":
    main()