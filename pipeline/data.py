"""Strict YOLO detection dataset loading, independent of Ultralytics settings."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math

import cv2
import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
BOX_BORDER_TOLERANCE_PIXELS = 0.01


def read_yaml(path):
    with Path(path).open(encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a YAML mapping: {Path(path).name}")
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def class_names(config):
    names = config.get("names")
    if isinstance(names, dict):
        names = {int(key): str(value) for key, value in names.items()}
        if sorted(names) != list(range(len(names))):
            raise ValueError("Class IDs must be consecutive, starting at zero")
        names = [names[index] for index in range(len(names))]
    if not isinstance(names, list) or not names or not all(isinstance(x, str) for x in names):
        raise ValueError("Data YAML must contain a nonempty names list or ID mapping")
    if len(set(names)) != len(names):
        raise ValueError("Class names must be unique")
    return names


def dataset_config(path):
    path = Path(path).resolve()
    config = read_yaml(path)
    root = (path.parent / config.get("path", ".")).resolve()
    return config, root, class_names(config)


def split_images(config, root, split):
    value = config.get(split)
    if not isinstance(value, str):
        raise ValueError(f"Split {split!r} must name an image directory")
    directory = (root / value).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing image directory for split {split!r}: {value}")
    paths = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"No images found in split {split!r}")
    for path in paths:
        if not path.resolve().is_relative_to(root):
            raise ValueError("Image paths must remain inside the dataset root")
    return paths


def label_path(image_path):
    parts = list(image_path.parts)
    positions = [index for index, part in enumerate(parts) if part == "images"]
    if not positions:
        raise ValueError("Dataset image paths must contain an images directory")
    parts[positions[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_boxes(path, n_classes, allow_missing=False, image_size=None):
    if not path.exists():
        if allow_missing:
            return None
        raise FileNotFoundError(f"Missing label: {path.name}; empty negatives need an empty .txt")
    boxes = []
    tolerance_x, tolerance_y = ((BOX_BORDER_TOLERANCE_PIXELS / image_size[0],
                                BOX_BORDER_TOLERANCE_PIXELS / image_size[1])
                               if image_size else (1e-6, 1e-6))
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = [float(value) for value in line.split()]
        except ValueError as error:
            raise ValueError(f"Invalid numeric label: {path.name}:{line_number}") from error
        if len(values) != 5 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Expected class xc yc w h: {path.name}:{line_number}")
        category, x, y, width, height = values
        valid_class = category.is_integer() and 0 <= category < n_classes
        valid_box = (0 < width <= 1 and 0 < height <= 1 and 0 <= x <= 1 and 0 <= y <= 1 and
                     x - width / 2 >= -tolerance_x and y - height / 2 >= -tolerance_y and
                     x + width / 2 <= 1 + tolerance_x and y + height / 2 <= 1 + tolerance_y)
        if not valid_class or not valid_box:
            raise ValueError(f"Invalid class or normalized box: {path.name}:{line_number}")
        box = (int(category), x, y, width, height)
        if box in boxes:
            raise ValueError(f"Duplicate label: {path.name}:{line_number}")
        boxes.append(box)
    return boxes


@dataclass
class Dataset:
    root: Path
    names: list
    split: str
    records: list

    def coco(self):
        images, annotations = [], []
        for record in self.records:
            images.append({key: record[key] for key in ("id", "file_name", "width", "height", "sha256")})
            for category, x, y, width, height in record["boxes"]:
                w, h = width * record["width"], height * record["height"]
                annotations.append({
                    "id": len(annotations) + 1, "image_id": record["id"],
                    "category_id": category, "bbox": [(x - width / 2) * record["width"],
                                                       (y - height / 2) * record["height"], w, h],
                    "area": w * h, "iscrowd": 0,
                })
        return {"info": {"description": "YOLO labels in original image coordinates"},
                "images": images, "annotations": annotations,
                "categories": [{"id": i, "name": name} for i, name in enumerate(self.names)]}

    def fingerprint(self):
        content = json.dumps(self.coco(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(content.encode()).hexdigest()

    def geometry_fingerprint(self):
        payload = {
            "split": self.split,
            "names": self.names,
            "records": [(r["file_name"], r["width"], r["height"], r["boxes"]) for r in self.records],
        }
        content = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(content.encode()).hexdigest()


def load_dataset(path, split="val", allow_missing=False, validation_issues=None):
    """Load strictly, or collect failed images for audit only when a list is supplied."""
    config, root, names = dataset_config(path)
    records = []
    label_paths = set()
    for image_path in split_images(config, root, split):
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Unreadable image: {image_path.relative_to(root)}")
            annotation_path = label_path(image_path)
            if annotation_path in label_paths:
                raise ValueError("Multiple image extensions share the same label path")
            label_paths.add(annotation_path)
            boxes = read_boxes(annotation_path, len(names), allow_missing, (image.shape[1], image.shape[0]))
        except (ValueError, FileNotFoundError) as error:
            if validation_issues is None:
                raise
            validation_issues.append({"file_name": image_path.relative_to(root).as_posix(),
                                      "error": str(error)})
            continue
        records.append({"id": len(records) + 1, "path": image_path,
                        "file_name": image_path.relative_to(root).as_posix(),
                        "width": image.shape[1], "height": image.shape[0],
                        "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                        "border_rounding_boxes": sum(
                            x - w / 2 < 0 or y - h / 2 < 0 or x + w / 2 > 1 or y + h / 2 > 1
                            for _, x, y, w, h in boxes or []),
                        "missing_label": boxes is None, "boxes": boxes or []})
    return Dataset(root, names, split, records)
