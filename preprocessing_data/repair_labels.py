"""Back up and repair out-of-frame/degenerate YOLO boxes without relabeling classes."""

import argparse
import hashlib
import math
import os
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.data import (BOX_BORDER_TOLERANCE_PIXELS, dataset_config, label_path,
                           split_images, write_json)


def repair_text(content, image_size, n_classes):
    """Keep valid rows verbatim; clip beyond tolerance and remove zero-area rows."""
    image_width, image_height = image_size
    rows, changes = [], []
    for number, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            rows.append(line)
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 5 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Malformed annotation on line {number}")
        category, x, y, width, height = values
        if not category.is_integer() or not 0 <= category < n_classes:
            raise ValueError(f"Unknown class on line {number}")
        if not 0 <= x <= 1 or not 0 <= y <= 1 or width < 0 or height < 0:
            raise ValueError(f"Invalid center/negative size on line {number}; manual review required")
        left, top, right, bottom = x - width / 2, y - height / 2, x + width / 2, y + height / 2
        clipped = (max(0., left), max(0., top), min(1., right), min(1., bottom))
        if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
            changes.append({"line": number, "action": "remove_zero_area", "before": line, "after": None})
            continue
        overflow = max(0, -left * image_width, -top * image_height,
                       (right - 1) * image_width, (bottom - 1) * image_height)
        if overflow > BOX_BORDER_TOLERANCE_PIXELS:
            left, top, right, bottom = clipped
            after = f"{int(category)} " + " ".join(f"{value:.12f}" for value in
                [(left + right) / 2, (top + bottom) / 2, right - left, bottom - top])
            changes.append({"line": number, "action": "clip_to_image", "overflow_pixels": overflow,
                            "before": line, "after": after})
            rows.append(after)
        else:
            rows.append(line)
    return ("\n".join(rows) + ("\n" if rows else "")) if changes else content, changes


def repair(data, output, apply=False):
    config, root, names = dataset_config(data)
    output = Path(output).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Backup/report directory must be separate from the dataset")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    planned, seen_labels = [], set()
    scanned = 0
    for split in (split for split in ["train", "val", "test"] if split in config):
        for image_path in split_images(config, root, split):
            label = label_path(image_path)
            if label in seen_labels:
                raise ValueError("Multiple images share the same label path")
            seen_labels.add(label)
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Unreadable image: {image_path.relative_to(root)}")
            original = label.read_bytes()
            try:
                corrected, changes = repair_text(original.decode("utf-8"),
                                                 (image.shape[1], image.shape[0]), len(names))
            except ValueError as error:
                raise ValueError(f"{label.relative_to(root)}: {error}") from error
            scanned += 1
            if changes:
                planned.append({"label": label, "original": original, "corrected": corrected.encode(),
                                "entry": {"split": split, "label": label.relative_to(root).as_posix(),
                                          "before_sha256": hashlib.sha256(original).hexdigest(),
                                          "after_sha256": hashlib.sha256(corrected.encode()).hexdigest(),
                                          "changes": changes}})
    report = {"status": "planned", "names": names, "images_scanned": scanned,
              "modified_files": len(planned),
              "clipped_boxes": sum(change["action"] == "clip_to_image" for item in planned
                                   for change in item["entry"]["changes"]),
              "removed_zero_area_boxes": sum(change["action"] == "remove_zero_area" for item in planned
                                             for change in item["entry"]["changes"]),
              "files": [item["entry"] for item in planned]}
    write_json(output / "repair.json", report)
    if apply:
        # All inputs are validated and backed up before the first annotation is replaced.
        for item in planned:
            backup = output / "original_labels" / item["entry"]["label"]
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(item["original"])
        for item in planned:
            if item["label"].read_bytes() != item["original"]:
                raise RuntimeError("Labels changed during repair; stop and review backups")
        for item in planned:
            temporary = item["label"].with_suffix(".repair-tmp")
            with temporary.open("xb") as handle:
                handle.write(item["corrected"])
            os.replace(temporary, item["label"])
        report["status"] = "applied"
        write_json(output / "repair.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="configs/data_urpc_raw.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Apply after saving originals; otherwise report only")
    args = parser.parse_args()
    report = repair(args.data, args.output, args.apply)
    print({key: value for key, value in report.items() if key != "files"})


if __name__ == "__main__":
    main()
