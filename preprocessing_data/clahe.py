"""Generate a Gray World + CLAHE dataset from either a YOLO directory or a data YAML."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from tqdm import tqdm
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.data import IMAGE_SUFFIXES, dataset_config, label_path, split_images


def color_correction(img):
    """Gray World white balance with gain cap (2.5x) to avoid red channel blow-out in deep water."""
    b, g, r = cv2.split(img.astype(np.float32))
    avg_b, avg_g, avg_r = max(float(np.mean(b)), 1.0), max(float(np.mean(g)), 1.0), max(float(np.mean(r)), 1.0)
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    kb, kg, kr = min(avg_gray / avg_b, 2.5), min(avg_gray / avg_g, 2.5), min(avg_gray / avg_r, 2.5)
    b = np.clip(b * kb, 0, 255)
    g = np.clip(g * kg, 0, 255)
    r = np.clip(r * kr, 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)


def apply_clahe(img):
    """Apply fixed-parameter CLAHE on the L channel in LAB space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def process_single_image(args):
    in_path, out_path = args
    img = cv2.imread(in_path)
    if img is None:
        raise ValueError(f"Unreadable input image: {Path(in_path).name}")
    enhanced = apply_clahe(color_correction(img))
    if not cv2.imwrite(out_path, enhanced):
        raise OSError(f"Could not write image: {Path(out_path).name}")


def process_dataset(input_folder, output_folder, workers=4):
    input_folder, output_folder = Path(input_folder), Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    image_files = sorted(path for path in input_folder.rglob("*")
                         if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_files:
        raise ValueError("Input split contains no supported images")
    tasks = []
    for filename in image_files:
        out_path = output_folder / filename.relative_to(input_folder)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tasks.append((str(filename), str(out_path)))
    print(f"\n[Enhance] Processing {len(image_files)} images from: {input_folder}")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(tqdm(executor.map(process_single_image, tasks), total=len(tasks), desc="Progress"))


def write_portable_yaml(dataset_dir, output_yaml, splits, names):
    output_yaml = Path(output_yaml)
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    rel_path = Path(os.path.relpath(Path(dataset_dir).resolve(), output_yaml.parent.resolve())).as_posix()
    payload = {
        "path": rel_path,
        **{split: f"images/{split}" for split in splits},
        "names": dict(enumerate(names)),
    }
    output_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def enhance_from_yaml(data_yaml, dst_root, requested_splits=None, workers=4, yaml_output=None):
    config, src_root, names = dataset_config(data_yaml)
    if dst_root.exists() or dst_root.is_relative_to(src_root) or src_root.is_relative_to(dst_root):
        raise ValueError("Output must be new and separate from the input dataset")
    splits = requested_splits or [split for split in ("train", "val", "test") if split in config]
    if not splits:
        raise ValueError("No valid splits found in data YAML")
    split_plans = []
    for split in splits:
        if split not in {"train", "val", "test"} or split not in config:
            raise ValueError(f"Split {split!r} is missing from {Path(data_yaml).name}")
        split_img_root = (src_root / config[split]).resolve()
        images = split_images(config, src_root, split)
        pairs = []
        for img_path in images:
            src_label = label_path(img_path)
            if not src_label.is_file():
                raise FileNotFoundError(f"Missing label for {img_path.relative_to(src_root)}")
            rel = img_path.resolve().relative_to(split_img_root)
            pairs.append((img_path, src_label, rel))
        split_plans.append((split, pairs))

    for split, pairs in split_plans:
        tasks = []
        for img_path, src_label, rel in pairs:
            dst_img = dst_root / "images" / split / rel
            dst_lbl = (dst_root / "labels" / split / rel).with_suffix(".txt")
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            tasks.append((str(img_path), str(dst_img)))
            shutil.copy2(src_label, dst_lbl)
        print(f"\n[Enhance] Processing {len(tasks)} images for split '{split}'")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(tqdm(executor.map(process_single_image, tasks), total=len(tasks), desc=f"Progress ({split})"))

    write_portable_yaml(dst_root, dst_root / "data.yaml", splits, names)
    if yaml_output is not None:
        write_portable_yaml(dst_root, yaml_output, splits, names)


def main():
    parser = argparse.ArgumentParser(description="Gray World color correction followed by CLAHE")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--data", type=Path, help="Data YAML (supports both <split>/images and images/<split>)")
    source_group.add_argument("--input", type=Path, help="Prepared YOLO root with images/<split> and labels/<split>")
    parser.add_argument("--output", type=Path, default=Path("URPC2020_CLAHE"))
    parser.add_argument("--yaml-output", type=Path, help="Optional portable data YAML path to write")
    parser.add_argument("--splits", nargs="+")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or (args.splits and len(set(args.splits)) != len(args.splits)):
        parser.error("workers must be positive; splits must be unique")
    if args.yaml_output and args.yaml_output.exists():
        parser.error("Output YAML already exists; choose a new --yaml-output path")
    dst_root = args.output.resolve()

    if args.data is not None:
        try:
            enhance_from_yaml(args.data, dst_root, args.splits, args.workers, args.yaml_output)
        except (ValueError, FileNotFoundError, OSError) as error:
            parser.error(str(error))
        return

    input_path = args.input or Path("URPC2020_YOLO")
    splits = args.splits or ["train", "val", "test"]
    src_root = input_path.resolve()
    if dst_root.exists() or dst_root.is_relative_to(src_root) or src_root.is_relative_to(dst_root):
        parser.error("Output must be new and separate from the input dataset")
    for folder_name in splits:
        if folder_name not in {"train", "val", "test"}:
            parser.error("Only train, val and test splits are supported")
        if not (src_root / "images" / folder_name).is_dir() or not (src_root / "labels" / folder_name).is_dir():
            parser.error(f"Missing images/labels for split {folder_name}")

    for folder_name in splits:
        process_dataset(src_root / "images" / folder_name, dst_root / "images" / folder_name, args.workers)
        shutil.copytree(src_root / "labels" / folder_name, dst_root / "labels" / folder_name)


if __name__ == "__main__":
    main()
