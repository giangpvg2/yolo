"""Generate a portable data YAML for a prepared YOLO dataset."""

import argparse
import os
from pathlib import Path
import yaml

DEFAULT_URPC_CLASSES = ["holothurian", "echinus", "scallop", "starfish"]


def detect_splits(dataset_dir):
    found = {}
    for split, candidates in [
        ("train", ["images/train", "train/images"]),
        ("val", ["images/val", "val/images", "valid/images"]),
        ("test", ["images/test", "test/images"]),
    ]:
        for rel in candidates:
            if (dataset_dir / rel).is_dir():
                found[split] = rel
                break
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("URPC2020_YOLO"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--names", nargs="+", default=DEFAULT_URPC_CLASSES)
    args = parser.parse_args()
    if not args.dataset.is_dir():
        parser.error("Dataset directory not found")
    output = args.output or args.dataset / "data.yaml"
    if output.exists():
        parser.error("Output YAML already exists; choose another --output")
    splits = detect_splits(args.dataset)
    if not splits:
        parser.error("No YOLO image split directories found under --dataset")
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "path": Path(os.path.relpath(args.dataset.resolve(), output.parent.resolve())).as_posix(),
        **splits,
        "names": dict(enumerate(args.names)),
    }
    output.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print("Created:", output)


if __name__ == "__main__":
    main()
