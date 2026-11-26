from pathlib import Path
import shutil
import subprocess
import sys

from pipeline.data import load_dataset

PROJECT = Path(__file__).resolve().parents[1]


def test_prepare_copies_and_refuses_overwrite(tmp_path):
    script = tmp_path / "preprocessing_data/scripts.sh"
    script.parent.mkdir()
    shutil.copy(PROJECT / "preprocessing_data/scripts.sh", script)
    for split in ["train", "valid", "test"]:
        for kind in ["images", "labels"]:
            directory = tmp_path / "URPC2020" / split / kind
            directory.mkdir(parents=True)
            (directory / "sample.txt").write_text(f"{split}/{kind}")
    subprocess.run(["bash", str(script)], cwd=tmp_path.parent, check=True, capture_output=True)
    source = tmp_path / "URPC2020/valid/labels/sample.txt"
    target = tmp_path / "URPC2020_YOLO/labels/val/sample.txt"
    assert source.read_bytes() == target.read_bytes()
    repeated = subprocess.run(["bash", str(script)], capture_output=True)
    assert repeated.returncode != 0
    assert source.read_bytes() == target.read_bytes()


def test_enhancement_and_portable_yaml(toy_data, tmp_path):
    source = toy_data.parent.parent / "dataset"
    output = tmp_path / "enhanced"
    subprocess.run([sys.executable, str(PROJECT / "preprocessing_data/clahe.py"),
                    "--input", str(source), "--output", str(output), "--splits", "train", "val",
                    "--workers", "1"], check=True, capture_output=True)
    generated = tmp_path / "separate-config/enhanced.yaml"
    subprocess.run([sys.executable, str(PROJECT / "preprocessing_data/create_yaml.py"),
                    "--dataset", str(output), "--output", str(generated), "--names", "a", "b"],
                   check=True, capture_output=True)
    raw, enhanced = load_dataset(toy_data), load_dataset(generated)
    assert len(enhanced.records) == len(raw.records)
    assert raw.records[0]["boxes"] == enhanced.records[0]["boxes"]
    assert raw.records[0]["sha256"] != enhanced.records[0]["sha256"]
    assert (source / "images/val/0.png").is_file()
    assert str(tmp_path) not in generated.read_text()


def test_enhance_from_yaml_supports_split_first_layout(tmp_path):
    import cv2
    import numpy as np
    import yaml

    rng = np.random.default_rng(11)
    raw_root = tmp_path / "archive/URPC2020"
    for split, folder in [("train", "train"), ("val", "valid"), ("test", "test")]:
        (raw_root / folder / "images").mkdir(parents=True)
        (raw_root / folder / "labels").mkdir(parents=True)
        image = rng.integers(10, 240, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(raw_root / folder / "images/0001.jpg"), image)
        (raw_root / folder / "labels/0001.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")

    raw_yaml = tmp_path / "configs/raw.yaml"
    raw_yaml.parent.mkdir(parents=True)
    raw_yaml.write_text(yaml.safe_dump({
        "path": "../archive/URPC2020",
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {0: "holothurian", 1: "echinus", 2: "scallop", 3: "starfish"},
    }, sort_keys=False), encoding="utf-8")

    enhanced_dir = tmp_path / "URPC2020_CLAHE"
    enhanced_yaml = tmp_path / "configs/clahe.yaml"
    subprocess.run(
        [sys.executable, str(PROJECT / "preprocessing_data/clahe.py"),
         "--data", str(raw_yaml), "--output", str(enhanced_dir),
         "--yaml-output", str(enhanced_yaml), "--workers", "1"],
        check=True, capture_output=True, text=True,
    )
    for split in ("train", "val", "test"):
        raw_ds = load_dataset(raw_yaml, split)
        enh_ds = load_dataset(enhanced_yaml, split)
        assert enh_ds.names == ["holothurian", "echinus", "scallop", "starfish"]
        assert len(enh_ds.records) == len(raw_ds.records) == 1
        assert enh_ds.records[0]["boxes"] == raw_ds.records[0]["boxes"]
        assert enh_ds.records[0]["sha256"] != raw_ds.records[0]["sha256"]


def test_prepare_dev_split_removes_cross_split_leakage_without_touching_val_or_test(toy_data, tmp_path):
    root = toy_data.parent.parent / "dataset"
    # Inject exact leakage: train/0.png identical to val/0.png.
    (root / "images/train/0.png").write_bytes((root / "images/val/0.png").read_bytes())

    dev_dir = tmp_path / "dev_dataset"
    dev_yaml = tmp_path / "configs/data_dev.yaml"
    report_md = tmp_path / "reports/data_decisions.md"
    result = subprocess.run(
        [sys.executable, str(PROJECT / "preprocessing_data/prepare_dev_split.py"),
         "--data", str(toy_data), "--output", str(dev_dir),
         "--yaml-output", str(dev_yaml), "--report", str(report_md), "--link-mode", "copy"],
        check=True, capture_output=True, text=True,
    )
    assert "exact_cross_split_duplicate" in result.stdout
    assert report_md.is_file()
    train_dev = load_dataset(dev_yaml, "train")
    val_dev = load_dataset(dev_yaml, "val")
    test_dev = load_dataset(dev_yaml, "test")
    assert len(train_dev.records) == 1
    assert len(val_dev.records) == 2 and len(test_dev.records) == 2
    assert not ({r["sha256"] for r in train_dev.records} & {r["sha256"] for r in val_dev.records})
