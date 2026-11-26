from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from dataset_audit import HashIndex, audit, image_scores
from pipeline.data import load_dataset, read_yaml
from pipeline.evaluation import count_errors, error_curve, evaluate, validate_predictions
import train as train_module

AUDIT_CONFIG = Path(__file__).resolve().parents[1] / "configs/audit.yaml"


def test_paths_are_relative_to_yaml(toy_data, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path.parent)
    dataset = load_dataset(toy_data)
    assert len(dataset.records) == 2
    assert dataset.coco()["annotations"][0]["bbox"] == [54, 54, 20, 20]
    assert dataset.coco()["annotations"][0]["area"] == 400


@pytest.mark.parametrize("label", ["0 nan .5 .1 .1", "2 .5 .5 .1 .1", "0.5 .5 .5 .1 .1",
                                   "0 .5 .5 0 .1", "0 .01 .5 .5 .5", "0 .5 .5 1.1 .1",
                                   "0 .5 .5 .1 .1 8", "not a label"])
def test_invalid_labels_fail(toy_data, label):
    (toy_data.parent.parent / "dataset/labels/val/0.txt").write_text(label)
    with pytest.raises(ValueError):
        load_dataset(toy_data)


def test_missing_and_empty_labels_are_distinct(toy_data):
    label = toy_data.parent.parent / "dataset/labels/val/0.txt"
    label.unlink()
    with pytest.raises(FileNotFoundError):
        load_dataset(toy_data)
    assert load_dataset(toy_data, allow_missing=True).records[0]["missing_label"]
    label.write_text("")
    record = load_dataset(toy_data).records[0]
    assert record["boxes"] == [] and not record["missing_label"]


def test_subpixel_border_rounding_is_preserved_but_larger_errors_fail(toy_data):
    label = toy_data.parent.parent / "dataset/labels/val/0.txt"
    x = 0.95 + 0.005 / 128
    label.write_text(f"0 {x} 0.5 0.1 0.1\n")
    record = load_dataset(toy_data).records[0]
    assert record["boxes"][0][1] == x
    assert record["border_rounding_boxes"] == 1
    label.write_text(f"0 {0.95 + 0.05 / 128} 0.5 0.1 0.1\n")
    with pytest.raises(ValueError):
        load_dataset(toy_data)


def test_image_content_is_part_of_fingerprint(toy_data):
    before = load_dataset(toy_data).fingerprint()
    path = toy_data.parent.parent / "dataset/images/val/0.png"
    cv2.imwrite(str(path), np.zeros((128, 128, 3), dtype=np.uint8))
    assert load_dataset(toy_data).fingerprint() != before


def test_perfect_coco_and_error_counts(dataset, perfect_predictions, tmp_path):
    original_predictions = deepcopy(perfect_predictions)
    original_truth = dataset.coco()
    rows = evaluate(dataset, perfect_predictions, tmp_path / "metrics")
    assert rows[0]["AP_small"] == pytest.approx(1)
    assert rows[0]["AR_small"] == pytest.approx(1)
    assert rows[0]["FP_per_image"] == 0
    assert rows[0]["FN_small"] == 0
    assert (tmp_path / "metrics/metrics.csv").is_file()
    assert perfect_predictions == original_predictions
    assert json.loads((tmp_path / "metrics/ground_truth.json").read_text()) == original_truth


def test_empty_predictions_count_all_small_as_missed(dataset, tmp_path):
    row = evaluate(dataset, [], tmp_path / "metrics")[0]
    assert row["AP_small"] == 0 and row["AR_small"] == 0
    assert row["FP_per_image"] == 0 and row["FN_small"] == 2


def test_wrong_class_and_duplicates_are_fp(dataset, perfect_predictions):
    predictions = deepcopy(perfect_predictions)
    predictions.append(deepcopy(predictions[0]))
    predictions[1]["category_id"] = 0
    counts = count_errors(dataset.coco(), predictions)
    assert sum(row["tp"] for row in counts) == 1
    assert sum(row["fp"] for row in counts) == 2
    assert sum(row["fn_small"] for row in counts) == 1


def test_confidence_filter_only_changes_error_counts(dataset, perfect_predictions, tmp_path):
    perfect_predictions[0]["score"] = 0.1
    row = evaluate(dataset, perfect_predictions, tmp_path / "metrics", confidence=0.25)[0]
    assert row["AP_small"] == pytest.approx(1)
    assert row["FN_small"] == 1


def test_froc_agrees_with_thresholded_matching(dataset, perfect_predictions):
    perfect_predictions[0]["score"] = 0.18
    duplicate = deepcopy(perfect_predictions[1])
    duplicate["score"] = 0.5
    perfect_predictions.append(duplicate)
    for row in error_curve(dataset.coco(), perfect_predictions):
        errors = count_errors(dataset.coco(), perfect_predictions, row["confidence"])
        assert row["TP"] == sum(error["tp"] for error in errors)
        assert row["FP"] == sum(error["fp"] for error in errors)
        assert row["recall_small"] == pytest.approx(1 - sum(error["fn_small"] for error in errors) / 2)


def test_matching_uses_all_gt_sizes(dataset, perfect_predictions):
    truth = dataset.coco()
    truth["annotations"][0].update(bbox=[0, 0, 128, 128], area=128 ** 2)
    perfect_predictions[0]["bbox"] = [0, 0, 128, 128]
    counts = count_errors(truth, perfect_predictions)
    assert counts[0]["tp"] == 1 and counts[0]["fp"] == 0 and counts[0]["gt_small"] == 0


def test_no_gt_metrics_are_undefined(toy_data, perfect_predictions, tmp_path):
    for label in (toy_data.parent.parent / "dataset/labels/val").glob("*.txt"):
        label.write_text("")
    row = evaluate(load_dataset(toy_data), perfect_predictions, tmp_path / "metrics")[0]
    assert row["AP_small"] is None and row["FN_small_rate"] is None
    assert row["FP_per_image"] == 1


@pytest.mark.parametrize("field,value", [("image_id", 999), ("category_id", 99),
                                        ("score", float("nan")), ("bbox", [0, 0, -1, 3])])
def test_bad_predictions_rejected(dataset, perfect_predictions, field, value):
    perfect_predictions[0][field] = value
    with pytest.raises(ValueError):
        validate_predictions(perfect_predictions, dataset.coco())


def test_audit_detects_split_leakage_and_separates_proposed_groups(toy_data, tmp_path):
    root = toy_data.parent.parent / "dataset"
    (root / "images/val/0.png").write_bytes((root / "images/train/0.png").read_bytes())
    output = tmp_path / "audit"
    stats = audit(toy_data, ["train", "val"], output, read_yaml(AUDIT_CONFIG), sample_size=1)
    assert stats["val"]["objects"] == 2
    duplicates = json.loads((output / "duplicates.json").read_text())
    assert any(group["cross_split"] for group in duplicates["exact"])
    dataset = load_dataset(toy_data)
    groups = json.loads((output / "groups_val.json").read_text())
    rows = evaluate(dataset, [], tmp_path / "metrics", groups=groups)
    assert any("/proposed/" in row["group"] for row in rows)
    groups["dataset_fingerprint"] = "incorrect"
    with pytest.raises(ValueError, match="does not match"):
        evaluate(dataset, [], tmp_path / "bad", groups=groups)


def test_insufficient_background_is_unknown():
    config = read_yaml(AUDIT_CONFIG)
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    group = image_scores(image, [(0, .5, .5, 1., 1.)], config)
    assert group["clutter"] == "Unknown"
    assert group["scores"]["background_fraction"] == 0


def test_audit_collects_invalid_images_without_partial_metrics(toy_data, tmp_path):
    root = toy_data.parent.parent / "dataset"
    for split in ["train", "val"]:
        (root / "labels" / split / "0.txt").write_text("0 .1 .1 .8 .8\n")
    output = tmp_path / "invalid-audit"
    with pytest.raises(ValueError, match="2 invalid images"):
        audit(toy_data, ["train", "val"], output, read_yaml(AUDIT_CONFIG))
    report = json.loads((output / "validation_issues.json").read_text())
    assert report["invalid_images_by_split"] == {"train": 1, "val": 1}
    assert not (output / "groups_val.json").exists()
    assert not (output / "summary.json").exists()


def test_hash_index_finds_all_neighbors():
    values = [0, 1, 3, 7, 255, 256, 512, 1024]
    index = HashIndex()
    for value in values:
        index.add(value, value)
    for query in values:
        actual = {value for _, value in index.neighbors(query, 2)}
        assert actual == {value for value in values if (value ^ query).bit_count() <= 2}


def test_prediction_replay_rejects_changed_dataset(dataset, perfect_predictions, toy_data, tmp_path):
    exported = tmp_path / "exported"
    evaluate(dataset, perfect_predictions, exported)
    script = Path(__file__).resolve().parents[1] / "eval.py"
    command = [sys.executable, str(script), "--data", str(toy_data), "--predictions",
               str(exported / "predictions.json"), "--output", str(tmp_path / "replay")]
    subprocess.run(command, check=True, capture_output=True)
    changed = dataset.records[0]["path"]
    cv2.imwrite(str(changed), np.zeros((128, 128, 3), dtype=np.uint8))
    command[-1] = str(tmp_path / "wrong-replay")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2
    assert "Reference GT does not match" in result.stderr


def test_per_class_and_crowding_metrics_exported(dataset, perfect_predictions, tmp_path):
    extra_fp = deepcopy(perfect_predictions[0])
    extra_fp["bbox"] = [0.0, 0.0, 16.0, 16.0]
    extra_fp["score"] = 0.8
    preds = deepcopy(perfect_predictions) + [extra_fp]
    out_dir = tmp_path / "per-class"
    rows = evaluate(dataset, preds, out_dir)
    assert rows[0]["GT"] == 2
    assert (out_dir / "metrics_per_class.csv").is_file()
    payload = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    per_class = {row["class_name"]: row for row in payload["per_class"]}
    assert set(per_class) == {"a", "b"}
    assert per_class["a"]["GT"] == 1 and per_class["a"]["TP"] == 1 and per_class["a"]["FP"] == 1
    assert per_class["b"]["GT"] == 1 and per_class["b"]["TP"] == 1 and per_class["b"]["FP"] == 0
    assert per_class["b"]["AP_small"] == pytest.approx(1.0)
    assert sum(item["TP"] for item in payload["per_class"]) == rows[0]["TP"]
    assert sum(item["FP"] for item in payload["per_class"]) == rows[0]["FP"]
    assert payload["crowding"]["max_gt_per_image"] == 1
    assert payload["crowding"]["images_gt_over_100"] == 0
    assert payload["crowding"]["AP_maxdet300"] == pytest.approx(rows[0]["AP"])


def test_crowded_image_over_100_objects_separates_coco100_from_maxdet300(toy_data, tmp_path):
    # Place 120 non-overlapping 4x4 small boxes on a 128x128 validation image.
    lines = []
    for idx in range(120):
        col, row = idx % 12, idx // 12
        xc = (col * 10 + 4) / 128
        yc = (row * 10 + 4) / 128
        lines.append(f"0 {xc:.8f} {yc:.8f} {4 / 128:.8f} {4 / 128:.8f}")
    (toy_data.parent.parent / "dataset/labels/val/0.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    ds = load_dataset(toy_data)
    preds = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": list(a["bbox"]), "score": 0.99}
             for a in ds.coco()["annotations"]]
    out_dir = tmp_path / "crowded"
    rows = evaluate(ds, preds, out_dir)
    payload = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    # Standard COCO caps detections per image at 100, while crowding diagnostic recovers all 120 at maxDets=300.
    assert rows[0]["AR100"] < 1.0
    assert payload["crowding"]["max_gt_per_image"] == 120
    assert payload["crowding"]["images_gt_over_100"] == 1
    assert payload["crowding"]["AR300"] == pytest.approx(1.0)
    assert payload["crowding"]["AP_maxdet300"] == pytest.approx(1.0)


def test_per_class_without_gt_reports_undefined_ap(toy_data, perfect_predictions, tmp_path):
    # Remove class 'b' GT from val split while keeping a class 'b' prediction.
    (toy_data.parent.parent / "dataset/labels/val/1.txt").write_text("")
    ds = load_dataset(toy_data)
    out_dir = tmp_path / "missing-class-gt"
    evaluate(ds, perfect_predictions, out_dir)
    payload = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    class_b = next(item for item in payload["per_class"] if item["class_name"] == "b")
    assert class_b["GT"] == 0 and class_b["images_with_gt"] == 0
    assert class_b["AP"] is None and class_b["AR_small"] is None and class_b["FN_small_rate"] is None
    assert class_b["FP"] == 1


def test_results_csv_nan_detection_and_telemetry_accumulation(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    results_csv = run_dir / "results.csv"
    results_csv.write_text("epoch, train/box_loss, val/cls_loss\n1, 1.25, 0.75\n2, 1.10, 0.65\n", encoding="utf-8")
    info = train_module.inspect_results_csv(results_csv)
    assert info == {"epochs_logged": 2, "finite_metrics": True, "nonfinite_columns": []}

    (run_dir / "experiment.json").write_text(json.dumps({"baseline": "B0-Raw"}), encoding="utf-8")

    def fake_resume():
        with results_csv.open("a", encoding="utf-8") as handle:
            handle.write("3, nan, 0.50\n")

    summary = train_module.collect_gpu_and_time(fake_resume, run_dir, mode="resume")
    assert summary["epochs_logged"] == 3
    assert summary["latest_session"]["epochs_in_session"] == 1
    assert summary["finite_metrics"] is False
    assert summary["latest_session"]["nonfinite_columns"] == ["train/box_loss"]
    saved = json.loads((run_dir / "experiment.json").read_text(encoding="utf-8"))
    assert "code_fingerprint" in saved and len(saved["telemetry"]["sessions"]) == 1


def test_stripped_checkpoint_rejected_for_resume(tmp_path):
    import argparse
    import torch

    stripped = tmp_path / "best.pt"
    torch.save({"epoch": -1, "optimizer": None, "model": {}}, stripped)
    parser = argparse.ArgumentParser()
    with pytest.raises(SystemExit):
        train_module.validate_resume_checkpoint(parser, stripped)


def test_pixel_modified_groups_allow_enhanced_images_but_reject_label_changes(toy_data, tmp_path):
    audit_dir = tmp_path / "audit"
    audit(toy_data, ["val"], audit_dir, read_yaml(AUDIT_CONFIG), sample_size=1)
    groups = json.loads((audit_dir / "groups_val.json").read_text(encoding="utf-8"))

    # Modify pixel content without changing resolution or labels (simulating CLAHE).
    val_img = toy_data.parent.parent / "dataset/images/val/0.png"
    cv2.imwrite(str(val_img), np.full((128, 128, 3), 42, dtype=np.uint8))
    enhanced_ds = load_dataset(toy_data)
    with pytest.raises(ValueError, match="does not match"):
        evaluate(enhanced_ds, [], tmp_path / "strict-fail", groups=groups)

    rows = evaluate(enhanced_ds, [], tmp_path / "allowed", groups=groups, allow_pixel_modified_groups=True)
    assert any("/proposed/" in row["group"] for row in rows)

    # Mutate a label box: even allow_pixel_modified_groups=True must reject it.
    (toy_data.parent.parent / "dataset/labels/val/0.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    mismatched_ds = load_dataset(toy_data)
    with pytest.raises(ValueError, match="geometry/labels do not match"):
        evaluate(mismatched_ds, [], tmp_path / "geom-fail", groups=groups, allow_pixel_modified_groups=True)
