import json

import pytest

from pipeline.data import load_dataset
from preprocessing_data.repair_labels import repair, repair_text


def test_clips_edges_preserves_classes_and_removes_degenerate_boxes():
    original = "2 .99 .5 .04 .2\n1 .5 .5 0 .2\n3 .5 .5 .1 .1\n"
    fixed, changes = repair_text(original, (100, 100), 4)
    assert fixed.splitlines() == ["2 0.985000000000 0.500000000000 0.030000000000 0.200000000000",
                                  "3 .5 .5 .1 .1"]
    assert [change["action"] for change in changes] == ["clip_to_image", "remove_zero_area"]


@pytest.mark.parametrize("content", ["4 .5 .5 .1 .1", "1 nan .5 .1 .1", "1 .5 .5 -.1 .1",
                                     "1 1.1 .5 .1 .1", "not a label"])
def test_ambiguous_errors_require_manual_review(content):
    with pytest.raises(ValueError):
        repair_text(content, (100, 100), 4)


def test_keeps_valid_and_subpixel_rounding_rows_verbatim():
    content = "0 0.950005 .5 .1 .1\n"
    assert repair_text(content, (100, 100), 4) == (content, [])


def test_dry_run_apply_backups_and_idempotence(toy_data, tmp_path):
    root = toy_data.parent.parent / "dataset"
    label = root / "labels/train/0.txt"
    original = "0 .99 .5 .04 .2\n0 .5 .5 0 .2\n"
    label.write_text(original)
    dry = repair(toy_data, tmp_path / "dry")
    assert dry["status"] == "planned" and label.read_text() == original
    applied = repair(toy_data, tmp_path / "applied", apply=True)
    assert applied["status"] == "applied" and applied["clipped_boxes"] == 1
    assert applied["removed_zero_area_boxes"] == 1
    assert (tmp_path / "applied/original_labels/labels/train/0.txt").read_text() == original
    assert len(load_dataset(toy_data, "train").records[0]["boxes"]) == 1
    assert json.loads((tmp_path / "applied/repair.json").read_text())["status"] == "applied"
    assert repair(toy_data, tmp_path / "again")["modified_files"] == 0


def test_bad_inputs_prevent_all_annotation_writes(toy_data, tmp_path):
    labels = toy_data.parent.parent / "dataset/labels/train"
    first = labels / "0.txt"
    first.write_text("0 .99 .5 .04 .2\n")
    (labels / "1.txt").write_text("99 .5 .5 .1 .1\n")
    with pytest.raises(ValueError):
        repair(toy_data, tmp_path / "failed", apply=True)
    assert first.read_text() == "0 .99 .5 .04 .2\n"
