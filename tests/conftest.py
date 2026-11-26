import cv2
import numpy as np
import pytest
import yaml


@pytest.fixture
def toy_data(tmp_path):
    root = tmp_path / "dataset"
    rng = np.random.default_rng(7)
    for split in ["train", "val", "test"]:
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for index in range(2):
            image = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
            cv2.imwrite(str(root / "images" / split / f"{index}.png"), image)
            (root / "labels" / split / f"{index}.txt").write_text(
                f"{index} 0.5 0.5 0.15625 0.15625\n")
    config = tmp_path / "configs" / "data.yaml"
    config.parent.mkdir()
    config.write_text(yaml.safe_dump({"path": "../dataset", "train": "images/train",
                                      "val": "images/val", "test": "images/test", "names": ["a", "b"]}))
    return config


@pytest.fixture
def dataset(toy_data):
    from pipeline.data import load_dataset
    return load_dataset(toy_data)


@pytest.fixture
def perfect_predictions(dataset):
    return [{"image_id": annotation["image_id"], "category_id": annotation["category_id"],
             "bbox": list(annotation["bbox"]), "score": 0.99} for annotation in dataset.coco()["annotations"]]
