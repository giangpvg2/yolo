"""COCO evaluation plus thresholded, class-aware one-to-one error counts."""

from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
import csv
import io
import math
import tempfile

from pipeline.data import BOX_BORDER_TOLERANCE_PIXELS, write_json

COCO_METRIC_KEYS = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
                    "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]


def validate_predictions(predictions, truth):
    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a COCO detection list")
    images = {image["id"] for image in truth["images"]}
    classes = {category["id"] for category in truth["categories"]}
    for prediction in predictions:
        if not isinstance(prediction, dict):
            raise ValueError("Every prediction must be a COCO detection mapping")
        if prediction.get("image_id") not in images or prediction.get("category_id") not in classes:
            raise ValueError("Prediction contains unknown image_id/category_id")
        box, score = prediction.get("bbox"), prediction.get("score")
        if (not isinstance(box, list) or len(box) != 4 or
                not all(isinstance(x, (int, float)) and math.isfinite(x) for x in box) or
                box[2] <= 0 or box[3] <= 0):
            raise ValueError("Prediction bbox must be finite xywh with positive width/height")
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Prediction score must be between zero and one")


def box_iou(first, second):
    x = max(0, min(first[0] + first[2], second[0] + second[2]) - max(first[0], second[0]))
    y = max(0, min(first[1] + first[3], second[1] + second[3]) - max(first[1], second[1]))
    intersection = x * y
    return intersection / (first[2] * first[3] + second[2] * second[3] - intersection)


def count_errors(truth, predictions, confidence=0.25, match_iou=0.5):
    """Match against every GT size first; FN_small counts unmatched small GT only."""
    annotations, detections = defaultdict(list), defaultdict(list)
    for annotation in truth["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    for index, prediction in enumerate(predictions):
        if prediction["score"] >= confidence:
            detections[prediction["image_id"]].append((index, prediction))
    results = []
    for image in truth["images"]:
        gt = annotations[image["id"]]
        matched, matches = set(), []
        ranked = sorted(detections[image["id"]], key=lambda item: -item[1]["score"])
        for prediction_index, prediction in ranked:
            candidates = [(box_iou(prediction["bbox"], annotation["bbox"]), index)
                          for index, annotation in enumerate(gt)
                          if index not in matched and annotation["category_id"] == prediction["category_id"]]
            overlap, index = max(candidates, default=(-1, -1))
            if overlap >= match_iou:
                matched.add(index)
                matches.append({"prediction_index": prediction_index, "gt_id": gt[index]["id"]})
        small = {index for index, annotation in enumerate(gt) if annotation["area"] <= 32 ** 2}
        results.append({"image_id": image["id"], "file_name": image["file_name"],
                        "tp": len(matched), "fp": len(ranked) - len(matched),
                        "gt_small": len(small), "fn_small": len(small - matched), "matches": matches})
    return results


def _build_coco_eval(truth_data, predictions, image_ids, category_ids=None, max_dets=(1, 10, 100)):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    truth = COCO()
    truth.dataset = deepcopy(truth_data)
    truth.createIndex()
    # Some cocoapi releases cannot loadRes([]).
    if predictions:
        detected = truth.loadRes([dict(prediction) for prediction in predictions])
    else:
        detected = COCO()
        detected.dataset = {**truth_data, "annotations": []}
        detected.createIndex()
    evaluator = COCOeval(truth, detected, "bbox")
    evaluator.params.imgIds = sorted(image_ids)
    if category_ids is not None:
        evaluator.params.catIds = sorted(category_ids)
    evaluator.params.maxDets = list(max_dets)
    evaluator.evaluate()
    evaluator.accumulate()
    return evaluator


def coco_metrics(truth_data, predictions, image_ids, category_ids=None):
    if not image_ids:
        return {key: None for key in COCO_METRIC_KEYS}
    with redirect_stdout(io.StringIO()):
        evaluator = _build_coco_eval(truth_data, predictions, image_ids, category_ids, max_dets=(1, 10, 100))
        evaluator.summarize()
    return {key: float(value) if value >= 0 else None
            for key, value in zip(COCO_METRIC_KEYS, evaluator.stats)}


def _valid_mean(values):
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else None


def crowding_metrics(truth_data, predictions, image_ids):
    """Audit COCO's 100-detection cap on dense images while keeping standard COCO separate."""
    per_image = Counter(ann["image_id"] for ann in truth_data["annotations"] if ann["image_id"] in image_ids)
    if not image_ids:
        return {"max_gt_per_image": 0, "images_gt_over_100": 0,
                "AP_maxdet300": None, "AP_small_maxdet300": None, "AR300": None, "AR_small_maxdet300": None}
    # pycocotools' summarize() hardcodes maxDets=100 for AP; aggregate accumulated tensors directly at maxDets=300.
    with redirect_stdout(io.StringIO()):
        evaluator = _build_coco_eval(truth_data, predictions, image_ids, max_dets=(300,))
    precision = evaluator.eval["precision"]  # shape: (iou, recall, cls, area, maxdet)
    recall = evaluator.eval["recall"]        # shape: (iou, cls, area, maxdet)
    return {
        "max_gt_per_image": max(per_image.values(), default=0),
        "images_gt_over_100": sum(count > 100 for count in per_image.values()),
        "AP_maxdet300": _valid_mean(precision[:, :, :, 0, 0]),
        "AP_small_maxdet300": _valid_mean(precision[:, :, :, 1, 0]),
        "AR300": _valid_mean(recall[:, :, 0, 0]),
        "AR_small_maxdet300": _valid_mean(recall[:, :, 1, 0]),
    }


def per_class_metrics(truth_data, predictions, errors, confidence=0.25):
    """Compute per-class COCO AP/AR and thresholded 1-to-1 error counts across the full split."""
    image_ids = {image["id"] for image in truth_data["images"]}
    image_count = len(image_ids)
    matched_gt_ids = {match["gt_id"] for row in errors for match in row["matches"]}
    rows = []
    for category in truth_data["categories"]:
        cat_id, name = category["id"], category["name"]
        gt_all = [ann for ann in truth_data["annotations"] if ann["category_id"] == cat_id]
        gt_small = [ann for ann in gt_all if ann["area"] <= 32 ** 2]
        tp = sum(ann["id"] in matched_gt_ids for ann in gt_all)
        predicted = sum(1 for pred in predictions
                        if pred["category_id"] == cat_id and pred["score"] >= confidence)
        fp = predicted - tp
        fn_small = sum(ann["id"] not in matched_gt_ids for ann in gt_small)
        rows.append({
            "category_id": cat_id,
            "class_name": name,
            "images": image_count,
            "images_with_gt": len({ann["image_id"] for ann in gt_all}),
            "GT": len(gt_all),
            "GT_small": len(gt_small),
            **coco_metrics(truth_data, predictions, image_ids, category_ids=[cat_id]),
            "TP": tp,
            "FP": fp,
            "FP_per_image": fp / image_count if image_count else None,
            "FN": len(gt_all) - tp,
            "FN_small": fn_small,
            "FN_small_rate": fn_small / len(gt_small) if gt_small else None,
        })
    return rows


def error_curve(truth, predictions, match_iou=0.5, confidence=0.25):
    """Full-split recall/FP curve for comparing operating points on development data."""
    matches = {match["prediction_index"]: match["gt_id"]
               for row in count_errors(truth, predictions, 0, match_iou) for match in row["matches"]}
    small_ids = {annotation["id"] for annotation in truth["annotations"] if annotation["area"] <= 32 ** 2}
    ranked = sorted(enumerate(predictions), key=lambda item: -item[1]["score"])
    thresholds = sorted({index / 100 for index in range(101)} | {0.001, confidence}, reverse=True)
    cursor, true_positives, false_positives, small_tp = 0, 0, 0, 0
    rows = []
    for threshold in thresholds:
        while cursor < len(ranked) and ranked[cursor][1]["score"] >= threshold:
            prediction_index = ranked[cursor][0]
            if prediction_index in matches:
                true_positives += 1
                small_tp += matches[prediction_index] in small_ids
            else:
                false_positives += 1
            cursor += 1
        rows.append({"confidence": threshold,
                     "recall": true_positives / len(truth["annotations"]) if truth["annotations"] else None,
                     "recall_small": small_tp / len(small_ids) if small_ids else None,
                     "FP_per_image": false_positives / len(truth["images"]),
                     "TP": true_positives, "FP": false_positives})
    return rows


def evaluation_groups(dataset, groups, allow_pixel_modified=False):
    selections = {"all": {record["id"] for record in dataset.records}}
    if groups is None:
        return selections
    if groups.get("split") != dataset.split:
        raise ValueError("Group manifest does not match this dataset split/geometry/labels")
    if not allow_pixel_modified and groups.get("dataset_fingerprint") != dataset.fingerprint():
        raise ValueError("Group manifest does not match this dataset split/geometry/labels")
    if allow_pixel_modified and "geometry_fingerprint" in groups:
        if groups.get("geometry_fingerprint") != dataset.geometry_fingerprint():
            raise ValueError("Group manifest geometry/labels do not match enhanced dataset")
    names = {record["file_name"]: record["id"] for record in dataset.records}
    if set(groups.get("images", {})) != set(names):
        raise ValueError("Group manifest must cover exactly the evaluated images")
    for file_name, group in groups["images"].items():
        status = "reviewed" if group.get("reviewed") is True else "proposed"
        for axis, allowed in [("severity", {"Clear", "Moderate", "Severe"}),
                              ("clutter", {"Smooth Seabed", "Hard Clutter", "Unknown"})]:
            if group.get(axis) not in allowed:
                raise ValueError(f"Invalid {axis} label in group manifest")
            selections.setdefault(f"{axis}/{status}/{group[axis]}", set()).add(names[file_name])
    return selections


def evaluate(dataset, predictions, output, confidence=0.25, match_iou=0.5, groups=None, metadata=None,
             allow_pixel_modified_groups=False):
    truth = dataset.coco()
    validate_predictions(predictions, truth)
    selections = evaluation_groups(dataset, groups, allow_pixel_modified=allow_pixel_modified_groups)
    errors = count_errors(truth, predictions, confidence, match_iou)
    gt_by_image = Counter(ann["image_id"] for ann in truth["annotations"])
    rows = []
    for group, image_ids in selections.items():
        counts = [error for error in errors if error["image_id"] in image_ids]
        gt_count = sum(gt_by_image[image_id] for image_id in image_ids)
        small_count = sum(error["gt_small"] for error in counts)
        false_negatives = sum(error["fn_small"] for error in counts)
        rows.append({"group": group, "images": len(image_ids),
                     **coco_metrics(truth, predictions, image_ids),
                     "TP": sum(error["tp"] for error in counts),
                     "FP": sum(error["fp"] for error in counts),
                     "FP_per_image": sum(error["fp"] for error in counts) / len(image_ids),
                     "GT": gt_count, "GT_small": small_count, "FN_small": false_negatives,
                     "FN_small_rate": false_negatives / small_count if small_count else None})
    class_rows = per_class_metrics(truth, predictions, errors, confidence)
    crowding = crowding_metrics(truth, predictions, selections["all"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "ground_truth.json", truth)
    write_json(output / "predictions.json", predictions)
    write_json(output / "errors_per_image.json", errors)
    curve = error_curve(truth, predictions, match_iou, confidence)
    with (output / "froc.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve[0].keys())
        writer.writeheader()
        writer.writerows(curve)
    write_json(output / "metrics.json", {"protocol": {
        "split": dataset.split, "dataset_fingerprint": dataset.fingerprint(),
        "area_coordinates": "original image pixels", "small_area_max": 32 ** 2,
        "input_box_border_tolerance_pixels": BOX_BORDER_TOLERANCE_PIXELS,
        "coco_iou": "0.50:0.05:0.95", "coco_max_dets": [1, 10, 100],
        "crowding_diagnostic_max_dets": [1, 10, 300],
        "error_confidence": confidence, "error_match_iou": match_iou,
        "error_matching": "score-ordered, class-aware, one-to-one, all GT sizes",
        "metric_scale": "AP/AR fractions in [0,1]; undefined metrics are null",
        **(metadata or {}),
    }, "metrics": rows, "per_class": class_rows, "crowding": crowding})
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    if class_rows:
        with (output / "metrics_per_class.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=class_rows[0].keys())
            writer.writeheader()
            writer.writerows(class_rows)
    return rows


def predict(dataset, weights, imgsz=640, device=None, nms_iou=0.7, max_det=300):
    from ultralytics import YOLO

    if not Path(weights).is_file():
        raise FileNotFoundError("Checkpoint not found; automatic downloads are disabled")
    model = YOLO(str(weights))
    if [model.names[index] for index in range(len(model.names))] != dataset.names:
        raise ValueError("Checkpoint class names/order differ from data YAML")
    image_ids = {record["path"].resolve(): record["id"] for record in dataset.records}
    predictions, seen = [], set()
    # A filename manifest streams from disk; passing a Python image list loads the entire split.
    with tempfile.TemporaryDirectory(prefix="yolo-eval-") as temporary:
        manifest = Path(temporary) / "images.txt"
        manifest.write_text("\n".join(str(path) for path in image_ids) + "\n", encoding="utf-8")
        results = model.predict(source=str(manifest), batch=1,
                                imgsz=imgsz, device=device, conf=0.001, iou=nms_iou,
                                max_det=max_det, rect=False, stream=True, save=False, verbose=False)
        for result in results:
            image_id = image_ids[Path(result.path).resolve()]
            seen.add(image_id)
            for box, score, category in zip(result.boxes.xyxy.cpu().tolist(),
                                             result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
                x1, y1, x2, y2 = box
                if x2 > x1 and y2 > y1:
                    predictions.append({"image_id": image_id, "category_id": int(category),
                                        "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
    if seen != set(image_ids.values()):
        raise RuntimeError("Prediction did not cover every image")
    return predictions
