"""Audit YOLO splits and propose reviewable visual-quality/background groups."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from pathlib import Path
import random

import cv2
import numpy as np

from pipeline.data import label_path, load_dataset, read_yaml, write_json


def image_scores(image, boxes, config):
    height, width = image.shape[:2]
    scale = min(1.0, config["analysis_size"] / max(height, width))
    image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = gray.astype(np.float32)
    local_mean = cv2.blur(normalized, (9, 9))
    local_variance = np.maximum(cv2.blur(normalized ** 2, (9, 9)) - local_mean ** 2, 0)
    contrast = float(np.sqrt(local_variance).mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    means = image.mean(axis=(0, 1))
    color_imbalance = float(1 - means.min() / max(float(means.max()), 1.0))
    components = [1 - min(contrast / config["contrast_reference"], 1),
                  1 - min(sharpness / config["sharpness_reference"], 1), color_imbalance]
    severity_score = float(np.dot(config["severity_weights"], components))
    low, high = config["severity_cutoffs"]
    severity = "Clear" if severity_score < low else "Moderate" if severity_score < high else "Severe"
    background = np.ones(gray.shape, dtype=np.uint8)
    h, w = gray.shape
    for _, x, y, box_w, box_h in boxes:
        x1, x2 = max(0, int((x - box_w / 2) * w)), min(w, int(np.ceil((x + box_w / 2) * w)))
        y1, y2 = max(0, int((y - box_h / 2) * h)), min(h, int(np.ceil((y + box_h / 2) * h)))
        background[y1:y2, x1:x2] = 0
    # Exclude GT boundaries as well, so box edges cannot inflate the clutter proxy.
    background = cv2.erode(background, np.ones((5, 5), dtype=np.uint8))
    available = background.astype(bool)
    density = float((cv2.Canny(gray, 80, 160)[available] > 0).mean()) if available.any() else None
    if available.mean() < 0.05:
        clutter = "Unknown"
    else:
        clutter = "Hard Clutter" if density >= config["clutter_edge_density"] else "Smooth Seabed"
    return {"severity": severity, "clutter": clutter, "reviewed": False,
            "scores": {"visual_degradation": severity_score, "local_contrast": contrast,
                       "laplacian_variance": sharpness, "color_imbalance": color_imbalance,
                       "background_edge_density": density, "background_fraction": float(available.mean())}}


def difference_hash(image):
    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (9, 8))
    value = 0
    for bit in (gray[:, 1:] > gray[:, :-1]).flat:
        value = (value << 1) | int(bit)
    return value


class HashIndex:
    """BK tree for Hamming-neighbor queries without an all-pairs image comparison."""

    def __init__(self):
        self.root = None

    def add(self, value, record):
        if self.root is None:
            self.root = [value, [record], {}]
            return
        node = self.root
        while True:
            distance = (node[0] ^ value).bit_count()
            if distance == 0:
                node[1].append(record)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [record], {}]
                return
            node = node[2][distance]

    def neighbors(self, value, maximum):
        pending = [self.root] if self.root else []
        while pending:
            node = pending.pop()
            distance = (node[0] ^ value).bit_count()
            if distance <= maximum:
                for record in node[1]:
                    yield distance, record
            pending.extend(child for edge, child in node[2].items()
                           if distance - maximum <= edge <= distance + maximum)


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_config(config):
    if config["analysis_size"] < 16 or config["contrast_reference"] <= 0 or config["sharpness_reference"] <= 0:
        raise ValueError("Invalid analysis size or quality reference")
    weights, cutoffs = config["severity_weights"], config["severity_cutoffs"]
    if len(weights) != 3 or min(weights) < 0 or not np.isclose(sum(weights), 1):
        raise ValueError("severity_weights must contain three nonnegative values summing to one")
    if len(cutoffs) != 2 or not 0 <= cutoffs[0] < cutoffs[1] <= 1:
        raise ValueError("severity_cutoffs must be increasing within [0,1]")
    if not 0 <= config["clutter_edge_density"] <= 1 or not 0 <= config["dhash_distance"] <= 16:
        raise ValueError("Invalid edge density or dHash distance")


def audit(data_path, splits, output, config, allow_missing=False, sample_size=300, seed=0):
    validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    datasets, issues = [], []
    for split in splits:
        split_issues = []
        datasets.append(load_dataset(data_path, split, allow_missing, split_issues))
        issues.extend({"split": split, **issue} for issue in split_issues)
    if issues:
        write_json(output / "validation_issues.json", {
            "status": "invalid", "invalid_images": len(issues),
            "invalid_images_by_split": dict(Counter(issue["split"] for issue in issues)),
            "issues": issues,
            "note": "At least one error per failed image; no partial metrics/groups are exported."})
        (output / "dataset_audit.md").write_text(
            "# Dataset audit: validation failed\n\n"
            f"Found {len(issues)} images with label/image validation errors.\n\n"
            "See validation_issues.json. Source data was not modified.\n"
            "No partial size statistics or groups were exported; correct or review errors before training.\n",
            encoding="utf-8")
        raise ValueError(f"{len(issues)} invalid images; see validation_issues.json in the audit output")
    manifest, objects, near_duplicates, samples = [], [], [], []
    exact, statistics = defaultdict(list), {}
    index = HashIndex()
    for dataset in datasets:
        split = dataset.split
        groups = {}
        counts = Counter()
        ratios = defaultdict(list)
        for record in dataset.records:
            image = cv2.imread(str(record["path"]))
            sha256 = record["sha256"]
            label = label_path(record["path"])
            label_sha256 = hashlib.sha256(label.read_bytes()).hexdigest() if label.exists() else None
            dhash = difference_hash(image)
            item = {"split": split, "file_name": record["file_name"], "image_id": record["id"],
                    "width": record["width"], "height": record["height"], "objects": len(record["boxes"]),
                    "missing_label": record["missing_label"], "sha256": sha256,
                    "label_sha256": label_sha256, "dhash": f"{dhash:016x}"}
            exact[sha256].append({"split": split, "file_name": record["file_name"]})
            if len(near_duplicates) < 1000:
                for distance, previous in index.neighbors(dhash, config["dhash_distance"]):
                    if previous["split"] != split and previous["sha256"] != sha256:
                        near_duplicates.append({"first": previous["file_name"], "second": record["file_name"],
                                                "first_split": previous["split"], "second_split": split,
                                                "hamming_distance": distance})
                        if len(near_duplicates) >= 1000:
                            break
            index.add(dhash, item)
            manifest.append(item)
            groups[record["file_name"]] = image_scores(image, record["boxes"], config)
            for category, _, _, width, height in record["boxes"]:
                area = width * height * record["width"] * record["height"]
                counts[dataset.names[category]] += 1
                ratios[dataset.names[category]].append(width * height)
                objects.append({"split": split, "file_name": record["file_name"],
                                "class_id": category, "class_name": dataset.names[category],
                                "relative_area": width * height, "original_area": area,
                                "coco_small": area <= 32 ** 2})
        statistics[split] = {
            "images": len(dataset.records), "objects": sum(counts.values()),
            "missing_labels": sum(record["missing_label"] for record in dataset.records),
            "border_rounding_boxes": sum(record["border_rounding_boxes"] for record in dataset.records),
            "classes": {name: {"count": counts[name], "relative_area_quantiles":
                        [float(x) for x in np.quantile(ratios[name], [0, .25, .5, .75, 1])] if ratios[name] else None}
                        for name in dataset.names},
            "severity": dict(Counter(group["severity"] for group in groups.values())),
            "clutter": dict(Counter(group["clutter"] for group in groups.values())),
            "joint_groups": dict(Counter(f"{group['severity']} / {group['clutter']}" for group in groups.values())),
        }
        write_json(output / f"groups_{split}.json", {
            "schema_version": 1, "split": split, "dataset_fingerprint": dataset.fingerprint(),
            "geometry_fingerprint": dataset.geometry_fingerprint(),
            "source": "fixed visual proxies; manual review required", "config": config, "images": groups})
        write_json(output / f"ground_truth_{split}.json", dataset.coco())
        if split in {"val", "test"}:
            buckets = defaultdict(list)
            for name, group in groups.items():
                buckets[(group["severity"], group["clutter"])].append(name)
            rng = random.Random(seed)
            for bucket in buckets.values():
                rng.shuffle(bucket)
            selected = []
            while len(selected) < min(sample_size, len(groups)):
                for key in sorted(buckets):
                    if buckets[key] and len(selected) < sample_size:
                        selected.append(buckets[key].pop())
            samples.extend({"split": split, "file_name": name, "severity": groups[name]["severity"],
                            "clutter": groups[name]["clutter"], "reviewed": False} for name in selected)
    duplicates = [{"sha256": sha, "cross_split": len({entry["split"] for entry in entries}) > 1,
                   "images": entries} for sha, entries in exact.items() if len(entries) > 1]
    write_json(output / "summary.json", statistics)
    write_json(output / "duplicates.json", {"exact": duplicates, "near_cross_split_candidates": near_duplicates,
               "near_candidate_limit_reached": len(near_duplicates) >= 1000,
               "note": "dHash candidates require review; this does not establish source independence"})
    write_csv(output / "manifest.csv", manifest, list(manifest[0]))
    write_csv(output / "objects.csv", objects,
              ["split", "file_name", "class_id", "class_name", "relative_area", "original_area", "coco_small"])
    write_csv(output / "review_sample.csv", samples, ["split", "file_name", "severity", "clutter", "reviewed"])
    lines = ["# Dataset audit", "", "Severity and clutter are proposed visual proxies, not physical ground truth.",
             "Missing labels allowed: " + str(allow_missing), "", "| Split | Images | Objects | Missing labels |",
             "|---|---:|---:|---:|"]
    for split, stats in statistics.items():
        lines.append(f"| {split} | {stats['images']} | {stats['objects']} | {stats['missing_labels']} |")
    lines += ["", "## Class coverage", "", "| Class | " + " | ".join(statistics) + " |",
              "|---|" + "---:|" * len(statistics)]
    for name in datasets[0].names:
        lines.append("| " + name + " | " + " | ".join(
            str(stats["classes"][name]["count"]) for stats in statistics.values()) + " |")
    if "train" in statistics and "val" in statistics:
        unseen = [name for name in datasets[0].names
                  if statistics["train"]["classes"][name]["count"] == 0
                  and statistics["val"]["classes"][name]["count"] > 0]
        if unseen:
            lines += ["", "Validation classes with no training objects: " + ", ".join(unseen) +
                      ". Review labels/protocol before interpreting macro AP."]
    lines += ["", f"Exact duplicate groups: {len(duplicates)}; cross-split groups: "
              f"{sum(group['cross_split'] for group in duplicates)}.",
              f"Near-duplicate candidates: {len(near_duplicates)} (cap 1000).", "",
              "See summary.json for per-class area quantiles and group counts; objects.csv for individual boxes.",
              "Review the stratified sample in review_sample.csv, then edit groups_<split>.json and set reviewed=true.",
              "Box coordinates are preserved; export rounding up to 0.01 pixel at image borders is accepted and counted.",
              "Do not tune group thresholds, model choices or annotations on the test set."]
    (output / "dataset_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="configs/data_urpc_raw.yaml")
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val"])
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs/audit.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/audit"))
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-missing-labels", action="store_true")
    args = parser.parse_args()
    if args.sample_size < 0 or len(set(args.splits)) != len(args.splits):
        parser.error("sample-size must be nonnegative; splits must be unique")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("Output directory must be new or empty")
    try:
        summary = audit(args.data, args.splits, args.output, read_yaml(args.config),
                        args.allow_missing_labels, args.sample_size, args.seed)
    except ValueError as error:
        parser.error(str(error))
    print({split: {key: stats[key] for key in ["images", "objects", "missing_labels"]}
           for split, stats in summary.items()})


if __name__ == "__main__":
    main()
