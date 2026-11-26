"""Evaluate a detector checkpoint or an exported COCO prediction list."""

import argparse
import json
from pathlib import Path

from pipeline.data import load_dataset
from pipeline.evaluation import evaluate, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="configs/data_urpc_raw.yaml")
    parser.add_argument("--split", choices=["val", "test", "train"], default="val")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--weights", type=Path)
    source.add_argument("--predictions", type=Path, help="COCO list with IDs from ground_truth.json")
    parser.add_argument("--reference-gt", type=Path, help="GT that defines imported prediction IDs; defaults to sibling ground_truth.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=Path)
    parser.add_argument("--confidence", type=float, default=0.25, help="FP/FN threshold, not AP filtering")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-missing-labels", action="store_true")
    parser.add_argument("--allow-pixel-modified-groups", action="store_true",
                        help="Allow raw severity/clutter groups on enhanced images with identical geometry and labels")
    args = parser.parse_args()
    if not (0.001 <= args.confidence <= 1 and 0 < args.match_iou <= 1 and 0 < args.nms_iou <= 1):
        parser.error("confidence must be in [0.001,1]; IoU thresholds in (0,1]")
    if args.imgsz < 32 or args.imgsz % 32 or args.max_det < 1:
        parser.error("imgsz must be a positive multiple of 32; max-det must be positive")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("Output directory must be new or empty")
    if args.weights and not args.weights.is_file():
        parser.error("Checkpoint not found; automatic weight downloads are disabled")
    dataset = load_dataset(args.data, args.split, args.allow_missing_labels)
    if args.reference_gt and not args.predictions:
        parser.error("reference-gt is only used with imported predictions")
    if args.predictions:
        reference = args.reference_gt or args.predictions.parent / "ground_truth.json"
        if not reference.is_file():
            parser.error("Imported predictions need --reference-gt from audit, or a sibling ground_truth.json")
        if json.loads(reference.read_text()) != dataset.coco():
            parser.error("Reference GT does not match current image IDs, content, class mapping or annotations")
    predictions = (json.loads(args.predictions.read_text()) if args.predictions else
                   predict(dataset, args.weights, args.imgsz, args.device, args.nms_iou, args.max_det))
    groups = json.loads(args.groups.read_text()) if args.groups else None
    rows = evaluate(dataset, predictions, args.output, args.confidence, args.match_iou, groups,
                    {"prediction_source": "imported" if args.predictions else "checkpoint",
                     "imgsz": None if args.predictions else args.imgsz,
                     "prediction_confidence": None if args.predictions else 0.001,
                     "nms_iou": None if args.predictions else args.nms_iou,
                     "prediction_max_det": None if args.predictions else args.max_det,
                     "missing_labels_allowed": args.allow_missing_labels,
                     "pixel_modified_groups_allowed": args.allow_pixel_modified_groups},
                    allow_pixel_modified_groups=args.allow_pixel_modified_groups)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
