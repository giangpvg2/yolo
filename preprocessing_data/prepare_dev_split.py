"""Create a leakage-checked, versioned development dataset and portable YAML (Gate D0)."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.data import dataset_config, label_path, load_dataset, write_json


def read_exclusion_list(inline_items, list_file):
    items = set(inline_items or [])
    if list_file is not None:
        for raw_line in Path(list_file).read_text(encoding="utf-8").splitlines():
            cleaned = raw_line.split("#", 1)[0].strip()
            if cleaned:
                items.add(cleaned)
    return items


def materialize_file(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"Unsupported link mode: {mode}")


def plan_dev_split(data_yaml, manual_train_exclusions=None, dedup_within_train=False):
    config, src_root, names = dataset_config(data_yaml)
    splits = [split for split in ("train", "val", "test") if split in config]
    if "train" not in splits or "val" not in splits:
        raise ValueError("Data YAML must define at least train and val splits")
    loaded = {split: load_dataset(data_yaml, split) for split in splits}
    manual = set(manual_train_exclusions or [])
    train_names = {record["file_name"] for record in loaded["train"].records}
    unknown = sorted(manual - train_names)
    if unknown:
        raise ValueError(f"Manual train exclusions not found in train split: {unknown[:5]}")

    holdout_by_hash = {}
    for split in ("val", "test"):
        if split in loaded:
            for record in loaded[split].records:
                holdout_by_hash.setdefault(record["sha256"], []).append(
                    {"split": split, "file_name": record["file_name"]}
                )

    kept_train, excluded_train = [], []
    seen_train_hashes = {}
    for record in loaded["train"].records:
        sha = record["sha256"]
        fname = record["file_name"]
        if sha in holdout_by_hash:
            excluded_train.append({
                "file_name": fname,
                "sha256": sha,
                "reason": "exact_cross_split_duplicate",
                "matched": holdout_by_hash[sha],
            })
        elif fname in manual:
            excluded_train.append({
                "file_name": fname,
                "sha256": sha,
                "reason": "manual_exclusion",
                "matched": [],
            })
        elif dedup_within_train and sha in seen_train_hashes:
            excluded_train.append({
                "file_name": fname,
                "sha256": sha,
                "reason": "exact_within_train_duplicate",
                "matched": [{"split": "train", "file_name": seen_train_hashes[sha]}],
            })
        else:
            seen_train_hashes[sha] = fname
            kept_train.append(record)

    if not kept_train or not any(record["boxes"] for record in kept_train):
        raise ValueError("Filtered train split has no labeled objects remaining")

    return {
        "config": config,
        "src_root": src_root,
        "names": names,
        "splits": splits,
        "loaded": loaded,
        "kept_records": {
            "train": kept_train,
            **{split: loaded[split].records for split in splits if split != "train"},
        },
        "excluded_train": excluded_train,
    }


def write_decision_markdown(report_path, summary):
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Quyết định chuẩn hóa tập dữ liệu phát triển (Gate D0)",
        "",
        f"- Thời điểm tạo (UTC): `{summary['created_utc']}`",
        f"- Cấu hình nguồn: `{summary['source_yaml']}`",
        f"- Cấu hình xuất ra: `{summary['yaml_output']}`",
        f"- Chế độ lưu tệp: `{summary['link_mode']}`",
        f"- Khử trùng lặp nội bộ tập `train`: `{summary['dedup_within_train']}`",
        "",
        "## Thống kê trước và sau khi lọc",
        "",
        "| Split | Ảnh gốc | Ảnh giữ lại | Ảnh loại bỏ | Hộp sau lọc |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, counts in summary["splits"].items():
        lines.append(
            f"| {split} | {counts['source_images']} | {counts['kept_images']} | "
            f"{counts['excluded_images']} | {counts['kept_objects']} |"
        )
    lines += ["", "## Danh sách ảnh loại khỏi `train`", ""]
    if not summary["excluded_train"]:
        lines.append("Không có ảnh nào bị loại bỏ.")
    else:
        for item in summary["excluded_train"]:
            matched = ", ".join(f"{m['split']}:{m['file_name']}" for m in item["matched"]) or "—"
            lines.append(f"- `{item['file_name']}` (`{item['reason']}`, đối chiếu: `{matched}`)")
    lines += [
        "",
        "> Tập `val` và `test` được giữ nguyên tuyệt đối để bảo toàn tính toàn vẹn khi đối chiếu benchmark.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_dev_split(
    data_yaml,
    output_dir,
    yaml_output,
    report_path=None,
    manual_train_exclusions=None,
    dedup_within_train=False,
    link_mode="copy",
    dry_run=False,
):
    plan = plan_dev_split(data_yaml, manual_train_exclusions, dedup_within_train)
    src_root = plan["src_root"]
    dst_root = Path(output_dir).resolve()
    yaml_output = Path(yaml_output)
    summary = {
        "status": "planned" if dry_run else "applied",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_yaml": str(Path(data_yaml)),
        "output_dir": str(dst_root),
        "yaml_output": str(yaml_output),
        "link_mode": link_mode,
        "dedup_within_train": dedup_within_train,
        "classes": plan["names"],
        "source_fingerprints": {s: plan["loaded"][s].fingerprint() for s in plan["splits"]},
        "splits": {
            split: {
                "source_images": len(plan["loaded"][split].records),
                "kept_images": len(plan["kept_records"][split]),
                "excluded_images": len(plan["loaded"][split].records) - len(plan["kept_records"][split]),
                "kept_objects": sum(len(r["boxes"]) for r in plan["kept_records"][split]),
            }
            for split in plan["splits"]
        },
        "excluded_train": plan["excluded_train"],
    }
    if dry_run:
        return summary

    if dst_root.exists() or dst_root.is_relative_to(src_root) or src_root.is_relative_to(dst_root):
        raise ValueError("Output dataset directory must be new and separate from the source dataset")
    if yaml_output.exists():
        raise ValueError(f"Output YAML already exists: {yaml_output}")

    for split in plan["splits"]:
        split_img_root = (src_root / plan["config"][split]).resolve()
        for record in plan["kept_records"][split]:
            src_img = record["path"].resolve()
            src_lbl = label_path(src_img)
            rel = src_img.relative_to(split_img_root)
            dst_img = dst_root / "images" / split / rel
            dst_lbl = (dst_root / "labels" / split / rel).with_suffix(".txt")
            materialize_file(src_img, dst_img, link_mode)
            materialize_file(src_lbl, dst_lbl, link_mode)

    yaml_payload = {
        "path": Path(os.path.relpath(dst_root, yaml_output.parent.resolve())).as_posix(),
        **{split: f"images/{split}" for split in plan["splits"]},
        "names": dict(enumerate(plan["names"])),
    }
    yaml_output.parent.mkdir(parents=True, exist_ok=True)
    yaml_output.write_text(yaml.safe_dump(yaml_payload, sort_keys=False), encoding="utf-8")
    (dst_root / "data.yaml").write_text(
        yaml.safe_dump({**yaml_payload, "path": "."}, sort_keys=False), encoding="utf-8"
    )
    summary["dev_fingerprints"] = {s: load_dataset(yaml_output, s).fingerprint() for s in plan["splits"]}
    write_json(dst_root / "split_manifest.json", summary)
    if report_path is not None:
        write_decision_markdown(report_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("configs/data_urpc_raw.yaml"))
    parser.add_argument("--output", type=Path, default=Path("datasets/URPC2020_dev"))
    parser.add_argument("--yaml-output", type=Path, default=Path("configs/data_dev.yaml"))
    parser.add_argument("--report", type=Path, default=Path("reports/data_decisions.md"))
    parser.add_argument("--exclude-train", nargs="*", default=[])
    parser.add_argument("--exclude-train-file", type=Path)
    parser.add_argument("--dedup-within-train", action="store_true")
    parser.add_argument("--link-mode", choices=["copy", "hardlink", "symlink"], default="copy")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        exclusions = read_exclusion_list(args.exclude_train, args.exclude_train_file)
        summary = prepare_dev_split(
            args.data, args.output, args.yaml_output,
            report_path=None if args.dry_run else args.report,
            manual_train_exclusions=exclusions,
            dedup_within_train=args.dedup_within_train,
            link_mode=args.link_mode,
            dry_run=args.dry_run,
        )
    except (ValueError, FileNotFoundError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
