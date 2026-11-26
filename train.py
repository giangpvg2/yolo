"""Train comparable YOLOv8n baselines, record GPU/time telemetry, and evaluate best.pt."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import time

from pipeline.data import dataset_config, load_dataset, read_yaml, write_json

BASELINES = {
    "B0-Raw": ("yolov8n.yaml", "data_urpc_raw.yaml"),
    "B0-Enhanced": ("yolov8n.yaml", "data_urpc_clahe.yaml"),
    "B1-Raw": ("yolov8n-p2.yaml", "data_urpc_raw.yaml"),
}
PROJECT_ROOT = Path(__file__).resolve().parent
CODE_SNAPSHOT_FILES = [
    "train.py",
    "eval.py",
    "dataset_audit.py",
    "pipeline/data.py",
    "pipeline/evaluation.py",
    "configs/train.yaml",
]


def environment():
    versions = {"python": platform.python_version()}
    for package in ["ultralytics", "torch", "torchvision", "numpy", "opencv-python", "PyYAML", "pycocotools"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def code_provenance():
    digest = hashlib.sha256()
    for relative in CODE_SNAPSHOT_FILES:
        path = PROJECT_ROOT / relative
        if path.is_file():
            digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    commit = None
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        if result.returncode == 0:
            commit = result.stdout.strip() or None
    except OSError:
        commit = None
    return {"code_fingerprint": digest.hexdigest(), "git_commit": commit}


def check_runtime(parser, skip_eval, device=None):
    versions = environment()
    for package, required in [("ultralytics", "8.3.232"), ("torch", "2.9.1"), ("torchvision", "0.24.1")]:
        if (versions[package] or "").split("+")[0] != required:
            parser.error(f"Install pinned requirements: {package}=={required} is required")
    if not skip_eval and versions["pycocotools"] is None:
        parser.error("Install pycocotools before training, or explicitly use --skip-eval")
    try:
        import torch
        from torchvision.extension import _has_ops
        if not _has_ops():
            raise RuntimeError("Missing torchvision operators")
    except (ImportError, RuntimeError, OSError):
        parser.error("Torch/Torchvision operators unavailable; install a matching CPU or CUDA pair (see README)")
    if device is not None and str(device).strip().lower() not in {"", "cpu", "mps"}:
        if not torch.cuda.is_available():
            parser.error(
                f"CUDA device {device!r} requested, but torch.cuda.is_available() is False. "
                "Enable a GPU runtime and install a CUDA build of torch==2.9.1 / torchvision==0.24.1."
            )


def validate_resume_checkpoint(parser, checkpoint_path):
    import torch

    try:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except Exception as error:
        parser.error(f"Unreadable checkpoint {checkpoint_path}: {error}")
    if not isinstance(ckpt, dict) or ckpt.get("optimizer") is None or ckpt.get("epoch", -1) < 0:
        parser.error(
            "Checkpoint has no active optimizer state (training already completed or stripped); "
            "pass an interrupted weights/last.pt checkpoint to --resume."
        )


def inspect_results_csv(results_path):
    if not Path(results_path).is_file():
        return {"epochs_logged": 0, "finite_metrics": True, "nonfinite_columns": []}
    nonfinite = set()
    rows = 0
    with Path(results_path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            for key, raw_value in row.items():
                if key is None or raw_value is None or not raw_value.strip():
                    continue
                try:
                    number = float(raw_value.strip())
                except ValueError:
                    continue
                if not math.isfinite(number):
                    nonfinite.add(key.strip())
    return {"epochs_logged": rows, "finite_metrics": not nonfinite, "nonfinite_columns": sorted(nonfinite)}


def collect_gpu_and_time(train_call, save_dir, mode):
    import torch

    save_dir = Path(save_dir)
    prior_epochs = inspect_results_csv(save_dir / "results.csv")["epochs_logged"]
    cuda_active = torch.cuda.is_available()
    if cuda_active:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started_utc = datetime.now(timezone.utc).isoformat()
    start_clock = time.perf_counter()
    train_call()
    if cuda_active:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_clock
    finished_utc = datetime.now(timezone.utc).isoformat()
    results_info = inspect_results_csv(save_dir / "results.csv")
    epochs_this_session = max(0, results_info["epochs_logged"] - prior_epochs)
    gpu_info = {"cuda_available": cuda_active, "gpu_name": None, "cuda_version": torch.version.cuda,
                "peak_vram_allocated_mb": None, "peak_vram_reserved_mb": None}
    if cuda_active and torch.cuda.device_count() > 0:
        gpu_info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "peak_vram_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
            "peak_vram_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2),
        })
    session = {
        "mode": mode,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "training_seconds": round(elapsed, 3),
        "epochs_in_session": epochs_this_session,
        "seconds_per_epoch": round(elapsed / epochs_this_session, 3) if epochs_this_session else None,
        **gpu_info,
        **results_info,
    }
    experiment_path = save_dir / "experiment.json"
    experiment = json.loads(experiment_path.read_text(encoding="utf-8")) if experiment_path.is_file() else {}
    sessions = list(experiment.get("sessions", [])) + [session]
    vram_peaks = [item["peak_vram_allocated_mb"] for item in sessions if item.get("peak_vram_allocated_mb") is not None]
    telemetry_summary = {
        "total_training_seconds": round(sum(item["training_seconds"] for item in sessions), 3),
        "epochs_logged": results_info["epochs_logged"],
        "max_peak_vram_allocated_mb": max(vram_peaks) if vram_peaks else None,
        "finite_metrics": all(item["finite_metrics"] for item in sessions),
        "latest_session": session,
        "sessions": sessions,
    }
    experiment.update({**code_provenance(), "telemetry": telemetry_summary})
    write_json(experiment_path, experiment)
    write_json(save_dir / "training_telemetry.json", telemetry_summary)
    return telemetry_summary


def evaluate_best(model, device, dataset=None, telemetry=None):
    from pipeline.evaluation import evaluate, predict
    trainer = model.trainer
    dataset = dataset or load_dataset(trainer.args.data, "val")
    predictions = predict(dataset, trainer.best, trainer.args.imgsz, device)
    evaluate(dataset, predictions, trainer.save_dir / "evaluation", metadata={
        "seed": trainer.args.seed, "imgsz": trainer.args.imgsz, "environment": environment(),
        **code_provenance(),
        "prediction_confidence": 0.001, "nms_iou": 0.7, "prediction_max_det": 300,
        "training_seconds": telemetry["total_training_seconds"] if telemetry else None,
        "peak_vram_allocated_mb": telemetry["max_peak_vram_allocated_mb"] if telemetry else None,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=BASELINES)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--resume", type=Path, help="Resume an interrupted last.pt with its original recipe")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print recipe without training or requiring data")
    args = parser.parse_args()
    if args.resume:
        if args.dry_run or any(value is not None for value in [args.baseline, args.config, args.project,
                args.data, args.seed, args.epochs, args.batch, args.imgsz, args.name]):
            parser.error("resume uses saved settings; do not combine it with recipe overrides or dry-run")
        if not args.resume.is_file():
            parser.error("Resume checkpoint not found")
        check_runtime(parser, args.skip_eval, args.device)
        validate_resume_checkpoint(parser, args.resume)
        from ultralytics import YOLO
        model = YOLO(str(args.resume))
        inferred_save_dir = args.resume.resolve().parent.parent
        telemetry = collect_gpu_and_time(
            lambda: model.train(resume=True, device=args.device),
            inferred_save_dir,
            mode="resume",
        )
        if not args.skip_eval:
            evaluate_best(model, args.device, telemetry=telemetry)
        return
    args.baseline = args.baseline or "B1-Raw"
    args.project = args.project or PROJECT_ROOT / "runs/detect"
    recipe = read_yaml(args.config or PROJECT_ROOT / "configs/train.yaml")
    for key in ["seed", "epochs", "batch", "imgsz"]:
        if getattr(args, key) is not None:
            recipe[key] = getattr(args, key)
    if recipe.get("pretrained") is not False:
        parser.error("This baseline protocol requires pretrained: false for every model")
    if recipe["epochs"] < 1 or recipe["batch"] < 1 or recipe["seed"] < 0:
        parser.error("epochs/batch must be positive and seed nonnegative")
    if recipe["imgsz"] < 32 or recipe["imgsz"] % 32:
        parser.error("imgsz must be a positive multiple of 32")
    model_yaml, data_yaml = BASELINES[args.baseline]
    data_path = args.data or PROJECT_ROOT / "configs" / data_yaml
    data_config, root, names = dataset_config(data_path)
    run_name = args.name or f"{args.baseline}_s{recipe['seed']}_{recipe['imgsz']}"
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        parser.error("name must be a single directory name")
    run_dir = args.project / run_name
    summary = {"baseline": args.baseline, "model": model_yaml, "recipe": recipe,
               "classes": names, "dataset_present": root.is_dir(), "environment": environment(),
               **code_provenance()}
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return
    check_runtime(parser, args.skip_eval, args.device)
    training = load_dataset(data_path, "train")
    validation = load_dataset(data_path, "val")
    if not any(record["boxes"] for record in training.records):
        parser.error("Training split has no objects")
    if run_dir.exists():
        parser.error("Run directory already exists; use --resume or a new --name")
    run_dir.mkdir(parents=True)
    # A runtime-only resolved copy avoids Ultralytics' global datasets_dir resolution.
    runtime_data = {"path": str(root), "names": names,
                    **{split: data_config[split] for split in ["train", "val", "test"] if split in data_config}}
    import yaml
    (run_dir / "data.runtime.yaml").write_text(yaml.safe_dump(runtime_data), encoding="utf-8")
    summary.update({"created_utc": datetime.now(timezone.utc).isoformat(),
                    "train_fingerprint": training.fingerprint(), "val_fingerprint": validation.fingerprint()})
    write_json(run_dir / "experiment.json", summary)
    from ultralytics import YOLO
    model = YOLO(model_yaml)
    telemetry = collect_gpu_and_time(
        lambda: model.train(**recipe, data=str((run_dir / "data.runtime.yaml").resolve()),
                            device=args.device, project=str(args.project.resolve()),
                            name=run_name, exist_ok=True),
        run_dir,
        mode="train",
    )
    if not args.skip_eval:
        evaluate_best(model, args.device, validation, telemetry=telemetry)


if __name__ == "__main__":
    main()
