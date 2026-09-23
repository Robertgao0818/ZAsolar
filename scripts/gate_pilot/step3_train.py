#!/usr/bin/env python3
"""Step 3: train YOLOv11n on the four training grids and lock tau.

Holdout grids are not linked into any dataset and are not predicted.
Each fold starts from the COCO yolo11n.pt weights. The final model does too.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from scripts.gate_pilot.common import (
    HOLDOUT_PV_GRID,
    OUT_ROOT,
    STEP0_DIR,
    TRAIN_PV_GRIDS,
    setup_logging,
)
from scripts.gate_pilot.threshold import fold_warning, select_tau

VENV_PYTHON = Path.home() / "venvs" / "yolo_gate" / "bin" / "python"
CHIP_ROOT = OUT_ROOT / "chips"
STEP2_DONE = OUT_ROOT / "step2" / "step2.done"
STEP3_DIR = OUT_ROOT / "step3"
RUNS_DIR = STEP3_DIR / "runs"
CONFIG_PATH = STEP3_DIR / "train_config.json"
IMGSZ = 416
EPOCHS = 100
PATIENCE = 20
CONF = 0.001
SEED = 20260923


def _load_feasibility() -> dict:
    if not STEP2_DONE.exists():
        raise RuntimeError("step 2 has not finished; refusing to train")
    feasibility = json.loads((STEP0_DIR / "feasibility.json").read_text(encoding="utf-8"))
    if feasibility["verdict"] != "pass":
        raise RuntimeError("step 0 did not pass")
    return feasibility


def _forbidden(selection: dict) -> set[str]:
    return {HOLDOUT_PV_GRID, selection["e_test"]}


def _assert_safe_names(paths: list[Path], forbidden: set[str]) -> None:
    for path in paths:
        for grid_id in forbidden:
            if grid_id in path.name or grid_id in path.parts:
                raise RuntimeError(f"holdout path leaked into training: {path}")


def _link_grids(grids: list[str], split: str, root: Path, forbidden: set[str]) -> int:
    if set(grids) & forbidden:
        raise RuntimeError(f"{split} contains a holdout grid: {grids}")
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for grid_id in grids:
        images = sorted((CHIP_ROOT / grid_id / "images").glob("*.png"))
        if len(images) != 2058:
            raise RuntimeError(f"{grid_id} has {len(images)} pngs, expected 2058")
        for image in images:
            stem = f"{grid_id}_{image.stem}"
            image_dest = image_dir / f"{stem}.png"
            label_src = CHIP_ROOT / grid_id / "labels" / f"{image.stem}.txt"
            label_dest = label_dir / f"{stem}.txt"
            if not label_src.exists():
                raise RuntimeError(f"missing label {label_src}")
            if not image_dest.exists():
                image_dest.symlink_to(image.resolve())
            if not label_dest.exists():
                label_dest.symlink_to(label_src.resolve())
            n += 1
    _assert_safe_names(list(image_dir.iterdir()), forbidden)
    return n


def _write_yaml(root: Path) -> Path:
    path = root / "data.yaml"
    path.write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: pv",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _probe_batch(log) -> int:
    probe_root = STEP3_DIR / "batch_probe"
    # Sixty-four chips from a training grid, so a requested batch of 32 is real.
    images = sorted((CHIP_ROOT / TRAIN_PV_GRIDS[0] / "images").glob("*.png"))[:64]
    for batch in (32, 16, 8, 4):
        root = probe_root / f"b{batch}"
        if root.exists():
            for child in root.rglob("*"):
                if child.is_symlink() or child.is_file():
                    child.unlink()
        image_dir = root / "images" / "train"
        label_dir = root / "labels" / "train"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image in images:
            stem = f"{TRAIN_PV_GRIDS[0]}_{image.stem}"
            dest = image_dir / f"{stem}.png"
            if not dest.exists():
                dest.symlink_to(image.resolve())
            src = CHIP_ROOT / TRAIN_PV_GRIDS[0] / "labels" / f"{image.stem}.txt"
            label = label_dir / f"{stem}.txt"
            if not label.exists():
                label.symlink_to(src.resolve())
        # Val has to exist. Point it at the same 64 chips.
        val_img = root / "images" / "val"
        val_lbl = root / "labels" / "val"
        val_img.mkdir(parents=True, exist_ok=True)
        val_lbl.mkdir(parents=True, exist_ok=True)
        for image in image_dir.iterdir():
            target = val_img / image.name
            if not target.exists():
                target.symlink_to(image.resolve())
        for label in label_dir.iterdir():
            target = val_lbl / label.name
            if not target.exists():
                target.symlink_to(label.resolve())
        yaml_path = _write_yaml(root)
        code = subprocess.run(
            [
                str(VENV_PYTHON), "-c",
                (
                    "import sys\n"
                    "from ultralytics import YOLO\n"
                    "model = YOLO('yolo11n.pt')\n"
                    "model.train(data=sys.argv[1], epochs=1, patience=100, batch=int(sys.argv[2]), "
                    "imgsz=416, flipud=0.5, fliplr=0.5, device=0, seed=20260923, "
                    "project=sys.argv[3], name='probe', exist_ok=True, workers=2, "
                    "single_cls=True, cache=False, plots=False, verbose=False)\n"
                ),
                str(yaml_path), str(batch), str(root / "run"),
            ],
            cwd=_ROOT,
        ).returncode
        if code == 0:
            log.info("batch %d fits", batch)
            return batch
        log.info("batch %d did not finish cleanly (exit %s); trying the next lower power of two", batch, code)
    raise RuntimeError("YOLOv11n did not fit at batch 4")


def _locked_config(log) -> dict:
    if CONFIG_PATH.exists():
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        log.info("reusing frozen train config batch=%s", config["batch"])
        return config
    version_text = subprocess.check_output(
        [
            str(VENV_PYTHON), "-c",
            "import torch, ultralytics\n"
            "print('TORCH ' + torch.__version__)\n"
            "print('CUDA ' + str(torch.version.cuda))\n"
            "print('ULTRALYTICS ' + ultralytics.__version__)\n",
        ],
        text=True,
    )
    versions = dict(line.split(" ", 1) for line in version_text.splitlines() if line.startswith(("TORCH ", "CUDA ", "ULTRALYTICS ")))
    config = {
        "model": "yolo11n.pt",
        "imgsz": IMGSZ,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "flipud": 0.5,
        "fliplr": 0.5,
        "batch": _probe_batch(log),
        "conf": CONF,
        "seed": SEED,
        "single_cls": True,
        "workers": 2,
        "cache": False,
        "torch": versions["TORCH"],
        "cuda": versions["CUDA"],
        "ultralytics": versions["ULTRALYTICS"],
        "note": "Ultralytics default augmentations, plus flipud=0.5. fliplr=0.5 is that default made explicit.",
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def _train(yaml_path: Path, run_name: str, config: dict, log) -> Path:
    weights = RUNS_DIR / run_name / "weights" / "best.pt"
    if weights.exists():
        log.info("reuse %s", weights)
        return weights
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    code = subprocess.run(
        [
            str(VENV_PYTHON), "-c",
            (
                "import sys\n"
                "from ultralytics import YOLO\n"
                "model = YOLO('yolo11n.pt')\n"
                "model.train(data=sys.argv[1], epochs=int(sys.argv[2]), patience=int(sys.argv[3]), "
                "batch=int(sys.argv[4]), imgsz=int(sys.argv[5]), flipud=float(sys.argv[6]), "
                "fliplr=float(sys.argv[7]), device=0, seed=int(sys.argv[8]), project=sys.argv[9], "
                "name=sys.argv[10], exist_ok=True, workers=2, single_cls=True, cache=False, plots=False)\n"
            ),
            str(yaml_path),
            str(config["epochs"]),
            str(config["patience"]),
            str(config["batch"]),
            str(config["imgsz"]),
            str(config["flipud"]),
            str(config["fliplr"]),
            str(config["seed"]),
            str(RUNS_DIR),
            run_name,
        ],
        cwd=_ROOT,
        env=env,
    ).returncode
    if code != 0 or not weights.exists():
        raise RuntimeError(f"{run_name} did not write {weights} (exit {code})")
    return weights


def _score_dir(weights: Path, image_dir: Path, config: dict) -> pd.DataFrame:
    script = r"""
import sys
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
weights, image_dir, out_csv, conf, imgsz = sys.argv[1:]
model = YOLO(weights)
rows = []
seen = set()
for result in model.predict(source=image_dir, conf=float(conf), imgsz=int(imgsz), stream=True, verbose=False, device=0):
    stem = Path(result.path).stem
    grid_id, index_s = stem.rsplit("_", 1)
    boxes = result.boxes
    score = 0.0
    if boxes is not None and len(boxes):
        score = float(boxes.conf.max().item())
    rows.append({"grid_id": grid_id, "chip_index": int(index_s), "score": score})
    seen.add(stem)
for path in sorted(Path(image_dir).glob("*.png")):
    if path.stem not in seen:
        grid_id, index_s = path.stem.rsplit("_", 1)
        rows.append({"grid_id": grid_id, "chip_index": int(index_s), "score": 0.0})
pd.DataFrame(rows).to_csv(out_csv, index=False)
"""
    out_csv = STEP3_DIR / f"raw_{image_dir.parent.parent.name}_{image_dir.name}.csv"
    # image_dir is .../images/val, so the name above is not unique. Caller passes a csv path via argv.
    code = subprocess.run(
        [str(VENV_PYTHON), "-c", script, str(weights), str(image_dir), str(out_csv), str(config["conf"]), str(config["imgsz"])],
        cwd=_ROOT,
    ).returncode
    if code != 0:
        raise RuntimeError(f"predict failed for {image_dir}")
    return pd.read_csv(out_csv)


def _attach_labels(scores: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for grid_id, part in scores.groupby("grid_id"):
        manifest = pd.read_csv(CHIP_ROOT / grid_id / "manifest.csv", usecols=["chip_index", "y_pre", "y_post"])
        merged = part.merge(manifest, on="chip_index", how="left")
        if merged["y_pre"].isna().any():
            raise RuntimeError(f"{grid_id} scores did not join to the manifest")
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    log = setup_logging("step3_train")
    feasibility = _load_feasibility()
    selection = feasibility["selection"]
    forbidden = _forbidden(selection)
    train_grids = [*TRAIN_PV_GRIDS, selection["e_train"]]
    if set(train_grids) & forbidden:
        raise RuntimeError("training list overlaps the holdout")
    STEP3_DIR.mkdir(parents=True, exist_ok=True)
    config = _locked_config(log)
    fold_frames = []
    warnings = []
    for left_out in TRAIN_PV_GRIDS:
        others = [grid_id for grid_id in train_grids if grid_id != left_out]
        root = STEP3_DIR / "data" / f"fold_{left_out}"
        _link_grids(others, "train", root, forbidden)
        _link_grids([left_out], "val", root, forbidden)
        yaml_path = _write_yaml(root)
        weights = _train(yaml_path, f"fold_{left_out}", config, log)
        score_path = STEP3_DIR / f"scores_{left_out}.csv"
        if score_path.exists():
            scored = pd.read_csv(score_path)
        else:
            scored = _attach_labels(_score_dir(weights, root / "images" / "val", config))
            scored.to_csv(score_path, index=False)
        if (
            set(scored["grid_id"]) != {left_out}
            or len(scored) != 2058
            or scored.duplicated(["grid_id", "chip_index"]).any()
        ):
            raise RuntimeError(
                f"fold {left_out} scored {len(scored)} rows for grids {set(scored['grid_id'])}"
            )
        warning = fold_warning(scored["y_post"].to_numpy(), scored["score"].to_numpy())
        warning["grid_id"] = left_out
        warnings.append(warning)
        log.info("fold %s pass rate at full post recall %.3f", left_out, warning["pass_rate"])
        fold_frames.append(scored)
    (STEP3_DIR / "fold_warnings.json").write_text(json.dumps(warnings, indent=2) + "\n", encoding="utf-8")
    pooled = pd.concat(fold_frames, ignore_index=True)
    chosen = select_tau(pooled["y_pre"].to_numpy(), pooled["y_post"].to_numpy(), pooled["score"].to_numpy())
    chosen["fold_pass_rate_above_50"] = [row["grid_id"] for row in warnings if row["above_limit"]]
    (STEP3_DIR / "tau.json").write_text(json.dumps(chosen, indent=2) + "\n", encoding="utf-8")
    log.info("tau=%s pre_recall=%.4f post_recall=%.4f pass_rate=%.4f", chosen["tau"], chosen["pre_recall"], chosen["post_recall"], chosen["pass_rate"])

    final_root = STEP3_DIR / "data" / "final"
    _link_grids(train_grids, "train", final_root, forbidden)
    _link_grids(train_grids, "val", final_root, forbidden)
    weights = _train(_write_yaml(final_root), "final", config, log)
    final_scores = _attach_labels(_score_dir(weights, final_root / "images" / "val", config))
    final_scores.to_csv(STEP3_DIR / "scores_final_train.csv", index=False)
    if len(final_scores) != 2058 * len(train_grids) or final_scores.duplicated(["grid_id", "chip_index"]).any():
        raise RuntimeError(f"final scores have {len(final_scores)} rows, expected {2058 * len(train_grids)}")
    post = final_scores["y_post"].to_numpy().astype(bool)
    recall = float((final_scores.loc[post, "score"] >= chosen["tau"]).mean())
    summary = {"train_y_post_recall_at_tau": recall, "tau": chosen["tau"], "n_post": int(post.sum())}
    (STEP3_DIR / "final_train_recall.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if recall < 1.0:
        log.error("final model train y_post recall %.4f; stopping before the holdout", recall)
        return 2
    log.info("final model recalls every training post-positive chip at the locked tau")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import logging
        logging.getLogger("step3_train").exception("step 3 failed")
        raise
