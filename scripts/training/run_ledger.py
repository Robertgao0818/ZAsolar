"""Training-run ledger: pure provenance recording for ``train.py``.

This module is intentionally **CPU-importable** — it never imports torch — so the
ledger logic can be unit-tested on machines without CUDA (train.py asserts CUDA at
import time and cannot be loaded there).

What it records (Phase 3 of the trainpool-normalization effort):

- ``build_run_manifest(...)`` — assemble a single run manifest dict (run_id,
  dataset provenance, init weights + sha256, seed, hyperparams, boundary-aware
  config, code provenance, metrics, output checkpoints).
- ``extend_training_history(...)`` — fold the run manifest into the existing
  ``training_history.json`` without clobbering the legacy ``history`` /
  ``best_ap50`` / ``best_f1`` keys.
- ``append_training_runs_index(...)`` — append a 1-line lightweight entry to the
  git-tracked index ``configs/training_runs.yaml`` (idempotent on run_id).
- ``write_run_manifest_detail(...)`` — write the full detail to the gitignored
  ``runs/<run_id>/run_manifest.json``.
- ``update_model_registry_training_set(...)`` — set ``training_set_id`` on a
  model-registry entry (opt-in; validated).
- ``seed_everything(...)`` — seed all RNGs (modules passed in / imported lazily,
  so this module stays torch-free for the common code paths).

The dataset ``build_id`` flows from the Phase-2 ``build_manifest.json`` emitted by
``scripts/training/build_unified_reviewall.py`` into the COCO build dir. Legacy
COCO dirs without that file degrade gracefully (null build_id).

Reuses the deterministic-ID and sha256 helpers from ``pipeline.manifests`` — does
not duplicate them.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# Reuse — do NOT duplicate — the deterministic build-id / hashing / git helpers.
from pipeline.manifests import (
    compute_file_sha256,
    compute_string_sha256,
    generate_build_id,
    _git_commit_hash,
    _git_is_dirty,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Git-tracked index of every training run (the only ledger product committed to
# git; the per-run detail under runs/ is gitignored).
DEFAULT_INDEX_PATH = REPO_ROOT / "configs" / "training_runs.yaml"
# Gitignored per-run detail root.
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"
DEFAULT_REGISTRY_PATH = REPO_ROOT / "configs" / "model_registry.yaml"
DEFAULT_BOUNDARY_RULES_PATH = (
    REPO_ROOT / "data" / "training_pool" / "boundary_trust_rules.yaml"
)


# ════════════════════════════════════════════════════════════════════════════
# Seeding (the ONLY new training *behavior*)
# ════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python ``random``, numpy, and torch RNGs.

    torch / numpy are imported lazily so this module remains importable on a
    box without torch. If ``deterministic`` is set, also flip cuDNN to the
    deterministic backend and request deterministic algorithms (guarded — some
    ops have no deterministic kernel and would otherwise raise).
    """
    import random

    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except Exception as exc:  # some ops lack a deterministic impl
                print(f"[SEED] use_deterministic_algorithms unavailable: {exc}")
    except Exception:
        # No torch (e.g. CPU-only ledger unit tests) — python+numpy seeded above.
        pass


def make_dataloader_seeding(seed: int):
    """Return ``(generator, worker_init_fn)`` for a reproducible train loader.

    ``generator`` is a seeded ``torch.Generator`` for shuffle order;
    ``worker_init_fn`` re-seeds numpy + python ``random`` per DataLoader worker
    so augmentation RNG is reproducible across runs. Returns ``(None, None)`` if
    torch is unavailable. Kept here (not in train.py) so train.py stays thin and
    the seeding contract is unit-testable.
    """
    try:
        import torch
    except Exception:
        return None, None

    generator = torch.Generator()
    generator.manual_seed(seed)

    def _worker_init_fn(worker_id: int) -> None:
        import random as _random
        worker_seed = (seed + worker_id) % (2 ** 32)
        try:
            import numpy as _np
            _np.random.seed(worker_seed)
        except Exception:
            pass
        _random.seed(worker_seed)

    return generator, _worker_init_fn


# ════════════════════════════════════════════════════════════════════════════
# Dataset provenance (build_manifest.json → build_id + sha256)
# ════════════════════════════════════════════════════════════════════════════

def _resolve_build_manifest(coco_dir: Path) -> tuple[str | None, str | None]:
    """Return ``(build_id, build_manifest_sha256)`` for a COCO build dir.

    Reads ``<coco_dir>/build_manifest.json`` (Phase-2 product). Legacy dirs
    without it degrade gracefully to ``(None, None)``.
    """
    manifest_path = Path(coco_dir) / "build_manifest.json"
    if not manifest_path.is_file():
        return None, None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None, compute_file_sha256(manifest_path)
    build_id = manifest.get("build_id")
    return build_id, compute_file_sha256(manifest_path)


def _file_sha256_or_none(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    return compute_file_sha256(p) if p.is_file() else None


# ════════════════════════════════════════════════════════════════════════════
# Run manifest assembly
# ════════════════════════════════════════════════════════════════════════════

# CLI hyperparameter knobs captured into the run manifest. Mirrors train.py's
# argparse surface (train.py:788-922). Boundary-aware fields are split into their
# own block per the Phase-3 schema.
HYPERPARAM_KEYS = (
    "lr1",
    "lr2",
    "epochs1",
    "epochs2",
    "batch_size",
    "chip_size",
    "num_workers",
    "no_amp",
    "boundary_band_iters",
    "reinit_mask_head",
    "reinit_box_predictor",
    "diff_lr_backbone_mult",
    "diff_lr_rpn_box_mult",
    "diff_lr_mask_mult",
    "eval_schedule",
    "early_stop_metrics",
    "early_stop_min_delta",
    "early_stop_patience",
    "best_ckpt_bulk_range",
    # C-2 recipe levers (warmup + EMA) — recorded so the manifest fingerprint
    # distinguishes a warmup/EMA run from its legacy-recipe sibling.
    "warmup_iters",
    "warmup_start_factor",
    "ema",
    "ema_decay",
    # C-3(b) recipe lever (boundary ignore band).
    "boundary_ignore_band",
)


def build_run_manifest(
    *,
    coco_dir: str | Path,
    init_weights: str | Path | None,
    seed: int,
    hyperparams: dict[str, Any],
    boundary_aware: dict[str, Any],
    spec_path: str | None = None,
    metrics: dict[str, Any] | None = None,
    output_checkpoints: list[str] | None = None,
    boundary_rules_path: str | Path | None = None,
    build_date: datetime | None = None,
) -> dict[str, Any]:
    """Assemble the run manifest dict.

    ``run_id`` is deterministic: ``generate_build_id`` over a fingerprint of
    dataset build_id + hyperparams + seed + init-weights sha256 + boundary-aware
    config. Timestamps are excluded from the fingerprint so identical inputs
    yield an identical run_id (the date *prefix* still reflects the build date).

    ``metrics.grid_level`` is a null placeholder, back-filled later by
    run-evaluation. Existing chip-level metrics (best_f1/best_ap50/history) flow
    in via ``metrics``.
    """
    coco_dir = Path(coco_dir)
    build_id, build_manifest_sha256 = _resolve_build_manifest(coco_dir)

    init_weights_str = str(init_weights) if init_weights is not None else None
    init_weights_sha256 = _file_sha256_or_none(init_weights)

    if boundary_rules_path is None:
        boundary_rules_path = DEFAULT_BOUNDARY_RULES_PATH
    boundary_rules_sha256 = _file_sha256_or_none(boundary_rules_path)

    hp = {k: hyperparams.get(k) for k in HYPERPARAM_KEYS}

    ba = {
        "per_instance_mask_trusted": boundary_aware.get("per_instance_mask_trusted"),
        "per_source_mask_weight": boundary_aware.get("per_source_mask_weight"),
        "freeze_mask_head": boundary_aware.get("freeze_mask_head"),
        "boundary_trust_rules_sha256": boundary_rules_sha256,
    }

    # Deterministic run-id fingerprint (NO timestamps).
    fingerprint = {
        "dataset_build_id": build_id,
        "dataset_build_manifest_sha256": build_manifest_sha256,
        "init_weights_sha256": init_weights_sha256,
        "seed": seed,
        "hyperparams": hp,
        "boundary_aware": ba,
    }
    fingerprint_json = json.dumps(fingerprint, sort_keys=True)
    spec_name = build_id.split("_")[0] if build_id else "train"
    run_id = generate_build_id(
        f"run_{spec_name}", fingerprint_json, build_date=build_date
    )

    manifest = {
        "run_id": run_id,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "coco_dir": str(coco_dir),
            "build_id": build_id,
            "build_manifest_sha256": build_manifest_sha256,
            "spec_path": spec_path,
        },
        "init_weights": init_weights_str,
        "init_weights_sha256": init_weights_sha256,
        "seed": seed,
        "hyperparams": hp,
        "boundary_aware": ba,
        "code_provenance": {
            "git_commit": _git_commit_hash(),
            "git_dirty": _git_is_dirty(),
            "entrypoint": "train.py",
        },
        "metrics": {
            "chip_level": metrics or {},
            "grid_level": None,  # back-filled by run-evaluation
        },
        "output_checkpoints": list(output_checkpoints or []),
    }
    return manifest


# ════════════════════════════════════════════════════════════════════════════
# training_history.json — fold in run_manifest, preserve legacy keys
# ════════════════════════════════════════════════════════════════════════════

def extend_training_history(
    history_path: str | Path, run_manifest: dict[str, Any]
) -> Path:
    """Add a top-level ``run_manifest`` block to ``training_history.json``.

    Preserves the legacy ``history`` / ``best_ap50`` / ``best_f1`` keys (and any
    other existing keys) untouched. Creates the file if absent.
    """
    history_path = Path(history_path)
    payload: dict[str, Any] = {}
    if history_path.is_file():
        try:
            payload = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    if not isinstance(payload, dict):
        payload = {}

    payload["run_manifest"] = run_manifest

    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return history_path


# ════════════════════════════════════════════════════════════════════════════
# configs/training_runs.yaml — git-tracked lightweight index (idempotent)
# ════════════════════════════════════════════════════════════════════════════

def _index_entry_from_manifest(run_manifest: dict[str, Any]) -> dict[str, Any]:
    chip = (run_manifest.get("metrics") or {}).get("chip_level") or {}
    return {
        "run_id": run_manifest.get("run_id"),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "coco_build_id": (run_manifest.get("dataset") or {}).get("build_id"),
        "seed": run_manifest.get("seed"),
        "git_commit": (run_manifest.get("code_provenance") or {}).get("git_commit"),
        "best_f1": chip.get("best_f1"),
        "best_ap50": chip.get("best_ap50"),
    }


def append_training_runs_index(
    index_path: str | Path, run_manifest: dict[str, Any]
) -> Path:
    """Append a lightweight 1-line entry to ``configs/training_runs.yaml``.

    Creates the file if missing (git-tracked per the index-in-git policy).
    Idempotent on ``run_id`` — re-appending the same run is a no-op.
    """
    index_path = Path(index_path)
    doc: dict[str, Any] = {"runs": []}
    if index_path.is_file():
        try:
            loaded = yaml.safe_load(index_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("runs"), list):
                doc = loaded
        except Exception:
            doc = {"runs": []}

    entry = _index_entry_from_manifest(run_manifest)
    run_id = entry["run_id"]
    existing_ids = {r.get("run_id") for r in doc["runs"] if isinstance(r, dict)}
    if run_id in existing_ids:
        return index_path  # idempotent: already recorded

    doc["runs"].append(entry)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Training Runs Index — git-tracked lightweight ledger.\n"
        "#\n"
        "# One line per completed training run. The full per-run detail lives in\n"
        "# runs/<run_id>/run_manifest.json (gitignored). Appended by\n"
        "# scripts/training/run_ledger.py at the end of train.py. Idempotent on run_id.\n"
    )
    index_path.write_text(
        header + yaml.safe_dump(doc, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return index_path


# ════════════════════════════════════════════════════════════════════════════
# runs/<run_id>/run_manifest.json — full gitignored detail
# ════════════════════════════════════════════════════════════════════════════

def write_run_manifest_detail(
    run_id: str,
    run_manifest: dict[str, Any],
    runs_root: str | Path | None = None,
) -> Path:
    """Write the full run manifest to ``runs/<run_id>/run_manifest.json``."""
    runs_root = Path(runs_root) if runs_root is not None else DEFAULT_RUNS_ROOT
    out_dir = runs_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "run_manifest.json"
    out_path.write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out_path


# ════════════════════════════════════════════════════════════════════════════
# model_registry.yaml — write training_set_id (opt-in, validated)
# ════════════════════════════════════════════════════════════════════════════

def update_model_registry_training_set(
    registry_path: str | Path,
    model_key: str,
    build_id: str | None,
) -> Path:
    """Set ``training_set_id = build_id`` on the registry entry ``model_key``.

    Validated: warns loudly if ``build_id`` is null (legacy COCO dir without a
    build_manifest.json) and raises if ``model_key`` is absent. Preserves the
    other fields of the entry; comment preservation is best-effort only — this
    repo has no ruamel.yaml, so PyYAML round-trips drop top-of-file comments and
    reflow. Caller is expected to gate this behind an opt-in flag.
    """
    registry_path = Path(registry_path)
    if build_id is None:
        print(
            "[REGISTRY][WARN] build_id is null (COCO dir had no "
            "build_manifest.json); writing training_set_id=null — provenance "
            "is incomplete."
        )

    doc = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or "models" not in doc:
        raise ValueError(
            f"{registry_path} has no top-level 'models' mapping; refusing to write."
        )
    models = doc["models"]
    if model_key not in models:
        raise KeyError(
            f"model_key '{model_key}' not found in {registry_path}. "
            f"Available: {sorted(models)}"
        )

    models[model_key]["training_set_id"] = build_id

    # Best-effort comment preservation: re-prepend the original top-of-file
    # comment header (PyYAML drops comments on dump).
    original = registry_path.read_text(encoding="utf-8")
    header_lines: list[str] = []
    for line in original.splitlines():
        if line.startswith("#") or line.strip() == "":
            header_lines.append(line)
        else:
            break
    header = "\n".join(header_lines)
    if header and not header.endswith("\n"):
        header += "\n"

    body = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)
    registry_path.write_text(header + body, encoding="utf-8")
    return registry_path


# ════════════════════════════════════════════════════════════════════════════
# Orchestration helper (called from train.py)
# ════════════════════════════════════════════════════════════════════════════

def record_run(
    *,
    output_dir: str | Path,
    coco_dir: str | Path,
    init_weights: str | Path | None,
    seed: int,
    hyperparams: dict[str, Any],
    boundary_aware: dict[str, Any],
    spec_path: str | None = None,
    metrics: dict[str, Any] | None = None,
    output_checkpoints: list[str] | None = None,
    register_as: str | None = None,
    index_path: str | Path | None = None,
    runs_root: str | Path | None = None,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """One-call wrapper used by train.py.

    Builds the run manifest, folds it into ``<output_dir>/training_history.json``,
    appends the git-tracked index, writes the gitignored per-run detail, and —
    only if ``register_as`` is given — writes ``training_set_id`` into the model
    registry. Returns the run manifest. Callers should wrap this in try/except so
    ledger failure never crashes a finished training run.
    """
    run_manifest = build_run_manifest(
        coco_dir=coco_dir,
        init_weights=init_weights,
        seed=seed,
        hyperparams=hyperparams,
        boundary_aware=boundary_aware,
        spec_path=spec_path,
        metrics=metrics,
        output_checkpoints=output_checkpoints,
    )
    run_id = run_manifest["run_id"]

    history_path = Path(output_dir) / "training_history.json"
    extend_training_history(history_path, run_manifest)

    if index_path is None:
        index_path = DEFAULT_INDEX_PATH
    append_training_runs_index(index_path, run_manifest)

    write_run_manifest_detail(run_id, run_manifest, runs_root=runs_root)

    if register_as:
        if registry_path is None:
            registry_path = DEFAULT_REGISTRY_PATH
        update_model_registry_training_set(
            registry_path,
            register_as,
            run_manifest["dataset"]["build_id"],
        )

    return run_manifest
