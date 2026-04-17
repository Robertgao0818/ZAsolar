"""Build manifest and dataset summary generation.

Every dataset build produces two provenance files:

- ``build_manifest.json`` — full reproducibility record (spec, source
  hashes, resolved paths, code provenance).
- ``dataset_summary.json`` — quick machine-readable stats (chip counts,
  ratios, filtered counts by reason).

Build IDs are deterministic: ``<spec_name>_<YYYYMMDD>_<short_hash>``
where *short_hash* is the first 8 characters of the SHA-256 of the
fully resolved spec content.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent


def _normalize_path(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# SHA-256 helpers
# ---------------------------------------------------------------------------

def compute_file_sha256(path: Path) -> str:
    """Return hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_string_sha256(text: str) -> str:
    """Return hex SHA-256 digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Build ID
# ---------------------------------------------------------------------------

def generate_build_id(
    spec_name: str,
    build_fingerprint_json: str,
    build_date: datetime | None = None,
) -> str:
    """Generate a deterministic human-readable build ID.

    Format: ``<spec_name>_<YYYYMMDD>_<8-char-hash>``

    The hash is derived from the effective build fingerprint, which should
    include the fully resolved spec plus any dynamic discovery inputs that
    materially affect dataset contents.
    """
    if build_date is None:
        build_date = datetime.now(timezone.utc)
    date_str = build_date.strftime("%Y%m%d")
    short_hash = compute_string_sha256(build_fingerprint_json)[:8]
    return f"{spec_name}_{date_str}_{short_hash}"


# ---------------------------------------------------------------------------
# Git provenance
# ---------------------------------------------------------------------------

def _git_commit_hash() -> str | None:
    """Return current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _git_is_dirty() -> bool | None:
    """Return True if working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


# ---------------------------------------------------------------------------
# Source inventory
# ---------------------------------------------------------------------------

@dataclass
class SourceFileEntry:
    path: str
    sha256: str
    role: str  # "annotation_gpkg", "audit_csv", "manifest_csv", "hn_shortlist_csv"


def build_source_inventory(
    annotation_paths: list[Path],
    audit_csv: Path | None = None,
    manifest_csv: Path | None = None,
    hn_shortlist_csvs: list[Path] | None = None,
) -> list[dict[str, str]]:
    """Compute SHA-256 for all input files and return inventory."""
    entries: list[dict[str, str]] = []

    for p in annotation_paths:
        if p.exists():
            entries.append({
                "path": _normalize_path(p),
                "sha256": compute_file_sha256(p),
                "role": "annotation_gpkg",
            })

    if audit_csv and audit_csv.exists():
        entries.append({
            "path": _normalize_path(audit_csv),
            "sha256": compute_file_sha256(audit_csv),
            "role": "audit_csv",
        })

    if manifest_csv and manifest_csv.exists():
        entries.append({
            "path": _normalize_path(manifest_csv),
            "sha256": compute_file_sha256(manifest_csv),
            "role": "manifest_csv",
        })

    for csv_path in (hn_shortlist_csvs or []):
        if csv_path.exists():
            entries.append({
                "path": str(csv_path),
                "sha256": compute_file_sha256(csv_path),
                "role": "hn_shortlist_csv",
            })

    return entries


# ---------------------------------------------------------------------------
# Build manifest
# ---------------------------------------------------------------------------

def write_build_manifest(
    build_dir: Path,
    *,
    build_id: str,
    spec_path: str,
    resolved_spec: dict[str, Any],
    resolved_spec_hash: str,
    regions: list[str],
    evaluation_regime: str,
    exclude_grids: list[str],
    excluded_grids_reason: str,
    source_inventory: list[dict[str, str]],
    split_strategy: str,
    split_seed: int,
    easy_neg_ratio: float,
    hard_negatives_config: list[dict[str, Any]],
    selected_annotations: list[dict[str, Any]],
    resolved_tile_roots: dict[str, str],
    resolved_output_root: str,
    entrypoint: str = "pipeline.dataset_builder",
) -> Path:
    """Write ``build_manifest.json`` to the build directory."""
    manifest = {
        "build_id": build_id,
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "spec_path": spec_path,
        "resolved_spec": resolved_spec,
        "resolved_spec_hash": resolved_spec_hash,
        "regions": regions,
        "evaluation_regime": evaluation_regime,
        "exclude_grids": exclude_grids,
        "excluded_grids_reason": excluded_grids_reason,
        "source_inventory": source_inventory,
        "split": {
            "strategy": split_strategy,
            "seed": split_seed,
        },
        "negatives": {
            "easy_neg_ratio": easy_neg_ratio,
        },
        "hard_negatives": hard_negatives_config,
        "selected_annotations": selected_annotations,
        "resolved_paths": {
            "tile_roots": resolved_tile_roots,
            "output_root": resolved_output_root,
        },
        "code_provenance": {
            "git_commit": _git_commit_hash(),
            "git_dirty": _git_is_dirty(),
            "entrypoint": entrypoint,
        },
    }

    out_path = build_dir / "build_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out_path


# ---------------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------------

@dataclass
class DatasetSummary:
    """Quick stats for a dataset build."""
    positive_chips: int = 0
    easy_neg_chips: int = 0
    reviewed_fp_hn_chips: int = 0
    small_fp_hn_chips: int = 0
    total_train_images: int = 0
    total_val_images: int = 0
    train_annotations: int = 0
    val_annotations: int = 0
    effective_easy_neg_ratio: float = 0.0
    effective_hn_ratio: float = 0.0
    filtered_counts: dict[str, int] = field(default_factory=dict)
    # filtered_counts keys: "tier_filtered", "audit_filtered",
    # "excluded_grids", "missing_source", "missing_tiles"
    per_region_grid_counts: dict[str, int] = field(default_factory=dict)


def write_dataset_summary(build_dir: Path, summary: DatasetSummary) -> Path:
    """Write ``dataset_summary.json`` to the build directory."""
    data = {
        "positive_chips": summary.positive_chips,
        "easy_neg_chips": summary.easy_neg_chips,
        "reviewed_fp_hn_chips": summary.reviewed_fp_hn_chips,
        "small_fp_hn_chips": summary.small_fp_hn_chips,
        "total_train_images": summary.total_train_images,
        "total_val_images": summary.total_val_images,
        "train_annotations": summary.train_annotations,
        "val_annotations": summary.val_annotations,
        "effective_easy_neg_ratio": round(summary.effective_easy_neg_ratio, 4),
        "effective_hn_ratio": round(summary.effective_hn_ratio, 4),
        "filtered_counts": summary.filtered_counts,
        "per_region_grid_counts": summary.per_region_grid_counts,
    }

    out_path = build_dir / "dataset_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out_path
