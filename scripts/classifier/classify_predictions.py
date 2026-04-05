"""
Post-hoc PV vs non-PV classification of detection predictions.

Reads predictions_metric.gpkg, crops each small detection from source tiles,
classifies with trained binary classifier, and writes parallel output files
alongside the originals (never overwriting).

Usage:
    # Single grid
    python scripts/classifier/classify_predictions.py \
        --grid-id G1238 \
        --model-path checkpoints/cls_pv_thermal/best_cls.pth

    # Batch mode
    python scripts/classifier/classify_predictions.py \
        --grid-ids G1238 G1689 G1690 \
        --model-path checkpoints/cls_pv_thermal/best_cls.pth

    # With config file
    python scripts/classifier/classify_predictions.py \
        --grid-id G1238 \
        --cls-config configs/classifier/cls_pv_thermal.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.grid_utils import TILES_ROOT

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# Same as build_cls_dataset.py
CHIP_SIZE = 400
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_cls_config(config_path: Path) -> dict:
    """Load classifier config, returning defaults for missing keys."""
    with open(config_path) as f:
        cfg = json.load(f)
    return cfg


def build_model(arch: str, num_classes: int = 2) -> nn.Module:
    """Build model architecture (weights loaded separately)."""
    from torchvision import models
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return model


def load_classifier(model_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    """Load trained classifier from checkpoint. Returns (model, config)."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # Try to load config from sibling config.json
    config_path = model_path.parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {"arch": "efficientnet_b0", "img_size": 224, "num_classes": 2}

    arch = config.get("arch", "efficientnet_b0")
    model = build_model(arch, config.get("num_classes", 2))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    return model, config


class ChipDataset(Dataset):
    """Dataset of cropped detection chips for classification."""

    def __init__(
        self,
        chips: list[np.ndarray],
        indices: list[int],
        img_size: int = IMG_SIZE,
    ):
        self.chips = chips
        self.indices = indices
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.chips)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        chip = self.chips[idx]
        # Resize to classifier input size
        chip = cv2.resize(chip, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_AREA)
        tensor = self.transform(chip)
        return tensor, self.indices[idx]


def _find_tile(lon: float, lat: float, grid_id: str,
               tiles_root: Path) -> Path | None:
    """Find tile GeoTIFF containing a lon/lat point."""
    import rasterio
    grid_dir = tiles_root / grid_id
    if not grid_dir.exists():
        return None
    for tif in grid_dir.glob(f"{grid_id}_*_*_geo.tif"):
        with rasterio.open(tif) as src:
            left, bottom, right, top = src.bounds
            if left <= lon <= right and bottom <= lat <= top:
                return tif
    return None


def extract_detection_chips(
    pred_gdf: gpd.GeoDataFrame,
    grid_id: str,
    tiles_root: Path,
    area_cutoff: float,
    chip_size: int = CHIP_SIZE,
) -> tuple[list[np.ndarray], list[int], list[int]]:
    """Extract chips for detections below area cutoff.

    Returns (chips, classified_indices, skipped_indices).
    """
    import rasterio
    from rasterio.windows import Window

    # Reproject to EPSG:4326 for tile lookup
    if pred_gdf.crs and pred_gdf.crs.to_epsg() != 4326:
        pred_4326 = pred_gdf.to_crs(epsg=4326)
    else:
        pred_4326 = pred_gdf

    chips = []
    classified_indices = []
    skipped_indices = []
    tile_cache: dict[str, rasterio.DatasetReader] = {}

    try:
        for idx in pred_gdf.index:
            row = pred_gdf.loc[idx]
            area = row.get("area_m2", 0)

            if area >= area_cutoff:
                continue  # large detections bypass classifier

            row_4326 = pred_4326.loc[idx]
            centroid = row_4326.geometry.centroid
            lon, lat = centroid.x, centroid.y

            tile_path = _find_tile(lon, lat, grid_id, tiles_root)
            if tile_path is None:
                skipped_indices.append(idx)
                continue

            tile_key = str(tile_path)
            if tile_key not in tile_cache:
                tile_cache[tile_key] = rasterio.open(tile_path)

            src = tile_cache[tile_key]
            py, px = src.index(lon, lat)

            x0 = max(0, int(px - chip_size // 2))
            y0 = max(0, int(py - chip_size // 2))
            x0 = min(x0, max(0, src.width - chip_size))
            y0 = min(y0, max(0, src.height - chip_size))

            w = min(chip_size, src.width - x0)
            h = min(chip_size, src.height - y0)

            if w < chip_size * 0.5 or h < chip_size * 0.5:
                skipped_indices.append(idx)
                continue

            window = Window(x0, y0, w, h)
            data = src.read(window=window)

            if w < chip_size or h < chip_size:
                padded = np.zeros(
                    (data.shape[0], chip_size, chip_size), dtype=data.dtype
                )
                padded[:, :h, :w] = data
                data = padded

            if np.all(data >= 245):
                skipped_indices.append(idx)
                continue

            img = data[:3].transpose(1, 2, 0)  # CHW -> HWC
            chips.append(img)
            classified_indices.append(idx)

    finally:
        for handle in tile_cache.values():
            handle.close()

    return chips, classified_indices, skipped_indices


@torch.no_grad()
def classify_chips(
    model: nn.Module,
    chips: list[np.ndarray],
    indices: list[int],
    device: torch.device,
    img_size: int = IMG_SIZE,
    batch_size: int = 64,
) -> dict[int, float]:
    """Run classifier on chips. Returns {index: pv_probability}."""
    if not chips:
        return {}

    dataset = ChipDataset(chips, indices, img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    results = {}
    softmax = nn.Softmax(dim=1)

    for batch_tensors, batch_indices in loader:
        batch_tensors = batch_tensors.to(device)
        outputs = model(batch_tensors)
        probs = softmax(outputs)

        # Assumes class 1 = "pv" (ImageFolder sorts alphabetically: non_pv=0, pv=1)
        pv_probs = probs[:, 1].cpu().numpy()

        for i, idx in enumerate(batch_indices.numpy()):
            results[int(idx)] = float(pv_probs[i])

    return results


def classify_grid(
    grid_id: str,
    model: nn.Module,
    device: torch.device,
    tiles_root: Path,
    results_dir: Path,
    pv_threshold: float = 0.5,
    area_cutoff: float = 30.0,
    img_size: int = 224,
    batch_size: int = 64,
) -> dict | None:
    """Classify predictions for one grid. Returns summary dict or None on error."""
    pred_path = results_dir / grid_id / "predictions_metric.gpkg"
    if not pred_path.exists():
        print(f"  WARN: {pred_path} not found, skipping")
        return None

    pred_gdf = gpd.read_file(pred_path)
    original_crs = pred_gdf.crs
    total = len(pred_gdf)
    if total == 0:
        print(f"  {grid_id}: 0 detections, skipping")
        return None

    print(f"  {grid_id}: {total} detections, extracting chips (area < {area_cutoff} m²)...")

    # Extract chips
    chips, classified_idx, skipped_idx = extract_detection_chips(
        pred_gdf, grid_id, tiles_root, area_cutoff,
    )

    print(f"    Classified: {len(classified_idx)}, "
          f"skipped: {len(skipped_idx)}, "
          f"large (bypassed): {total - len(classified_idx) - len(skipped_idx)}")

    # Classify
    pv_scores = classify_chips(model, chips, classified_idx, device, img_size, batch_size)

    # Assign scores to all predictions
    # Default: large detections bypass classifier (assume PV)
    pred_gdf["cls_score"] = 1.0
    pred_gdf["cls_label"] = "pv"
    pred_gdf["cls_applied"] = False

    # Detections that were actually classified by the model
    for idx, score in pv_scores.items():
        pred_gdf.at[idx, "cls_score"] = score
        pred_gdf.at[idx, "cls_label"] = "pv" if score >= pv_threshold else "non_pv"
        pred_gdf.at[idx, "cls_applied"] = True

    # Skipped small detections: extraction failed (no tile / blank / too small)
    # NOT classified — keep cls_applied=False, cls_score=1.0 (benefit of doubt)

    # Count results
    n_actually_classified = len(pv_scores)
    n_large_bypassed = total - len(classified_idx) - len(skipped_idx)
    n_extraction_failed = len(skipped_idx)
    n_pv = (pred_gdf["cls_label"] == "pv").sum()
    n_non_pv = (pred_gdf["cls_label"] == "non_pv").sum()

    print(f"    Actually classified: {n_actually_classified}, "
          f"extraction failed: {n_extraction_failed}, "
          f"large bypassed: {n_large_bypassed}")
    print(f"    Results: {n_pv} PV, {n_non_pv} non-PV "
          f"(removed {n_non_pv} / {n_actually_classified} classified)")

    # --- Save outputs ---
    out_dir = results_dir / grid_id

    # Full annotated gpkg (EPSG:32734)
    pred_gdf.to_file(str(out_dir / "predictions_metric_cls.gpkg"), driver="GPKG")

    # EPSG:4326 export
    export_gdf = pred_gdf.to_crs(epsg=4326) if pred_gdf.crs and pred_gdf.crs.to_epsg() != 4326 else pred_gdf
    export_gdf.to_file(str(out_dir / "predictions_cls.geojson"), driver="GeoJSON")

    # Filtered (PV only)
    filtered = pred_gdf[pred_gdf["cls_label"] == "pv"].copy()
    filtered.to_file(str(out_dir / "predictions_metric_cls_filtered.gpkg"), driver="GPKG")

    # Summary JSON
    summary = {
        "grid_id": grid_id,
        "model_path": str(model_path_global),
        "pv_threshold": pv_threshold,
        "area_cutoff_m2": area_cutoff,
        "total_detections": total,
        "classified_count": n_actually_classified,
        "extraction_failed_count": n_extraction_failed,
        "large_bypassed_count": n_large_bypassed,
        "pv_count": int(n_pv),
        "non_pv_count": int(n_non_pv),
        "filtered_count": len(filtered),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / "cls_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# Global reference for summary (set in main)
model_path_global: Path = Path(".")


def main():
    global model_path_global

    parser = argparse.ArgumentParser(
        description="Classify predictions as PV vs non-PV"
    )
    parser.add_argument("--grid-id", type=str, default=None,
                        help="Single grid ID to process")
    parser.add_argument("--grid-ids", nargs="+", default=None,
                        help="Multiple grid IDs to process")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Path to classifier checkpoint")
    parser.add_argument("--cls-config", type=Path, default=None,
                        help="Classifier config JSON (defaults for model-path, threshold, etc.)")
    parser.add_argument("--pv-threshold", type=float, default=None,
                        help="P(PV) threshold for keeping detection (default: 0.5)")
    parser.add_argument("--area-cutoff", type=float, default=None,
                        help="Only classify detections below this area (default: 30)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tiles-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()

    # Load config defaults, then override with CLI args
    config = {}
    if args.cls_config and args.cls_config.exists():
        config = load_cls_config(args.cls_config)
        print(f"Loaded config: {args.cls_config}")

    model_path = args.model_path or Path(config.get("classifier_model", ""))
    pv_threshold = args.pv_threshold if args.pv_threshold is not None else config.get("pv_threshold", 0.5)
    area_cutoff = args.area_cutoff if args.area_cutoff is not None else config.get("area_cutoff_m2", 30.0)
    img_size = config.get("classifier_img_size", 224)
    tiles_root = args.tiles_root or TILES_ROOT
    results_dir = args.results_dir or RESULTS_DIR

    model_path_global = model_path

    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    # Determine grid list
    grid_ids = []
    if args.grid_id:
        grid_ids = [args.grid_id]
    elif args.grid_ids:
        grid_ids = args.grid_ids
    else:
        print("ERROR: Specify --grid-id or --grid-ids")
        sys.exit(1)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading classifier from {model_path} (device={device})...")
    model, model_config = load_classifier(model_path, device)
    img_size = model_config.get("img_size", img_size)
    print(f"  Arch: {model_config.get('arch', '?')}, "
          f"threshold={pv_threshold}, area_cutoff={area_cutoff}")

    # Process each grid
    summaries = []
    for grid_id in grid_ids:
        summary = classify_grid(
            grid_id, model, device, tiles_root, results_dir,
            pv_threshold=pv_threshold,
            area_cutoff=area_cutoff,
            img_size=img_size,
            batch_size=args.batch_size,
        )
        if summary:
            summaries.append(summary)

    # Print overall summary
    if summaries:
        total_orig = sum(s["total_detections"] for s in summaries)
        total_filtered = sum(s["filtered_count"] for s in summaries)
        total_removed = sum(s["non_pv_count"] for s in summaries)
        print(f"\n=== Overall ===")
        print(f"  Grids processed: {len(summaries)}")
        print(f"  Original detections: {total_orig}")
        print(f"  After classifier filter: {total_filtered} "
              f"(removed {total_removed}, {total_removed/total_orig*100:.1f}%)")


if __name__ == "__main__":
    main()
