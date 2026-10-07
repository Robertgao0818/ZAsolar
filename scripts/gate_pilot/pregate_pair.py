#!/usr/bin/env python3
"""Paired Durban trial: the same grids with the presence pregate off, then on.

The production detector stays unchanged unless this script passes
``--pregate-head``. There is no citywide switch. Imagery is not downloaded
and tiles are not deleted. A small pair cannot show that the gate misses
nothing across the city.

``print-plan`` writes the commands and does not need tiles.
``run`` stops before the first GPU call when a listed grid has no tiles.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import shapely

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.inference.presence_pregate import (  # noqa: E402
    DINO_CACHE,
    TRIAL_THRESHOLD,
    default_head_paths,
)

DETECT = ROOT / "detect_direct.py"
FINALIZE = ROOT / "finalize.py"
DEFAULT_POSTPROC = ROOT / "configs" / "postproc" / "v4_canonical.json"
DEFAULT_LAYER = "vexcel_ortho_2025_12_2026_01"
DEFAULT_OUTPUT = (
    Path.home() / "zasolar_data" / "gate_pilot_2026-09" / "durban_pregate_pair_20261008"
)
SMALL_BATCH_MAX = 10
GRID_RE = re.compile(r"DBN\d{4}")
MISS_AREA_M2 = 0.01
PAIR_READING = (
    "Compare work_s_ratio and process_s_ratio. work_s is the inference-loop "
    "wall clock plus finalize, so it already includes tile reads, the DINO "
    "pregate, detection, and postprocessing. process_s also includes model "
    "load. chip_skip_fraction is only the share of chips that skipped the "
    "detector. It is not the share of time saved. This pair does not show "
    "that the gate misses nothing citywide, and 0.45 was confirmed on the "
    "Cape Town 26-grid pilot rather than on official Vexcel Durban imagery."
)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--grids", nargs="+", required=True, help="Explicit DBN#### ids. No citywide default.")
    parser.add_argument("--model-path", type=Path, required=True, help="Mask R-CNN checkpoint. No default.")
    parser.add_argument("--imagery-layer", default=DEFAULT_LAYER)
    parser.add_argument(
        "--pregate-head", action="append", default=None,
        help="Override the four vit30 fold heads. Repeat for each head.",
    )
    parser.add_argument("--pregate-threshold", type=float, default=TRIAL_THRESHOLD)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--detector-score-threshold", type=float, default=0.05)
    parser.add_argument("--postproc-config", type=Path, default=DEFAULT_POSTPROC)
    parser.add_argument(
        "--allow-larger-batch", action="store_true",
        help=f"Allow more than {SMALL_BATCH_MAX} grids. Still not a citywide switch.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pregate_pair.py",
        description="Run one Durban batch twice and compare detector polygons.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_common(sub.add_parser("print-plan", help="Print the paired commands and exit."))
    _add_common(sub.add_parser("run", help="Run the pair. Refuses missing tiles."))
    return parser


def normalize_grids(raw_grids: list[str], *, allow_larger_batch: bool) -> tuple[str, ...]:
    grids = []
    seen = set()
    for raw in raw_grids:
        grid_id = raw.strip().upper()
        if not GRID_RE.fullmatch(grid_id):
            raise ValueError(f"not a Durban grid id: {raw!r}")
        if grid_id in seen:
            raise ValueError(f"duplicate grid id: {grid_id}")
        seen.add(grid_id)
        grids.append(grid_id)
    if not grids:
        raise ValueError("at least one Durban grid is required")
    if len(grids) > SMALL_BATCH_MAX and not allow_larger_batch:
        raise ValueError(
            f"{len(grids)} grids exceeds the small-batch cap of {SMALL_BATCH_MAX}. "
            "There is no citywide switch. Pass --allow-larger-batch only for a "
            "deliberate larger pair."
        )
    return tuple(grids)


def resolve_heads(raw_heads: list[str] | None) -> tuple[Path, ...]:
    if raw_heads:
        paths = tuple(Path(item).expanduser().resolve() for item in raw_heads)
    else:
        paths = default_head_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "pregate head is missing (this script does not train or download one):\n"
            + "\n".join(missing)
        )
    labels = [path.parent.name for path in paths]
    if any(not label for label in labels) or len(labels) != len(set(labels)):
        raise ValueError(f"pregate head labels must be unique and non-empty: {labels}")
    return paths


def build_plan(args: argparse.Namespace) -> dict:
    grids = normalize_grids(args.grids, allow_larger_batch=args.allow_larger_batch)
    model = Path(args.model_path).expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(f"detector checkpoint is not a file: {model}")
    postproc = Path(args.postproc_config).expanduser()
    if not postproc.is_absolute():
        postproc = ROOT / postproc
    postproc = postproc.resolve()
    if not postproc.is_file():
        raise FileNotFoundError(f"postproc config is not a file: {postproc}")
    heads = resolve_heads(args.pregate_head)
    if not (0.0 <= float(args.pregate_threshold) <= 1.0):
        raise ValueError(f"pregate threshold must be inside [0, 1], got {args.pregate_threshold}")
    output_root = Path(args.output_root).expanduser().resolve()
    steps = []
    for grid_id in grids:
        for arm in ("off", "on"):
            out_dir = output_root / arm / grid_id
            raw_path = out_dir / "raw_detections.pkl"
            detect = [
                sys.executable, str(DETECT),
                "--grid-id", grid_id,
                "--region", "durban",
                "--imagery-layer", args.imagery_layer,
                "--model-run", f"durban_pregate_pair_{arm}",
                "--model-path", str(model),
                "--output-dir", str(out_dir),
                "--batch-size", str(args.batch_size),
                "--num-workers", str(args.num_workers),
                "--device", args.device,
                "--detector-score-threshold", str(args.detector_score_threshold),
                "--stage-timing",
            ]
            if arm == "on":
                for head in heads:
                    detect.extend(["--pregate-head", str(head)])
                detect.extend(["--pregate-threshold", str(args.pregate_threshold)])
            steps.append({
                "kind": "detect",
                "grid_id": grid_id,
                "arm": arm,
                "argv": detect,
                "cwd": str(ROOT),
                "output_dir": str(out_dir),
                "raw_path": str(raw_path),
            })
            steps.append({
                "kind": "finalize",
                "grid_id": grid_id,
                "arm": arm,
                "argv": [
                    sys.executable, str(FINALIZE),
                    "--input", str(raw_path),
                    "--output-dir", str(out_dir),
                    "--postproc-config", str(postproc),
                    "--merge-mode", "per-detection",
                ],
                "cwd": str(ROOT),
                "output_dir": str(out_dir),
                "gpkg_path": str(out_dir / "predictions_metric.gpkg"),
            })
    return {
        "schema": "durban_pregate_pair_plan_v1",
        "citywide": False,
        "downloads_imagery": False,
        "deletes_tiles": False,
        "region": "durban",
        "imagery_layer": args.imagery_layer,
        "grids": list(grids),
        "threshold": float(args.pregate_threshold),
        "threshold_source": "cape_town_26_grid_owner_confirmed_0.45_not_validated_on_durban_vexcel",
        "keep_rule": "max head score >= threshold",
        "heads": [str(path) for path in heads],
        "model_path": str(model),
        "postproc_config": str(postproc),
        "merge_mode": "per-detection",
        "output_root": str(output_root),
        "small_batch_max": SMALL_BATCH_MAX,
        "reading": PAIR_READING,
        "steps": steps,
    }


def _union(geometries) -> shapely.Geometry:
    valid = [geom for geom in geometries if geom is not None and not geom.is_empty]
    if not valid:
        return shapely.geometry.Polygon()
    return shapely.union_all(valid)


def classify_coverage(
    reference,
    other,
    *,
    miss_m2: float = MISS_AREA_M2,
    reduced_m2: float = MISS_AREA_M2,
) -> list[dict]:
    """Classify each reference polygon against the union of the other arm.

    The reference arm is the run with the gate off. ``missed`` means the
    gated union covers at most ``miss_m2``. ``reduced`` means a larger hole
    remains. ``retained`` means the uncovered sliver is within tolerance.
    """
    if miss_m2 < 0 or reduced_m2 < 0:
        raise ValueError("area tolerances must be >= 0")
    covered_by = _union(other)
    rows = []
    for index, geom in enumerate(reference):
        if geom is None or geom.is_empty or float(geom.area) <= 0:
            rows.append({
                "reference_index": index,
                "status": "empty",
                "area_m2": 0.0,
                "intersection_m2": 0.0,
                "uncovered_m2": 0.0,
                "covered_fraction": None,
            })
            continue
        area = float(geom.area)
        intersection = float(geom.intersection(covered_by).area)
        uncovered = area - intersection
        if intersection <= miss_m2:
            status = "missed"
        elif uncovered > reduced_m2:
            status = "reduced"
        else:
            status = "retained"
        rows.append({
            "reference_index": index,
            "status": status,
            "area_m2": area,
            "intersection_m2": intersection,
            "uncovered_m2": uncovered,
            "covered_fraction": intersection / area,
        })
    return rows


def summarize_coverage(rows: list[dict], *, gated_only_area_m2: float | None = None) -> dict:
    def total(status: str, key: str) -> float:
        return float(sum(row[key] for row in rows if row["status"] == status))

    considered = [row for row in rows if row["status"] != "empty"]
    reference_area = float(sum(row["area_m2"] for row in considered))
    covered_area = float(sum(row["intersection_m2"] for row in considered))
    counts = {status: sum(row["status"] == status for row in rows) for status in ("missed", "reduced", "retained", "empty")}
    return {
        "n_reference": counts["missed"] + counts["reduced"] + counts["retained"],
        "n_missed": counts["missed"],
        "n_reduced": counts["reduced"],
        "n_retained": counts["retained"],
        "n_empty": counts["empty"],
        "reference_area_m2": reference_area,
        "covered_area_m2": covered_area,
        "missed_area_m2": total("missed", "area_m2"),
        "reduced_uncovered_m2": total("reduced", "uncovered_m2"),
        "covered_fraction": (covered_area / reference_area) if reference_area else None,
        "gated_only_area_m2": gated_only_area_m2,
        "area_tolerance_m2": MISS_AREA_M2,
        "limitation": "A small-batch pair cannot show that the gate misses nothing citywide.",
    }


def coverage_from_gpkgs(off_path: Path, on_path: Path) -> tuple[list[dict], dict]:
    import geopandas as gpd

    off_path = Path(off_path)
    on_path = Path(on_path)
    if not off_path.is_file() or not on_path.is_file():
        raise FileNotFoundError(f"missing predictions: off={off_path} on={on_path}")
    off = gpd.read_file(off_path, layer="predictions")
    on = gpd.read_file(on_path, layer="predictions")
    if off.crs is None or on.crs is None:
        raise RuntimeError("both prediction layers need a CRS")
    if str(off.crs) != str(on.crs):
        on = on.to_crs(off.crs)
    off_geoms = list(off.geometry)
    on_geoms = list(on.geometry)
    rows = classify_coverage(off_geoms, on_geoms)
    gated_only = float(_union(on_geoms).difference(_union(off_geoms)).area)
    return rows, summarize_coverage(rows, gated_only_area_m2=gated_only)


def _seconds(payload: dict, key: str) -> float:
    value = payload.get(key)
    if value is None:
        raise KeyError(key)
    number = float(value)
    if number < 0:
        raise ValueError(f"{key} must be >= 0")
    return number


def compare_timing(off: dict, on: dict) -> dict:
    """Join one grid's off and on stage records.

    ``work_s`` is loop wall plus finalize. ``process_s`` adds the detect
    subprocess, including model load. Neither ratio is the chip-skip fraction.
    """
    off_loop = _seconds(off, "loop_wall_s")
    on_loop = _seconds(on, "loop_wall_s")
    off_finalize = _seconds(off, "finalize_process_s")
    on_finalize = _seconds(on, "finalize_process_s")
    off_detect = _seconds(off, "detect_process_s")
    on_detect = _seconds(on, "detect_process_s")
    off_work = off_loop + off_finalize
    on_work = on_loop + on_finalize
    off_process = off_detect + off_finalize
    on_process = on_detect + on_finalize
    skip = on.get("chip_skip_fraction")
    return {
        "off_loop_wall_s": off_loop,
        "on_loop_wall_s": on_loop,
        "off_batch_wait_s": off.get("batch_wait_s"),
        "on_batch_wait_s": on.get("batch_wait_s"),
        "off_pregate_s": off.get("pregate_s"),
        "on_pregate_s": on.get("pregate_s"),
        "off_detect_s": off.get("detect_s"),
        "on_detect_s": on.get("detect_s"),
        "off_finalize_process_s": off_finalize,
        "on_finalize_process_s": on_finalize,
        "off_detect_process_s": off_detect,
        "on_detect_process_s": on_detect,
        "off_work_s": off_work,
        "on_work_s": on_work,
        "off_process_s": off_process,
        "on_process_s": on_process,
        "work_s_ratio_on_over_off": (on_work / off_work) if off_work else None,
        "process_s_ratio_on_over_off": (on_process / off_process) if off_process else None,
        "chip_skip_fraction": None if skip is None else float(skip),
        "chip_skip_fraction_is_not_time_saved": True,
        "reading": PAIR_READING,
    }


def assemble_pair_report(grids: list[dict]) -> dict:
    missed = sum(item["coverage"]["n_missed"] for item in grids)
    reduced = sum(item["coverage"]["n_reduced"] for item in grids)
    retained = sum(item["coverage"]["n_retained"] for item in grids)
    reference_area = sum(item["coverage"]["reference_area_m2"] for item in grids)
    covered_area = sum(item["coverage"]["covered_area_m2"] for item in grids)
    off_work = sum(item["timing"]["off_work_s"] for item in grids)
    on_work = sum(item["timing"]["on_work_s"] for item in grids)
    off_process = sum(item["timing"]["off_process_s"] for item in grids)
    on_process = sum(item["timing"]["on_process_s"] for item in grids)
    return {
        "schema": "durban_pregate_pair_report_v1",
        "citywide": False,
        "n_grids": len(grids),
        "n_missed": missed,
        "n_reduced": reduced,
        "n_retained": retained,
        "reference_area_m2": reference_area,
        "covered_area_m2": covered_area,
        "covered_fraction": (covered_area / reference_area) if reference_area else None,
        "off_work_s": off_work,
        "on_work_s": on_work,
        "work_s_ratio_on_over_off": (on_work / off_work) if off_work else None,
        "off_process_s": off_process,
        "on_process_s": on_process,
        "process_s_ratio_on_over_off": (on_process / off_process) if off_process else None,
        "chip_skip_fraction_is_not_time_saved": True,
        "limitation": "A small-batch pair cannot show that the gate misses nothing citywide.",
        "reading": PAIR_READING,
        "grids": grids,
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def existing_raw_paths(plan: dict) -> list[str]:
    found = []
    for step in plan["steps"]:
        if step["kind"] == "detect" and Path(step["raw_path"]).exists():
            found.append(step["raw_path"])
    return found


def missing_tiles(grids: list[str], imagery_layer: str) -> list[str]:
    import argparse

    import detect_direct

    missing = []
    for grid_id in grids:
        probe = argparse.Namespace(grid_id=grid_id, imagery_layer=imagery_layer)
        try:
            detect_direct.resolve_tile_paths(probe, "durban")
        except FileNotFoundError as exc:
            missing.append(f"{grid_id}: {exc}")
    return missing


def _run_step(step: dict) -> float:
    started = time.perf_counter()
    subprocess.run(step["argv"], cwd=step["cwd"], check=True)
    return time.perf_counter() - started


def execute(plan: dict) -> dict:
    if plan.get("downloads_imagery") or plan.get("deletes_tiles") or plan.get("citywide"):
        raise RuntimeError("refusing a plan that downloads, deletes tiles, or enables the gate citywide")
    occupied = existing_raw_paths(plan)
    if occupied:
        raise FileExistsError(
            "refusing to overwrite an existing paired raw artifact:\n" + "\n".join(occupied)
        )
    if not DINO_CACHE.is_dir():
        raise FileNotFoundError(
            f"DINOv2 cache is not a directory: {DINO_CACHE}. "
            "This script does not download weights."
        )
    missing = missing_tiles(plan["grids"], plan["imagery_layer"])
    if missing:
        raise FileNotFoundError(
            "tiles are missing. This script does not download imagery and does not "
            "wait on the pregate before a separate download finishes:\n" + "\n".join(missing)
        )
    process_times: dict[tuple[str, str], dict] = {}
    for step in plan["steps"]:
        elapsed = _run_step(step)
        bucket = process_times.setdefault((step["grid_id"], step["arm"]), {})
        if step["kind"] == "detect":
            bucket["detect_process_s"] = elapsed
        else:
            bucket["finalize_process_s"] = elapsed
            out_dir = Path(step["output_dir"])
            (out_dir / "process_timing.json").write_text(
                json.dumps(bucket, indent=2) + "\n", encoding="utf-8",
            )
    per_grid = []
    for grid_id in plan["grids"]:
        coverage_rows = []
        grid_dir_off = Path(plan["output_root"]) / "off" / grid_id
        grid_dir_on = Path(plan["output_root"]) / "on" / grid_id
        rows, coverage = coverage_from_gpkgs(
            grid_dir_off / "predictions_metric.gpkg",
            grid_dir_on / "predictions_metric.gpkg",
        )
        coverage_rows.extend({"grid_id": grid_id, **row} for row in rows)
        off_stage = _load_json(grid_dir_off / "stage_timing.json")
        on_stage = _load_json(grid_dir_on / "stage_timing.json")
        off_stage.update(process_times[(grid_id, "off")])
        on_stage.update(process_times[(grid_id, "on")])
        per_grid.append({
            "grid_id": grid_id,
            "coverage": coverage,
            "polygons": coverage_rows,
            "timing": compare_timing(off_stage, on_stage),
        })
    report = assemble_pair_report(per_grid)
    output_root = Path(plan["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "pair_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    _write_polygon_csv(output_root / "pair_coverage.csv", per_grid)
    _write_timing_csv(output_root / "pair_timing.csv", per_grid)
    return report


def _write_polygon_csv(path: Path, per_grid: list[dict]) -> None:
    import csv

    fieldnames = [
        "grid_id", "reference_index", "status", "area_m2",
        "intersection_m2", "uncovered_m2", "covered_fraction",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for grid in per_grid:
            for row in grid["polygons"]:
                writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_timing_csv(path: Path, per_grid: list[dict]) -> None:
    import csv

    fieldnames = [
        "grid_id", "chip_skip_fraction", "work_s_ratio_on_over_off",
        "process_s_ratio_on_over_off", "off_work_s", "on_work_s",
        "off_process_s", "on_process_s", "on_pregate_s", "on_detect_s",
        "on_batch_wait_s", "off_detect_s", "off_batch_wait_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for grid in per_grid:
            timing = grid["timing"]
            writer.writerow({"grid_id": grid["grid_id"], **{key: timing.get(key, "") for key in fieldnames if key != "grid_id"}})


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = build_plan(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[pregate_pair] {exc}", file=sys.stderr)
        return 2
    if args.command == "print-plan":
        print(json.dumps(plan, indent=2))
        return 0
    try:
        report = execute(plan)
    except (FileExistsError, FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"[pregate_pair] {exc}", file=sys.stderr)
        return 2
    print(
        "[pregate_pair] grids={n} missed={missed} reduced={reduced} retained={kept} "
        "work_s_ratio={ratio}".format(
            n=report["n_grids"],
            missed=report["n_missed"],
            reduced=report["n_reduced"],
            kept=report["n_retained"],
            ratio=report["work_s_ratio_on_over_off"],
        )
    )
    print(f"[pregate_pair] {report['reading']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
