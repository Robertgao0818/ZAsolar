#!/usr/bin/env python3
"""Durban citywide inference with the presence pregate on and solar_cls off.

One GPU process loads the Mask R-CNN and the four fold heads once, then
walks every grid. Copying tiles into /dev/shm and ``finalize.py`` run on
the CPU at the same time, so the GPU is not idle while the next grid is
read or the previous grid is polygonized.

The paired trial script stays a 10-grid, two-arm comparison. This driver
does not download imagery, does not delete source tiles, and does not call
the classifier. Launch it inside tmux::

    python -u scripts/gate_pilot/durban_citywide.py print-plan --model-path ...
    tmux new -d -s durban_citywide \\
      'python -u scripts/gate_pilot/durban_citywide.py run --model-path ...'

Re-running skips grids that already have an ok or empty sentinel.
``raw_detections.pkl`` is kept after finalize unless ``--reap-raw`` is set.
A finished grid is still skipped: the ok/empty sentinel is checked before
the raw file, so keeping the pickle does not schedule another finalize.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

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
    Path.home() / "zasolar_data" / "gate_pilot_2026-09" / "durban_citywide_pregate_20261008"
)
DEFAULT_SHM = Path("/dev/shm/durban_pregate_tiles")
GRID_RE = re.compile(r"DBN\d{4}")
EXPECTED_TILES = 4
EMPTY_MARK = "no polygons after vectorization"
ENGINE_NAME = "ENGINE.json"
# One bad grid is retried on the next resume. Five failures in a row means
# the worker itself is broken, and the remaining grids must not be marked fail.
CONSECUTIVE_FAIL_STOP = 5
READING = (
    "One GPU process loads the detector and the presence pregate once. "
    "Tile staging and finalize run on CPU beside that process. Staging keeps "
    "a few grids on a size-capped disk and deletes each copy after finalize; "
    "source tiles stay. solar_cls is not called, and there is no ungated arm. "
    "chip_skip_fraction is the share of chips that skipped the detector, not "
    "the share of time saved. A grid with fewer than 4 tiles is still inferred; "
    "its coverage is partial."
)


def _log(message: str) -> None:
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def default_tile_root() -> Path:
    from core import region_registry

    return Path(region_registry.get_imagery_layer_path("durban", DEFAULT_LAYER))


def discover_grids(tile_root: Path) -> list[dict]:
    """Every DBN directory that already contains at least one geo tif."""
    root = Path(tile_root)
    if not root.is_dir():
        raise FileNotFoundError(f"tile root is not a directory: {root}")
    found = []
    for child in sorted(path for path in root.iterdir() if path.is_dir()):
        if not GRID_RE.fullmatch(child.name):
            continue
        tifs = sorted(child.glob(f"{child.name}_*_geo.tif"))
        if not tifs:
            continue
        found.append({
            "grid_id": child.name,
            "tile_count": len(tifs),
            "complete": len(tifs) == EXPECTED_TILES,
            "bytes": sum(path.stat().st_size for path in tifs),
            "source_dir": str(child),
        })
    if not found:
        raise FileNotFoundError(f"no DBN geo tifs under {root}")
    return found


def sentinel_path(state_dir: Path, grid_id: str, kind: str) -> Path:
    return Path(state_dir) / f"infer_{grid_id}.{kind}"


def terminal_kind(state_dir: Path, grid_id: str) -> str | None:
    for kind in ("ok", "empty"):
        if sentinel_path(state_dir, grid_id, kind).is_file():
            return kind
    return None


def write_sentinel(state_dir: Path, grid_id: str, kind: str) -> None:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    for name in ("ok", "empty", "fail"):
        sentinel_path(state_dir, grid_id, name).unlink(missing_ok=True)
    target = sentinel_path(state_dir, grid_id, kind)
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary.write_text("", encoding="utf-8")
    os.replace(temporary, target)


def assign_actions(grids: list[dict], state_dir: Path, output_root: Path) -> list[dict]:
    """Choose skip, finalize_only, or detect.

    An ok or empty sentinel is skip even when ``raw_detections.pkl`` is
    still in the grid directory. Keeping that pickle after finalize does
    not rerun a finished grid. A pickle without a terminal sentinel is
    finalize_only: detect finished and finalize did not.
    """
    assigned = []
    for grid in grids:
        kind = terminal_kind(state_dir, grid["grid_id"])
        raw = Path(output_root) / grid["grid_id"] / "raw_detections.pkl"
        if kind in ("ok", "empty"):
            action = "skip"
        elif raw.is_file():
            action = "finalize_only"
        else:
            action = "detect"
        assigned.append({**grid, "action": action})
    return assigned


def engine_record(plan: dict) -> dict:
    record = {
        "schema": "durban_citywide_pregate_v1",
        "threshold": plan["threshold"],
        "heads": list(plan["heads"]),
        "model_path": plan["model_path"],
        "postproc_config": plan["postproc_config"],
        "merge_mode": "per-detection",
        "solar_cls": False,
        "gpu_slots": 1,
        "imagery_layer": plan["imagery_layer"],
    }
    # Absent unless rescue is on, so a flags-off run still matches an
    # engine file written before this flag existed.
    if plan.get("pregate_rescue"):
        record["pregate_rescue"] = True
        record["pregate_rescue_min_score"] = float(plan["pregate_rescue_min_score"])
    return record


def check_engine(state_dir: Path, engine: dict) -> None:
    path = Path(state_dir) / ENGINE_NAME
    markers = list(Path(state_dir).glob("infer_*.ok")) + list(Path(state_dir).glob("infer_*.empty"))
    if not path.is_file():
        if markers:
            raise RuntimeError(
                f"{state_dir} has inference sentinels but no {ENGINE_NAME}. "
                "Refusing to mix them with this run."
            )
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous != engine and markers:
        raise RuntimeError(
            "output state belongs to a different engine. "
            f"found {previous}, this run is {engine}. "
            "Use a new --output-root, or remove the old sentinels on purpose."
        )


def _under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return path.resolve() != root.resolve()


def stage_grid(source_dir: Path, dest_dir: Path, shm_root: Path) -> int:
    """Copy one grid's geo tifs onto the staging disk. Source files stay."""
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    shm_root = Path(shm_root)
    if not _under_root(dest_dir, shm_root):
        raise RuntimeError(f"refusing to stage outside {shm_root}: {dest_dir}")
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)
    total = 0
    for source in sorted(source_dir.glob(f"{source_dir.name}_*_geo.tif")):
        temporary = dest_dir / (source.name + ".partial")
        shutil.copyfile(source, temporary)
        os.replace(temporary, dest_dir / source.name)
        total += source.stat().st_size
    if total <= 0:
        raise FileNotFoundError(f"no geo tifs staged from {source_dir}")
    return total


def release_staged(dest_dir: Path, shm_root: Path) -> None:
    """Delete a staged copy. Never deletes the source tree or the shm root."""
    dest_dir = Path(dest_dir)
    shm_root = Path(shm_root)
    if not dest_dir.exists():
        return
    if not _under_root(dest_dir, shm_root):
        raise RuntimeError(f"refusing to delete outside {shm_root}: {dest_dir}")
    shutil.rmtree(dest_dir)


def reap_intermediates(grid_dir: Path, keep_raw: bool = True) -> None:
    """Drop heavy intermediates after polygons exist. Keep the gate table.

    ``raw_detections.pkl`` stays unless ``keep_raw`` is False. ``masks/``
    and ``vectors/`` are removed either way. Resume still skips a grid that
    has an ok or empty sentinel, so leaving the pickle in place does not
    turn a finished grid into finalize_only.
    """
    grid_dir = Path(grid_dir)
    raw = grid_dir / "raw_detections.pkl"
    if raw.exists() and not keep_raw:
        raw.unlink()
    for name in ("masks", "vectors"):
        path = grid_dir / name
        if path.is_dir():
            shutil.rmtree(path)


def classify_finalize_result(returncode: int, log_text: str, gpkg_size: int) -> dict:
    if returncode != 0:
        return {"rc": int(returncode), "empty": False, "error": f"finalize exit {returncode}"}
    if EMPTY_MARK in log_text:
        return {"rc": 0, "empty": True}
    if gpkg_size > 0:
        return {"rc": 0, "empty": False}
    return {"rc": 1, "empty": False, "error": "finalize wrote no geopackage"}


def _detect_args(args: argparse.Namespace, heads: tuple[Path, ...], output_root: Path) -> dict:
    return {
        "grid_id": "DBN0000",
        "region": "durban",
        "imagery_layer": args.imagery_layer,
        "model_run": "durban_citywide_pregate",
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "output_dir": str(output_root),
        "output_raw": None,
        "pregate_head": [str(path) for path in heads],
        "pregate_threshold": float(args.pregate_threshold),
        "pregate_allow_non400": False,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "device": args.device,
        "chip_size": 400,
        "overlap": 0.25,
        "detections_per_img": 300,
        "nms_iou_threshold": 0.5,
        "detector_score_threshold": float(args.detector_score_threshold),
        "mask_threshold": 0.3,
        "raw_mask_storage": "crop",
        "parity_mode": "direct",
        "max_chips": None,
        "hash_tiles_content": False,
        "dry_run_windows": False,
        "stage_timing": True,
        "profile": False,
        "pregate_rescue": bool(getattr(args, "pregate_rescue", False)),
        "pregate_rescue_min_score": float(getattr(args, "pregate_rescue_min_score", 0.3)),
    }


def build_plan(args: argparse.Namespace) -> dict:
    tile_root = Path(args.tile_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    model = Path(args.model_path).expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(f"detector checkpoint is not a file: {model}")
    postproc = Path(args.postproc_config).expanduser()
    if not postproc.is_absolute():
        postproc = ROOT / postproc
    postproc = postproc.resolve()
    if not postproc.is_file():
        raise FileNotFoundError(f"postproc config is not a file: {postproc}")
    if args.pregate_head:
        heads = tuple(Path(item).expanduser().resolve() for item in args.pregate_head)
    else:
        heads = default_head_paths()
    missing = [str(path) for path in heads if not path.is_file()]
    if missing:
        raise FileNotFoundError("pregate head is missing:\n" + "\n".join(missing))
    if not (0.0 <= float(args.pregate_threshold) <= 1.0):
        raise ValueError(f"pregate threshold must be inside [0, 1], got {args.pregate_threshold}")
    if int(args.stage_ahead) < 1:
        raise ValueError("--stage-ahead must be at least 1")
    if int(args.stage_workers) < 1 or int(args.finalize_workers) < 1:
        raise ValueError("worker counts must be at least 1")
    discovered = discover_grids(tile_root)
    if args.grids:
        wanted = []
        seen = set()
        for raw in args.grids:
            grid_id = raw.strip().upper()
            if not GRID_RE.fullmatch(grid_id):
                raise ValueError(f"not a Durban grid id: {raw!r}")
            if grid_id in seen:
                raise ValueError(f"duplicate grid id: {grid_id}")
            seen.add(grid_id)
            wanted.append(grid_id)
        by_id = {grid["grid_id"]: grid for grid in discovered}
        missing_grids = [grid_id for grid_id in wanted if grid_id not in by_id]
        if missing_grids:
            raise FileNotFoundError("tiles are missing for: " + ", ".join(missing_grids))
        discovered = [by_id[grid_id] for grid_id in wanted]
    grids = assign_actions(discovered, output_root / "state", output_root)
    shm_root = None if args.no_stage else Path(args.shm_root).expanduser().resolve()
    plan = {
        "schema": "durban_citywide_pregate_plan_v1",
        "citywide": True,
        "solar_cls": False,
        "off_arm": False,
        "downloads_imagery": False,
        "deletes_source_tiles": False,
        "gpu_slots": 1,
        "region": "durban",
        "imagery_layer": args.imagery_layer,
        "tile_root": str(tile_root),
        "output_root": str(output_root),
        "state_dir": str(output_root / "state"),
        "log_dir": str(output_root / "logs"),
        "model_path": str(model),
        "postproc_config": str(postproc),
        "merge_mode": "per-detection",
        "threshold": float(args.pregate_threshold),
        "keep_rule": "max head score >= threshold",
        "heads": [str(path) for path in heads],
        "stage_tiles": shm_root is not None,
        "shm_root": str(shm_root) if shm_root is not None else "",
        "shm_budget_bytes": int(float(args.shm_budget_gib) * 1024**3),
        "stage_ahead": int(args.stage_ahead),
        "stage_workers": int(args.stage_workers),
        "finalize_workers": int(args.finalize_workers),
        "n_grids": len(grids),
        "n_partial": sum(1 for grid in grids if not grid["complete"]),
        "partial_grid_ids": [grid["grid_id"] for grid in grids if not grid["complete"]],
        "n_skip": sum(1 for grid in grids if grid["action"] == "skip"),
        "n_finalize_only": sum(1 for grid in grids if grid["action"] == "finalize_only"),
        "n_detect": sum(1 for grid in grids if grid["action"] == "detect"),
        "grids": grids,
        "reading": READING,
        # Default keeps raw_detections.pkl. --reap-raw restores the old delete.
        "reap_raw": bool(getattr(args, "reap_raw", False)),
        "pregate_rescue": bool(getattr(args, "pregate_rescue", False)),
        "pregate_rescue_min_score": float(getattr(args, "pregate_rescue_min_score", 0.3)),
    }
    if plan["pregate_rescue"] and not (0.0 <= plan["pregate_rescue_min_score"] <= 1.0):
        raise ValueError(
            "pregate rescue min score must be inside [0, 1], "
            f"got {plan['pregate_rescue_min_score']}"
        )
    plan["detect_args"] = _detect_args(args, heads, output_root)
    plan["engine"] = engine_record(plan)
    return plan


def _die_with_parent() -> None:
    """Ask Linux to signal this process when its parent exits.

    The GPU worker is not a daemon: a daemon cannot start the tile-reading
    workers. Without this, killing the citywide parent would leave the model
    resident on the GPU.
    """
    try:
        import ctypes
        import signal

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        # PR_SET_PDEATHSIG. Failure is not fatal; the parent still joins.
        if libc.prctl(1, int(signal.SIGTERM)) != 0:
            return
    except Exception:
        return


def _forkserver_warmup() -> None:
    return None


def _start_dataloader_forkserver() -> None:
    """Start the tile-worker server before this process touches CUDA."""
    import multiprocessing as mp

    ctx = mp.get_context("forkserver")
    proc = ctx.Process(target=_forkserver_warmup)
    proc.start()
    try:
        proc.join(timeout=60)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)
            raise RuntimeError("dataloader forkserver did not start within 60s")
        if proc.exitcode != 0:
            raise RuntimeError(f"dataloader forkserver exited {proc.exitcode}")
    finally:
        # Same pipe pair as a DataLoader worker. Close it once the warmup exits.
        try:
            proc.close()
        except ValueError:
            pass


def _raise_nofile_limit() -> tuple[int, int]:
    """Lift the soft open-file cap to the hard cap. The hard cap stays put."""
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        except (ValueError, OSError):
            pass
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    return int(soft), int(hard)


def _prepare_gpu_env(shm_root: str) -> None:
    if shm_root:
        os.environ["SOLAR_TILES_ROOT"] = shm_root
    else:
        os.environ.pop("SOLAR_TILES_ROOT", None)
    os.environ["SOLAR_DATALOADER_START"] = "forkserver"


def _boot_gpu_runtime(detect_args: dict):
    """Import the detector, start the forkserver, then load weights."""
    import detect_direct

    _start_dataloader_forkserver()
    template = argparse.Namespace(**detect_args)
    return detect_direct, detect_direct.load_runtime(template)


def gpu_process_main(detect_args: dict, shm_root: str, jobs, results) -> None:
    """Load both models once and detect grids as their ids arrive."""
    _die_with_parent()
    soft, hard = _raise_nofile_limit()
    print(f"[durban_citywide] nofile soft={soft} hard={hard}", flush=True)
    _prepare_gpu_env(shm_root)
    detect_direct, runtime = _boot_gpu_runtime(detect_args)
    try:
        while True:
            grid_id = jobs.get()
            if grid_id is None:
                break
            grid_args = argparse.Namespace(**detect_args)
            grid_args.grid_id = grid_id
            grid_args.output_dir = Path(detect_args["output_dir"]) / grid_id
            grid_args.output_raw = None
            try:
                rc = detect_direct.infer_grid(grid_args, runtime)
                results.put({"grid_id": grid_id, "rc": int(rc), "error": ""})
            except Exception as exc:
                results.put({
                    "grid_id": grid_id,
                    "rc": 1,
                    "error": f"{type(exc).__name__}: {exc}",
                })
    finally:
        detect_direct.close_runtime(runtime)


class GpuClient:
    """One spawned process. The parent never initializes CUDA."""

    def __init__(self, plan: dict):
        import multiprocessing as mp

        self.ctx = mp.get_context("spawn")
        self.jobs = self.ctx.Queue()
        self.results = self.ctx.Queue()
        self.proc = self.ctx.Process(
            target=gpu_process_main,
            args=(plan["detect_args"], plan["shm_root"], self.jobs, self.results),
        )
        self.proc.start()

    def detect(self, grid: dict) -> dict:
        self.jobs.put(grid["grid_id"])
        while True:
            try:
                result = self.results.get(timeout=5)
            except queue.Empty:
                if not self.proc.is_alive():
                    raise RuntimeError(
                        f"GPU worker exited while detecting {grid['grid_id']} "
                        f"(exit {self.proc.exitcode})"
                    )
                continue
            if result.get("grid_id") != grid["grid_id"]:
                raise RuntimeError(f"GPU worker returned an unexpected result: {result}")
            return result

    def close(self) -> None:
        if self.proc.is_alive():
            self.jobs.put(None)
            self.proc.join(timeout=60)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=10)


def finalize_one(plan: dict, grid_id: str) -> dict:
    out_dir = Path(plan["output_root"]) / grid_id
    log_dir = Path(plan["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"finalize_{grid_id}.log"
    argv = [
        sys.executable, str(FINALIZE),
        "--input", str(out_dir / "raw_detections.pkl"),
        "--output-dir", str(out_dir),
        "--postproc-config", plan["postproc_config"],
        "--merge-mode", "per-detection",
        "--allow-overwrite-canonical",
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT, check=False)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    gpkg = out_dir / "predictions_metric.gpkg"
    size = gpkg.stat().st_size if gpkg.is_file() else 0
    result = classify_finalize_result(completed.returncode, text, size)
    if result["rc"] == 0:
        reap_intermediates(out_dir, keep_raw=not bool(plan.get("reap_raw", False)))
    return result


def run_pipeline(plan: dict, *, detect, finalize, stage, release, log=_log) -> dict:
    """Run staging, one GPU detect slot, and CPU finalize side by side.

    ``detect`` and ``finalize`` are called from worker threads. Production
    passes a GPU-process client and a ``finalize.py`` subprocess. Tests pass
    plain functions. Two detects are never in flight.

    The staged copy stays until finalize returns. ``v4_canonical`` reads those
    tiles for per-polygon color, and the artifact records the staged paths.
    Deleting them when detect returns makes every grid that has polygons fail.
    """
    state_dir = Path(plan["state_dir"])
    state_dir.mkdir(parents=True, exist_ok=True)
    # Repair grids that already have a detection dump before detecting new ones.
    pending = deque(grid for grid in plan["grids"] if grid["action"] == "finalize_only")
    pending.extend(grid for grid in plan["grids"] if grid["action"] == "detect")
    stage_ahead = int(plan["stage_ahead"])
    budget = int(plan["shm_budget_bytes"])
    do_stage = bool(plan["stage_tiles"])
    held: dict[str, dict] = {}
    ready: deque[dict] = deque()
    stage_futs: dict = {}
    fin_futs: dict = {}
    detect_fut = None
    detect_grid = None
    counts = {"ok": 0, "empty": 0, "fail": 0, "skip": plan["n_skip"]}
    consecutive_fails = 0
    stopped = False

    def held_bytes() -> int:
        return sum(int(grid["bytes"]) for grid in held.values()) if do_stage else 0

    def finish(grid_id: str, result: dict) -> None:
        nonlocal consecutive_fails, stopped
        if result.get("rc"):
            write_sentinel(state_dir, grid_id, "fail")
            counts["fail"] += 1
            consecutive_fails += 1
            error = result.get("error") or ""
            log(f"fail {grid_id} {error}".rstrip())
            if not stopped and consecutive_fails >= CONSECUTIVE_FAIL_STOP:
                stopped = True
                log(f"stop: {consecutive_fails} consecutive failures")
            return
        if not stopped:
            consecutive_fails = 0
        kind = "empty" if result.get("empty") else "ok"
        write_sentinel(state_dir, grid_id, kind)
        counts[kind] += 1
        done = counts["ok"] + counts["empty"] + counts["fail"]
        log(f"{kind} {grid_id}  finished={done} skip={counts['skip']} / {plan['n_grids']}")

    def release_after(grid: dict | None) -> None:
        if grid is None:
            return
        try:
            release(grid)
        except Exception as exc:
            log(f"release {grid['grid_id']} after finalize: {exc}")

    with ThreadPoolExecutor(max_workers=int(plan["stage_workers"])) as stage_pool, \
            ThreadPoolExecutor(max_workers=1) as detect_pool, \
            ThreadPoolExecutor(max_workers=int(plan["finalize_workers"])) as fin_pool:
        while pending or stage_futs or ready or detect_fut is not None or fin_futs:
            if stopped:
                # Grids not yet detected stay unmarked so a later resume can
                # retry them. Copies already staged are deleted here.
                pending.clear()
                while ready:
                    dropped = ready.popleft()
                    release_after(held.pop(dropped["grid_id"], None))
            progressed = False
            while pending and not stopped and len(held) < stage_ahead:
                grid = pending[0]
                if do_stage and held and held_bytes() + int(grid["bytes"]) > budget:
                    break
                pending.popleft()
                held[grid["grid_id"]] = grid
                progressed = True
                if do_stage:
                    log(f"stage {grid['grid_id']}")
                    stage_futs[grid["grid_id"]] = stage_pool.submit(stage, grid)
                elif grid["action"] == "finalize_only":
                    log(f"finalize {grid['grid_id']}")
                    fin_futs[grid["grid_id"]] = fin_pool.submit(finalize, grid)
                else:
                    ready.append(grid)
            finished_stage = [grid_id for grid_id, fut in stage_futs.items() if fut.done()]
            for grid_id in finished_stage:
                fut = stage_futs.pop(grid_id)
                progressed = True
                grid = held.get(grid_id)
                try:
                    fut.result()
                except Exception as exc:
                    held.pop(grid_id, None)
                    error = f"stage: {exc}"
                    if grid is not None:
                        try:
                            release(grid)
                        except Exception as release_exc:
                            error = f"{error}; release: {release_exc}"
                    finish(grid_id, {"rc": 1, "error": error})
                    continue
                if grid is None:
                    finish(grid_id, {"rc": 1, "error": "stage finished without a held copy"})
                    continue
                if stopped and grid["action"] != "finalize_only":
                    release_after(held.pop(grid_id, None))
                    continue
                if grid["action"] == "finalize_only":
                    log(f"finalize {grid['grid_id']}")
                    fin_futs[grid["grid_id"]] = fin_pool.submit(finalize, grid)
                else:
                    ready.append(grid)
            if detect_fut is None and ready and not stopped:
                detect_grid = ready.popleft()
                log(f"detect {detect_grid['grid_id']}")
                detect_fut = detect_pool.submit(detect, detect_grid)
                progressed = True
            if detect_fut is not None and detect_fut.done():
                grid = detect_grid
                detect_grid = None
                try:
                    result = detect_fut.result()
                except Exception as exc:
                    result = {"rc": 1, "error": f"detect: {exc}"}
                detect_fut = None
                progressed = True
                if result.get("rc"):
                    held.pop(grid["grid_id"], None)
                    try:
                        release(grid)
                    except Exception as exc:
                        result = {"rc": 1, "error": f"release: {exc}"}
                    finish(grid["grid_id"], result)
                else:
                    # Keep the staged tiles. Finalize reads them for color.
                    log(f"finalize {grid['grid_id']}")
                    fin_futs[grid["grid_id"]] = fin_pool.submit(finalize, grid)
            finished_final = [grid_id for grid_id, fut in fin_futs.items() if fut.done()]
            for grid_id in finished_final:
                fut = fin_futs.pop(grid_id)
                progressed = True
                try:
                    result = fut.result()
                except Exception as exc:
                    result = {"rc": 1, "error": f"finalize: {exc}"}
                finish(grid_id, result)
                release_after(held.pop(grid_id, None))
            if not progressed:
                waiting = list(stage_futs.values()) + list(fin_futs.values())
                if detect_fut is not None:
                    waiting.append(detect_fut)
                if not waiting:
                    break
                wait(waiting, timeout=0.2)
    return counts


def _production_stage(plan: dict, grid: dict) -> int:
    return stage_grid(
        Path(grid["source_dir"]),
        Path(plan["shm_root"]) / grid["grid_id"],
        Path(plan["shm_root"]),
    )


def _production_release(plan: dict, grid: dict) -> None:
    if not plan["stage_tiles"]:
        return
    release_staged(Path(plan["shm_root"]) / grid["grid_id"], Path(plan["shm_root"]))


def execute(plan: dict, *, detect=None, finalize=None, stage=None, release=None) -> dict:
    if plan.get("solar_cls") or plan.get("off_arm") or plan.get("downloads_imagery"):
        raise RuntimeError("this driver does not run solar_cls, an ungated arm, or a download")
    if int(plan.get("gpu_slots", 0)) != 1:
        raise RuntimeError("this driver keeps a single GPU worker")
    output_root = Path(plan["output_root"])
    state_dir = Path(plan["state_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    Path(plan["log_dir"]).mkdir(parents=True, exist_ok=True)
    check_engine(state_dir, plan["engine"])
    (state_dir / ENGINE_NAME).write_text(
        json.dumps(plan["engine"], indent=2) + "\n", encoding="utf-8",
    )
    if plan["stage_tiles"]:
        shm_root = Path(plan["shm_root"])
        shm_root.mkdir(parents=True, exist_ok=True)
    if not DINO_CACHE.is_dir():
        raise FileNotFoundError(
            f"DINOv2 cache is not a directory: {DINO_CACHE}. This script does not download weights."
        )
    _log(
        "grids={n_grids} detect={n_detect} finalize_only={n_finalize_only} "
        "skip={n_skip} partial={n_partial}".format(**plan)
    )
    summary = {key: plan[key] for key in (
        "n_grids", "n_partial", "partial_grid_ids", "n_skip", "n_detect",
        "n_finalize_only", "threshold", "gpu_slots", "stage_tiles", "shm_root",
        "reading",
    )}
    (output_root / "citywide_plan.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    client = None
    if detect is None and plan["n_detect"]:
        client = GpuClient(plan)
        detect = client.detect
    if finalize is None:
        finalize = lambda grid: finalize_one(plan, grid["grid_id"])
    if stage is None:
        stage = lambda grid: _production_stage(plan, grid)
    if release is None:
        release = lambda grid: _production_release(plan, grid)
    try:
        counts = run_pipeline(
            plan, detect=detect, finalize=finalize, stage=stage, release=release,
        )
    finally:
        if client is not None:
            client.close()
    (output_root / "citywide_counts.json").write_text(
        json.dumps(counts, indent=2) + "\n", encoding="utf-8",
    )
    return counts


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tile-root", type=Path, default=None,
                        help="Directory of DBN####/ tile folders. Defaults to the Durban registry layer.")
    parser.add_argument("--grids", nargs="*", default=None,
                        help="Optional subset. Default is every grid that has a geo tif.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--imagery-layer", default=DEFAULT_LAYER)
    parser.add_argument("--pregate-head", action="append", default=None)
    parser.add_argument("--pregate-threshold", type=float, default=TRIAL_THRESHOLD)
    parser.add_argument("--pregate-rescue", action="store_true",
                        help="Re-run the detector on a rejected chip when a kept "
                             "neighbour detection overlaps its window. Off by default.")
    parser.add_argument("--pregate-rescue-min-score", type=float, default=0.3,
                        help="Score a kept detection needs to trigger --pregate-rescue.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--postproc-config", type=Path, default=DEFAULT_POSTPROC)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--detector-score-threshold", type=float, default=0.05)
    parser.add_argument("--no-stage", action="store_true",
                        help="Read tiles from the registry path. Use only on fast local disk.")
    parser.add_argument("--shm-root", type=Path, default=DEFAULT_SHM)
    parser.add_argument("--shm-budget-gib", type=float, default=20.0)
    parser.add_argument("--stage-ahead", type=int, default=3,
                        help="How many grids may occupy staging space, including the one on the GPU.")
    parser.add_argument("--stage-workers", type=int, default=2)
    parser.add_argument("--finalize-workers", type=int, default=2)
    parser.add_argument("--reap-raw", action="store_true",
                        help="Delete raw_detections.pkl after a successful finalize. "
                             "Default keeps it so postproc can be rerun on CPU. "
                             "masks/ and vectors/ are still removed.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="durban_citywide.py",
        description="Durban citywide detect_direct with the presence pregate. No solar_cls.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_common(sub.add_parser("print-plan", help="Show the grid list and engine, then exit."))
    _add_common(sub.add_parser("run", help="Run the city. Resume skips finished grids."))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.tile_root is None:
        args.tile_root = default_tile_root()
    try:
        plan = build_plan(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[durban_citywide] {exc}", file=sys.stderr)
        return 2
    if args.command == "print-plan":
        shown = dict(plan)
        shown.pop("detect_args", None)
        print(json.dumps(shown, indent=2))
        return 0
    try:
        counts = execute(plan)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[durban_citywide] {exc}", file=sys.stderr)
        return 2
    _log(
        "done ok={ok} empty={empty} fail={fail} skip={skip}".format(**counts)
    )
    print(f"[durban_citywide] {READING}", flush=True)
    return 0 if counts["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
