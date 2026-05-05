#!/usr/bin/env python3
# DEPRECATED 2026-05-05: Migrated to solar_backdating subrepo (V1.4 sub-line pivot).
# Authoritative copy: /home/gaosh/projects/solar_backdating/scripts/temporal/download_geid_historical_direct.py
# This file is frozen; scheduled for removal after 2026-05-31. Bug fixes go to subrepo first.
"""Experimental direct downloader for GEID historical tiles.

This bypasses the slow per-task ``downloader.exe`` loop once historical
date-version pairs are known or can be learned from seed ``*_datever.dat``
files.  It preserves the GEID output layout used by ``score_anchor_presence.py``:

    <save_to>/<task_name>/<zoom>/<x>/gesh_<x>_<y>_<zoom>.jpg

Historical URL shape verified from GEID 6.48 SSL captures:

    https://khmdb.google.com/flatfile?db=tm&f1-<quadkey>-i.<img_ver>-<date_ver_hex>

Generic workflow:
    1. Run a tiny GEID CLI seed for each requested date (or reuse a partial run).
    2. Run this script on the full task CSV.  It learns date-version values
       from existing seed/partial ``*_datever.dat`` files and downloads the
       remaining tiles directly and concurrently.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import struct
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RE_ROOT = PROJECT_ROOT.parent / "geid_reverse_engineering"
DEFAULT_CIPHER_KEY = RE_ROOT / "artifacts" / "cipher_key.bin"

USER_AGENT = (
    "GoogleEarth/7.3.6.9345(Windows;Microsoft Windows (6.2.9200.0);"
    "en;kml:2.2;client:Pro;type:default)"
)
GE_DIGIT_MAP = (0, 3, 1, 2)
JPEG_SOI = b"\xff\xd8\xff"
COMMENT_RE = re.compile(rb"\*AD\*(\d{4}):(\d{2}):(\d{2})\*")


@dataclass(frozen=True)
class DateVersion:
    date_ver: int
    img_ver: int

    @property
    def suffix(self) -> str:
        return f"i.{self.img_ver}-{self.date_ver:x}"


@dataclass(frozen=True)
class TileJob:
    task_idx: int
    task_name: str
    requested_date: str
    x: int
    y: int
    z: int
    out_path: Path
    date_version: DateVersion


@dataclass
class VersionIndex:
    """Learned historical date-version values from GEID ``*_datever.dat`` files."""

    exact: dict[tuple[str, int, int, int], DateVersion]
    parent: dict[tuple[str, int, str], Counter[DateVersion]]
    by_date: dict[str, Counter[DateVersion]]
    sources: list[dict[str, object]]

    def add(self, *, requested_date: str, x: int, y: int, z: int, date_version: DateVersion, source: str) -> None:
        self.exact[(requested_date, z, x, y)] = date_version
        self.parent.setdefault((requested_date, z, quadkey(x, y, z)[:-1]), Counter())[date_version] += 1
        self.by_date.setdefault(requested_date, Counter())[date_version] += 1
        self.sources.append(
            {
                "requested_date": requested_date,
                "x": x,
                "y": y,
                "z": z,
                "date_ver": date_version.date_ver,
                "date_ver_hex": f"{date_version.date_ver:x}",
                "img_ver": date_version.img_ver,
                "source": source,
            }
        )


def new_version_index() -> VersionIndex:
    return VersionIndex(exact={}, parent={}, by_date={}, sources=[])


def windows_path_to_wsl(path: str | os.PathLike[str]) -> Path | None:
    text = str(path).strip().strip('"')
    match = re.match(r"^([A-Za-z]):[\\/]*(.*)$", text)
    if not match:
        return None
    drive = match.group(1).lower()
    rest = match.group(2).replace("\\", "/")
    return Path("/mnt") / drive / rest if rest else Path("/mnt") / drive


def resolve_save_root(save_to: str | os.PathLike[str]) -> Path:
    return windows_path_to_wsl(save_to) or Path(save_to)


def quadkey(x: int, y: int, z: int) -> str:
    out: list[str] = []
    for i in range(z - 1, -1, -1):
        x_bit = (x >> i) & 1
        y_bit = (y >> i) & 1
        out.append(str(GE_DIGIT_MAP[(x_bit << 1) | y_bit]))
    return "".join(out)


def bbox_to_tile_range(lon_min: float, lon_max: float, lat_min: float, lat_max: float, z: int) -> tuple[int, int, int, int]:
    n = 2 ** (z - 1)
    factor = n / 360.0
    x_min = int((lon_min + 180.0) * factor)
    x_max = int((lon_max + 180.0) * factor)
    y_min = int((lat_min + 180.0) * factor)
    y_max = int((lat_max + 180.0) * factor)
    return x_min, x_max, y_min, y_max


def iter_tile_coords(task: Mapping[str, str]) -> Iterable[tuple[int, int, int]]:
    z_from = int(task.get("zoom_from") or 21)
    z_to = int(task.get("zoom_to") or z_from)
    left = float(task["left_longitude"])
    right = float(task["right_longitude"])
    top = float(task["top_latitude"])
    bottom = float(task["bottom_latitude"])
    for z in range(z_from, z_to + 1):
        x_min, x_max, y_min, y_max = bbox_to_tile_range(left, right, bottom, top, z)
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                yield x, y, z


def parse_date_version(value: str) -> tuple[str, DateVersion]:
    """Parse DATE=DATEVER:IMGVER or DATE=DATEVER,IMGVER."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("date version must be DATE=DATEVER:IMGVER")
    date_text, rest = value.split("=", 1)
    parts = re.split(r"[:,]", rest)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("date version must be DATE=DATEVER:IMGVER")
    date_ver = int(parts[0], 0)
    img_ver = int(parts[1], 0)
    return date_text.strip()[:10], DateVersion(date_ver=date_ver, img_ver=img_ver)


def read_tasks_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        return [dict(row) for row in csv.DictReader(fh) if row.get("task_name")]


def datever_path_for_task(task: Mapping[str, str]) -> Path:
    task_name = str(task["task_name"])
    task_root = resolve_save_root(task["save_to"]) / task_name
    return task_root.parent / f"{task_name}_datever.dat"


def is_valid_date_version(date_version: DateVersion | None) -> bool:
    return date_version is not None and date_version.date_ver != 0xFFFFFFFF and date_version.img_ver != 0xFFFFFFFF


def read_datever_entries(path: Path, expected: int) -> list[DateVersion | None] | None:
    if not path.exists():
        return None
    raw = path.read_bytes()
    if len(raw) < expected * 8:
        return None
    out: list[DateVersion | None] = []
    for i in range(expected):
        date_ver, img_ver = struct.unpack("<II", raw[i * 8 : i * 8 + 8])
        item = DateVersion(date_ver=date_ver, img_ver=img_ver)
        out.append(item if is_valid_date_version(item) else None)
    return out


def write_datever(path: Path, versions: list[DateVersion]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b"".join(struct.pack("<II", item.date_ver, item.img_ver) for item in versions)
    path.write_bytes(raw)


def extract_capture_date(data: bytes) -> str:
    match = COMMENT_RE.search(data[:4096])
    if not match:
        return ""
    return f"{match.group(1).decode()}-{match.group(2).decode()}-{match.group(3).decode()}"


def _single_counter_value(counter: Counter[DateVersion]) -> DateVersion | None:
    if len(counter) != 1:
        return None
    return next(iter(counter))


def learn_versions_from_tasks(tasks: Sequence[Mapping[str, str]]) -> VersionIndex:
    index = new_version_index()
    for task in tasks:
        task_name = str(task["task_name"])
        requested_date = str(task.get("date") or task.get("requested_date") or "")[:10]
        coords = list(iter_tile_coords(task))
        entries = read_datever_entries(datever_path_for_task(task), len(coords))
        if entries is None:
            continue
        source = str(datever_path_for_task(task))
        n_valid = 0
        for (x, y, z), item in zip(coords, entries):
            if item is None:
                continue
            index.add(requested_date=requested_date, x=x, y=y, z=z, date_version=item, source=source)
            n_valid += 1
        if n_valid:
            index.sources.append({"task_name": task_name, "requested_date": requested_date, "valid_entries": n_valid, "source": source})
    return index


def resolve_learned_version(
    *,
    requested_date: str,
    x: int,
    y: int,
    z: int,
    index: VersionIndex,
    date_overrides: Mapping[str, DateVersion],
    allow_date_fallback: bool,
) -> tuple[DateVersion | None, str]:
    exact = index.exact.get((requested_date, z, x, y))
    if exact is not None:
        return exact, "learned_exact_tile"

    parent_key = (requested_date, z, quadkey(x, y, z)[:-1])
    parent = index.parent.get(parent_key)
    if parent:
        parent_value = _single_counter_value(parent)
        if parent_value is not None:
            return parent_value, "learned_parent_quadkey"
        return None, "ambiguous_parent_quadkey"

    if requested_date in date_overrides:
        return date_overrides[requested_date], "manual_date_version"

    by_date = index.by_date.get(requested_date)
    if by_date and allow_date_fallback:
        date_value = _single_counter_value(by_date)
        if date_value is not None:
            return date_value, "learned_requested_date"
        return None, "ambiguous_requested_date"

    return None, "unresolved"


class HistoricalClient:
    def __init__(self, *, session_id: str, cipher_key: bytes, host: str, timeout: float):
        self.cipher_key = cipher_key
        self.host = host
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Encoding": "identity",
                "Content-Type": "application/octet-stream",
                "Cookie": f'$Version="0"; SessionId="{session_id}"; State="1"',
            }
        )

    def fetch_tile(self, *, x: int, y: int, z: int, date_version: DateVersion) -> bytes:
        qk = quadkey(x, y, z)
        url = f"https://{self.host}/flatfile?db=tm&f1-{qk}-{date_version.suffix}"
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code == 200:
                    wire = response.content
                    if len(wire) > len(self.cipher_key):
                        raise RuntimeError(f"tile {len(wire)}B exceeds cipher key {len(self.cipher_key)}B")
                    data = bytes(a ^ b for a, b in zip(wire, self.cipher_key))
                    if not data.startswith(JPEG_SOI):
                        raise RuntimeError(f"decrypted tile is not JPEG: {data[:8].hex()}")
                    return data
                if response.status_code in (401, 403):
                    raise RuntimeError(f"auth failed HTTP {response.status_code}; SessionId may be expired")
                if response.status_code == 404:
                    raise FileNotFoundError(url)
                raise RuntimeError(f"HTTP {response.status_code}: {response.content[:120]!r}")
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_exc = exc
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"network error after retries: {last_exc}")


def build_jobs(
    tasks: list[Mapping[str, str]],
    date_overrides: Mapping[str, DateVersion],
    version_index: VersionIndex,
    *,
    min_bytes: int,
    write_datever_files: bool,
    allow_date_fallback: bool,
) -> tuple[list[TileJob], list[dict[str, object]]]:
    jobs: list[TileJob] = []
    manifests: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []
    for task_idx, task in enumerate(tasks, 1):
        task_name = str(task["task_name"])
        requested_date = str(task.get("date") or task.get("requested_date") or "")[:10]
        coords = list(iter_tile_coords(task))
        task_root = resolve_save_root(task["save_to"]) / task_name
        datever_path = datever_path_for_task(task)
        existing_entries = read_datever_entries(datever_path, len(coords)) or [None] * len(coords)

        task_jobs = 0
        skipped = 0
        version_sources: Counter[str] = Counter()
        resolved_versions: list[DateVersion] = []
        can_write_datever = True
        for (x, y, z), existing_version in zip(coords, existing_entries):
            out_path = task_root / str(z) / str(x) / f"gesh_{x}_{y}_{z}.jpg"
            if existing_version is not None:
                date_version = existing_version
                version_source = "existing_datever"
            else:
                date_version, version_source = resolve_learned_version(
                    requested_date=requested_date,
                    x=x,
                    y=y,
                    z=z,
                    index=version_index,
                    date_overrides=date_overrides,
                    allow_date_fallback=allow_date_fallback,
                )
            if date_version is None:
                can_write_datever = False
                if out_path.exists() and out_path.stat().st_size >= min_bytes:
                    skipped += 1
                    version_sources["not_needed_existing_jpg"] += 1
                    continue
                unresolved.append(
                    {
                        "task_name": task_name,
                        "requested_date": requested_date,
                        "x": x,
                        "y": y,
                        "z": z,
                        "reason": version_source,
                    }
                )
                version_sources[version_source] += 1
                continue
            resolved_versions.append(date_version)
            version_sources[version_source] += 1
            if out_path.exists() and out_path.stat().st_size >= min_bytes:
                skipped += 1
                continue
            jobs.append(
                TileJob(
                    task_idx=task_idx,
                    task_name=task_name,
                    requested_date=requested_date,
                    x=x,
                    y=y,
                    z=z,
                    out_path=out_path,
                    date_version=date_version,
                )
            )
            task_jobs += 1
        if write_datever_files and can_write_datever and len(resolved_versions) == len(coords):
            write_datever(datever_path, resolved_versions)
        manifests.append(
            {
                "idx": task_idx,
                "task_name": task_name,
                "requested_date": requested_date,
                "tile_count": len(coords),
                "skipped_existing": skipped,
                "queued": task_jobs,
                "version_sources": dict(version_sources),
                "datever_path": str(datever_path),
            }
        )
    if unresolved:
        examples = "; ".join(
            f"{u['task_name']} {u['requested_date']} ({u['x']},{u['y']},{u['z']}) {u['reason']}"
            for u in unresolved[:8]
        )
        raise ValueError(
            f"{len(unresolved)} tiles have no historical date-version. "
            "Run a tiny GEID CLI seed for the missing date/area, pass that seed CSV with "
            f"--learn-from-tasks-csv, or pass --date-version manually. Examples: {examples}"
        )
    return jobs, manifests


def run_jobs(client: HistoricalClient, jobs: list[TileJob], *, workers: int, log_file: Path | None) -> tuple[int, int]:
    ok = fail = 0
    log_fh = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_fh = log_file.open("a", encoding="utf-8")

    def work(job: TileJob) -> dict[str, object]:
        started = time.monotonic()
        try:
            data = client.fetch_tile(x=job.x, y=job.y, z=job.z, date_version=job.date_version)
            job.out_path.parent.mkdir(parents=True, exist_ok=True)
            job.out_path.write_bytes(data)
            return {
                "status": "ok",
                "task_idx": job.task_idx,
                "task_name": job.task_name,
                "requested_date": job.requested_date,
                "x": job.x,
                "y": job.y,
                "z": job.z,
                "path": str(job.out_path),
                "bytes": len(data),
                "capture_date": extract_capture_date(data),
                "elapsed_s": round(time.monotonic() - started, 3),
            }
        except Exception as exc:
            return {
                "status": "fail",
                "task_idx": job.task_idx,
                "task_name": job.task_name,
                "requested_date": job.requested_date,
                "x": job.x,
                "y": job.y,
                "z": job.z,
                "path": str(job.out_path),
                "error": str(exc),
                "elapsed_s": round(time.monotonic() - started, 3),
            }

    last_print = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, job) for job in jobs]
            for fut in as_completed(futures):
                rec = fut.result()
                if rec["status"] == "ok":
                    ok += 1
                else:
                    fail += 1
                if log_fh is not None:
                    log_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    log_fh.flush()
                now = time.monotonic()
                if now - last_print >= 2.0:
                    done = ok + fail
                    print(f"direct historical: {done}/{len(jobs)} ok={ok} fail={fail}", file=sys.stderr, flush=True)
                    last_print = now
    finally:
        if log_fh is not None:
            log_fh.close()
    return ok, fail


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks-csv", type=Path, required=True)
    ap.add_argument(
        "--learn-from-tasks-csv",
        type=Path,
        action="append",
        default=[],
        help="Additional seed/partial task CSVs whose existing *_datever.dat files should be learned.",
    )
    ap.add_argument("--date-version", action="append", default=[], type=parse_date_version, help="DATE=DATEVER:IMGVER, repeatable.")
    ap.add_argument("--session-id")
    ap.add_argument("--session-id-file", type=Path)
    ap.add_argument("--cipher-key", type=Path, default=DEFAULT_CIPHER_KEY)
    ap.add_argument("--host", default="khmdb.google.com")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--min-bytes", type=int, default=1024)
    ap.add_argument("--log-file", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--write-datever", action="store_true", help="Write fallback datever.dat files when they are absent or invalid.")
    ap.add_argument(
        "--no-date-fallback",
        action="store_true",
        help="Do not reuse a unique learned date-version across all tiles for the same requested date.",
    )
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    date_overrides = dict(args.date_version)
    tasks = read_tasks_csv(args.tasks_csv)
    learning_tasks = list(tasks)
    for seed_csv in args.learn_from_tasks_csv:
        learning_tasks.extend(read_tasks_csv(seed_csv))
    version_index = learn_versions_from_tasks(learning_tasks)
    jobs, manifests = build_jobs(
        tasks,
        date_overrides,
        version_index,
        min_bytes=args.min_bytes,
        write_datever_files=args.write_datever and not args.dry_run,
        allow_date_fallback=not args.no_date_fallback,
    )

    learned_date_versions = {
        date: {"date_ver": value.date_ver, "date_ver_hex": f"{value.date_ver:x}", "img_ver": value.img_ver, "count": count}
        for date, counter in version_index.by_date.items()
        for value, count in counter.items()
    }

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tasks_csv": str(args.tasks_csv),
        "learn_from_tasks_csv": [str(p) for p in args.learn_from_tasks_csv],
        "tasks": len(tasks),
        "queued_tiles": len(jobs),
        "manual_date_versions": {k: {"date_ver": v.date_ver, "date_ver_hex": f"{v.date_ver:x}", "img_ver": v.img_ver} for k, v in date_overrides.items()},
        "learned_date_versions": learned_date_versions,
        "learned_exact_tiles": len(version_index.exact),
        "learned_parent_quadkeys": len(version_index.parent),
        "tasks_detail": manifests,
    }
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({k: v for k, v in summary.items() if k != "tasks_detail"}, indent=2), file=sys.stderr)
    if args.dry_run or not jobs:
        return 0

    if args.session_id:
        session_id = args.session_id.strip()
    elif args.session_id_file:
        session_id = args.session_id_file.read_text(encoding="utf-8").strip().splitlines()[0]
    else:
        raise SystemExit("--session-id or --session-id-file is required unless --dry-run")
    if not args.cipher_key.exists():
        raise SystemExit(f"cipher key not found: {args.cipher_key}")

    client = HistoricalClient(session_id=session_id, cipher_key=args.cipher_key.read_bytes(), host=args.host, timeout=args.timeout)
    ok, fail = run_jobs(client, jobs, workers=args.workers, log_file=args.log_file)
    print(f"summary: queued={len(jobs)} ok={ok} fail={fail}", file=sys.stderr)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
