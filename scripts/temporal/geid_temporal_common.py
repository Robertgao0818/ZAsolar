# DEPRECATED 2026-05-05: Migrated to solar_backdating subrepo (V1.4 sub-line pivot).
# Authoritative copy: /home/gaosh/projects/solar_backdating/scripts/temporal/geid_temporal_common.py
# This file is frozen; scheduled for removal after 2026-05-31. Bug fixes go to subrepo first.
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Mapping, Sequence

TRUE_VALUES = {"1", "true", "t", "yes", "y", "pv", "present"}
FALSE_VALUES = {"0", "false", "f", "no", "n", "non_pv", "non-pv", "absent"}
DATE_COLUMNS = ("capture_date", "actual_capture_date", "dominant_actual_date", "requested_date", "date")


@dataclass(frozen=True)
class PresenceObservation:
    anchor_id: str
    capture_date: date
    pv_present: bool | None
    pv_score: float | None = None
    requested_date: date | None = None
    quality_flag: str = "ok"
    source_row: int | None = None


@dataclass(frozen=True)
class InstallInterval:
    anchor_id: str
    status: str
    latest_absent_date: date | None
    earliest_present_date: date | None
    install_interval_start: date | None
    install_interval_end: date | None
    n_observations: int
    n_absent: int
    n_present: int
    confidence: str
    notes: str = ""

    def as_row(self) -> dict[str, object]:
        return {
            "anchor_id": self.anchor_id,
            "status": self.status,
            "latest_absent_date": format_date(self.latest_absent_date),
            "earliest_present_date": format_date(self.earliest_present_date),
            "install_interval_start": format_date(self.install_interval_start),
            "install_interval_end": format_date(self.install_interval_end),
            "n_observations": self.n_observations,
            "n_absent": self.n_absent,
            "n_present": self.n_present,
            "confidence": self.confidence,
            "notes": self.notes,
        }


def parse_iso_date(value: object) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    # Accept YYYY-MM-DD and longer ISO timestamps.
    text = text[:10]
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def format_date(value: date | None) -> str:
    return value.isoformat() if value else ""


def years_to_dates(start_year: int, end_year: int, month_day: str = "06-15") -> list[str]:
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")
    if not re.fullmatch(r"\d{2}-\d{2}", month_day):
        raise ValueError("month_day must use MM-DD format")
    # Validate by parsing every year; this catches invalid Feb-29 requests.
    out = []
    for year in range(start_year, end_year + 1):
        d = parse_iso_date(f"{year}-{month_day}")
        if d is None:
            raise ValueError(f"invalid date: {year}-{month_day}")
        out.append(d.isoformat())
    return out


def parse_boolish(value: object) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null", ""}:
        return None
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    return None


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def presence_from_row(row: Mapping[str, object], *, threshold: float = 0.5) -> tuple[bool | None, float | None]:
    """Extract binary presence from a CSV row.

    Priority:
    1. Explicit ``pv_present`` column if parseable.
    2. Score columns (``pv_score``, ``presence_score``, ``score``, ``prob``)
       thresholded at ``threshold``.
    3. Unknown/None.
    """
    explicit = parse_boolish(row.get("pv_present"))
    score = None
    for col in ("pv_score", "presence_score", "score", "prob", "pv_prob"):
        if col in row:
            score = parse_float(row.get(col))
            if score is not None:
                break
    if explicit is not None:
        return explicit, score
    if score is not None:
        return score >= threshold, score
    return None, None


def observation_from_row(row: Mapping[str, object], *, row_idx: int | None = None, threshold: float = 0.5) -> PresenceObservation | None:
    anchor_id = str(row.get("anchor_id", "")).strip()
    if not anchor_id:
        return None
    capture = None
    for col in DATE_COLUMNS:
        capture = parse_iso_date(row.get(col))
        if capture is not None:
            break
    if capture is None:
        return None
    requested = parse_iso_date(row.get("requested_date"))
    present, score = presence_from_row(row, threshold=threshold)
    return PresenceObservation(
        anchor_id=anchor_id,
        capture_date=capture,
        requested_date=requested,
        pv_present=present,
        pv_score=score,
        quality_flag=str(row.get("quality_flag", "ok") or "ok"),
        source_row=row_idx,
    )


def _first_stable_present_index(obs: Sequence[PresenceObservation], min_consecutive_present: int) -> int | None:
    if min_consecutive_present <= 1:
        for i, item in enumerate(obs):
            if item.pv_present is True:
                return i
        return None
    for i in range(0, len(obs) - min_consecutive_present + 1):
        window = obs[i : i + min_consecutive_present]
        if all(item.pv_present is True for item in window):
            return i
    return None


def infer_install_interval(
    anchor_id: str,
    observations: Sequence[PresenceObservation],
    *,
    min_consecutive_present: int = 1,
) -> InstallInterval:
    """Infer the install-date interval from binary historical presence.

    The expected physical process is monotonic: absent before installation,
    present after installation. Later absences after a present observation are
    treated as ambiguity rather than proof of panel removal, because they are
    often caused by imagery date drift, clouds/shadows, georegistration error,
    or classifier failure.
    """
    valid = sorted(
        [o for o in observations if o.anchor_id == anchor_id and o.pv_present is not None],
        key=lambda o: (o.capture_date, o.source_row or 0),
    )
    n_obs = len(valid)
    n_present = sum(1 for o in valid if o.pv_present is True)
    n_absent = sum(1 for o in valid if o.pv_present is False)

    if n_obs == 0:
        return InstallInterval(anchor_id, "no_valid_observations", None, None, None, None, 0, 0, 0, "low")
    if n_present == 0:
        return InstallInterval(
            anchor_id,
            "not_seen",
            valid[-1].capture_date,
            None,
            valid[-1].capture_date,
            None,
            n_obs,
            n_absent,
            n_present,
            "medium",
            "All valid observations are absent; installation is after the last observation or not visible in GEID.",
        )

    first_present_idx = _first_stable_present_index(valid, min_consecutive_present)
    if first_present_idx is None:
        return InstallInterval(
            anchor_id,
            "ambiguous_sporadic_positive",
            max((o.capture_date for o in valid if o.pv_present is False), default=None),
            min((o.capture_date for o in valid if o.pv_present is True), default=None),
            max((o.capture_date for o in valid if o.pv_present is False), default=None),
            min((o.capture_date for o in valid if o.pv_present is True), default=None),
            n_obs,
            n_absent,
            n_present,
            "low",
            f"No run of {min_consecutive_present} consecutive present observations.",
        )

    earliest_present = valid[first_present_idx].capture_date
    earlier_absences = [o.capture_date for o in valid[:first_present_idx] if o.pv_present is False]
    latest_absent = max(earlier_absences) if earlier_absences else None
    later_absences = [o.capture_date for o in valid[first_present_idx + 1 :] if o.pv_present is False]

    if latest_absent is None:
        status = "already_present"
        confidence = "medium" if later_absences else "high"
        notes = "First valid observation is already present; installation predates the first capture."
        if later_absences:
            status = "ambiguous_nonmonotonic"
            confidence = "low"
            notes += " Later absent observations violate monotonicity."
        return InstallInterval(
            anchor_id,
            status,
            None,
            earliest_present,
            None,
            earliest_present,
            n_obs,
            n_absent,
            n_present,
            confidence,
            notes,
        )

    status = "appears"
    confidence = "high"
    notes = "First stable present observation follows at least one absent observation."
    if later_absences:
        status = "ambiguous_nonmonotonic"
        confidence = "low"
        notes += " Later absent observations violate monotonicity."

    return InstallInterval(
        anchor_id,
        status,
        latest_absent,
        earliest_present,
        latest_absent,
        earliest_present,
        n_obs,
        n_absent,
        n_present,
        confidence,
        notes,
    )


def safe_task_token(value: object, *, max_len: int = 120) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    token = token.strip("._-") or "task"
    return token[:max_len]


def join_task_root(root: str, *parts: object) -> str:
    """Join GEID task output path parts while preserving root path style.

    Canonical local storage is WSL/POSIX under ``~/zasolar_data``.  Windows
    drive and UNC roots are still supported for explicit downloader staging.
    """
    root_text = str(root).strip()
    use_backslash = bool(re.match(r"^[A-Za-z]:[\\/]", root_text)) or root_text.startswith("\\\\")
    sep = "\\" if use_backslash else "/"
    out = root_text.rstrip("\\/")
    for part in parts:
        token = str(part).strip().strip("\\/")
        if token:
            out += sep + token
    return out


def win_join(root: str, *parts: object) -> str:
    """Backward-compatible wrapper for older callers/tests."""
    return join_task_root(root, *parts)


def build_geid_task_rows(
    anchors: Sequence[Mapping[str, object]],
    dates: Sequence[str],
    *,
    save_root_win: str,
    zoom_from: int,
    zoom_to: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for anchor in anchors:
        anchor_id = str(anchor.get("anchor_id", "")).strip()
        if not anchor_id:
            raise ValueError("anchor row missing anchor_id")
        region = safe_task_token(anchor.get("region_key", "unknown_region"))
        grid_id = safe_task_token(anchor.get("grid_id", "unknown_grid"))
        for required in ("chip_lon_min", "chip_lon_max", "chip_lat_min", "chip_lat_max"):
            if required not in anchor or str(anchor[required]).strip() == "":
                raise ValueError(f"anchor {anchor_id} missing {required}")
        lon_min = float(anchor["chip_lon_min"])
        lon_max = float(anchor["chip_lon_max"])
        lat_min = float(anchor["chip_lat_min"])
        lat_max = float(anchor["chip_lat_max"])
        for d_str in dates:
            d = parse_iso_date(d_str)
            if d is None:
                raise ValueError(f"invalid date: {d_str}")
            date_token = d.strftime("%Y%m%d")
            task_name = safe_task_token(f"{anchor_id}_{date_token}")
            rows.append(
                {
                    "grid_id": anchor_id,
                    "task_name": task_name,
                    "save_to": join_task_root(save_root_win, region, grid_id, safe_task_token(anchor_id), str(d.year)),
                    "map_type": "",
                    "date": d.isoformat(),
                    "zoom_from": zoom_from,
                    "zoom_to": zoom_to,
                    "left_longitude": f"{min(lon_min, lon_max):.10f}",
                    "right_longitude": f"{max(lon_min, lon_max):.10f}",
                    "top_latitude": f"{max(lat_min, lat_max):.10f}",
                    "bottom_latitude": f"{min(lat_min, lat_max):.10f}",
                }
            )
    return rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
