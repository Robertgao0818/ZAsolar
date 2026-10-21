"""Regression: tiles already present (ROOT/tiles or --existing) must be written to the manifest."""
import csv, subprocess, sys
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/imagery/dl_native_v2.py"


def fake_tile(p, x, y):
    p.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(p, "w", driver="GTiff", width=2000, height=2000, count=3, dtype="uint8",
                       crs="EPSG:32734", transform=from_origin(x, y + 100, 0.05, 0.05)) as o:
        o.write(np.zeros((3, 2000, 2000), np.uint8))


def test_existing_tiles_go_into_manifest(tmp_path):
    root, ext = tmp_path / "root", tmp_path / "old"
    fake_tile(ext / "tiles/E100_N200.tif", 100, 200)
    fake_tile(root / "tiles/E200_N200.tif", 200, 200)
    args = [sys.executable, str(SCRIPT), "--root", str(root), "--existing", str(ext / "tiles"),
            "--only-tiles", "E100_N200,E200_N200"]
    for _ in range(2):  # second run must not duplicate rows
        subprocess.run(args, check=True, capture_output=True)
    man = [r[0] for r in csv.reader(open(root / "manifest_tiles_v2.csv"))]
    src = {r[0]: r[2] for r in csv.reader(open(root / "manifest_tiles_v2_src.csv"))}
    assert sorted(man) == ["E100_N200.tif", "E200_N200.tif"]
    assert src == {"E100_N200.tif": "existing", "E200_N200.tif": "downloaded"}
    # legacy consumers open ROOT/tiles/<tile>: the existing tile must be reachable there (symlink, not a copy)
    link = root / "tiles/E100_N200.tif"
    assert link.is_symlink() and link.resolve() == (ext / "tiles/E100_N200.tif").resolve()
    assert sum(1 for _ in open(root / "manifest_tiles_v2_src.csv")) == 2


def test_sidecar_updates_when_path_changes(tmp_path):
    root, a_dir, b_dir = tmp_path / "root", tmp_path / "A", tmp_path / "B"
    fake_tile(a_dir / "E100_N200.tif", 100, 200)
    base = [sys.executable, str(SCRIPT), "--root", str(root), "--only-tiles", "E100_N200"]
    subprocess.run(base + ["--existing", str(a_dir)], check=True, capture_output=True)
    (root / "tiles/E100_N200.tif").unlink()          # simulate A going away (link and source)
    (a_dir / "E100_N200.tif").rename(tmp_path / "gone.tif")
    fake_tile(b_dir / "E100_N200.tif", 100, 200)
    subprocess.run(base + ["--existing", str(b_dir)], check=True, capture_output=True)
    rows = list(csv.reader(open(root / "manifest_tiles_v2_src.csv")))
    assert rows[-1][1] == str(root / "tiles/E100_N200.tif") and (root / "tiles/E100_N200.tif").resolve() == (b_dir / "E100_N200.tif").resolve()
    assert [r[0] for r in csv.reader(open(root / "manifest_tiles_v2.csv"))] == ["E100_N200.tif"]


def test_invalid_root_file_not_manifested_and_dangling_link_replaced(tmp_path):
    root, ext = tmp_path / "root", tmp_path / "old"
    fake_tile(ext / "E100_N200.tif", 100, 200); fake_tile(ext / "E200_N200.tif", 200, 200)
    (root / "tiles").mkdir(parents=True)
    (root / "tiles/E100_N200.tif").write_bytes(b"corrupt")                     # invalid real file: keep, do not manifest
    (root / "tiles/E200_N200.tif").symlink_to(tmp_path / "nowhere.tif")        # dangling link: replace
    subprocess.run([sys.executable, str(SCRIPT), "--root", str(root), "--existing", str(ext),
                    "--only-tiles", "E100_N200,E200_N200"], check=True, capture_output=True)
    man = {r[0]: r for r in csv.reader(open(root / "manifest_tiles_v2.csv"))}
    assert list(man) == ["E200_N200.tif"]
    assert (root / "tiles/E100_N200.tif").read_bytes() == b"corrupt"
    assert (root / "tiles/E200_N200.tif").resolve() == (ext / "E200_N200.tif").resolve()
    assert int(man["E200_N200.tif"][1]) == (ext / "E200_N200.tif").stat().st_size


def test_corrupt_after_registration_drops_row_and_resize_keeps_one_row(tmp_path):
    root, ext = tmp_path / "root", tmp_path / "old"
    fake_tile(ext / "E100_N200.tif", 100, 200); fake_tile(root / "tiles/E200_N200.tif", 200, 200)
    args = [sys.executable, str(SCRIPT), "--root", str(root), "--existing", str(ext), "--only-tiles", "E100_N200,E200_N200"]
    subprocess.run(args, check=True, capture_output=True)
    (root / "tiles/E100_N200.tif").unlink(); (root / "tiles/E100_N200.tif").write_bytes(b"corrupt")
    man = root / "manifest_tiles_v2.csv"
    rows = list(csv.reader(open(man))); rows = [r if r[0] != "E200_N200.tif" else [r[0], "1", r[2], r[3]] for r in rows]
    csv.writer(open(man, "w", newline="")).writerows(rows)          # simulate a stale size for E200
    subprocess.run(args, check=True, capture_output=True)
    names = [r[0] for r in csv.reader(open(man))]
    assert names == ["E200_N200.tif"]
    assert int(next(csv.reader(open(man)))[1]) == (root / "tiles/E200_N200.tif").stat().st_size


def test_registered_tile_lost_and_redownload_fails_drops_row(tmp_path):
    import os
    root = tmp_path / "root"; fake_tile(root / "tiles/E100_N200.tif", 100, 200)
    args = [sys.executable, str(SCRIPT), "--root", str(root), "--only-tiles", "E100_N200"]
    subprocess.run(args, check=True, capture_output=True)
    assert [r[0] for r in csv.reader(open(root / "manifest_tiles_v2.csv"))] == ["E100_N200.tif"]
    (root / "tiles/E100_N200.tif").unlink()
    env = dict(os.environ, DL_NATIVE_TRIES="0")   # no network: every fetch fails immediately
    subprocess.run(args, check=True, capture_output=True, env=env)
    assert list(csv.reader(open(root / "manifest_tiles_v2.csv"))) == []
