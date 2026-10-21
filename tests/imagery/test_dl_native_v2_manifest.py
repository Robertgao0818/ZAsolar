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
