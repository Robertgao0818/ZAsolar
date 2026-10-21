import csv, json, subprocess, sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_coverage  # noqa: E402


def w(p, header, rows):
    with open(p, "w", newline="") as f:
        c = csv.writer(f); c.writerow(header); c.writerows(rows)
    return str(p)


def test_pass_with_scored_and_excluded(tmp_path):
    aoi = w(tmp_path / "a.csv", ["tile"], [["a"], ["b"], ["c"]])
    sc = w(tmp_path / "s.csv", ["tile", "max_score"], [["a", 0.1], ["b", 0.9]])
    ex = w(tmp_path / "e.csv", ["tile", "reason"], [["c", "no_image_data"]])
    assert check_coverage.main(["--city", "t", "--aoi", aoi, "--scored", sc, "--score-col", "max_score",
                                "--exclusions", ex, "--out-dir", str(tmp_path / "o")]) == 0
    s = json.load(open(tmp_path / "o/coverage_summary.json"))
    assert (s["scored"], s["excluded"], s["unscored"]) == (2, 1, 0)


def test_fail_on_unscored_nan_score_and_empty_reason(tmp_path):
    aoi = w(tmp_path / "a.csv", ["tile"], [["a"], ["b"], ["c"]])
    sc = w(tmp_path / "s.csv", ["tile", "max_score"], [["a", 0.1], ["b", ""]])
    ex = w(tmp_path / "e.csv", ["tile", "reason"], [["c", ""]])
    rc = check_coverage.main(["--city", "t", "--aoi", aoi, "--scored", sc, "--score-col", "max_score",
                              "--exclusions", ex, "--out-dir", str(tmp_path / "o")])
    assert rc == 2
    s = json.load(open(tmp_path / "o/coverage_summary.json"))
    assert s["unscored"] == 2 and s["exclusions_without_reason"] == 1 and s["result"] == "FAIL"


def test_reference_on_unscored(tmp_path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box
    aoi = w(tmp_path / "a.csv", ["tile", "minx", "miny", "maxx", "maxy"], [["a", 0, 0, 100, 100], ["b", 100, 0, 200, 100]])
    sc = w(tmp_path / "s.csv", ["tile"], [["a"]])
    ref = tmp_path / "r.gpkg"
    gpd.GeoDataFrame(geometry=[box(10, 10, 20, 20), box(150, 10, 160, 20), box(500, 500, 501, 501)], crs="EPSG:32734").to_file(ref)
    rc = check_coverage.main(["--city", "t", "--aoi", aoi, "--scored", sc, "--reference", str(ref),
                              "--crs", "EPSG:32734", "--out-dir", str(tmp_path / "o")])
    s = json.load(open(tmp_path / "o/coverage_summary.json"))
    assert rc == 2 and s["reference_by_status"]["unscored"]["n"] == 1 and s["reference_by_status"]["outside_aoi"]["n"] == 1


def test_ct_census_run_gate_requires_complete_download(tmp_path):
    """Shell wiring: grid with infer .ok but dl .fail (partial download) must fail the gate."""
    import re
    sh = (ROOT / "scripts/ct_census_run.sh").read_text()
    block = sh[sh.index("# ── Coverage gate"):sh.index("# ── Phase C")]
    st, logs = tmp_path / "state", tmp_path / "logs"; st.mkdir(); logs.mkdir()
    (st / "glist.txt").write_text("CPT1\nCPT2\n")
    import os, time
    for f in ["dl_CPT1.ok", "infer_CPT1.ok", "infer_CPT2.ok", "dl_CPT2.fail", "dl_CPT3.ok", "infer_CPT3.ok"]: (st / f).touch()
    (st / "glist.txt").write_text("CPT1\nCPT2\nCPT3\n")
    t = time.time(); os.utime(st / "infer_CPT3.ok", (t - 100, t - 100))  # CPT3: download completed AFTER inference
    script = (f'set -uo pipefail\nSTATE={st}; LOGS={logs}; GLIST={st}/glist.txt; ZAS={ROOT}\n'
              'log(){ echo "$*"; }; write_status(){ :; }\n' + block + "\necho GATE_PASSED\n")
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={"PATH": __import__("os").environ["PATH"]})
    assert r.returncode == 2 and "GATE_PASSED" not in r.stdout
    assert (st / "coverage_gate/unscored_units.csv").read_text().split() == ["grid", "CPT2", "CPT3"]


def _infer_one_harness(tmp_path):
    import re
    sh = (ROOT / "scripts/ct_census_run.sh").read_text()
    body = re.search(r"\ninfer_one\(\)\{\n(.*?)\n  local OUT=", sh, re.S).group(1)
    st, logs = tmp_path / "state", tmp_path / "logs"; st.mkdir(); logs.mkdir()
    zas = tmp_path / "zas"; (zas / "scripts/lib").mkdir(parents=True); (zas / "scripts/lib/census_caliber.sh").write_text("")
    fn = "infer_one(){\n" + body + "\n  echo RAN_INFER\n}\n"
    return st, (f"STATE={st}; LOGS={logs}; ZAS={zas}\n" + fn)


def test_infer_reruns_after_download_completes_and_skips_incomplete(tmp_path):
    import os, time
    st, pre = _infer_one_harness(tmp_path)
    run = lambda g: subprocess.run(["bash", "-c", pre + f"infer_one {g}"], capture_output=True, text=True).stdout
    (st / "infer_A.ok").touch(); t = time.time(); os.utime(st / "infer_A.ok", (t - 100, t - 100)); (st / "dl_A.ok").touch()
    assert "RAN_INFER" in run("A") and not (st / "infer_A.ok").exists()      # stale marker dropped, re-run
    (st / "dl_B.ok").touch(); (st / "infer_B.ok").touch()
    assert "RAN_INFER" not in run("B")                                        # fresh marker kept
    assert "RAN_INFER" not in run("C")                                        # no dl_C.ok -> not inferred


def test_stale_dl_fail_cleared_before_downloader():
    sh = (ROOT / "scripts/ct_census_run.sh").read_text()
    assert sh.index('rm -f "$STATE"/dl_*.fail') < sh.index("xargs -P \"$DL_WORKERS\"")
