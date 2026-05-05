# DEPRECATED 2026-05-05: Migrated to solar_backdating subrepo (V1.4 sub-line pivot).
# Authoritative copy: /home/gaosh/projects/solar_backdating/tests/temporal/test_score_anchor_presence.py
# This file is frozen; scheduled for removal after 2026-05-31. Bug fixes go to subrepo first.
import tempfile
import unittest
from pathlib import Path

from scripts.temporal.score_anchor_presence import (
    build_manual_decision_index,
    build_presence_rows,
    extract_geid_capture_date,
    resolve_task_dir,
    write_qa_html,
)


class ScoreAnchorPresenceTests(unittest.TestCase):
    def test_resolve_task_dir_handles_windows_and_posix_save_roots(self):
        self.assertEqual(
            resolve_task_dir(r"D:\ZAsolar\geid_raw\temporal_pv", "a1_20190615"),
            Path("/mnt/d/ZAsolar/geid_raw/temporal_pv/a1_20190615"),
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.assertEqual(resolve_task_dir(str(root), "a1_20190615"), root / "a1_20190615")

    def test_extract_geid_capture_date_from_jpeg_comment(self):
        with tempfile.TemporaryDirectory() as td:
            jpg = Path(td) / "gesh_1_2_21.jpg"
            jpg.write_bytes(b"\xff\xd8\xff\xe0xxxx*AD*2018:12:04*yyyy")
            self.assertEqual(extract_geid_capture_date(jpg), "2018-12-04")

    def test_build_presence_rows_creates_manual_template_from_downloaded_chip(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_dir = root / "a1_20190615"
            task_dir.mkdir()
            (task_dir / "gesh_1_2_21.jpg").write_bytes(b"\xff\xd8\xff\xe0*AD*2018:12:04*")
            anchors = [
                {
                    "anchor_id": "a1",
                    "region_key": "johannesburg",
                    "grid_id": "G0922",
                }
            ]
            tasks = [
                {
                    "grid_id": "a1",
                    "task_name": "a1_20190615",
                    "save_to": str(root),
                    "date": "2019-06-15",
                }
            ]

            rows = build_presence_rows(anchors, tasks, manual_decisions={})

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["anchor_id"], "a1")
            self.assertEqual(row["region_key"], "johannesburg")
            self.assertEqual(row["grid_id"], "G0922")
            self.assertEqual(row["requested_date"], "2019-06-15")
            self.assertEqual(row["capture_date"], "2018-12-04")
            self.assertEqual(row["actual_capture_dates"], "2018-12-04")
            self.assertEqual(row["decision_source"], "manual_template")
            self.assertEqual(row["quality_flag"], "needs_review")
            self.assertEqual(row["pv_present"], "")
            self.assertEqual(row["n_jpg"], 1)
            self.assertTrue(row["sample_chip_path"].endswith("gesh_1_2_21.jpg"))

    def test_manual_decisions_override_template_fields(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_dir = root / "a1_20190615"
            task_dir.mkdir()
            (task_dir / "gesh_1_2_21.jpg").write_bytes(b"\xff\xd8\xff\xe0*AD*2018:12:04*")
            anchors = [{"anchor_id": "a1", "region_key": "johannesburg", "grid_id": "G0922"}]
            tasks = [{"grid_id": "a1", "task_name": "a1_20190615", "save_to": str(root), "date": "2019-06-15"}]
            manual = build_manual_decision_index(
                [
                    {
                        "anchor_id": "a1",
                        "requested_date": "2019-06-15",
                        "pv_present": "1",
                        "pv_score": "0.91",
                        "quality_flag": "ok",
                        "notes": "visible on roof",
                    }
                ]
            )

            row = build_presence_rows(anchors, tasks, manual_decisions=manual)[0]

            self.assertEqual(row["decision_source"], "manual")
            self.assertEqual(row["quality_flag"], "ok")
            self.assertEqual(row["pv_present"], "1")
            self.assertEqual(row["pv_score"], "0.91")
            self.assertEqual(row["notes"], "visible on roof")

    def test_manual_unusable_decision_leaves_presence_blank(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_dir = root / "a1_20190615"
            task_dir.mkdir()
            (task_dir / "gesh_1_2_21.jpg").write_bytes(b"\xff\xd8\xff\xe0*AD*2018:12:04*")
            anchors = [{"anchor_id": "a1", "region_key": "johannesburg", "grid_id": "G0922"}]
            tasks = [{"grid_id": "a1", "task_name": "a1_20190615", "save_to": str(root), "date": "2019-06-15"}]
            manual = build_manual_decision_index(
                [
                    {
                        "anchor_id": "a1",
                        "requested_date": "2019-06-15",
                        "pv_present": "",
                        "pv_score": "",
                        "quality_flag": "unusable",
                        "notes": "too blurry",
                    }
                ]
            )

            row = build_presence_rows(anchors, tasks, manual_decisions=manual)[0]

            self.assertEqual(row["decision_source"], "manual")
            self.assertEqual(row["quality_flag"], "unusable")
            self.assertEqual(row["pv_present"], "")
            self.assertEqual(row["pv_score"], "")
            self.assertEqual(row["notes"], "too blurry")

    def test_write_qa_html_includes_three_class_controls_and_csv_export(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "qa.html"
            write_qa_html(
                out,
                [
                    {
                        "anchor_id": "a1",
                        "requested_date": "2019-06-15",
                        "capture_date": "2018-12-04",
                        "quality_flag": "needs_review",
                        "n_jpg": 1,
                        "sample_chip_path": "",
                        "chip_dir": str(Path(td) / "chips"),
                    }
                ],
            )

            html = out.read_text(encoding="utf-8")
            self.assertIn("data-label='present'", html)
            self.assertIn("data-label='absent'", html)
            self.assertIn("data-label='unusable'", html)
            self.assertIn("data-label='unsure'", html)
            self.assertIn("Manual decisions CSV", html)
            self.assertIn("quality_flag = 'unusable'", html)

    def test_missing_chip_rows_are_explicit_and_do_not_guess_presence(self):
        with tempfile.TemporaryDirectory() as td:
            anchors = [{"anchor_id": "a1", "region_key": "johannesburg", "grid_id": "G0922"}]
            tasks = [{"grid_id": "a1", "task_name": "a1_20190615", "save_to": td, "date": "2019-06-15"}]

            row = build_presence_rows(anchors, tasks, manual_decisions={})[0]

            self.assertEqual(row["decision_source"], "missing_chip")
            self.assertEqual(row["quality_flag"], "missing_chip")
            self.assertEqual(row["pv_present"], "")
            self.assertEqual(row["pv_score"], "")
            self.assertEqual(row["capture_date"], "")
            self.assertEqual(row["n_jpg"], 0)


if __name__ == "__main__":
    unittest.main()
