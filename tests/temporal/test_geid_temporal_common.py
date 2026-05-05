# DEPRECATED 2026-05-05: Migrated to solar_backdating subrepo (V1.4 sub-line pivot).
# Authoritative copy: /home/gaosh/projects/solar_backdating/tests/temporal/test_geid_temporal_common.py
# This file is frozen; scheduled for removal after 2026-05-31. Bug fixes go to subrepo first.
import csv
import tempfile
import unittest
from datetime import date
from pathlib import Path

from scripts.temporal.geid_temporal_common import (
    PresenceObservation,
    build_geid_task_rows,
    infer_install_interval,
    join_task_root,
    observation_from_row,
    read_csv_rows,
    write_csv_rows,
    years_to_dates,
)


class TemporalCommonTests(unittest.TestCase):
    def test_years_to_dates_inclusive(self):
        self.assertEqual(years_to_dates(2018, 2020, "06-15"), ["2018-06-15", "2019-06-15", "2020-06-15"])

    def test_observation_from_score_row(self):
        obs = observation_from_row(
            {"anchor_id": "a1", "capture_date": "2019-02-03", "pv_score": "0.73"},
            threshold=0.5,
        )
        self.assertIsNotNone(obs)
        assert obs is not None
        self.assertEqual(obs.anchor_id, "a1")
        self.assertEqual(obs.capture_date, date(2019, 2, 3))
        self.assertTrue(obs.pv_present)
        self.assertAlmostEqual(obs.pv_score or 0, 0.73)

    def test_infer_appearance_interval(self):
        obs = [
            PresenceObservation("a1", date(2017, 6, 15), False),
            PresenceObservation("a1", date(2018, 6, 15), False),
            PresenceObservation("a1", date(2019, 6, 15), True),
            PresenceObservation("a1", date(2020, 6, 15), True),
        ]
        interval = infer_install_interval("a1", obs)
        self.assertEqual(interval.status, "appears")
        self.assertEqual(interval.latest_absent_date, date(2018, 6, 15))
        self.assertEqual(interval.earliest_present_date, date(2019, 6, 15))
        self.assertEqual(interval.confidence, "high")

    def test_infer_already_present(self):
        obs = [
            PresenceObservation("a1", date(2018, 6, 15), True),
            PresenceObservation("a1", date(2019, 6, 15), True),
        ]
        interval = infer_install_interval("a1", obs)
        self.assertEqual(interval.status, "already_present")
        self.assertIsNone(interval.latest_absent_date)
        self.assertEqual(interval.earliest_present_date, date(2018, 6, 15))

    def test_infer_nonmonotonic(self):
        obs = [
            PresenceObservation("a1", date(2018, 6, 15), False),
            PresenceObservation("a1", date(2019, 6, 15), True),
            PresenceObservation("a1", date(2020, 6, 15), False),
        ]
        interval = infer_install_interval("a1", obs)
        self.assertEqual(interval.status, "ambiguous_nonmonotonic")
        self.assertEqual(interval.confidence, "low")

    def test_join_task_root_preserves_posix_and_windows_styles(self):
        self.assertEqual(join_task_root("/home/gaosh/zasolar_data/geid_raw", "a", "b"), "/home/gaosh/zasolar_data/geid_raw/a/b")
        self.assertEqual(join_task_root(r"D:\ZAsolar\geid_raw", "a", "b"), r"D:\ZAsolar\geid_raw\a\b")
        self.assertEqual(
            join_task_root(r"\\wsl.localhost\Ubuntu\home\gaosh\zasolar_data", "a", "b"),
            r"\\wsl.localhost\Ubuntu\home\gaosh\zasolar_data\a\b",
        )

    def test_build_geid_task_rows_uses_posix_canonical_root(self):
        anchors = [
            {
                "anchor_id": "johannesburg_G0922_a000001",
                "region_key": "johannesburg",
                "grid_id": "G0922",
                "chip_lon_min": "28.0",
                "chip_lon_max": "28.1",
                "chip_lat_min": "-26.2",
                "chip_lat_max": "-26.1",
            }
        ]
        rows = build_geid_task_rows(
            anchors,
            ["2019-06-15", "2020-06-15"],
            save_root_win="/home/gaosh/zasolar_data/geid_raw/temporal_anchor_presence",
            zoom_from=21,
            zoom_to=21,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["task_name"], "johannesburg_G0922_a000001_20190615")
        self.assertEqual(rows[0]["top_latitude"], "-26.1000000000")
        self.assertEqual(rows[0]["bottom_latitude"], "-26.2000000000")
        self.assertIn("johannesburg/G0922/johannesburg_G0922_a000001/2019", rows[0]["save_to"])

    def test_csv_write_read_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rows.csv"
            write_csv_rows(path, [{"a": 1, "b": "x"}], ["a", "b"])
            rows = read_csv_rows(path)
            self.assertEqual(rows, [{"a": "1", "b": "x"}])


if __name__ == "__main__":
    unittest.main()
