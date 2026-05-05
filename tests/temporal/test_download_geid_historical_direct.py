# DEPRECATED 2026-05-05: Migrated to solar_backdating subrepo (V1.4 sub-line pivot).
# Authoritative copy: /home/gaosh/projects/solar_backdating/tests/temporal/test_download_geid_historical_direct.py
# This file is frozen; scheduled for removal after 2026-05-31. Bug fixes go to subrepo first.
import struct
import tempfile
import unittest
from pathlib import Path

from scripts.temporal.download_geid_historical_direct import (
    DateVersion,
    build_jobs,
    learn_versions_from_tasks,
    read_datever_entries,
    write_datever,
)


def _task(
    root: Path,
    task_name: str,
    date: str,
    *,
    left: str = "28.0490913",
    right: str = "28.0495033",
) -> dict[str, str]:
    return {
        "task_name": task_name,
        "save_to": str(root),
        "date": date,
        "zoom_from": "21",
        "zoom_to": "21",
        "left_longitude": left,
        "right_longitude": right,
        "top_latitude": "-26.2174644",
        "bottom_latitude": "-26.2178764",
    }


class DownloadGeidHistoricalDirectTests(unittest.TestCase):
    def test_read_datever_entries_marks_invalid_entries_as_none(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sample_datever.dat"
            p.write_bytes(struct.pack("<II", 1033630, 296) + struct.pack("<II", 0xFFFFFFFF, 0xFFFFFFFF))

            entries = read_datever_entries(p, 2)

            self.assertEqual(entries, [DateVersion(1033630, 296), None])

    def test_build_jobs_learns_exact_tile_date_version_from_seed_task(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            seed = _task(root, "seed_20190615", "2019-06-15")
            full = _task(root, "full_20190615", "2019-06-15")
            coords_count = 9
            write_datever(root / "seed_20190615_datever.dat", [DateVersion(1033630, 296)] * coords_count)

            index = learn_versions_from_tasks([seed])
            jobs, manifest = build_jobs(
                [full],
                {},
                index,
                min_bytes=1024,
                write_datever_files=False,
                allow_date_fallback=True,
            )

            self.assertEqual(len(jobs), coords_count)
            self.assertEqual({job.date_version for job in jobs}, {DateVersion(1033630, 296)})
            self.assertEqual(manifest[0]["version_sources"], {"learned_exact_tile": coords_count})

    def test_build_jobs_can_fallback_to_unique_requested_date_version(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            seed = _task(root, "seed_20190615", "2019-06-15")
            full = _task(root, "full_20190615", "2019-06-15", left="28.0800000", right="28.0805033")
            write_datever(root / "seed_20190615_datever.dat", [DateVersion(1033630, 296)] * 9)

            index = learn_versions_from_tasks([seed])
            jobs, manifest = build_jobs(
                [full],
                {},
                index,
                min_bytes=1024,
                write_datever_files=False,
                allow_date_fallback=True,
            )

            self.assertEqual({job.date_version for job in jobs}, {DateVersion(1033630, 296)})
            self.assertEqual(manifest[0]["version_sources"], {"learned_requested_date": len(jobs)})

    def test_build_jobs_rejects_ambiguous_learned_date_versions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            seed1 = _task(root, "seed1_20190615", "2019-06-15")
            seed2 = _task(root, "seed2_20190615", "2019-06-15", left="28.0600000", right="28.0605033")
            full = _task(root, "full_20190615", "2019-06-15", left="28.0800000", right="28.0805033")
            write_datever(root / "seed1_20190615_datever.dat", [DateVersion(1033630, 296)] * 9)
            write_datever(root / "seed2_20190615_datever.dat", [DateVersion(1034000, 296)] * 9)
            index = learn_versions_from_tasks([seed1, seed2])

            with self.assertRaises(ValueError) as ctx:
                build_jobs(
                    [full],
                    {},
                    index,
                    min_bytes=1024,
                    write_datever_files=False,
                    allow_date_fallback=True,
                )

            self.assertIn("no historical date-version", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
