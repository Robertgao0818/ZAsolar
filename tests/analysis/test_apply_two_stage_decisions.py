import json
import tempfile
import unittest
from pathlib import Path

from scripts.analysis.apply_two_stage_decisions import grid_for_path, load_decisions


def _write_jsonl(rows):
    fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for r in rows:
        fh.write(json.dumps(r) + "\n")
    fh.close()
    return Path(fh.name)


def _row(grid, pid, *, auto_drop, action, path="/p/Gx/predictions_metric.gpkg", review=False):
    return {
        "candidate_id": f"{grid}_pred{pid:06d}",
        "grid_id": grid,
        "pred_id": pid,
        "predictions_path": path,
        "auto_drop": auto_drop,
        "production_action": action,
        "requires_human_review": review,
    }


class ApplyTwoStageDecisionsTests(unittest.TestCase):
    def test_drop_and_keep_collected(self):
        p = _write_jsonl([
            _row("G1", 0, auto_drop=True, action="drop"),
            _row("G1", 1, auto_drop=False, action="keep"),
        ])
        decisions, conflicts, violations = load_decisions([p])
        self.assertEqual(len(decisions), 2)
        self.assertEqual(conflicts, [])
        self.assertEqual(violations, [])

    def test_cross_file_conflict_resolves_to_keep(self):
        # same prediction, one file says drop, the other keep -> conservative keep wins
        path = "/p/G1/predictions_metric.gpkg"
        a = _write_jsonl([_row("G1", 5, auto_drop=True, action="drop", path=path)])
        b = _write_jsonl([_row("G1", 5, auto_drop=False, action="keep", path=path)])
        decisions, conflicts, violations = load_decisions([a, b])
        key = (str(Path(path).resolve()), 5)
        self.assertIn(key, decisions)
        self.assertFalse(decisions[key]["auto_drop"])  # non-drop wins
        self.assertEqual(len(conflicts), 1)

    def test_integrity_violation_flagged(self):
        # auto_drop=true but action != drop is a fail-closed integrity violation
        p = _write_jsonl([_row("G1", 0, auto_drop=True, action="review")])
        _decisions, _conflicts, violations = load_decisions([p])
        self.assertEqual(len(violations), 1)
        self.assertIn("auto_drop=true", violations[0])

    def test_grid_for_path_prefers_grid_id_field_over_path(self):
        # reviewed-gpkg layout: parent.name is 'review', must use grid_id field
        dmap = {0: {"grid_id": "G0890", "pred_id": 0}}
        self.assertEqual(
            grid_for_path(dmap, "/r/G0890/review/G0890_reviewed.gpkg"), "G0890"
        )
        # fallback to path parsing when grid_id absent
        self.assertEqual(grid_for_path({0: {}}, "/r/run/G0816/predictions_metric.gpkg"), "G0816")


if __name__ == "__main__":
    unittest.main()
