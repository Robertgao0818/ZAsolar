"""Unit tests for C-3(a) Phase 0 primitives (core.training.c3a_phase0).

Covers (per deliverable F):
  - stratified sampling logic (quota allocation + deterministic indices)
  - chip-window enumeration parity with scan_chips_from_tile
  - audit CSV schema validation
  - gate threshold decision (synthetic audit data)

All synthetic; CPU only; no tiles, no model.
"""
from __future__ import annotations

import unittest

from shapely.geometry import box

from core.training.c3a_phase0 import (
    AUDIT_CSV_COLUMNS,
    AUDIT_LABEL_VALUES,
    DEFAULT_GATE_THRESHOLD,
    allocate_stratified_quota,
    compute_gate,
    enumerate_chip_windows,
    make_audit_id,
    make_chip_uid,
    proposal_overlaps_gt,
    sample_indices,
    stratum_key,
    stratum_sub_seed,
    validate_audit_rows,
)


# ──────────────────────────────────────────────────────────────────────────
# Chip-window enumeration
# ──────────────────────────────────────────────────────────────────────────
class ChipWindowTests(unittest.TestCase):
    def test_stride_and_edge_skip_match_scan_chips(self):
        # chip_size=400 overlap=0.25 -> stride=300.  A 700x500 tile.
        wins = list(enumerate_chip_windows(700, 500, 400, 0.25))
        # x offsets: 0, 300, 600 ; y offsets: 0, 300
        # Edge chips with side < chip_size//2 (=200) are skipped.
        # x=600 -> w = 700-600 = 100 < 200 -> skipped for all rows.
        # y=300 -> h = 500-300 = 200 == 200 (not < 200) -> kept.
        xs = sorted({w[0] for w in wins})
        ys = sorted({w[1] for w in wins})
        self.assertEqual(xs, [0, 300])
        self.assertEqual(ys, [0, 300])
        # full chips at (0,0),(300,0): w=400,h=400 ; at (0,300),(300,300): h=200
        by_origin = {(w[0], w[1]): (w[2], w[3]) for w in wins}
        self.assertEqual(by_origin[(0, 0)], (400, 400))
        self.assertEqual(by_origin[(300, 0)], (400, 400))
        self.assertEqual(by_origin[(0, 300)], (400, 200))

    def test_invalid_overlap_raises(self):
        with self.assertRaises(ValueError):
            list(enumerate_chip_windows(400, 400, 400, 1.0))
        with self.assertRaises(ValueError):
            list(enumerate_chip_windows(400, 400, 0, 0.25))


# ──────────────────────────────────────────────────────────────────────────
# Stratified quota
# ──────────────────────────────────────────────────────────────────────────
class StratifiedQuotaTests(unittest.TestCase):
    def test_proportional_split_sums_to_target(self):
        counts = {"ct:aerial_2025": 600, "jhb:vexcel_2024": 400}
        quota = allocate_stratified_quota(counts, 180)
        self.assertEqual(sum(quota.values()), 180)
        # proportional: 600/1000*180=108, 400/1000*180=72
        self.assertEqual(quota["ct:aerial_2025"], 108)
        self.assertEqual(quota["jhb:vexcel_2024"], 72)

    def test_never_exceeds_availability(self):
        counts = {"a": 10, "b": 1000}
        quota = allocate_stratified_quota(counts, 500)
        self.assertLessEqual(quota["a"], 10)
        self.assertEqual(sum(quota.values()), 500)

    def test_target_exceeds_total_caps_at_total(self):
        counts = {"a": 30, "b": 20}
        quota = allocate_stratified_quota(counts, 200)
        self.assertEqual(quota["a"], 30)
        self.assertEqual(quota["b"], 20)
        self.assertEqual(sum(quota.values()), 50)

    def test_largest_remainder_distributes_leftover(self):
        # 7 over 3 equal strata -> floors 2,2,2 + 1 remainder
        counts = {"a": 100, "b": 100, "c": 100}
        quota = allocate_stratified_quota(counts, 7)
        self.assertEqual(sum(quota.values()), 7)
        self.assertEqual(sorted(quota.values()), [2, 2, 3])

    def test_zero_target_and_empty_strata(self):
        self.assertEqual(allocate_stratified_quota({"a": 5}, 0), {"a": 0})
        self.assertEqual(allocate_stratified_quota({}, 10), {})
        self.assertEqual(allocate_stratified_quota({"a": 0}, 10), {"a": 0})

    def test_sample_indices_deterministic_and_bounded(self):
        a = sample_indices(100, 10, seed=42)
        b = sample_indices(100, 10, seed=42)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 10)
        self.assertTrue(all(0 <= i < 100 for i in a))
        self.assertEqual(a, sorted(a))
        # quota > available -> clamps
        self.assertEqual(len(sample_indices(5, 99, seed=1)), 5)
        self.assertEqual(sample_indices(0, 5, seed=1), [])

    def test_stratum_sub_seed_process_stable(self):
        # Pinned values: zlib.crc32 is platform/process stable, so a change
        # here means the sampling universe silently shifted under --seed.
        self.assertEqual(stratum_sub_seed(42, "cape_town:aerial_2025"), 42 + 8868)
        self.assertEqual(stratum_sub_seed(42, "johannesburg:vexcel_2024"), 42 + 8979)
        # distinct strata must not collide on the derived seed
        self.assertNotEqual(
            stratum_sub_seed(42, "cape_town:aerial_2025"),
            stratum_sub_seed(42, "johannesburg:vexcel_2024"),
        )


# ──────────────────────────────────────────────────────────────────────────
# Background-region proposal classification
# ──────────────────────────────────────────────────────────────────────────
class ProposalOverlapTests(unittest.TestCase):
    def test_proposal_inside_gt_counts_as_labeled(self):
        gt = [box(0, 0, 100, 100)]
        prop = box(10, 10, 20, 20)  # fully inside GT
        self.assertTrue(proposal_overlaps_gt(prop, gt, iof_threshold=0.1))

    def test_disjoint_proposal_is_background(self):
        gt = [box(0, 0, 50, 50)]
        prop = box(200, 200, 220, 220)
        self.assertFalse(proposal_overlaps_gt(prop, gt, iof_threshold=0.1))

    def test_tiny_clip_below_threshold_is_background(self):
        gt = [box(0, 0, 100, 100)]
        # proposal mostly outside; only 5% of its area overlaps GT
        prop = box(95, 95, 195, 195)  # area=10000, overlap with GT = 5x5=25
        self.assertFalse(proposal_overlaps_gt(prop, gt, iof_threshold=0.1))


# ──────────────────────────────────────────────────────────────────────────
# Audit CSV schema
# ──────────────────────────────────────────────────────────────────────────
def _audit_row(label="", cap="", **over):
    row = {c: "" for c in AUDIT_CSV_COLUMNS}
    row.update({
        "audit_id": "ct:aerial_2025:G1:T:0_0__p00",
        "chip_uid": "ct:aerial_2025:G1:T:0_0",
        "region": "cape_town",
        "imagery_layer": "aerial_2025",
        "grid_id": "G1",
        "tile_stem": "T",
        "x0": "0", "y0": "0", "chip_size": "400",
        "proposal_index": "0", "score": "0.06",
        "audit_label": label, "ignore_area_cap_m2": cap,
    })
    row.update(over)
    return row


class AuditSchemaTests(unittest.TestCase):
    def test_valid_rows_pass(self):
        rows = [_audit_row("confirmed_pv"), _audit_row("lookalike"),
                _audit_row("ignore_candidate", cap="25.0"), _audit_row("")]
        self.assertEqual(validate_audit_rows(rows), [])

    def test_invalid_label_caught(self):
        errs = validate_audit_rows([_audit_row("definitely_pv")])
        self.assertTrue(any("invalid audit_label" in e for e in errs))

    def test_ignore_candidate_requires_cap(self):
        errs = validate_audit_rows([_audit_row("ignore_candidate", cap="")])
        self.assertTrue(any("ignore_area_cap_m2" in e for e in errs))

    def test_ignore_candidate_cap_must_be_numeric(self):
        errs = validate_audit_rows([_audit_row("ignore_candidate", cap="big")])
        self.assertTrue(any("not numeric" in e for e in errs))

    def test_missing_column_caught(self):
        bad = _audit_row("confirmed_pv")
        del bad["score"]
        errs = validate_audit_rows([bad])
        self.assertTrue(any("missing columns" in e for e in errs))

    def test_all_label_values_are_accepted(self):
        rows = []
        for lab in AUDIT_LABEL_VALUES:
            cap = "10.0" if lab == "ignore_candidate" else ""
            rows.append(_audit_row(lab, cap=cap))
        self.assertEqual(validate_audit_rows(rows), [])

    def test_id_helpers(self):
        uid = make_chip_uid("cape_town", "aerial_2025", "G1", "G1_0_0_geo", 300, 0)
        self.assertEqual(uid, "cape_town:aerial_2025:G1:G1_0_0_geo:300_0")
        self.assertEqual(make_audit_id(uid, 3), uid + "__p03")
        self.assertEqual(stratum_key("ct", "aerial_2025"), "ct:aerial_2025")


# ──────────────────────────────────────────────────────────────────────────
# Gate decision
# ──────────────────────────────────────────────────────────────────────────
def _gate_rows(specs):
    """specs: list of (chip_uid, region, layer, [labels...]) -> audit rows."""
    rows = []
    for chip_uid, region, layer, labels in specs:
        for j, lab in enumerate(labels):
            cap = "10.0" if lab == "ignore_candidate" else ""
            rows.append({
                "audit_id": f"{chip_uid}__p{j:02d}",
                "chip_uid": chip_uid, "region": region, "imagery_layer": layer,
                "grid_id": "G", "tile_stem": "T", "x0": "0", "y0": "0",
                "chip_size": "400", "proposal_index": str(j), "score": "0.06",
                "proposal_area_m2": "", "max_iof_vs_gt": "0.0", "n_gt_in_chip": "1",
                "chip_png": "", "audit_label": lab, "ignore_area_cap_m2": cap,
                "audit_notes": "", "reviewed_at": "",
            })
    return rows


class GateDecisionTests(unittest.TestCase):
    def test_pass_at_or_above_threshold(self):
        # 20 chips, 1 affected -> 5% == threshold -> PASS (>=)
        specs = [(f"c{i}", "ct", "aerial_2025",
                  ["confirmed_pv"] if i == 0 else ["lookalike"]) for i in range(20)]
        r = compute_gate(_gate_rows(specs), threshold=0.05)
        self.assertAlmostEqual(r.affected_rate, 0.05)
        self.assertEqual(r.decision, "PASS")
        self.assertEqual(r.n_chips_affected, 1)
        self.assertEqual(r.n_chips_total, 20)

    def test_kill_below_threshold(self):
        # 100 chips, 2 affected -> 2% < 5% -> KILL
        specs = [(f"c{i}", "ct", "aerial_2025",
                  ["confirmed_pv"] if i < 2 else ["not_pv_other"]) for i in range(100)]
        r = compute_gate(_gate_rows(specs), threshold=0.05)
        self.assertAlmostEqual(r.affected_rate, 0.02)
        self.assertEqual(r.decision, "KILL")

    def test_chip_affected_if_any_proposal_is_pv(self):
        # one chip with [lookalike, confirmed_pv] is affected once, not twice
        specs = [("c0", "ct", "aerial_2025", ["lookalike", "confirmed_pv"]),
                 ("c1", "ct", "aerial_2025", ["not_pv_other"])]
        r = compute_gate(_gate_rows(specs), threshold=0.05)
        self.assertEqual(r.n_chips_affected, 1)
        self.assertEqual(r.n_chips_total, 2)
        self.assertEqual(r.n_proposals_confirmed_pv, 1)
        self.assertEqual(r.n_proposals_lookalike, 1)

    def test_uncertain_does_not_count_as_decided_or_affected(self):
        specs = [("c0", "ct", "aerial_2025", ["uncertain"])]
        r = compute_gate(_gate_rows(specs), threshold=0.05, min_decided_chips=1)
        self.assertEqual(r.n_chips_decided, 0)
        self.assertEqual(r.decision, "INSUFFICIENT_DATA")

    def test_insufficient_data_when_nothing_decided(self):
        specs = [("c0", "ct", "aerial_2025", [""]),
                 ("c1", "ct", "aerial_2025", [""])]
        r = compute_gate(_gate_rows(specs), threshold=0.05, min_decided_chips=1)
        self.assertEqual(r.decision, "INSUFFICIENT_DATA")

    def test_per_stratum_breakdown(self):
        specs = [
            ("a0", "cape_town", "aerial_2025", ["confirmed_pv"]),
            ("a1", "cape_town", "aerial_2025", ["lookalike"]),
            ("b0", "johannesburg", "vexcel_2024", ["not_pv_other"]),
        ]
        r = compute_gate(_gate_rows(specs), threshold=0.05)
        strata = {s.stratum: s for s in r.per_stratum}
        self.assertIn("cape_town:aerial_2025", strata)
        self.assertIn("johannesburg:vexcel_2024", strata)
        self.assertEqual(strata["cape_town:aerial_2025"].n_chips_affected, 1)
        self.assertEqual(strata["cape_town:aerial_2025"].n_chips_total, 2)
        self.assertEqual(strata["johannesburg:vexcel_2024"].n_chips_affected, 0)

    def test_invalid_threshold_raises(self):
        with self.assertRaises(ValueError):
            compute_gate([], threshold=0.0)
        with self.assertRaises(ValueError):
            compute_gate([], threshold=1.5)

    def test_default_threshold_is_five_percent(self):
        self.assertAlmostEqual(DEFAULT_GATE_THRESHOLD, 0.05)

    def test_sampled_chips_denominator_counts_zero_proposal_chips(self):
        # 1 audited chip (affected), but 20 chips were sampled overall.
        # True rate = 1/20 = 5% (PASS), not 1/1 = 100%.
        audit = _gate_rows([("c0", "ct", "aerial_2025", ["confirmed_pv"])])
        manifest = [
            {"chip_uid": f"c{i}", "region": "ct", "imagery_layer": "aerial_2025"}
            for i in range(20)
        ]
        r = compute_gate(audit, threshold=0.05, sampled_chips=manifest)
        self.assertEqual(r.n_chips_total, 20)
        self.assertEqual(r.n_chips_affected, 1)
        self.assertAlmostEqual(r.affected_rate, 0.05)
        self.assertEqual(r.decision, "PASS")

    def test_sampled_chips_denominator_can_flip_to_kill(self):
        # Same single affected chip, but 100 sampled -> 1% < 5% -> KILL.
        audit = _gate_rows([("c0", "ct", "aerial_2025", ["confirmed_pv"])])
        manifest = [
            {"chip_uid": f"c{i}", "region": "ct", "imagery_layer": "aerial_2025"}
            for i in range(100)
        ]
        r = compute_gate(audit, threshold=0.05, sampled_chips=manifest)
        self.assertEqual(r.n_chips_total, 100)
        self.assertAlmostEqual(r.affected_rate, 0.01)
        self.assertEqual(r.decision, "KILL")

    def test_audit_chip_absent_from_manifest_still_counted(self):
        # Defensive: an audited chip not in the manifest still adds to the
        # denominator (manifest drift must not hide an affected chip).
        audit = _gate_rows([("orphan", "ct", "aerial_2025", ["confirmed_pv"])])
        manifest = [{"chip_uid": "c0", "region": "ct", "imagery_layer": "aerial_2025"}]
        r = compute_gate(audit, threshold=0.05, sampled_chips=manifest)
        self.assertEqual(r.n_chips_total, 2)  # c0 (sampled) + orphan (audited)
        self.assertEqual(r.n_chips_affected, 1)


if __name__ == "__main__":
    unittest.main()
