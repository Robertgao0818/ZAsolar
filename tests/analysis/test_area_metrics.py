"""Unit tests for core.area_metrics — the Tier-1 aggregate-area statistics
kernel extracted from scripts/analysis/area_aggregate_eval.py (2026-06-12).

Covers σ_Bw (B-weighted dispersion), RMSE, agg_F1, through-origin β, OLS R²,
bootstrap CI determinism (fixed seed), and the empty / single-row edge cases.
Synthetic per-grid rows only — no GeoPackage I/O.
"""

import math
import unittest

from core.area_metrics import _bootstrap_ci, _ols_regression, summarize


def _row(region, run, gt_total, pred_total, inter, area_F1,
         model_version="v3c", imagery_layer="vexcel", grid_id="G0001"):
    """Build a per-grid row carrying exactly the keys summarize() reads."""
    abs_err = pred_total - gt_total
    return {
        "region": region,
        "model_run": run,
        "model_version": model_version,
        "imagery_layer": imagery_layer,
        "grid_id": grid_id,
        "gt_total_m2": gt_total,
        "pred_total_m2": pred_total,
        "inter_m2": inter,
        "area_F1": area_F1,
        "abs_error_m2": abs_err,
        "abs_rel_error": abs(abs_err) / gt_total if gt_total else 0.0,
        "signed_rel_error": abs_err / gt_total if gt_total else 0.0,
        "pred_gt_ratio": pred_total / gt_total if gt_total else 0.0,
    }


class SummarizeShapeTests(unittest.TestCase):
    def test_empty_rows_return_empty_list(self):
        self.assertEqual(summarize([]), [])

    def test_one_bucket_per_region_run_pair_sorted(self):
        rows = [
            _row("jhb", "runB", 1000.0, 950.0, 900.0, 0.9),
            _row("ct", "runA", 1000.0, 950.0, 900.0, 0.9),
            _row("ct", "runA", 2000.0, 2100.0, 1900.0, 0.92),
        ]
        out = summarize(rows)
        self.assertEqual(len(out), 2)
        # sorted by (region, model_run): ct/runA first, then jhb/runB.
        self.assertEqual((out[0]["region"], out[0]["model_run"]), ("ct", "runA"))
        self.assertEqual((out[1]["region"], out[1]["model_run"]), ("jhb", "runB"))
        self.assertEqual(out[0]["n_grids"], 2)
        self.assertEqual(out[1]["n_grids"], 1)

    def test_passthrough_fields_taken_from_first_item(self):
        rows = [
            _row("ct", "runA", 1000.0, 950.0, 900.0, 0.9,
                 model_version="mvX", imagery_layer="layerY"),
            _row("ct", "runA", 2000.0, 2100.0, 1900.0, 0.92,
                 model_version="mvZ", imagery_layer="layerW"),
        ]
        out = summarize(rows)
        self.assertEqual(out[0]["model_version"], "mvX")
        self.assertEqual(out[0]["imagery_layer"], "layerY")


class SummarizeMetricTests(unittest.TestCase):
    def _multi(self):
        return [
            _row("ct", "runA", 1000.0, 950.0, 900.0, 0.92),
            _row("ct", "runA", 2000.0, 2200.0, 1800.0, 0.86),
            _row("ct", "runA", 500.0, 480.0, 450.0, 0.90),
            _row("ct", "runA", 1500.0, 1700.0, 1400.0, 0.88),
        ]

    def test_agg_f1_matches_manual_aggregate(self):
        out = summarize(self._multi())[0]
        gt = 1000.0 + 2000.0 + 500.0 + 1500.0
        pred = 950.0 + 2200.0 + 480.0 + 1700.0
        inter = 900.0 + 1800.0 + 450.0 + 1400.0
        agg_R = inter / gt
        agg_P = inter / pred
        agg_F1 = 2 * agg_R * agg_P / (agg_R + agg_P)
        self.assertEqual(out["agg_area_R"], round(agg_R, 4))
        self.assertEqual(out["agg_area_P"], round(agg_P, 4))
        self.assertEqual(out["agg_area_F1"], round(agg_F1, 4))

    def test_bulk_ratio_matches_manual(self):
        out = summarize(self._multi())[0]
        gt = 1000.0 + 2000.0 + 500.0 + 1500.0
        pred = 950.0 + 2200.0 + 480.0 + 1700.0
        self.assertEqual(out["bulk_pred_gt_ratio"], round(pred / gt, 4))

    def test_sigma_bw_matches_manual_b_weighted_dispersion(self):
        items = self._multi()
        out = summarize(items)[0]
        gt = sum(r["gt_total_m2"] for r in items)
        ratios = [r["pred_total_m2"] / r["gt_total_m2"] for r in items]
        mean_ratio = sum(ratios) / len(ratios)
        sigma_bw = math.sqrt(sum(
            (r["gt_total_m2"] / gt) * (rt - mean_ratio) ** 2
            for r, rt in zip(items, ratios)
        ))
        self.assertEqual(out["std_ratio_Bw"], round(sigma_bw, 4))
        self.assertEqual(out["cv_ratio_Bw"], round(sigma_bw / mean_ratio, 4))

    def test_rmse_matches_manual(self):
        items = self._multi()
        out = summarize(items)[0]
        eps = [r["pred_total_m2"] - r["gt_total_m2"] for r in items]
        rmse = math.sqrt(sum(e ** 2 for e in eps) / len(eps))
        self.assertEqual(out["rmse_m2"], round(rmse, 2))

    def test_through_origin_beta_matches_manual(self):
        items = self._multi()
        out = summarize(items)[0]
        Bs = [r["gt_total_m2"] for r in items]
        As = [r["pred_total_m2"] for r in items]
        beta_o = sum(b * a for b, a in zip(Bs, As)) / sum(b ** 2 for b in Bs)
        self.assertEqual(out["thru0_slope"], round(beta_o, 4))

    def test_ols_r2_matches_helper(self):
        items = self._multi()
        out = summarize(items)[0]
        Bs = [r["gt_total_m2"] for r in items]
        As = [r["pred_total_m2"] for r in items]
        reg = _ols_regression(Bs, As)
        self.assertEqual(out["ols_slope"], round(reg["slope"], 4))
        self.assertEqual(out["ols_intercept_m2"], round(reg["intercept"], 2))
        self.assertEqual(out["ols_r2"], round(reg["r2"], 4))

    def test_bootstrap_ci_is_deterministic_across_calls(self):
        # summarize seeds _bootstrap_ci(seed=0) implicitly; two identical
        # inputs must produce identical CI bounds.
        items = self._multi()
        a = summarize(items)[0]
        b = summarize(items)[0]
        for k in ("f1_pg_CI95_lo", "f1_pg_CI95_hi",
                  "std_ratio_CI95_lo", "std_ratio_CI95_hi",
                  "rmse_CI95_lo", "rmse_CI95_hi"):
            self.assertEqual(a[k], b[k], k)
            self.assertFalse(math.isnan(a[k]), k)


class SingleAndTwoRowEdgeTests(unittest.TestCase):
    def test_single_row_std_and_bootstrap_are_nan(self):
        out = summarize([_row("ct", "runC", 1000.0, 900.0, 850.0, 0.89)])[0]
        self.assertEqual(out["n_grids"], 1)
        # std_ratio uses ddof=1 with n<2 -> nan; bootstrap n<3 -> nan.
        self.assertTrue(math.isnan(out["std_ratio"]))
        self.assertTrue(math.isnan(out["std_ratio_CI95_lo"]))
        self.assertTrue(math.isnan(out["std_ratio_CI95_hi"]))
        self.assertTrue(math.isnan(out["f1_pg_CI95_lo"]))
        self.assertTrue(math.isnan(out["rmse_CI95_lo"]))
        # OLS / through-origin need n>=2 -> None / nan.
        self.assertIsNone(out["ols_slope"])
        self.assertIsNone(out["ols_r2"])
        self.assertTrue(math.isnan(out["thru0_slope"]))
        # std_logratio needs >=2 valid (A>0,B>0) pairs -> nan with one grid.
        self.assertTrue(math.isnan(out["std_logratio"]))

    def test_two_rows_std_defined_bootstrap_still_nan(self):
        out = summarize([
            _row("ct", "runD", 1000.0, 950.0, 900.0, 0.90),
            _row("ct", "runD", 2000.0, 1900.0, 1800.0, 0.88),
        ])[0]
        self.assertEqual(out["n_grids"], 2)
        # n>=2 -> std_ratio defined (not nan); bootstrap n<3 -> still nan.
        self.assertFalse(math.isnan(out["std_ratio"]))
        self.assertTrue(math.isnan(out["std_ratio_CI95_lo"]))
        self.assertFalse(math.isnan(out["std_logratio"]))
        self.assertIsNotNone(out["ols_slope"])
        self.assertFalse(math.isnan(out["thru0_slope"]))

    def test_zero_pred_grid_excluded_from_logratio_but_kept_in_totals(self):
        items = [
            _row("ct", "runE", 1000.0, 0.0, 0.0, 0.0),
            _row("ct", "runE", 1500.0, 1400.0, 1300.0, 0.90),
            _row("ct", "runE", 2000.0, 2100.0, 1900.0, 0.92),
        ]
        out = summarize(items)[0]
        self.assertEqual(out["n_grids"], 3)
        self.assertEqual(out["pred_total_m2"], round(0.0 + 1400.0 + 2100.0, 2))
        # log-ratio uses only A>0 & B>0 -> 2 valid pairs -> defined.
        self.assertFalse(math.isnan(out["std_logratio"]))


class BootstrapCIHelperTests(unittest.TestCase):
    def test_bootstrap_returns_nan_below_three(self):
        lo, hi = _bootstrap_ci([1.0, 2.0], lambda v: float(v.mean()))
        self.assertTrue(math.isnan(lo))
        self.assertTrue(math.isnan(hi))

    def test_bootstrap_is_seed_deterministic(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        a = _bootstrap_ci(vals, lambda v: float(v.mean()), seed=0)
        b = _bootstrap_ci(vals, lambda v: float(v.mean()), seed=0)
        self.assertEqual(a, b)
        # Different seed -> (very likely) different bounds.
        c = _bootstrap_ci(vals, lambda v: float(v.mean()), seed=1)
        self.assertNotEqual(a, c)
        self.assertLessEqual(a[0], a[1])


class OLSHelperTests(unittest.TestCase):
    def test_perfect_line_gives_r2_one(self):
        reg = _ols_regression([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
        self.assertAlmostEqual(reg["slope"], 2.0)
        self.assertAlmostEqual(reg["intercept"], 0.0)
        self.assertAlmostEqual(reg["r2"], 1.0)

    def test_single_point_returns_none(self):
        reg = _ols_regression([1.0], [2.0])
        self.assertIsNone(reg["slope"])
        self.assertIsNone(reg["r2"])

    def test_constant_x_returns_none(self):
        reg = _ols_regression([3.0, 3.0, 3.0], [1.0, 2.0, 3.0])
        self.assertIsNone(reg["slope"])


if __name__ == "__main__":
    unittest.main()
