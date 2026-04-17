import math
import unittest

import geopandas as gpd
from shapely.geometry import box

from scripts.analysis.cluster_level_eval import (
    build_overlap_clusters,
    compute_cluster_metrics,
    summarize_cluster_metrics,
)


def _gdf(geoms):
    return gpd.GeoDataFrame({"geometry": geoms}, crs="EPSG:3857")


class ClusterLevelEvalTests(unittest.TestCase):
    def test_build_overlap_clusters_merges_one_pred_with_two_gt_into_single_component(self):
        gt = _gdf([box(0, 0, 1, 1), box(1.0, 0, 2.0, 1)])
        pred = _gdf([box(0, 0, 2.0, 1)])

        clusters = build_overlap_clusters(gt, pred, edge_iou_threshold=0.0)

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["gt_indices"], {0, 1})
        self.assertEqual(clusters[0]["pred_indices"], {0})

    def test_compute_cluster_metrics_supports_many_to_many_matching_with_cardinality_penalty(self):
        gt = _gdf([box(0, 0, 1, 1), box(1.0, 0, 2.0, 1)])
        pred = _gdf([box(0, 0, 2.0, 1)])
        cluster = {"gt_indices": {0, 1}, "pred_indices": {0}}

        metrics = compute_cluster_metrics(gt, pred, cluster)

        self.assertAlmostEqual(metrics["area_precision"], 1.0)
        self.assertAlmostEqual(metrics["area_recall"], 1.0)
        self.assertAlmostEqual(metrics["area_f1"], 1.0)
        self.assertAlmostEqual(metrics["cardinality_penalty"], 0.5)
        self.assertAlmostEqual(metrics["balanced_score"], 0.5)

    def test_summarize_cluster_metrics_counts_fp_and_fn_components(self):
        gt = _gdf([box(0, 0, 1, 1), box(10, 10, 11, 11)])
        pred = _gdf([box(0, 0, 1, 1), box(20, 20, 21, 21)])

        summary = summarize_cluster_metrics(
            gt,
            pred,
            edge_iou_threshold=0.0,
            match_coverage_threshold=0.5,
            match_purity_threshold=0.5,
        )

        self.assertEqual(summary["matched_cluster_count"], 1)
        self.assertEqual(summary["fp_cluster_count"], 1)
        self.assertEqual(summary["fn_cluster_count"], 1)
        self.assertAlmostEqual(summary["cluster_precision"], 0.5)
        self.assertAlmostEqual(summary["cluster_recall"], 0.5)
        self.assertAlmostEqual(summary["cluster_f1"], 0.5)

    def test_summarize_cluster_metrics_treats_fragmented_prediction_as_match_when_coverage_and_purity_pass(self):
        gt = _gdf([box(0, 0, 2, 1)])
        pred = _gdf([box(0, 0, 1, 1), box(1, 0, 2, 1)])

        summary = summarize_cluster_metrics(
            gt,
            pred,
            edge_iou_threshold=0.0,
            match_coverage_threshold=0.9,
            match_purity_threshold=0.9,
        )

        self.assertEqual(summary["matched_cluster_count"], 1)
        self.assertEqual(summary["fp_cluster_count"], 0)
        self.assertEqual(summary["fn_cluster_count"], 0)
        self.assertAlmostEqual(summary["mean_cardinality_penalty"], 0.5)
        self.assertAlmostEqual(summary["mean_balanced_score"], 0.5)


if __name__ == "__main__":
    unittest.main()
