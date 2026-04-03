"""
太阳能板检测与评估完整流水线
Solar Panel Detection & Evaluation Pipeline

功能：
  1. 使用 geoai SolarPanelDetector 对 tiles/ 中的 GeoTIFF 进行检测
  2. 加载 g1238.gpkg / g1238.geojson 真值数据
  3. 基于 IoU 匹配进行多阈值评估
  4. 生成可视化图表
  5. 输出汇总报告和逐 tile CSV

依赖安装：
  pip install geoai-py geopandas shapely scikit-learn matplotlib seaborn rasterio
"""

import argparse
import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
from shapely.geometry import box
from shapely.ops import unary_union

from core.grid_utils import (
    COMBINED_ANNOTATION_GPKG,
    DEFAULT_GRID_ID,
    get_metric_crs,
    get_grid_paths,
    normalize_grid_id,
)

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════
# 配置常量
# ════════════════════════════════════════════════════════════════════════
BASE_DIR      = Path(__file__).parent
GRID_ID       = DEFAULT_GRID_ID
TILES_DIR     = BASE_DIR / "tiles" / GRID_ID
GT_GPKG       = BASE_DIR / "data" / "annotations" / f"{GRID_ID}.gpkg"
GT_GEOJSON    = BASE_DIR / "data" / "annotations" / f"{GRID_ID.lower()}.geojson"
OUTPUT_DIR    = BASE_DIR / "results" / GRID_ID
MASKS_DIR     = OUTPUT_DIR / "masks"
VECTORS_DIR   = OUTPUT_DIR / "vectors"
BUILDINGS_GPKG = BASE_DIR / "buildings.gpkg"

# geoai 检测参数
CONFIDENCE_THRESHOLD = 0.3
MASK_THRESHOLD       = 0.3
MIN_OBJECT_AREA      = 5      # 后处理面积过滤（m²），<5m² 几乎全是碎片
MAX_ELONGATION       = 8.0    # 后处理长宽比过滤（V1.2 校准后从 4.0 放宽）
# 大面积检测放宽 elongation（商业长条板 elong 9-10+）
ELONGATION_TIERED = [
    # (min_area_m2, max_elongation)
    (100, 15.0),   # 大型商业板
    (0,    8.0),   # 住宅板（默认）
]
MIN_SOLIDITY         = 0.0    # 暂不限制 solidity（TP/FP 分布重叠太大）
SHADOW_RGB_THRESH    = 60     # RGB 三通道均 < 此值视为阴影
POST_CONF_THRESHOLD  = 0.85   # 后处理置信度过滤（V3 Model C 校准，recall≥94%）
# 大面积检测使用更低的置信度阈值（商业太阳能板 confidence 系统性偏低）
CONF_TIERED = [
    # (min_area_m2, conf_threshold)  — 从大到小匹配，第一个命中的生效
    (200, 0.70),   # 超大型商业板
    (100, 0.65),   # 大型商业板
    (0,   0.85),   # 住宅板（默认）
]
OVERLAP              = 0.25
CHIP_SIZE            = (400, 400)
BATCH_SIZE           = 4

# 评估参数
IOU_THRESHOLDS       = [0.1, 0.2, 0.3, 0.5, 0.7]
DEFAULT_IOU          = 0.3
INPUT_CRS            = "EPSG:4326"   # QGIS 标注/交换、原始瓦片地理参考
DEFAULT_METRIC_CRS   = "EPSG:32734"
METRIC_CRS           = DEFAULT_METRIC_CRS
EXPORT_CRS           = INPUT_CRS     # 导出回 QGIS 时统一使用 4326

# 输出文件路径
PREDICTIONS_PATH         = OUTPUT_DIR / "predictions.geojson"
PREDICTIONS_METRIC_PATH  = OUTPUT_DIR / "predictions_metric.gpkg"
CONFIG_PATH              = OUTPUT_DIR / "config.json"
CONFIDENCE_HIST_PATH     = OUTPUT_DIR / "confidence_histogram.png"
PR_CURVE_PATH            = OUTPUT_DIR / "precision_recall_curve.png"
IOU_METRICS_PATH         = OUTPUT_DIR / "iou_threshold_metrics.png"
EVALUATION_CSV_PATH      = OUTPUT_DIR / "evaluation_per_tile.csv"
SIZE_STRATIFIED_CSV_PATH = OUTPUT_DIR / "size_stratified_metrics.csv"
SCRIPT_SHA256            = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def set_grid_context(grid_id: str = DEFAULT_GRID_ID,
                     output_subdir: str | None = None) -> None:
    """更新当前运行使用的 grid 路径上下文。"""
    global GRID_ID
    global TILES_DIR
    global GT_GPKG
    global GT_GEOJSON
    global OUTPUT_DIR
    global MASKS_DIR
    global VECTORS_DIR
    global METRIC_CRS
    global PREDICTIONS_PATH
    global PREDICTIONS_METRIC_PATH
    global CONFIG_PATH
    global CONFIDENCE_HIST_PATH
    global PR_CURVE_PATH
    global IOU_METRICS_PATH
    global EVALUATION_CSV_PATH
    global SIZE_STRATIFIED_CSV_PATH
    global ERROR_ANALYSIS_PATH
    global FN_ANALYSIS_PATH
    global PRESENCE_METRICS_PATH
    global FOOTPRINT_METRICS_PATH
    global AREA_ERROR_METRICS_PATH

    paths = get_grid_paths(grid_id, output_subdir=output_subdir)
    GRID_ID = paths.grid_id
    TILES_DIR = paths.tiles_dir
    GT_GPKG = paths.gt_gpkg
    GT_GEOJSON = paths.gt_geojson
    OUTPUT_DIR = paths.output_dir
    MASKS_DIR = OUTPUT_DIR / "masks"
    VECTORS_DIR = OUTPUT_DIR / "vectors"
    METRIC_CRS = get_metric_crs(GRID_ID)
    PREDICTIONS_PATH = OUTPUT_DIR / "predictions.geojson"
    PREDICTIONS_METRIC_PATH = OUTPUT_DIR / "predictions_metric.gpkg"
    CONFIG_PATH = OUTPUT_DIR / "config.json"
    CONFIDENCE_HIST_PATH = OUTPUT_DIR / "confidence_histogram.png"
    PR_CURVE_PATH = OUTPUT_DIR / "precision_recall_curve.png"
    IOU_METRICS_PATH = OUTPUT_DIR / "iou_threshold_metrics.png"
    EVALUATION_CSV_PATH = OUTPUT_DIR / "evaluation_per_tile.csv"
    SIZE_STRATIFIED_CSV_PATH = OUTPUT_DIR / "size_stratified_metrics.csv"
    ERROR_ANALYSIS_PATH = OUTPUT_DIR / "error_analysis.csv"
    FN_ANALYSIS_PATH = OUTPUT_DIR / "fn_analysis.csv"
    PRESENCE_METRICS_PATH = OUTPUT_DIR / "presence_metrics.csv"
    FOOTPRINT_METRICS_PATH = OUTPUT_DIR / "footprint_metrics.csv"
    AREA_ERROR_METRICS_PATH = OUTPUT_DIR / "area_error_metrics.csv"


set_grid_context(DEFAULT_GRID_ID)


# ════════════════════════════════════════════════════════════════════════
# 辅助函数：CRS 统一
# ════════════════════════════════════════════════════════════════════════
def ensure_crs(gdf: gpd.GeoDataFrame,
               assumed_crs: str,
               label: str) -> gpd.GeoDataFrame:
    """为缺失 CRS 的 GeoDataFrame 补默认 CRS。"""
    if gdf.crs is None:
        print(f"  [INFO] {label} 无 CRS，假设为 {assumed_crs}")
        gdf = gdf.set_crs(assumed_crs)
    return gdf


def to_metric_crs(gdf: gpd.GeoDataFrame,
                  assumed_crs: str,
                  label: str) -> gpd.GeoDataFrame:
    """统一到米制计算 CRS。"""
    gdf = ensure_crs(gdf, assumed_crs=assumed_crs, label=label)
    if str(gdf.crs) != METRIC_CRS:
        gdf = gdf.to_crs(METRIC_CRS)
    return gdf


def to_export_crs(gdf: gpd.GeoDataFrame,
                  assumed_crs: str,
                  label: str) -> gpd.GeoDataFrame:
    """统一到 QGIS 友好的导出 CRS。"""
    gdf = ensure_crs(gdf, assumed_crs=assumed_crs, label=label)
    if str(gdf.crs) != EXPORT_CRS:
        gdf = gdf.to_crs(EXPORT_CRS)
    return gdf


# ════════════════════════════════════════════════════════════════════════
# 辅助函数：实验可追溯性
# ════════════════════════════════════════════════════════════════════════
def _json_ready(value):
    """将配置对象标准化为可稳定序列化的 JSON 值。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in sorted(value.items())}
    return value


def load_postproc_config(config_path: str | Path) -> dict:
    """从 JSON 文件加载后处理参数，返回可用于覆盖默认值的 dict。

    支持的键: post_conf_threshold, min_object_area, max_elongation
    (未来扩展: classifier_model, nms_strategy, dual_elongation 等)
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"后处理配置文件不存在: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    known_keys = {
        "post_conf_threshold", "min_object_area", "max_elongation",
    }
    params = {k: v for k, v in cfg.items() if k in known_keys}
    extra = set(cfg.keys()) - known_keys - {"_meta"}
    if extra:
        print(f"  [INFO] 后处理配置中忽略未知键: {extra}")
    print(f"  [POSTPROC] 从 {path.name} 加载: {params}")
    return params


def build_detection_config(
    chip_size=None,
    overlap=None,
    min_object_area=None,
    confidence_threshold=None,
    mask_threshold=None,
    post_conf_threshold=None,
    max_elongation=None,
    output_dir=None,
    model_path=None,
) -> dict:
    """构建检测阶段配置快照，用于结果复用校验。"""
    return {
        "grid_id": GRID_ID,
        "tiles_dir": Path(TILES_DIR).resolve(),
        "output_dir": Path(output_dir or OUTPUT_DIR).resolve(),
        "script_sha256": SCRIPT_SHA256,
        "model_path": str(Path(model_path).resolve()) if model_path else None,
        "chip_size": chip_size or CHIP_SIZE,
        "overlap": overlap if overlap is not None else OVERLAP,
        "min_object_area": (
            min_object_area if min_object_area is not None else MIN_OBJECT_AREA
        ),
        "confidence_threshold": (
            confidence_threshold
            if confidence_threshold is not None else CONFIDENCE_THRESHOLD
        ),
        "mask_threshold": (
            mask_threshold if mask_threshold is not None else MASK_THRESHOLD
        ),
        "post_conf_threshold": (
            post_conf_threshold
            if post_conf_threshold is not None else POST_CONF_THRESHOLD
        ),
        "max_elongation": (
            max_elongation if max_elongation is not None else MAX_ELONGATION
        ),
        "min_solidity": MIN_SOLIDITY,
        "shadow_rgb_thresh": SHADOW_RGB_THRESH,
        "batch_size": BATCH_SIZE,
        "input_crs": INPUT_CRS,
        "metric_crs": METRIC_CRS,
        "export_crs": EXPORT_CRS,
    }


def write_run_config(
    config_path: Path,
    config: dict,
    *,
    result_count: int | None = None,
    evaluation_config: dict | None = None,
) -> None:
    """将当前实验配置写入 config.json。"""
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": _json_ready(config),
        "artifacts": {
            "predictions_metric": "predictions_metric.gpkg",
            "predictions_export": "predictions.geojson",
        },
    }
    if result_count is not None:
        payload["result_count"] = int(result_count)
    if evaluation_config is not None:
        payload["evaluation_config"] = evaluation_config
    config_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_run_config(config_path: Path) -> dict | None:
    """读取 config.json。读取失败时返回 None。"""
    if not config_path.exists():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def append_evaluation_config(config_path: Path, evaluation_config: dict) -> None:
    """Append evaluation_config to existing config.json (non-destructive)."""
    payload = load_run_config(config_path)
    if payload is None:
        return
    payload["evaluation_config"] = evaluation_config
    config_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_geoai_input_raster(tif_path: Path, scratch_dir: Path) -> Path:
    """为 geoai 生成兼容输入，避免 YCbCr 源图复制 profile 后导致 mask 写出失败。"""
    with rasterio.open(str(tif_path)) as src:
        profile = src.profile.copy()
        photometric = str(profile.get("photometric", "")).lower()

        if photometric != "ycbcr":
            return tif_path

        scratch_dir.mkdir(parents=True, exist_ok=True)
        prepared_path = scratch_dir / f"{tif_path.stem}_geoai_input.tif"
        if prepared_path.exists():
            return prepared_path

        write_profile = profile.copy()
        for key in ("photometric", "compress", "jpeg_quality", "jpegtablesmode"):
            write_profile.pop(key, None)
        write_profile.update(driver="GTiff", compress="lzw")

        with rasterio.open(str(prepared_path), "w", **write_profile) as dst:
            for band_idx in range(1, src.count + 1):
                dst.write(src.read(band_idx), band_idx)

    print(f"    [INFO] 为 geoai 准备兼容输入: {prepared_path.name}")
    return prepared_path


def is_empty_geometry_result_error(exc: Exception) -> bool:
    """识别 geoai 在空矢量结果上抛出的已知异常。"""
    return "Assigning CRS to a GeoDataFrame without a geometry column" in str(exc)


def should_reuse_predictions(
    output_dir: Path,
    config: dict,
    *,
    force: bool = False,
) -> bool:
    """判断现有预测结果是否可直接复用。"""
    predictions_exist = (
        (output_dir / "predictions_metric.gpkg").exists()
        or (output_dir / "predictions.geojson").exists()
    )
    if not predictions_exist:
        return False

    if force:
        print("[INFO] --force 已指定，忽略现有预测结果并重新检测")
        return False

    saved = load_run_config(output_dir / "config.json")
    if saved is None:
        raise RuntimeError(
            f"{output_dir / 'config.json'} 缺失或损坏，无法确认现有结果来自哪套配置。"
            " 请使用 --force 重新检测。"
        )

    saved_config = saved.get("config")
    current_config = _json_ready(config)
    if saved_config != current_config:
        raise RuntimeError(
            "现有预测结果与当前配置不一致，继续评估会变成“新代码评估旧结果”。"
            " 请使用 --force 重新检测。"
        )

    print(f"[INFO] 检测配置一致，复用已有结果: {output_dir}")
    return True


# ════════════════════════════════════════════════════════════════════════
# 辅助函数：空间 NMS 去重
# ════════════════════════════════════════════════════════════════════════
def spatial_nms(gdf: gpd.GeoDataFrame, iou_threshold: float = 0.5) -> gpd.GeoDataFrame:
    """
    空间非极大值抑制：合并 IoU > threshold 的重复检测。
    保留面积较大的多边形（通常更完整）。
    """
    if len(gdf) <= 1:
        return gdf

    keep = [True] * len(gdf)
    sindex = gdf.sindex

    for i in range(len(gdf)):
        if not keep[i]:
            continue
        geom_i = gdf.iloc[i].geometry
        candidates = list(sindex.intersection(geom_i.bounds))
        for j in candidates:
            if j <= i or not keep[j]:
                continue
            geom_j = gdf.iloc[j].geometry
            try:
                inter = geom_i.intersection(geom_j).area
                union = geom_i.area + geom_j.area - inter
                if union > 0 and (inter / union) > iou_threshold:
                    # 保留面积较大的
                    if geom_i.area >= geom_j.area:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
            except Exception:
                continue

    pre_nms = len(gdf)
    result = gdf[keep].copy()
    if pre_nms > len(result):
        print(f"\n空间 NMS: 移除了 {pre_nms - len(result)} 个重复检测")
    return result


# ════════════════════════════════════════════════════════════════════════
# 第一步：检测太阳能板
# ════════════════════════════════════════════════════════════════════════
def detect_solar_panels(
    chip_size=None,
    overlap=None,
    min_object_area=None,
    confidence_threshold=None,
    mask_threshold=None,
    post_conf_threshold=None,
    max_elongation=None,
    output_dir=None,
    save_config=True,
    model_path=None,
) -> gpd.GeoDataFrame:
    """
    使用 geoai SolarPanelDetector 对每张 GeoTIFF 进行检测。
    参数可覆盖模块级常量，用于参数搜索。
    """
    _chip_size = chip_size or CHIP_SIZE
    _overlap = overlap if overlap is not None else OVERLAP
    _min_object_area = min_object_area if min_object_area is not None else MIN_OBJECT_AREA
    _confidence_threshold = confidence_threshold if confidence_threshold is not None else CONFIDENCE_THRESHOLD
    _mask_threshold = mask_threshold if mask_threshold is not None else MASK_THRESHOLD
    _post_conf_threshold = (
        post_conf_threshold if post_conf_threshold is not None else POST_CONF_THRESHOLD
    )
    _max_elongation = max_elongation if max_elongation is not None else MAX_ELONGATION
    _output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    _masks_dir = _output_dir / "masks"
    _vectors_dir = _output_dir / "vectors"
    _predictions_path = _output_dir / "predictions.geojson"
    _predictions_metric_path = _output_dir / "predictions_metric.gpkg"
    _config_path = _output_dir / "config.json"
    _config = build_detection_config(
        chip_size=_chip_size,
        overlap=_overlap,
        min_object_area=_min_object_area,
        confidence_threshold=_confidence_threshold,
        mask_threshold=_mask_threshold,
        post_conf_threshold=_post_conf_threshold,
        max_elongation=_max_elongation,
        output_dir=_output_dir,
        model_path=model_path,
    )

    _output_dir.mkdir(parents=True, exist_ok=True)
    _masks_dir.mkdir(parents=True, exist_ok=True)
    _vectors_dir.mkdir(parents=True, exist_ok=True)

    # 只处理带有地理参考信息的 _geo.tif 文件
    geo_tifs = sorted(TILES_DIR.glob(f"{GRID_ID}_*_*_geo.tif"))
    if not geo_tifs:
        # 回退到普通 .tif（不包含 _geo 后缀且不含 mosaic/mask 等关键词）
        geo_tifs = sorted([
            f for f in TILES_DIR.glob(f"{GRID_ID}_*_*.tif")
            if "_geo" not in f.stem and "mosaic" not in f.stem and "mask" not in f.stem
        ])

    if not geo_tifs:
        print("[ERROR] tiles/ 目录下未找到任何 GeoTIFF 文件")
        sys.exit(1)

    print(f"找到 {len(geo_tifs)} 张待检测的 GeoTIFF 文件")

    # ── 自动检测 GPU/CUDA ─────────────────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"\n[GPU] GPU acceleration enabled: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            device = "cpu"
            print("\n[WARN] CUDA GPU not found, using CPU (slower)")
    except ImportError:
        device = "cpu"
        print("\n[WARN] PyTorch not installed, using CPU")

    # ── 路径 A：geoai SolarPanelDetector ──────────────────────────────
    try:
        import geoai

        print(f"\n[路径 A] 使用 geoai.SolarPanelDetector 进行检测 (device={device})...")
        detector_kwargs = {"device": device}
        if model_path is not None:
            detector_kwargs["model_path"] = model_path
            print(f"[MODEL] 使用自定义模型权重: {model_path}")
        detector = geoai.SolarPanelDetector(**detector_kwargs)
        # Override geoai defaults with our pipeline thresholds
        detector.confidence_threshold = _confidence_threshold
        detector.min_object_area = _min_object_area

        # 预加载建筑轮廓（避免每个 tile 重复加载）
        bldg_union_cache = None
        bldg_crs_cache = None
        if BUILDINGS_GPKG.exists():
            try:
                _buildings = gpd.read_file(str(BUILDINGS_GPKG))
                if len(_buildings) > 0:
                    bldg_metric = to_metric_crs(
                        _buildings, assumed_crs=INPUT_CRS, label="建筑轮廓"
                    )
                    bldg_metric["geometry"] = bldg_metric.geometry.buffer(2)
                    bldg_crs_cache = _buildings.crs  # 原始 CRS
                    bldg_union_cache = {}  # 按 CRS 缓存
                    print(f"[建筑掩膜] 已加载 {len(_buildings)} 个建筑轮廓 (buffer=2m)")
                del _buildings
            except Exception as e:
                print(f"[WARN] 加载建筑轮廓失败: {e}")

        all_gdfs = []
        for idx, tif_path in enumerate(geo_tifs, 1):
            tile_name = tif_path.stem
            print(f"  [{idx}/{len(geo_tifs)}] 检测中: {tile_name}")

            try:
                prepared_tif_path = prepare_geoai_input_raster(
                    tif_path, _masks_dir / "_geoai_input_cache"
                )

                # 生成掩膜 → 矢量化
                mask_path = _masks_dir / f"{tile_name}_mask.tif"
                masks_result = detector.generate_masks(
                    str(prepared_tif_path),
                    output_path=str(mask_path),
                    confidence_threshold=_confidence_threshold,
                    mask_threshold=_mask_threshold,
                    min_object_area=_min_object_area,
                    overlap=_overlap,
                    chip_size=_chip_size,
                    batch_size=BATCH_SIZE,
                    verbose=False,
                )

                # 矢量化：正交化多边形
                vector_path = _vectors_dir / f"{tile_name}_vectors.geojson"
                try:
                    gdf_tile = geoai.orthogonalize(
                        input_path=masks_result,
                        output_path=str(vector_path),
                        epsilon=0.2,
                    )
                except Exception as exc:
                    if is_empty_geometry_result_error(exc):
                        print("    -> 未检测到可矢量化多边形")
                        continue
                    raise

                if gdf_tile is not None and len(gdf_tile) > 0:
                    # --- 从 mask band 2 回填 confidence ---
                    try:
                        import rasterstats as _rs
                        _conf_stats = _rs.zonal_stats(
                            gdf_tile, str(mask_path), band=2,
                            stats=["mean"], nodata=0,
                        )
                        gdf_tile["confidence"] = [
                            (s["mean"] / 255.0) if s["mean"] is not None else 0.0
                            for s in _conf_stats
                        ]
                    except Exception as _e:
                        print(f"    [WARN] confidence 回填失败: {_e}")

                    # --- 颜色过滤：去除阴影和反光 ---
                    try:
                        import rasterstats
                        stats_r = rasterstats.zonal_stats(gdf_tile, str(tif_path), band=1, stats=['mean'], nodata=0)
                        stats_g = rasterstats.zonal_stats(gdf_tile, str(tif_path), band=2, stats=['mean'], nodata=0)
                        stats_b = rasterstats.zonal_stats(gdf_tile, str(tif_path), band=3, stats=['mean'], nodata=0)
                        gdf_tile["mean_r"] = [s['mean'] if s['mean'] is not None else 0 for s in stats_r]
                        gdf_tile["mean_g"] = [s['mean'] if s['mean'] is not None else 0 for s in stats_g]
                        gdf_tile["mean_b"] = [s['mean'] if s['mean'] is not None else 0 for s in stats_b]

                        # 阴影过滤（RGB 三通道均低于阈值）+ 过曝过滤
                        is_shadow = ((gdf_tile["mean_r"] < SHADOW_RGB_THRESH)
                                     & (gdf_tile["mean_g"] < SHADOW_RGB_THRESH)
                                     & (gdf_tile["mean_b"] < SHADOW_RGB_THRESH))
                        is_too_bright = (gdf_tile["mean_r"] > 250) & (gdf_tile["mean_g"] > 250) & (gdf_tile["mean_b"] > 250)
                        valid_mask = ~(is_shadow | is_too_bright)

                        pre_count = len(gdf_tile)
                        gdf_tile = gdf_tile[valid_mask].copy()
                        if pre_count > len(gdf_tile):
                            print(f"    -> 颜色过滤: 移除了 {pre_count - len(gdf_tile)} 个极端阴影/反光")
                    except Exception as e:
                        print(f"    [WARN] RGB过滤失败: {e}")

                    # --- 建筑掩膜已禁用（OSM 数据滞后） ---

                    if len(gdf_tile) > 0:
                        # 添加几何属性用于后续过滤
                        gdf_tile = geoai.add_geometric_properties(gdf_tile)
                        # 添加来源 tile 信息
                        gdf_tile["source_tile"] = tile_name
                        all_gdfs.append(gdf_tile)
                        print(f"    → 检测到 {len(gdf_tile)} 个候选特征")
                    else:
                        print(f"    → 颜色过滤后无多边形剩余")
                else:
                    print(f"    → 未检测到太阳能板")

            except Exception as e:
                print(f"    [WARNING] 处理 {tile_name} 时出错: {e}")
                continue

        if not all_gdfs:
            print("[ERROR] 所有 tile 均未检测到太阳能板")
            sys.exit(1)

        # 合并所有检测结果
        pred_gdf = pd.concat(all_gdfs, ignore_index=True)
        pred_gdf = ensure_crs(
            pred_gdf,
            assumed_crs=METRIC_CRS,
            label="检测结果",
        )
        pred_gdf = pred_gdf.to_crs(METRIC_CRS)

        # 空间 NMS 去重：chip 重叠导致同一目标被重复检测
        pred_gdf = spatial_nms(pred_gdf, iou_threshold=0.5)

        # 面积过滤
        pre_filter_count = len(pred_gdf)
        if "area_m2" in pred_gdf.columns:
            pred_gdf = pred_gdf[pred_gdf["area_m2"] >= _min_object_area].copy()

        # 长宽比过滤：分面积段（大面积商业板 elongation 系统性偏高）
        if "elongation" in pred_gdf.columns and "area_m2" in pred_gdf.columns:
            elong_keep = pd.Series(False, index=pred_gdf.index)
            for min_area, max_elong in ELONGATION_TIERED:
                tier_mask = (pred_gdf["area_m2"] >= min_area) & ~elong_keep
                elong_keep |= tier_mask & (pred_gdf["elongation"] <= max_elong)
            pred_gdf = pred_gdf[elong_keep].copy()
            elong_desc = ", ".join(f"≥{a}m²→≤{e}" for a, e in ELONGATION_TIERED)
            elong_info = f"elongation({elong_desc})"
        elif "elongation" in pred_gdf.columns and _max_elongation < 999:
            pred_gdf = pred_gdf[pred_gdf["elongation"] <= _max_elongation].copy()
            elong_info = f"elongation<={_max_elongation}"
        else:
            elong_info = ""

        post_filter_count = len(pred_gdf)
        print(f"\n后处理过滤: {post_filter_count} / {pre_filter_count} 个多边形保留"
              f"（area>={_min_object_area}m² + {elong_info}）")

        # 确保有 confidence 字段
        if "confidence" not in pred_gdf.columns:
            # geoai orthogonalize 输出的字段可能叫 score 或 probability
            for alt_col in ["score", "probability", "prob", "conf"]:
                if alt_col in pred_gdf.columns:
                    pred_gdf["confidence"] = pred_gdf[alt_col]
                    break
            else:
                print("[INFO] 未找到置信度字段，使用默认值 0.5")
                pred_gdf["confidence"] = 0.5

        # 置信度过滤：分面积段阈值（大面积商业板 confidence 系统性偏低）
        pre_conf_count = len(pred_gdf)
        if "area_m2" in pred_gdf.columns:
            keep_mask = pd.Series(False, index=pred_gdf.index)
            for min_area, thresh in CONF_TIERED:
                tier_mask = (pred_gdf["area_m2"] >= min_area) & ~keep_mask
                keep_mask |= tier_mask & (pred_gdf["confidence"] >= thresh)
            pred_gdf = pred_gdf[keep_mask].copy()
            tier_desc = ", ".join(f"≥{a}m²→{t}" for a, t in CONF_TIERED)
            print(f"置信度过滤(分段): {len(pred_gdf)} / {pre_conf_count} 个多边形保留"
                  f"（{tier_desc}）")
        else:
            pred_gdf = pred_gdf[pred_gdf["confidence"] >= _post_conf_threshold].copy()
            print(f"置信度过滤: {len(pred_gdf)} / {pre_conf_count} 个多边形保留"
                  f"（confidence>={_post_conf_threshold}）")

        pred_gdf.to_file(str(_predictions_metric_path), driver="GPKG")
        export_gdf = to_export_crs(
            pred_gdf, assumed_crs=METRIC_CRS, label="预测结果"
        )
        export_gdf.to_file(str(_predictions_path), driver="GeoJSON")
        if save_config:
            write_run_config(_config_path, _config, result_count=len(pred_gdf))
        print(f"\n[OK] metric predictions saved: {_predictions_metric_path}")
        print(f"[OK] QGIS export saved: {_predictions_path} ({EXPORT_CRS})")
        print(f"[OK] detection config saved: {_config_path}")
        print(f"    总计 {len(pred_gdf)} 个太阳能板检测多边形")
        return pred_gdf

    except ImportError:
        print("[INFO] geoai 包未安装, 尝试路径 B...")
    except Exception as e:
        print(f"[WARNING] geoai 路径 A 出错: {e}\n尝试路径 B...")

    # ── 路径 B：samgeo SAM 模型备选 ──────────────────────────────────
    try:
        from samgeo import SamGeo

        print("\n[路径 B] 使用 samgeo (SAM) + 文字提示 'solar panel' 进行检测...")
        sam = SamGeo(
            model_type="vit_h",
            automatic=False,
            device=device,
        )

        all_gdfs = []
        for idx, tif_path in enumerate(geo_tifs, 1):
            tile_name = tif_path.stem
            print(f"  [{idx}/{len(geo_tifs)}] 检测中: {tile_name}")

            try:
                mask_path = _masks_dir / f"{tile_name}_sam_mask.tif"
                vector_path = _vectors_dir / f"{tile_name}_sam_vectors.geojson"

                sam.set_image(str(tif_path))
                sam.text_predict(
                    text="solar panel",
                    output=str(mask_path),
                    box_threshold=0.24,
                    text_threshold=0.24,
                )
                sam.raster_to_vector(str(mask_path), str(vector_path))

                gdf_tile = gpd.read_file(str(vector_path))
                if len(gdf_tile) > 0:
                    gdf_tile["source_tile"] = tile_name
                    gdf_tile["confidence"] = 0.5  # SAM 不提供逐对象置信度
                    all_gdfs.append(gdf_tile)
                    print(f"    → 检测到 {len(gdf_tile)} 个候选多边形")
                else:
                    print(f"    → 未检测到太阳能板")
            except Exception as e:
                print(f"    [WARNING] 处理 {tile_name} 时出错: {e}")
                continue

        if not all_gdfs:
            print("[ERROR] samgeo 路径也未检测到任何太阳能板")
            sys.exit(1)

        pred_gdf = pd.concat(all_gdfs, ignore_index=True)
        pred_gdf = to_metric_crs(pred_gdf, assumed_crs=INPUT_CRS, label="检测结果")
        pred_gdf.to_file(str(_predictions_metric_path), driver="GPKG")
        export_gdf = to_export_crs(
            pred_gdf, assumed_crs=METRIC_CRS, label="预测结果"
        )
        export_gdf.to_file(str(_predictions_path), driver="GeoJSON")
        if save_config:
            write_run_config(_config_path, _config, result_count=len(pred_gdf))
        print(f"\n[OK] metric predictions saved: {_predictions_metric_path}")
        print(f"[OK] QGIS export saved: {_predictions_path} ({EXPORT_CRS})")
        print(f"[OK] detection config saved: {_config_path}")
        return pred_gdf

    except ImportError:
        print("[FATAL] geoai 和 samgeo 均未安装，无法执行检测")
        print("  请运行:  pip install geoai-py")
        print("  或:      pip install segment-geospatial")
        sys.exit(1)


# ════════════════════════════════════════════════════════════════════════
# 第二步：加载真值数据
# ════════════════════════════════════════════════════════════════════════
def load_ground_truth() -> gpd.GeoDataFrame:
    """加载真值多边形，并统一到米制计算 CRS。"""
    print("\n" + "=" * 60)
    print("加载真值数据 (Ground Truth)...")

    gt = None
    if GT_GPKG.exists():
        try:
            import pyogrio
            layers = pyogrio.list_layers(str(GT_GPKG))
            if len(layers) > 1:
                # 多图层时，读取多边形数量最多的图层
                best_layer, best_count = None, 0
                for layer_name, _ in layers:
                    gdf_tmp = gpd.read_file(str(GT_GPKG), layer=layer_name)
                    print(f"  图层 '{layer_name}': {len(gdf_tmp)} 个多边形")
                    if len(gdf_tmp) > best_count:
                        best_layer, best_count = layer_name, len(gdf_tmp)
                        gt = gdf_tmp
                print(f"  → 选择图层 '{best_layer}' ({best_count} 个多边形)")
            else:
                gt = gpd.read_file(str(GT_GPKG))
                print(f"  已加载 GPKG: {GT_GPKG.name} ({len(gt)} 个多边形)")
        except Exception as e:
            print(f"  [WARNING] 读取 GPKG 失败: {e}")

    if gt is None and GT_GEOJSON.exists():
        try:
            gt = gpd.read_file(str(GT_GEOJSON))
            print(f"  已加载 GeoJSON: {GT_GEOJSON.name} ({len(gt)} 个多边形)")
        except Exception as e:
            print(f"  [ERROR] 读取 GeoJSON 也失败: {e}")
            sys.exit(1)

    if gt is None and COMBINED_ANNOTATION_GPKG.exists():
        try:
            combined = gpd.read_file(str(COMBINED_ANNOTATION_GPKG))
            name_cols = [c for c in combined.columns if c.lower() in {"name", "gridcell_id", "grid_id"}]
            gt = None
            if "gridcell_id" in combined.columns:
                mask = combined["gridcell_id"].astype(str) == GRID_ID
                if mask.any():
                    gt = combined.loc[mask].copy()
            if gt is None and "grid_id" in combined.columns:
                mask = combined["grid_id"].astype(str) == GRID_ID
                if mask.any():
                    gt = combined.loc[mask].copy()
            if gt is None:
                for col in ["Name", "name", "unique_id", "panel_id", "id"]:
                    if col in combined.columns:
                        mask = combined[col].astype(str).str.startswith(f"{GRID_ID}_", na=False)
                        if mask.any():
                            gt = combined.loc[mask].copy()
                            break
            if gt is not None:
                print(
                    f"  已从 {COMBINED_ANNOTATION_GPKG.name} 过滤 {GRID_ID} 标注 "
                    f"({len(gt)} 个多边形)"
                )
        except Exception as e:
            print(f"  [WARNING] 读取合并标注失败: {e}")

    if gt is None:
        print("[ERROR] 未找到任何真值文件 (g1238.gpkg 或 g1238.geojson)")
        sys.exit(1)

    # 统一投影
    gt = to_metric_crs(gt, assumed_crs=INPUT_CRS, label="真值数据")

    # 确保都是有效的几何体
    gt = gt[gt.geometry.notnull() & gt.is_valid].copy()
    print(f"  输入 CRS 按 {INPUT_CRS} 解释，计算统一到 {METRIC_CRS}")
    print(f"  有效多边形: {len(gt)} 个")
    return gt


def load_predictions() -> gpd.GeoDataFrame:
    """加载预测结果，并统一到米制计算 CRS。"""
    pred_path = None
    if PREDICTIONS_METRIC_PATH.exists():
        pred_path = PREDICTIONS_METRIC_PATH
    elif PREDICTIONS_PATH.exists():
        pred_path = PREDICTIONS_PATH
    else:
        print(f"[ERROR] 预测文件不存在: {PREDICTIONS_METRIC_PATH} / {PREDICTIONS_PATH}")
        sys.exit(1)

    pred = gpd.read_file(str(pred_path))
    print(f"  已加载预测结果: {len(pred)} 个多边形 ({pred_path.name})")

    assumed_crs = METRIC_CRS if pred_path == PREDICTIONS_METRIC_PATH else EXPORT_CRS
    pred = to_metric_crs(pred, assumed_crs=assumed_crs, label="预测结果")

    pred = pred[pred.geometry.notnull() & pred.is_valid].copy()
    return pred


# ════════════════════════════════════════════════════════════════════════
# 第三步：IoU 匹配与评估
# ════════════════════════════════════════════════════════════════════════
def compute_iou(geom_a, geom_b) -> float:
    """计算两个几何体的交并比 (IoU)"""
    try:
        if geom_a.is_empty or geom_b.is_empty:
            return 0.0
        intersection = geom_a.intersection(geom_b).area
        union = geom_a.area + geom_b.area - intersection
        if union == 0:
            return 0.0
        return intersection / union
    except Exception:
        return 0.0


def iou_matching(gt: gpd.GeoDataFrame,
                 pred: gpd.GeoDataFrame,
                 iou_threshold: float = 0.3,
                 merge_preds: bool = True,
                 return_match_details: bool = False,
                 ) -> dict:
    """
    基于空间 IoU 的匹配，支持两种模式：

    merge_preds=False（严格一对一模式）:
      对每个 GT 多边形，找到 IoU 最高的单个预测多边形匹配。

    merge_preds=True（多对一合并模式，默认）:
      对每个 GT 多边形，将所有与之相交的预测多边形 union 合并后再计算 IoU。
      适用于标注者用一个大多边形覆盖屋顶上多组面板，而检测器将其拆分为
      多个小多边形的情况。

    return_match_details=True 时额外返回 "match_details" 列表，每个元素为:
      {"gt_idx", "pred_indices", "iou", "intersection_area", "gt_area", "pred_area"}

    返回:
      {
        "tp": int, "fp": int, "fn": int,
        "precision": float, "recall": float, "f1": float,
        "matched_pred_indices": set,
        "matched_gt_indices": set,
        "iou_scores": list  # 每个 TP 对应的 IoU 值
      }
    """
    pred_sindex = pred.sindex  # 空间索引
    matched_pred = set()
    matched_gt = set()
    iou_scores = []
    match_details = [] if return_match_details else None

    if merge_preds:
        # ── 多对一合并模式 ────────────────────────────────────────────
        # 对每个 GT，找出所有与之相交的 pred，合并后计算 IoU
        gt_match_results = []  # (gt_idx, merged_iou, pred_indices_set, gt_geom, merged_pred_geom)

        for gt_idx, gt_row in gt.iterrows():
            gt_geom = gt_row.geometry
            # 空间索引粗筛
            candidate_idxs = list(pred_sindex.intersection(gt_geom.bounds))

            # 精筛：只保留真正相交的
            intersecting_idxs = []
            for pidx in candidate_idxs:
                pred_geom = pred.iloc[pidx].geometry
                try:
                    if gt_geom.intersects(pred_geom):
                        intersecting_idxs.append(pidx)
                except Exception:
                    continue

            if not intersecting_idxs:
                continue

            # 合并所有相交的预测多边形
            merged_pred_geom = unary_union(
                [pred.iloc[pidx].geometry for pidx in intersecting_idxs]
            )

            iou_val = compute_iou(gt_geom, merged_pred_geom)
            if iou_val >= iou_threshold:
                gt_match_results.append(
                    (gt_idx, iou_val, set(intersecting_idxs), gt_geom, merged_pred_geom)
                )

        # 按 IoU 降序处理（贪心），避免 pred 被重复分配
        gt_match_results.sort(key=lambda x: x[1], reverse=True)

        for gt_idx, iou_val, pidx_set, gt_geom, merged_pred_geom in gt_match_results:
            if gt_idx in matched_gt:
                continue
            # 检查是否有至少一个 pred 尚未被分配
            available = pidx_set - matched_pred
            if not available:
                continue
            matched_gt.add(gt_idx)
            matched_pred.update(pidx_set)  # 所有参与合并的 pred 都标记为已匹配
            iou_scores.append(iou_val)
            if return_match_details:
                try:
                    inter_area = gt_geom.intersection(merged_pred_geom).area
                except Exception:
                    inter_area = 0.0
                match_details.append({
                    "gt_idx": gt_idx,
                    "pred_indices": pidx_set,
                    "iou": iou_val,
                    "intersection_area": inter_area,
                    "gt_area": gt_geom.area,
                    "pred_area": merged_pred_geom.area,
                })

    else:
        # ── 严格一对一模式 ────────────────────────────────────────────
        candidate_pairs = []

        for gt_idx, gt_row in gt.iterrows():
            gt_geom = gt_row.geometry
            candidate_idxs = list(pred_sindex.intersection(gt_geom.bounds))
            for pred_idx in candidate_idxs:
                pred_geom = pred.iloc[pred_idx].geometry
                iou_val = compute_iou(gt_geom, pred_geom)
                if iou_val >= iou_threshold:
                    candidate_pairs.append((gt_idx, pred_idx, iou_val))

        candidate_pairs.sort(key=lambda x: x[2], reverse=True)

        for gt_idx, pred_idx, iou_val in candidate_pairs:
            if gt_idx not in matched_gt and pred_idx not in matched_pred:
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)
                iou_scores.append(iou_val)
                if return_match_details:
                    gt_geom = gt.loc[gt_idx].geometry
                    pred_geom = pred.iloc[pred_idx].geometry
                    try:
                        inter_area = gt_geom.intersection(pred_geom).area
                    except Exception:
                        inter_area = 0.0
                    match_details.append({
                        "gt_idx": gt_idx,
                        "pred_indices": {pred_idx},
                        "iou": iou_val,
                        "intersection_area": inter_area,
                        "gt_area": gt_geom.area,
                        "pred_area": pred_geom.area,
                    })

    tp = len(matched_gt)
    fn = len(gt) - tp
    fp = len(pred) - len(matched_pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    result = {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "matched_pred_indices": matched_pred,
        "matched_gt_indices": matched_gt,
        "iou_scores": iou_scores,
    }
    if return_match_details:
        result["match_details"] = match_details
    return result


def evaluate_at_multiple_thresholds(gt: gpd.GeoDataFrame,
                                     pred: gpd.GeoDataFrame,
                                     merge_preds: bool = True,
                                     ) -> pd.DataFrame:
    """在多个 IoU 阈值下计算评估指标"""
    results = []
    for iou_thr in IOU_THRESHOLDS:
        metrics = iou_matching(gt, pred, iou_threshold=iou_thr,
                               merge_preds=merge_preds)
        results.append({
            "IoU_Threshold": iou_thr,
            "TP": metrics["tp"],
            "FP": metrics["fp"],
            "FN": metrics["fn"],
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "F1": round(metrics["f1"], 4),
        })
    return pd.DataFrame(results)


def classify_predictions(gt: gpd.GeoDataFrame,
                          pred: gpd.GeoDataFrame,
                          iou_threshold: float = 0.3
                          ) -> gpd.GeoDataFrame:
    """
    为每个预测多边形标记 TP/FP 分类，以便后续置信度分析。
    """
    pred = pred.copy()
    metrics = iou_matching(gt, pred, iou_threshold=iou_threshold)
    pred["is_tp"] = False
    for pred_idx in metrics["matched_pred_indices"]:
        pred.at[pred.index[pred_idx], "is_tp"] = True
    return pred


# ════════════════════════════════════════════════════════════════════════
# 第四步：逐 Tile 统计
# ════════════════════════════════════════════════════════════════════════
def evaluate_per_tile(gt: gpd.GeoDataFrame,
                      pred: gpd.GeoDataFrame
                      ) -> pd.DataFrame:
    """
    将评估结果按 tile 分解统计。
    使用 tiles/ 的网格参数重建每个 tile 的边界框。
    """
    import rasterio

    geo_tifs = sorted(TILES_DIR.glob(f"{GRID_ID}_*_*_geo.tif"))
    if not geo_tifs:
        geo_tifs = sorted([
            f for f in TILES_DIR.glob(f"{GRID_ID}_*_*.tif")
            if "_geo" not in f.stem and "mosaic" not in f.stem
        ])

    rows = []
    for tif_path in geo_tifs:
        tile_name = tif_path.stem
        try:
            with rasterio.open(str(tif_path)) as src:
                tile_bounds = src.bounds
                tile_crs    = src.crs

            # 创建 tile 边界多边形
            tile_box = gpd.GeoDataFrame(
                geometry=[box(tile_bounds.left, tile_bounds.bottom,
                              tile_bounds.right, tile_bounds.top)],
                crs=tile_crs,
            ).to_crs(METRIC_CRS).geometry[0]

            # 筛选落入该 tile 的 GT 和 Pred
            gt_in_tile   = gt[gt.geometry.intersects(tile_box)]
            pred_in_tile = pred[pred.geometry.intersects(tile_box)]

            if len(gt_in_tile) == 0 and len(pred_in_tile) == 0:
                rows.append({
                    "tile": tile_name,
                    "gt_count": 0, "pred_count": 0,
                    "TP": 0, "FP": 0, "FN": 0,
                    "precision": 0.0, "recall": 0.0, "f1": 0.0,
                })
                continue

            gt_in_tile   = gt_in_tile.reset_index(drop=True)
            pred_in_tile = pred_in_tile.reset_index(drop=True)

            if len(gt_in_tile) > 0 and len(pred_in_tile) > 0:
                m = iou_matching(gt_in_tile, pred_in_tile, iou_threshold=DEFAULT_IOU)
            elif len(pred_in_tile) > 0:
                m = {"tp": 0, "fp": len(pred_in_tile), "fn": 0,
                     "precision": 0.0, "recall": 0.0, "f1": 0.0}
            else:
                m = {"tp": 0, "fp": 0, "fn": len(gt_in_tile),
                     "precision": 0.0, "recall": 0.0, "f1": 0.0}

            rows.append({
                "tile": tile_name,
                "gt_count": len(gt_in_tile),
                "pred_count": len(pred_in_tile),
                "TP": m["tp"], "FP": m["fp"], "FN": m["fn"],
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1": round(m["f1"], 4),
            })
        except Exception as e:
            print(f"  [WARNING] 处理 {tile_name} 逐 tile 统计时出错: {e}")
            rows.append({"tile": tile_name, "error": str(e)})

    df = pd.DataFrame(rows)
    df.to_csv(str(EVALUATION_CSV_PATH), index=False, encoding="utf-8-sig")
    print(f"\n[OK] per-tile evaluation saved: {EVALUATION_CSV_PATH}")
    return df


def evaluate_by_size(gt: gpd.GeoDataFrame,
                     pred: gpd.GeoDataFrame,
                     bins=None,
                     labels=None) -> pd.DataFrame:
    """
    按 GT 面积分层评估，便于重点观察大面积面板在不同 IoU 阈值下的召回变化。
    """
    if bins is None:
        bins = [0, 5, 20, 50, 100, float("inf")]
    if labels is None:
        labels = ["<5m2", "5-20m2", "20-50m2", "50-100m2", ">100m2"]

    gt_metric = gt.to_crs(METRIC_CRS).copy()
    gt_metric["area_m2"] = gt_metric.geometry.area
    gt_metric["size_class"] = pd.cut(
        gt_metric["area_m2"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    rows = []
    for iou_thr in IOU_THRESHOLDS:
        metrics = iou_matching(gt_metric, pred, iou_threshold=iou_thr)
        matched = gt_metric.index.isin(list(metrics["matched_gt_indices"]))
        gt_metric["matched"] = matched

        for size_class, subset in gt_metric.groupby("size_class", observed=False):
            if len(subset) == 0:
                continue
            matched_count = int(subset["matched"].sum())
            gt_count = int(len(subset))
            fn_count = gt_count - matched_count
            recall = matched_count / gt_count if gt_count > 0 else 0.0
            rows.append({
                "IoU_Threshold": iou_thr,
                "size_class": str(size_class),
                "gt_count": gt_count,
                "matched_gt": matched_count,
                "fn_count": fn_count,
                "recall": round(recall, 4),
                "mean_area_m2": round(float(subset["area_m2"].mean()), 2),
                "median_area_m2": round(float(subset["area_m2"].median()), 2),
            })

    df = pd.DataFrame(rows)
    df.to_csv(str(SIZE_STRATIFIED_CSV_PATH), index=False, encoding="utf-8-sig")
    print(f"\n[OK] size-stratified evaluation saved: {SIZE_STRATIFIED_CSV_PATH}")
    return df


# ════════════════════════════════════════════════════════════════════════
# 第五步：可视化
# ════════════════════════════════════════════════════════════════════════
def set_plot_style():
    """设置统一的图表风格"""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 11,
    })


def plot_confidence_histogram(pred_classified: gpd.GeoDataFrame):
    """
    图1: 置信度直方图，TP/FP 分色
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    tp_conf = pred_classified.loc[pred_classified["is_tp"], "confidence"]
    fp_conf = pred_classified.loc[~pred_classified["is_tp"], "confidence"]

    bins = np.linspace(0, 1, 25)

    ax.hist(tp_conf, bins=bins, alpha=0.7, color="#2ecc71", edgecolor="white",
            label=f"TP ({len(tp_conf)})", zorder=3)
    ax.hist(fp_conf, bins=bins, alpha=0.7, color="#e74c3c", edgecolor="white",
            label=f"FP ({len(fp_conf)})", zorder=2)

    ax.set_xlabel("Confidence Score", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("Confidence Distribution: TP vs FP", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_xlim(0, 1)

    fig.tight_layout()
    fig.savefig(str(CONFIDENCE_HIST_PATH))
    plt.close(fig)
    print(f"  [OK] saved: {CONFIDENCE_HIST_PATH.name}")


def plot_precision_recall_curve(gt: gpd.GeoDataFrame,
                                 pred: gpd.GeoDataFrame):
    """
    图2: 以置信度阈值为横轴的 Precision-Recall 曲线
    """
    set_plot_style()

    if "confidence" not in pred.columns:
        print("  [SKIP] 无 confidence 字段，跳过 PR 曲线")
        return

    conf_thresholds = np.arange(0.05, 1.0, 0.05)
    precisions = []
    recalls = []

    for conf_thr in conf_thresholds:
        pred_filtered = pred[pred["confidence"] >= conf_thr].reset_index(drop=True)
        if len(pred_filtered) == 0:
            precisions.append(1.0)  # 无预测 → 精度为 1（无误检）
            recalls.append(0.0)    # 无预测 → 召回为 0
            continue
        m = iou_matching(gt, pred_filtered, iou_threshold=DEFAULT_IOU)
        precisions.append(m["precision"])
        recalls.append(m["recall"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(conf_thresholds, precisions, "o-", color="#3498db",
            linewidth=2, markersize=5, label="Precision")
    ax.plot(conf_thresholds, recalls, "s-", color="#e67e22",
            linewidth=2, markersize=5, label="Recall")

    ax.set_xlabel("Confidence Threshold", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title(f"Precision & Recall vs Confidence Threshold (IoU={DEFAULT_IOU})",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(str(PR_CURVE_PATH))
    plt.close(fig)
    print(f"  [OK] saved: {PR_CURVE_PATH.name}")


def plot_iou_threshold_metrics(metrics_df: pd.DataFrame):
    """
    图3: IoU 阈值 vs Precision / Recall / F1
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(metrics_df["IoU_Threshold"], metrics_df["Precision"],
            "o-", color="#2ecc71", linewidth=2.5, markersize=8, label="Precision")
    ax.plot(metrics_df["IoU_Threshold"], metrics_df["Recall"],
            "s-", color="#3498db", linewidth=2.5, markersize=8, label="Recall")
    ax.plot(metrics_df["IoU_Threshold"], metrics_df["F1"],
            "D-", color="#9b59b6", linewidth=2.5, markersize=8, label="F1 Score")

    ax.set_xlabel("IoU Threshold", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Detection Metrics at Different IoU Thresholds",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_xlim(0.05, 0.75)
    ax.set_ylim(-0.05, 1.05)

    # 添加数据标注
    for _, row in metrics_df.iterrows():
        ax.annotate(f"{row['F1']:.2f}",
                     xy=(row["IoU_Threshold"], row["F1"]),
                     textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=9, color="#9b59b6")

    fig.tight_layout()
    fig.savefig(str(IOU_METRICS_PATH))
    plt.close(fig)
    print(f"  [OK] saved: {IOU_METRICS_PATH.name}")


# ════════════════════════════════════════════════════════════════════════
# Installation-level evaluation profile (V1.2)
# ════════════════════════════════════════════════════════════════════════

def evaluate_presence(matching_result: dict,
                      grid_id: str,
                      output_dir: Path) -> pd.DataFrame:
    """Installation-level presence metrics (merge, IoU>=0.1).

    Writes presence_metrics.csv with one row.
    """
    row = {
        "grid_id": grid_id,
        "gt_count": matching_result["tp"] + matching_result["fn"],
        "pred_count": matching_result["tp"] + matching_result["fp"],
        "tp": matching_result["tp"],
        "fp": matching_result["fp"],
        "fn": matching_result["fn"],
        "precision": matching_result["precision"],
        "recall": matching_result["recall"],
        "f1": matching_result["f1"],
    }
    df = pd.DataFrame([row])
    csv_path = output_dir / "presence_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[INSTALL] Presence metrics → {csv_path}")
    print(f"  P={row['precision']:.4f}  R={row['recall']:.4f}  F1={row['f1']:.4f}")
    return df


def evaluate_footprint(matching_result: dict,
                       output_dir: Path) -> pd.DataFrame:
    """Footprint quality metrics: IoU and Dice distributions for matched pairs.

    Writes footprint_metrics.csv with summary statistics.
    """
    details = matching_result.get("match_details", [])
    if not details:
        print("\n[INSTALL] Footprint metrics: no matches to evaluate")
        empty = pd.DataFrame([{
            "n_matches": 0, "mean_iou": 0, "median_iou": 0,
            "p25_iou": 0, "p75_iou": 0,
            "iou_ge_0.3_rate": 0, "iou_ge_0.5_rate": 0,
            "mean_dice": 0, "median_dice": 0,
        }])
        csv_path = output_dir / "footprint_metrics.csv"
        empty.to_csv(csv_path, index=False)
        return empty

    ious = []
    dices = []
    for d in details:
        ious.append(d["iou"])
        denom = d["gt_area"] + d["pred_area"]
        dice = 2 * d["intersection_area"] / denom if denom > 0 else 0.0
        dices.append(dice)

    ious_arr = np.array(ious)
    dices_arr = np.array(dices)

    row = {
        "n_matches": len(ious),
        "mean_iou": float(ious_arr.mean()),
        "median_iou": float(np.median(ious_arr)),
        "p25_iou": float(np.percentile(ious_arr, 25)),
        "p75_iou": float(np.percentile(ious_arr, 75)),
        "iou_ge_0.3_rate": float((ious_arr >= 0.3).mean()),
        "iou_ge_0.5_rate": float((ious_arr >= 0.5).mean()),
        "mean_dice": float(dices_arr.mean()),
        "median_dice": float(np.median(dices_arr)),
    }
    df = pd.DataFrame([row])
    csv_path = output_dir / "footprint_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[INSTALL] Footprint metrics → {csv_path}")
    print(f"  IoU: mean={row['mean_iou']:.3f} median={row['median_iou']:.3f}"
          f"  @0.3={row['iou_ge_0.3_rate']:.1%} @0.5={row['iou_ge_0.5_rate']:.1%}")
    print(f"  Dice: mean={row['mean_dice']:.3f} median={row['median_dice']:.3f}")
    return df


def evaluate_area_error(matching_result: dict,
                        gt: gpd.GeoDataFrame,
                        output_dir: Path) -> pd.DataFrame:
    """Per-match area error bucketed by GT area size class.

    Writes area_error_metrics.csv.
    """
    details = matching_result.get("match_details", [])
    if not details:
        print("\n[INSTALL] Area error metrics: no matches to evaluate")
        csv_path = output_dir / "area_error_metrics.csv"
        pd.DataFrame().to_csv(csv_path, index=False)
        return pd.DataFrame()

    records = []
    for d in details:
        gt_area = d["gt_area"]
        pred_area = d["pred_area"]
        abs_err = pred_area - gt_area
        rel_err = abs_err / gt_area if gt_area > 0 else 0.0
        records.append({
            "gt_area_m2": gt_area,
            "pred_area_m2": pred_area,
            "abs_error_m2": abs_err,
            "rel_error": rel_err,
        })

    match_df = pd.DataFrame(records)

    bins = [0, 5, 20, 50, 100, float("inf")]
    labels = ["<5m2", "5-20m2", "20-50m2", "50-100m2", ">100m2"]
    match_df["size_class"] = pd.cut(
        match_df["gt_area_m2"], bins=bins, labels=labels, right=False
    )

    rows = []
    for sc in labels:
        subset = match_df[match_df["size_class"] == sc]
        if len(subset) == 0:
            continue
        rows.append({
            "size_class": sc,
            "n_matches": len(subset),
            "mean_abs_error_m2": float(subset["abs_error_m2"].mean()),
            "median_abs_error_m2": float(subset["abs_error_m2"].median()),
            "mean_rel_error": float(subset["rel_error"].mean()),
            "median_rel_error": float(subset["rel_error"].median()),
        })

    result_df = pd.DataFrame(rows)
    csv_path = output_dir / "area_error_metrics.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"\n[INSTALL] Area error metrics → {csv_path}")
    for _, r in result_df.iterrows():
        print(f"  {r['size_class']:>10s}: n={int(r['n_matches']):3d}"
              f"  abs_err={r['mean_abs_error_m2']:+.1f}m²"
              f"  rel_err={r['mean_rel_error']:+.1%}")
    return result_df


# ════════════════════════════════════════════════════════════════════════
# 误检/漏检分类分析
# ════════════════════════════════════════════════════════════════════════
ERROR_ANALYSIS_PATH = OUTPUT_DIR / "error_analysis.csv"
FN_ANALYSIS_PATH    = OUTPUT_DIR / "fn_analysis.csv"


def analyze_errors(gt: gpd.GeoDataFrame,
                   pred: gpd.GeoDataFrame,
                   pred_classified: gpd.GeoDataFrame) -> None:
    """
    对 FP 和 FN 进行分类分析，输出 CSV 和控制台汇总。
    FP 分类基于 RGB 均值和几何属性；FN 按面积分档。
    """
    print("\n" + "=" * 60)
    print("误检/漏检分类分析 (Error Analysis)...")

    # ── FP 分类 ─────────────────────────────────────────────────────
    fp = pred_classified[~pred_classified["is_tp"]].copy()

    fp["error_type"] = "other"

    # 1) 极暗 → shadow
    is_shadow = (fp["mean_r"] < 50) & (fp["mean_g"] < 50) & (fp["mean_b"] < 50)
    fp.loc[is_shadow, "error_type"] = "shadow"

    # 2) 偏暗 → dark_object
    is_dark = ((fp["mean_r"] < 70) & (fp["mean_g"] < 70) & (fp["mean_b"] < 70)
               & (fp["error_type"] == "other"))
    fp.loc[is_dark, "error_type"] = "dark_object"

    # 3) 蓝色偏高 → blue_object（泳池/蓝车/蓝屋顶）
    has_blue = (fp["mean_b"] > fp["mean_r"] * 1.3) & (fp["error_type"] == "other")
    fp.loc[has_blue, "error_type"] = "blue_object"

    # 4) 面积过小 → small_fragment
    is_small = (fp["area_m2"] < 3) & (fp["error_type"] == "other")
    fp.loc[is_small, "error_type"] = "small_fragment"

    # 5) 过于细长 → elongated
    is_elongated = (fp["elongation"] > 4) & (fp["error_type"] == "other")
    fp.loc[is_elongated, "error_type"] = "elongated"

    # 保存 FP 分析
    fp_out_cols = ["source_tile", "error_type", "mean_r", "mean_g", "mean_b",
                   "area_m2", "elongation", "solidity"]
    fp_out = fp[[c for c in fp_out_cols if c in fp.columns]].copy()
    fp_out.to_csv(str(ERROR_ANALYSIS_PATH), index=False, encoding="utf-8-sig")

    print(f"\n  FP 分类统计 ({len(fp)} 个误检):")
    for etype, count in fp["error_type"].value_counts().items():
        pct = count / len(fp) * 100
        print(f"    {etype:20s}: {count:3d} ({pct:.1f}%)")

    # ── FN 分析 ─────────────────────────────────────────────────────
    metrics = iou_matching(gt, pred, iou_threshold=DEFAULT_IOU)
    fn_gt = gt.loc[~gt.index.isin(metrics["matched_gt_indices"])].copy()

    if len(fn_gt) > 0:
        # 计算面积（统一使用米制计算 CRS）
        fn_metric = fn_gt.to_crs(METRIC_CRS)
        fn_gt["area_m2"] = fn_metric.geometry.area

        fn_gt["size_class"] = pd.cut(
            fn_gt["area_m2"],
            bins=[0, 5, 20, 50, float("inf")],
            labels=["<5m2", "5-20m2", "20-50m2", ">50m2"],
        )

        fn_out = fn_gt[["area_m2", "size_class"]].copy()
        fn_out.to_csv(str(FN_ANALYSIS_PATH), index=False, encoding="utf-8-sig")

        print(f"\n  FN 面积分布 ({len(fn_gt)} 个漏检):")
        for sc, count in fn_gt["size_class"].value_counts().sort_index().items():
            print(f"    {sc:10s}: {count:3d}")
        print(f"    mean area: {fn_gt['area_m2'].mean():.1f} m2, "
              f"median: {fn_gt['area_m2'].median():.1f} m2")
    else:
        print("\n  FN: 0 个漏检")

    print(f"\n  [OK] saved: {ERROR_ANALYSIS_PATH.name}, {FN_ANALYSIS_PATH.name}")


# ════════════════════════════════════════════════════════════════════════
# 第六步：汇总报告
# ════════════════════════════════════════════════════════════════════════
def print_report(gt: gpd.GeoDataFrame,
                 pred: gpd.GeoDataFrame,
                 pred_classified: gpd.GeoDataFrame,
                 metrics_df: pd.DataFrame,
                 evaluation_profile: str = "installation"):
    """输出格式化评估报告"""

    default_metrics = iou_matching(gt, pred, iou_threshold=DEFAULT_IOU)

    tp_conf = pred_classified.loc[pred_classified["is_tp"], "confidence"]
    fp_conf = pred_classified.loc[~pred_classified["is_tp"], "confidence"]

    report = f"""
{'=' * 50}
  太阳能板检测评估报告
  Solar Panel Detection Evaluation Report
{'=' * 50}
计算 CRS            : {METRIC_CRS}
导出 CRS            : {EXPORT_CRS}
真值多边形总数     : {len(gt)}
预测多边形总数     : {len(pred)}
{'─' * 50}"""

    for _, row in metrics_df.iterrows():
        report += f"""
IoU 阈值 = {row['IoU_Threshold']}:
  TP              : {row['TP']}
  FP              : {row['FP']}
  FN              : {row['FN']}
  Precision       : {row['Precision']:.4f}
  Recall          : {row['Recall']:.4f}
  F1 Score        : {row['F1']:.4f}
{'─' * 50}"""

    tp_mean = f"{tp_conf.mean():.4f}" if len(tp_conf) > 0 else "N/A"
    fp_mean = f"{fp_conf.mean():.4f}" if len(fp_conf) > 0 else "N/A"
    tp_std  = f"{tp_conf.std():.4f}" if len(tp_conf) > 1 else "N/A"
    fp_std  = f"{fp_conf.std():.4f}" if len(fp_conf) > 1 else "N/A"

    report += f"""
confidence stats (IoU={DEFAULT_IOU}):
  mean confidence (TP) : {tp_mean}
  mean confidence (FP) : {fp_mean}
  std confidence  (TP) : {tp_std}
  std confidence  (FP) : {fp_std}
{'=' * 50}

输出文件：
  - {PREDICTIONS_METRIC_PATH}
  - {PREDICTIONS_PATH}
  - {CONFIDENCE_HIST_PATH}
  - {PR_CURVE_PATH}
  - {IOU_METRICS_PATH}
  - {EVALUATION_CSV_PATH}
"""

    if evaluation_profile == "installation":
        # Append installation-level metrics summary
        presence_csv = OUTPUT_DIR / "presence_metrics.csv"
        footprint_csv = OUTPUT_DIR / "footprint_metrics.csv"
        area_csv = OUTPUT_DIR / "area_error_metrics.csv"

        report += f"""
{'─' * 50}
Installation Profile (V1.3):
  evaluation_profile : installation
  label_definition   : installation_footprint
  note               : V1.3 evaluates reviewed predictions against installation-level GT"""

        if presence_csv.exists():
            pres_df = pd.read_csv(presence_csv)
            if len(pres_df) > 0:
                r = pres_df.iloc[0]
                report += f"""
  ── Presence (merge, IoU>=0.1) ──
    Precision       : {r['precision']:.4f}
    Recall          : {r['recall']:.4f}
    F1              : {r['f1']:.4f}"""

        if footprint_csv.exists():
            fp_df = pd.read_csv(footprint_csv)
            if len(fp_df) > 0 and fp_df.iloc[0].get("n_matches", 0) > 0:
                r = fp_df.iloc[0]
                report += f"""
  ── Footprint Quality ──
    mean IoU        : {r['mean_iou']:.4f}
    median IoU      : {r['median_iou']:.4f}
    IoU>=0.3 rate   : {r['iou_ge_0.3_rate']:.1%}
    IoU>=0.5 rate   : {r['iou_ge_0.5_rate']:.1%}
    mean Dice       : {r['mean_dice']:.4f}"""

        if area_csv.exists():
            area_df = pd.read_csv(area_csv)
            if len(area_df) > 0:
                report += f"""
  ── Area Error (by size class) ──"""
                for _, r in area_df.iterrows():
                    report += f"""
    {r['size_class']:>10s}: n={int(r['n_matches']):3d}  abs={r['mean_abs_error_m2']:+.1f}m²  rel={r['mean_rel_error']:+.1%}"""

        report += f"""
  ── Installation Metric Files ──
    - {presence_csv}
    - {footprint_csv}
    - {area_csv}
{'=' * 50}
"""

    print(report)

    # 同时保存为文本文件
    report_path = OUTPUT_DIR / "evaluation_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"[OK] report saved: {report_path}")


# ════════════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="太阳能板检测与评估流水线"
    )
    parser.add_argument(
        "--grid-id",
        default=DEFAULT_GRID_ID,
        help=f"目标 grid，默认 {DEFAULT_GRID_ID}",
    )
    parser.add_argument(
        "--output-subdir",
        default=None,
        help="结果输出到 results/<grid>/<subdir>/",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略已有 predictions/config.json，重新执行检测",
    )
    parser.add_argument(
        "--chip-size",
        type=int,
        default=None,
        help="检测 chip 边长像素，传 400 等价于 (400, 400)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=None,
        help="检测 chip overlap，例如 0.25",
    )
    parser.add_argument(
        "--min-object-area",
        type=float,
        default=None,
        help="后处理最小面积阈值（m²）",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="模型置信度阈值",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=None,
        help="掩膜阈值",
    )
    parser.add_argument(
        "--post-conf-threshold",
        type=float,
        default=None,
        help="后处理置信度阈值（基于 mask band2 回填值）",
    )
    parser.add_argument(
        "--max-elongation",
        type=float,
        default=None,
        help="后处理长宽比上限，超过此值的预测被过滤",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="自定义模型权重路径（.pth），默认使用 geoai 内置权重",
    )
    parser.add_argument(
        "--evaluation-profile",
        choices=["installation", "legacy_instance"],
        default="installation",
        help="评估模式: installation (V1.2三层指标) 或 legacy_instance (旧版兼容)",
    )
    parser.add_argument(
        "--data-scope",
        choices=["val_tiles", "full_grid", "cross_city"],
        default="full_grid",
        help="数据范围标签，记录到 config.json",
    )
    parser.add_argument(
        "--postproc-config",
        default=None,
        help="后处理参数 JSON 文件路径（由 calibration_sweep 生成），"
             "覆盖 post_conf_threshold / min_object_area / max_elongation",
    )
    return parser.parse_args()


def main(force: bool = False,
         grid_id: str = DEFAULT_GRID_ID,
         output_subdir: str | None = None,
         chip_size: int | None = None,
         overlap: float | None = None,
         min_object_area: float | None = None,
         confidence_threshold: float | None = None,
         mask_threshold: float | None = None,
         post_conf_threshold: float | None = None,
         max_elongation: float | None = None,
         model_path: str | None = None,
         evaluation_profile: str = "installation",  # V1.3: name preserved for compatibility; evaluates reviewed predictions vs installation-level GT
         data_scope: str = "full_grid",
         postproc_config: str | None = None):
    # 后处理配置文件覆盖：文件中的值作为 fallback，CLI 显式参数优先
    if postproc_config is not None:
        pp = load_postproc_config(postproc_config)
        if post_conf_threshold is None and "post_conf_threshold" in pp:
            post_conf_threshold = pp["post_conf_threshold"]
        if min_object_area is None and "min_object_area" in pp:
            min_object_area = pp["min_object_area"]
        if max_elongation is None and "max_elongation" in pp:
            max_elongation = pp["max_elongation"]

    set_grid_context(normalize_grid_id(grid_id), output_subdir=output_subdir)

    print("╔════════════════════════════════════════════════════════╗")
    print("║  太阳能板检测 & 评估流水线                             ║")
    print("║  Solar Panel Detection & Evaluation Pipeline          ║")
    print("╚════════════════════════════════════════════════════════╝\n")
    print(f"[GRID] {GRID_ID}")
    print(f"[OUT ] {OUTPUT_DIR}")
    print(f"[EVAL] profile={evaluation_profile}, scope={data_scope}")

    # ── Step 1: 检测 ──────────────────────────────────────────────────
    detect_chip_size = (chip_size, chip_size) if chip_size is not None else None
    detection_config = build_detection_config(
        chip_size=detect_chip_size,
        overlap=overlap,
        min_object_area=min_object_area,
        confidence_threshold=confidence_threshold,
        mask_threshold=mask_threshold,
        post_conf_threshold=post_conf_threshold,
        max_elongation=max_elongation,
        output_dir=OUTPUT_DIR,
        model_path=model_path,
    )
    if should_reuse_predictions(OUTPUT_DIR, detection_config, force=force):
        pred = load_predictions()
    else:
        pred = detect_solar_panels(
            chip_size=detect_chip_size,
            overlap=overlap,
            min_object_area=min_object_area,
            confidence_threshold=confidence_threshold,
            mask_threshold=mask_threshold,
            post_conf_threshold=post_conf_threshold,
            max_elongation=max_elongation,
            output_dir=str(OUTPUT_DIR),
            model_path=model_path,
        )
        pred = load_predictions()  # 重新加载以确保 CRS 统一

    # ── Step 2: 加载真值 ──────────────────────────────────────────────
    try:
        gt = load_ground_truth()
    except SystemExit:
        print("\n[INFO] 无真值数据，跳过评估 (detection-only mode)")
        print(f"[INFO] 检测结果已保存:")
        print(f"       GPKG: {OUTPUT_DIR / 'predictions_metric.gpkg'}")
        print(f"       GeoJSON: {OUTPUT_DIR / 'predictions.geojson'}")
        print(f"[INFO] 可在 QGIS 中加载 predictions_metric.gpkg 进行标注审查")
        return

    # ── Step 3: 多阈值评估（两种模式对比） ───────────────────────────
    print("\n" + "=" * 60)
    print("IoU 多阈值评估...")
    print("\n>> 模式 1: 多对一合并匹配 (merge_preds=True)")
    print("   适配标注风格: 一个 GT 多边形覆盖多组面板")
    metrics_df = evaluate_at_multiple_thresholds(gt, pred)
    print("\n" + metrics_df.to_string(index=False))

    print("\n>> 模式 2: 严格一对一匹配 (merge_preds=False)")
    print("   传统模式: 每个预测只能匹配一个 GT")
    metrics_df_strict = evaluate_at_multiple_thresholds(
        gt, pred, merge_preds=False
    )
    print("\n" + metrics_df_strict.to_string(index=False))

    # ── Step 4: 分类预测 (TP/FP)（使用合并模式） ─────────────────────
    pred_classified = classify_predictions(gt, pred, iou_threshold=DEFAULT_IOU)

    # ── Step 4b: 误检/漏检分类分析 ────────────────────────────────────
    try:
        analyze_errors(gt, pred, pred_classified)
    except (KeyError, Exception) as e:
        print(f"  [WARN] Error analysis skipped: {e}")

    # ── Step 5: 逐 Tile 评估 ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("逐 Tile 评估（合并匹配模式）...")
    tile_df = evaluate_per_tile(gt, pred)
    if "gt_count" in tile_df.columns:
        non_empty = tile_df[tile_df["gt_count"] > 0]
        if len(non_empty) > 0:
            print(f"\n有太阳能板的 tile 数: {len(non_empty)}")
            print(non_empty[["tile", "gt_count", "pred_count", "TP", "FP", "FN",
                             "precision", "recall", "f1"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("按面积分层评估（重点关注大面积面板）...")
    size_df = evaluate_by_size(gt, pred)
    large_df = size_df[size_df["size_class"].isin(["50-100m2", ">100m2"])]
    if len(large_df) > 0:
        print("\n大面积面板召回（按 IoU 阈值）:")
        print(large_df[["IoU_Threshold", "size_class", "gt_count", "matched_gt", "fn_count", "recall"]].to_string(index=False))

    # ── Step 5b: Installation-level 三层评估 (V1.3: reviewed predictions vs installation GT) ──
    if evaluation_profile == "installation":
        print("\n" + "=" * 60)
        print("Installation-level 三层评估 (presence / footprint / area)...")
        install_match = iou_matching(
            gt, pred, iou_threshold=0.1, merge_preds=True,
            return_match_details=True,
        )
        evaluate_presence(install_match, GRID_ID, OUTPUT_DIR)
        evaluate_footprint(install_match, OUTPUT_DIR)
        evaluate_area_error(install_match, gt, OUTPUT_DIR)

    # ── Step 6: 可视化（基于合并模式指标） ────────────────────────────
    print("\n" + "=" * 60)
    print("生成可视化图表...")
    plot_confidence_histogram(pred_classified)
    plot_precision_recall_curve(gt, pred)
    plot_iou_threshold_metrics(metrics_df)

    # ── Step 7: 最终报告 ──────────────────────────────────────────────
    print_report(gt, pred, pred_classified, metrics_df, evaluation_profile)

    # ── Step 8: 追加 evaluation_config 到 config.json ────────────────
    append_evaluation_config(CONFIG_PATH, {
        "evaluation_profile": evaluation_profile,
        "label_definition": "installation_footprint",
        "data_scope": data_scope,
        "annotation_tier_mix": "T1+T2",
    })

    print("\n[DONE] Pipeline finished!")


if __name__ == "__main__":
    args = parse_args()
    try:
        main(
            force=args.force,
            grid_id=args.grid_id,
            output_subdir=args.output_subdir,
            chip_size=args.chip_size,
            overlap=args.overlap,
            min_object_area=args.min_object_area,
            confidence_threshold=args.confidence_threshold,
            mask_threshold=args.mask_threshold,
            post_conf_threshold=args.post_conf_threshold,
            max_elongation=args.max_elongation,
            model_path=args.model_path,
            evaluation_profile=args.evaluation_profile,
            data_scope=args.data_scope,
            postproc_config=args.postproc_config,
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
