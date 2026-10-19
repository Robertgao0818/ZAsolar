# Phase-1 evaluation v3

本次按审查 M1/M2 修正验证划分与按网格置信区间，全部从既有 GPU 预测重计分。`val_sel300` 是训练用于挑选权重的 300 条记录，`val_holdout` 为余下 1,403 条并作为主验证结果；完整结果与日志对账见 [TEST_REPORT_v3.md](TEST_REPORT_v3.md)。两栏分开才能避免把选模分数误当成留出表现；忽略区分会混入选模偏差。

可搬运交付与三个架构的精确命令见 [pod_bundle_v3/README_v3.md](pod_bundle_v3/README_v3.md)，完整静态文件校验为 `pod_bundle_v3/MD5SUMS`。包内包含未改动的模型/标签源码、初始化权重、固定 zaeval 评分器和新版本评分入口；不包含影像、像素掩膜或训练检查点。

本目录实际成功运行路径为 `rescore_full_attempt2_v3/`；第一次因旧路径守卫误判输出目录名而在评分前停止，其空目录和日志保留。执行记录见 [RUN_v3.md](RUN_v3.md)、[LOG.md](LOG.md)。完整 CPU 重计分的命令是：

```bash
CUDA_VISIBLE_DEVICES='' python -B rescore_saved_v3.py --eval-dir ./test_run_full_gpu_v2 --out ./rescore_full_attempt2_v3
CUDA_VISIBLE_DEVICES='' python -B verify_saved_v3.py --eval-dir ./rescore_full_attempt2_v3 --out ./rescore_full_attempt2_v3/independent_pixel_iou_v3.json
CUDA_VISIBLE_DEVICES='' python -B verify_results_v3.py --eval-dir ./rescore_full_attempt2_v3 --out ./rescore_full_attempt2_v3/regression_v3.json
```

这些路径是已完成运行的复现记录，入口拒绝已有输出。再次运行必须换新的 `_v3` 输出目录，不能覆盖已有结果。`run_rescore_attempt2_v3.py` 是这次后台分阶段驱动：只跳过有成功状态文件的完整阶段，失败阶段不自动重试。每步连续两次失败就停止并保留证据。
