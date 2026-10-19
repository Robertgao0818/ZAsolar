#!/usr/bin/env python3
"""Build the final Chinese report exclusively from completed, checked v3 artifacts."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'rescore_full_attempt2_v3'


def main():
    doc=json.loads((BASE/'metrics.json').read_text())
    independent=json.loads((BASE/'independent_pixel_iou_v3.json').read_text())
    regression=json.loads((BASE/'regression_v3.json').read_text())
    contracts=json.loads((ROOT/'contract_checks_v3.json').read_text())
    integrity=json.loads((ROOT/'originals_integrity_confirmed_v3.json').read_text())
    bundle=json.loads((ROOT/'bundle_verification_v3.json').read_text())
    assert independent['matched'] and regression['passed'] and contracts['passed'] and integrity['matched'] and bundle['passed']
    stages={name:json.loads((ROOT/f'{name}_attempt2_status_v3.json').read_text()) for name in ('rescore','independent','regression')}
    assert all(v['returncode']==0 for v in stages.values())
    rec=doc['training_log_reconciliation']
    fmt=lambda v:'N/A' if v is None else f'{v:.8f}'
    def estimate(scope,name,metric):
        b=doc['splits'][scope][name]['bootstrap_95ci'][metric]
        return f"{fmt(b['estimate'])} [{fmt(b['low'])}, {fmt(b['high'])}]"
    table=['| 类别与指标 | val_sel300（选模诊断） | val_holdout（主验证） |','|---|---:|---:|']
    for name in ('PV','SWH'):
        a,b=(doc['splits'][s][name] for s in ('val_sel300','val_holdout'))
        table.append(f"| {name} 计分瓦片 / 网格 | {a['n_tiles']} / {len(a['grids'])} | {b['n_tiles']} / {len(b['grids'])} |")
        for metric,label in [('pixel_iou','像素 IoU'),('object_f1_0.5','对象 F1@0.5')]:
            table.append(f'| {name} {label}，95% CI | {estimate("val_sel300",name,metric)} | {estimate("val_holdout",name,metric)} |')
    evtable=['| 类别 | 计分瓦片 / 网格 | 像素 IoU，95% CI | 对象 F1@0.5，95% CI |','|---|---:|---:|---:|']
    for name,c in doc['splits']['eval'].items():
        evtable.append(f'| {name} | {c["n_tiles"]} / {len(c["grids"])} | {estimate("eval",name,"pixel_iou")} | {estimate("eval",name,"object_f1_0.5")} |')
    logtable=['| 类别 | 旧日志最佳步像素 IoU | v2f val_sel300 保存预测 IoU | 绝对差 |','|---|---:|---:|---:|']
    for name in ('PV','SWH'):
        r=rec[name]
        logtable.append(f'| {name} | {fmt(r["logged_iou"])} | {fmt(r["saved_mask_iou"])} | {fmt(r["absolute_difference"])} |')
    counts=['| scope / 类别 | TP | FP | FN | 有效像素 |','|---|---:|---:|---:|---:|']
    for scope,classes in doc['splits'].items():
        for name,c in classes.items():
            v=c['pixel'];counts.append(f'| {scope} / {name} | {v["tp"]} | {v["fp"]} | {v["fn"]} | {v["n_valid"]} |')
    pv=doc['splits']['val_holdout']['PV']['bootstrap_95ci']
    sw=doc['splits']['val_holdout']['SWH']['bootstrap_95ci']
    report=f'''# Phase-1 评估修复测试报告 v3

2026-10-18，Pacific/Auckland。审查 M1 的验证划分和 M2 的网格不确定性计算已实现，并从上次 GPU 保存的全部 2,335 张预测重新计分；本次没有模型推断、没有使用 GPU（用于模型并行计算的显卡）。旧检查点日志可以定位最佳步，但不能证明 v2f 选模复现通过，原因与实测差值如下。这个区分决定了结果能支持哪些研究结论；忽略它会把运行正确误当成训练验证完全复现。

## M1：还原训练选模子集，以留出行为主结果

`val_sel300` 为训练固定选模用的 300 条网格—影像记录，`val_holdout` 为其余 1,403 条，后者是主验证指标。两组均覆盖 14 个验证网格；完整 eval 保持五个授权网格、632 条。PV 是光伏板，SWH 是太阳能热水器及泳池加热器。IoU 是交集除以并集；F1 综合查准率与查全率；`@0.5` 表示预测对象与真值对象的 IoU 至少 0.5 才能一对一匹配。

本次原样调用经过源码检查的 `hn_clean_v2f.load_M()`，它只读取 v2f/v2b 两张 CSV 清单，不读取坐标缓存。它先按 v2b 的 `(grid,image)` 键重排，再追加新键；随后按训练 `load_bundle()` 的公开资格规则保留 `valid>100000` 且不与 eval 共用影像的 val 行，最后 `sample(300, random_state=1)`。未执行会读取坐标缓存及 Li 清单的 `load_bundle()`，也未复制或修改训练函数。独立检查与原样 `load_M()` 的 300 个身份、原始行号和抽样次序逐一相等。元数据及摘要在 `metadata_v3/`；完整分组名单在 `{BASE.name}/scope_rows_v3.csv`。这样能防止先排序或搬迁路径改变选样；若忽略行序，固定随机种子仍可能抽出不同样本。

当前冻结清单的合格 val 相对顺序恰好已与训练顺序一致，直接从它抽样也重合 300/300；这只是本快照事实，代码没有依赖这个巧合。将池反序时仅重合 {contracts['reversed_order_selection_overlap']}/300，检验证明顺序确实影响选样。原始 1,768 条 val 中，18 条有效像素资格不足，47 条与 eval 共用影像被排除，合格池为 1,703 条。

{chr(10).join(table)}

主验证 PV 像素 IoU 为 {fmt(pv['pixel_iou']['estimate'])}、对象 F1@0.5 为 {fmt(pv['object_f1_0.5']['estimate'])}；SWH 对应为 {fmt(sw['pixel_iou']['estimate'])}、{fmt(sw['object_f1_0.5']['estimate'])}。它说明这个旧检查点对 PV 的辨识明显强于 SWH；忽略类别差异，会把光伏板表现误认为热水器也可靠。本次只有一个旧 MiT 检查点，不能据此做三架构排名。

`val_sel300` 的 SWH 对象计分只有 5 个可见真值、22 个预测、零个匹配，所以 F1 与网格重抽样区间均为零。这个退化区间只表示重抽已有网格无法产生原本没有的正确匹配，并不表示总体 F1 已被精确确定为零；忽略这一点会把样本中没有命中误当成没有统计不确定性。

两组之间有 9 张不同影像因相邻网格各有一行而同时出现，两侧各 9 行；仍严格保留你要求的 300/1,403 行划分。因此 `val_holdout` 是不在本轮 v2f 协议固定 300 条中的记录，并不是完全独立的地理留出集；忽略跨网格共图会高估独立性。这也意味着按网格区间依赖“网格近似独立”的假设，不能消除共享影像带来的相关性。

此外，实际复算权重来自旧 v2c，其原始选模池与本轮 v2f 池不同，不能保证这 1,403 行从未用于那个旧权重的实际选模。本次验证的是 v2f 协议分组与计分实现，旧权重的这组分数不应被视为已消除其历史选模偏差的正式性能；忽略这一点，仍可能过度解释留出结果。

对象在每个 scope 内重新合并、匹配，未采用全池减选模计数。切跨两组的对象在各自可见部分计分；像素计数可以相加回原全池，对象计数通常不能。完整对象阈值 0.3/0.5/0.7、面积组与边界计分仍保存在 `metrics.json` 中。边界仍沿用同 scope 邻瓦片 halo（接上的邻瓦片边缘像素），缺邻居时保留 1 m 排除规则；新分组边界分数不声称等同于旧全池边界分数。

## 与检查点训练日志的对账

权重仍为 `{doc['checkpoint']}`，SHA256 为 `{doc['checkpoint_sha256']}`。SHA256 是识别文件内容的摘要。其旁 `{rec['log_path']}` 有 12 次训练后验证，没有 `obj_f1` 字段。原样旧 `train2.py` 明确按 `v['iou']>best` 保存 `best.pt`，日志中第一个最高像素 IoU 的步骤是 **{rec['best_step']}**，对应 IoU **{fmt(rec['logged_validation']['iou'])}**。权重文件本身不含步数，此绑定来自旧文件名与保存规则，并非嵌入式步数证据。

实际模型/归一化源码为 `{doc['training_source']}`，SHA256 为 `{doc['training_sha256']}`，MD5 为 `961e2e8e1a3657d1f72aca27c968f83f`；固定评分器提交为 `{doc['scorer']['commit']}`，数据清单 MD5 为 `{doc['manifest_md5']}`。MD5 同样是文件内容摘要，用于核对已冻结来源；本次没有把无法在本机解析的声称提交号当作已验证来源。

{chr(10).join(logtable)}

对账状态为 **`{rec['status']}`**。旧 `best.pt` 使用 v2c 标签和旧验证池，并按像素 IoU 选模；本次是 v2f 标签/资格规则下的 `val_sel300`，没有同检查点的 v2f `best_objf1` 日志或配置。因此差值不能归因于低精度误差，也不能报告“复现通过”。忽略标签、样本池与选模准则变化，会把口径差异误判为流水线错误或改善。

新入口对正式 `best_objf1.pt` 自动找日志中第一个严格最佳的 `obj_f1` 步，排除不保存的 step 0；支持缺日志明确 N/A、来源/配置不匹配明确标记、差值超过诊断容差明确 mismatch。PV 绝对 IoU 容差设为 0.001，仅作运行诊断而非 bf16（保留较大数值范围的低精度浮点格式）误差上界。有 SWH 日志时也报告其差值。单元检查刻意让对象 F1 的最佳步不同于像素 IoU 最佳步，并设置并列最佳，已验证选择第一个正确步骤。没有为了获得对账通过而重推断。

## M2：按网格 bootstrap 的 95% 区间

Bootstrap 指反复有放回地重抽整组观测，以估计指标在样本构成变化下的波动。本次把完整网格作为一组：每次抽取与原范围同样数量的网格，汇总各网格 TP/FP/FN 后计算像素 IoU 和对象 F1@0.5，取 2.5%/97.5% 百分位。TP/FP/FN 分别是真阳性、假阳性、假阴性的整数计数。固定 20,000 次、随机种子 20261018；没有平均瓦片或网格的比例，避免给小网格过大权重。

PV 在两组 val 各抽 14 个网格、eval 抽 5 个。SWH 在两组 val 各抽 **8** 个标注合格网格，eval 抽 4 个；其中 G1979 是有标注资格但零 SWH 真值的网格，必须保留其假阳性贡献。审查中“7 个 val 网格”的表述不等于当前清单的标注资格数量。若跳过零目标网格，会把负样本识别错误漏掉。

分母为零的重复样本不擅自当零，记录 undefined 次数并仅从已定义样本取分位数；全部未定义则区间与点估计为 N/A。`bootstrap_95ci` 保存参数、网格名单、原始整数、已定义/未定义次数。固定种子让未来各架构同 scope 同类别可以用同一组网格抽样做配对比较；配对比较是把同一网格在两架构上的差值一起计算，其区间应单独生成，不能只凭两个区间重叠判断差异。

{chr(10).join(evtable)}

全部 eval 像素计数与对象阈值 0.3/0.5/0.7 的总计、逐网格 TP/FP/FN 均与 v2 相同。这说明 v3 没有改变原 eval 预测或评分口径，只补上区间；忽略此回归检查会让实现变更混入结果变化。区间条件于当前检查点和这些观测网格，五个 eval 网格时只是粗略不确定性估计，不涵盖训练种子、标注误差或整个城市的抽样设计。

## 独立核验与运行证据

`verify_saved_v3.py` 只导入 NumPy/rasterio（分别用于数组计算和地理栅格读写），不导入本流水线或评分器。它逐张读取全部 2,335 张保存预测及允许的真值掩膜，检查尺寸、地理变换、坐标系、单波段 uint8 和类别编码，独立累加六个 scope/类别组合的整数，并核对每个网格计数。全部相等、像素 IoU 差为零；合并 val 两栏也精确还原 v2 全池。忽略这些检查可能掩盖路径、掩膜或汇总错误。

{chr(10).join(counts)}

`verify_results_v3.py` 另从输出逐网格整数独立重算 12 组指标的 20,000 次抽样及分位数，全部精确相等；另核对未改动 eval 与 val 像素分区守恒。契约检查覆盖不等分母时必须用总计之比、固定种子重复性、全零及部分未定义样本，以及日志的正确选步。对象计算仍依赖固定 zaeval 实现，这些检查不等于重新独立审计全部对象匹配算法。

成功重计分耗时 {stages['rescore']['runtime_s']:.3f} 秒；独立保存掩膜核验 {stages['independent']['runtime_s']:.3f} 秒；汇总与 bootstrap 回归核验 {stages['regression']['runtime_s']:.3f} 秒（均含解释器启动）。运行在独立 tmux 会话；tmux 是让后台进程不依赖聊天会话持续存活的工具。完整日志、命令及退出码见 `rescore_attempt2_v3.log`、`independent_attempt2_v3.log`、`regression_attempt2_v3.log` 及对应状态 JSON。

## 可搬运包、失败记录与硬规则

`pod_bundle_v3/README_v3.md` 给出三个架构在绝对路径和相对路径模式下的精确命令，分别使用各自 `best_objf1.pt`/`log.jsonl`，输出到新的 `eval_v3/<arch>_s0/`。新入口提供按训练行序划分、逐网格区间和日志对账；权重配置含 script_md5 时还会在推断前断言源码一致。完整 MD5SUMS 覆盖 {bundle['files']} 个静态文件、{bundle['bytes']} 字节，全部核验通过；它不包含自身及运行时缓存。影像、掩膜和训练检查点需目标 pod 已有，未打入包或上传。

完整绝对路径预检核查 2,335 条；搬迁后四瓦片相对路径预检的 300 张抽样顺序摘要与绝对路径完全相同。真实旧 MiT 权重 CPU 严格加载通过，未做新前向；三个架构的命令入口用记录器核查，共 9 个调用，未运行模型。评分器归档和源码摘要也通过。三架构模型构造及权重逻辑没有改变，不能把命令核验声称为三个架构的新实测。

本次有三次**分开的首次检查失败**。契约检查第一次错误预期原始 val 顺序必须不同，修正后第二次通过；随后首次重计分在评分前因旧路径守卫误判 `test_run_full_gpu_v2/metrics.json` 而停止，修正为仅对明确授权的输出 JSON 直接取摘要后，使用新目录第二次通过；首次原文件完整性检查还发现宿主程序正在自动追加本会话的 `codex_task1c.log`，复查证明其原始内容前缀完全未变。每次失败后都有成功检查，没有连续两次失败后仍继续；无自动重试。空的 `rescore_full_v3/`、首次失败状态及完整性检查 JSON 均保留。{integrity['static_files_exactly_unchanged']} 个原始静态文件前后摘要相等，宿主会话日志只追加、原内容未覆盖；任务本身仅按授权追加 LOG.md。

所有新写入都在当前目录；标签、v2f 数据、外部源码与既有预测未修改。不调用 load_bundle/prep/_validate_body，不读取原始 Li 测试数据或坐标缓存，不杀进程、不操作 PID 374974、不上传。GPU 预测齐全，本次无需 nvidia-smi 或 GPU 推断；未来入口仍先检查 nvidia-smi，只用 GPU 0、batch=1、7.5 GB 分配预算、峰值保留显存 <8 GB。SegNeXt 的随机 NMF（内部随机初始化分解的一种特征处理）仍受批量/遍历顺序影响；即使使用 bf16，也不承诺与训练 batch=4 完全相同。

来源与状态文件：`contract_checks_v3.json`、`metadata_v3/selection_provenance_v3.json`、`{BASE.name}/metrics.json`、`{BASE.name}/independent_pixel_iou_v3.json`、`{BASE.name}/regression_v3.json`、`bundle_full_preflight_v3.json`、`bundle_relative_preflight_v3.json`、`bundle_cli_checks_v3.json`、`bundle_verification_v3.json`、`originals_integrity_confirmed_v3.json`。执行状态另见 RUN_v3.md；原 v1/v2 文档保留。
'''
    with (ROOT/'TEST_REPORT_v3.md').open('x') as f:f.write(report)
    with (ROOT/'RUN_v3.md').open('x') as f:f.write(f'''# Phase-1 v3 run record

Status: complete. M1 split and M2 grid-cluster CIs implemented and re-scored on saved masks only.

Source of truth: TEST_REPORT_v3.md; {BASE.name}/metrics.json; contract_checks_v3.json; bundle_verification_v3.json.

Decisions: exact training load_M identity/index/draw order; val_sel300=300, val_holdout=1403 (main), eval=632. Legacy run_v2c/best.pt log best pixel-IoU step={rec['best_step']}; reconciliation={rec['status']} because legacy v2c labels/pool and no obj_f1 log/config. No claimed v2f best_objf1 reproduction or historical selection independence of the legacy checkpoint's holdout score. 9 shared images cross selection/holdout, documented without altering requested partition. SWH labelled empty grids included (8 val / 4 eval).

Uncertainty: 20,000 grid bootstrap draws, seed 20261018, percentile 95%, ratios of summed integers. All 12 interval recalculations and all 2335 raster checks passed. Eval object/pixel counts unchanged; val pixel partition sums to v2 full pool.

Execution: three isolated initial check failures (selection-test expectation, saved-output path guard, live-transcript hash expectation); corrected checks passed with no consecutive pair. Distinct attempt paths/logs/status retained. No inference, no GPU, no process termination or PID 374974 access, no original Li test/cache reads, no external mutations or uploads. Original static artifacts hash-stable; host transcript original prefix preserved while its external logger appends; LOG.md appended by authorization. Clean pod_bundle_v3/ with all static files in MD5SUMS and exact per-architecture commands in README_v3.md.
''')
    print(json.dumps({'report':'TEST_REPORT_v3.md','run':'RUN_v3.md','status':'complete'}))


if __name__=='__main__':main()
