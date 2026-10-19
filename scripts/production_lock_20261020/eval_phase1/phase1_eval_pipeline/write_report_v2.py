#!/usr/bin/env python3
"""Write final reports only after successful full evaluation, render and independent checks."""
import csv,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'test_run_full_gpu_v2'
def main():
    status=json.loads((ROOT/'full_gpu_driver_status_v2.json').read_text())
    assert len(status['stages'])==3 and all(r['returncode']==0 for r in status['stages'])
    d=json.loads((BASE/'metrics.json').read_text());ind=json.loads((BASE/'independent_pixel_iou_both_v2.json').read_text())
    render=json.loads((BASE/'overlays/render_manifest.json').read_text());entries=json.loads((BASE/'predictions.json').read_text())
    assert not d['partial'] and len(entries)==2335 and ind['matched'] and ind['checked_saved_masks']==2335
    assert len(render['tiles'])==632 and len(render['full_grid_overlays'])==5
    assert all(Path(x['path']).is_file() for x in render['full_grid_overlays'])
    monitors=[json.loads(x) for x in (ROOT/'gpu_monitor_v2.jsonl').read_text().splitlines()]
    samples=[]
    for r in monitors:
        for line in r['compute_apps'].splitlines():
            parts=[p.strip() for p in line.split(',')]
            if len(parts)==2 and parts[0]!='3331':
                try:samples.append((int(parts[0]),int(parts[1])))
                except ValueError:pass
    sampled=max((m for pid,m in samples),default=0)
    fmt=lambda v:'N/A' if v is None else f'{v:.8f}'
    lines=['# GPU 全量端到端测试报告 v2','',
      '2026-10-18，Pacific/Auckland。单个 GPU 作业完成全部合格 val 和五个 eval_orig10 网格；评分、渲染及独立复算均通过。GPU 是用于模型并行计算的显卡。本次结果证明流水线能在真实检查点与完整授权范围上运行；它不是三架构比较，忽略这一点会将旧检查点或样本范围差异误当成架构优劣。','',
      '## 检查点与源码','',
      f"- 检查点：`{d['checkpoint']}`；SHA256：`{d['checkpoint_sha256']}`。SHA256 是用于识别文件内容是否相同的摘要。CPU 比较所有权重键和尺寸通过，正式执行也严格加载通过，未使用 smoke 后备检查点。",
      f"- 实际训练源码：`{d['training_source']}`；SHA256：`{d['training_sha256']}`。用户指的本机 train2_pinned_v2f.py 是清单/缓存审核脚本，没有 build_model，未执行。声称提交 2b2fdea 本地无法核验；不能将它记作已验证来源。pod 若有真正提供 build_model 的同名模块，bundle 优先导入，记录实际摘要。忽略这一来源差异会损害模型实现的可追溯性。",
      f"- 评分器：zaeval-2，提交 `{d['scorer']['commit']}`；原样导入固定工作树。数据清单 MD5：`{d['manifest_md5']}`。MD5 也是文件完整性摘要，本任务用它核对已冻结的清单。",
      '- 未执行 load_bundle/prep/_validate_body：前者读取禁止的坐标缓存及 Li 清单，后两者写死 CUDA。原样导入模型构造、norm、object_counts，数据选择来自冻结 CSV。','',
      '## 范围与全量指标','',
      f"共 {len(entries):,} 条网格-瓦片记录、{sum(e['windows'] for e in entries):,} 个滑窗。val 为全部 1,703 条训练合格池，含 14 个网格；eval 为全部 632 条、5 个网格。原始 val 1,768 条中，65 条被训练资格规则排除；这里不是固定 300 张选点子集。同 split 共用影像保留各网格自己的掩膜，不解释为去重地理总体。忽略这些范围差异会造成错误性能比较。",
      '', 'PV 指光伏板；SWH 指太阳能热水器及泳池加热器。IoU 是交集除以并集；P 为预测对象中匹配正确的比例，R 为真值对象被找到的比例，F1 综合二者。对象采用一对一 IoU 匹配，不能直接等同于训练的 best_objf1 命中规则。','',
      '| split | 类别 | 计分瓦片 | 像素 IoU | 对象 P/R/F1 @0.3 | 对象 P/R/F1 @0.5 | 对象 P/R/F1 @0.7 | 边界 F 0.25m | 边界 F 0.5m |',
      '|---|---|---:|---:|---|---|---|---:|---:|']
    for sp in ('val','eval'):
        for name in ('PV','SWH'):
            c=d['splits'][sp][name]
            ob=[' / '.join(fmt(c['objects']['by_iou'][t][k]) for k in ('precision','recall','f1')) for t in ('0.3','0.5','0.7')]
            bf=[fmt(next(x for x in c['boundary']['by_tol'] if x['tol_m']==tol)['F']) for tol in (.25,.5)]
            lines.append('| '+' | '.join([sp,name,str(c['n_tiles']),fmt(c['pixel']['iou']),*ob,*bf])+' |')
    lines+=['','SWH 只在 swh_labelled=True 的网格计分，未标注不等于没有设备。255 像素不参与像素评分。边界 F 比较轮廓是否在给定米数容差内重合，缺少边缘邻居时沿用评分器 1 m 排除规则。完整面积组、原始匹配与边界排除计数见 `test_run_full_gpu_v2/metrics.json` 和 `metrics.md`；忽略标签资格或排除规则会误读 SWH 与边界得分。','',
      '小面板召回按真值真实面积分组，下表为对象 IoU=0.5；其余阈值完整保存在 metrics.md。','',
      '| split | 类别 | <4 m² 召回（命中/真值） | 4–10 m² | >=10 m² |','|---|---|---|---|---|']
    for sp in ('val','eval'):
        for name in ('PV','SWH'):
            bins=d['splits'][sp][name]['objects']['by_iou']['0.5']['bins']
            cells=[f"{fmt(v['recall'])} ({v['hits']}/{v['n']})" if v['n'] else 'N/A (0 GT)' for v in bins.values()]
            lines.append('| '+' | '.join([sp,name,*cells])+' |')
    lines+=['','## 耗时与峰值显存','',
      f"推断配置为 `{d['precision']}`、batch=1、640×640 滑窗、步长 480，Hann 权重融合。bfloat16 是保留较大数值范围的低精度格式；窗口概率与拼接为 float32。它与此前 CPU float32 的四瓦片 smoke 不是同一数值配置或样本，不能直接据差值判断模型变化。",
      '', '| 阶段 | 实测秒数 |','|---|---:|',
      *[f"| {r['stage']}（包含解释器启动） | {r['wall_runtime_s']:.3f} |" for r in status['stages']],
      f"| 驱动全部阶段总墙钟时间 | {status['wall_runtime_s']:.3f} |",
      f"| 模型推断循环（含瓦片读取与预测写盘） | {d['inference_runtime_s']:.3f} |",
      f"| 评估内部总计（含评分） | {d['runtime_s']:.3f} |",
      f"| 独立 PV/SWH 复算（额外） | {ind['runtime_s']:.3f} |",'',
      f"显卡：{d['gpu']['name']}。PyTorch 峰值已分配 {d['gpu']['peak_allocated_bytes']:,} bytes（{d['gpu']['peak_allocated_bytes']/1e9:.3f} GB）；峰值保留 {d['gpu']['peak_reserved_bytes']:,} bytes（{d['gpu']['peak_reserved_bytes']/1e9:.3f} GB）。nvidia-smi 每 15 秒抽样的计算进程显存峰值 {sampled:,} MiB（{sampled*2**20/1e9:.3f} GB），后者包含 CUDA 上下文，但属于抽样下界，并非连续瞬时峰值。显存分配器预设不超过 7.5 GB，峰值保留值通过 <8 GB 断言。若只看权重大小而忽略中间激活与上下文，可能低估运行资源。",'',
      '## 独立复算与空间渲染','',
      '独立脚本 recompute_pixel_iou_v2.py 仅导入 NumPy 与 rasterio，不导入本流水线或 zaeval。逐个读取全部 2,335 张保存的预测及原始 GT（人工真值），核对尺寸、坐标变换、坐标系、单波段 uint8 和 0/1/2 编码，重新累加有效像素的整数 TP/FP/FN。它验证保存掩膜与评分报告的一致性；不能单独证明模型推断等同于 CPU float32，也不能替代对象匹配实现的独立审计。','',
      '| split | 类别 | TP | FP | FN | 并集 | IoU 绝对误差 |','|---|---|---:|---:|---:|---:|---:|']
    for sp,classes in ind['splits'].items():
        for name,c in classes.items():lines.append(f"| {sp} | {name} | {c['tp']} | {c['fp']} | {c['fn']} | {c['union']} | {c['absolute_error']} |")
    lines+=['','所有整数与报告完全一致，IoU 绝对误差为 0（阈值 1e-12）。这证明读盘后的掩膜与报告采用同一像素结果；跳过此检查可能掩盖保存类别、路径或地理配准错误。证据为 `independent_pixel_iou.json` 和 `independent_pixel_iou_both_v2.json`。','',
      '已渲染全部 632 张 eval 叠图、5 张缩略图总览和 5 张完整地理网格叠图；完整格文件为 `test_run_full_gpu_v2/overlays/G####_full_grid.png`。整格概览分辨率 0.20 m，按原始地理坐标拼接，裁到真实倾斜轮廓；原始预测保持 0.05 m。PV 为青色、SWH 为橙色，真值轮廓为实线，预测半透明填充，255 区域变暗。渲染完成后另做视觉 QA，记录于 visual_qa_v2.md。','',
      '## 覆盖与可搬运包','',
      'coverage_eval_orig10.csv 确认五格实际所需瓦片数 127/127/126/126/126，均与 v2f manifest 相同，文件全部存在；无缺瓦片，所以缺失文件大小列表为空。逐瓦片文件大小与投影轮廓相交面积在 coverage_eval_orig10_tiles_v2.csv。28 个 bbox（轴对齐外接矩形）空位均在倾斜轮廓之外，与 lead 的检查一致。若按 bbox 填满而不检查真实轮廓，会产生虚假的缺图结论。','',
      'pod_bundle/ 包含固定脚本与 requirements_pinned.txt、原样训练/标签辅助源码、MiT/MSCAN 初始化权重、val/eval 多边形与元数据、git archive 8044d14 的评分器压缩包和展开源码。推荐入口与精确三架构命令在 pod_bundle/README_v2.md；两种数据路径模式分别用绝对路径和搬迁后四对实际文件通过预检，且完整绝对路径 2,335 条记录通过。最终校验见 MD5SUMS_v2 与 RELEASE_NOTES_v2.md。三架构均做 CPU 构造/小前向，只有本次 mit_b2 实际 checkpoint 完成 GPU 全量评估；忽略这个范围会错误宣称三臂实测完成。没有上传。','',
      '## 失败与硬性约束','',
      '评估、渲染和独立复算实际阶段均首次通过，没有连续两次执行失败。前期本地来源查询发现 ds3m_ft 不在 Git 仓库中，所检查的 ZAsolar 仓库也没有 2b2fdea；这属于已记录的来源核验缺口，没有反复查询或执行未知源码。既有任务文件保持原样，仅 LOG.md 按授权追加；新版本与输出使用新名称。所有本次写入均在工作目录内，未修改标签、v2f 数据或外部源码，未读取 Li 测试数据与坐标缓存，未终止进程、未操作 PID 374974、未 git push 或上传。CUDA_VISIBLE_DEVICES=0 只给了这一个真实检查点 E2E 子进程，其余预检与渲染用 CPU。','']
    with (ROOT/'TEST_REPORT_v2.md').open('x') as f:f.write('\n'.join(lines))
    with (ROOT/'RUN_v2.md').open('x') as f:f.write('# Phase-1 follow-up run record\n\nStatus: complete after full GPU evaluation, independent checks, coverage and portable bundle verification.\n\nSource of truth: TEST_REPORT_v2.md, visual_qa_v2.md, full_gpu_driver_status_v2.json, coverage_eval_orig10.csv, pod_bundle/README_v2.md and pod_bundle/RELEASE_NOTES_v2.md.\n\nDecisions: only one GPU run; run_v2c/best.pt strictly compatible; actual training model source is unchanged train2_v2e.py because local claimed pinned file is an audit helper with no build_model. Claimed commit 2b2fdea remains unverified. Full eligible val=1703, eval=632; no Li/cache access; pod entry uses explicit bundled MSCAN init and separate PV/SWH recomputation. All original task files preserved; LOG.md appended by authorization. No uploads.\n')
    with (ROOT/'README_v2.md').open('x') as f:f.write('# Phase-1 evaluation follow-up\n\n完整 GPU 结果见 [TEST_REPORT_v2.md](TEST_REPORT_v2.md)；原始 CPU smoke 报告与代码保留。完整网格叠图在 test_run_full_gpu_v2/overlays/G####_full_grid.png。覆盖见 coverage_eval_orig10.csv。\n\nPod 最终命令、依赖与数据布局见 [pod_bundle/README_v2.md](pod_bundle/README_v2.md) 和 [RELEASE_NOTES_v2.md](pod_bundle/RELEASE_NOTES_v2.md)；优先用 run_phase1_eval_v2.sh（显式 MSCAN 初始化、双类独立复算）。原始 run_phase1_eval.sh 仍按用户要求保留。没有上传任何文件。\n')
    with (ROOT/'LOG.md').open('a') as f:f.write(f"- Full follow-up COMPLETE: 2335 rows / {sum(e['windows'] for e in entries)} windows, 632 eval tile overlays and five full-grid mosaics, PV/SWH independent integers and IoU exactly equal. GPU peak reserved {d['gpu']['peak_reserved_bytes']} bytes; driver wall {status['wall_runtime_s']:.3f} s. TEST_REPORT_v2.md and RUN_v2.md written. Bundle and visual QA records linked. No evaluation failures or uploads.\n")
    print('TEST_REPORT_v2.md, RUN_v2.md and README_v2.md created.')
if __name__=='__main__':main()
