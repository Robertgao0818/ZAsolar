**判定：sound with changes（方法成立，但需要修改使用方式）。** 人工确认的误检可以作为难负样本（HN，即模型容易误认成光伏的真实背景）；目前的清洗也没有发现坐标错配。不过，它只排除了与**已有标签**冲突的采样点，尚未证明剩余区域没有漏标光伏。忽略这个区别，可能会把真光伏反复当作背景训练，尤其伤害小板的召回率——即真实光伏被找出的比例。

本次全程只读，仅做 CPU 检查，没有创建或修改文件，没有运行训练、模型推理或 Gemini 调用。下文“本次实查”指在内存中读取缓存、影像元数据和标签后计算；这些结果没有另存文件。

**1．来源有依据，但可信度应落实到对象及其周边，不能落实到全部背景像素。**

源头确实经过人工审阅。库存生成器读取 `v3c_targeted_hn_aerial_2025` 和旧版审阅文件，保留对象的来源路径、行号与审阅状态；训练集构建器只选择 `status == "delete"` 的对象。[inventory.py:3–13](/home/gao/data/ZAsolar/geid_temporal/cls_hardneg_inventory_v1_20261014/inventory.py:3)、[build_ct_seg_trainset_v2.py:326–348](/home/gao/projects/ZAsolar_segv1_wt/scripts/training/build_ct_seg_trainset_v2.py:326)

本次实查发现：

- 库存中的 793 个 `delete` 对象，均能在原始文件对应行找到相同删除状态；没有状态或行号不一致。
- 原始 145,998 个 HN 坐标全部落在 v2b 的类别 `2` 上。但这个旧类别混合了普通误检和已确认热水器，因此原始缓存并非纯粹的普通背景误检集。[train.py:23–33](/home/gao/capetown_eval/ds3m_ft/train.py:23)、[patch_ct_seg_trainset_v2c.py:3–13](/home/gao/projects/ZAsolar_segv1_wt/scripts/training/patch_ct_seg_trainset_v2c.py:3)
- 清洗后 118,143 个点全部位于原始 `delete` 多边形内，共涉及 **504 个不同对象、30 个网格**；全部坐标合法，492 张相关瓦片的影像与标签网格对齐。

这足以支持“有根据的负样本候选”，不足以支持“已经验证没有光伏”。人工删除一个检测对象，可能是因为它整体错误、混合了屋顶和小板，或者边界严重错误；它不自动证明整个多边形以及周围裁剪都是背景。

此外，汇总中的“8 个有可信手标网格”是**原始 HN 网格与手标来源的交集**。保留 HN 后，30 个网格中只有 **7 个**有该手标图层，另 **23 个**没有；后者承担约 **79.7% 的 HN 分支抽样机会**。`hand_label=True` 本身也只是判断该网格存在图层，没有检查该网格是否标注完整。[hn_clean.py:36–42](/home/gao/capetown_eval/ds3m_ft/hn_clean/hn_clean.py:36)

**2．删除规则方向正确；0.3 米只是暂定保护距离，而且当前实现并非严格圆形半径。**

清洗确实要求：

```text
当前位置是背景，并且不在已标光伏或可信手标的膨胀区域内。
```

可信手标来自指定的两组 `G*_SAM2_*` 文件，排除 Google Earth 标签；投影后还删除面积 ≥20,000 m² 的几何。[hn_clean.py:18–23](/home/gao/capetown_eval/ds3m_ft/hn_clean/hn_clean.py:18)、[hn_clean.py:34–43](/home/gao/capetown_eval/ds3m_ft/hn_clean/hn_clean.py:34)

**27,855 个删除点的总数核实无误，但解释需要修正。** 本次逐点读取 v2d2 标签得到：

- **27,809 个是类别 `2`，即太阳能热水／泳池加热设备。**
- **46 个是 `255`，即忽略区。**
- **0 个是已标光伏类别 `1`。**

这与汇总的 `not_bg=27855` 一致。因此，这次清洗主要完成了类别分流，不能当作“已清除大量漏标真光伏”的证据。[hn_clean_summary.json:5–10](/home/gao/capetown_eval/ds3m_ft/hn_clean/hn_clean_summary.json:5)

对三个具体规则的判断如下：

- **删除 `mask==255` 的 HN 采样中心是正确的。** 忽略区可能是光伏边界、无影像区域或不确定的热水器，不应视为确认背景。训练也同时从交叉熵损失和 Dice 损失中排除了 `255`；两者分别衡量逐像素分类错误和预测区域重合程度。[build_ct_seg_trainset_v1.py:46–64](/home/gao/projects/ZAsolar_segv1_wt/scripts/training/build_ct_seg_trainset_v1.py:46)、[build_ct_seg_trainset_v2.py:408–412](/home/gao/projects/ZAsolar_segv1_wt/scripts/training/build_ct_seg_trainset_v2.py:408)、[train2_v2d2.py:75–81](/home/gao/capetown_eval/ds3m_ft/train2_v2d2.py:75)
- **六次默认膨胀不是严格的 0.3 米圆形缓冲。** 它沿上下左右扩展，形成菱形距离范围；部分直线距离不足 0.3 米的对角位置可能落在范围外。代码没有显式指定圆形结构。[hn_clean.py:34–39](/home/gao/capetown_eval/ds3m_ft/hn_clean/hn_clean.py:34)、[SciPy 官方说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_dilation.html)
- **光伏距离是从标签内核算起。** 标签制作已将光伏边界内外设为忽略区，所以 `mm==1` 通常比原始光伏多边形小；再向外膨胀六像素，不能解释成“原始板边外另加 0.3 米”。[build_ct_seg_trainset_v1.py:58–63](/home/gao/projects/ZAsolar_segv1_wt/scripts/training/build_ct_seg_trainset_v1.py:58)、[hn_clean.py:34](/home/gao/capetown_eval/ds3m_ft/hn_clean/hn_clean.py:34)

不过，本次用直线距离独立复核后，保留点中仍没有点距光伏内核 ≤0.3 米；在有可信手标的保留点中，距原始手标最近约 1.77 米。**几何实现值得修正，但没有证据说明它已污染当前这批保留点。**

0.3 米可以暂留为探索参数，不能认定为最优。建议按原始光伏多边形计算真实距离，并比较 0.3、0.5、1 米会改变哪些对象。单纯扩大半径无法保护完全没有标签的光伏。

**3．实际训练强度：15% 裁剪来自 HN，没有 HN 专属损失加权。**

实际代码与开头注释略有出入：

- 非空采样池下，比例为 **70% 光伏、5% 热水器、15% HN、10% 随机有效像素**。[train2_v2d2.py:41–45](/home/gao/capetown_eval/ds3m_ft/train2_v2d2.py:41)
- HN 分支先等概率选瓦片，再等概率选该瓦片内缓存点。因此不是网格等权，也不是全部 HN 点等权；点少的瓦片会使每个点被更频繁地重复抽取。[train2_v2d2.py:44–48](/home/gao/capetown_eval/ds3m_ft/train2_v2d2.py:44)
- 默认裁剪 **640×640 像素，即 32×32 米**；围绕抽样点随机平移，边缘限制在瓦片内。README 中的 512 像素切片索引不决定这里的训练裁剪大小。[train2_v2d2.py:18](/home/gao/capetown_eval/ds3m_ft/train2_v2d2.py:18)、[train2_v2d2.py:46–50](/home/gao/capetown_eval/ds3m_ft/train2_v2d2.py:46)
- 类别权重默认是背景／光伏／热水器 **1／1／2**，总损失为交叉熵加 **0.5×光伏 Dice**；没有 HN 点或 HN 裁剪的额外权重。[train2_v2d2.py:20](/home/gao/capetown_eval/ds3m_ft/train2_v2d2.py:20)、[train2_v2d2.py:74–81](/home/gao/capetown_eval/ds3m_ft/train2_v2d2.py:74)

**风险来自整张裁剪的标签，不只是采样点。** HN 坐标只选择取图位置，训练仍使用整张裁剪原有的标签。本次按代码复现前 12,000 次取样，得到 1,749 张 HN 裁剪，其中 **426 张（24.4%）含已标光伏**。这些光伏会正常参与正类训练；但附近若有漏标小板，它们会作为背景受罚。删除中心坐标也不会修复这些背景标签，其他采样分支仍可能取到它们。[train2_v2d2.py:47–56](/home/gao/capetown_eval/ds3m_ft/train2_v2d2.py:47)、[train2_v2d2.py:78–80](/home/gao/capetown_eval/ds3m_ft/train2_v2d2.py:78)

因此，15% 本身不能判为过高，却也没有证据说明它最合适。小板还面临两个独立问题：正样本按光伏像素而非对象抽取，小对象获得的机会更少；边界忽略带会使它们失去更大比例的有效监督。[hn_clean.py:27–29](/home/gao/capetown_eval/ds3m_ft/hn_clean/hn_clean.py:27)、[train2_v2d2.py:42–47](/home/gao/capetown_eval/ds3m_ft/train2_v2d2.py:42)、[v2d2 README:9](/home/gao/data/ZAsolar/geid_temporal/ct_seg_trainset_v2d2_20261017/README.md:9)

**HN 不能直接解释 v2d1 相对 v2c 的退步。** 本次逐数组比较，v2c、v2d1、v2d2 的 409 个 HN 数组完全相同；v2c 与 v2d1 的采样比例、损失实现也相同。[train2.py:38–58](/home/gao/capetown_eval/ds3m_ft/train2.py:38)、[train2_v2d1.py:40–60](/home/gao/capetown_eval/ds3m_ft/train2_v2d1.py:40)、[train2.py:84–91](/home/gao/capetown_eval/ds3m_ft/train2.py:84)、[train2_v2d1.py:86–93](/home/gao/capetown_eval/ds3m_ft/train2_v2d1.py:86)

对象精确率——预测对象中正确对象的比例——从 **0.892 降至 0.864**，已在相同 `h0.9_0.5` 配置的结果中核实；小板命中从 **11/12 降至 8/12**，但分母只有 12，估计仍很不稳定。[原版结果:5](/home/gao/capetown_eval/ds3m_ft/evalobj/TABLE.md:5)、[v2d1 结果:4](/home/gao/capetown_eval/ds3m_ft/evalobj_v2d1/TABLE_v2d1.md:4)

这意味着应检查标签变化与 HN 裁剪内容的交互，以及训练随机差异，不能直接归因于 HN 比例。**“未改瓦片新增误检”这项，我没有独立重做对象匹配；退步的因果来源仍未验证。**

**建议按以下优先级推进；这些是后续建议，本次没有执行写入或训练。**

1. **先审计现有 HN，优先看完整裁剪。**  
   每个活跃 HN 网格随机抽最多 **10 个不同原始对象**，共 **214 个**；记录随机种子，并在对象内随机选一个缓存点。每例看一张近景和按训练规则生成的 32 米裁剪，叠加原始删除边界、当前光伏标签及忽略区。分别记录“点下有真光伏”和“裁剪内有被标背景的真光伏”。另查距已标光伏最近的点、混合屋顶对象及少数点保留下来的瓦片，风险样本单独报告。这样才能区分中心可靠、周边漏标两类问题；只审中心会漏掉实际训练风险。

2. **审计发现真光伏时，修标签或设为忽略，不能只删坐标。**  
   明确真光伏改为正类；无法判断的对象或局部区域设为 `255`。Gemini 可以先分流“疑似光伏／非光伏／不确定”，人工复核疑似与不确定，并抽查其“非光伏”判断。按网格和实际采样机会报告污染情况，不能把多个同屋顶像素当作独立证据。

3. **修正距离定义和清洗报告。**  
   使用原始多边形的米制直线距离；同时报告非背景中的 `1/2/255` 分解，以及各排除条件的重叠和新增删除量。当前 `near_pv`、`near_hand` 是前序条件过滤后的新增数，两个零不能解释成没有任何原始重叠。[hn_clean.py:40–47](/home/gao/capetown_eval/ds3m_ft/hn_clean/hn_clean.py:40)

4. **以后用一个受控对照决定 HN 比例。**  
   固定标签、初始化、随机种子、训练步数和评估规则，先比较 **15% 与 0% HN**；腾出的比例分配给随机有效像素，其他设置保持一致。若确有精确率收益但损害小板召回，再试 5%。同时报告小板逐对象得失和网格间差异，避免只凭总体像素指标选择模型。当前不建议增加 HN 损失权重。

5. **应该刷新 HN，但把当前模型的误检当作待审候选。**  
   在训练网格上利用已有预测筛选高置信、重复出现且类型多样的候选，先人工或辅助审阅，再加入确认背景；其中发现的真光伏应补入正类。保留旧 HN 的一部分，避免只追逐某次模型的错误。验证和测试网格上的发现用于评估，不回灌训练，否则后续指标会失去独立性。现有脚本确实对训练／验证网格排除了指定保留网格。[train2_v2d2.py:28–29](/home/gao/capetown_eval/ds3m_ft/train2_v2d2.py:28)

6. **先核实 v2e 的配套状态，再使用。**  
   v2e README 写着 HN 缓存尚未重建，但当前磁盘已有新的 v2e 缓存；本次确认其 HN 数组与 v2d2 相同，其他坐标数组有变化。现有验证记录也说明了这一点，因此 README 已部分过时；指定位置仍未找到 `train2_v2e.py`。[v2e README:39](/home/gao/data/ZAsolar/geid_temporal/ct_seg_trainset_v2e_20261017/README.md:39)、[v2e 缓存 README:2–7](/home/gao/capetown_eval/ds3m_ft/hn_clean_v2e_20261017/README.md:2)  
   不应因 HN 恰好相同就沿用整个 v2d2 缓存。当前缓存名称只绑定清单文件的 MD5——即文件内容校验值——没有绑定标签栅格和手标内容；将来原地改标签而不改清单，可能继续加载旧缓存。[hn_clean.py:16–17](/home/gao/capetown_eval/ds3m_ft/hn_clean/hn_clean.py:16)、[train2_v2d2.py:30–32](/home/gao/capetown_eval/ds3m_ft/train2_v2d2.py:30)

Durban 的 Gemini HN 标签质量本次**未验证**；本次追溯的 Cape Town 数据和缓存读取链没有纳入它。它不能为当前 HN 的可信度提供额外支持。[train2_v2d2.py:23–32](/home/gao/capetown_eval/ds3m_ft/train2_v2d2.py:23)