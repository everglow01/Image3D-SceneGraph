# MCMC 分片恢复不对称

**当前阶段：E006已完成，E007阶段A已完成且预定配对逐点阈值一致，阶段B已实现、远端未验证；唯一长期根因未证明。** 更新日期：2026-10-27。

## 主问题

同样经历 relocation，为什么有的 Gaussian 恢复，有的很快再次死亡？为什么这类差异在某个 owner 的局部来源池内持续积累？哪种机制真正造成差异，而不只是与差异同时出现？

每个 owner 的点可接收两相机监督；owner、物理 GPU、相机执行 rank、局部来源池不能混为一谈。死亡阈值判定不等于点已删除，也不等于完全没有渲染贡献。

## 先记住的事实与边界

- 原始 apartment 长训存在质量异常与严重分片低透明度；不是简单漏更新或未跑完。
- raw_loss 改善颜色，但没有消除弱分片失衡；floor、矩处理、位置日程不能视为已通过修法。
- 目标相机尺寸错误有隔离正确性证据，但修复后的真实 12k 仍有恢复不对称。
- 死亡先于首次 relocation；搬迁即时缩小死亡数差，但后续恢复不对称。旧总量日志不能识别谁再次死亡。
- 完整生命周期 observer 尚未运行：控制 A/A 在第一步已分歧，旧 Gate1 失败，Gate2 未运行。
- 相同 B0 下 B1 不同已定位；固定 B0 v2 五次的 80 项比较在原容差内，不豁免下游差异，也不证明长期因果。

## 三个子问题

<a id="q1"></a>
### Q1 · 搬迁后为什么恢复不同？

区分目标、来源、未参与点；slot 不是持续不变的 Gaussian 身份。总量时序已排除 relocation 是两条 12k 轨迹最初死亡源，但完整 observer 尚未运行，不能回答谁再次死亡。下一步需要可信的逐点生命周期，关联死亡阈值、render cutoff、实际图像贡献与持续存活；屏幕半径非零不能独立证明恢复。

<a id="q2"></a>
### Q2 · 为什么差异在分片内持续积累？

先将 Q1 的点级失败汇总到 owner 的来源复用与恢复情况，再区分触发和维持失衡的机制。尺寸配对中的弱 owner 会变化，不支持固定 rank1 故障解释，也未证明来源池反馈；映射和池组织干预尚未启动。

<a id="q3"></a>
### Q3 · 哪种机制真正造成差异？

raw_loss 改善颜色但未治愈失衡，其他早期候选各有失败，尚无可推广的唯一修法。根据生命周期证据，在同源状态上选择一个候选做可反驳的单因素干预：先短窗口机制，再检验失衡形成、维持及长期质量，不把相关性当成因果。

## 三条支线及返回条件

<a id="b1"></a>
### B1 · 观测可信性与浮点非确定性

生命周期控制 A/A 失败引出了本支线。已定位 B0 相同而 B1 不同；固定 B0 v2 的 80 项比较在容差内，但既有下游参数差仍需解释。返回 Q1 需结构性非干预检查及独立 A/A 基线支持的指标验收；E006已完成连通传播：固定同一B1的参数/更新量位级一致，改变实测B1产生部分超容差差异。独立非干预验收已按E007预算实现，结构前置和正式运行待独立远端授权；旧Gate1未通过。

<a id="b2"></a>
### B2 · 异尺寸分布式渲染正确性

错误的目标相机尺寸会污染梯度解释。排序控制后，尺寸修复通过合成与短生命周期门禁，真实 12k 配对仍不对称。后续须明确冻结原库或修复环境，不混合归因；正确性修复不代表长期根因闭合或默认晋级。

<a id="b3"></a>
### B3 · 来源复制容量与数值稳定性

floor、目标矩候选失败及复制计数问题引出了本支线。容量候选通过小 CUDA 门禁，但多周期质量和来源池门禁阻塞；无有效监督时，容量限制不能创造恢复信号。保留失败机制，返回逐点恢复和局部来源池主线，不自动晋级 10k/30k。

## 调查因果顺序

最初长训异常 → 候选筛查与五臂 → 失败机制/容量限制 → 上游尺寸正确性 → 真实 12k 配对仍不对称 → 总量时序不足，准备逐点观测 → A/A 阻塞 → 首步/反向边界定位 → 固定 B0 有限重复性证据。

这不是一条已证明的退化机制链，而是“为何做下一项实验”的研究路径。

## 尚未闭合的机制

数值扰动可能触发离散阈值差异；局部 Adam/几何状态可能放大扰动；来源复用和监督不足可能维持不对称。但“微扰 → 点的命运 → 来源池变化 → 持续分片失衡 → 长期质量退化”仍缺直接连接与干预证据。

## 下一步与执行状态

**迁移验收阶段已完成（保留监控终态缺项）：** 2026-10-23 固定 B0 前置 3 项通过，真实 80 项重复比较和 40 项新旧对齐比较均在原容差内；执行任务正常完成，唯一监控日志已见结果但面板结束标志缺失。详见 [验收报告](branches/b1_observation_reliability/experiments/e004_fixed_b0/reports/workspace-20261023-v1.md)。本次不重跑补标志；下次启动前处理监控终态问题。迁移验收不是长期失衡的新因果证据。

**科学下一步：** [E006 单步连通传播](branches/b1_observation_reliability/experiments/e006_connected_propagation/README.md) 已完成10个独立单步对照和结果核验，同输入门禁通过；改变B1会传播为部分超容差的梯度、参数与更新量差异，各case透明度阈值聚合计数相同。详见[结果报告](branches/b1_observation_reliability/experiments/e006_connected_propagation/reports/connected-20261023-v1.md)。下一步按[E007设计](branches/b1_observation_reliability/experiments/e007_observer_admission/README.md)：先只读补齐E006逐点阈值比较，再做独立无中间观测A/A与非干预验收；通过后直接用同批观察数据分析Q1，不重复训练。三类E006带探针对照不能替代该验收。阶段A已远端完成，预定配对及CPU/CUDA口径共360份差异位图的XOR均为0，见[E007阶段A结果](branches/b1_observation_reliability/experiments/e007_observer_admission/reports/masks-20261026-v1.md)。降低该单步数值触发候选优先级，不追加E006重复；阶段B入口、独立结构前置及条件Q1分析已实现；等待Git同步和独立远端授权，未执行。

| 研究项 | 状态与约束 |
|---|---|
| 连通 B1→P2→P3 传播（已完成） | E006已实际重算全量P2/P3与更新量，不拼接历史输出；仅支持冻结单步传播，未证明点命运或长期质量因果。 |
| 独立 A/A 与非干预验收（E007已实现、未运行） | 3次校准＋3次独立原生留出＋3次观察，固定12001—12200窗口；校准与留出分离。阈值/停止条件见设计，不追认旧Gate。 |
| 阶段B预算与回归条件（尚未实测） | 最多1800全局更新、正式流程6小时、新增产物30GiB；失败不加次数/放宽标准，通过即分析同批Q1轨迹。已实现，远端前置及正式运行另行授权。 |

通过有限范围观测准入后，才解释同批Q1短窗口的来源、目标、未参与点轨迹及分组贡献，再按证据选择一个候选机制。E007的1个百分点研究尺度与0.1个百分点自然波动上限是设计假设，不是既有实测结果或统计等价证明。旧Gate1不追认，旧六分支不自动恢复。

## 历史实验索引

下表列研究项，不是新执行次数。原生 A/A 与生命周期观测共享一次执行。详细执行/验收状态在各运行记录，不以表格上的“已有结论”替代门禁。

| 归属 | 实验 | 已有结论 |
|---|---|---|
| Q1 | [MCMC-Q1-E001 · 死亡起点与搬迁后恢复的时序审计](questions/q1_post_relocation_survival/experiments/e001_asymmetry_onset/README.md) | 两条 12k 轨迹中死亡先于首次 relocation；固定点数阶段搬迁即时缩小分片死亡数差，但后续训练又增加死亡。 |
| Q1 | [MCMC-Q1-E002 · 12k 后逐点生命周期观测与门禁](questions/q1_post_relocation_survival/experiments/e002_lifecycle_trace/README.md) | 原库 control_a/control_b 各完成 200 步，但第一步参数/Adam 已不同；Gate1 失败，完整 observer 与 repair 分支未运行。 |
| Q2 | [MCMC-Q2-E001 · 原始 apartment 退化与分片状态核验](questions/q2_rank_accumulation/experiments/e001_initial_audit/README.md) | 原始 MCMC 30k 的 rank1 低透明度占比约 96.164%，rank0 约 17.4182%；参数有限、两 rank Adam step 一致。 |
| Q3 | [MCMC-Q3-E001 · 初始候选机制与参考实现审查](questions/q3_causal_mechanism/experiments/e001_candidate_research/README.md) | 形成四个单因素候选；目标旧矩保留也是参考实现行为，不能直接定性为本项目独有 bug。 |
| Q3 | [MCMC-Q3-E002 · 五臂真实场景稳定性对照](questions/q3_causal_mechanism/experiments/e002_stability_suite/README.md) | 五臂均实际执行，baseline/raw_loss 完成，floor、目标矩清零、位置日程三臂失败；raw_loss 改善颜色但未消除分片低透明度失衡。 |
| B1 | [MCMC-B1-E001 · 原生控制 A/A 准入比较](branches/b1_observation_reliability/experiments/e001_native_aa/README.md) | 使用 Q1 生命周期实验的同一次执行：12001 第一步起参数/Adam 不同，严格门禁失败。 |
| B1 | [MCMC-B1-E002 · 首步 P0—P4 分歧定位](branches/b1_observation_reliability/experiments/e002_first_divergence/README.md) | P0/P1 位级相同；首差在 P2，少量梯度和下游参数超容差。 |
| B1 | [MCMC-B1-E003 · 完整 B0／B1 反向边界定位](branches/b1_observation_reliability/experiments/e003_backward_boundary/README.md) | 完整 B0 与图像梯度位级一致，B1 已不同；首差位于 rasterizer backward 内、反向通信之前。 |
| B1 | [MCMC-B1-E004 · 固定 B0 五次重复与六点敏感性](branches/b1_observation_reliability/experiments/e004_fixed_b0/README.md) | v1 前置因首次导入改变 Python RNG 而停止；v2 审计初始化后，两相机各五次，80 项比较全部在原容差内且无非有限。 |
| B1 | [MCMC-B1-E005 · 反向源码与 ULP／累加规模只读分析](branches/b1_observation_reliability/experiments/e005_readonly_localization/README.md) | 源码和差异分布最支持浮点原子累加顺序候选；近零大 ULP 不等于大绝对误差。 |
| B1 | [MCMC-B1-E006 · 单步 B1→P2→P3 连通传播](branches/b1_observation_reliability/experiments/e006_connected_propagation/README.md) | 10个对照完成；同输入参数/更新量位级一致，不同B1出现部分超容差传播；未证明长期根因。 |
| B1 | [MCMC-B1-E007 · E006收口与观察工具准入](branches/b1_observation_reliability/experiments/e007_observer_admission/README.md) | 阶段A完成：预定配对逐点阈值及两口径一致，降低该单步数值触发候选优先级；不证明任意跨组相等。阶段B已实现、未运行，通过后同批轨迹回归Q1。 |
| B2 | [MCMC-B2-E001 · 原库异尺寸单卡／双卡等价性诊断](branches/b2_camera_size_correctness/experiments/e001_native_equivalence/README.md) | 异尺寸出现差异，但首轮同尺寸对照也失败，需先排除排序混杂。 |
| B2 | [MCMC-B2-E002 · 统一 Gaussian 顺序的尺寸对照](branches/b2_camera_size_correctness/experiments/e002_order_controls/README.md) | 排序控制通过，异尺寸问题仍在，修复目标收敛为每目标相机的 width/height。 |
| B2 | [MCMC-B2-E003 · 目标相机尺寸隔离修复门禁](branches/b2_camera_size_correctness/experiments/e003_size_repair/README.md) | 12 个合成用例、每例 5 步通过原容差。 |
| B2 | [MCMC-B2-E004 · 尺寸修复的原生生命周期门禁](branches/b2_camera_size_correctness/experiments/e004_native_lifecycle/README.md) | 原库与修复的 12 步生命周期门禁通过；本地保留 protocol、summary 与逐文件核验。 |
| B2 | [MCMC-B2-E005 · 真实 apartment 12k 尺寸配对](branches/b2_camera_size_correctness/experiments/e005_scene_pair/README.md) | 两臂完成 12k；修复未消除死亡总量和恢复差异，较弱 owner 从 rank1 变为 rank0。 |
| B3 | [MCMC-B3-E001 · 低透明度退化受控检查](branches/b3_relocation_capacity/experiments/e001_degeneration_probe/README.md) | 完成各臂和独立 unit/监控核验；外层执行面板结束标记缺失，不能伪造 finished。 |
| B3 | [MCMC-B3-E002 · 图像恢复梯度与尺度下溢机制](branches/b3_relocation_capacity/experiments/e002_mechanism_details/README.md) | 有效可见微型例图像恢复梯度可远大于正则；floor 压力例出现尺度下溢。 |
| B3 | [MCMC-B3-E003 · 容量约束候选的小型 CUDA 门禁](branches/b3_relocation_capacity/experiments/e003_capacity_gate/README.md) | 小型 CUDA 门禁通过；保留真实复制计数与候选范围。 |
| B3 | [MCMC-B3-E004 · 容量候选多周期恢复检查](branches/b3_relocation_capacity/experiments/e004_capacity_cycles/README.md) | 受控研究完成但门禁阻塞：可见例 PSNR 差约 −1.2601 dB，无监督／不平衡条件有来源耗尽；未晋级长训。 |

## 归档入口

- [远端存储精简与旧产物可用性](storage-retirement-20261023-v1.md)：已用量降至249.49GB；部分旧二进制已删除，E006全量内容与冻结来源保留。
- [代码、测试和辅助证据迁移清单](migration.md)
- [完整历史回顾（本地旧报告）](../../outputs/analysis/mcmc-investigation-retrospective-20261022-v1/report.md)
- [工作区执行规则与索引检查](../README.md)

- [研究历史档案](history/README.md)：codex 72 条决策与两份合同的 24 节研究细节逐字保存；旧章节入口和冻结约束继续有效。
