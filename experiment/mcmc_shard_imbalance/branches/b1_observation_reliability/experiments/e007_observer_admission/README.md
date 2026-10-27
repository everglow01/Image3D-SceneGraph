# E006收口与观察工具准入

2026-10-27 · **MCMC-B1-E007** · **阶段A远端验证完成；阶段B代码已实现，结构前置和正式运行均未执行。**

## 1. 目的与当前授权

主线是：Q1迁移后的点为何恢复或再次死亡 → Q2差异如何在owner局部来源池积累 → Q3单因素干预验证因果。[E006结果](../e006_connected_propagation/reports/connected-20261023-v1.md)只证明冻结单步中B1差异可以传到参数更新，尚未证明点命运、来源池或长期质量变化。

本设计将支线限定为：**已有数据只读收口 → 一轮有上限的观察工具验收（同时采集Q1数据）→ 通过即分析Q1**。不继续追加固定B0重复实验，不自动开长训。

| 阶段 | 做什么 | 如何影响实验导向 |
|---|---|---|
| A | 比较E006已有快照的逐点阈值掩码，不训练 | 决定是否提高或降低数值扰动候选的优先级 |
| B | 3次校准、3次独立原生留出、3次带观察器对照，各200步 | 判断观察工具在预定尺度内是否可用于探索 |
| Q1 | 验收通过后直接分析B阶段的3次观察数据 | 回答哪些点恢复失败、何时失败、与哪些因素相关 |
| Q2/Q3 | 根据Q1结果只选一个候选机制 | 另设计可反驳的因果对照，不提前启动 |

用户随后要求按阶段继续实施，并在手动Git同步后明确授权阶段A远端验证。阶段A前置及真实补分析已完成，执行/监控均退出0，360份差异位图的XOR数量均为0，详见[阶段A结果](reports/masks-20261026-v1.md)。本结论限预定配对，不证明native与注入组任意跨组相等；降低该单步数值触发候选优先级。随后按用户要求实现了独立`run_b.py`及结构前置。B尚未进行远端验证或正式运行；A的入口、结果与旧Gate1失败状态保持不变。

### 从现有实现确认的限制

- 旧`run_mcmc_lifecycle_trace.py`的control仍安装`LifecycleObserver(observed=False)`，会做逐步哈希并包装抽样，不能复用为无中间观测基线。
- 原生trainer在`stop_after_iteration`终点仍会触发Validation；新入口必须在该调用前截停，不能运行旧runner后声称只使用Train。
- [旧生命周期实验](../../../../questions/q1_post_relocation_survival/experiments/e002_lifecycle_trace/README.md)没有完成完整observer运行。现有代码可复用，不等于已通过新的验收。
- [存储退役记录](../../../../storage-retirement-20261023-v1.md)已说明部分旧二进制不可用；E006原始内容及绑定来源保留，但实施时仍需重新核实。

## 2. 阶段A：用已有E006快照收口，不再训练

### 问题
“相同死亡总数”之下，是否存在不同点跨过阈值？这决定数值扰动在主线中的优先级，而不是决定是否跳过工具验收。

### 数据与方法
- 只读E006已保留的10case P0/P3及manifest；使用已有same/native/varied配对，不新增挑选最坏case。
- 两owner分别比较全量逐slot死亡诊断掩码、alive→dead和dead→alive掩码；输出XOR数量及可复核的压缩slot集合，不仅比较总数。
- 保留CPU float64 sigmoid诊断口径；另在指定远端以冻结Torch/CUDA版本、原dtype重算sigmoid阈值诊断，并与float64分开报告。后者仍是离线重算，不能称实际strategy执行过的死亡判定。
- 阈值`.005`与render cutoff `1/255`分别标注；检查shape、有限性、owner/slot身份。E006没有迁移或拓扑变化，才允许按同一slot逐点对齐。
- 优先只解码所需opacity叶子；文件验真一次后复用，不反复加载六阶段全部数组。原始文件（含硬链接）始终只读。

### 分流
| 结果 | 对主线的影响 |
|---|---|
| 全量掩码一致 | E006收口为“参数差异，但此单步未见逐点阈值差异”；降低数值扰动候选优先级，不追加次数/步数找效果。 |
| 掩码不同 | 保留“可能触发离散命运差异”的候选，但仍需Q1轨迹与后续干预，不宣称长期因果。 |
| 原dtype与float64不一致 | 标注阈值数值口径敏感；Q1以实际执行路径及明确阶段记录为准，不用离线诊断冒充策略事件。 |

阶段A结果不改变阶段B的case数、验收阈值或窗口，不修改旧E006报告/协议来追认新结果。

## 3. 阶段B：一次独立验收，同时收集可用于Q1的轨迹

### 冻结条件
- 只研究E006对应的原库baseline；不同时引入尺寸修复、raw_loss、floor、矩重置或来源池重组。
- 使用E006绑定的iteration12000父checkpoint、Train replay、初始化和环境；父checkpoint SHA为`080952acf0ef989b160b64d9bb8e17f8acb7adf2bf7e18b19e3ae1d94d3c2105`，core SHA为`5941370c16bb04301db2f59477468eebe62326201d12c487980a1dfb51dcde8a`。
- 各case从相同父状态独立恢复，运行12001—12200共200步，名义学习率日程仍为30000。
- 12100发生首次目标迁移；跟踪该cohort至12200下一次迁移之前。随后允许原生12200策略及noise正常完成，统一在Validation入口保存终点并截停。cohort终点与训练终点明确不同。
- 本轮是新协议的有限范围准入，不追认旧Gate1，不恢复旧六分支。

### 对照与固定次序
| 组 | 次数 | 用途 |
|---|---:|---|
| C0/C1/C2：无中间探针基线 | 3 | 校准本窗口自然波动的描述性范围 |
| U0/U1/U2：独立无中间探针留出组 | 3 | 检查校准范围能否覆盖新的原生运行 |
| O0/O1/O2：完整候选观察器 | 3 | 验收观察器，并采集迁移后逐点轨迹 |

固定顺序`C0,C1,C2,U0,O0,O1,U1,U2,O2`，交错验收顺序降低单向时间漂移；不得按结果换顺序或补样本。

校准只用C组；校准摘要及阈值文件先落盘、哈希冻结，才开放U/O运行。U/O不参与调阈值。任一U/O超限即停止剩余case，保留失败/证据不足状态，不自动重试。

### 无中间探针基线的真实边界
- C/U不安装LifecycleObserver，不包装B1、不挂逐步optimizer/strategy/sampling钩子、不计算中间哈希、不额外做贡献渲染。
- 保留原生trainer本来就有的训练日志、事件诊断、资源管理和GPU锁；“无中间探针”不是完全无日志。
- 所有组使用相同的终点接口：包装`evaluate_views`，在第一次调用、且原生progress已证实完成12200时，保存传入model的完整参数、终点RNG及共同日志索引，然后抛`ProbeComplete`。不调用真正evaluate，不伪造评估返回值。
- 终点接口不声称持有完整Adam/strategy/camera内部状态，不为取它们新增逐步钩子；其缺失明确写入证据边界。父状态身份通过原生恢复与来源校验保证，不伪称逐case捕获了完整P0。
- 先验证冻结配置在12001—12199无Validation触发；提前进入evaluate即失败。终点所有训练更新已结束，采集后不再进入下一步。
- Train之外的图像解码设共同拒绝守卫；懒加载split元数据允许，Test图像和Validation图像均不加载。

### 结构性非干预检查：先于正式case
复用`unchanged_rng`、`reuse_record`、`regularizer_gradient`、`verify_alpha`和现有微型测试思想，在指定远端独立opt-in验证：
- 原调用恰好一次；不替换梯度，不修改参数、Adam、strategy活跃状态或输入，不恢复RNG掩盖副作用。
- 额外统计与输出前后RNG一致；采样原调用自然消耗的RNG与观察逻辑消耗明确分开。
- 额外贡献渲染只在O组、no_grad下发生，明确单独计数，不谎称两组forward总数一致。
- 贡献质量守恒、owner/相机映射、累计正则梯度口径、目标/来源/未参与身份均有负例测试。
- 共同终点接口验证最后一次noise/策略已完成、Validation/Test解码0、下一迭代0；双rank异常协同停止，不悬挂collective。
- 结构检查通过不等于真实数值路径完全不受调度影响；后者由独立C/U/O比较提供有限证据。

## 4. 判据：提前冻结、保守分流，不宣称小样本统计等价

### 本轮设计尺度（已批准设计假设，不是既有实测结论）
为了不把微小聚合变化解释成主线机制，本轮仅拟支持**至少1个百分点的owner死亡率差异**这一尺度的探索；自然重复波动上限设为其十分之一，即**0.1个百分点**。这不是最小真实效应或通用科学阈值；更小的效应本轮不作可靠结论。

### 共同主指标
各owner分别、不可用全局平均掩盖一侧：
1. 12100、12200原生策略前/后的低透明度比例（读取原生health日志，必须字段齐全）。
2. 两个事件的迁移数量/本owner点数。
3. 12081—12100及12181—12200的平均Train reconstruction loss（共同原生日志，不额外渲染）。
4. 参数有限性、点数、完成步数、事件时序和相机batch序列。

针对每个连续指标m：
- C组参考为三次值的中位数；半宽`E_m=max(C组max-min, 数值分辨下限)`。
- 比例的下限为`1/N_owner`；loss的下限沿用`2e-6 + 2e-4*abs(C中位数)`。不乘扩大系数。
- 比例E不得超过0.001（0.1个百分点）；loss E不得超过`2e-6 + 1e-3*abs(C中位数)`。若校准本身超上限，结论是“当前窗口自然波动过大，无法支持拟定分辨尺度”，不放宽上限。
- 每个U/O指标都须落在冻结`C中位数±E_m`内；所有指标同时通过才允许使用O数据作本窗口探索。
- 参数的原元素容差`2e-6+2e-4*abs(reference)`继续完整报告，不扩大、不改旧门禁。它不再单独充当新协议的逐点轨迹一致性要求；终点参数超容差不等价于观察器造成偏差。
- 终点RNG记录供诊断；迁移抽样量自然变化可导致不同运行RNG不同，不能一概判为观察器消耗。单次观察操作前后RNG不变仍是硬要求。

### 三种结论
- **有限范围准入：**结构检查、C范围、全部独立U/O共同指标通过。仅表述“在此窗口、此观察器及分辨尺度下，未检出超过冻结范围的额外扰动”；不是全局等价证明，也不是长期因果证据。
- **观察器可疑：**全部3次U通过而O失败。停止解释O的生命周期数据；只定位一类最直接的额外观测操作，修订方案另行确认，不自动加新对照。
- **基线不稳定/证据不足：**C或U失败。不能归罪尚未证实的观察器，也不补样本到通过；保留原生总量现象，给出下一步主线测量能力缺口。

交错顺序中若O提前失败、其余U尚未运行，只能记录“观察组超限、独立基线覆盖不足”；不能用一次U通过代替全部U通过，也不为补归因而自动继续运行。

跨case经过迁移后，同一slot不保证是同一Gaussian；不以终点slot的差异当作同一点命运改变。有限准入只允许运行内身份追踪及跨重复的群体描述，不足以支持精确逐点跨运行因果配对。

## 5. 验收通过就回归Q1，不再重复训练

O组在验收时已使用准备投入Q1的同一版本观察器；验收后直接分析其数据，不增加另一轮200步。

### 观察器最小增量
复用`scripts/mcmc_lifecycle_observer.py:LifecycleObserver`的真实采样包装、贡献计算和梯度分析，旧文件保持冻结。新实验旁置薄适配器，仅补齐主线所需的持久化：
- 12100实际target→source映射、owner、事件号，明确slot身份在事件前后改变。
- 迁移后初始alive状态；12001—12100不预造cohort。
- 12101—12200每次Adam后全量alive位图，压缩保存；在12200第二次迁移前结束该cohort，禁止混入下一代。
- 每点first_death、first_recovery、末态、累积image-gradient residual/正则/更新量，以及事件前后必要opacity值。读取观察器已有数组，不改活跃训练张量。
- 已有分组贡献渲染只能解释owner×目标/来源/未参与组的合成权重，不能冒充每点可见性或loss改善；`total−analytic_regularizer`是图像梯度残差诊断，不是精确因果分解。

这些增量必须纳入阶段B被验收的版本，不能验收旧观察器、分析时再偷偷增加功能。

### Q1主结果与返回条件
- 每owner分别给出目标/来源/未参与点的初始恢复比例、首次再死亡时间、100步内再死亡率、首次恢复时间及恢复后持续alive步数。
- 初始未alive的目标单独归类，不能算作“先恢复后再死亡”；同一source被复制多次按真实复用次数记录。
- 比较失败点与存活点的监督残差、正则、更新量与来源复用关系；报告3次O的一致/不一致趋势，不只挑最漂亮一次。
- 首个交付是“恢复失败发生在哪里、何时、与哪些候选因素相关”的表，不写唯一根因结论。

## 6. Q2/Q3按结果选择，不现在扩成多分支

| Q1结果 | 下一步方向 |
|---|---|
| 失败点图像梯度残差弱，正则/历史Adam更新占主导 | 优先设计监督不足/更新机制的单因素短窗干预；分组贡献只能辅助，必要的逐点贡献测量需另验收。 |
| 失败反复集中于高复用来源或特定局部池 | 优先Q2来源池关联与组织机制；区分owner、物理GPU和相机rank。 |
| E006有逐点阈值差异，且Q1跨重复命运不稳定 | 保留数值触发候选；另设计来源匹配与扰动控制，不直接恢复旧B1注入或长训。 |
| 三次O趋势不一致或没有明确差异 | 如实报告本窗口不足，不自动扩大步数或增加重复。 |

只选一个有证据支持、可被反驳的机制进入Q3。短窗干预先验证机制与恢复指标，再决定是否值得做持续失衡及长期质量；不提前部署修法或晋级默认。

## 7. 运行、存储与失败预算（未来需单独授权）

- 阶段A：不训练，新增摘要/压缩索引上限1GiB，分析上限1小时。
- 阶段B真实case最多9×200=1800次全局更新；3600次训练相机forward和3600次loss backward；O组三次另有1200次无梯度贡献forward；六参数组双rank合计21600次Adam组step。
- 每case两次全局迁移事件，共18次；各owner调用合计36次。保留原生逐步noise，不把本轮预算写成E006的strategy=0。
- 微型结构前置预算已在`run_b.py`单列冻结，见下文；不计入真实1800步，旧阶段A授权变量不能开启。
- 单一执行任务＋唯一只读监控、GPU lease/flock/once、失败不重启；整套正式case及比较上限6小时。首个case过慢若按剩余数已无法在预算内完成，停止并报告，不减对照偷偷继续。
- 每卡reserved上限18GiB；主存memory.max/high/swap无限额；运行磁盘剩余低于4GiB停止。
- 全套新增产物硬上限30GiB，准入需预算加8GiB余量。原已用249.49GB是清理完成的时点值，**新实验会增加占用，不承诺仍低于250GB**；正式申请列出当时实测估算。
- 父checkpoint只复制一次，只读共享；每case只保存终点完整参数及必要RNG、原生日志、O的压缩位图/点表，不再保存每步完整P0—P3/Adam快照；不自动删旧数据或新失败证据。
- 核验清理后真实来源可用性；旧历史checkpoint/快照被删的路径不能替代新基线。E006保留的硬链接内容禁止原地写。
- 本机只标准库/AST/NoTorch/Ruff；数组、Torch/CUDA、CPU张量分析也只在指定远端。新源码只经Git、用户手动同步；不上传源码包或远端写源码。只下载必要文本到隔离incoming，标准库核SHA。

## 8. 实施时应复用的代码与验证

阶段A入口保持冻结；阶段B已在独立文件中实现，复用范围如下：
- `e006_connected_propagation/analyze.py`的配对与opacity口径；以新分析入口补充全量mask，不改旧分析实现或历史comparison。
- `scripts/mcmc_first_divergence.py`的typed快照、安全路径、canonical hash、`ProbeComplete`及原容差比较。
- `scripts/mcmc_lifecycle_observer.py`上述观察方法；不调用旧runner的Gate1/六分支入口。
- `scripts/run_mcmc_lifecycle_trace.py:worker`仅借鉴原生resume参数与provenance验证，移除旧control安装观察器和终点Validation执行的做法。
- E006 `publish_status`、worker双rank拒绝模式及独立终态；项目`FileLease`、`require_idle_gpu`、`memory_telemetry`、`failure_record`。

设计文档检查：目录链接、design-only元数据、旧验收断言、`git diff --check`，确认不出现“已运行/已通过”的新预填记录。

未来代码验证重点：
1. 本机NoTorch负例：“总数相同但掩码不同”、迁移后slot不能直接配对、阈值恰等于`.005`、校准与留出数据不可混用、超限不能改阈值、原始硬链接只读、超预算/重复提交拒绝。
2. 远端合成：实际dtype阈值重算、结构非干预、真采样调用一次、恢复/再死亡分类、终点在策略后Validation前、双rank同步拒绝。
3. 正式前检查Git提交、来源SHA、资源、独立授权及协议SHA；完整结果记录execution/data validity/numerical outcome/scientific scope分开。
4. 结果一旦满足有限准入，立即交付Q1分析及候选排序；不另开E008继续寻找更漂亮的重复性。


## 9. 阶段A实现与运行边界

- 入口：`run.py`（`prepare/run/worker/monitor`）；计算：`masks.py`。没有阶段B、训练或自动追加case参数。
- 验真链：冻结E006 protocol/comparison → 10case complete → worker → P0/P3 manifest → 所需opacity文件SHA及canonical SHA。只重验所需数组，不声称重新读取全部E006快照。
- `cpu_float64`复用E006稳定sigmoid，计数须与冻结comparison一致；`cuda_original_dtype`在原Torch/CUDA/GPU型号下用float32执行sigmoid和阈值比较。两者均是离线诊断，不是策略实际执行记录。
- 保存360份可复核的压缩XOR位图：固定10配对×2owner×2口径×6掩码，以及10case×2owner×6跨口径掩码。位序为little、slot从0开始，显式记录长度、XOR数量、padding、SHA；不把配对累计数当唯一点数。
- 本阶段真实训练更新、rasterizer forward/backward、strategy均0；40次CUDA sigmoid和40次CPU sigmoid。独立远端合成前置另有4次CUDA sigmoid，无训练。
- 上限1小时、新增1GiB；磁盘准入9GiB、运行剩余4GiB停止；每卡reserved≤18GiB，主存无限额。只写新E007 runs目录，来源（包括硬链接）只读。
- 执行任务持GPU锁并用once防重；子进程继承锁，父进程监督时限、磁盘和显存；只读monitor不持锁。失败保留终态，无自动重试。

### 本轮执行顺序（已完成，不重复提交）

1. 提交新源码并由用户手动Git同步；核对远端host、仓库、提交和两卡空闲。不通过源码包或远端写源码替代。
2. 明确获准后，在指定远端设置`IMAGE3D_AUTHORIZE_OBSERVER_MASK_TESTS=1`，仅运行`tests/test_masks_remote.py`。前置失败不prepare、不运行真实分析；保留前置测试日志与终态。
3. 新目录下执行`python -m experiment.mcmc_shard_imbalance.branches.b1_observation_reliability.experiments.e007_observer_admission.run --prepare --authorize-prepare --output-root <新的E007运行目录>`；核验产出的protocol SHA。
4. 使用专用`image3d-observer-masks-v1.service`执行同模块`--run --authorize-analysis --protocol-sha256 <SHA> --output-root <同目录>`；主存三项限制均为infinity。外层启动脚本将前置、prepare及分析串联，任何失败即停止；用显式子shell保证面板写终态。
5. 唯一只读监控调用同模块`--monitor --protocol-sha256 <SHA> --output-root <同目录>`；同时保留执行与监控面板终态。只下载必要JSON/日志，不下载原始数组。
6. 阶段A结果归档并解释后，再继续阶段B；本入口不会根据结果自动启动B。

本机只运行NoTorch、AST、Ruff与目录回归。阶段A已取得远端合成前置和真实数组分析结果；本机只核文本SHA与统计，不重算张量。阶段B准入与主线因果证据仍缺席，不登记为通过。

## 10. 阶段B实现状态与执行准入

**已实现，未远端验证；当前不能填写观察工具准入通过或Q1实测结论。**

### 文件职责

- `run_b.py`：独立来源/代码/前置报告绑定，父checkpoint单次只读复制，9case编排、资源与失败监控；模块名为`experiment.mcmc_shard_imbalance.branches.b1_observation_reliability.experiments.e007_observer_admission.run_b`。
- `admission.py`：标准库读取原生训练与owner健康日志，冻结C0/C1/C2范围，独立验证U/O。逐case超限即停止，不调整阈值、不补跑剩余case。
- `observer.py`：共同终点边界及冻结`LifecycleObserver`的旁置适配器。C/U不构造观察器；O保存事件快照、逐步alive位图和已有累计诊断，不改原有观测算法。
- `cohort_analysis.py`：只有全部留出通过后才读取O0/O1/O2；从逐步位图独立重建首次死亡/恢复并核对保存记录，统计首次恢复后的连续alive步数，明确窗口结束的右删失。终点参数原容差比较仅描述，不作为放宽原门禁的工具。

### 两个实现边界

1. 原生日志并不保证含独立的每owner实际迁移计数字段。因此每owner采用原生策略前dead_count作为该冻结原库全量迁移的数量依据，同时核对两owner之和等于原生全局实际`mcmc_relocated_count`且新增点数为0；O另存真实target→source采样映射。这是明确标注的派生计数，不伪造C/U的逐点采样观测。
2. 共同终点只保存参数、RNG和原生窗口日志，不声称取得完整终点Adam/strategy/camera内部状态。C/U的更新预算由冻结trainer与连续200步原生日志核验，不为了计数插入optimizer钩子。

O的本地观察/策略钩子使用双rank失败汇总，避免一侧拒绝后另一侧继续下游通信。附加贡献渲染本身仍是分布式算子，不能承诺局部异常总能在其内部协同返回；所有组统一在实际子进程初始化进程组时设置120秒collective超时，沿用冻结gsplat的初始化warmup，由spawn管理同组进程退出。没有修改NCCL算法/传输参数，六小时整套上限仍保留。

### 独立结构前置（尚未执行）

需新变量`IMAGE3D_AUTHORIZE_OBSERVER_ADMISSION_TESTS=1`，以及指向本实验runs内**新文件**的`IMAGE3D_OBSERVER_ADMISSION_PREFLIGHT_REPORT`。pytest的`--basetemp`也必须位于本实验独立runs目录；先创建空前置目录，不复用旧测试临时目录。

仅运行`tests/test_admission_remote.py`，合成场景为每owner8点、两张64×64合成Train张量：无观察与有观察各4步，使用真实双卡渲染、Adam和MCMC策略。没有读取真实Train/Validation/Test图像。

冻结预算：8次全局合成更新；16次训练相机forward、8次额外分组贡献forward、2次独立全点模型贡献参考forward；16次训练loss backward、2次独立解析正则校验autograd调用；96次Adam参数组step；1次双rank协同拒绝、1次主动消耗Python RNG的负例。负例只恢复它自己的测试状态，不允许生产观察器恢复RNG掩盖副作用。

前置验证：原采样/反向调用、参数/Adam/梯度/RNG不被观测改写、独立贡献参考、alpha守恒拒绝、解析正则梯度、无中间观测路径、真实轨迹重算与事件身份、终点截停、Test/Validation解码拒绝、双rank失败边界。仅成功后写两rank证据及总报告；报告绑定当次Git提交和全部执行依赖SHA。

### 后续获授权后的执行顺序

1. 先提交新源码，由用户手动Git同步；核对host、提交、冻结来源、两卡空闲、磁盘、当前未占用的新输出目录。**本轮没有提交、同步或启动实验。**
2. 在主存三项限制均为infinity的独立任务中跑上述合成前置。pytest整体退出0且报告完整后，才进入prepare。
3. 用新模块执行`--prepare --authorize-prepare --preflight-report <新报告> --output-root <新的admission-*目录>`；prepare拒绝缺失、失败、单rank、陈旧提交或SHA不符的前置证据。复核新protocol SHA。
4. 单一执行任务在`image3d-observer-admission-v1.service`中执行`--run --authorize-execution --protocol-sha256 <SHA> --output-root <同目录>`；主存无限额，CUDA可见卡按现场核验设置。唯一只读监控调用`--monitor`，不持GPU锁。完整面板封装在正式申请时生成，不在本轮远端部署脚本。
5. C组三次结束后冻结校准；U/O依预定交错顺序运行。任何超限以`admission_blocked/exit_code=2`保留原因和已通过留出，停止后续；执行故障为`failed/1`，都不产出成功Q1报告。
6. 全部准入通过后，在同次执行预算内生成参数差异描述和Q1关联报告，不重新训练。新的远端科学结果、面板终态与资源证据到齐后再登记运行记录；不能用本机测试代替。

来源核验包括已完成阶段A、冻结原库父状态和Train replay。输出准入按30GiB上限加8GiB余量，父checkpoint只复制一次并设为只读；每case保留必要终点/轨迹而非每步全量Adam快照。若已观测case耗时推算剩余固定套件不能在预算内完成，提前停止，不缩减对照冒充成功。
