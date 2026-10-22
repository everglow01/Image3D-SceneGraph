# Trainer 研究历史合同

以下 19 节来自 `docs/gaussian-trainer-contract.md` 的研究区段，按原顺序逐字归档。**迁移文档位置不取消冻结条款**；“仅实现”“待验证”等保留为原时点陈述，不能当作现行执行许可或把后续结果倒写入旧合同。

原路径仍以仓库根解释；现行问题见 [主课题](../README.md)，各实验 protocol 和 [归档清单](archive-manifest.json) 保留原身份。产品合同在原位置保留同名章节入口。

<a id="trainer-001"></a>
<!-- archive:trainer-001:begin -->
## 异尺寸相机分布式一致性诊断（2026-10-19）

用户批准指定远端原安装gsplat的独立小门禁，不修改生产渲染器/MCMC/通信配置。`smoke_distributed_camera_sizes.py`用9个确定性合成用例，每例16全局Gaussian、SH3、5次fresh Adam更新；覆盖同尺寸64×64与1361×2048、历史宽差1361/1372×2048、放大64×64/96×80、相机交换/所有权交换、不等长7/9分片及一分片不可见。历史用例仅复用尺寸，K/pose/点/目标均合成，不是apartment重放。固定全参数、相机、目标、环境/core和脚本哈希，原包源码/二进制前后校验。

单卡完整模型采用dense非分布式路径，每相机L1+SSIM损失除2累积后只做一次Adam更新；双卡每卡一图、unpacked且损失除2，不使用DDP额外平均。沿用clamp规则，噪声、拓扑与正则关闭；首步额外对比公开单卡packed路径与dense参考。逐rank在collective前保存raw RGB/alpha、六组梯度、每步参数/Adam两矩/step误差，主判据atol=2e-6/rtol=2e-4，非有限不得填零冒充通过；报告边缘32px及内部超容差元素数。首步图像/梯度与后续轨迹误差分开，不能由Adam微小梯度放大单独归因通信。

同尺寸或packed参考失败标control_blocked；缺用例/缺rank/缺更新标incomplete；数值不一致仍执行剩余用例并最终非零退出，只有全部完整且一致才passed。真实子进程/资源失败独立记录、不自动重试。专用unit image3d-camera-size-equivalence-20261019-v1.service和新outputs/analysis目录，唯一只读监控；每例300秒、unit3600秒、每卡reserved18GiB、磁盘20GiB准入/4GiB停止、GPU租约；RAM max/high/swap无限额、host_group=None。新代码只Git同步，本机禁止Torch；不读取Test/RGB或checkpoint，不启动10k/30k，不自行修复或推广。门禁不证明该问题解释全部长期退化。

<!-- archive:trainer-001:end -->

<a id="trainer-002"></a>
<!-- archive:trainer-002:begin -->
### 排序隔离 order-v2（2026-10-19）

上一轮legacy-v1控制失败保留。新profile按rank拼接顺序建立全模型参考，并以显式全局ID逆映射比较所有参数、梯度和Adam状态；canonical顺序用于内容hash以及equal_small首步排序敏感性观测，该附加观测不替代主控制。原9例保持原定义，另增加equal_small/historical/mixed三个unique例：z=2+.05*i，xyz与尺度同比缩放，真实float32深度必须无并列。共12例，每例16点/SH3/5更新；边带为min(32,max(1,min(width,height)//4))，避免小图边带覆盖全图。主容差仍2e-6/2e-4，只有完整、所有同尺寸及packed控制通过且原库异尺寸仍不一致，才允许下一阶段隔离尺寸修复验证；控制失败停止，不自动补跑。专用image3d-camera-size-order-20261019-v2.service，新目录与唯一只读监控，unit4200秒、每例300秒，原显存18GiB、磁盘20/4GiB、RAM无限额和GPU锁不变。本轮尚未修改安装gsplat、生产入口、MCMC或通信配置；后续补丁必须独立commit并保留原CUDA二进制，不直接推广。

<!-- archive:trainer-002:end -->

<a id="trainer-003"></a>
<!-- archive:trainer-003:begin -->
### 目标相机尺寸隔离修复 repair-v1（2026-10-19）

仅order-v2完整且全部同尺寸/packed控制通过、异尺寸仍不一致时允许新候选。`prepare_gsplat_camera_sizes.py`锁定原rendering.py SHA，复制原包并只修改Python渲染层、保留所有CUDA二进制。分布式汇集每rank宽高；每rank一相机/dense/普通pinhole EWA范围内，同尺寸保留原投影调用，不同尺寸逐目标viewmat/K及对应width/height调用原kernel并可微拼接。禁止用max画布或resize代替目标尺寸语义；未覆盖模式明确拒绝。不触碰原安装包、生产核心、公开默认和其他overlay。

repair-v1使用相同12例/16点/5更新、相同输入与同序参考/容差，独立记录base environment及补丁profile/全文件SHA，不冒充原库身份。基线repair_ready、原包SHA、canonical初始参数/目标/K/pose、参考ID/optimizer/core均须配对；每rank核验实际导入路径。任一数值/资源/身份失败保留并停止余例，缺失记录维持未完成，只有全矩阵完整通过才passed。专用image3d-camera-size-repair-20261019-v1.service、新实验目录和唯一只读监控；每例300秒/unit4200秒、显存18GiB/磁盘20-4GiB/RAM无限额/GPU锁保留。不推广生产、不读Test、不重跑场景或10k/30k；小门禁不当真实场景长期质量与3M性能证明。

<!-- archive:trainer-003:end -->

<a id="trainer-004"></a>
<!-- archive:trainer-004:begin -->
### 原生生命周期与保留日程的实验结束（2026-10-21）

尺寸overlay正式进入native训练环境身份：marker/profile/实际导入路径/完整文件SHA/来源rendering SHA及原二进制对应必须有效，禁止与AbsGrad或MCMC库变体混用；原库与尺寸环境不能互换checkpoint。此接线改变core，不用旧core门禁替代新生命周期验收。

内部`train_gaussians(stop_after_iteration=...)`和MCMC CLI `--stop-after-iteration`只设置正常执行端点，不改变effective config的名义iterations/LR/SH/拓扑日程。默认不传保持原行为；显式端点为1..nominal整数且必须晚于resume起点，不用于prepare/readiness/external trainer。实际终点执行完整Validation、终态checkpoint和best模型合并；result.iteration及checkpoint目录为实际步，额外字段记录nominal_iterations/requested_stop_iteration/termination_reason，不把12k称为完成30k。取消仍不写完整checkpoint。首次正式研究仅原库与尺寸修复、同core/同apartment replay/seed/2048/3M，每臂12k、名义30k，不叠加其他MCMC修改、不发布生产。生命周期和场景实际结果另行记录，不以实现完成冒充验收。


<!-- archive:trainer-004:end -->

<a id="trainer-005"></a>
<!-- archive:trainer-005:begin -->
### 同核心 apartment 尺寸修复 12k 配对（2026-10-21）

`run_camera_size_scene_pair.py`仅在已授权远端执行，先核验同core原库/尺寸overlay原生生命周期passed和跨环境checkpoint拒绝，再冻结来源Job `20261010_172759_d4dda4b3` 的replay、初始化、相机、配置、脚本、门禁与完整安装包SHA。两臂fresh初始化，原MCMC配置、2048最长边、双卡全局3M及seed不变；名义30k日程保留，实际正常结束于12k/24k相机样本。原库baseline先行，repair仅用已验收尺寸overlay，不叠加其他消融。

Validation固定3k/5k/7k/10k/12k，逐次核对完整351个冻结视角、有限指标/RGB及两rank健康；末态checkpoint组件流式核验SHA、provenance和12k/30k scheduler，不在本机反序列化模型。两臂必须同输入/core/config及实际相机序列。主要比较同一步尤其12k终态raw/display质量与分片健康，best模型只作补充；若原库本轮未复现历史退化，不宣称修复消灭该退化。

新目录 `outputs/analysis/apartment-camera-size-pair-20261021-v1/`，一个串行执行任务及唯一只读监控。每臂独立systemd unit、6小时上限、18GiB/卡reserved、20GiB磁盘准入/4GiB停止与GPU租约；RAM max/high/swap无限额、host_group=None。独立臂失败保留、不重试；另一臂仍须资源准入，两臂不完整不报成功。现有replay完整性可校验Test文件字节，但不解码或训练/评价Test RGB；不运行SOR、final-fit、30k完整训练或发布推广，不改原包/生产/通信配置。

<!-- archive:trainer-005:end -->

<a id="trainer-006"></a>
<!-- archive:trainer-006:begin -->
## 固定B0 v2实测结果（2026-10-22）

6230db4在独立v2目录完成前置3项及双rank各五次原backward；初始化导入仅改变Python RNG，测量内全部输入/RNG不变、无额外热身。协议ff6e21e7…，全部20配对/80张量/5.4亿配对元素中严格与bitwise差异29,719,460、超原容差0、非有限0；90.1303%非零差异≤4ULP、97.7905%≤16ULP。独立CPU复核80配对与32份频次/range数组一致，状态numerically_equivalent不等于位级相同；只代表这组输入/环境/五次，不证明全局误差上界，原六分支Gate仍不追认。

预选三异常及三匹配对照的历史六组Adam分析支持局部更新敏感性，且不能只以“旧矩为零”判断：709801的四元数处在旧v与当前g²混合区，零旧矩的对照413834却给相同更新；1488267的shN分量sigma/eps约.21才是eps主导。新B1没有经projection/通信回传生成新P2/P3，不把跨运行并列数据拼接成完整因果链；来源池恢复不对称及长期退化仍未证明。未改CUDA/确定性/通信/优化器或旧验收标准，后续统计非干预门槛须另行设计。

<!-- archive:trainer-006:end -->

<a id="trainer-007"></a>
<!-- archive:trainer-007:begin -->
## 固定B0初始化边界修复（2026-10-22，仅实现）

用户要求修复后手动同步。新profile为`mcmc_fixed_b0_five_repeats_v2`、unit为`image3d-fixed-b0-v2.service`，保留首轮v1失败记录。在共享`replay_five`入口、五次计数和RNG基线建立前，按源协议SHA核验已安装`gsplat.csrc`原生扩展的loader/路径/文件SHA，先直接载入该二进制，再载入后端并确认`_C`对象身份及函数存在；缺扩展或身份不符即失败，不进入JIT fallback，不调用任何额外forward/backward作热身。初始化前后的四类RNG摘要与变化列表写入独立`initialization.json`，不恢复RNG；测量从初始化后的实际RNG状态开始，五次内部的全部输入/RNG严格检查原样保留。

初始化审计SHA由worker记录绑定，完成校验要求零热身调用且首轮RNG摘要等于初始化后摘要；比较报告分别携带两rank审计，不把初始化消耗解释成CUDA核随机性。微型前置和正式回放共用该修复路径。本机只做禁Torch回归，远端修复后的前置/新实验尚未运行；不覆盖旧失败目录，不自行同步或重试。

<!-- archive:trainer-007:end -->

<a id="trainer-008"></a>
<!-- archive:trainer-008:begin -->
## 固定B0首轮前置结果（2026-10-22）

a6bea0a首轮远端前置1失败/2通过，真实回放未开始。第一份微型backward输出保存后Python RNG guard失败；NumPy/TorchCPU/CUDA RNG未变。静态导入链及已有状态的私有CPU推演支持首次lazy backend导入rich.style模块顶层getrandbits(24)副作用，但没有动态调用栈。原检查不放松，不把初始化消耗恢复掉；本轮无重试或源码修复，也无真实五次范围/长期因果结论。后续可另行将已安装扩展显式加载与审计置于测量前，不允许以额外未计数forward/backward作热身。历史实现定义及旧失败结果保留，新的尝试必须使用独立目录。

<!-- archive:trainer-008:end -->

<a id="trainer-009"></a>
<!-- archive:trainer-009:begin -->
## 固定B0五次重复（2026-10-22，仅实现，远端待验证）

`run_mcmc_fixed_b0.py`是另行获授权的窄回放入口，旧一级协议“不授权回放”的历史记录保持不变。只取`backward-boundary-20261022-v1`协议`d17d6f12…`的a臂，两相机各自原B0；校验完整源完成记录/比较身份、manifest/NPY/规范SHA、包/core/研究依赖与实际运行环境。CPU读取不恢复训练图，按原dtype/shape/stride/None/settings搬至对应GPU并回读核SHA；拒绝不支持的布局、非法索引、非有限和非本相机来源。两个独立worker各在同一进程内调用安装库原`_RasterizeToPixels.backward`五次，共十次，不调用forward/loss backward/通信/optimizer/strategy；无额外warmup或重试。保留原输出分配，检查输入/RNG不变、结果不与输入/尚存上一轮输出别名；记录当前worker RNG而不恢复历史RNG或掩盖副作用。微型前置用解析构造的一像素上下文，十次原backward另计，不混入实测预算。

每相机内部全十组i<j比较，j为参考，合计二十组/八十个张量比较；沿用atol=2e-6/rtol=2e-4，输出严格/bitwise/绝对相对误差、ULP、非有限、逐元素十配对差异与超容差频次及五次range。signed zero在ULP中距离为零，但位级差异独立统计。原生a/B1距离单列，不算第六次重复。频次和极值重复率不是独立Gaussian计数；只给本输入/环境/五次的观测范围，不证明全局有界，不追认旧Gate通过。

回放前固定owner0/223955、owner1/709801、owner1/1488267三个先前异常点，并为各点按同owner/可见相机模式、源A/A目标组P2/P3均在容差内筛选对照；以log1p总tile、log轴比、log四元数norm、log opacity平方距离最小且slot小者优先，排除已选点，无候选失败。六点不是无偏样本或因果匹配设计。独立CPU分析记录几何、两相机本轮B1五次变化，以及**历史一级A/A** P0/P2/P3/Adam矩/实际更新与float64标量公式；后者不是重跑Adam，更不把新B1拼成新的P2/P3传播轨迹。旧T1仅作提名依据另绑身份。本轮不回答MCMC长期因果，不自动改六分支验收。

RAM max/high/swap无限额、host_group=None，18GiB/卡reserved、20GiB磁盘准入/4GiB停止与6h上限、继承GPU lease和独立`image3d-fixed-b0-v1.service`保留。新代码仅Git手动同步，远端前置通过才冻结选点/运行；新输出目录不覆盖、不复用失败目录。完成仅`replay_complete`，非训练完成；本机只禁Torch轻量检查，真实CUDA与统计检查必须显式远端opt-in，不能将未执行记作通过。

<!-- archive:trainer-009:end -->

<a id="trainer-010"></a>
<!-- archive:trainer-010:begin -->
## Loss／rasterizer 反向边界一级实验（2026-10-22）

`run_mcmc_backward_boundary.py`复用冻结T1恢复/资源/比较流程，仅在独立进程作用域适配profile与worker，不改旧脚本、12文件核心或安装包。原库a/b各独立从同一12k恢复一步，保留P0—P4，新增严格顺序P0/P1/B0/B1/P2/P3/P4：B0是原`_RasterizeToPixels.backward`入口，保存全部十个ctx.saved_tensors（含colors、last_ids、交换后的投影输入和排序数组）、实际dL/dRGB与dL/dalpha及width/height/tile/absgrad/needs_input_grad；B1是同一原调用返回、反向通信前的v_means2d/v_conics/v_colors/v_opacities/v_backgrounds。原raster backward恰调用一次，两边界均CUDA同步并写独立CPU非pickle快照；检查快照前后live状态、输入/输出张量摘要及RNG不变，失败不恢复掩盖。

协议单独绑定边界脚本与B0/B1完整字段，完成记录绑定边界报告/快照SHA，缺rank/阶段/字段拒绝。归因要求P0/P1及B0保存输入与设置严格位级相同：图像梯度先不同只定位loss backward区间，不能直接称SSIM；B0完整输入位级相同而B1不同才定位rasterizer backward；B0/B1相同但P2不同则定位下游；内部保存输入不同或RNG异常不做rasterizer归因。仍报告严格不等及既有2e-6/2e-4数值误差，不能用输入容差内替代位级相同。

资源边界沿用T1：RAM无限额、18GiB/卡reserved、6h/阶段、快照磁盘准入与4GiB停止、GPU继承租约、独立unit `image3d-backward-boundary-{a,b}-v1.service`。只做一级两臂一步，不关闭SSIM/改Adam/通信/确定性，不自动执行loss-only或fixed-input replay，不恢复旧六分支。正常完成为probe_complete而非完整训练，无Validation评价/final checkpoint/Test解码；源码同步仅Git，真实执行前必须通过微型原生边界检查。本条是实现与执行定义，不表示真实结果已通过。

<!-- archive:trainer-010:end -->

<a id="trainer-011"></a>
<!-- archive:trainer-011:begin -->
## A/A First-Divergence T1（2026-10-21，仅实现）

旧Gate 1实测在原库control_a/control_b的12001步末参数/Adam严格比较失败，完整observed未启动、Gate 2被阻断；原六分支保持停止，不以新判据追认旧门禁通过。`run_mcmc_first_divergence.py`与`mcmc_first_divergence.py`提供独立原库T1：两独立进程组a/b各从同一已验证12k完整状态恢复，只执行12001一次更新（每臂1更新/2相机样本，合计2/4），名义30k/LR/SH/噪声/优化器配置不变。不提供T2/T3或自动串行追加入口，不加入贡献渲染、cohort、消融或确定性设置。

P0在原生恢复后、首次相机采样/forward之前，保存参数、逐组Adam状态/param_groups、策略、原相机order/cursor、RNG及梯度存在性；P1在实际backward调用之前保存原RGB/alpha/GT、真实scaled backward loss（loss/world_size）、原重建/正则loss项、实际相机/SH/LR及renderer元数据；P2在原backward返回且任何Adam step之前保存六组真实梯度/存在性、参数与Adam摘要和RNG；P3在全部六个Adam step完成后、策略之前保存参数/逐组Adam/策略/RNG；P4在原策略及位置噪声完成后保存状态。12001不应refine或变点数，违反即失败。方法包装恢复原引用，backward和策略恰调用一次，每个优化器恰step一次，不另做反向或重算训练。

每阶段先同步当前CUDA设备，逐张量detach并独立复制到CPU，再写非pickle NPY及带dtype/shape/原device/stride、规范SHA、文件SHA/bytes的JSON manifest；不保留所有阶段的GPU clone。快照前后RNG和live state摘要不变，不能恢复RNG掩盖干扰。逐组比较means/scales/quats/opacities/sh0/shN以及exp_avg/exp_avg_sq/step，缺必需字段/文件、阶段、rank或SHA不匹配不得验收。运行环境记录Torch/gsplat/CUDA/cuDNN/NCCL、GPU、确定性/TF32标志及实际Adam类/defaults/param_groups；记录配置不等于证明内部内核选择。不得打印全量环境或凭据。

方向固定A=a、B=b，浮点逐元素判据为abs(A-B)<=2e-6+2e-4*abs(B)。分块float64统计max_abs_error、mean_abs_error（有限对分母）、max_rel_error、超容差数/比例、首个超容差和严格不等坐标、两侧/并集非有限数、规范SHA相等、数值严格不等及逐元素bitwise不等数（保留+0/-0差异）。相对误差以abs(B)为分母：0/0记0，B=0而A非零记unbounded并以null+计数表示，不能用大相对误差替代逐元素容差。NaN/Inf即使两边相同仍失败；整数/bool/索引、RNG、Adam step、结构和梯度存在性严格比较，不广播。输出位级一致、数值等价但非位级一致、超容差/非有限或结构失败，分别给最早非位级及超容差阶段，辅助renderer元数据与主数值流分开。P0不严格相同标restored_state_mismatch，禁止将后续首差当可靠阶段定位；缺梯度即使两臂相同亦不能通过。

P4写完后使用专用ProbeComplete信号结束探针，worker只捕获此专用类型，其他错误保留；不执行完整Validation/checkpoint/模型合并，不伪造TrainingResult、final checkpoint或best。原生恢复可能读取Validation相机/懒加载视图元数据，但不评价或解码Validation RGB；Test仅允许原有replay/合同完整性字节校验，不解码/训练/渲染/评价。输出probe_complete仅表示五阶段一步快照完整，不是正常训练生命周期完成。新增同步可能改变调度，未复现只能报告同步探针未复现；不用于吞吐基准或宣称旧问题已修复，不把首次差异阶段直接解释为具体算子因果。

prepare/run/worker/compare均受指定远端主机/项目及各自显式授权限制，本机不能运行真实张量快照或比较。父attempt/checkpoint复制到独立目录，无可写别名；协议绑定源pair/core/env/package/config/replay、研究runner与依赖SHA，不改12个冻结核心。运行保留GPU空闲/继承租约、独立systemd cgroup、RAM无限额/host_group=None、18GiB每卡reserved、6h上限、20GiB准入/4GiB停止，并按父组件大小保守预留两臂快照空间、逐张量写入再检查磁盘。不重试、不覆盖旧产物、不改公开默认/通信；代码开关不代替用户执行授权。本次只有实现、禁Torch轻量测试及未运行的远端测试代码，真实T1仍not_run。

<!-- archive:trainer-011:end -->

<a id="trainer-012"></a>
<!-- archive:trainer-012:begin -->
## MCMC 生命周期观测 Gate 1／2（2026-10-21，仅实现）

`run_mcmc_lifecycle_trace.py`与`mcmc_lifecycle_observer.py`实现`mcmc_lifecycle_observer_v1`。本阶段仅实现与本机禁Torch轻量测试，不意味着任何真实恢复、CUDA观测一致性或机制结论已通过。Gate 3不提供自动入口，源仍是已冻结的apartment原库/尺寸修复各自12k完整checkpoint。

固定窗口12001—12200，名义30000；每环境control_a/control_b/observed三个独立恢复分支，共6分支、1200次新更新。所有分支从对应12k源恢复，不能接着上一分支末态；原库与尺寸修复checkpoint不能互换。12100建立目标/来源/未参与三组，追踪到12200第二次relocation之前的100次优化；12200原操作仍正常完成。每rank保存一次owner/slot/event来源与目标映射，来源reuse不包含原点自身，公式总副本数另为reuse+1。

独立入口临时包装原策略、原sampler、原renderer及优化器实例钩子，原操作只调用一次，不改12个冻结核心文件或依赖包。共同审计比较每步参数、Adam矩/step/参数组、策略、相机order/cursor、RNG及真实来源抽样的规范内容摘要；先检验control_a==control_b，再检验control_a==observed，任一不一致明确失败，不以PSNR或放宽容差替代。first_render审计处于首次恢复渲染之后、backward之前，相机已消费12001批次；不将此记录冒充未推进游标的12k状态。完整checkpoint再与保存时的真实末态逐字段摘要比较，排除序列化时间和封装字节差异。

observed记录各组opacity、最短/最长轴、逐点kappa和Adam分布；区分搬迁后未存活、存活后首次死亡、曾死亡后恢复及再次死亡。图像梯度以总梯度减去局部正则的解析梯度获得，明确logit/log-scale坐标和world_size因子，属于浮点残差，并非独立精确梯度测量；原backward与grad不被替换。按最终存活类别附加100步图像梯度、正则、Adam矩和logit更新暴露统计，相关性不当清零动量的因果证明。

完整诊断从Gate 1开始即包含每步额外no_grad的六通道owner×cohort标签渲染：几何/opacity/相机/遮挡保留，标签替代颜色，累计T·alpha作为覆盖贡献，而非图像质量提升。各通道和须对齐原alpha，辅助alpha亦须对齐原alpha，预声明atol2e-6/rtol2e-4；任一失败不回退到K代理。该检查仅为贡献计算守恒容差，训练状态和采样一致性仍严格相等。cohort尚未建立时所有点归未参与组并明确标记，不与事件后的组统计混算。不声称已有每点投影半径或逐点合成贡献，当前输出为owner×cohort累计权重。

Gate 2只读Gate 1 observed日志，核验六分支完成记录、每rank/每步/两次来源抽样及诊断文件SHA、身份映射和完整100步窗口。Gate 1未通过或文件缺失/改变时禁止可信Gate 2报告；其输出状态为observed_not_causal_intervention。报告只研究已形成不对称后的维持机制，不推断早期首因。

未来准备需`--prepare --authorize-prepare`；未来执行需`--run --authorize-execution --arm ... --branch ... --protocol-sha256 ...`及正确远端/独立systemd unit。内部worker还须继承GPU租约。每分支unit名`image3d-mcmc-lifecycle-{arm}-{branch}-v1.service`，RAM max/high/swap无限额、18GiB/卡reserved、6h阶段上限、磁盘20GiB准入/4GiB停止及额外checkpoint空间预算保留；监控显式从12000计数，新200更新/400样本不冒充12200新更新。禁止自动重试、改算法、运行Test、SOR、final-fit或Gate 3。`--gate1`与`--gate2`仅分析既有文件；源码中的执行开关不是用户授权的替代品。本轮不运行上述准备/执行命令。

<!-- archive:trainer-012:end -->

<a id="trainer-013"></a>
<!-- archive:trainer-013:begin -->
## MCMC 稳定性修复与单因素验证（2026-10-13）

用户授权按诊断、重定位数值门禁、RGB 损失、目标动量、位置学习率的顺序修复与实验，每项独立提交。实验只在 i-94B8D131 的指定项目内，经 Git 同步，在独立 outputs 目录执行；现有两个 apartment Job、生产依赖、公开默认和 Test 保持不变。场景对照从 MCMC Job 的冻结 replay fresh 初始化，seed20260729、2048、双卡、全局3M、名义30k日程保持一致；3k无完整优化器/RNG，不能伪称精确恢复。旧 AbsGrad 计划不是本轮执行计划。

首项仅补诊断：MCMC 第1步、每次配置的 refinement 间隔、Validation步和末步收集逐rank的小摘要，明确优化器更新后/策略前及重定位、增点、噪声后的作用域；记录存活/低透明度数量、opacity/max-axis-scale分位数、最大原点距离、位置LR和本rank相机的原始RGB范围。Validation另有按rank排序的参数摘要；历史顶层参数健康字段仍是rank0，不冒充全模型。所有新评估逐视角增加 `rgb_range` 的越界通道计数和极值，raw/display分数及选择逻辑不变。统计不消耗RNG、不修改张量/优化器，不把96%低透明度擅自新增为硬失败。

其余修复只有通过对应数值与质量对照后才可判断有效，不能一次混改后归因。隔离重定位修复必须匹配包装层/内核并绑定源码和二进制身份；保留旧目标动量是参考实现共有行为，其清零只作为单独实验变体。已有内部 `loss.clamp_render=false` 和 `learning_rate.position.delay_multiplier=1.0` 分别用于单字段对照，不改公开 schema10 默认。位置LR对位置噪声的联动必须披露。完整场景实验的启动、资源门禁和真实结果随后记录，不以准备完成冒充训练完成。

第二项的最小隔离实现为 `scripts/prepare_gsplat_mcmc.py`：核对安装包三个文件的审计SHA，只复制包到全新目录，保留原 `csrc*.so` 光栅化二进制；独立编译重定位小扩展，而非重编整个光栅化器。包装层向扩展传递实际 `min_opacity`，内核在尺度计算前执行上下限；`relocate`/`sample_add`均显式传参，旧动量处理不变。每次编译最多一个并行任务，不修改生产依赖或全局编译缓存。`overlay.json`绑定全部包源码和二进制，训练provenance检查实际文件集合与SHA后将其身份加入environment hash，旧环境不能冒充修复态或跨环境resume。上游公式依据是2026-10-12归档的RelocationCUDA.cu SHA `9f77224f…`，不宣称某已发布版本含修复。

`scripts/smoke_mcmc_stability.py`仅用于获授权远端双卡验证：原库诊断态与显式 `--relocation` 修复态分开执行；验证健康/96%退化两分片均可见且不推进CUDA RNG，6个重定位公式输入（含ratio超上限）、两条拓扑入口各两个opacity下限、无floor时与原二进制对照（atol2e-6/rtol2e-4）、SH3/64px/60点/12更新的实际分布式trainer、增点、Validation、末态checkpoint和合并。输出独立且拒绝覆盖；这些不是apartment质量证据。当前仅源码SHA/锚点、AST、标准库补丁回归和Ruff通过，CUDA编译/实际GPU运行尚未通过，禁止据此启动后续场景臂。

2026-10-13远端验证更新：用户手动拉取后确认i-94B8D131为79eae01，trainer/evaluation/config/隔离补丁共81项回归通过（3.23s）；此前旧版76项不算新实现验收。两L2空闲、compute capability8.9，CUDA12.2工具链已存在但不在默认PATH。提交MCMC诊断与重定位独立GPU门禁-v1（原库双卡smoke→隔离编译→修复态smoke，8GiB MemoryMax、禁用任务swap、30分钟上限）时被权限层要求本轮精确主机/任务授权而拦截。只读确认输出目录/once均不存在、unit未创建、无GPU计算进程、后端active；未换通道重试，CUDA与apartment实验仍未运行。证据为outputs/analysis/mcmc-stability-implementation-20261013-v1/remote-regression.json。

2026-10-13 GPU门禁已通过：用户精确授权后，执行任务20261013-113308-8473、只读监控20261013-113419-54ef，代码79eae01，受限unit运行56.526s/exit0。原库和隔离修复库两L2均完成SH3/12更新/60→66点、增点、Validation、末态checkpoint与合并；两rank健康统计准确报告[0,96]个dead且不推进CUDA RNG。修复态每rank6组公式、4组拓扑/下限检查通过，尺度最大绝对误差1.862645149230957e-8，无floor时与原二进制在预声明容差内一致。原库/修复态summary SHA分别1e713f9e…/b1a1012f…，隔离overlay身份f034e1f3…，产物outputs/analysis/mcmc-stability-gates-20261013-v1。面板外层命令提前exit使GP结束标志缺失，列表STOP不能当作失败；以独立unit成功、exit-code=0及两份完整报告交叉验证，不伪造GP状态。此为实现门禁，不是apartment质量通过；场景单因素实验仍须fresh冻结配置和资源保护。

独立 `--variant destination_moments` 只在SHA锁定的原版ops中增加目标索引的优化器状态清零，不编译新CUDA、不应用floor修复、保留step及无关行。环境身份为 `mcmc_destination_moments_overlay_v1`；训练入口拒绝同时启用两个overlay标记。`smoke_mcmc_stability.py --destination-moments`验证六参数组搬迁目标在5次零梯度更新中不漂移、未搬迁行仍有动量、step从100到105，并运行同一12步trainer。它是算法消融，不是默认行为或参考实现缺陷的既定结论。

场景对照入口 `scripts/run_mcmc_stability_ablation.py` 只服务冻结apartment Job，复用既有runner、replay、GPU租约、逐rank遥测、TrainingMonitor、SOR与Validation，不新增训练平台。五臂为baseline/raw_loss/relocation_floor/destination_moments/position_lr：raw_loss和position_lr复用既有单字段override，库变体保持全部配置相同；各臂fresh、30k/60k相机样本，严格同camera-sequence SHA、同training core，需三种环境当前core的双卡门禁。准备阶段先冻结输入/配置/overlay/gate/入口哈希，再按臂运行，baseline完成前拒绝其他臂。MCMC显式允许既有 `--resource-telemetry-dir`，Project原限制不放宽。

每场景臂使用独立systemd unit，主存硬上限8GiB/MemoryHigh7GiB/无swap，14GiB主机空闲准入、运行中6GiB底线、cgroup7.5GiB停止、两卡各18GiB reserved上限，全局3M不变；每stage6小时，unit6.5小时，磁盘20GiB准入/4GiB停止。MemoryHigh先回收缓存，不以额外剪枝或降分辨率避资源失败；失败保留证据，不自动增预算或重跑。先研究主训练失稳并统一SOR/held-out Selection，不做新final-fit或发布模型；结果报告raw/display/尾部失败视角/RGB越界/各rank终态与资源，不自动决定推广。所有源码先提交经Git同步，本机只运行无Torch的配置/AST回归。

<!-- archive:trainer-013:end -->

<a id="trainer-014"></a>
<!-- archive:trainer-014:begin -->
### 退化修复第二轮：诊断先行（2026-10-17）

用户授权继续验证和修复。先修复unpacked双卡屏幕健康统计：使用gsplat元数据中all-to-all前的本分片投影，沿相机维归并可见性和最大屏幕半径；不把全局渲染buffer误作本分片索引。packed路径排除零半径。新增`screen_health_views`标明有诊断数据的视角数，0表示不可据此判断真实可见性；`screen_health_scope`明确本分片/参与投影相机作用域。统计不修改模型、RNG、渲染结果或选择指标。此诊断修复改变训练核心身份，旧core的完整实验与门禁只保留为历史证据。新张量回归、双卡验证未运行前，不宣称修复已获远端验收，不推广生产默认。

每个MCMC attempt新增独立`artifacts/mcmc-rank-N.jsonl`，在已有诊断频率的策略前/后写入本卡摘要，后摘要必须早于参数校验和全卡汇总。增加最小轴/最小log尺度及零/次正规/NaN/Inf分类；仅有限值参与分位数。成功渲染记录实际瓦片交集数K及24K索引数组估算（不含排序工作区），不是失败帧预分配统计。异常记录本rank阶段、步数和原始cause栈，禁止为记录错误再分配CUDA张量或执行collective；磁盘记录失败不能替换原训练异常。不改变优化器、搬迁、噪声、阈值或预算。

`scripts/smoke_mcmc_degeneration.py`是第二轮的远端复现门禁，不是算法候选或完整场景训练。原库/旧floor/旧moments分别在全新目录运行；实际100步Adam正则更新覆盖step0与9300，184真实副本覆盖公式上限，40轮固定低透明度公式压力覆盖尺度动态范围，复用已有12步双卡trainer/Validation/checkpoint。进程内钩子仅用于该小门禁，记录钳制前的真实ratio输入（非unique-source分布）、瓦片分配前分块包围盒估算和分配后的实际K；小例中逐项核对估算/实际、原函数输出和RNG，不宣称任意未来CUDA输入均位级等价。报告绑定脚本/复用smoke哈希、训练核心和overlay环境。期望终态为`diagnostic_complete`且`stability_gate=blocked_by_reproduced_failures`，不能被既有passed门禁替代。运行需专用`image3d-mcmc-degeneration-20261017-v1.service`，主存max/high/swap均无限额，GPU租约、空闲检查、单卡18GiB峰值检查保留；调度仍必须经面板任务和只读监控、限时unit，禁止本机执行或直接用面板同步exec承载GPU任务。尚未远端运行，不能据此启动完整场景或推广候选。

第二轮补充`--mechanism-details`使用独立`image3d-mcmc-mechanism-details-20261017-v1.service`和全新输出目录。只运行两卡各自单高斯合成渲染及固定opacity的尺度压力，不执行真实场景或重复12步trainer。packed/unpacked各7个阈值上下样本记录真实图像/正则logit梯度与alpha/投影/K；这不是双卡全场景逐点梯度。尺度压力逐轮保存3轴输入/输出/GPU log及零/负有限/NaN/±Inf分类，JSON不写非标准NaN。所有观测先落盘再断言，不把预期下溢路径当观测；summary仍为诊断完成/稳定性阻塞，训练核心不变。

<!-- archive:trainer-014:end -->

<a id="trainer-015"></a>
<!-- archive:trainer-015:begin -->
### 来源容量准入候选（capacity-v1，2026-10-17）

此候选仅隔离研究，不改schema10或生产默认，也不与raw_loss、floor、目标动量或position_lr变体混合。`prepare_gsplat_mcmc.py --variant capacity`复制审计原包到新目录，以共享`mcmc_capacity_ops.py`替换relocate/sample_add入口，原CUDA公式和光栅化二进制不变；额外锁定strategy/mcmc.py的SHA，记录实际接纳数量而非请求数。完整包源码/二进制及候选helper均纳入overlay身份；trainer核心的身份/计数接入变化必须重新通过同core门禁，禁止复用旧passed报告作为新场景准入。

令`t=max(min_opacity,1/255)`，来源alpha有限且<1，容量由`1-(1-alpha)^(1/N)>t`严格不等式导出，并受原binomial表最大N限制。每次只做一批原opacity加权提议，按来源容量稳定接纳并保留提议/目标对应顺序；不通过展开每个来源的容量槽占用巨量显存。内存O(points+requests)，排序O(requests log requests)，可能未用尽所有来源容量，不无限重采样。仅对实际获准来源按真实副本数计算公式。写入前检查logit往返仍高于t、尺度log与exp往返为正常正有限值；失败则拒绝整个来源组，绝不部分复制后改变N、强抬透明度、压尺度或修补NaN。未获准的死亡点及其优化器状态原样保留；增点可不足请求数，cap仍为上限而非必须达到的目标。

保持原动量语义：relocate仅清接受来源行、保留目标行；sample_add保留来源状态、新行置零，Adam共享step不变；正则/噪声/LR不改。`mcmc_relocated_count`记录实际全局接纳数，新增`mcmc_deferred_count`记录未完成搬迁。此候选只阻断不满足容量/数值约束的复制，不保证点长期高于cutoff、不解决跨rank来源池隔离，不保证任意尺度/相机的K或显存受限，需独立质量与资源对照。

新增CPU回归覆盖两条拓扑入口、183请求最多50新副本、严格阈值/饱和输入、原公式非法输出整组无写入、目标顺序和成熟Adam状态；CUDA `smoke_mcmc_stability.py --capacity`验证真实kernel/51总副本中心透明度、近死40轮无强迫收缩及既有双卡12步trainer完整生命周期。未获独立新候选GPU任务授权前不运行；小门禁通过也不等于30k稳定性或质量通过。

原库/capacity门禁通过`smoke_mcmc_stability.py --guarded`运行，固定在`image3d-mcmc-capacity-20261017-v1.service`及指定远端仓库；复用GPU租约/空闲准入、逐rank显存遥测和TrainingMonitor，运行中检查18GiB reserved/每臂300秒及最终12步完整性。外层unit900秒，磁盘20GiB准入/4GiB停止；主存max/high/swap无限额且host_group=None，无主存主动停止。输出仅新建研究目录，原库/capacity分别保留probe及资源/退出记录，子进程只执行既有小smoke。

<!-- archive:trainer-015:end -->

<a id="trainer-016"></a>
<!-- archive:trainer-016:begin -->
### Capacity 多周期受控验证（2026-10-17）

用户明确授权原库/capacity的三合成用例，每rank128点、每例最多1000更新，不含10k/30k/Test/生产变更。`run_mcmc_capacity_cycles.py`只在指定远端与`image3d-mcmc-capacity-cycles-20261017-v1.service`运行；复用已通过的同core两环境，不改core/配置默认。每rank4个来源初始opacity=.5，其余1e-6；两个64px固定合成相机与黑背景、可见点z=2，负对照全在z=-2，失衡仅rank1在相机后。模型/优化器及参考图像初始哈希须配对；这是刻意构造的困难状态，不是旧失败现场重放或held-out场景质量。

初始Adam step9300，opacity两矩设为稳态正则梯度及平方，其他矩为0；使用名义30k LR的9301..10300、原100步搬迁间隔、固定local cap128。核心正则乘128/1500000保持全局3M下的逐点正则量级，明确为虚拟归一化而非3M模型。两实现同初始seed，不承诺采样路径分化后的噪声轨迹一致。图像梯度统计为总梯度减去独立autograd正则梯度，可能有float舍入残差。记录前后健康、实际K/投影、可用来源容量、candidate接纳数与实际步数；来源耗尽按原alive门槛在搬迁前同步终止，不伪造剩余更新。固定点数排除通过少增点降低负载的混淆，但不证明大模型资源表现。

预声明探索性安全否决线：可见与失衡用例两实现均完成1000更新、两分片终态均有有效可见点、candidate原始合成PSNR相对原库下降不超过0.5dB；无图像信号负对照可耗尽来源，不把共同失败称为candidate新增退化。任何缺失/输入不配对/非有限或否决均阻止给出后续10k资格；即使通过，本脚本也不启动真实场景。保留六用例失败继续且不重试；逐case独立取消/日志/遥测目录，18GiB reserved/每例900秒、磁盘20GiB准入/4GiB停止、unit6000秒、GPU串行锁；RAM max/high/swap无限额，不设置主存保护。摘要的受控提前结束必须与逐行真实进度、耗尽rank、最终健康和显存遥测吻合。

<!-- archive:trainer-016:end -->

<a id="trainer-017"></a>
<!-- archive:trainer-017:begin -->
### 主存修复与重新基线（memory-v2，2026-10-13）

上述8GiB/7GiB策略保留为失败v1的历史合同，不再用于新实验。v1在10k Validation出现匿名内存与shmem合计超过memory.high、反复回收/缺页，最终遥测陈旧保护自动终止；不是OOM或质量失败，不恢复无optimizer/RNG的best模型分片。用户明确授权先修复主存、验证后重新启动基线并放宽内存限制。

运行时新增仅释放已解码图像的`TrainingViews.clear_cache()`，保留sources/相机/采样顺序；主训练进入Validation前清Train缓存，验证返回后清Validation缓存，不使用会销毁sources的`clear()`。健康统计由CPU张量直接转NumPy，避免百万Python浮点对象，保留原float64/分位数算法及数值口径。验证每个视角加载前记录完成数/总数，分开标记健康统计与逐rank汇总；只有新MCMC协议启用300秒无Validation进度停止门禁，120秒遥测陈旧与原有保护不关闭。缓存/RNG/指标等价由远端回归验证，不宣称释放对象必然等于RSS即时归还。

新profile为`apartment_mcmc_single_factor_memory_v2`，unit后缀`20261013-v2`、全新输出目录与once。主存硬限12GiB、`MemoryHigh=infinity`（协议记null、cgroup须为max）、SwapMax0，18GiB整机空闲准入、6GiB运行余量、11.5GiB任务主动停止；逐卡18GiB/global3M、30k/2048/seed及五臂单因素保持不变。12GiB上限依据约19.4GiB现场可用及6GiB共享主机余量，不能将“尽可能放宽”解释为取消所有保护；运行中不自动加预算或重试。

新核心必须重跑三环境小型双卡门禁，并通过`scripts/smoke_mcmc_validation_memory.py`：读取旧MCMC 30k检查点的哈希验证模型分片（仅作资源负载，不续训），实际3M点/SH3/2048、分配Adam两矩，每rank三次完整351视角Validation；每轮先暖Train图像缓存再释放，检查缓存释放后sources仍可用、CUDA RNG不变、各轮逐视角PSNR差<1e-4、每轮<600秒且任务内存低于11.5GiB。父控制器复用run_stage/cgroup/遥测/进度门禁，失败不生成passed。此负载不是失败10k堆的完全重演，也不是新场景画质证据。准备新协议必须绑定当前core的内存门禁summary及三环境报告，旧v1报告不能替代；源训练/config/replay、协议和失败记录均保留，重启必须fresh。完整基线还需跨过10k、30k保存、SOR和Selection才能宣称完成，不保证任何未来负载绝不失败。

2026-10-13验证结果：42c26f6远端161项回归通过；任务20261013-162310-eedc/只读监控20261013-162445-be3a正常exit0。三环境两rank均通过新core `55ea2ce2…` 门禁。真实3M/SH3/2048、Adam两矩常驻的三轮351-view验证全部完成，逐轮最慢rank耗时46.818/45.377/45.275秒，每rank每轮实际释放Train缓存536205312bytes，逐视角PSNR跨轮最大差0、CUDA RNG不变。2秒采样任务主存峰值6724980736bytes约6.26GiB，high/max/oom/oom_kill均0，逐卡reserved峰值3391094784bytes；这是代表负载门禁而非原10k堆的严格前后对照或完整30k稳定性证明。新协议已冻结在`outputs/analysis/mcmc-stability-ablation-20261013-v2/protocol.json`，SHA `05f55c4bada35cfc7e5c4c35b1c63eb41f15f24e1556054cd71d863184568d33`，内存summary SHA `561292e7…`；独立记录后才启动新基线，不为文档提交改变远端核心。


<!-- archive:trainer-017:end -->

<a id="trainer-018"></a>
<!-- archive:trainer-018:begin -->
### 用户明确取消任务主存限制（unlimited-host-memory-v3，2026-10-16）

用户在raw_loss最终checkpoint因11.5GiB主动停止后，明确要求取消12GiB硬限及所有任务主存保护；此授权取代上文memory-v2的主存资源政策，不重写v2协议、预算、失败结论或产物。无主存限制可能触发整机OOM并影响面板/其他服务，不能承诺只有训练进程会被杀；不改主机swap配置或其他服务限额。

`scripts/run_mcmc_stability_ablation.py --prepare --unlimited-host-memory`只在新的输出目录生成`apartment_mcmc_single_factor_unlimited_host_memory_v3`协议，unit后缀`20261016-v3`。host policy的`memory_max_bytes`、`memory_high_bytes`、`memory_swap_max_bytes`、`startup_available_bytes`、`minimum_available_bytes`、`stop_current_bytes`全部记null；实际启动必须显式`MemoryMax=infinity`、`MemoryHigh=infinity`、`MemorySwapMax=infinity`，运行入口核对本unit及祖先cgroup没有继承的有限max/high/swap限制。无主存限额不等于物理内存无限，不关闭内核OOM机制。

新模式不向`run_stage`传host_group，因此不调用`host_failure`，不再按18GiB启动余量/6GiB运行余量/11.5GiB任务总量提前拒绝或终止，也不生成原受限模式的内部host-memory时间线；需要内存观察时使用独立只读面板监控，不将观测重新接成停止条件。原CUDA遥测、逐视角进度、显存/磁盘/时长保护与GPU串行租约仍在，它们不是本次取消的主存限制。新模式通过协议显式选择，不能用run时开关覆盖已冻结v2。

此改动仅在研究runner，不改trainer/absgrad_resources或训练核心SHA；三环境GPU门禁及3M Validation报告仍验证同一核心，但受限内存门禁是历史执行证据，不是新无限额政策的安全证明。新协议重新绑定runner及所有输入/config SHA，仍不读Test RGB、不改线上默认/模型、不执行新final-fit。取消主存限额本身不授权重试已失败臂、拼装未验证checkpoint或覆盖旧现场。已失败raw_loss的八次验证是SOR前质量证据，不能标成完整实验通过。


<!-- archive:trainer-018:end -->

<a id="trainer-019"></a>
<!-- archive:trainer-019:begin -->
### 串行总控与失败继续（2026-10-16）

用户随后明确要求一次性串行跑完任务、最后统一查看，并指定“如果失败就跳到下一个串行任务”。这覆盖此前每臂失败停止续接/等待人工审查的调度规则，并授权本次新目录fresh重跑；不是旧目录的自动重试或checkpoint恢复。`scripts/run_mcmc_stability_suite.py`在独立suite目录准备新的无限额v3协议，顺序为baseline→raw_loss→relocation_floor→destination_moments→position_lr；重跑新基线以保持五臂同资源政策和同冻结身份，旧成功/失败臂不覆盖、不伪装成新协议结果。

总控持专用flock和一次性标记、每臂同步等待独立systemd任务；单臂继续复用GPU租约和显存/磁盘/时长保护，RAM/high/swap均infinity，没有主存主动停止线。每臂前重新查GPU；有残留占用则该臂记失败，不强行并行或杀其他进程。基线不可用时依赖它的候选记录skipped/matched_baseline_unavailable，不绕过基线门禁。单臂非零退出或完整性核验失败记录后继续下一臂，不重试；整个主机OOM、总控被杀或协议准备失败仍可能阻止后续调度，不能承诺故障后总控永不退出。

每个成功臂在继续前自动核验30k/60k/world2、八次351-view Validation、相机序列与新基线一致、完整最终checkpoint六组件流式大小/SHA及元数据SHA、三阶段退出结果、模型和Selection评估SHA；不把GPU利用率、模型分片或退出码0单独当成功。每臂review及日志即时独立落盘，结束写suite.summary.json（含相对新基线的质量差值）与suite.exit.json；有任一失败/跳过时最终返回1，但不是在首个失败时停队列。suite和单臂入口都绑定进新协议，训练核心保持不变。总控运行在远端面板任务，不依赖本地会话续接；另外只有一个独立只读监控。最终集中审查时仍按每臂分别归档/提交结果，不在训练中修改远端Git或发布模型，不读取Test RGB、不做final-fit。

<!-- archive:trainer-019:end -->
