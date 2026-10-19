# Project-owned Gaussian trainer and initialization contract

Stage 2C owns the 3D Gaussian model and optimization lifecycle. Runtime code must not invoke Nerfstudio, Splatfacto, or another full external trainer. The only external training primitive is the R2.2-pinned `gsplat==1.5.3+pt23cu121` rasterizer.

## Model and rendering

`image3d_scenegraph.gaussian.model.GaussianModel` owns these learnable tensors:

- normalized-space means `N x 3`;
- log scales `N x 3`, activated with `exp`;
- quaternion rotations `N x 4`, normalized before rendering;
- opacity logits `N`, activated with `sigmoid`;
- degree-3 spherical-harmonic coefficients `N x 16 x 3`.

Optimizer groups are position, feature, opacity, scaling, and rotation. Their effective rates, exponential schedules, SH schedule, loss weights, mutation cadence, and budgets come from the R2.4 resolved configuration. The renderer boundary is `gaussian.render.render_gaussians`; no project module imports an external trainer. `RenderCamera.camera_from_normalized` is a rigid transform in normalized camera units: dataset similarity scale is removed before rendering, and gsplat receives explicit normalized-unit near/far planes (`0.01` / `1e10`).

Training samples one deterministic train view per iteration, uses project-owned L1 + differentiable SSIM, validates finite gradients/parameters, and selects either the frozen `default_v1` or `mcmc_v1` strategy from the hashed effective configuration. Before CUDA/distributed launch, `gaussian_geometry_readiness_v1` records robust camera-center, normalized-point, 3NN-scale, and bounded projection-risk evidence. A single camera simultaneously at least 100× median and 10× p99 center radius, or a 50% 3NN-floor scale collapse, fails with stable reason codes. `run_gaussian_training.py --readiness-only` performs this preparation/gate and exits without launching a trainer. CUDA OOM, cancellation, non-finite loss/gradient/parameter, invisible initialization, and unsupported input remain explicit failures rather than successful partial output.

`default_v1` otherwise retains the Project v7 behavior and gsplat `DefaultStrategy`, including configured densification/pruning/reset and no hard Gaussian cap. Its existing `pruning.max_world_scale` rule is now applied once before the first render as well as by later strategy refinement, so a few oversized seeds cannot allocate tiles during the first 500 iterations; the same deterministic mask applies to fresh and frozen initialization. Experimental `mcmc_v1` uses the installed Apache-2.0 gsplat `MCMCStrategy` inside the same model/optimizer/checkpoint/Validation lifecycle. It adds activated opacity and scale mean regularization, method-specific initialization, per-step position noise, and relocation/growth refinement. Its 3,000,000 cap is global: each distributed rank receives an exact quotient/remainder share. Default pruning, opacity reset, and recovery-prune are disabled, including the pre-render Default-only scale mask. Before each refinement, a synchronized live-opacity guard fails every rank with counts and opacity quantiles if any rank has no relocation source, avoiding gsplat 1.5.3's empty multinomial CUDA failure.

COLMAP `SIMPLE_RADIAL`, `RADIAL`, and `OPENCV` distortion is explicitly inverted by differentiable image resampling before pinhole rendering. This is necessary because gsplat 1.5.3's distortion UT projection is documented as non-differentiable to all inputs. Distortion is never silently discarded.

The trainer loads only `train` and `validation` IDs and asserts that no `test` ID enters runtime. LPIPS remains `not_run: dependency_not_audited_in_r2_10`; R2.11 owns dependency audit and final isolated test evaluation.

The 2026-09-15 UHD extension permits an explicit longest edge through 3840px while keeping 1280px as the default. Runtime view collections retain ordered camera/path metadata and decode RGB on demand into a 512 MiB byte-bounded LRU cache per collection, rather than retaining every decoded image per rank. A frame larger than the budget is returned uncached; active frame tensors and model/optimizer memory are outside the cache budget. Header inspection, metadata-only ID/strategy checks, slicing and Train+Validation concatenation do not decode or retain the whole dataset. Pixel conversion, LANCZOS resize, scaled intrinsics, distortion resampling and uint8-to-float transfer remain unchanged. Cache state consumes no RNG, is absent from checkpoints, and is rebuilt after resume; dataset/config/code/environment checks still apply. The same loader serves main training, bounded final-fit and evaluation. `--no-intermediate-previews` is an explicit research runner option: every scheduled Validation metric and selection decision remains, but training PNGs are saved only at the last iteration. Ordinary calls retain intermediate PNGs.

Gaussian evaluation schema 2 preserves the historical top-level and per-view `psnr`/`ssim` fields as the primary `raw_float_v1` profile: metrics are computed directly from the unclamped gsplat float render and the float reference. It additionally reports `display_psnr`/`display_ssim` under `display_clamped_uint8_v1`, applying the preview path independently to prediction and reference (`clamp(0,1)`, multiply by 255, floor to uint8, divide by 255) before scoring. `quality_profiles` records both identities. These profiles are parallel reports: display metrics do not replace raw metrics, change Validation selection, or rewrite historical evaluation artifacts. Frozen-candidate and Test-consumption records remain schema 1 so previously frozen candidates retain their authorization identity.

## 异尺寸相机分布式一致性诊断（2026-10-19）

用户批准指定远端原安装gsplat的独立小门禁，不修改生产渲染器/MCMC/通信配置。`smoke_distributed_camera_sizes.py`用9个确定性合成用例，每例16全局Gaussian、SH3、5次fresh Adam更新；覆盖同尺寸64×64与1361×2048、历史宽差1361/1372×2048、放大64×64/96×80、相机交换/所有权交换、不等长7/9分片及一分片不可见。历史用例仅复用尺寸，K/pose/点/目标均合成，不是apartment重放。固定全参数、相机、目标、环境/core和脚本哈希，原包源码/二进制前后校验。

单卡完整模型采用dense非分布式路径，每相机L1+SSIM损失除2累积后只做一次Adam更新；双卡每卡一图、unpacked且损失除2，不使用DDP额外平均。沿用clamp规则，噪声、拓扑与正则关闭；首步额外对比公开单卡packed路径与dense参考。逐rank在collective前保存raw RGB/alpha、六组梯度、每步参数/Adam两矩/step误差，主判据atol=2e-6/rtol=2e-4，非有限不得填零冒充通过；报告边缘32px及内部超容差元素数。首步图像/梯度与后续轨迹误差分开，不能由Adam微小梯度放大单独归因通信。

同尺寸或packed参考失败标control_blocked；缺用例/缺rank/缺更新标incomplete；数值不一致仍执行剩余用例并最终非零退出，只有全部完整且一致才passed。真实子进程/资源失败独立记录、不自动重试。专用unit image3d-camera-size-equivalence-20261019-v1.service和新outputs/analysis目录，唯一只读监控；每例300秒、unit3600秒、每卡reserved18GiB、磁盘20GiB准入/4GiB停止、GPU租约；RAM max/high/swap无限额、host_group=None。新代码只Git同步，本机禁止Torch；不读取Test/RGB或checkpoint，不启动10k/30k，不自行修复或推广。门禁不证明该问题解释全部长期退化。

### 排序隔离 order-v2（2026-10-19）

上一轮legacy-v1控制失败保留。新profile按rank拼接顺序建立全模型参考，并以显式全局ID逆映射比较所有参数、梯度和Adam状态；canonical顺序用于内容hash以及equal_small首步排序敏感性观测，该附加观测不替代主控制。原9例保持原定义，另增加equal_small/historical/mixed三个unique例：z=2+.05*i，xyz与尺度同比缩放，真实float32深度必须无并列。共12例，每例16点/SH3/5更新；边带为min(32,max(1,min(width,height)//4))，避免小图边带覆盖全图。主容差仍2e-6/2e-4，只有完整、所有同尺寸及packed控制通过且原库异尺寸仍不一致，才允许下一阶段隔离尺寸修复验证；控制失败停止，不自动补跑。专用image3d-camera-size-order-20261019-v2.service，新目录与唯一只读监控，unit4200秒、每例300秒，原显存18GiB、磁盘20/4GiB、RAM无限额和GPU锁不变。本轮尚未修改安装gsplat、生产入口、MCMC或通信配置；后续补丁必须独立commit并保留原CUDA二进制，不直接推广。

### 目标相机尺寸隔离修复 repair-v1（2026-10-19）

仅order-v2完整且全部同尺寸/packed控制通过、异尺寸仍不一致时允许新候选。`prepare_gsplat_camera_sizes.py`锁定原rendering.py SHA，复制原包并只修改Python渲染层、保留所有CUDA二进制。分布式汇集每rank宽高；每rank一相机/dense/普通pinhole EWA范围内，同尺寸保留原投影调用，不同尺寸逐目标viewmat/K及对应width/height调用原kernel并可微拼接。禁止用max画布或resize代替目标尺寸语义；未覆盖模式明确拒绝。不触碰原安装包、生产核心、公开默认和其他overlay。

repair-v1使用相同12例/16点/5更新、相同输入与同序参考/容差，独立记录base environment及补丁profile/全文件SHA，不冒充原库身份。基线repair_ready、原包SHA、canonical初始参数/目标/K/pose、参考ID/optimizer/core均须配对；每rank核验实际导入路径。任一数值/资源/身份失败保留并停止余例，缺失记录维持未完成，只有全矩阵完整通过才passed。专用image3d-camera-size-repair-20261019-v1.service、新实验目录和唯一只读监控；每例300秒/unit4200秒、显存18GiB/磁盘20-4GiB/RAM无限额/GPU锁保留。不推广生产、不读Test、不重跑场景或10k/30k；小门禁不当真实场景长期质量与3M性能证明。

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

### 退化修复第二轮：诊断先行（2026-10-17）

用户授权继续验证和修复。先修复unpacked双卡屏幕健康统计：使用gsplat元数据中all-to-all前的本分片投影，沿相机维归并可见性和最大屏幕半径；不把全局渲染buffer误作本分片索引。packed路径排除零半径。新增`screen_health_views`标明有诊断数据的视角数，0表示不可据此判断真实可见性；`screen_health_scope`明确本分片/参与投影相机作用域。统计不修改模型、RNG、渲染结果或选择指标。此诊断修复改变训练核心身份，旧core的完整实验与门禁只保留为历史证据。新张量回归、双卡验证未运行前，不宣称修复已获远端验收，不推广生产默认。

每个MCMC attempt新增独立`artifacts/mcmc-rank-N.jsonl`，在已有诊断频率的策略前/后写入本卡摘要，后摘要必须早于参数校验和全卡汇总。增加最小轴/最小log尺度及零/次正规/NaN/Inf分类；仅有限值参与分位数。成功渲染记录实际瓦片交集数K及24K索引数组估算（不含排序工作区），不是失败帧预分配统计。异常记录本rank阶段、步数和原始cause栈，禁止为记录错误再分配CUDA张量或执行collective；磁盘记录失败不能替换原训练异常。不改变优化器、搬迁、噪声、阈值或预算。

`scripts/smoke_mcmc_degeneration.py`是第二轮的远端复现门禁，不是算法候选或完整场景训练。原库/旧floor/旧moments分别在全新目录运行；实际100步Adam正则更新覆盖step0与9300，184真实副本覆盖公式上限，40轮固定低透明度公式压力覆盖尺度动态范围，复用已有12步双卡trainer/Validation/checkpoint。进程内钩子仅用于该小门禁，记录钳制前的真实ratio输入（非unique-source分布）、瓦片分配前分块包围盒估算和分配后的实际K；小例中逐项核对估算/实际、原函数输出和RNG，不宣称任意未来CUDA输入均位级等价。报告绑定脚本/复用smoke哈希、训练核心和overlay环境。期望终态为`diagnostic_complete`且`stability_gate=blocked_by_reproduced_failures`，不能被既有passed门禁替代。运行需专用`image3d-mcmc-degeneration-20261017-v1.service`，主存max/high/swap均无限额，GPU租约、空闲检查、单卡18GiB峰值检查保留；调度仍必须经面板任务和只读监控、限时unit，禁止本机执行或直接用面板同步exec承载GPU任务。尚未远端运行，不能据此启动完整场景或推广候选。

第二轮补充`--mechanism-details`使用独立`image3d-mcmc-mechanism-details-20261017-v1.service`和全新输出目录。只运行两卡各自单高斯合成渲染及固定opacity的尺度压力，不执行真实场景或重复12步trainer。packed/unpacked各7个阈值上下样本记录真实图像/正则logit梯度与alpha/投影/K；这不是双卡全场景逐点梯度。尺度压力逐轮保存3轴输入/输出/GPU log及零/负有限/NaN/±Inf分类，JSON不写非标准NaN。所有观测先落盘再断言，不把预期下溢路径当观测；summary仍为诊断完成/稳定性阻塞，训练核心不变。

### 来源容量准入候选（capacity-v1，2026-10-17）

此候选仅隔离研究，不改schema10或生产默认，也不与raw_loss、floor、目标动量或position_lr变体混合。`prepare_gsplat_mcmc.py --variant capacity`复制审计原包到新目录，以共享`mcmc_capacity_ops.py`替换relocate/sample_add入口，原CUDA公式和光栅化二进制不变；额外锁定strategy/mcmc.py的SHA，记录实际接纳数量而非请求数。完整包源码/二进制及候选helper均纳入overlay身份；trainer核心的身份/计数接入变化必须重新通过同core门禁，禁止复用旧passed报告作为新场景准入。

令`t=max(min_opacity,1/255)`，来源alpha有限且<1，容量由`1-(1-alpha)^(1/N)>t`严格不等式导出，并受原binomial表最大N限制。每次只做一批原opacity加权提议，按来源容量稳定接纳并保留提议/目标对应顺序；不通过展开每个来源的容量槽占用巨量显存。内存O(points+requests)，排序O(requests log requests)，可能未用尽所有来源容量，不无限重采样。仅对实际获准来源按真实副本数计算公式。写入前检查logit往返仍高于t、尺度log与exp往返为正常正有限值；失败则拒绝整个来源组，绝不部分复制后改变N、强抬透明度、压尺度或修补NaN。未获准的死亡点及其优化器状态原样保留；增点可不足请求数，cap仍为上限而非必须达到的目标。

保持原动量语义：relocate仅清接受来源行、保留目标行；sample_add保留来源状态、新行置零，Adam共享step不变；正则/噪声/LR不改。`mcmc_relocated_count`记录实际全局接纳数，新增`mcmc_deferred_count`记录未完成搬迁。此候选只阻断不满足容量/数值约束的复制，不保证点长期高于cutoff、不解决跨rank来源池隔离，不保证任意尺度/相机的K或显存受限，需独立质量与资源对照。

新增CPU回归覆盖两条拓扑入口、183请求最多50新副本、严格阈值/饱和输入、原公式非法输出整组无写入、目标顺序和成熟Adam状态；CUDA `smoke_mcmc_stability.py --capacity`验证真实kernel/51总副本中心透明度、近死40轮无强迫收缩及既有双卡12步trainer完整生命周期。未获独立新候选GPU任务授权前不运行；小门禁通过也不等于30k稳定性或质量通过。

原库/capacity门禁通过`smoke_mcmc_stability.py --guarded`运行，固定在`image3d-mcmc-capacity-20261017-v1.service`及指定远端仓库；复用GPU租约/空闲准入、逐rank显存遥测和TrainingMonitor，运行中检查18GiB reserved/每臂300秒及最终12步完整性。外层unit900秒，磁盘20GiB准入/4GiB停止；主存max/high/swap无限额且host_group=None，无主存主动停止。输出仅新建研究目录，原库/capacity分别保留probe及资源/退出记录，子进程只执行既有小smoke。

### Capacity 多周期受控验证（2026-10-17）

用户明确授权原库/capacity的三合成用例，每rank128点、每例最多1000更新，不含10k/30k/Test/生产变更。`run_mcmc_capacity_cycles.py`只在指定远端与`image3d-mcmc-capacity-cycles-20261017-v1.service`运行；复用已通过的同core两环境，不改core/配置默认。每rank4个来源初始opacity=.5，其余1e-6；两个64px固定合成相机与黑背景、可见点z=2，负对照全在z=-2，失衡仅rank1在相机后。模型/优化器及参考图像初始哈希须配对；这是刻意构造的困难状态，不是旧失败现场重放或held-out场景质量。

初始Adam step9300，opacity两矩设为稳态正则梯度及平方，其他矩为0；使用名义30k LR的9301..10300、原100步搬迁间隔、固定local cap128。核心正则乘128/1500000保持全局3M下的逐点正则量级，明确为虚拟归一化而非3M模型。两实现同初始seed，不承诺采样路径分化后的噪声轨迹一致。图像梯度统计为总梯度减去独立autograd正则梯度，可能有float舍入残差。记录前后健康、实际K/投影、可用来源容量、candidate接纳数与实际步数；来源耗尽按原alive门槛在搬迁前同步终止，不伪造剩余更新。固定点数排除通过少增点降低负载的混淆，但不证明大模型资源表现。

预声明探索性安全否决线：可见与失衡用例两实现均完成1000更新、两分片终态均有有效可见点、candidate原始合成PSNR相对原库下降不超过0.5dB；无图像信号负对照可耗尽来源，不把共同失败称为candidate新增退化。任何缺失/输入不配对/非有限或否决均阻止给出后续10k资格；即使通过，本脚本也不启动真实场景。保留六用例失败继续且不重试；逐case独立取消/日志/遥测目录，18GiB reserved/每例900秒、磁盘20GiB准入/4GiB停止、unit6000秒、GPU串行锁；RAM max/high/swap无限额，不设置主存保护。摘要的受控提前结束必须与逐行真实进度、耗尽rank、最终健康和显存遥测吻合。

### 主存修复与重新基线（memory-v2，2026-10-13）

上述8GiB/7GiB策略保留为失败v1的历史合同，不再用于新实验。v1在10k Validation出现匿名内存与shmem合计超过memory.high、反复回收/缺页，最终遥测陈旧保护自动终止；不是OOM或质量失败，不恢复无optimizer/RNG的best模型分片。用户明确授权先修复主存、验证后重新启动基线并放宽内存限制。

运行时新增仅释放已解码图像的`TrainingViews.clear_cache()`，保留sources/相机/采样顺序；主训练进入Validation前清Train缓存，验证返回后清Validation缓存，不使用会销毁sources的`clear()`。健康统计由CPU张量直接转NumPy，避免百万Python浮点对象，保留原float64/分位数算法及数值口径。验证每个视角加载前记录完成数/总数，分开标记健康统计与逐rank汇总；只有新MCMC协议启用300秒无Validation进度停止门禁，120秒遥测陈旧与原有保护不关闭。缓存/RNG/指标等价由远端回归验证，不宣称释放对象必然等于RSS即时归还。

新profile为`apartment_mcmc_single_factor_memory_v2`，unit后缀`20261013-v2`、全新输出目录与once。主存硬限12GiB、`MemoryHigh=infinity`（协议记null、cgroup须为max）、SwapMax0，18GiB整机空闲准入、6GiB运行余量、11.5GiB任务主动停止；逐卡18GiB/global3M、30k/2048/seed及五臂单因素保持不变。12GiB上限依据约19.4GiB现场可用及6GiB共享主机余量，不能将“尽可能放宽”解释为取消所有保护；运行中不自动加预算或重试。

新核心必须重跑三环境小型双卡门禁，并通过`scripts/smoke_mcmc_validation_memory.py`：读取旧MCMC 30k检查点的哈希验证模型分片（仅作资源负载，不续训），实际3M点/SH3/2048、分配Adam两矩，每rank三次完整351视角Validation；每轮先暖Train图像缓存再释放，检查缓存释放后sources仍可用、CUDA RNG不变、各轮逐视角PSNR差<1e-4、每轮<600秒且任务内存低于11.5GiB。父控制器复用run_stage/cgroup/遥测/进度门禁，失败不生成passed。此负载不是失败10k堆的完全重演，也不是新场景画质证据。准备新协议必须绑定当前core的内存门禁summary及三环境报告，旧v1报告不能替代；源训练/config/replay、协议和失败记录均保留，重启必须fresh。完整基线还需跨过10k、30k保存、SOR和Selection才能宣称完成，不保证任何未来负载绝不失败。

2026-10-13验证结果：42c26f6远端161项回归通过；任务20261013-162310-eedc/只读监控20261013-162445-be3a正常exit0。三环境两rank均通过新core `55ea2ce2…` 门禁。真实3M/SH3/2048、Adam两矩常驻的三轮351-view验证全部完成，逐轮最慢rank耗时46.818/45.377/45.275秒，每rank每轮实际释放Train缓存536205312bytes，逐视角PSNR跨轮最大差0、CUDA RNG不变。2秒采样任务主存峰值6724980736bytes约6.26GiB，high/max/oom/oom_kill均0，逐卡reserved峰值3391094784bytes；这是代表负载门禁而非原10k堆的严格前后对照或完整30k稳定性证明。新协议已冻结在`outputs/analysis/mcmc-stability-ablation-20261013-v2/protocol.json`，SHA `05f55c4bada35cfc7e5c4c35b1c63eb41f15f24e1556054cd71d863184568d33`，内存summary SHA `561292e7…`；独立记录后才启动新基线，不为文档提交改变远端核心。


### 用户明确取消任务主存限制（unlimited-host-memory-v3，2026-10-16）

用户在raw_loss最终checkpoint因11.5GiB主动停止后，明确要求取消12GiB硬限及所有任务主存保护；此授权取代上文memory-v2的主存资源政策，不重写v2协议、预算、失败结论或产物。无主存限制可能触发整机OOM并影响面板/其他服务，不能承诺只有训练进程会被杀；不改主机swap配置或其他服务限额。

`scripts/run_mcmc_stability_ablation.py --prepare --unlimited-host-memory`只在新的输出目录生成`apartment_mcmc_single_factor_unlimited_host_memory_v3`协议，unit后缀`20261016-v3`。host policy的`memory_max_bytes`、`memory_high_bytes`、`memory_swap_max_bytes`、`startup_available_bytes`、`minimum_available_bytes`、`stop_current_bytes`全部记null；实际启动必须显式`MemoryMax=infinity`、`MemoryHigh=infinity`、`MemorySwapMax=infinity`，运行入口核对本unit及祖先cgroup没有继承的有限max/high/swap限制。无主存限额不等于物理内存无限，不关闭内核OOM机制。

新模式不向`run_stage`传host_group，因此不调用`host_failure`，不再按18GiB启动余量/6GiB运行余量/11.5GiB任务总量提前拒绝或终止，也不生成原受限模式的内部host-memory时间线；需要内存观察时使用独立只读面板监控，不将观测重新接成停止条件。原CUDA遥测、逐视角进度、显存/磁盘/时长保护与GPU串行租约仍在，它们不是本次取消的主存限制。新模式通过协议显式选择，不能用run时开关覆盖已冻结v2。

此改动仅在研究runner，不改trainer/absgrad_resources或训练核心SHA；三环境GPU门禁及3M Validation报告仍验证同一核心，但受限内存门禁是历史执行证据，不是新无限额政策的安全证明。新协议重新绑定runner及所有输入/config SHA，仍不读Test RGB、不改线上默认/模型、不执行新final-fit。取消主存限额本身不授权重试已失败臂、拼装未验证checkpoint或覆盖旧现场。已失败raw_loss的八次验证是SOR前质量证据，不能标成完整实验通过。


### 串行总控与失败继续（2026-10-16）

用户随后明确要求一次性串行跑完任务、最后统一查看，并指定“如果失败就跳到下一个串行任务”。这覆盖此前每臂失败停止续接/等待人工审查的调度规则，并授权本次新目录fresh重跑；不是旧目录的自动重试或checkpoint恢复。`scripts/run_mcmc_stability_suite.py`在独立suite目录准备新的无限额v3协议，顺序为baseline→raw_loss→relocation_floor→destination_moments→position_lr；重跑新基线以保持五臂同资源政策和同冻结身份，旧成功/失败臂不覆盖、不伪装成新协议结果。

总控持专用flock和一次性标记、每臂同步等待独立systemd任务；单臂继续复用GPU租约和显存/磁盘/时长保护，RAM/high/swap均infinity，没有主存主动停止线。每臂前重新查GPU；有残留占用则该臂记失败，不强行并行或杀其他进程。基线不可用时依赖它的候选记录skipped/matched_baseline_unavailable，不绕过基线门禁。单臂非零退出或完整性核验失败记录后继续下一臂，不重试；整个主机OOM、总控被杀或协议准备失败仍可能阻止后续调度，不能承诺故障后总控永不退出。

每个成功臂在继续前自动核验30k/60k/world2、八次351-view Validation、相机序列与新基线一致、完整最终checkpoint六组件流式大小/SHA及元数据SHA、三阶段退出结果、模型和Selection评估SHA；不把GPU利用率、模型分片或退出码0单独当成功。每臂review及日志即时独立落盘，结束写suite.summary.json（含相对新基线的质量差值）与suite.exit.json；有任一失败/跳过时最终返回1，但不是在首个失败时停队列。suite和单臂入口都绑定进新协议，训练核心保持不变。总控运行在远端面板任务，不依赖本地会话续接；另外只有一个独立只读监控。最终集中审查时仍按每臂分别归档/提交结果，不在训练中修改远端Git或发布模型，不读取Test RGB、不做final-fit。

## 双卡绝对梯度实验（2026-09-30）

内部 `absgrad_ablation_v1` 使用配置 schema 11，在 schema 10 的 Project 参数上仅增加 `densification.absgrad`（默认 `false`）。公开配置、MCMC 配置及其 schema 10 哈希不变。双臂使用该内部 profile，唯一 changed leaf 为 `densification.absgrad`；通过 `--resolved-config-json` 进入既有 runner，不增加 API／前端入口。MCMC 拒绝绝对梯度。该实验称为“AbsGS 启发的绝对梯度消融”，不是完整 AbsGS 复现，不构成默认推广。

双卡仍按高斯分片、每 rank 每步一个相机、`packed=False`，参数梯度仍使用 `(loss / world_size).backward()`。绝对梯度只用于增密：CUDA 按像素取绝对值后累加，在 backward 完成后、策略更新前，把相机所属卡的 `[1, N_global, 2]` 统计反向 All-to-All 回传并还原成 `[world_size, N_local, 2]`。相机维度不提前求和、不额外乘除 world size，保留 DefaultStrategy 的屏幕归一化、可见性计数和逐相机范数累积。每次重用该次前向记录的分片长度，支持增密后的不等长分片；任一卡缺失／非法统计则集体失败，不能以零或 `.grad.abs()` 代替。增密结束后停止生成／回传统计，Validation 和固定拓扑 final-fit 保持原路径。

安装的 gsplat 1.5.3 不支持此模式。`scripts/prepare_gsplat_absgrad.py --output-dir <新的独立目录>` 只对审计 SHA 匹配的 `rendering.py` 创建独立包副本：保留通信后张量和分片长度，限定非 packed、每卡一个相机、经典 3DGS、无颜色分块，不改 CUDA 内核。仅实验进程通过 `PYTHONPATH` 选择副本；不修改 `.venv` 或生产环境。副本有明确能力标记，原库拒绝开启态；`overlay.json` 记录前后源码与 CUDA 二进制哈希，checkpoint 的 environment hash 额外绑定副本渲染源码 SHA。

远端小测试入口 `scripts/smoke_distributed_absgrad.py`：先以未补丁库 `--baseline-only` 保存双卡有符号参考，再以隔离副本 `--reference-dir` 核对关闭态／参数梯度、单卡全模型逐相机参考、不等长分片、不可见分片及梯度抵消；最后运行 64×64、13 个初始点、12 步的双卡合成 trainer，覆盖增密、增密结束、Validation、checkpoint 与模型合并。数值容差预声明 `atol=2e-6, rtol=2e-4`。不是正式场景质量实验，不读取真实 Test，失败目录保留，输出不覆盖。

2026-09-30 远端两张 L2 实现冒烟通过：代码 `b27fb2d`，任务 `20260930-104232-f8e3` exit 0，产物 `outputs/analysis/distributed-absgrad-smoke-20260930-v2/`。93 项 CPU 回归、每 rank 58 项数值对照通过；关闭态与原库图像差 0，所有对照最大绝对误差 `9.31e-10`，绝对梯度对全模型逐相机参考最大误差 `5.82e-11`。抵消样例有符号范数约 `1.69e-7–3.58e-7`、绝对统计范数约 `15.98`。真实 trainer 完成 12 更新／24 样本、3 次增密（13→104）、增密结束、Validation、最终 checkpoint 和模型合并。原库源码 SHA 未变；首次 v1 在准备副本时因补丁锚点不唯一而失败，保留失败目录并收窄锚点后以 v2 复验。此结果仅为实现门禁：合成 loss 从约 0.90985 到 0.91134，未证明收敛或质量改善；未验证大场景资源、SH3 数值、恢复续训或正式场景 A/B，不能宣称候选质量通过。

2026-09-30 下一轮先运行 signed 对照：`scripts/smoke_distributed_absgrad.py --sh-degree 3` 使用非零高阶 SH 系数，核对原库与隔离库、全模型与分片的参数梯度及绝对统计，参考与候选必须采用相同 SH 阶数；原 SH0 检查仍是默认。`scripts/run_absgrad_signed_baseline.py` 在通过 SH3 检查后核对冻结源协议和配置，生成仅 absgrad 不同的双臂配置，但只执行 signed 的 30k/60k、SOR、377-view Validation 和 2k/4k Train-only。两臂配置复用 1920、seed20260729、recovery-prune；旧目录 seed3500 是 Mapper 种子帧数。候选 absolute 不自动启动：质量阈值、Train ROI 和候选资源上限须在候选结果产生前冻结，当前 signed 数据不能被称为配对质量结论。保留 native 模型、checkpoint 和评估，暂不重复生成 PLY/ZIP。启动沿用20GiB空闲准入，每阶段至少8GiB；运行中每10秒检查，低于4GiB或单阶段超过6小时请求取消，最多等待120秒保存checkpoint，再仅终止本次子进程组（TERM后30秒KILL）。这些是运行保护，不是质量通过标准；fresh-only、不覆盖、不自动恢复、不运行Test，replay完整性校验仍会哈希Test文件字节而不加载Test RGB或指标。

2026-10-08 状态补充：signed正式基线、SH3门禁、16个固定Train视角/24ROI诊断和同模型重复推理校准已完成；absolute正式臂仍未授权/未启动。原20GiB磁盘准入已通过，不降低门槛。完整指标、产物索引与仍未批准的质量/资源增长门禁见[绝对梯度实验状态](absgrad-experiment-status.md)。此前按日期记录的未验证项保留为历史边界，不代表本次状态。

2026-10-08 候选准备补充：`scripts/run_absgrad_candidate.py`新增草案生成、只读预检与显式执行入口；主训练/final-fit CLI仅在显式`--resource-telemetry-dir`时记录逐rank累计reserved显存。训练核心、signed runner与默认路径不变。候选每10秒检查逐步高斯数、逐rank遥测及主训练两倍wall；遥测缺失/过期/损坏或预算、采样序列不一致均拒绝完成，不以cap/剪枝代替停止。遥测含合并阶段，统计窗口比signed原result更宽；SOR/独立selection仅保留磁盘/硬超时保护。31项不加载模型的本地回归及Ruff通过，尚未远端GPU验证、批准门禁或启动absolute；详见上述状态索引的候选入口和运行限制。 用户随后已明确授权提交、Git同步、远端验证及通过后启动正式候选；另建批准合同，历史草案保留。实际执行成败以启动审计和任务记录为准。

## Experimental Train+Validation final-fit

`gaussian_final_fit=train_validation_v1` is an explicit post-selection delivery phase for native Project/MCMC only; default `off` preserves all historical behavior. The ordinary trainer first selects its best model using held-out Validation, common SOR runs if enabled, and that source model receives the unchanged held-out `gaussian_evaluation`. Final-fit then loads exactly Train∪Validation, verifies disjoint Test IDs, and runs 2,000 updates from the hash-bound source. It uses fresh Adam, the configured final position LR, 0.1× configured feature/opacity/scale/rotation LRs, maximum SH, and the source L1/SSIM/clamp policy. Topology is frozen: no Default/MCMC strategy step, split/duplicate/prune/reset, MCMC relocation/noise, or method regularizer may run; source/final Gaussian counts must match.

The phase reuses distributed native rasterization and deterministic shuffle-without-replacement, records world size and `2,000 × world_size` camera samples, and emits its own model, progress, schema-1 record, and schema-2 `fit_validation` evaluation. That evaluation has `quality_role=in_sample_after_train_validation_fit` and `selection_eligible=false`; it is reconstruction evidence for images used by final-fit, never a held-out quality or promotion metric. The record binds dataset/effective-config/source-model/selection-evaluation/final-model/fit-evaluation hashes, split ID hashes/counts, fixed profile/LRs, loss/resources, unchanged topology, and `test_rgb=not_loaded`. The bounded phase has no checkpoint/resume; cancellation, OOM, non-finite state, hash drift, or incomplete distributed shards fail the requested Job without falling back to the pre-fit model. `vggt_visibility_v1` is rejected in v1 because its current Train-only decision plus held-out Validation reporting contract is incompatible with a model already fit on Validation.

`gaussian_raw_model` remains the trainer-selected pre-SOR snapshot. With final-fit enabled, `gaussian_selection_model` is the SOR-complete source, `gaussian_evaluation` remains its held-out selection report, and `gaussian_model`/`scene_splat` identify the final-fit delivery model. Export verifies the final-fit lineage; an SOR record remains bound to the source model and both records enter the bundle. The original final trainer checkpoint remains the base-training lifecycle checkpoint and is not misrepresented as a final-fit checkpoint.

### Research-only equal-budget control and final Test comparison

`scripts/run_gaussian_final_fit.py --train-only-control` uses the same optimizer loop and fixed numeric profile, but samples only Train and records the distinct `train_only_control_v1` identity. It remains research-only, not a public Job/API option. The 2026-09-15 two-arm UHD experiment adds explicit export support only with the matching `train_only_control_v1` record, `input_splits=[train]`, `control_validation` role and source/final/evaluation hashes; no-record or mismatched-role control exports are rejected. SOR remains bound to the Selection source rather than the final model. Validation is loaded only as a reporting set, never included in the control's optimizer view list. Its final schema-2 evaluation uses `split=control_validation`, `quality_role=held_out_after_train_only_control`, and `selection_eligible=false`; it cannot replace the original Validation-selected source or trigger checkpoint/step selection. Omitting the flag preserves the original Train+Validation profile and profile hash.

`scripts/evaluate_gaussian_final_fit.py` freezes the MCMC Selection / Train-only / Train+Validation comparison after checking identical source/config, fresh-optimizer profiles, two-rank 2,000-update/4,000-sample budgets, exact sampled IDs, model counts/SH, environment and lineage. CPU-only `freeze` opens no dataset RGB. `evaluate` requires both the frozen protocol SHA and explicit `--authorize-test`, checks all candidates before first consumption, and reuses the existing once-per-candidate Test evaluator over every Test view. `report` recomputes paired raw/display distributions only from complete hash-bound Test reports, never selects a model or promotes a default. Training code hashes remain separately disclosed when reusing historical Train+Validation output. This is held-out RGB supervision with shared SfM cameras, not fully held-out geometry, a new scene or continuous-view stability. Note that existing contract validation with a dataset root hashes all image files, including Test bytes; `test_rgb=not_loaded` means no Test image tensor is loaded, not that no file byte was read. The execution/authorization protocol is [gaussian-final-fit-comparison.md](gaussian-final-fit-comparison.md).

## Initialization schema v1

Sparse initialization parses project/COLMAP `points3D.txt` and records:

- source SHA-256;
- minimum track support and maximum reprojection error;
- non-finite/support/error/budget rejection counts;
- deterministic selected-row and complete selection hashes.

New ordinary-COLMAP Global primary / `global_recovery_v1` text outputs use `fixed_model_pixel_reprojection_v1` before calibration diagnostics and the initialization 4-pixel filter: project fixed XYZ through the corresponding text camera and distortion into the original `images.txt` observations, take mean Euclidean pixel residual per point, then report unweighted point summaries. Never interpret Global's stored solver `ERROR` as pixels or repair it by multiplying by a focal length. Preserve the original `points3D.txt` bytes as `points3D.source-error.txt` and bind original cameras/images/points plus normalized points in `pixel-reprojection.json`; replace only the error column. Raw and undistorted models are recomputed independently. Self-built COLMAP+VGGT Global uses the same boundary. Incremental and historical/reused models remain unchanged. Unsupported cameras, missing point references, non-finite residuals, non-positive observed depth and points without observations fail explicitly; no RGB, new geometry or Test is consumed.

Dense initialization reads an RGB PLY and a matching Stage 1 support sidecar when provided. It applies, in order:

1. finite coordinate/confidence checks;
2. minimum support and confidence;
3. coordinate-wise quantile outlier rejection;
4. deterministic normalized-space voxel deduplication, preferring support then confidence;
5. deterministic Morton-stratified maximum budget.

It records every rejection count, source/sidecar hashes, settings, selected-row hash, and selection hash. Source PLY/sidecars are read-only. Selected points are transformed by the dataset contract's `normalized_from_world`; RGB initializes SH DC, higher bands are zero, rotations identity, and scales come from bounded nearest-neighbor spacing.

`dataset.with_initialization` returns a new hashed dataset contract pointing at a verified initialization asset. It never mutates a frozen source contract. Every complete native preparation also publishes `gaussian/replay/` with the same dataset JSON, camera source, initialization NPZ/diagnostics, and registered undistorted images under unchanged relative paths. Same-filesystem assets are hardlinked and otherwise copied. `replay.json` binds dataset, camera, image, and initialization hashes/counts/bytes; an existing replay directory is validated rather than overwritten. `--initialization frozen` accepts this replay root for `project` or `mcmc`, strictly checks NPZ fields/dtypes/shapes/finiteness/positive scales plus diagnostics hashes/counts, and never repeats point selection or 3NN scale estimation. COLMAP databases, descriptors, and matches are not replay assets.

`run_gaussian_training.py --prepare-only` is a separate fresh-initialization research stage: it performs the existing geometry-readiness gate, writes initialization/config/dataset/replay, and returns without optimization, attempts or checkpoints. The internal adapter option `gaussian_prepare_only=True` exposes this stop boundary to the two-arm experiment driver only, returning `gaussian_prepared`, not a completed reconstruction; it is not a public upload form field. Both formal arms then use `--initialization frozen` against that shared replay. See [video-4k-train-only-comparison.md](video-4k-train-only-comparison.md).

`derive_gaussian_pose_repair.py` is the explicit remediation boundary for a retained failed workspace. It requires operator-supplied image IDs and never trains or overwrites its source. The derivative removes those images from the camera source and existing split assignments, removes their sparse-track observations, drops tracks below length three, preserves the remaining parent sparse point coordinates without claiming a new bundle adjustment, recomputes the unchanged mean/max normalization and sparse 3NN initialization, and emits a separately hashed replay plus `repair.json` binding parent and child hashes. Such output is labeled `repaired_derivative`; it cannot retroactively make the original SfM arm successful or comparable.

## Checkpoint and resume

The trainer serializes project-owned model tensors, optimizer, schedule iteration, Default or MCMC strategy state, Python/NumPy/Torch CPU+CUDA RNG, camera order, and finite JSON metric history into the existing R2.5 `CheckpointState`. Attempts and checkpoints retain R2.5 atomic non-overwrite behavior and dataset/config/code/environment provenance checks. Before a terminal or cancellation checkpoint, the trainer releases the no-longer-needed CPU Train/Validation image views; this lowers distributed host-memory peak without omitting or changing any checkpoint state. The effective-config hash prevents resuming a Default checkpoint as MCMC or vice versa; distributed resume still requires the original world size.

Fresh/retry/resume remain distinct. Each attempt keeps its model/result/progress under immutable `attempts/{attempt_id}/artifacts`; a resume attempt cannot overwrite its parent. It loads only a provenance-compatible parent checkpoint. Iteration-indexed view sampling plus restored optimizer/model/RNG makes interrupted execution tolerance-reproducible without claiming cross-GPU bitwise identity.

## Worker and assets

`project_3dgs + gaussian_splat` owns the common Gaussian job lifecycle. The adapter runs COLMAP preparation and `scripts/run_gaussian_training.py` as cancellable subprocesses under the R2.6 serial worker. Trainer dispatch accepts default native `project`, isolated Official comparison `graphdeco`, or native experimental `mcmc`. Project and MCMC use the same Torch/CUDA/gsplat availability probe and distributed execution path. Omitted Project requests enable the existing recovery-gated prune; explicit/environment `off` remains a kill switch, while Graphdeco/MCMC default off and MCMC rejects `on`. Raw Gaussian hyperparameters are not exposed by the public job form.

Public `vggt_visibility_v1` keeps its frozen extent-based depth semantics. The internal-only `vggt_visibility_oversized_center_v1` candidate changes only oversized Gaussian support, contradiction, and capture-envelope tests from the projected 3-sigma depth extent to the Gaussian center depth; ordinary rows remain extent-based. The CLI defaults to `extent`, records the selected policy/profile, and the product adapter exposes no raw policy control. Frozen MCMC evidence rejected the center candidate: it removed every v1 row plus 2,135 mostly high-opacity Gaussians and reduced large free-space proxies, but regressed Validation by 1.341 dB / 0.00783 SSIM. It remains reproducible only and cannot redefine the public profile or frontend derivative.

Complete jobs may publish:

- `gaussian_raw_model`: immutable Validation-selected trainer output before common SOR;
- `gaussian_model`: model passed to common Validation/export after optional SOR;
- `gaussian_training_result`: final iteration, strategy/cap, loss, validation, resource, and checkpoint record;
- `gaussian_progress`: JSONL iteration/validation/topology events;
- `gaussian_dataset`: effective dataset/initialization contract;
- `gaussian_replay_dataset` and `gaussian_replay_record`: frozen self-contained training-input contract and provenance.

These are internal Stage 2C training assets, not the R2.12 canonical/browser Gaussian export. Failed/cancelled partial work stays under lifecycle attempt diagnostics and out of successful manifest assets.

## Verified Stage 2C evidence

Generated ignored evidence:

- CUDA learn/resume smoke: `outputs/experiments/r2_7/synthetic-smoke-20260730-d`.
  - 12 known-camera 64x64 views;
  - cancel at iteration 20, resume to iteration 40;
  - loss `0.14069837 -> 0.09866242`;
  - validation PSNR `16.01746`, SSIM `0.80934`;
  - 23,068,672 peak reserved bytes.
- R2.8 public indoor sparse evidence: `outputs/experiments/r2_8/room-sparse-100iter-v1`.
  - frozen `room` 26/3/3 contract, 20,000 budgeted sparse initialization;
  - 100-iteration train loss `0.54790366 -> 0.53736663`, final 18,765 Gaussians;
  - validation PSNR `2.22707`, SSIM `0.189397`, peak reserved 94,371,840 bytes;
  - same-config resume from iteration 50 to 100 reproduced final loss/PSNR exactly and SSIM within `1e-8`; model tensors differed at most `2.1e-7`, consistent with the frozen CUDA tolerance rather than bitwise identity.
- R2.10 validation-only ablation: `outputs/experiments/r2_10/terrains-validation-100iter-v8`.
  - frozen dataset hash `0c1c2d409aca649e1eba5e1aa1229601dd14bee031cb66d30a495f2d03459b33`;
  - identical config hash `125c31960d7c2185b2f30ee456c5aae5b281577454f5e5fc27a00d008cfa0624`, seed, train/validation split, resolution, and 100 iterations;
  - sparse: 8,745 initial, 8,387 final, loss `0.60263 -> 0.45565`, validation PSNR `5.89716`, SSIM `0.48045`, peak reserved 77,594,624 bytes;
  - dense: 2,000,000 source -> 1,830,684 supported -> 1,782,973 inlier -> 441,565 voxel -> 20,000 budgeted initial; 17,871 final, loss `0.49077 -> 0.36141`, validation PSNR `6.07229`, SSIM `0.47854`, peak reserved 98,566,144 bytes;
  - dense-minus-sparse: PSNR `+0.17513 dB`, SSIM `-0.001907`, Gaussian count `+9,484`, peak reserved `+20,971,520` bytes.

This short 100-iteration public geometry-development run proves executable sparse/dense policies and mixed validation/resource behavior; it does not promote dense initialization, establish final quality, substitute for the primary `room` indoor benchmark, or use held-out test metrics. R2.11–R2.16 still own full evaluation, canonical/browser export, frozen RTX profile, and upload-to-view delivery.


2026-10-09 独立质量探索补充：用户批准另立`absgrad_quality_exploration_v1`，原候选资源失败仍保留。新入口通过显式参数/独立批准合同选择操作预算，不修改训练核心、算法单变量或旧profile；必须受独立16GiB/无swap的cgroup约束，主机22GiB准入、6GiB余量与14GiB任务软停止线覆盖所有阶段。紧急资源停止不调用既有checkpoint取消路径，仅终止该任务进程组，先保存失败证据；正常完成仍执行原checkpoint合同。新6M/18GiB/6h是操作停止线而非算法cap或资源通过证据。完整两阶段配对报告必须另做视觉审查，不产生旧门禁PASS或默认推广；当前RAM准入未过、未启动新正式任务，实际状态见实验索引。


2026-10-09 保存峰值修复：1.5M代表状态实测主峰位于完整状态gather/rank打包；3M旧正常/取消保存均触发隔离12GiB cgroup OOM，不是再次整机OOM。正常保存改为逐rank逐组件落盘和流式封装，公开checkpoint合同不变、内部容器显式版本化并兼容旧读取；取消不保存完整checkpoint。终态不用整份reload取得hash，模型合并直接文件load/save。该改动改变trainer核心hash；后续严格配对需同新提交fresh重跑signed，不绕过原身份门禁或改写旧结果。优化效果及模型等价须以远端回归/同规模实测为准。


2026-10-09后续用户批准新schema3匹配实验独立主机预算：cgroup12GiB/SwapMax0，MemAvailable18GiB启动、运行6GiB余量、memory.current11.5GiB停止。原schema2的16GiB/22GiB/14GiB保持不变。只在新同代码fresh signed+absolute入口显式传入，仍保护全部阶段，不改变训练算法或默认配置。新的匹配身份须包含流式保存/SH3实测SHA及当前core/environment一致性，旧candidate不能借新预算绕过旧身份。

## 2026-10-09 审查修复与代码职责

本次用户授权修复与梳理代码，但明确不启动实验。新增`gaussian/model_io.py`统一best模型、最终模型、Train-only分片和checkpoint模型组件的文件I/O；训练梯度、样本顺序、增密、SOR、两段预算及冻结质量门槛不变。CPU文件加载采用mmap且不先read_bytes；这不是已实测的真实训练内存改善结论。

训练/final-fit在生命周期开始时清零CUDA峰值；其内部`evaluate_views`调用评估时显式`reset_memory_peak=false`，不得重置训练累计值。独立评估仍可重置自己的进程计数器，返回`memory_peak_scope`区分两种口径。10秒遥测仍是轮询观测，不构成硬实时上限。旧结果不重写，也不把旧原生峰值当成已修正的全程统计。

`gaussian/absgrad_experiment.py`集中冻结协议、身份核验、四阶段与匹配顺序；candidate/matched CLI保持原参数接口并变为薄入口，配对报告不再导入其他CLI。新入口不再从旧视频实验脚本导入工具函数。`absgrad_resources.py`负责进程、准入、监控、停止及结构化终态。`scripts/launch_absgrad_matched_pair.sh`将现有systemd启动/只读监控行为纳入版本控制；新增脚本不是启动授权，本轮没有执行其训练或监控模式。

新正式监控写入每阶段`*.memory.jsonl`：总量、anon/file等memory.stat字段、各PID RSS/PSS及rank阶段；缺失字段保持缺失，不以0冒充未知。时间线逐行flush，上限64MiB，超过后明确失败而非静默丢弃。rank阶段另保留追加式事件与原子当前快照。仍按memory.current总量停止，不扣除文件缓存、不改变12GiB/18GiB/6GiB/11.5GiB或旧profile预算。准入、子进程执行和结果校验处于统一阶段终态边界，退出码0但身份校验失败不会产生完成标记；matched顶层另保留failure.json。

资源报告分为historical_reference与matched_pair；新配对高斯倍率读取两臂同窗口监控数据，native显存/时间与telemetry显存分别比较，不混用历史分母。training_provenance在原8文件之外纳入model_io、evaluation、checkpoint、absgrad_resources，防止职责迁移漏出源码身份范围。旧批准合同、验收SHA和产物不改；旧验收core与新代码不匹配时继续拒绝执行，必须重新取得独立授权和验收后才能训练。


## 2026-10-09 恢复signed后继续配对（schema4）

用户授权继续剩余配对，选择取消任务内存上限，并明确将整机MemAvailable停止线由6GiB改为2GiB。仅显式`--continue-recovered-signed`/`--continue-authorized`和独立批准schema4 `absgrad_recovered_matched_pair_v1`使用此策略：新systemd unit的MemoryMax/MemoryHigh/MemorySwapMax均为infinity，合同中的任务内存上限和停止线记为null；非核心调度层将软件阈值转换为正无穷后复用原监控。保留18GiB准入、2秒采样、低于2GiB主动TERM/5秒KILL，以及6M高斯/每rank18GiB reserved/每阶段6小时/磁盘20-8-4GiB边界；无上限不是可用内存保证，也不是旧资源门禁通过。旧schema1/2/3不改。

只接受绑定SHA的本次30k signed离线恢复：核验原协议、失败记录、全部恢复来源、最终checkpoint组件与provenance、模型SHA、原相机序列/预算及原核心/环境一致。新独立目录只执行signed的SOR、377-view Validation及2k/4k Train-only，然后fresh absolute四阶段和全量配对评估。原失败记录不改写，不生成假的signed主训练result或成功exit；`signed/train-recovery.json`和两臂协议记录原训练提交与恢复来源。训练核心12文件SHA保持7f85ff7fb74e6d7ea0dd3224c6c897325d4585754753f80b8450f662abb6700b，Git变动仅调度/报告/测试/合同；不称两臂主训练Git提交相同。

signed原生全生命周期结果缺失，恢复是另一个进程，因此配对报告的native/telemetry显存倍率、native/完整stage耗时倍率记为null并解释窗口不完整；保留原观测值、失败状态和可比的高斯计数，不补造数据、不拼接恢复耗时为原生结果。全部冻结质量门槛、60 endpoint ROI视觉审查、Test不加载和不推广默认保持不变。续跑失败不自动retry或resume。


## 2026-10-10 absolute增密阈值单变量对照（schema5）

用户要求先提高absolute增密阈值做对照，其他参数不变。显式`--absolute-threshold-control`/`--threshold-authorized`选择固定一次`0.0002→0.0008`实验，不提供自动调参扫描。新schema5 `absgrad_threshold_0008_v1`绑定已完成absolute 0.0002的协议、配置、训练退出/原生结果与完整数值报告SHA，以及原报告保护的模型/评估文件。新候选与旧absolute配置必须且只能相差`densification.gradient_threshold`；absgrad仍true。signed只作原质量门槛参照，两臂旧训练/模型/报告均不重跑、不覆盖。

新absolute从相同冻结初始化/replay、seed20260729起步，30k/60k→相同SOR→377-view Validation→2k/4k固定拓扑Train-only和377-view评估。恢复剪枝、opacity reset、学习率、SH3、分辨率1920、15000步结束增密、相机序列及全部冻结ROI不变。训练核心12文件与环境必须匹配旧控制；不同调度Git提交不冒充相同完整提交。仍无任务RAM/Swap上限，18GiB准入、2GiB整机运行余量、6M高斯/每rank18GiB reserved/6h及磁盘20-8-4GiB边界不变。

报告`threshold_control`为主对照：对每个端点的30 ROI及377相同Validation视角比较新absolute与旧absolute，而非把signed冒充阈值对照。两个完整absolute生命周期的资源比可报告，但不作为多次测量置信界；相对恢复signed的不可比生命周期值仍null。原signed质量门槛仍独立报告，超过旧absolute不等于恢复signed质量或推广通过。新两端点60 ROI对照仍需静态视觉审查；旧失败/视觉否决不重写，Test/默认/新seed/新场景不执行。

## Geometry reuse before native training

Explicit same-store multi-image geometry reuse changes only preparation, not the Project/MCMC optimizer, initialization package, training budgets, Validation, final-fit, export or Test isolation. The source must be a completed standard Job with a hash-bound geometry bundle; copied geometry is verified and dataset splits must remain identical. Each destination retains its own Job ID and model. See [the geometry-reuse contract](gaussian-geometry-reuse-contract.md). The rig-neighbor retrieval profile is a geometry-only experimental option, not a new Gaussian training strategy or a claim of quality equivalence.
