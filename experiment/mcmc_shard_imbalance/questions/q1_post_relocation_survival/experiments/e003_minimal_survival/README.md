# 最小逐点生命周期

2026-10-28 · **MCMC-Q1-E003** · **两条训练已有完整描述；独立冷进程前置与零训练补渲染已通过。原套件的隔离渲染失败记录不追认。**

## 1. 为什么另立协议

[E007正式校准](../../../../branches/b1_observation_reliability/experiments/e007_observer_admission/reports/admission-20261027-v1.md)因两个loss范围超限停止，U/O没有运行。[既有日志分析](../../../../branches/b1_observation_reliability/experiments/e007_observer_admission/reports/admission-log-analysis-20261027-v1.md)表明：记录差异在迁移前已经出现，迁移后窗口广泛增大，但三组owner总体不对称方向仍一致。

用户批准改变证据目标：**先验收最小记录操作，再描述同一次带观测运行中T/S/U的命运；附带固定相机下迁移即时效果，不再用200步跨run loss范围作为描述性研究的前提。** 这不是放宽E007门槛、恢复其未运行O组或证明完整非干预性。依赖E007的来源/工具与失败证据，不要求把E007追认为通过。

所有结果必须明确`full_numerical_noninterference_proven=false`；纯读取和状态保护仍可能改变CUDA时序。记录有效不等于不带观测的反事实轨迹相同，更不是因果机制或长期质量证明。

## 2. 固定两条轨迹

- 固定T0、T1，每条从同一12000父checkpoint独立启动，执行12001—12200共200步，名义学习率日程30000。
- 父checkpoint SHA：`080952acf0ef989b160b64d9bb8e17f8acb7adf2bf7e18b19e3ae1d94d3c2105`；冻结core：`5941370c16bb04301db2f59477468eebe62326201d12c487980a1dfb51dcde8a`。
- 原库、原Train replay、每owner150万点、全局300万cap；不改采样、noise、梯度、Adam或迁移规则，不同时引入尺寸修复/raw_loss/floor等方案。
- 12100建立T目标、S实际被抽中来源、U未参与三组。来源重复按实际采样序列记数，S是唯一来源集合；同一来源的副本不是独立样本。
- 生命周期观察到12200第二次迁移之前；随后原生第二次策略/noise照常完成，在Validation入口保存共同终点并截停。Test/Validation像素解码与评价禁止。
- 两条轨迹分别报告；不把slot当跨run身份，不为结果不一致补第三次，也不将两次启动当充分的统计独立重复。

## 3. 最小记录器

`observer.py`不继承旧完整`LifecycleObserver`的安装逻辑，不挂optimizer/梯度/训练渲染钩子、不逐步全模型哈希、不做分组贡献渲染。

只旁置包装原策略post、原`_relocate_gs`和原`_multinomial_sample`：

1. 12100 `_relocate_gs`之前（Adam之后）保存六参数typed快照。
2. 原始采样恰好调用一次，原返回tensor原样传回，只复制用于身份记录的索引；不重采样、不覆盖原参数。
3. 原迁移返回后、noise之前保存六参数快照、T→S映射、标签及原dtype/CUDA sigmoid得到的初始alpha。零目标也形成有效空T/S、全U记录。
4. 12101—12200所有Adam完成、策略开始之前，保存alive（>.005）与below cutoff（<1/255）位图，完整100步。
5. 12200第二次迁移前保存末态opacity/alpha/标签与索引；原native health与本记录的同相位计数必须一致。

每次纯观察均检查RNG不变，不能通过恢复RNG掩盖消耗。完整参数/梯度/Adam/strategy读前读后摘要只在微型前置中测试，真实轨迹不做逐步全模型摘要。错误协调只放在双rank共同post边界；零目标rank跳过原采样，不在rank局部采样钩子中插入collective，避免另一rank永远等不到。

额外拒绝协调通信与原训练collective分开。保证的是原调用不被替换/重排，不声称总通信数量不增加，也不宣称调度完全不受影响。

## 4. 主线交付

`analyze.py`在远端CPU读取typed快照及位图，独立重建初始未存活、首次死亡/恢复、恢复后连续alive及窗口右删失。每owner、每T/S/U分开给出：

- 迁移后立即未存活数/比例。
- 初始存活者的100步再死亡率和首次死亡时间；初始未存活者不混入再死亡分母。
- 曾恢复、恢复后再次死亡、末态死亡及其对owner总死亡的贡献。
- 恢复后的首次连续alive段及右删失数量，不能把100步窗口当终身生存。
- 两条轨迹的owner1−owner0描述性方向，空分母、零差或反向明确输出；不自动选因果候选。

来源复用固定分层1、2–3、≥4；初始opacity分层≤.005、(.005,.01]、(.01,.05]、>.05。T/S输出交叉分层的点数、初始存活分母、再死亡率及等权来源家族再死亡率；U没有来源家族，不伪造复用标签。各分层的家族数显式保留，不输出p值或把百万点当百万独立样本。

不采集或伪造逐点图像贡献、梯度、Adam机制解释；这些若以后需要，须另立最小问题与验收。

## 5. 迁移即时渲染：两条训练结束后隔离执行

prepare从冻结Train IDs按`SHA256(UTF-8 image_id), image_id`排序选择8张，固定相机元数据、归一化、原分辨率配置、SH3和loss配置进入协议；不根据本轮loss或历史尖峰挑视角。

对每条轨迹的迁移前/实际迁移后快照，各渲染两遍同一组8相机，共32次相机forward；两条合计64次。独立worker持同一GPU租约，在两个训练子进程都退出后才启动，no_grad、无优化器/策略/反向。保留原双rank渲染路径及原库，不借此夹带已知尺寸问题的修复；结论限定该冻结渲染实现。

输出：
- 同状态重复的RGB/alpha元素差与loss差，容差沿用`2e-6 + 2e-4*abs(reference)`。
- 迁移前后固定相机RGB/alpha差、L1/SSIM/loss变化；这是实际迁移的即时渲染效应，不是离散来源分叉或后100步放大的因果证明。
- 同状态数值差超原容差时，标记即时渲染解释不可用，不调整容差、不将已有有效生命周期数据伪装成失败或成功准入。结构/有限性/身份错误仍是整套停止条件。

不保存不必要的整批图像，不追加训练，不从历史12k伪造此次12100迁移前状态。前状态是同一次事件“不施加本次迁移”的即时参照，不能说明长期跳过迁移的质量。

## 6. 独立新版结构前置（2026-10-27已通过）

仅指定主机/仓库、显式`IMAGE3D_AUTHORIZE_MINIMAL_SURVIVAL_TESTS=1`及新的`IMAGE3D_MINIMAL_SURVIVAL_PREFLIGHT_REPORT`路径允许运行。报告和pytest `--basetemp`必须在本实验runs内的独立新目录。

前置单独使用`image3d-minimal-survival-preflight-v1.service`、`RuntimeMaxSec=900`、主存max/high/swap无限额；进入GPU租约并确认空闲后才初始化CUDA。原预编译扩展及Torch/CUDA/GPU必须匹配冻结来源，不允许JIT fallback。

微型每owner8点、两张64×64合成视图，无观察C与最小观察T各4步：

| 全局预算 | 数量 |
|---|---:|
| 合成更新 | 8 |
| 训练相机forward / loss backward | 各16 |
| Adam参数组step | 96 |
| 两状态重复只读渲染forward | 8 |
| 真实图像解码 | 0 |

检查纯观察前后活跃状态、原抽样tensor身份/内容/次数、原训练collective序列、T/S/U映射/位图重建、零目标记录fixture、Test/Validation解码拒绝、终点截停、双rank拒绝及RNG负例。空目标fixture不执行额外训练/策略调用。

只有全部断言和两rank证据齐全才写总报告。报告绑定新提交及旧53份依赖和全部新实现/测试SHA；E007前置不能替代此报告，也不能把“实现存在”写成远端通过。

## 7. 固定预算与失败停止

| 正式套件项目 | 上限 |
|---|---:|
| 全局更新 | 400 |
| 训练相机forward / loss backward | 各800 |
| 双rank六参数组Adam step | 4800 |
| 全局迁移事件 / owner迁移调用 | 4 / 8 |
| 分组贡献forward | 0 |
| 隔离只读渲染forward | 64 |
| 正式全流程（训练、渲染、分析） | 60分钟 |
| 每卡reserved显存 | 18GiB |
| 主存max/high/swap | 全部无限额 |
| 正式新增产物 / 磁盘准入 | 12GiB / 16GiB |
| 必须保留空闲磁盘 | 4GiB |
| 子进程collective超时 | 120秒 |

原父checkpoint只复制一次、设只读；每条轨迹保存迁移前/后及共同终点，不逐步保存全量参数。prepare按SH3 fp32参数形状及父文件大小估算输出，实际运行持续核预算。

新目录、once/flock、单执行＋唯一只读监控；首次结构/身份/资源错误停止、留终态、无自动重试；两条描述方向不一致并非程序失败，不加第三次。同状态渲染仅数值不一致时保留原值并标记不可解释，绝不改门槛。

最终成功状态仅为`descriptive_trajectories_complete`；旧Gate1和E007均保持未通过，`full_numerical_noninterference_proven=false`、`long_term_causality_proven=false`。

## 8. 入口与后续流程

模块：`experiment.mcmc_shard_imbalance.questions.q1_post_relocation_survival.experiments.e003_minimal_survival.run`。

- `--prepare --authorize-prepare --preflight-report <新报告> --output-root <新survival-*目录>`。
- `--run --authorize-execution --protocol-sha256 <明确SHA> --output-root <同目录>`，独立`image3d-minimal-survival-v1.service`内。
- `--monitor --protocol-sha256 <同SHA> --output-root <同目录>`只读；worker/render-worker只允许内部固定case及继承GPU租约。

用户手动同步3d5a7df并授权后，上述前置、prepare及正式两轨迹已执行；T0隔离渲染因RNG守卫失败停止，T1渲染未启动。详见[部分结果与核验](reports/survival-20261027-v1.md)。不得重试原once、重训补渲染或将失败报告改写为通过；后续修复/只读补渲染需另行授权。

本机只做标准库/mock/AST/目录检查，不加载模型或用本机GPU验证。原完整套件失败与事后CPU重算的训练描述分开登记；完整数值非干预和长期因果仍未证明。


## 9. 后续只读核查（2026-10-28）

[精确复用次数与RNG源码核查](reports/source-reuse-readonly-20261028-v1.md)已完成，无新训练或渲染。高k来源迁移前opacity也更高，原始风险不能作为复制次数的因果效应；粗基线分层后没有统一单调关系，未取得逐点投影覆盖。RNG发现csrc与backend预加载边界遗漏、前置进程已热而正式进程全新的覆盖缺口。该只读核查当时没有动态证据；随后独立前置与补渲染已完成，见§11。原失败状态不改。


## 10. 隔离渲染初始化修复（2026-10-28，本地实现，未远端验证）

用户批准修复并明确不重跑训练。`render_probe.py`现在在测量守卫之前复用既有固定B0 `initialize_backend()`：先核预编译csrc身份，再显式导入`gsplat.cuda._backend`并核其`_C`为同一扩展。初始化前后Python、NumPy、Torch CPU、CUDA RNG摘要单独保存，允许如实记录初始化消耗；没有forward/backward热身，不恢复RNG，不删除原守卫。原`prime_package`与训练路径不改变；`run.py`只在已有render分支传入新的审计输出目录。

每次只读相机forward分别记录load、render、loss、export四阶段的RNG摘要；发生消耗时保留失败阶段、变化的组件及原异常，不吞错。图像加载在独立阶段完成，render阶段将已加载视图交给原冻结渲染函数，不增加相机forward次数。审计目录不得落在来源快照目录内。

### 零训练冷进程前置

新增`tests/test_cold_render_remote.py`，独立变量`IMAGE3D_AUTHORIZE_COLD_RENDER_TESTS=1`及新路径`IMAGE3D_COLD_RENDER_PREFLIGHT_REPORT`，仍须新runs目录和新basetemp。使用`image3d-survival-cold-render-v1.service`、15分钟上限、主存三项无限额。

两rank通过spawn启动新进程，在任何gsplat forward之前确认backend未加载；8点SH3的两份手工合成状态（不是训练或迁移），两张64×64合成PNG经真实加载路径进入probe。每rank4次相机forward，全局8次；训练更新、反向、optimizer、strategy调用均0，真实数据集图像解码0。禁止调用训练/Adam/backward，核模型未改写、同状态重复性、初始化0热身、全部分阶段RNG摘要和主动消耗Python RNG的拒绝负例。此检查不再依赖“同进程此前已经训练过”的暖状态。

只有独立前置成功才写绑定新提交/执行依赖/双rank证据的报告。旧Q1或E007前置都不能替代。本地没有运行这个GPU测试，也没有取得修复通过的动态证据。

### 独立只读补渲染入口

模块为`experiment.mcmc_shard_imbalance.questions.q1_post_relocation_survival.experiments.e003_minimal_survival.render_recovery`，没有训练路径。

- 来源固定为原`survival-20261027-v1`，协议`7390dacff982e1add40cb3dd99b6b1c4943eb771aa0edf86ceee120b27286c97`、原提交3d5a7df及两条训练complete SHA。
- 原协议中的59份旧代码按原Git对象校验，不拿已修复工作区冒充旧执行版本；原库包、replay dataset、选中Train图像及全部完成文件绑定同时验真。失败终态必须仍是失败。
- prepare将旧来源身份与新修复代码/新冷进程前置绑定成新协议；输出必须为新的`render-recovery-*`目录，不能写入原T0/T1或旧失败渲染目录，不复制父checkpoint、不重新计算训练轨迹。
- 固定顺序只补T0、T1，各用原8相机、前后两状态各重复两遍，合计64相机forward；新训练/反向/optimizer/strategy调用均0。
- 正式上限30分钟、每卡reserved≤18GiB、输出≤1GiB、磁盘准入≥5GiB且保留4GiB空闲；主存三项无限额、collective超时120秒。独立`image3d-survival-render-recovery-v1.service`，once/flock/租约，首次失败停止，不自动重试。
- 命令：`--prepare --authorize-prepare --preflight-report <新冷进程报告> --output-root <新目录>`；执行用`--run --authorize-render-only --protocol-sha256 <新SHA>`及同目录。唯一只读监控仍用`--monitor`，不占GPU；监控可等待至65分钟但正式执行最多30分钟，不因此延长计算预算。
- 同状态A/A数值超原容差仍只标记相应即时解释不可用；任何RNG/结构/来源错误立即失败。成功也只写`render_only_complete`，原整套失败和长期因果/完整非干预未证明状态不变。

以上为修复实现时的状态。用户随后手动同步f5b248b，并明确授权“先跑冷进程前置，通过后补渲染，不重训”；执行结果见下节，未恢复或覆盖旧任务。


## 11. 独立补渲染结果（2026-10-28）

[实测报告](reports/render-recovery-20261028-v1.md)与[独立运行记录](records/render-recovery-20261028-v1.json)：冷进程前置1 passed/14.21秒，随后仅用原T0/T1快照完成64次相机forward；新增训练、反向、optimizer和策略调用均0。执行与唯一监控均exit0，原训练产物SHA及原失败终态完整保留。

初始化阶段仅Python RNG变化，正式256份分阶段测量审计全部不变。32个同状态重复配对的RGB/alpha全部位级一致；这证明本次初始化边界修复通过，不等于对旧失败的唯一调用来源完成动态追踪。

固定8张Train相机上，T0平均loss由0.065316620变为0.064676861（−0.979%），T1由0.065511346变为0.064955371（−0.849%）；两条均4张下降、4张上升。实际relocation产生稳定的即时图像变化，但不能据此认定长期质量改善、解释来源S后续死亡或E007跨run分化。后续机制干预仍须另行设计和授权，不重训补样本。


## 12. 家族内配对只读结果（2026-10-28）

[Q1-A家族配对报告](reports/family-pair-readonly-20261028-v1.md)：先固定20/100步为主结果，再核验relocation后、noise前的真实T→S配对。四组共513,600条边、六类参数30,302,400个元素全部位级一致，配对初始存活状态也一致。

排除初始未活家族后，每个家族内对T成员平均，再家族等权。四组20步S−T首次死亡风险差为+35.36—36.67个百分点，100步为+35.14—36.86个百分点；k=1也有同方向差异。混合目标结局、各目标首次死亡及右删失单列，“最早目标”仅作带k偏差的补充。

这确认了同家族、同参数起点下的命运不对称，不能直接证明来源Adam清零是原因。初始参数相同不保证后续noise/梯度/渲染竞争相同；本次未读取不存在于事件快照中的Adam历史，也未新采集梯度。全部使用原产物CPU只读分析，0训练、0渲染、0远端文件写入；新公共前缀及短窗干预尚未实施。
