# 来源Adam状态短窗干预

2026-10-29 · **MCMC-Q1-E004 / Q1-B** · **零训练恢复及A/B各20步已完成；来源Adam状态的短窗作用已测得，监控面板终态缺失。**

本轮结果与证据见[单份主记录](records/adam-intervention-recovery-20261029-v1.json)。下文保留设计与历史阶段记录。

## 1. 问题与证据边界

[Q1-A家族配对](../e003_minimal_survival/reports/family-pair-readonly-20261028-v1.md)确认：实际relocation后、noise前S/T六参数位级一致，但初始存活家族等权后，S的20/100步首次死亡风险仍比T高约35—37个百分点。这不足以证明Adam清零致死。

本实验只检验：**同一完整公共起点下，保留来源S迁移前的Adam两矩，是否改变其短期更新和死亡，同时对T/U与重建loss产生什么代价。** 干预对象是六组参数的来源两矩，不只是opacity；首步opacity代数对照只能解释对应的局部更新，不能代替六组状态干预的系统响应。

一个公共前缀不是充分独立重复。结果不能自动证明统计显著性、算法总体改善、长期owner不对称，或15k—30k的百万级分片退化根因。E007与旧Gate失败不追认，生产默认不变。

## 2. 两阶段，分别授权

### 阶段一：100步新公共前缀与Gate 0

- 从冻结12000父checkpoint恢复，名义30000步日程不变，执行12001—12100一次。
- 12100 Adam后、relocation前保存六参数、Adam两矩/step/超参数、strategy、RNG、相机order/cursor的完整typed相位快照。它不是已经完成该步策略的可直接续训checkpoint。
- 原采样、relocation恰好调用一次，保存真实T→S、唯一S、k与T/S/U。relocation后、noise前保存完整相位快照及初始alpha，检查每条T→S六参数位级一致。
- Gate 0核S两矩清零、T/U两矩位级保留、step未改且六组一致、原生Adam类与Parameter映射正确、状态完整且有限。仅支持冻结的非AMSGrad、非fused、非capturable、非differentiable、无weight decay的原Adam配置；实际lr/betas/eps与foreach选项从状态读取，不伪造默认值。
- 来源清零前与目标迁移后的两矩非零比例、符号与绝对值分位数逐组报告。`m/(sqrt(v)+eps)`仅是历史状态尺度，**不是尚未发生的更新**。目标未被此函数清零，不意味着其实际历史必然非零。
- 允许原12100 noise执行一次；保存S/T位置扰动。在Validation入口截停，复用原streaming writer原子保存完整12100 checkpoint，含scheduler、history、camera游标、strategy和四类RNG。不做Validation或导出，不伪造评价值。
- 产生 `gate0.json`，状态为`structural_verified_review_required`。结构通过与动量信号分布分开；不设临时“够大”阈值。如果来源全部六组两矩原本都为0，`effective_intervention_available=false`阻止两臂准备。
- **前缀任务到此退出、释放GPU，绝不自动进入A/B。** Gate 0分析是只读，但生成新前缀需要100次新更新授权，不能使用旧T0/T1六参数快照冒充完整Adam历史。

### 阶段二：用户审阅Gate 0后，A→B各20步

两臂从同一12100完整checkpoint恢复，执行12101—12120：

| 对象 | A：原生 | B：保留来源历史 |
|---|---|---|
| 模型六参数、相机、strategy、RNG | 公共checkpoint | 完全相同 |
| S的exp_avg/exp_avg_sq | 原生清零结果 | 恢复为12100 Adam后、relocation前的值 |
| T/U两矩 | 原生状态 | 位级不变 |
| step、lr、betas、eps等 | 原生状态 | 位级不变 |

原`inject_noise_to_position`不读取Adam两矩，因此共享同一次12100 noise后再恢复S两矩，不会改变这个公共noise结果。**S/T起点相等性检查在noise前；noise后S/T位置可以不同，但A/B同槽位起点仍必须相同。**

恢复完整状态的摘要必须匹配prefix；B白名单只允许唯一S的两个moment字段变化，不能改step、其他状态或参数。checkpoint恢复是显式原生恢复；观察守卫不能靠恢复RNG掩盖消耗。

两臂不重抽映射、不重跑公共前缀、不重复12100 noise。20步内禁止新relocation/拓扑改变。实际相机batch、SH、学习率、原生RNG/noise抽取日程须一致，但**不要求干预后的模型、梯度、Adam长期SHA一致**。

## 3. 首步机制证据

12101的原backward只执行一次，不注入或替换梯度，不做额外forward/backward：

- 六组记录实际更新和梯度摘要；opacity保存逐点总参数梯度、解析正则梯度、两矩、step、超参数及更新前后logit。
- 这里的总参数梯度含重建及正则，不能称为独立测得的纯图像梯度。
- 对A臂已取得的同一份opacity总梯度，离线代入“清零/保留”两种历史两矩计算Adam更新，区分**固定梯度的局部状态效应**与**实际训练系统响应**。
- 每臂用自己的实际梯度、step与状态预测真实更新，对参数、更新量、两矩使用原容差核对。代数不匹配则标记机制解释不可用，不修改原优化器或放宽门槛。
- 保存源/目标原生位置noise前后值；同时保存1/5/10/20步opacity实际更新及moment状态，并将首步更新/梯度按后来首次死亡或右删失分组，只作描述。

首步跨臂梯度仍可能存在CUDA浮点差异；后续干预也会通过模型反馈改变其他Gaussian的梯度，不能把全部差异算成直接局部Adam效应。

## 4. 生存、风险转移与loss相位

- 20步逐点alive/cutoff位图；1/5/10/20步报告首次alpha≤.005的风险。初始未活单列，恢复不抹去首死，未见死亡右删失。诊断死亡不是点被删除。
- 主结果为各owner的S风险`B−A`；并列T（目标点数与家族等权口径）、U、混合目标结局及逐目标死亡先后。
- 同一个新公共前缀允许跨臂配对slot；不能与旧T0/T1槽位配对。目标家族保留0…k个死亡成员分布，不将多个T压成一个二元状态。
- 报告各臂owner1−owner0差距及其变化，不把owner与硬件GPU或相机执行rank混同。
- 原生loss/L1/SSIM/正则按相同步骤和相机配对。**12101 loss在首次干预更新前；12102才可能反映第一次更新。12120 loss仍在第20次更新前**，本轮不额外渲染更新后终点。
- S改善而T/U或loss变差时必须报告代价；不自动宣布算法改善、不临时挑允许恶化阈值、不追加C臂或第三次重复。

## 5. 预算与保护

| 阶段 | 全局更新 | 双rank相机forward | loss backward | Adam参数组step |
|---|---:|---:|---:|---:|
| 公共前缀 | 100 | 200 | 200 | 1200 |
| A | 20 | 40 | 40 | 240 |
| B | 20 | 40 | 40 | 240 |
| 合计 | 140 | 280 | 280 | 1680 |

正式仅12100一次全局relocation（两owner调用），额外渲染0。公共前缀与两臂各30分钟上限，人工审阅间隔不持GPU；collective120秒。单卡reserved≤18GiB，RAM max/high/swap无限额。新输出≤16GiB，准入20GiB并始终保留4GiB；prepare计入全状态快照与原子checkpoint暂存峰值。

合成前置独立：每owner8点SH3，公共前缀4步、A/B各2步，共8更新、16 forward/backward、96 Adam组step，真实图像解码0；另每rank两次一次性位置noise等价性fixture调用，不增加模型forward/backward。上限15分钟，主存无限额。rank1设置零目标，验证rank局部采样跳过不会导致共同边界死锁。

各阶段新目录、once、GPU lease、单执行＋唯一只读监控、原子终态；失败不自动重启，不覆盖或清理旧结果。只允许新实验自身原子checkpoint暂存的原生收尾，不碰用户文件或其他attempt。

## 6. 实现与前置

- `probe.py`复用typed snapshot、原生resume与optimizer hook模式、共同边界consensus和位图编码，不安装旧完整观察器，不修改生产trainer或gsplat。
- `analyze.py`远端CPU按绑定快照重验Gate 0、首步Adam代数、家族风险及原生loss；本机fresh import不加载Torch/NumPy/gsplat。
- `run.py`分别提供prefix与arms授权，绑定实现来源、当前执行依赖、父checkpoint与独立前置（历史v2执行绑定64份，首次恢复/两臂执行绑定67份）。旧Q1或E007前置不能替代。
- `tests/test_intervention_remote.py`仅指定主机且 `IMAGE3D_AUTHORIZE_SOURCE_ADAM_TESTS=1` 才收集；报告路径变量为 `IMAGE3D_SOURCE_ADAM_PREFLIGHT_REPORT`，报告和pytest basetemp必须为本实验新runs目录。
- 前置验证原生checkpoint写入/恢复、两矩白名单、step与T/U保留、首步代数、映射/家族分布/删失、原调用预算、观察不变、RNG负例、零目标owner及双rank拒绝；测试代码存在不代表已经通过。

## 7. 执行流程与后续授权边界

1. 本地实现/轻量验证完成 → 用户授权提交 → 用户手动Git同步。
2. 明确授权新合成前置与100步公共前缀后，核目标主机/提交/GPU/磁盘，使用独立unit执行。前置unit为 `image3d-source-adam-preflight-v1.service`，prefix为 `image3d-source-adam-prefix-v1.service`，arms为 `image3d-source-adam-arms-v1.service`，三者主存限制均无限额。
3. 模块 `experiment.mcmc_shard_imbalance.questions.q1_post_relocation_survival.experiments.e004_source_adam_intervention.run`：
   - `--prepare --authorize-prepare --preflight-report <新报告> --output-root <新adam-intervention目录>`。
   - `--prefix --authorize-prefix --protocol-sha256 <prefix协议SHA> --output-root <同目录>`，只能在prefix unit内运行。完成后停在Gate 0。
4. 用户审阅实际moment分布、明确批准两臂后：
   - `--prepare-arms --authorize-prepare-arms --acknowledge-gate0-review --gate0-sha256 <已审阅SHA> --protocol-sha256 <prefix SHA> --output-root <同目录>`。
   - `--arms --authorize-arms --arms-protocol-sha256 <新arms协议SHA> --protocol-sha256 <prefix SHA> --output-root <同目录>`，只能在arms unit内运行。
5. 各阶段唯一只读监控用 `--monitor --monitor-phase prefix|arms --protocol-sha256 <prefix SHA> --output-root <同目录>`，不会停止或重启执行。

两阶段之间保持同一实现提交与执行依赖；若实现需要修复，应另立尝试与前置，不能覆盖已有once或把旧Gate 0绑定到变化后的代码。用户已提交并同步e98c828，2026-10-29按授权执行前置，但在合成A初始化失败，真实公共前缀未启动；下面记录此次终态，不自动按上面的流程重试。


## 8. 2026-10-29前置失败

[失败报告](reports/adam-intervention-20261029-v1-preflight.md)：执行与唯一监控均exit1，pytest 1 failed/73.05秒。8点合成prefix已完成4全局更新，但合成A在零来源rank1上对空moment张量求摘要时触发`stride(-1)=0`的Float→Byte view错误；rank0收到共同拒绝。完整前置未通过，不能用局部合成Gate 0放行真实数据。

没有preflight通过报告、正式协议或正式输出目录；真实100步公共前缀与A/B更新均0。原失败保留，未修代码或补跑。建议在本实验局部处理空来源摘要、保留冻结helper及零来源用例，再经提交/手动同步与新尝试授权进行验证。


## 9. 空来源摘要局部修复（2026-10-29，尚未远端验证）

按用户“按建议修复，然后开始实验”，`probe.py`新增局部`moment_digest`：空张量只按冻结摘要的tensor/dtype/原shape编码生成摘要，零数据字节不做Float→Byte视图；非空张量原样交给原`digest_state`。来源moment的before/expected/actual/reference/after五处摘要均使用该适配，不改共享helper、RNG守卫、参数/两矩干预或T/U保护。

本机用无Torch的空张量替身复现原字节视图报错路径，并检查原shape/dtype参与摘要、非空委托不变。远端前置新增CPU/CUDA的NumPy来源空张量及显式零stride张量检查，覆盖六组参数的空来源形状；在RNG不变守卫下执行，不增加模型forward/backward/Adam调用，原rank1零来源用例完整保留。

旧e98c828前置失败报告不改为通过。修复后的前置尚未执行；新尝试仍需先提交并由用户手动Git同步，使用新的目录与once，前置通过后只跑100步公共前缀，在真实Gate 0后停止。用户已授权这个执行范围，但未授权真实A/B；不得直接修改远端源码或复用旧失败目录。


## 10. v2公共前缀完成、收尾失败（2026-10-29）

[完整报告](reports/prefix-gate0-20261029-v2.md)：a269eb1新版合成前置1 passed/65.48秒，随后真实公共前缀100步完成，保存原生12100 checkpoint。执行包装器在finally调用缺失的Monitor.record时失败；原case.exit、complete和顶层gate0未落盘，执行与监控仍exit1，不能补造历史成功。

已只读验证两rank计数、全部快照SHA、100步六组学习率及checkpoint组件，并仅在CPU解码后核完整状态摘要与保存时一致。实际S数量owner0/1为80,384/142,316，T为87,900/168,827；opacity两矩在S清零前与T保留后均100%非零，S清零后均零，六组step均12100且不变，S/T六参数在noise前位级一致。来源一阶矩正负约53%/47%，尚不能宣称统一方向或有益效果。

另发现原verify_worker错误读取顶层position_lr，以及Gate 0 JSON betas列表/元组精确比较问题。独立核查按冻结日程与实际嵌套诊断字段核lr，只做JSON表示归一化，不改数值或原数据。真实A/B未准备或启动，未重跑公共前缀。

后续优先修编排/验真接口并设计零训练重新绑定，复用已验真checkpoint；不覆盖旧失败目录或补写原退出码。当前尚未实施这些修复/恢复，也未获真实A/B执行授权。


## 11. 三接口修复与零训练重新绑定（2026-10-29，仅本地实现）

本次不重跑100步，也不补写旧任务缺失的case.exit/complete/gate0。v1/v2失败记录及只读报告保持原样；本节描述新实现，不代表远端恢复已经执行。

### 三项修复

- Monitor补齐原共享runner要求的record接口。测试实际启动仅退出的标准库子进程，覆盖成功、非零退出和资源拒绝，确认exit/失败记录及资源摘要正常落盘；不再只mock run_stage。
- 学习率按冻结Train相机几何计算scene_scale，逐步核原指数日程和六组lr；诊断行再核mcmc_health_by_rank中两owner的position_lr。普通行不假设存在顶层position_lr，但不跳过lr验证；缺owner、重复owner、错SH/lr及缺终点诊断均拒绝。
- Gate 0使用排序键、禁止NaN的精确JSON表示比较，tuple/list仅按JSON数组表示归一化，不删字段、不舍入、不加容差。typed数组与来源状态的位级检查不变。

### 独立恢复入口

`prefix_recovery.py`只提供prepare/recover/monitor及`--authorize-recovery`，没有训练入口。新目录必须为本实验RUNS直属`adam-intervention-recovery-*`。

来源固定为a269eb1的`adam-intervention-20261029-v2`及原协议SHA，12100 checkpoint hash固定为`ed3257af8a07625b8abeacfe6df4016d24c9db9e8a4cca95336fa11888f326f9`。旧64份代码按旧Git对象验真，允许的接口/分析文件之外，训练核心与probe必须仍匹配旧SHA；核心数值分析函数也核AST不变。旧GPU前置只作为旧提交的历史证据，不能冒充当前接口已通过GPU前置。

新CPU前置记录提交与依赖SHA；是否失效按实际恢复代码、运行依赖及该CPU测试/runner fixture的变化判断，无关提交和其他实验测试变化不要求重跑。恢复协议仍核完整执行文件集合。前置覆盖真实runner收尾、小型原生checkpoint谱系复制/CPU摘要、实际100步日志与原生日程函数对齐及JSON表示。该前置不实例化模型或优化器，不调用训练、forward/backward、optimizer.step，不初始化CUDA或恢复活跃RNG。

恢复时重新验真原100步worker、快照、两owner Gate 0、原失败终态与checkpoint组件，CPU解码两rank完整状态并与保存时摘要精确匹配。在新目录复制12000与12100两个完整checkpoint及原attempt描述，保留字节级谱系、核SHA并设只读；约4GiB同机复制，不下载权重，不复制数据集或整套相位快照。旧相位快照仅经绑定路径只读引用，未来A/B产物只能写新目录。

恢复入口校验实现、前置和来源一次；复制后核源文件未变及目标完整性，不在收尾重复实现/前置校验。完成时写新的gate0.json与recovery-complete.json，明确`original_suite_status=failed`、`original_child_exit_code=null`、新训练/渲染/反向/optimizer/strategy均0。新恢复成功不追认旧任务成功，不创建假的prefix/complete.json或case.exit.json。

### 校验职责（2026-10-30，本地精简）

准备与执行是独立入口，各自进行一次共同输入校验。恢复证明读取不再回调全量来源验证；同一入口复用已取得的证明、来源文件摘要及恢复时重建的Gate 0，不再次扫描同一批checkpoint或重算Gate 0。

worker只保留协议身份、继承的GPU lease、原生加载和本rank实际恢复状态/S两矩干预保护，不重新验证整套历史来源或A臂产物。父进程保证A验收成功后才运行B；每臂首步/轨迹分析只做一次，最终比较直接复用结果。收尾不再重验历史输入，也不在刚写完complete后立即再全量读回。

必要的复制后完整性、实际干预白名单、Train边界、失败停止和资源限制不变；没有新增缓存、持久化证明或跳过校验开关。旧64文件仍按原Git对象核对历史来源，但当前测试文件不再要求保持历史版本；实际训练核心与probe约束不变。

### 后续A/B边界

正常prefix仍须原native_complete证明；恢复来源须独立recovered_payload证明，不能混用。未来arms协议绑定证明类型、路径和SHA，并仍要求用户审阅新Gate 0 SHA和两臂独立授权。对恢复根，即使传prefix授权也直接拒绝重新训练。本次恢复任务不会准备或启动A/B。

### CPU预算与后续操作

独立unit为`image3d-source-adam-recovery-v1.service`，30分钟上限，主存三项无限额，输出8GiB、准入12GiB、保留4GiB。只持恢复自身锁/once，不占GPU租约；CPU与磁盘复制仍有资源成本。失败不自动重试或删除部分证据。唯一只读监控不停止服务。

先提交并由用户手动Git同步，然后：

1. 使用新目录运行CPU前置，unit为`image3d-source-adam-recovery-cpu-preflight-v1.service`、15分钟、主存无限额。设置`CUDA_VISIBLE_DEVICES=`、`IMAGE3D_AUTHORIZE_PREFIX_RECOVERY_CPU_TESTS=1`与`IMAGE3D_PREFIX_RECOVERY_CPU_PREFLIGHT_REPORT=<新报告>`；pytest basetemp必须在本实验RUNS内新目录。
2. 模块`experiment.mcmc_shard_imbalance.questions.q1_post_relocation_survival.experiments.e004_source_adam_intervention.prefix_recovery`执行`--prepare --authorize-recovery --preflight-report <新CPU报告> --output-root <新恢复目录>`。
3. 同模块在恢复unit内执行`--recover --authorize-recovery --protocol-sha256 <新恢复协议SHA> --output-root <同目录>`，强制隐藏CUDA，完成后停止供审阅。
4. 唯一只读监控用`--monitor --protocol-sha256 <同SHA> --output-root <同目录>`。

本节实现现已通过远端CPU前置和零训练恢复；用户随后明确授权的A/B各20步也已完成，见顶部主记录。此恢复入口自身仍不启动A/B。


## 12. 两矩拆分短窗（2026-10-30，本地实现，未执行）

目标是判断首步两矩作用能否转化为持续20步存活，不修改生产默认，也不延长至100步或30k。

| 条件 | 来源S干预 | 执行安排 |
|---|---|---|
| A | 两矩清零 | 只读复用已完成参考 |
| B | 六组两矩保留 | 只读复用已完成参考 |
| V | 六组仅保留exp_avg_sq | 新增20步 |
| M | 六组仅保留exp_avg | 新增20步 |
| O | 仅opacity保留两矩 | 新增20步 |

参考固定为`adam-intervention-recovery-20261029-v1`，新三臂均从其同一12100完整checkpoint独立恢复。T/U、S未选矩、参数、step与超参数均不变。新产物写入`adam-intervention-moment-split-*`；复用原复制函数在新目录保留原生12000/12100谱系，不向旧训练目录写入，不再生成一套恢复证明。

`probe.retains_moment`明确选择字段，原A/B含义不变；`run.py`复用原worker/lease/Monitor/逐臂验收，`moment_split.py`负责固定参考与汇总。完整输入校验仍只在独立入口执行一次，worker不扫描历史。只读取本轮需要的checkpoint、pre-relocation/cohort、首步参考、日程与报告，不重跑旧A/B分析或扫描无关旧日志。新旧实现来源分别记录，不能把旧运行冒充新代码的重复实验。

**正式预算：** V→M→O，每臂20步，共60全局更新、120相机forward、120 backward、720 Adam参数组step。额外渲染、公共前缀更新和新relocation均为0。三臂共享30分钟执行上限，单卡reserved≤18GiB、主存无限额；输出16GiB、准入20GiB、保留4GiB。任何一臂执行/验收失败即停，M臂不会临时裁剪更新或更改lr，也不自动跳过失败继续O。

**输出：** 五条件并列各owner的S/T/U 1/5/10/20步首次死亡、T家族等权与混合结局、owner差距、首步实际更新与两矩、固定历史A梯度的代数对照，以及新臂分别相对A/B的同相机loss差。12101 loss作为更新前基线，主要均值用12102—12120；不额外渲染第20次更新后的终点。每臂分析一次，汇总直接复用。

### 验证和启动方式

本机只做标准库/mock/AST检查。新字段选择另有远端CPU测试`tests/test_moment_split_remote.py`，仅在指定主机、`CUDA_VISIBLE_DEVICES=`和`IMAGE3D_AUTHORIZE_MOMENT_SPLIT_CPU_TESTS=1`时启用；pytest basetemp须为本实验runs内新目录。该测试用8点合成状态覆盖A/B/V/M/O及空/非空来源，各执行一次真实CPU Adam，最多60参数组step，0 forward/backward/真实训练。它不运行Gaussian模型、不初始化CUDA、不重跑旧GPU前置；结果使用测试日志，不新增验收证书。

提交并手动Git同步、明确授权后，先运行上述CPU测试，通过才准备和执行。复用原模块`experiment.mcmc_shard_imbalance.questions.q1_post_relocation_survival.experiments.e004_source_adam_intervention.run`：

1. `--prepare-split --authorize-prepare-split --output-root <新目录>`：验证参考，复制续训谱系，最后写新协议；不训练。
2. 在`image3d-source-adam-moment-split-v1.service`内执行`--split --authorize-split --protocol-sha256 <新协议SHA> --output-root <同目录>`，并沿用主存无限额、30分钟上限的unit设置。
3. 唯一只读监控使用`--monitor --monitor-phase split --protocol-sha256 <同SHA> --output-root <同目录>`。

旧prefix/arms模式不能接收split协议。当前仅实现上述能力，CPU数值测试与真实三臂均未执行；不为这次实现重新运行有效的历史实验。
