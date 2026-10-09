# AbsGrad 训练与实验脚本审查

审查日期：2026-10-09。代码基准：`d581ff1`；远端失败实验实际执行提交仍为 `b2c7ac4`。本文是审查与职责索引，不是新训练授权；`codex.md`仍为唯一执行计划。

## 结论与范围

问题不只是脚本数量多，而是训练、序列化、资源统计、历史实验身份和进程调度的职责互相穿插。此前流式checkpoint修复解决了最终完整状态保存的已测峰值，但未统一其他模型读写路径；正式监控又缺少阶段级内存组成，导致真实任务失败后仍不能定位峰值。

本次深入检查8个AbsGrad/保存诊断专用脚本，以及它们实际调用的训练、final-fit、evaluation、checkpoint、runtime和资源模块。对整个`scripts/`的80个Python文件仅建立职责/依赖索引，**不是全仓库80个脚本的逐项正确性背书**。未修改训练算法、预算、源码或生产设置，未启动GPU任务。

## 一、按优先级排列的发现

### 1. P1：训练内Validation重置了整段训练的CUDA峰值计数

- 位置：`src/image3d_scenegraph/gaussian/evaluation.py:89-90`；调用来自`trainer.py:494-496`，结束统计在`trainer.py:562-565`；Train-only同样在评估之后读取峰值（`trainer.py:809-831`）。
- 问题：`evaluate_model()`无条件对CUDA模型调用`reset_peak_memory_stats()`，但训练最终把该全局计数器的结果作为训练峰值返回。早期峰值一旦高于重置时的当前占用，就可能从原生结果中消失；allocated和reserved都受这一统计窗口问题影响。
- 10秒遥测会保留**它已采到**的最大值，但不能保证在每次重置前采到所有瞬态峰值。因此不能把它描述成不受重置影响的完整累计峰值。
- 验证：用AST提取真实函数并注入模拟CUDA计数器，不加载Torch；进入Validation前峰值19，重置时当前占用2，函数将峰值改成2。模拟数字仅证明控制流，不是远端显存实测。
- 建议：训练过程的累计计数器由训练生命周期独占，嵌套评估不应重置；独立评估可以在自己的独立进程中重置。明确区分训练全程峰值、独立评估峰值和轮询观测峰值。
- 边界：这是显存统计缺陷，不是此次主机RAM停止的根因证据；不据此推翻旧实验已观测到的资源超限。

### 2. P1：内存优化只统一了最终checkpoint，没有覆盖生产生命周期的其他模型读写

- 训练中每次最佳Validation改善仍执行`candidate_path.write_bytes(_model_bytes(model))`（`trainer.py:543`）。
- `_model_bytes()`先构造CPU状态，再写入`BytesIO`并返回完整bytes（`trainer.py:1554-1565,1611-1614`）；不是流式文件写出。
- Train-only最终分片保存仍走相同路径（`trainer.py:839-844`）。其初始化又先`path.read_bytes()`加载完整模型，再切出本rank分片（`trainer.py:995-1015`）；两个rank分别加载全模型。
- 单卡终态也仍有完整`read_bytes()`/`write_bytes()`；不能笼统称“模型保存/加载已全部流式化”。
- 实际风险：大模型在Validation/模型选择或Train-only边界产生不必要的整模型CPU字节缓冲，可能在到达最终checkpoint前就触及任务限制。
- 诊断缺口：保存profiler先释放每rank的512MiB代表缓存，再进行保存；真实训练同时有Train和Validation两个独立缓存集合，每集合上限512MiB，每rank两份。profiler也调用了best snapshot，但没有真实Validation留下的内存状态。其6M保存成功不能代替真实生命周期预算验证（`profile_gaussian_checkpoint.py:113-125,170-171`，`runtime.py:48,77`）。
- 建议：先统一所有模型快照调用方的文件式读写，不能简单把`_save_torch_file`的`xb`用在需要替换的best模型上；best候选应写自己的新临时文件后原子替换。分片加载复用已经存在的Path加载能力。随后单独测训练→Validation→best保存→继续训练的边界。
- 边界：本次失败发生于3000步Validation及best写出之后，但没有内存组成证据，**不能宣称上述缓冲就是唯一或主要根因**。

### 3. P2：正式任务只会执行停止策略，没有保留足以解释内存峰值的证据

- 位置：`absgrad_resources.py:78-84,305-308`。
- 正式快照只有整机MemAvailable、cgroup memory.current、memory.events；没有memory.stat的anon/file、进程RSS/PSS或训练阶段。
- 正常轮询快照未持久化，只在失败/退出时记录最后一次。外层监控也只采总量/事件，而且本次面板任务日志为空。
- 同一仓库的保存profiler已经能记录memory.stat、RSS/PSS和phase，但这些能力没有进入真实任务监控；因此再次失败后仍需猜测缓存、匿名内存或序列化。
- 验证：给真实`host_snapshot()`提供包含anon/file的模拟cgroup，返回值仍只有上述3个字段。此次终态证据也确实没有这些组成项。
- 建议：只复用必需的读取逻辑，写一条有界、持续落盘的内存时间线，加少量明确阶段标记。先保留总量停止线，不擅自扣除file cache或提高阈值；监控持久证据应能在训练cgroup退出后保留。

### 4. P2：准入及阶段后校验失败，绕过统一的结构化终态记录

- 位置：`run_absgrad_candidate.py:226-229,240-246`；`run_absgrad_matched_pair.py:97-143`。
- `require_resources()`在`run_stage()`的try/finally外执行。若GPU忙、磁盘/RAM不足或Git身份变化，阶段目录已经建立，但不会有该阶段的failure/exit JSON。
- 训练退出后的采样序列、预算、评估身份核验也在`run_stage()`外；子进程exit0不代表整个阶段核验成功，顶层却没有统一experiment failure记录。
- 验证：提取真实`run_pipeline()`，模拟准入抛错；输出子目录存在，但failure/exit记录数量为0，且没有启动子进程。
- 当前外层shell通常仍会写exit-code和traceback，不能说失败完全无记录；缺陷是结构化事实分散、消费者必须猜状态。
- 建议：一个明确的顶层异常边界记录`stage/phase/reason`；区分子进程返回值与阶段后校验结果。无需新建通用工作流引擎。
- 本次面板finished=false/exit_code=null的具体原因仍未定位，不能把它直接归因于此项。

### 5. P2：schema3配对资源报告仍混用历史signed与新signed的比较口径

- 位置：`evaluate_absgrad_pair.py:260-278`。
- `gaussian_ratio_to_signed`的分母仍来自旧proposal的高斯门槛除以2，即历史signed；没有读取新signed的`train.exit.json.resources.max_observed_global_gaussians`来生成真正同代码的高斯倍率。
- 显存/时间有另加的`current_matched_*`字段，但历史字段仍使用不带historical前缀的名字；telemetry_scope的说明也没有明确区分两种基线。
- 影响：新两臂完成后，报告使用者容易把历史倍率当作新匹配倍率；目前尚未生成真实配对结果，不声称已导致实际错误质量结论。
- 建议：明确分成`historical_reference`和`matched_pair`两组；新signed/absolute高斯数量使用相同监控窗口，原资源失败独立保留。

## 二、当前真实调用关系

```text
面板执行任务 + 输出目录中的 launch-command.sh（不受Git跟踪）
  └─ systemd-run：cgroup、环境、一次性锁、外层退出记录
      └─ run_absgrad_matched_pair.py：身份校验、新signed/absolute顺序
          ├─ run_absgrad_candidate.run_pipeline(arm="signed")
          ├─ run_absgrad_candidate.run_pipeline(arm="absolute")
          │   └─ absgrad_resources.run_stage：进程、停止线、遥测
          │       ├─ run_gaussian_training.py
          │       │   └─ trainer.train_gaussians
          │       │       ├─ runtime：Train/Validation图像缓存
          │       │       ├─ evaluation：训练内Validation
          │       │       ├─ best-model：旧bytes路径
          │       │       └─ final checkpoint：新流式路径
          │       ├─ filter_gaussian_sor.py
          │       ├─ evaluate_gaussian.py：SOR后完整Validation
          │       └─ run_gaussian_final_fit.py --train-only-control
          │           └─ trainer.final_fit_gaussians + Validation/模型保存
          └─ evaluate_absgrad_pair.py：两端点ROI与全局指标
```

额外依赖是混乱的主要来源：

1. matched入口导入candidate CLI；配对报告也导入candidate，并按profile动态导入matched入口。CLI同时充当库。
2. 三个实验入口借用`run_video_4k_comparison.py`的`revision/require_resources`；该脚本顶层又导入runtime和geometry adapters，调度器因此间接加载Torch与无关几何实现。
3. `absgrad_resources.py`同时装有本次日期/unit/旧门槛、新政策、JSON写出、GPU遥测和进程生命周期。应保留保护能力，但把“本次实验身份”与“如何运行进程”分清。
4. `trainer.py`1643行同时负责主训练、final-fit、评估桥接、模型格式、rank容器、RNG、checkpoint和资源统计，保存策略因此在多个位置分叉。
5. 最关键的systemd启动和外部监控命令保存在Git忽略的outputs目录，仅调度Python脚本受Git约束；复现执行边界还依赖额外审计文件。

## 三、脚本职责与应保留的位置

| 脚本 | 真实职责 | 梳理结论 |
|---|---|---|
| `prepare_gsplat_absgrad.py` | 生成SHA固定的隔离gsplat副本 | 保留为准备工具；不修改生产库 |
| `smoke_distributed_absgrad.py` | SH0/SH3、双rank梯度、小trainer及取消检查 | 保留为GPU验收，不当正式场景预算证明 |
| `smoke_absgrad_resources.py` | 独立cgroup与局部OOM冒烟 | 保留为独立环境验收，不能据此扩大训练授权 |
| `profile_gaussian_checkpoint.py` | 指定规模的代表保存负载 | 保留为诊断工具，标明不覆盖真实Validation生命周期 |
| `run_absgrad_signed_baseline.py` | 历史signed入口，自己实现四阶段与停止逻辑 | 历史兼容，不再作为当前新配对入口 |
| `run_absgrad_candidate.py` | schema1/2旧候选入口，同时承载当前共享pipeline | 混合职责；共享pipeline应移出CLI，历史入口保留 |
| `run_absgrad_matched_pair.py` | schema3新配对、严格身份及signed收据 | 当前配对唯一入口；变薄，不再增加第四个runner |
| `evaluate_absgrad_pair.py` | 两端点30ROI、377逐视角、资源报告 | 保留，但依赖共享协议数据而非导入runner CLI |
| `run_gaussian_training.py` | 数据/初始化/readiness及训练CLI适配 | 保留，不塞本次实验常量 |
| `filter_gaussian_sor.py` | 模型后处理 | 保留并复用；本次未改其算法 |
| `evaluate_gaussian.py` | 独立模型评估CLI | 保留，与训练内评估的资源统计边界分开 |
| `run_gaussian_final_fit.py` | 固定拓扑Train-only/Train+Validation适配 | 保留；统一底层模型读写 |
| `run_video_4k_comparison.py` | 旧视频两臂实验 | 不是公共utility，移除新AbsGrad对它的运行依赖 |

整个scripts目录的80个文件还混合了生产适配器、setup、smoke、一次性实验和离线分析。完整索引见`outputs/analysis/absgrad-code-review-20261009-v1/script-index.json`。先标清入口和依赖，不一次性移动80个文件：适配器、测试、冻结命令与旧协议可能依赖现路径。

## 四、建议收敛为四个职责层，而非另建实验框架

1. **实验协议与顺序**：仅决定已批准身份、两臂配置、四阶段、完成校验。把现有共享pipeline/协议读取移到一个轻量共享模块；CLI只解析参数与调用。旧profile不能被新合同隐式覆盖。
2. **进程与资源**：复用现有资源模块，统一准入、启动、采样、终止、退出证据；移除对旧视频实验的工具函数依赖。把现有外层启动行为纳入可审查的版本控制，而非继续生成新runner。
3. **训练算法**：trainer保留训练循环、策略、相机采样与生命周期调用；不改变absgrad、阈值、SOR或质量合同。
4. **模型/checkpoint I/O**：集中模型文件读写、rank封装和合并；checkpoint.py仍只负责公开原子发布与完整性合同。旧读取兼容保留，legacy helper是否删除必须先核对测试/诊断依赖，不能因为看似未调用就删。

优先级为：先修正资源统计/证据与遗漏I/O路径，再做小范围职责提取，最后才考虑新实验。结构迁移也会改变core hash；应同步核对`training_provenance()`的源文件集合（当前8文件），不能把逻辑移出被哈希文件后让新实现漏出身份边界。任何新正式运行仍需单独确认，本次不将审查扩大为重构或训练。

## 五、验证结果与限制

- 67项已有非模型回归通过（0.49s）：candidate、pair纯指标、checkpoint文件合同、profiling标准库逻辑。
- 本地测试安装import拦截器：任何Torch/gsplat导入都会失败；未训练、渲染或做模型验证。
- 三项新增审查复现使用AST提取真实函数、标准库模拟对象：峰值计数器重置、阶段准入失败无结构化记录、正式快照缺内存组成，均复现。
- 复现脚本：`outputs/analysis/absgrad-code-review-20261009-v1/checks/reproduce_review.py`；它不修改生产源码。
- 现有matched测试主要验证合同模板/旧新profile隔离；四阶段测试mock了子进程。配对测试验证纯指标门禁，没有覆盖新matched完整两臂执行及配对渲染。小GPU smoke不能替代真实大场景内存生命周期检查。
- 本次未修复上述源码问题，也未做新远端GPU验证。

## 六、失败产物清理状态：被权限层阻塞，未删除

已确认旧absolute与新matched都没有活动GPU进程，最新unit为failed。仅选择4个失败运行的`.best-model-rank-000/001.pt`，共928,323,568 bytes（885.318MiB）；未选择日志、进度、配置、协议、成功signed、SH3证据或成功保存诊断产物。

精确路径、大小、inode和SHA256已保存在`outputs/analysis/absgrad-code-review-20261009-v1/cleanup-manifest.json`。它们只是无法用于完整训练resume的模型分片，不是声称文件损坏。

永久删除调用被权限层拒绝，要求用户明确确认清单中的具体路径；没有换通道删除、移动或覆盖。**本轮删除数量为0，回收空间为0。**已有终态审计仍保留删除前事实，清理结果以后续独立记录为准。

## 七、后续修复落实（同日，未启动实验）

用户随后明确授权按上述review修复代码，并授权删除清单中的四条路径。本节为后续状态，不改写前面审查时的发现和权限拒绝事实。

- **清理完成**：2026-10-09 03:30:02Z，在重新核对主机/路径、空闲GPU、失败unit及四文件inode/size/SHA后，持GPU文件锁删除这四个分片。删除文件合计928,323,568 bytes；逐个确认不存在，日志/协议和其他产物未动。回执为`outputs/analysis/absgrad-code-review-20261009-v1/cleanup-authorized-receipt.txt`。
- **发现1已修复**：嵌套Validation显式禁止重置峰值，独立评估保留其自身重置语义，结果标明统计范围。
- **发现2已修复代码路径**：新增`model_io.py`，生产best、终态、分片、合并和checkpoint模型保存统一直接文件写；best使用原子替换，其他目标拒绝覆盖；文件加载不再先read_bytes。真实资源下降尚未实测，不宣称解决了3005步失败的已知根因。
- **发现3已修复**：正式监控复用进程内存读取，持久记录memory.stat、RSS/PSS与阶段时间线；rank阶段既有追加事件也有当前快照，短阶段不再只依赖两秒采样碰巧命中。
- **发现4已修复**：准入和阶段后置验证纳入run_stage；phase明确为admission/execution/verification，返回码0但验证失败仍记录失败。matched顶层覆盖两臂与完成记录写出。
- **发现5已修复**：报告显式分历史/新配对两组，新signed高斯计数来自自己的train.exit记录，显存和时间按对应窗口比较。

新的职责关系：

```text
scripts/launch_absgrad_matched_pair.sh  systemd/只读watch，需另行授权运行
scripts/run_absgrad_candidate.py       历史CLI适配
scripts/run_absgrad_matched_pair.py    匹配CLI适配
        └─ gaussian/absgrad_experiment.py   协议、身份、两臂与四阶段
             └─ gaussian/absgrad_resources.py  进程、采样、停止与终态
scripts/evaluate_absgrad_pair.py       只依赖共享实验协议，不导入CLI
训练/评估 ─ gaussian/model_io.py         共用模型文件I/O
训练 ─ gaussian/checkpoint.py            公开checkpoint原子合同
```

本次没有移动整个scripts目录，没有引入通用工作流框架。旧signed四阶段实现保留历史兼容；核心训练策略与rank checkpoint协调仍在trainer，未做无关的大拆分。资源模块中的旧/新冻结预算保留，不借整理改变数值。

验证：128项本地非模型检查通过（含新双臂mock调度、准入/后验失败、原子文件写失败、峰值reset语义、内存组成/时间线、配对分母）；测试拦截真实Torch/gsplat导入。新增真实模型参数/RNG/分片等价测试仅写入`tests/test_gaussian_trainer.py`，本轮未执行。Shell语法和改动文件Ruff通过；未提交、未推送、未向远端同步源码、未运行模型/GPU验收或新实验。

### 提交与同步授权补充

用户已进一步授权提交并经Git同步本轮修复到既有远端main；上文“未提交/同步”为修复完成时状态。个人草稿和outputs证据不入Git，原合同/验收SHA不改，不启动实验、模型验证或重启服务。实际同步以两端提交SHA一致为准。
