# 绝对梯度消融：当前状态与执行边界

更新日期：2026-10-10。此文是结果索引；`codex.md`仍是唯一执行计划，训练合同见[gaussian-trainer-contract.md](gaussian-trainer-contract.md)。名称仅为“AbsGS启发的绝对梯度消融”，不是完整AbsGS复现。

## 0.0008对照最终结论：执行成功，数值与静态视觉门槛均失败（2026-10-10）

任务`20261010-105009-a387`在`49bc64a`上于04:33:21 UTC退出0，总耗时约1h43m；主训练30k/60k样本、SOR、377-view Selection、2k/4k样本Train-only及ROI数值评估全部exit0。用户随后授权继续视觉审查且不重训；本次只读取已有图像，没有重新渲染、加载Test或启动新实验。

- 全部60/60 endpoint ROI经12张四列图（reference/signed/absolute0.0002/absolute0.0008）逐项审查。两组各60张图的SHA匹配，reference/signed像素完全一致；118个受保护源文件在写入前后复验不变。
- **明显恢复但不全面通过**：旧候选10项墙角和8项Train窗边椅子的严重模糊/结构丢失明显减轻；椅脚局部收益保留，地毯细纹收益明显收缩。
- **明确静态视觉否决2项**：同一Validation视角2907的Selection与Train-only窗边椅子ROI。两个端点均额外检查已有1078×1267原尺寸裁剪；相对signed，右侧椅座/下部暗色轮廓被更大灰白雾状模糊淹没。signed亦有伪影，不是无缺陷参考。差值分别为−4.22810dB/−0.09486 SSIM和−3.93794dB/−0.08318。这是同一视角两个端点，不是两个独立视角。
- 全局raw均值：Selection 22.007329/0.805141，Train-only 22.280625/0.808734。最终相对旧absolute +2.78976dB/+0.02774，相对signed +0.05866dB/−0.000762；均值门槛通过，但Train地毯增益、墙角SSIM和全局SSIM P10仍失败（最终P10 ΔSSIM −0.013112）。
- 主训练5213.07秒，结束1,249,994高斯、SOR后1,236,705；相对旧absolute高斯数少约68.8%、主阶段耗时少约38.5%。全阶段采样任务峰11.0595GiB、整机可用最低10.8554GiB，OOM事件0。单次运行，不作置信区间；恢复signed的完整生命周期倍率仍null，旧资源失败不改写。
- 最终`FAIL_NUMERICAL_AND_VISUAL`，不推广默认、不自动扩展实验。AI静态审查非用户人工验收；动态遮挡跳变未评估，58项未见明确硬否决不等于全面通过。

远端新增三份审查文件已通过文件API写入并逐项SHA复验，目录：`outputs/experiments/absgrad-threshold-0008-20261010-v1/experiment/absolute-candidate/paired-quality/`。

| 文件 | SHA256 |
|---|---|
| `report.json`（原始数值，不覆盖） | `a488940a7ee3c263f638f948dbf8ee4e47e36de659eb067b9477e2da2c0462c5` |
| `visual-review.json` | `a72843e5427577ee02a247df3ea64977ddfbf74e0327ffb709fc0aff880fd5cf` |
| `FINAL_REPORT.md` | `781ac863a5142907fa665f123413272c7d3fcd5488439095b108313cd7abe335` |
| `review-completion.json` | `b53c00770486ea817e42262757f4ab21d611ffdfba9b02879a72f8d92783ea0d` |

原数值报告/experiment完成记录的visual_pending是生成时快照，不覆盖；以新增`review-completion.json`闭合审查。本地逐项报告与拼图位于`outputs/analysis/absgrad-threshold-0008-review-20261010-v1/`。下方启动与准备文字为历史时点，不是当前仍在运行。

## 0.0008对照启动快照（历史：2026-10-10 02:50 UTC）

远端已同步`49bc64a`，167项回归及实际策略/单leaf/旧控制身份核验通过。新任务`20261010-105009-a387`，只读监控`20261010-105009-ddb0`，目录`outputs/experiments/absgrad-threshold-0008-20261010-v1/`。02:50:52Z train阶段active，仍在初始化、尚无首个optimizer更新记录；已核验仅新absolute阈值0.0008，无signed重跑目录。

候选config hash `45b107feba2b193fc7647c9f3ce7fc98731ac1aea3ac6059c364f67c86a7d9d5`，训练核心未变。RAM/Swap无上限、2GiB整机可用停止线生效；启动采样OOM事件0。**尚无新质量结论；运行期间远端HEAD固定49bc64a，不pull后续文档改动。**

## 新实验准备：absolute阈值0.0008对照

用户授权仅提高absolute增密阈值，其他参数不变。新schema5一次固定`0.0002→0.0008`，从相同冻结初始化fresh训练；已完成absolute0.0002为主对照，signed作质量参照，两旧臂不重跑。候选与旧absolute只允许增密阈值一个leaf变化；相同seed/采样/30k及2k预算/reset/prune/SOR/ROI与2GiB整机余量保持不变。126项本地非模型检查通过，训练核心SHA不变；**此处是准备记录，实际远端验收/启动以后置记录为准。** 下方数值与视觉失败仍是旧0.0002实验的有效结论，不抹去或改写。

## 旧0.0002最终结论：数值与静态视觉门槛均失败（2026-10-10）

评估修复`14c3cdf`已通过Git同步。补评任务`20261010-101642-8ecc` exit0，耗时105.719秒；没有重训，原模型/评估SHA不变。两端点全部60 ROI及各377个Validation逐视角报告已生成，全部60组静态对照完成AI视觉审查。

- **局部改善成立**：Selection / Train-only的Train椅脚组均值ΔPSNR +3.0213 / +2.4390dB、ΔSSIM +0.07090 / +0.06935；地毯+1.8991 / +2.1881dB、+0.15590 / +0.16880，两组每端点均4/4双指标改善。
- **质量门槛失败**：Validation椅脚3670的PSNR差-1.0298 / -0.7724dB；墙角控制1728为-14.2007 / -15.1843dB，SSIM亦大幅退步；全局raw均值/P10均失败。
- **明确视觉否决**：两端点墙角条纹/边界大面积消失、窗边椅背被模糊色块遮挡，共20项明确否决。椅脚/地毯改善与其他区域的严重退步并存，不能只挑局部展示。
- 审查为AI静态图像审查，每crop最长320x250；不是用户人工验收或动态漫游。时间性遮挡跳变未评估，不记为通过。已有明确视觉否决足以判失败。
- 本次结论限单场景、单seed、冻结绝对梯度配置，不泛化为完整AbsGS算法无效；不推广默认、无Test、新seed/场景。

远端最终证据根：`outputs/experiments/absgrad-continued-matched-20261009-v1/experiment/absolute-candidate/paired-quality-20261010-v1/`：

| 文件 | 内容 / SHA256 |
|---|---|
| `report.json` | 两端点数值与377逐视角；`9948702bde9a0361360999287f1329f765b43db7e246b5ad8642a21af81ca1ae` |
| `visual-review.json` | 全60项观察与图像SHA；`92c6ad1a9aab53d547481fc034b8b0801d0082ca1f58b27adda8fd798d814843` |
| `FINAL_REPORT.md` | 完整数值/逐项视觉结论；`dcddc567bd21d7c4453cbfc33549a247e44f6deb9ea34b7ef06a757c3aa4b69e` |
| `review-completion.json` | 补评及静态审查完成、质量失败；`5a429e8de32e859dc099f81e33b98c8e95164f7c018b8c78c00e194004cb36f6` |
| `contacts/` | 全60组reference/signed/absolute对照 |

旧实验退出1和旧失败证据保持不变；数值报告生成时的visual_pending亦不覆盖，以新增`review-completion.json`闭合审查。原两倍资源失败保留，恢复signed的原生生命周期指标不补造。本地报告：`outputs/analysis/absgrad-roi-completion-20261010-v1/review-results/FINAL_REPORT.md`。收尾核验GPU空闲、四项共享服务active，8081页面/API均200。

## 2026-10-10：训练完成，修复ROI评估入口

续跑在2026-10-09 12:13:12Z退出1，但signed/absolute的所有训练、SOR、Selection及Train-only评估均已完成且模型SHA核验一致。失败仅发生在ROI渲染入口的CUDA未初始化峰值重置；远端冷进程复现，显式初始化后通过。用户授权修复并仅补全ROI报告/视觉审查，不重训；121项本地非模型检查通过，后续执行写新目录，不覆盖旧paired-quality失败证据。

四组评估均377/377：Selection signed 21.945152/0.806016，absolute 19.199299/0.776501；Train-only signed 22.221965/0.809496，absolute 19.490868/0.780997。两端点全局质量门槛已失败，局部ROI结论仍待报告与审查。续跑采样任务总峰13.789GiB、整机可用最低10.265GiB，OOM事件0；取消任务上限后完成训练，不代表原两倍资源门禁通过。

## 最新执行：剩余配对已启动（2026-10-09 09:18 UTC）

远端已通过Git同步至`fba6048`；161项CPU回归与恢复模型/检查点身份核验通过。执行任务`20261009-171722-8a17`，只读监控`20261009-171722-f714`，新目录`outputs/experiments/absgrad-continued-matched-20261009-v1/`。signed没有重新训练，SOR已exit0（20.95秒），377-view selection正在运行，之后按合同执行Train-only、fresh absolute及配对报告。

已实测确认新unit的MemoryMax/MemoryHigh/MemorySwapMax均infinity，整机可用内存低于2GiB时仍停止。09:18:07Z采样任务约1.425GiB、主机可用约18.479GiB、OOM事件0；这些是运行快照，不是最终峰值或完整质量结果。**运行期间保持远端HEAD为fba6048，不pull后续文档提交。**

## 续跑准备：取消任务上限、保留2GiB整机余量

用户已授权继续剩余配对并将整机可用内存停止线明确改为2GiB。新增schema4恢复模式，signed复用校验通过的30k模型，只执行后3阶段；absolute仍从冻结初始化fresh开始。所有旧失败/合同保留，原生生命周期显存/耗时比因signed离线恢复不再伪作完整匹配。120项本地非模型回归、Ruff和shell语法通过，12文件训练核心SHA不变；**此条仅记录准备，远端同步/验证和实际启动另记。**

## 最新状态：三万步保存恢复完成（2026-10-09 08:47 UTC）

新同代码配对 `32e3927` 的 signed 已完成30k更新/60k样本及最终377/377 Validation，随后在checkpoint保存阶段触发11.5GiB保护，原任务退出1；不是CUDA或cgroup OOM。保存期间匿名内存约7.74GiB基本不变，文件页缓存约2.04→3.69GiB，总量达到11.54GiB。原失败目录和终态保留。

用户明确授权后，独立任务 `20261009-164657-6f45` 在无任务级RAM/Swap上限、无11.5GiB软件停止条件下，从校验通过的暂存组件恢复保存，耗时28.964秒、exit0。新目录：`outputs/experiments/absgrad-save-recovery-20261009-v1/`。

- 完整checkpoint：`recovered-training/attempts/train-001/checkpoints/iteration_000030000/`；现有加载器和组件SHA校验通过。
- 合并模型：`recovered-training/attempts/train-001/artifacts/model.pt`，1,493,441高斯；与原两个rank分片按顺序逐参数一致。
- checkpoint hash：`a41d9fd748015b49cc5f2d33bfd07c35c89f1892b44b120251165ef89def72a7`。
- model SHA256：`cb9da1b10198ed9574726daf256709b9acfa86cad14b9851a5ab3ead9aeaf915`。
- `complete.json`与`recovery-provenance.json`明确记录离线恢复、原任务失败及源文件SHA不变；无重训、不伪造原训练result或配对complete。
- OOM事件0；`memory.peak`不可用，没有实测总峰值结论。取消限制只作用于此次恢复保存，未修改后续训练入口的预算。

**SOR、Train-only、absolute及配对质量审查仍未完成；此次保存成功不等于完整实验通过。** 下表及后续旧记录中的signed正式基线指历史基线，不能替代这次新配对。

## 已完成／未完成

| 项目 | 当前状态 |
|---|---|
| 双卡实现、SH3门禁 | CPU102项、每rank58项数值对照及12步合成trainer通过；不是质量结论 |
| signed正式基线 | 30k更新/60k样本→SOR→377-view Validation→2k/4k Train-only完成 |
| 观察区域 | 6个Validation ROI、16个Train相机/24个独立Train ROI已冻结并诊断 |
| 重复推理 | 同模型同16个Train视角一次额外推理，显示uint8逐通道与raw/display指标差均0 |
| 磁盘准入 | 2026-10-08 01:56:33Z可用302.30GiB，原20GiB线通过；每次执行仍须复核 |
| 质量／资源增长门禁 | 用户授权按上述计划验证后训练；新建批准合同，历史草案保持原样；不提供训练seed方差 |
| absolute执行控制 | 31项本地轻量回归、132项远端回归及SH3／真实双rank遥测冒烟通过 |
| absolute正式臂 | 已中断：5418步，高斯数及rank 0显存越界；取消期间发生主机OOM。无完整质量结果，不自动重试 |
| Test、生产默认推广 | 未执行、未授权 |

## signed证据

- 训练提交：`9cf78ad84abdc2da61f03512af09a3484b7e75bd`；seed20260729、双卡、最长边1920。
- 双臂内部schema11仅差`densification.absgrad`；主预算、增密阈值/结束点、recovery-prune、SOR及Train-only保持一致。
- 主训练3016个Train相机各19–20次，包含第15000步的增密前各9–10次；Train-only各1–2次。与历史Project两段采样序列相同，不等于每ROI有效可见次数相同。
- SOR后Validation：raw PSNR21.916840/SSIM0.803270；2k Train-only后：22.147273/0.806472，均377/377成功。
- Train-only角色为`held_out_after_train_only_control`且`selection_eligible=false`，不能替代原selection。
- 模型SHA256：`9113b393646a4b1e254355791b79b498cadec8969f0d2bbd7f791427bd7f2475`。
- 隔离gsplat rendering SHA256：`f1e345b2c943f2d5541d91d5b7d89e76c91467ab419a5dfe6448d2a1ba2ada9e`；生产依赖未改。

3670组Train椅脚/地毯仍有细节欠拟合。新旧关闭态全局均值接近但局部漂移较大，且代码/环境不同，不能当作重复seed噪声或AbsGrad效应。额外推理仅确认16个相机的显示像素和指标复现；旧raw预测张量未保存，未做raw像素逐元素比较。

## 结果索引

以下路径相对仓库根；生成产物不入Git。远端根为`/usr/local/3dgs_new/Image3D-SceneGraph`。

| 目录 | 入口 |
|---|---|
| `outputs/experiments/absgrad-signed-20260930-v1/experiment/` | `protocol.json`、`complete.json`及signed原生模型/评估 |
| `outputs/analysis/absgrad-signed-completion-20261008-v1/` | `REPORT.md`、`summary.json`、`sampling-audit.json`、`roi-audit.json` |
| `outputs/analysis/absgrad-signed-train-roi-20261008-v1/` | `REPORT.md`、`train-roi.json`、`train-roi-metrics.json`、`contact-sheets/` |
| `outputs/analysis/absgrad-signed-render-repeatability-20261008-v1/` | `REPORT.md`、`result.json`、`contract.json` |
| `outputs/analysis/absgrad-stage2b-gate-proposal-20261008-v1/` | 先读`CALIBRATION_UPDATE.md`，再读原`PROPOSAL.md`与`gate-proposal.json`；仍是草案 |
| `outputs/analysis/absgrad-results-sync-20261008-v1/` | 本轮同步`manifest.json`、最新`next-step.json`；以远端校验结果确认同步完成 |

历史报告保留其生成时状态，不改写原哈希证据。低磁盘/待Train渲染/校准首次拒绝是历史事实，当前状态以上表为准。

## 已授权执行顺序

1. 按用户授权另建批准合同：主目标Train/Validation两ROI各+0.5dB/+0.01 SSIM等为本轮工程门禁；不能冒充统计置信界。推理稳定性与视觉门禁可用于限定单场景探索，但训练波动仍未知。
2. 资源增长限制采用高斯数、主训练wall和每rank reserved相对signed的2倍。终止/失败门禁不得变成新的剪枝或高斯cap算法。既有signed runner保持原样；新候选入口已实现数量、逐rank显存与主阶段时间保护，已通过远端合成GPU与遥测验证；正式场景已资源越界并被主机OOM中断，见下方终态。
3. 用户已授权absolute fresh 30k/60k、SOR、Validation和2k/4k；须先通过验证。不从signed成品续训，不重复跑signed runner（它始终只跑关闭态）。
4. Git同步和远端验证已完成，正式臂曾启动但已在5418步中断；所有6个Validation与24个Train ROI、377个全局逐视角结果及失败均保留，不能事后改框/挑阶段。
5. 全部门禁通过至多`PASS_FOR_REPLICATION`；第二seed/场景、Test与默认推广均需独立授权。

此前4d673b4仅同步文档与结果。当前候选准备新增执行入口和资源监控，并为主训练／final-fit CLI增加显式启用的遥测参数；训练核心、配置、隔离库与signed执行脚本未改。未来协议分别记录实际执行提交和signed历史提交，并核对源码/配置/隔离库身份；不得重写旧实验protocol以伪造相同Git HEAD。


## 候选执行入口（验证后启动已授权）

`scripts/run_absgrad_candidate.py`有三个显式子命令：

- `gate-template --output <新文件>`：只生成`DRAFT_NOT_APPROVED`合同，不加载模型、不开GPU、不批准实验。
- `preflight --signed-experiment <signed实验根> --gate-contract <批准后的新合同> --gate-sha256 <合同SHA256> --output-dir <新实验目录>`：只读核对合同、signed证据、源文件和单变量配置，不创建候选目录；不是GPU环境验收。
- `execute`：参数与preflight相同；只有已批准合同才允许继续，限制主机`i-94B8D131`及既有远端仓库。批准标记是执行前置条件，不代替用户授权；不得自动把草案改为批准。

本地草案和轻量验证记录位于`outputs/analysis/absgrad-candidate-preparation-20261008-v1/`，不入Git。批准时另建合同文件，不覆盖草案或历史报告。模板绑定质量草案的SHA及其`proposed_quality_gates`、Train ROI SHA、两倍资源草案与原20/8/4GiB政策；不采用旧报告中的18GiB选项。若修改阈值，应先改合同/实现/测试并重新审核，不能仅改一个运行参数。

### 运行保护与身份

- 外层持有`outputs/.gpu.lock`；目录必须全新且位于`outputs/experiments/`下。没有自动resume。
- 启动核对干净且受Git跟踪的代码、训练核心相对于signed提交的差异、模型/配置/源协议哈希、训练code/environment hash、隔离rendering和现场两张L2。新CLI/监控代码由实际执行提交记录，不宣称全仓库与signed同提交。
- 保持fresh 30k/60k→SOR→377-view Validation→固定拓扑2k/4k Train-only；每步两个相机，主训练与Train-only相机序列摘要分别必须等于signed。评估不完整或held-out角色错误会阻止写完成标记。
- 仅候选显式传入`--resource-telemetry-dir`。每rank独立线程每10秒保存累计`torch.cuda.max_memory_reserved`，不改变训练循环、梯度、随机种子或策略；不使用`nvidia-smi used_memory`冒充PyTorch reserved。
- 监控增量读取每步高斯数和两个rank遥测，每10秒检查一次；进程正常退出也做最后检查。主训练2倍wall、主训练及Train-only的高斯数量/逐rank显存超限均失败，不加入cap或剪枝。
- 显存采样包括模型合并阶段，比signed原result中的训练峰值统计窗口更宽；这是保守停止保护。公平资源对照仍分别使用原生result的同口径峰值与新增监控峰值，不能混称严格同口径2倍测量。SOR及独立selection阶段沿用磁盘/6小时保护，没有逐rank reserved保护。
- 初始遥测最多等待600秒，已有遥测超过120秒未刷新、损坏、缺rank或退出时不完整均失败。这是待批准合同中的运行健全性限制，不是算法参数。10秒轮询不是硬实时显存上限，不能保证先于OOM拦截。
- 失败优先写取消标记，等待120秒，再向本任务进程组TERM，30秒后必要时KILL。主训练走既有checkpoint取消路径；final-fit原本不保存阶段checkpoint，不宣称能够恢复。保留失败目录，不重启服务、不停止其他任务。

### 验证范围

31项本地轻量回归覆盖配置单变量、草案拒绝、身份篡改、缺失/陈旧/损坏遥测、资源边界、增量日志、采样摘要、正常/异常子进程、停止升级、四阶段调度及默认入口。只运行标准库模拟、配置逻辑与短CPU子进程，没有模型训练、渲染、Test或GPU验证；Ruff通过。

完成标记为`absolute_training_complete_quality_pending`，不是质量通过。六个Validation ROI、24个Train ROI、全部视角指标与视觉否决仍须在获得候选结果后完整评审；本入口不自动渲染额外ROI、不挑选最好阶段、不自动产生`PASS_FOR_REPLICATION`。

2026-10-08授权补充：用户明确要求提交、Git同步、远端验证并允许开始训练。旧草案不改写；批准合同另存于`outputs/analysis/absgrad-candidate-launch-20261008-v1/`。现有SH3冒烟新增显式`--resource-telemetry`，验证真实双rank累计reserved、12步/24样本进度与结束记录后才进入正式训练。最新运行状态由该独立目录及面板任务记录确认，不能把授权或本地测试当作已启动证据。

### 正式启动观察

- 训练提交：`65513f129e99e2cc906bd52d7b75cb73b8700300`；后续文档更新不改变实验protocol。
- 执行任务：`20261008-163110-7086`；只读监控：`20261008-163110-d0f7`。
- 产物：`outputs/experiments/absgrad-absolute-20261008-v1/`；启动审计：`outputs/analysis/absgrad-candidate-launch-20261008-v1/startup.json`。
- 2026-10-08 08:33:50Z观察640更新、1,038,293高斯，两rank峰值reserved为1,486,880,768／1,507,852,288 bytes；两次观察从7步推进到640步。
- 132项远端回归、每rank58项SH3数值检查、12步/24样本合成trainer及真实显存遥测均通过。主训练、SOR、全Validation和Train-only是否完成，仍以之后的退出记录及完整评估为准。

## 最新终态：资源越界及主机OOM中断

2026-10-08 09:01:50Z复核，本轮已停止，GPU空闲，服务恢复。第5400步高斯数达到3,048,410（上限2,975,056）；最后rank 0 reserved为8.75GiB（上限约7.60GiB）。08:51:35取消标记已写入，08:52:17内核global_oom杀死rank 0 PID306870；面板服务随后以oom-kill退出并自动重启。不是助手重启，也不是整机重启。

最后记录5418更新／10836样本，无训练result、阶段exit记录或完整checkpoint；SOR、完整Validation和Train-only未执行。外层shell留下的exit-code=0不能替代完成证据，面板任务实际无正常结束标志。资源门禁已不通过，质量无法评估；不降低门禁、不自动resume或重跑。

证据入口：`outputs/analysis/absgrad-candidate-interruption-20261008-v1/{REPORT.md,terminal-audit.json}`。原启动观察保留为历史。运行中主机RAM保护和面板服务cgroup隔离暴露缺口；目前只做审计，未修改服务/训练逻辑，也未证明具体哪次内存分配导致OOM。


## 2026-10-09：另立完整质量探索协议

用户选择继续完整质量实验，并批准新质量探索计划。原候选资源失败结论保留，新协议不是原门禁的放宽后“通过”，也不产生`PASS_FOR_REPLICATION`。

- 新候选CLI显式加`--quality-exploration`（gate-template/preflight/execute）；独立schema2合同与旧profile互不接受，质量/ROI合同不改，原两倍资源只作报告。
- 仍fresh、同replay/初始化/seed1920/增密阈值与预算，只改absgrad；8个核心文件、生产库、signed runner不改，因此不追加signed训练。
- 新停止线：6,000,000高斯、每rank18GiB reserved、每阶段6h；磁盘20/8/4GiB不变。数量线只停止，不截断增密或剪枝。
- 主机RAM：启动/阶段须≥22GiB MemAvailable；运行低于6GiB或任务memory.current≥14GiB紧急停止。每2秒检查所有阶段；逐rankreserved仍每10秒采集，不宣称硬实时GPU保护。
- 新任务必须处于唯一瞬态systemd service，实际MemoryMax=16GiB、SwapMax=0、memory.oom.group=1，不能仍在gpu-panel.service里；不修改既有服务配置。紧急停止仅自身进程组TERM，5秒后必要时KILL，不写cancel文件进入完整checkpoint保存。旧profile保持原120/30秒行为。
- 先落不可覆盖的stage.failure.json再终止，外部只读监控记录systemd/面板退出与完成文件；不以shell零退出码代替实验完成。正常最终checkpoint/合并仍可能触发新预算，届时保留失败，不擅改训练核心。
- `scripts/evaluate_absgrad_pair.py`逐端点报告全部24Train/6Validation ROI、377逐视角raw差、全局均值/P10、原资源倍率，输出固定三列ROI图片；引用/渲染身份严格核验。数值通过时仍待全部视觉审查，不自动填通过；raw全局与uint8裁剪指标分开。
- `scripts/smoke_absgrad_resources.py`仅限远端受限CPU隔离检查与128MiB小分配/96MiB任务cgroup OOM冒烟，不制造整机OOM、不使用GPU。

本地39项轻量回归、Ruff及diff检查通过，不包含模型训练/渲染/验证。2026-10-09只读复核两张L2空闲、实际面板/cloud/TURN active；主机可用约19.3GiB，低于22GiB。新正式任务尚未启动；远端验证、准入与启动以后置实际记录为准。无完整质量结果，实验仍未完成。旧中断报告上传拒绝未重试或绕过。


### 远端验收与当前阻塞（2026-10-09）

- 执行代码`8e15887dc297e6da791f8b1b2299d18ecbe82e4c`已Git快进同步。用户补充点名授权后才执行；首次权限拦截未绕过。
- **141项远端回归通过**（10.42s），只读preflight通过。独立cgroup隔离exit0；128MiB小分配/96MiB cgroup约束得到`oom-kill`/信号9，内核为`CONSTRAINT_MEMCG`，仅杀该测试unit。面板/cloud/TURN active且原启动时间未改变。
- 批准合同SHA`610e9002fba4c2c80c05707be11e75d5144292136b79b38bfeff072f59b19885`及明确授权经文件API上传，5份小日志取回SHA一致；旧中断审计上传没有重试。
- 01:49:52Z可用RAM **19.18GiB < 22GiB**，还差约2.82GiB；因此新SH3/GPU验证与正式训练尚未运行，正式输出目录尚不存在。最长20分钟只读等待准入，不创建训练/修改远端；满足后仍需重查所有准入与SH3，超时不降低门槛。
- 证据：`outputs/analysis/absgrad-quality-preparation-20261009-v1/{REPORT.md,verification.json,remote-*.log}`。该本地REPORT/verification尚未上传，远端同目录有原始日志、新合同/授权；完整质量实验仍未完成。


## 保存峰值定位与流式优化（2026-10-09）

旧诊断`outputs/analysis/checkpoint-memory-old-20261009-v1/`由e971ba1执行，6个case中3M正常/取消均受限OOM；控制器exit0只说明诊断收集完，不代表case全通过。1.5M主峰出现在gather和rank再打包，匿名内存为主。

新诊断`outputs/analysis/checkpoint-memory-stream-20261009-v1/`由57dfe50执行，0.5M/1.5M/3M/6M正常与取消全部成功；13高斯双rank全部状态/RNG解码等价。正常保存总/匿名采样峰值：1.5M为5.176/2.631GB（旧8.158/7.968GB），3M为8.463/3.378GB，6M为11.918/4.999GB。各指标最大值可能不同时间，不能相加；代表负载不是完整场景峰值保证。旧/新都在独立12GiB无swap cgroup，服务未重启。

148项远端回归通过；公开checkpoint合同与旧读取兼容，内部rank容器版本改变、训练核心hash改变。下一轮新增`run_absgrad_matched_pair.py`同提交fresh signed+absolute，schema3独立合同及匹配收据，旧signed仅历史参照；不绕过旧candidate身份门禁。真实SH3/12步模型等价及取消验收仍待运行。正式22GiB准入未改、完整配对尚未启动，不宣称质量通过。


### 新匹配实验预算已明确批准

2026-10-09真实SH3与12步小trainer/取消验收通过（任务20261009-103842-1e50）；历史小模型最大参数差1.28616e-6，在原容差内，生产gsplat未改。用户随后批准仅新schema3匹配实验改用**12GiB任务上限/18GiB主机准入/6GiB整机余量/11.5GiB任务停止/无swap**。旧schema1/2的22GiB政策不改，也不把旧资源失败翻成通过。新协议SHA`793b8e75e165503f3190d8c26220328eb9c623820ecdca6d3b737834f9392223`绑定已验证保存/SH3证据；所有新鲜signed/absolute阶段同代码、同环境，训练core为9f3878de…，完整预算、ROI/quality、6M/18GiB/6h界限不变。

新预算仅是受限执行决策：600万代表负载成功不保证完整训练一定完成，触线仍停止且不自动加预算。正式启动结果另记。


### 新同代码配对实验已启动（2026-10-09 02:57:49Z观察）

- 实际执行提交`b2c7ac443564f7fcca331eb0ef7f76f7fddf505e`；151项远端回归通过，合同/来源/保存验收身份及现场18GiB准入通过。
- 执行`20261009-105648-27e3`、只读监控`20261009-105707-0863`；目录`outputs/experiments/absgrad-streaming-matched-20261009-v1/`。
- fresh signed主训练已推进至108更新/216样本、999,901高斯。独立cgroup实际12GiB/无swap，观测memory.current约7.897GiB、oom事件0；面板/cloud/TURN active。这不是完整训练或质量结果。
- 顺序为新signed完整四阶段→同代码absolute完整四阶段→两个端点冻结ROI/377-view配对；旧signed/失败保留。后台等待终态，不自动重试、放宽限制或加载Test。
- 启动审计`outputs/analysis/absgrad-streaming-matched-launch-20261009-v1/startup.json`本地记录。活动远端保持执行HEAD；后续仅文档提交暂不向运行中仓库快进，避免阶段身份门禁失败。实验完成后再同步最新状态。


### 最新终态：新signed触及任务内存停止线（2026-10-09 03:05:21Z）

新匹配实验尚未完成。fresh signed在3005更新/6010样本、1,183,595高斯处停止；保护采样记录3000步。`train.failure.json`原因为`task_memory_at_11.5_gib`：memory.current为12,419,674,112 bytes（11.5667GiB），整机仍可用11.9626GiB。memory.events的max/oom/oom_kill均0，内核本轮无OOM；子进程TERM退出（-15）、外层exit-code1，面板/cloud/TURN均active且启动时间未变。GPU已空闲。

第3000步训练内Validation已完成377/377，两rank best-model文件存在；不是SOR后selection或最终checkpoint。无完整checkpoint、主训练result或signed/pair complete；absolute尚未创建，后续SOR/正式Validation/Train-only/配对质量未执行。

代表负载保存优化通过不等于12GiB覆盖完整训练。此次触线发生于首次Validation/best-model写出后，尚未进入最终流式checkpoint；缺少触线时memory.stat与RSS/PSS，具体内存组成未定位，不据此修改预算或宣称保存优化无效。面板任务缺结束标志，但原生退出/失败记录与systemd一致；其元数据缺失原因未确认。

证据：`outputs/analysis/absgrad-streaming-matched-interruption-20261009-v1/{REPORT.md,terminal-audit.json,train.failure.json,train.exit.json,driver.log,train.log}`，仅本地归档，4份下载文件SHA与远端一致。旧失败保留，不自动重跑/resume/加预算，质量不可评估，Test与默认未动。远端保持执行提交b2c7ac4，未同步后续文档或修改原始产物。

### 代码审查修复完成，实验保持停止

2026-10-09，用户要求按review修复、梳理代码，并明确“先不要启动实验”。已完成峰值统计、模型文件I/O、正式内存时间线、准入/后验终态及同代码资源分母修复；共享实验逻辑移入absgrad_experiment，candidate/matched变为薄CLI，新增版本化systemd启动/只读watch脚本。128项本地非模型检查通过，模型/GPU等价与实际资源复测未运行；不宣称真实训练内存问题已解决。

按用户对精确清单的授权，仅删除两次失败运行的4个best模型分片，合计928,323,568 bytes，删除后已核验；日志、协议、进度、成功signed及诊断证据保留。历史终态审计中的“文件存在”是删除前事实，新增清理回执单独记录。

代码核心哈希已改变且新增模块纳入身份范围，旧SH3/保存验收不得直接沿用；旧合同及历史产物未改。远端仍保持原执行代码，本轮没有Git同步或新任务。详见`docs/absgrad-code-review-20261009.md`后续修复节。

### 新修复验收通过，准备重新配对

2026-10-09，用户已点名i-94B8D131授权验证并继续实验。e9c5b95的185项远端回归、10个受限保存case通过；600万正常保存采样总峰10.63GiB，13点双rank组件/RNG等价。首次验收包装在保存结束后因noclobber打开既有GPU锁失败，未执行SH3；没有重跑保存。独立SH3任务20261009-143829-2008随后exit0，各rank58项、12步/取消通过，历史模型最大差8.94e-7，生产库未变。新证据core为7f85ff7f…，本次仅更新共享实验模块的证据引用、另建批准合同；资源预算/算法/ROI不改。新正式目录为absgrad-review-matched-20261009-v1，是否启动及完成仍以后置证据为准。

### 修复后新配对已启动（2026-10-09 06:53:15Z观察）

用户手动pull至32e3927后，新合同SHA f3d9c001…及授权文件API上传核对一致，54项绑定逻辑回归通过。执行20261009-145126-7f89、监控20261009-145126-8e33；全新目录outputs/experiments/absgrad-review-matched-20261009-v1。fresh signed真实训练已到481更新/962样本、999901高斯，任务总内存约4.957GiB，12GiB无swap限制有效、OOM事件0，新增内存组成/阶段时间线可读。

尚未完成signed，更无absolute或配对质量结果。后台等待终态；活动远端HEAD固定32e3927，不同步之后的文档提交，不自动重试或调整预算。旧failed unit在留存终态且MainPID0后仅清除failed状态以复用合同固定名称，没有重启业务服务或改变旧失败文件。
