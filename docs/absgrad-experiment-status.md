# 绝对梯度消融：当前状态与执行边界

更新日期：2026-10-08。此文是结果索引；`codex.md`仍是唯一执行计划，训练合同见[gaussian-trainer-contract.md](gaussian-trainer-contract.md)。名称仅为“AbsGS启发的绝对梯度消融”，不是完整AbsGS复现。

## 已完成／未完成

| 项目 | 当前状态 |
|---|---|
| 双卡实现、SH3门禁 | CPU102项、每rank58项数值对照及12步合成trainer通过；不是质量结论 |
| signed正式基线 | 30k更新/60k样本→SOR→377-view Validation→2k/4k Train-only完成 |
| 观察区域 | 6个Validation ROI、16个Train相机/24个独立Train ROI已冻结并诊断 |
| 重复推理 | 同模型同16个Train视角一次额外推理，显示uint8逐通道与raw/display指标差均0 |
| 磁盘准入 | 2026-10-08 01:56:33Z可用302.30GiB，原20GiB线通过；每次执行仍须复核 |
| 质量／资源增长门禁 | 草案，未批准；重复推理不提供训练seed方差 |
| absolute正式臂 | 未授权、未启动；不能自动接续signed |
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

## 下一步，不自动执行

1. 确认质量门禁依据和数值：主目标Train/Validation两ROI各+0.5dB/+0.01 SSIM等仅为工程草案；不能冒充统计置信界。推理稳定性与视觉门禁可用于限定单场景探索，但训练波动仍未知。
2. 确认资源增长限制：草案为高斯数、主训练wall和每rank reserved相对signed的2倍。终止/失败门禁不得变成新的剪枝或高斯cap算法。既有signed runner仅实现磁盘/6小时保护，不宣称候选数量/显存增长实时保护已实现。
3. 明确授权absolute fresh 30k/60k、SOR、Validation和2k/4k。不从signed成品续训，不重复跑signed runner（它始终只跑关闭态）。
4. 实现候选执行控制并做必要回归后才启动；所有6个Validation与24个Train ROI、377个全局逐视角结果及失败均保留，不能事后改框/挑阶段。
5. 全部门禁通过至多`PASS_FOR_REPLICATION`；第二seed/场景、Test与默认推广均需独立授权。

本次仅文档提交与结果同步，不改变训练源码。未来协议须分别记录实际执行提交和signed历史提交，并核对源码/配置/隔离库身份；不得重写旧实验protocol以伪造相同Git HEAD。
