# 阶段B独立结构前置结果

2026-10-27 · MCMC-B1-E007 · **结构前置通过；正式观察工具准入未运行。**

## 授权与身份

用户手动同步后仅授权阶段B远端结构前置。本次未执行`prepare`，未启动9组×200步正式实验，未读取真实Train/Validation/Test图像；未修改远端源码、旧产物或服务。

- 主机/仓库：`i-94B8D131` / `/usr/local/3dgs_new/Image3D-SceneGraph`。
- 执行提交：`539dc881a794b2523f2149c5bb2b69154583cb3d`，`main`，tracked干净。
- 执行任务：`20261027-095704-45a2`；唯一只读监控：`20261027-095704-b818`，均`finished=true / running=false / exit_code=0`。
- UTC执行：01:57:04—01:57:16；监控01:57:34完成。pytest **1 passed in 11.74s**；systemd运行12.462秒。
- 两张NVIDIA L2；Torch `2.3.1+cu121`、CUDA `12.1`，与冻结来源相符。原gsplat包哈希及预编译扩展检查通过。

## 验证了什么

每owner 8点、64×64合成视图，无观察与有观察各4步，真实双卡渲染、Adam和MCMC策略均执行。

- 被测观察调用前后参数、Adam、梯度及RNG摘要不变；主动消耗Python RNG的负例被拒绝。
- 真实rasterizer前反向、采样和Adam调用计数符合冻结预算；无观察路径没有cohort/audit产物。
- 独立全点模型的分组贡献参考与观察结果在原容差内一致；alpha守恒反例被拒绝。
- 解析正则梯度与独立float64 autograd参考一致。
- 保存的来源/目标映射、存活位图可独立重建首次死亡和恢复；仅为合成场景检查，不是主线真实Q1证据。
- 非Train解码在实际读取前被拒绝，终点在评价前截停。
- 单rank故意拒绝被两rank协同报告。这是预期负例，不是本次任务失败。

| 全局预算项 | 实际核验 |
|---|---:|
| 合成更新 | 8 |
| 训练相机forward | 16 |
| 分组贡献forward | 8 |
| 独立贡献参考forward | 2 |
| loss backward | 16 |
| 解析参考autograd调用 | 2 |
| Adam参数组step | 96 |
| 双rank协同拒绝 / RNG负例 | 各1 |
| 真实图像解码 | 0 |

## 资源与证据

- `MemoryMax/MemoryHigh/MemorySwapMax`均为`infinity`；任务上限30分钟，collective超时120秒；没有修改NCCL算法或传输参数。
- 远端全部运行文件合计582828字节。本地仅下载11份必要文本、10670字节，逐文件SHA一致；未下载合成数组。
- 总报告与双rank证据由远端`verify_preflight`复核；本机仅标准库核53份执行依赖SHA及下载文件，不执行模型验证。
- 前置报告SHA：`818f3122bc799a5b0cc98de6621fa438975f2606a0e597be94fe35652004726e`。
- 远端报告：`experiment/mcmc_shard_imbalance/branches/b1_observation_reliability/experiments/e007_observer_admission/runs/admission-preflight-20261027-v1/preflight.json`。
- 01:59:19 UTC收尾：两卡无计算进程、各3MiB，`image3d-cloud`仍active、健康接口`ok`；tracked源码未变，阶段A证据再次通过。
- 未单独采集运行峰值显存/主存，不用任务结束后的空闲值冒充峰值。

详见[运行记录](../records/admission-preflight-20261027-v1.json)。

## 对主线的含义与下一步

结构前置证明本次微型场景中的被测接口、观察状态保护和失败边界按约定工作，**不能替代真实200步独立C/U/O准入**，也不是统计等价或长期因果证明。旧Gate1不追认，Q1真实轨迹仍未取得。

下一步仅在明确授权后，使用本报告准备正式协议并运行固定9case；失败不加次数、不放宽阈值，通过后才分析同批Q1轨迹。报告绑定执行提交及全部执行依赖：在`539dc88`上可供后续prepare核验；若更换提交或修改依赖，必须按门禁重新取得匹配的结构前置证据，不能把本报告改写成新提交的结果。
