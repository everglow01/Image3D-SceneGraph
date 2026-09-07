# SfM 小规模匹配审计与健康检查回归（2026-09-07）

## 状态与来源

- 原四格：`outputs/experiments/20260902_030611_a94e38dd-sfm-frontend-2x2-v2/`。
- 新审计：`outputs/experiments/20260907-sfm-small-match-audit-v1/`。
- 远端项目：`/usr/local/3dgs_new/Image3D-SceneGraph`。
- 已完成 CPU 审计 Job：`20260907-105158-ce45`，退出码 0。
- `results.json` 已下载至本地同名目录；完整健康报告保留在远端。
- 此次 CPU 审计使用本地未提交代码的隔离快照，不是某个新 Git 提交。实际运行包 `code-v2.tar.gz` SHA-256 为 `b14528d9c097f95b0cb3db8fe21b277d438eec582a1f2e3f57cd6561615d8027`；包内 `source-record.json` 保存逐文件 SHA-256。旧 `source-record.json` 与旧 `code.tar.gz` 是更早的本地准备产物，不能冒充实际运行包。
- 用户随后要求新代码只经 Git push/pull 同步；源码包不再作为发布方式。以后提交不能追溯性地冒充本次快照的来源。
- 没有运行新匹配、BA、Gaussian 训练或 Test RGB 评估。官方参考实现推理尚未运行；不要把“已导入包/已下载权重”记为对照通过。

## 健康检查真实回归

同一模型重新运行 `sfm_pose_health_v2`，原模型和 v1 报告不修改。

| 模型 | v2 结果 | 独立时间跳变数量 | 解释 |
|---|---|---:|---|
| SIFT + Brute-force 原始主体 | passed | 0 | 基线保持健康 |
| ALIKED + Brute-force 原始主体 | failed | 81 | 空间灾难与时间异常均存在 |
| ALIKED + LightGlue 原始主体 | failed | 46 | 保留原有 36 台空间异常候选 |
| ALIKED + LightGlue 旧 core repair | failed | 13 | 不再因 max/median=99.83 而逃过两层检查 |
| ALIKED + LightGlue Global | failed | 0 | 位姿异常，并明确记录空点云 |

v2 不降低空间阈值，不增加自动删除数量或轮数。时间异常独立于空间异常集合：至少 12 个正速度步长，单步速度达到正速度 p90 的 100 倍，且位移至少达到相机稳健中心距离的中位数，才触发新的 hard reason。时间异常本身不产生新的自动删除名单。旧 core repair 现在被拒绝，不等于已经修复了残余相机。注册间隙仍为软警告。

`sfm_frontend_factorial_v2` 选择注册数最多的原始模型（点数、模型路径依次打破并列），不再让两相机碎片掩盖 992 相机主体失败；产品门槛单独记录，缺失评估为未知，健康版本不同的实验不可混比。

## 八对图像的既有匹配检查

工具：`scripts/analyze_sfm_small_matches.py`。冻结图像帧号对：

```text
61–62、87–90、710–713、728–730、729–730、827–828
468–469、575–576（健康对照）
```

按 selection 中的文件名对应，不跨数据库按 image ID 硬配。读取关键点、候选对应和 verified 对应，记录 keypoint/selection/contract 哈希；不重新提取或匹配。以下误差来自 SIFT 主体估计位姿与 OPENCV 内参的 Sampson 残差，折算为近似像素单位，**SIFT 位姿不是地面真值**。

| 帧对 | 配置 | verified 数 | 残差中位数（近似 px） | 残差 >4 px 比例 |
|---|---|---:|---:|---:|
| 710–713 | SIFT + Brute-force | 0 | 不适用 | 不适用 |
| 710–713 | ALIKED + Brute-force | 68 | 255.11 | 100% |
| 710–713 | ALIKED + LightGlue | 242 | 3.03 | 35.12% |
| 728–730 | SIFT + Brute-force | 244 | 0.22 | 0.82% |
| 728–730 | ALIKED + Brute-force | 2,412 | 1.51 | 3.32% |
| 728–730 | ALIKED + LightGlue | 2,727 | 1.38 | 0.59% |
| 468–469 | SIFT + Brute-force | 2,400 | 0.24 | 0% |
| 468–469 | ALIKED + Brute-force | 828 | 0.24 | 0% |
| 468–469 | ALIKED + LightGlue | 2,878 | 0.27 | 0% |

- 710–713 提供了明显的两视图几何不一致线索；不能只用 verified 数量断言重建可靠。
- 728–730 的多数 ALIKED-LightGlue 对应与 SIFT 估计的极线关系并不严重冲突。灾难性相机不能全部简单解释为“匹配全错”；退化、视差与后续多视图求解仍需检查。
- 827–828 缺少完整 SIFT 参考相机，不能补造误差值。
- SIFT-LightGlue 的这八对在原库中都未留下匹配记录，标为未执行/缺失，不当成零匹配或几何失败。
- 这里的 4 px 只是离线摘要阈值，没有被加入生产门槛，也没有用它挑选新默认算法。

## ALIKED ONNX 预算：静态证据已确认

已安装模型 `external/colmap-features/aliked-n16rot.onnx` 的 SHA-256：

```text
39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547
```

ONNX 静态图的 `/Constant_91_output_0` 为 4096：

- `/TopK` 的 K 经 `/Reshape_5` 来自这个常量；
- `max_keypoints` 经 `/Clip_1`，上限 `/Cast_4_output_0` 同样来自该常量。

所以当前导出模型最多产生 4096 个候选，再过滤填充区域；传入 8192 不会突破这个限制。COLMAP C++ 端确实传入请求预算，并非本项目漏传参数。这解释了实际每图约 4092、最大 4096 的现象，但不能证明错误位姿由这个上限导致。

## 官方参考实现对照：已准备，未执行

脚本：`scripts/run_sfm_reference_audit.py`，对照范围为上述八对 ALIKED 特征：

1. 同一数据库关键点/描述子，官方 LightGlue 固定 9 层，关闭 depth/width 自适应、Flash 与 TF32，对照原库候选对应。
2. 两对代表性输入再以 CPU ONNX Runtime 执行相同 ONNX 模型，区分模型图与 C++ 包装差异。
3. 两张代表性图片对照官方 ALIKED N16Rot 与 ONNX，显式复现 COLMAP 的右/下复制填充和半像素坐标约定，检查预算和描述子。

脚本只加载本地权重，运行时不下载；匹配器权重用严格 state-dict 检查，源码版本和工作区状态写入报告。它目前只有本地辅助函数测试，实际推理仍待执行，不能视为已经验证可运行或结果一致。

已准备的资源：

- 已安装 LightGlue 仓库提交：`2f23ca2ea9638cecad7f7220795210fc6b8353c3`。
- 已安装 ALIKED 仓库提交：`683d7c65197395c0b3f01ebe76e1084a27e73a65`。
- 官方 LightGlue 参考权重来自 `https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/aliked_lightglue.pth`，47,632,827 bytes，SHA-256 `d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503`。
- 审计目录 `tool-deps/` 隔离安装 ONNX 1.17.0、CPU ONNX Runtime 1.20.1、NumPy 1.26.4 及其依赖，没有改写项目 venv。
- 本地 512 项 CPU 测试通过；未运行新 GPU 推理。远端任务创建因外部代码执行权限被拒绝，没有产生新的 reference Job ID。

后续代码必须先经 Git 同步；参考实验用新输出文件、独立任务锁和单卡串行，不能用前述快照执行替代新的 Git 来源记录。结果未出前，ALIKED 路径仍是嫌疑，LightGlue 的几何责任仍未隔离。
