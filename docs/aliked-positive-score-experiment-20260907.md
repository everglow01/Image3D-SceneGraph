# ALIKED 正分输出契约与小片段几何回归（2026-09-07）

## 冻结范围

- 前置证据：`docs/sfm-small-match-audit-20260907.md`。本轮由用户明确要求“开始做下一轮”。
- 代码只经 Git 同步，复用 `sfm-small-match-audit` 一个工作分支。
- 原始数据：`outputs/experiments/20260902_030611_a94e38dd-sfm-frontend-2x2-v2/aliked-lightglue/`，selection SHA-256 为 `9c877d158c3b87051f09ca72d04c86a50cf18add59969d53483dba1a10ba8151`。
- 新实验根：`outputs/experiments/20260907-aliked-positive-v1/`。原数据库、模型与结果不修改。
- 只用远端单张 NVIDIA L2，串行执行；不训练 Gaussian，不消费 Test 评估，不做千帧 exhaustive 或自动恢复。

## 修正及版本边界

`scripts/patches/colmap-4-aliked-positive-scores-v1.patch` 仅修改 pinned COLMAP `aliked.cc`：读取已有 scores 输出，拒绝非有限值、剔除非正值，在原 valid-keypoint 循环中保持坐标转换和描述子索引。日志逐图记录候选数、非正分数数量和最终保留数。

**不额外用输出 `score >0.2` 筛选。** 原检测输入 min_score=0.2 保持，正的亚像素输出分数即使低于该值也保留。ONNX、权重、请求预算 8192 / 实际上限 4096 均不变。

构建工具 `scripts/build_colmap_aliked_score_filter.py`：

- 原 COLMAP commit：`8bac7b9aab8f1a86dac377538666b10d45544f47`，要求原源码干净。
- 在 `external/colmap-4-aliked-positive-v1/` 建立独立源码/构建/安装目录，拒绝覆盖已有目录。
- 复用原安装的 ONNX Runtime 1.24.1，CPU/GPU 架构、编译器及主要 CMake 选项保持。
- 记录项目提交、补丁 SHA、原/新二进制 SHA、ONNX Runtime SHA 与 source diff；原安装二进制哈希构建前后必须不变。
- 不修改 production binary resolver、API profile、前端默认值。此候选仅供实验，不能冒充原 `aliked_n16rot_v1` 干净证据。

构建 Job `20260907-131309-0f33`，代码 `f195524`。构建期间逐文件比较两端 PoseLib source，内容一致；构建结果以下方实测为准。

## 实验协议

工具：`scripts/run_aliked_score_filter_experiment.py`。每个阶段拒绝覆盖输出，保存完整命令、退出码、耗时、超时状态与两秒采样的整卡显存峰值；显存不是精确 allocator peak。

### 提取 smoke

- 同一组 468、469、710、713 帧，逐图校验原 selection 中的 SHA。
- 原版/修正版分别以 CPU 和 CUDA 提取，全部写入新数据库。
- 以文件名对齐，不使用跨库 image ID；修正版每一点必须在原版集合中找到数值对应（位置误差 <0.01 px，描述子最大分量误差 <1e-4）。这些是数值一致性验收，不是几何质量门槛。
- 同时验证行数一致、输出数量不增、日志确实移除非正分候选，日志 retained 数量与数据库一致。
- smoke 不通过时不启动 geometry；两阶段必须使用同一构建与源 selection。

### 小片段 geometry A/B

| 片段 | 原 selection 帧号 | 数量 |
|---|---|---:|
| 问题区 | 696–743 | 48 |
| 健康对照 | 448–495 | 48 |

每段三格：原 ALIKED-LightGlue、正分修正 ALIKED-LightGlue、SIFT-Brute-force 对照。

- 每格重新提取和匹配，不修补旧数据库或复用旧匹配。
- **只有 ALIKED 提取二进制改变**；两格匹配、Mapper、model_converter 均用原安装二进制。
- 相同 exhaustive、`default_v1` verification、shared OPENCV、seed=0、4 threads；Mapper 保持 Gaussian baseline 的 global BA tolerance=1e-6。
- 每个提取/匹配/求解命令最多 1200 秒；超时是资源/未完成结果，不是 pose failure。
- 评估全部原始 incremental models，以注册数最多、其次点数、最后路径选代表模型，不能用小碎片掩盖主体。
- 使用原 `sfm_pose_health_v2` 与 12/70%/80% 产品门槛；时间覆盖只针对当前片段，不能宣称全视频覆盖。>2 秒间隙仍为软警告。
- 保存每个模型的健康报告、时间线、相机与稀疏点 text。无 Global、core repair 或其它 solver/阈值切换。

## 实测结果

尚未完成；本节在实际 Job 结束并核对结果后更新。

## 结论边界

去除占位点、数值索引正确、减少计算输入、改善几何是不同结论。原版若在 48 帧片段中未复现错误，只能说明本轮没有建立几何修复的因果证据，不能把两版均通过写成“修复原长视频”。SIFT 是对照，不是 ground truth；不会据此自动推广生产默认。
