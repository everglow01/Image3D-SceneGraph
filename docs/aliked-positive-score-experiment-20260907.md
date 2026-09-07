# ALIKED 正分输出契约与小片段几何回归（2026-09-07）

## 冻结范围

- 前置证据：`docs/sfm-small-match-audit-20260907.md`。本轮由用户明确要求“开始做下一轮”。
- 代码只经 Git 同步，复用 `sfm-small-match-audit` 一个工作分支。
- 原始数据：`outputs/experiments/20260902_030611_a94e38dd-sfm-frontend-2x2-v2/aliked-lightglue/`，selection SHA-256 为 `9c877d158c3b87051f09ca72d04c86a50cf18add59969d53483dba1a10ba8151`。
- 成功实验根：`outputs/experiments/20260907-aliked-positive-v1-r2/`。首次 `...-v1/` 保留两个基础设施失败现场；原数据库、模型与结果均未修改。
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

首次构建 Job `20260907-131309-0f33` 使用代码 `f195524`，编译和安装完成，但独立安装缺少 `libonnxruntime.so.1`，因此在 `colmap -h` 验证点失败且没有完成标志。没有删除或覆盖这个目录；`6c1ff7b` 增加同哈希 ONNX Runtime 库的隔离安装和严格 resume，`85df820` 修正 porcelain 状态解析。恢复验证 Job `20260907-134528-8881` 退出 0，最终 build record SHA-256 为 `c93047d8930549e9fcf9c9838b6c91de042ffc319973c3d8c507929b0854fa28`：

```text
patch                 eb39a571044322b8dd691bdded3bd77fa5d27b12b4bacd0d3541807151e3fc60
原 binary             20d23f7f45b3b3b2200d4abf61137ba0c51c1a2036936839b7ba34223ee493a5
候选 binary           a0c889e3d098380b180ad8a67770fde52098ec0bfd71c5727aea9b27ddd7b1ea
ONNX Runtime          ab76afb97c497fe47e230467d5d2ec21a7faf19dc5005dedef4d0ccab8f471f1
CUDA provider         33a95160a22804c9e06fff3f4e15cc37f3c8b16da11d088bf117dcb4731e912b
```

构建期间逐文件比较原/候选 PoseLib source，内容一致。原 binary 哈希在构建前后不变。

## 实验协议

工具：`scripts/run_aliked_score_filter_experiment.py`。每个阶段拒绝覆盖输出，保存完整命令、退出码、耗时和超时状态；整卡显存原计划每两秒采样，但运行期间所有 `nvidia-smi` 子请求都没有返回可用值，首次失败明确为超时。`9be0394` 将该旁路 telemetry 改为 fail-soft，所以结果中的显存为 null，不能作资源结论。

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

### 构建与 smoke

首次 smoke Job `20260907-134720-dbf9` 没有产生算法结果：旁路 `nvidia-smi` 采样超时抛出异常，`finally` 已终止当时的提取子进程。该失败现场留在 `...-v1/`，不能当作 ALIKED 失败。修正后 Job `20260907-135408-43f8` 使用代码 `9be0394`，退出 0；`smoke/results.json` SHA-256 为 `57c757fcb5d92f962118be5ee7fcd2a31bd6ed97abd431900af7417457f11c33`。

| 帧 | CPU 原版 → 正分版 | CUDA 原版 → 正分版 |
|---|---:|---:|
| 468 | 4,094 → 4,094 | 4,094 → 4,094 |
| 469 | 4,085 → 4,085 | 4,085 → 4,085 |
| 710 | 4,032 → 827 | 4,096 → 827 |
| 713 | 4,032 → 436 | 4,096 → 437 |

四帧在两个执行后端均通过：所有修正版保留点在同后端原版数据库中的坐标距离为 0，描述子最大分量误差为 0，关键点/描述子行数一致，日志 retained 数与数据库一致。CPU/CUDA 在帧 713 的非正值边界上相差一个正分点；没有把它改写为同一特征集。

### 六格 geometry

Job `20260907-135825-3d54` 使用代码 `9be0394`，正式状态 `finished=true, exit_code=0`。`geometry/results.json` SHA-256 为 `4b0b3808deaba7349b26681fa24bec1a9cc8a4461bf317edcc7899308b23e4a4`。

| 片段 | 格 | 特征总数 | candidate 对应 | verified 对应 | 主体注册 | 稀疏点 | pose v2 | max/median |
|---|---|---:|---:|---:|---:|---:|---|---:|
| 问题 | ALIKED 原版 | 196,451 | 317,124 | 287,720 | 48/48 | 12,161 | passed | 3.05 |
| 问题 | ALIKED 正分版 | 96,574 | 184,769 | 174,949 | 48/48 | 10,483 | passed | 40.21 |
| 问题 | SIFT 对照 | 186,301 | 77,939 | 74,474 | 36/48 | 6,485 | passed | 5.33 |
| 健康 | ALIKED 原版 | 196,324 | 983,801 | 967,869 | 48/48 | 18,427 | passed | 1.50 |
| 健康 | ALIKED 正分版 | 164,202 | 956,709 | 944,699 | 48/48 | 15,945 | passed | 1.52 |
| 健康 | SIFT 对照 | 475,496 | 754,842 | 747,248 | 48/48 | 40,510 | passed | 1.62 |

- 六个代表模型均无 v2 hard reason、时间 discontinuity 或 covisibility 分裂，positive-depth fraction 均为 1。两段 ALIKED 两格都注册全部 48 帧。
- 问题段原版本身没有复现千帧灾难；因此“两版均通过”不能证明修复了原长视频。
- 问题段正分版另外产生一个 4-camera fragment，主体仍按冻结 representative policy 选 48-camera model。
- SIFT 问题段主体注册率为 75%、片段时间覆盖为 100%，满足 12/70%/80% 门槛；中间有 5.324 秒注册间隙，按现有合同仍是软警告。
- 冻结 Job 的 `clip_product_passed` 同时包含产品门槛和 pose 条件。提交 `f9a302e` 已把未来报告改为独立的 `product_gate_passed` 与 `acceptance_passed`；没有回写本次 JSON。六个本次代表模型两项实际都通过。

问题段存在阻止推广的软异常：正分版主体中 frame 712 的相机中心距离达到 median 的 **40.21×**，而原版最大为 3.05×；正分版 `p99/median=10.11`、`max/p99=3.98`，原版分别为 3.03 和 1.00。正分版相邻平移速度 p90 为 4.276 arbitrary units/s、max 为 122.451，原版为 1.791/19.791。max 只约为本臂 p90 的 28.6×，没有达到冻结的 100× catastrophic 时间门槛，所以 health PASS 与该软风险并不矛盾。坐标仍是任意单位；这些比率是尺度不变量。当前证据不能断言 frame 712 的物理位姿一定错误，但明确不能称几何改善。

### 只读端点与轨迹归因

提交 `a33b46a` 增加同源数据库的只读比较；Job `20260907-152451-3bca` 退出 0。`comparison.json` SHA-256 为 `37b9e3b9809f28535c803470f5d9e355a7bf2379488d65e5da94dea2a484aae8`，并绑定上述 geometry result SHA。96 张图的正分特征都按原顺序映射回原版数据库，最大坐标和描述子误差均为 0，所以原版中映射不到正分版的特征可归为非正分端点。

| 片段 | 表 | 原版对应 | 涉及非正分端点 | 原版剩余 | 正分版 | 剩余集合 Jaccard | 逐 pair Jaccard p50 |
|---|---|---:|---:|---:|---:|---:|---:|
| 问题 | candidates | 317,124 | 142,436（44.91%） | 174,688 | 184,769 | 75.09% | 32.47% |
| 问题 | verified | 287,720 | 124,222（43.17%） | 163,498 | 174,949 | 77.16% | 44.22% |
| 健康 | candidates | 983,801 | 28,487（2.90%） | 955,314 | 956,709 | 96.84% | 99.16% |
| 健康 | verified | 967,869 | 24,895（2.57%） | 942,974 | 944,699 | 97.28% | 99.94% |

因此早先两对控制“没有零分匹配端点”的结论只适用于那两对，不能外推：问题片段中非正分占位点大量进入 candidate，且有大量通过两视图验证。过滤后也不是简单删除这些边；问题段 retained verified 集合中 16,089 条仅原版保留、27,540 条仅正分版产生，说明输入 token 变化实质改变了 LightGlue 输出或后续验证。

相机中心按共同 48 张图做无反射 Sim3 对齐，只量化两个解的差异，不把任一解当真值：

| 片段 | Sim3 后 residual / 原版 median radius | p50 | p90 | max |
|---|---|---:|---:|---:|
| 问题 | 原版 ↔ 正分版 | 108.67% | 203.21% | 258.58% |
| 健康 | 原版 ↔ 正分版 | 0.33% | 0.66% | 1.40% |

健康段说明过滤在足够支持的区域基本保持了同一轨迹。问题段两格则收敛到根本不同的解；结合 SIFT 在弱区缺失 12 张注册，合理解释是该区可靠特征本来就不足，原版的 hard-health PASS 也不能充当物理正确性的证明。已确认非正分点污染 view graph，但不能因为原版解表面更紧凑就保留已知无效输入，也不能因为修正版剔除了它们就宣称新解正确。

### 计算量观察

| 片段 | 指标 | 原版 → 正分版 | 变化 |
|---|---|---:|---:|
| 问题 | 特征输入 | 196,451 → 96,574 | -50.84% |
| 问题 | candidate 对应 | 317,124 → 184,769 | -41.74% |
| 问题 | verified 对应 | 287,720 → 174,949 | -39.19% |
| 问题 | matching 时间 | 416.13s → 184.17s | -55.74% |
| 问题 | Mapper 时间 | 196.70s → 109.61s | -44.28% |
| 健康 | 特征输入 | 196,324 → 164,202 | -16.36% |
| 健康 | verified 对应 | 967,869 → 944,699 | -2.39% |
| 健康 | matching 时间 | 374.13s → 347.59s | -7.09% |
| 健康 | Mapper 时间 | 118.63s → 118.95s | +0.27% |

这是每格单次运行的 wall-clock 观察，不是性能基准；CUDA/ORT 冷启动不同，且显存采样全部不可用。可以确认无效输入显著减少，但不能把更快和更少对应自动解释为更准确。

## 结论边界

去除占位点、数值索引正确、减少计算输入、改善几何是不同结论。本轮前两项成立，也确认非正分点真实污染了弱区 view graph；几何改善不成立。原版在 48 帧问题段没有复现千帧灾难，两版也没有物理真值，只能判定“过滤契约正确但单独推广被阻止”。SIFT 是场景支持对照，不是 ground truth。生产二进制、ONNX、profile 与默认值保持不变；下一次若继续归因，应先在同一 48 帧输入上做原版/正分版 **ALIKED brute-force** A/B，以移除 LightGlue token-context 变量，而不是直接重跑千帧或 Gaussian。
