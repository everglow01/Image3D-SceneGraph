# ALIKED 交叉对齐、孤立分量边界与有界恢复（2026-09-08）

## 冻结范围

用户明确要求完成三项工作：

1. 对主体与 `model_7` 做朝向约束和共同三维点约束的交叉对齐；
2. 检查 299.101–305.424 秒孤立 38 帧分量的边界 RGB 与匹配；
3. 根据前两项结果选择并运行恰好一次有界重建实验。

约束为：实例 `i-94B8D131`，项目 `/usr/local/3dgs_new/Image3D-SceneGraph`；新代码只经 Git；读取不超过既有 446 帧；独立新数据库和输出；每个诊断最多 300 秒、每个算法阶段最多 1,200 秒；只用单张 L2；不覆盖旧数据、不改生产默认、不运行 Gaussian/Test。

## 任务一：三种独立交叉对齐

只读 Job `20260908-110135-02a9` 使用提交 `265d944dece0190e955ad50fcdbb43c0611442b4`，正式 `finished=true, exit_code=0`。新报告：

```text
outputs/experiments/20260907-aliked-positive-v1-r2/dense-bruteforce/cross-alignment-v1.json
SHA-256 2b90c5bb22d6003aba8e4e95396888f1aeb463cddda1a41883412c2fe841090d
```

报告复用上一轮主体/`model_7` 的九张共同相机以及 172 对双向唯一、至少由两张共同图支持的三维点。三种方法分别只拟合一次 proper Sim3，不做 RANSAC、不删点、不选择最优方法：

| Sim3 来源 | 相机中心残差 / 共同相机半径 p50 | 相机朝向残差 p50 | 共同点残差 / 同一半径 p50 |
|---|---:|---:|---:|
| 九个相机中心 | 0.2037 | 110.77° | 56.12× |
| 九个相机朝向 | 0.9765 | 2.24° | 39.44× |
| 172 对共同点 | 33.1777 | 51.38° | 3.65× |

朝向约束能解释朝向，却不能同时解释相机中心和点；点约束也不能同时解释相机。因此，上一轮的矛盾不只是近线状中心导致旋转拟合不稳定，仍没有一个坐标变换同时支持三类证据。共同相机只覆盖 1.997 秒，且两模型独立优化出不同 OPENCV 内参；这不是 ground truth 判决，但继续拒绝把 `model_7` 当作可直接合并的可靠桥。

## 任务二：38 帧孤立分量边界

同一只读 Job 写入：

```text
boundary-v1/results.json
SHA-256 1beba232feb2512d9ca256fbe2eb12ff3d2bfb47d0d591c0493ab2b007a78228
boundary-v1/boundary-pairs.jpg
SHA-256 53f895c9c0c59c45e38600639acc73b40282c2f627232c53e5ca1f0040128bc8
```

- 38 帧正分特征数为 120–2,330，中位数 738.5，不是所有帧都没有特征。
- 它与其它 408 帧之间共有 15,504 个 exhaustive 图像对；旧 Brute-force 数据库中，所有跨分量对的 stored candidate 和 verified 数均为 0。
- 其中恰有 27 对跨分量图像的时间跨度不超过 2 秒：左边界 16 对、右边界 11 对。
- 最近左边界是 298.103↔299.101 秒（0.998 秒），特征数 414/622；最近右边界是 305.424↔306.755 秒（1.331 秒），特征数 124/310。
- 冻结预览显示左边界仍共享窗外城市、窗框和绿色墙面，右边界处于窗边向室内顶棚转向过程；这是可见重叠的定性证据，不是匹配真值。

源码核验补充了数据库语义：pinned COLMAP 在写库前将少于 `TwoViewGeometry.min_num_inliers=15` 的 candidate 集合清空。因此 stored zero 的严格含义是旧 ALIKED-Brute-force 没有提供达到冻结门槛的候选，不是证明原始描述子绝对一个提议都没有。

## 任务三：唯一有界重建验证

### 冻结方案

基于前两项，优先补匹配而不是合并模型或直接更换求解器。提交 `3c8eec2a959048112d78ebaa62d79b80ab24aeaa` 在运行前将协议写入 `codex.md` 和独占 `protocol.json`：

- SQLite backup 复制 446 帧正分 ALIKED + exhaustive Brute-force 数据库；
- 只删除上述 27 个原本为空的跨分量 pair rows；
- 全部特征以及其余 99,208 对 matches/two-view rows 的 canonical digest 必须保持一致；
- 只对这 27 对运行 pinned `ALIKED_LIGHTGLUE`，阈值 0.1、`default_v1` geometry verification；
- 若没有新增 verified 边，不运行 Mapper；若有，运行一次相同 incremental Mapper：seed 0、4 threads、global BA tolerance `1e-6`；
- 仍用 `sfm_pose_health_v2` 与 12/70%/80% 产品门槛；不做 Global/core recovery、内参或阈值修改、模型合并、Gaussian/Test。

正式 Job：`20260908-130658-6f94`，`finished=true, exit_code=0`。结果：

```text
outputs/experiments/20260908-aliked-boundary-lightglue-v1/protocol.json
SHA-256 52896425b175005a419491e025adbfb777deb12851b0db1f8f97fc101c578725
outputs/experiments/20260908-aliked-boundary-lightglue-v1/results.json
SHA-256 cbcd1f3f669d4c6ffe683378b894078207f9bce689345427b7a959d709894cef
```

两份小报告已下载至本地同名 ignored 目录；数据库、模型和阶段日志保留在远端独立目录。

### 匹配结果

27 对中只有左侧最近边界对产生可写入结果：

```text
frame_c001788_pts26829271.jpg  298.103s  DB image 186
frame_c001794_pts26919117.jpg  299.101s  DB image 187
candidate 29, verified 24, config UNCALIBRATED (3)
```

其余 26 对仍为 stored zero。新增总量就是 1 对、29 candidate、24 verified。它使输入 verified 图从 408+38 两个连通分量变成一个连通分量，但两个端点当时都不在最终主体注册集合中；而且 24 小于 unchanged Mapper 的 `abs_pose_min_num_inliers=30`。即使这 24 个 verified 对应都能连接到已三角化且独立的三维点，单靠该边也达不到图像注册前的可见点门槛。本轮没有降低门槛来强行接受边界帧。

### Mapper 与产品结果

匹配阶段 19.46 秒，Mapper 632.34 秒，均未超时。旁路 `nvidia-smi` 采样仍没有可用显存值，不能作显存结论；Job 后核验 L2 为 0 MiB/0%。Mapper 输出 9 个模型，主体结果为：

| 指标 | 原 446 帧正分 BF | 增补 27 对局部 LG |
|---|---:|---:|
| 主体注册 | 247/446（55.38%） | 247/446（55.38%） |
| 时间覆盖 | 79.49% | 79.49% |
| 最大注册间隙 | 13.843s | 13.843s |
| 稀疏点 | 23,753 | 23,234 |
| 注册观测 | 287,115 | 281,490 |
| max/median 相机半径 | 3.4065 | 3.4143 |
| pose-health v2 | passed | passed |
| 产品门槛 / 接受 | failed / false | failed / false |

新旧主体注册图像集合逐名完全相同：共同 247，丢失 0，新增 0。新增边右端 image 187 仍位于一个与旧 `model_3` 完全相同的 25-camera 局部模型；左端 image 186 没有进入任何输出模型。它没有把该局部模型接入主体。

虽然注册集合相同，Mapper 并未产生字节相同的几何。两主体相机中心 Sim3 对齐后，残差 / 原主体相机半径为 p50 3.03%、p90 11.48%、max 47.28%；新内参 fx/fy 为 1124.15/1128.36，原为 1114.99/1116.14。没有真值，不能把差异直接称为精度退化；但稀疏点减少 519（2.18%）、观测减少 5,625（1.96%），且产品覆盖没有收益，因此没有推广理由。

源数据库前后 SHA-256 不变，脚本同时核验复制库中的特征与全部非目标匹配 digest 不变。生成数据库 SHA 与源数据库不同是目标 27 对按协议替换及 SQLite 文件布局变化的正常结果，不是源库被修改。

## 三项任务结论

1. 相机中心、相机朝向和共同点三种交叉拟合仍互相矛盾，`model_7` 不可直接合并。
2. 孤立 38 帧边界有视觉重叠，但旧 Brute-force 未提供达到 15 对应门槛的跨分量边。
3. 局部 LightGlue 仅恢复一条 24-inlier 边，修复了抽象输入图连通性，却没有满足注册支持或改善最终产品指标。**图连通不等于几何可注册。**

因此，本轮选择的唯一实验已经给出否定结果，应停止继续扩大同类局部 LightGlue 配对或事后降低 Mapper/verification 门槛。生产二进制、默认 matcher/profile、健康阈值及原产物保持不变；没有 Gaussian/Test。后续若重新开启研究，应提出新的可证伪假设（例如独立跟踪产生多帧连续轨迹，或固定且有依据的自标定对照），而不是继续在本轮结果后无界试配。

代码运行前全量验证为 **526 passed**（一个既有 Starlette 弃用警告），Ruff 与 `git diff --check` 通过。
