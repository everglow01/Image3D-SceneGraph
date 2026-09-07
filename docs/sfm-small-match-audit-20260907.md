# SfM 小规模匹配审计与健康检查回归（2026-09-07）

## 状态与来源

- 原四格：`outputs/experiments/20260902_030611_a94e38dd-sfm-frontend-2x2-v2/`。
- 新审计：`outputs/experiments/20260907-sfm-small-match-audit-v1/`。
- 远端项目：`/usr/local/3dgs_new/Image3D-SceneGraph`。
- 已完成 CPU 审计 Job：`20260907-105158-ce45`，退出码 0。
- `results.json` 已下载至本地同名目录；完整健康报告保留在远端。
- 此次 CPU 审计使用本地未提交代码的隔离快照，不是某个新 Git 提交。实际运行包 `code-v2.tar.gz` SHA-256 为 `b14528d9c097f95b0cb3db8fe21b277d438eec582a1f2e3f57cd6561615d8027`；包内 `source-record.json` 保存逐文件 SHA-256。旧 `source-record.json` 与旧 `code.tar.gz` 是更早的本地准备产物，不能冒充实际运行包。
- 用户随后要求新代码只经 Git push/pull 同步；源码包不再作为发布方式。以后提交不能追溯性地冒充本次快照的来源。
- CPU 健康回归与后续小规模 GPU 参考推理均已完成；没有重跑整段 COLMAP、BA、Gaussian 训练或 Test 评估。参考推理从 Git 提交运行，详见下文。

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

## 官方参考实现对照：已完成

整个后续过程只维护一个工作分支 `sfm-small-match-audit`；代码经 push、远端 fast-forward pull 同步，没有继续上传源码包。

| Job | Git 提交 | 结果 |
|---|---|---|
| `20260907-115416-829c` | `6590561` | 旧官方权重缺少非学习的 `confidence_thresholds` 缓冲区，严格加载失败；未开始推理 |
| `20260907-120426-6e63` | `df48ea0` | 匹配器与提取器对照完成，退出码 0 |
| `20260907-121135-ad09` | `3d2d772` | 增补关键点分数核验，退出码 0 |
| `20260907-121812-6501` | `c79ad31` | 两对零分数过滤控制完成，退出码 0 |

脚本：`scripts/run_sfm_reference_audit.py`，对照范围为上述八对 ALIKED 特征：

1. 同一数据库关键点/描述子，官方 LightGlue 固定 9 层，关闭 depth/width 自适应、Flash 与 TF32，对照原库候选对应。
2. 两对代表性输入再以 CPU ONNX Runtime 执行相同 ONNX 模型，区分模型图与 C++ 包装差异。
3. 两张代表性图片对照官方 ALIKED N16Rot 与 ONNX，显式复现 COLMAP 的右/下复制填充和半像素坐标约定，检查预算和描述子。

脚本只加载本地权重，运行时不下载；源码版本和工作区状态写入报告。旧参考权重省略当前实现中按层数计算的 `confidence_thresholds`，只为这个固定缓冲区补入模型构造值，其余权重保持严格 state-dict 检查。两个参考仓库的工作区均干净。

已准备的资源：

- 已安装 LightGlue 仓库提交：`2f23ca2ea9638cecad7f7220795210fc6b8353c3`。
- 已安装 ALIKED 仓库提交：`683d7c65197395c0b3f01ebe76e1084a27e73a65`。
- 官方 LightGlue 参考权重来自 `https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/aliked_lightglue.pth`，47,632,827 bytes，SHA-256 `d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503`。
- 审计目录 `tool-deps/` 隔离安装 ONNX 1.17.0、CPU ONNX Runtime 1.20.1、NumPy 1.26.4 及其依赖，没有改写项目 venv。
- 本地 512 项 CPU 测试通过；后续参考推理在单张 L2 上串行完成。

### LightGlue：没有发现明显的包装/权重接入偏差

- 八对相同数据库特征的官方 PyTorch ↔ 原 COLMAP 候选集合 Jaccard 为 **98.33%–100%**，不是位姿准确率。
- 710–713、468–469 两对的官方 PyTorch ↔ CPU ONNX 输出匹配集合完全相同。
- 可直接按名称对应的 **120 个 ONNX 权重张量**与参考权重逐值相同；不冒充所有导出张量逐值验证。
- 单次匹配观测的 PyTorch 峰值 allocated 约 1.17 GB；第一对包含冷启动时间。这不是整段 exhaustive 吞吐或显存基准，也不能解释 SIFT-LightGlue 的资源需求。

这削弱了“LightGlue 主要因为本项目包装错误而产生这些对应”的解释，但不证明算法在所有场景都正确，也不排除少量差异影响增量求解。

### ALIKED：固定 TopK 与包装未过滤零分点的组合缺陷

对帧 710，同一原始图片、同一右/下填充和 N16Rot 权重：

| 实现/预算 | 有效坐标点数 | 零分点 | 分数 >0.2 |
|---|---:|---:|---:|
| 官方 PyTorch / 8192 | 786 | 不适用 | 不适用 |
| CPU ONNX / 2048 | 2032 | 1205 | 791 |
| CPU ONNX / 8192 | 4032 | 3205 | 791 |

ONNX 把固定 TopK 选出的零分位置也返回，而 pinned COLMAP `src/colmap/feature/aliked.cc:240–280` 只取 keypoints/descriptors、过滤越界坐标，没有用第三个 scores 输出来剔除零分点。这个接口问题已得到模型输出与 C++ 源码双重证据，不是“ALIKED 算法本身一定有缺陷”的证明。

健康帧 468 的官方实现产生 6276 点；CPU ONNX 请求 8192 得到 4094 个有效点，全部 >0.2，其中 99.85% 在官方点的半像素邻域内。弱纹理帧的零分 TopK 并列可能随执行后端产生不同位置：CPU ONNX 输出与旧 CUDA 数据库不能视为同一特征集合。

### 两对控制：过滤无效特征不等于已经证明几何获益

同一 CPU ONNX 产生的特征，只切换 `score >0.2` 过滤，再送入同一官方 LightGlue：

| 帧对 | 未过滤输入数 | 过滤后输入数 | 未过滤匹配数 | 过滤后匹配数 |
|---|---|---|---:|---:|
| 710–713 | 4032 / 4032 | 791 / 399 | 22 | 22 |
| 468–469 | 4094 / 4085 | 4094 / 4077 | 2876 | 2875 |

未过滤时，两对均未观察到零分端点进入最终匹配；健康对有 4 个匹配涉及不高于 0.2 的非零分端点。过滤显著降低弱纹理对的无效输入，但本次没有测出几何质量增益。后续 48 帧同源回归扩大了检查范围：问题段原版 verified 对应中 43.17% 涉及至少一个非正分端点，因此此处“两对未观察到”不能外推；见 `docs/aliked-positive-score-experiment-20260907.md`。

**不能把原数据库上的 280 个匹配与 CPU 特征过滤后的 22 个匹配称作 A/B 改善**：原 GPU 与此次 CPU 的特征集合不同。这个控制真正可比的是 22→22。也不能从两对结果推断所有零分点都安全，或宣布原 SfM OOM 根因已全部解决。

最终报告：`reference-r4-results.json`，SHA-256 `3da3325da4a9bc73380cef7ed122669be32213167b2f85c667dde89487bbe223`。r2/r3 报告和首次失败日志均保留，不覆盖历史。

## 有界结论与未执行项

1. 判断漏洞修复通过真实模型回归，旧修复模型不再被误放行。
2. 已确认 ALIKED ONNX 输出/包装层需要修正的零分点问题与 4096 上限；LightGlue 对当前同一特征的参考输出高度一致。
3. 尚未修改 pinned COLMAP 二进制或发布修正版 ONNX；生产算法、特征 profile、默认配置均未更换。
4. 该下一阶段已在独立候选中完成：正分输出契约通过 CPU/CUDA 数值 smoke，但 48 帧问题片段的原版没有复现长视频灾难，修正版反而出现 `max/median=40.21` 的软位姿不稳定，不能推广。生产 COLMAP、ONNX 与默认 profile 仍未修改；完整结果见 `docs/aliked-positive-score-experiment-20260907.md`。
