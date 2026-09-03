# COLMAP / SfM 几何来源优化调研与分阶段实施路线

> 日期：2026-09-01  
> 状态：Phase 1 特征提取、Phase 2 局部匹配、Phase 3 图像对策略、Phase 4 两视图几何/View Graph 与 Phase 5 相机标定均已接入代码；当前 8192 点配置及相机 profile 的真实 geometry A/B 尚待运行证据。2026-09-03 的独立产品决策已将视频 `standard_v2` 设为新 Job 默认，历史 v1 仍可显式选择
> 范围：RGB 图像进入后，从局部特征提取、匹配、两视图几何验证、相机标定、SfM、三角化与 BA，一直到 3DGS 数据集之前  
> 约束：坐标仍是归一化任意单位；不使用 Test 选择算法；模型权重不得在 Job 运行时下载

## 1. 结论摘要

当前最合适的第一步不是引入另一套 Python SfM 框架，而是利用项目已经安装并固定的 **COLMAP 4.0.0 原生 ALIKED / LightGlue / Global Mapper 能力**：

1. 项目本地 `external/colmap-4-cuda/install/bin/colmap` 已报告 COLMAP 4.0.0、CUDA、ONNX，并提供：
   - `SIFT`、`ALIKED_N16ROT`、`ALIKED_N32` 特征；
   - `SIFT_BRUTEFORCE`、`SIFT_LIGHTGLUE`、`ALIKED_BRUTEFORCE`、`ALIKED_LIGHTGLUE` 局部匹配；
   - incremental `mapper` 与集成 GLOMAP 的 `global_mapper`。
2. 因此第一批实验不需要新增 HLoc、Kornia 或另一套 PyTorch 环境，也不需要自定义 COLMAP 数据库导入器。这样改动最小、资产合同不变、现有 Mapper/恢复/诊断/3DGS 都能继续复用。
3. **SIFT 不是当前 `colmap_matcher` 字段所表达的东西。** SIFT 是检测器/描述子；旧 `colmap_matcher=exhaustive|sequential` 实际控制“选择哪些图像对”，局部描述子则使用 `SIFT_BRUTEFORCE`。当前 API 已把四层拆开：
   - `sfm_feature_profile`：提取什么特征；
   - `sfm_local_matcher`：已选择图像对内部如何匹配；
   - `sfm_pairing`：选择哪些图像对；
   - `sfm_geometric_verification`：如何执行两视图几何验证；
   - `sfm_camera_calibration`：采用哪种相机模型与内参共享/分组策略。
4. 实施顺序应严格沿 RGB→SfM 流水线推进：
   1. 特征提取：SIFT 与 ALIKED N16Rot；
   2. 局部匹配：Brute-force 与 LightGlue；
   3. 图像对选择：exhaustive、video sequential+loop、vocab-tree；
   4. 两视图几何验证与 view graph；
   5. 相机模型与内参共享策略；
   6. incremental Mapper 与 Global Mapper；
   7. 三角化、BA 和增量恢复；
   8. SfM 阶段缓存/重放与端到端 3DGS A/B。
5. 第一实现批次建议只增加两个冻结特征 profile：
   - `sift_v1`：保持现有 SIFT、最多 8192 点、SIFT brute-force；
   - `aliked_n16rot_v1`：ALIKED N16Rot、最多 8192 点、ALIKED brute-force。

这样第一批只改变“特征来源”，图像对策略、几何验证、Mapper、BA、选帧和训练器都保持不变。LightGlue 留到第二批，避免一次同时改变提取器和匹配器而无法归因。

---

## 2. Phase 1 前基线与当前状态

### 2.1 Phase 1 前普通 COLMAP / Project 3DGS 基线

> 本节保留启动 Phase 1 时的基线，用于解释改动来源；Phase 1–4 完成后的有效实现见第 8 节实施记录。

当时基线路径是：

```text
上传 RGB / 视频选帧
→ COLMAP feature_extractor（未显式指定 type，因此为 SIFT）
→ exhaustive_matcher 或 sequential_matcher
   （只改变图像对策略，局部匹配仍为默认 SIFT_BRUTEFORCE）
→ COLMAP 两视图几何验证
→ incremental mapper
→ standard_v2 可选初始注册扩展
→ standard_v2 可选局部注册恢复
→ 最终 BA
→ image_undistorter
→ cameras.json + points3D.txt + points.ply
→ dataset/splits/3DGS initialization
→ graphdeco | project | mcmc
```

当时关键实现：

- `scripts/run_colmap_sparse.py`
  - `feature_extractor` 只传数据库、图片、single-camera、GPU 和线程参数；没有传 `FeatureExtraction.type`，所以实际是 SIFT。
  - `exhaustive_matcher` / `sequential_matcher` 没有传 `FeatureMatching.type`，所以实际是 `SIFT_BRUTEFORCE`。
  - 固定运行 incremental `mapper`。
  - 已记录 extraction/matching/mapping/undistortion 等阶段耗时。
- `src/image3d_scenegraph/geometry/adapters.py`
  - Project Gaussian 默认 `colmap_matcher=exhaustive`。
  - video 可实验性选择 `sequential`，并强制使用已固定 SHA-256 的 FAISS vocab tree 做 loop detection。
  - coarse progress 只有 feature extraction `0.16`、feature matching `0.20`、mapping `0.26` 等固定跳点。
- `scripts/run_vggt_ba_sparse.py`
  - VGGT-BA 局部窗口跟踪内部已使用 ALIKED + VGGSfM；
  - 但用于最终 COLMAP 数据库、三角化、遗漏图注册和 classified fallback 的特征仍是普通 COLMAP SIFT + brute-force。
- `scripts/run_colmap_vggt_dense.py`
  - 也有一套单独写出的 SIFT→pairing→incremental Mapper 命令。

### 2.2 Phase 1 前问题及当前状态

1. **已解决（Phase 1）：** diagnostics 不再把 detector 固定写成 SIFT，feature profile 和模型 provenance 已独立记录。
2. **已解决（Phase 2/3）：** local matcher 与 pairing 已拆成 `sfm_local_matcher` 和 `sfm_pairing`，不再用一个 `matcher` 名称混淆两层。
3. **已解决（Phase 1–5）：** 普通 COLMAP、COLMAP+VGGT、Project ordinary geometry 与 VGGT-BA 最终 COLMAP database 共享稳定 feature/local-matcher/pairing/geometric-verification/camera-calibration 控制。
4. **已解决（Phase 1–5）：** 前端按 feature → local matcher → pairing → geometric verification → camera calibration capability 显示并提交五条独立控制轴。
5. **已解决（Phase 4）：** diagnostics schema 3 区分 tentative candidates、候选保留内点、Guided 新增内点和最终 verified correspondences，并汇总 verified View Graph 的 degree/component/孤立节点与视频软 gap 桥接证据。
6. **已解决（Phase 5）：** diagnostics schema 4 增加 raw sparse camera profile/model/group provenance 与逐图 `camera_id`；完整焦距、主点、畸变、注册、track 和 reprojection 证据保存在独立相机诊断资产中，合理性 warning 保持 soft。
7. **仍开放：** `--max-image-size` 当前用于 undistortion，但 `run_colmap_sparse.py` 没有把它传给 `FeatureExtraction.max_image_size`。对于未预缩放的多图输入，提取阶段可能仍按原图运行；这会改变历史基线，不能在首次算法 A/B 中静默修改。
8. **仍开放（Phase 1.5）：** COLMAP database 尚未冻结“仅完成特征提取”的只读快照，每个 matcher A/B 仍会重复提取特征。

### 2.3 已具备但尚未产品化的能力

`scripts/setup_colmap_cuda.py` 已在安装验证中检查：

```text
AlikedExtraction.max_num_features
SiftMatching.lightglue_model_path
AlikedMatching.lightglue_model_path
FeatureMatching.guided_matching
FeatureMatching.skip_geometric_verification
image_list_path
ImageReader.camera_model
ImageReader.single_camera
ImageReader.single_camera_per_image
```

上述几何验证 marker 会在 `exhaustive_matcher`、`sequential_matcher`、`vocab_tree_matcher` 与 standard-v2 使用的 `matches_importer` 上分别检查；ImageReader marker 用于 Phase 5 共享/分组相机 capability。

本机安装的 COLMAP 4.0.0 help 还显示以下模型与官方 SHA-256：

| 资产 | COLMAP 默认来源 | SHA-256 |
|---|---|---|
| ALIKED N16Rot extractor | `aliked-n16rot.onnx` | `39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547` |
| ALIKED N32 extractor | `aliked-n32.onnx` | `a077728a02d2de1a775c66df6de8cfeb7c6b51ca57572c64c680131c988c8b3c` |
| ALIKED brute-force matcher | `bruteforce-matcher.onnx` | `3c1282f96d83f5ffc861a873298d08bbe5219f59af59223f5ceab5c41a182a47` |
| SIFT LightGlue | `sift-lightglue.onnx` | `e0500228472b43f92b3d36881a09b3310d3b058b56187b246cc7b9ab6429096e` |
| ALIKED LightGlue | `aliked-lightglue.onnx` | `b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d` |

这些默认值包含 URL，COLMAP 可自动下载，但本项目明确禁止 Job 运行时下载。因此接入前必须由 setup 脚本预下载、校验，并在命令中传本地绝对路径。

---

## 3. 候选局部特征与匹配方案

### 3.1 第一优先级：COLMAP 4 原生路径

| 实验 | 提取器 | 局部匹配器 | 主要价值 | 代价/风险 | 优先级 |
|---|---|---|---|---|---|
| 当前基线 | SIFT | SIFT brute-force | 历史可比、无需权重、成熟 | 弱纹理/模糊/大视角变化可能不足；图像对多时昂贵 | 保留默认 |
| 经典增强 | DSP/affine SIFT | SIFT brute-force + guided matching | 无 learned 权重；COLMAP 官方建议用于增加对应 | 提取与匹配更慢；必须作为完整 profile A/B | 中 |
| 提取器 A/B | ALIKED N16Rot | ALIKED brute-force | 轻量 learned 特征；旋转版本适合未知相机方向；原生 ONNX | 需要两份模型资产；权重许可需单独确认；质量依赖场景 | **第一批** |
| 更大提取器 | ALIKED N32 | ALIKED brute-force | 论文中通常比 N16 有更高容量 | 更慢；只有 N16Rot 不足时才值得增加 | 后续 |
| 纯匹配器 A/B | SIFT | SIFT LightGlue | 保持 SIFT 特征不变，可直接测 learned matcher 的净收益 | ONNX 模型、GPU/Provider 兼容和显存要验证 | **第二批** |
| learned 完整组合 | ALIKED N16Rot | ALIKED LightGlue | 提取与匹配均为 learned 路径 | 不能用于第一轮单因素归因 | 第二批 |

ALIKED 论文报告其稀疏可变形描述子头只在选定关键点计算描述子，并给出较高吞吐；但论文也指出大尺度/大视角联合变化、纹理分布不均和默认互近邻匹配仍可能失败。LightGlue 通过 adaptive depth/width 提前停止和关键点裁剪提升匹配效率，但剪枝收益与硬件、关键点数有关，不能直接把论文/RTX 3080 数字外推到本项目。

### 3.2 第二优先级：只有 COLMAP 原生候选不足时再引入

| 方案 | 优点 | 为什么不作为第一批 |
|---|---|---|
| HLoc（SuperPoint/DISK/SIFT + LightGlue/NN/AdaLAM） | Apache-2.0；成熟地写 HDF5、导入 COLMAP/PyCOLMAP、支持 retrieval pairing | 额外 Python/PyTorch/子模块/权重体系；项目已有 COLMAP 4 原生 ALIKED/LightGlue，第一批引入 HLoc 属于重复基础设施 |
| XFeat / LighterGlue | Apache-2.0 仓库；作者报告 VGA 下很高 CPU/GPU 吞吐；可做 sparse/semi-dense | 没有原生 COLMAP 4 接口或完整 SfM 示例；需要自建 DB 导入、模型缓存和几何验证桥接；应在原生候选失败后再评估 |
| DISK + LightGlue | Apache 路径、HLoc 支持、对部分视角变化有优势 | 需要 HLoc/PyTorch 集成，不能直接复用当前 C++ ONNX 路径 |
| SuperPoint + LightGlue | 研究与 HLoc 使用广泛 | LightGlue 仓库明确提示 SuperPoint 预训练权重/推理实现有单独限制；不作为产品优先候选 |
| LoFTR / RoMa 等 detector-free dense matcher | 弱纹理和宽基线可能更强，直接产生稠密对应 | 对每个 pair 成本和显存高；需将坐标去重为 COLMAP keypoint 索引；大规模视频图像对不适合作为第一步 |
| MASt3R-SfM / DUSt3R 类完整替代 | 可绕过传统特征→增量 SfM 的部分限制 | 属于新的完整几何后端而不是可替换的单阶段；许可、显存、尺度/相机合同和失败回退都需独立评审 |

现有 VGGT-BA 已经覆盖“基础模型相机 + ALIKED/VGGSfM track + BA”的研究方向。普通 COLMAP 路径不应再复制一套 VGGSfM 运行时来解决第一阶段特征 A/B。

---

## 4. 图像对选择和局部匹配必须分开

### 4.1 建议术语

```text
feature extractor  = 每张图检测 keypoint 并生成 descriptor
local matcher      = 对一个已选图像对匹配 descriptor
pairing            = 决定测试哪些图像对
geometric verify   = RANSAC/F/E/H 等两视图几何过滤
mapper              = 从 verified view graph 求相机和稀疏点
```

建议 API 最终使用：

```text
sfm_feature_profile       = sift_v1 | aliked_n16rot_v1 | ...
sfm_local_matcher         = bruteforce | lightglue
sfm_pairing               = exhaustive | sequential_loop | vocab_tree
sfm_geometric_verification = default_v1 | guided_v1
sfm_camera_calibration     = shared_opencv_v1 | shared_simple_radial_v1 | auto_grouped_simple_radial_v1
sfm_mapper                 = incremental | global
```

现有 `colmap_matcher=exhaustive|sequential` 暂时保留为兼容字段，但前端标签应写成“图像对策略”，不能再称为“关键点匹配算法”。在完成兼容迁移前，不删除旧字段，也不改变历史 Job 解释。

### 4.2 pairing 的推荐使用范围

- 多图、小规模、无顺序：`exhaustive` 是可复现基线。
- 视频：`sequential_loop`，邻域 overlap + descriptor-compatible vocab tree loop detection。
- 大规模无序多图：优先测试 `vocab_tree` 或 HLoc retrieval；不能使用“按文件名顺序”的 sequential 假设。
- `transitive_matcher` 可在已较强的图上补边，但会改变匹配图，应作为独立实验而不是默认尾处理。
- ALIKED 必须使用 ALIKED 专用 vocabulary tree；当前固定的 Flickr100K FAISS tree 是 SIFT tree，不能复用给 ALIKED。

---

## 5. SfM 标定、Mapper 与 BA 的改进方向

### 5.1 相机模型与内参共享

当前 Gaussian baseline 对所有输入固定 shared `OPENCV`。这对同一视频相机是合理的历史基线，但对多图输入存在两个问题：

1. 上传图片可能来自不同设备或变焦状态，强制单相机会错误共享内参。
2. 对普通镜头，`OPENCV` 的自由度高于 `SIMPLE_RADIAL`；在视图较少或弱纹理时可能自标定不稳。

Phase 5 已冻结为三个高层 profile：

- `shared_opencv_v1`：一个共享 `OPENCV` camera；这是 `project_3dgs` ordinary COLMAP 与 VGGT-BA 的历史默认；
- `shared_simple_radial_v1`：一个共享 `SIMPLE_RADIAL` camera；这是 direct `colmap` 与 `colmap_vggt` 的历史默认；
- `auto_grouped_simple_radial_v1`：仅 multi-image，以归一化 Make + Model + 可选 LensModel + FocalLength/35mm equivalent + 解码宽高 + EXIF Orientation 做项目自有确定性分组；设备/焦距/方向证据缺失或 EXIF 无效时每图独立。

自动分组不使用 COLMAP 原生 Auto，因为同一设备不同变焦状态可能被错误合并；也不使用文件名、上传顺序或目录结构推断相机。只读取分组所需标签，不读取/持久化 GPS、序列号或完整 EXIF。重复共享组分别用 `image_list_path + single_camera=1` 提取，singleton 合并为一次 `single_camera_per_image=1` 提取；随后严格校验数据库 image→camera 分区。

`colmap_vggt` 当前 dense unprojection 不支持 OPENCV 切向畸变，因此明确拒绝 `shared_opencv_v1`，不静默降级。VGGT-BA 是视频共享相机路径，可在 OPENCV 与 SIMPLE_RADIAL 间 A/B，但拒绝 auto-grouped；standard-v2 恢复帧用 `existing_camera_id` 继承共享相机。主点默认继续固定；只有注册完成且共享视图足够时，才把“最终 global BA refine principal point”作为后续独立实验。焦距、主点和畸变原始数值只进入诊断资产，不成为前端调参控件。

### 5.2 incremental Mapper 与 Global Mapper

COLMAP 4 已集成 GLOMAP 为 `global_mapper`。GLOMAP 作者报告相对 incremental Mapper 可快 10–100 倍并保持相当质量，但这是作者报告，不是本项目证据。COLMAP 文档同时提示：global mapper 对 outlier 更敏感，并依赖较好的焦距先验；没有可靠先验时应先在数据库副本上运行 `view_graph_calibrator`。

建议顺序：

1. 先让 ALIKED/LightGlue 产生更稳定的 verified view graph；
2. 在**多图或短视频**上对同一 database 做：
   - incremental mapper；
   - `view_graph_calibrator` + global mapper；
3. Global 通过后再考虑长视频；
4. standard_v2 的 seed-list、image-registrator、non-clearing triangulation 和 recovery 当前都是围绕 incremental 路径设计，不能直接宣称与 global mapper 等价。

Global Mapper 是最有潜力降低当前 Mapping 耗时的选项。历史远端证据中，Mapper 占 COLMAP 时间 `5596.110 / 6909.314 s`，因此只优化 SIFT 提取或 pair matching 不足以解决整个几何阶段耗时。

### 5.3 两视图几何、三角化和 BA

按低风险到高风险排序：

1. **先只记录，不调阈值**：每 pair candidate/inlier 数、inlier ratio、view-graph degree/component。
2. `guided_matching`：COLMAP 官方建议用于获得更多对应，但会增加成本；只作为独立 profile。
3. DSP-SIFT + affine shape：经典鲁棒性 profile，与 learned 路径平行，不叠加到基线。
4. `Mapper.ba_use_gpu=1`：COLMAP 4 支持 GPU BA；几何与训练串行，因此理论上可利用训练前空闲 GPU，但必须测实际显存、回退和数值一致性。
5. Global Mapper 已默认提供 GPU global positioning/BA 选项，先用其默认冻结配置，不提前暴露几十个 raw 参数。
6. CASPAR BA 仍是实验后端，不进入第一轮。
7. `tri_ignore_two_view_tracks=0` 可能增加小集合的点，但也会增加弱约束点；只能独立 A/B。

---

## 6. 性能优化与进度优化

### 6.1 性能优化优先级

1. **阶段重放而不是重复计算**
   - 特征提取后冻结只读 `features.db` 快照；
   - 每个 local matcher arm 从该快照复制到独立工作数据库；SQLite 文件不能以可写 hardlink 共享；
   - 匹配完成后再冻结 `matches.db`，供 incremental/global Mapper A/B；
   - 快照记录图片集合 hash、feature profile、模型 SHA、COLMAP build 和 descriptor type。
2. **减少无用 pair**
   - 视频 sequential+loop；无序集合 vocab retrieval；避免对上千帧做 O(N²) exhaustive。
3. **Global Mapper A/B**
   - 当前最大实际瓶颈是 Mapper，不是训练前所有步骤平均分布。
4. **有证据地限制提取分辨率/关键点预算**
   - `FeatureExtraction.max_image_size` 和 max features 应进入冻结 profile；
   - 第一轮保持历史输入像素和 8192 上限，不同时改变分辨率；
   - 后续再做 1280/2048/原图与 4096/8192 的成本曲线。
5. **GPU/线程**
   - 保持 COLMAP 自己的 GPU index 能力；不要为每张图启动 Python 模型；
   - 记录实际 GPU/provider，避免 ONNX 静默落到 CPU；
   - Mapper CPU 线程和 GPU BA 分开测，不能用训练显存估算几何显存。
6. **不重复下载/编译**
   - 所有 ONNX 与 vocab tree 由 dry-run-by-default setup 脚本安装并校验；Job 只探测和使用。

### 6.2 进度反馈

当前固定 `0.16→0.20→0.26` 只表明换了 stage，长时间停在一个数字会被误解为卡死。建议原子 `progress.json` 扩展为：

```json
{
  "stage": "colmap_feature_extraction",
  "elapsed_seconds": 123.4,
  "completed_images": 317,
  "total_images": 1000,
  "tested_pairs": null,
  "expected_pairs": null,
  "heartbeat_at": "..."
}
```

实现原则：

- feature extraction：从 database 中读取已写 keypoint/descriptor 的 image 数，可给精确 `completed/total`；
- exhaustive matching：可用 `N(N-1)/2` 给 expected pair；
- sequential/vocab：若 loop/retrieval 目标图尚未完全确定，则只显示 tested/matched pair 数与 elapsed，不伪造百分比；
- mapping：显示 elapsed、最新可安全获得的 registered count/sparse point count；若 COLMAP 没有稳定中间输出，只显示 heartbeat，不解析不稳定日志来制造百分比；
- 前端显示“阶段 + 计数 + 已用时间”，整体进度仍保持单调；
- 结束后 `colmap_timing.json` 继续提供权威分阶段耗时。

---

## 7. 诊断与消融合同

### 7.1 SfM diagnostics schema 4

Phase 1–3 的 schema 2 已拆开 feature、local matcher 与 pairing；Phase 4 的 schema 3 增加 geometric verification 与 View Graph。Phase 5 升级到 schema 4，把相机 profile/模型/分组摘要和逐图 `camera_id` 纳入 run provenance 与 run ID：

```json
{
  "feature": {"profile": "aliked_n16rot_v1", "extractor": "ALIKED_N16ROT"},
  "local_matcher": {"profile": "lightglue", "name": "ALIKED_LIGHTGLUE"},
  "pairing": {"name": "exhaustive"},
  "geometric_verification": {
    "profile": "guided_v1",
    "guided_matching": true,
    "skip_geometric_verification": false,
    "raw_parameter_policy": "colmap_build_defaults",
    "implementation": "colmap"
  },
  "camera_calibration": {
    "profile": "auto_grouped_simple_radial_v1",
    "camera_model": "SIMPLE_RADIAL",
    "sharing_policy": "focal_aware_groups",
    "initial_camera_count": 3,
    "final_camera_count": 2,
    "warning_count": 0,
    "diagnostics_path": "diagnostics/sfm_camera_calibration.json"
  },
  "view_graph": {
    "profile": "sfm_verified_view_graph_v1",
    "edge_definition": "nonempty_two_view_geometry"
  },
  "mapper": {"name": "incremental"}
}
```

Pair shard schema 2 继续分别记录 tentative candidates、其中通过验证的 candidates、Guided 新增 correspondence、最终 verified count 和 rejected candidates，避免出现“最终内点数大于候选数”时的错误内点率。feature shard 仍为 schema 1，pair shard 仍为 schema 2，因为相机分组不改变其语义。主 schema 1/2/3 继续可读：历史 Project Gaussian Job 推断为 shared OPENCV；`scripts/analyze_sfm_view_graph.py` 接受 schema 4 但仍只读 pair index，不修改 accepted Job。

独立 `diagnostics/sfm_camera_calibration.json` 来自 `image_undistorter` 前的 raw sparse model，记录初始/最终 camera groups、EXIF focal prior、注册率、具名 focal/principal-point/distortion、初始到最终焦距变化、track/reprojection 汇总与软合理性 warning。Gaussian undistorted PINHOLE camera 不得冒充原始标定结果。

前端现有关键点和 pair canvas 本身不依赖 descriptor 内容，因此继续复用同一 Viewer；run provenance 增加 geometric verification，匹配页增加紧凑 View Graph 摘要，不另建图形系统。

### 7.2 每个 geometry arm 必须记录的指标

**特征/匹配：**

- 每图 keypoint count 的 min/P10/median/P90/max；
- tested pair、candidate matches、verified inliers；
- pair inlier ratio 分布；
- view graph connected components、largest component ratio、degree 分布；
- feature/matching 秒数、database bytes、峰值 CPU RSS/显存（可取得时）。

**SfM：**

- registered count/rate；
- 视频 temporal coverage、最大 gap、gap 总超额；gap 继续是 soft warning；
- sparse point count、observation count、mean/median track length；
- mean/median reprojection error；
- 焦距比、畸变参数异常数、被拒绝相机数；
- mapping/triangulation/BA 秒数；
- standard_v2 扩展/恢复增益和成本。

**有 ground truth 时：**

- 经过 Sim(3) 对齐后的相机中心误差、旋转误差；
- 点云 accuracy/completeness；
- 可复用现有 `scripts/evaluate_eth3d_scene.py`，不能因算法切换另造不可比指标。

**以后具备训练资源时：**

- 固定同一 trainer/config/split，仅改变 geometry；
- Validation PSNR/SSIM、floater/haze 指标、训练 wall time 与 peak memory；
- Test 不参与选择；坐标仍是 arbitrary units。

### 7.3 A/B 公平性

每次只改变一个因子，并冻结：

```text
原始 RGB hash
视频 selection sidecar/hash
输入尺寸
关键点上限
相机共享策略
pairing
几何验证参数
Mapper/BA
random seed
Train/Validation/Test split policy
```

第一轮 SIFT 与 ALIKED 都使用最多 8192 点、相同图像、相同 pairing、相同 incremental Mapper。实际检测点数不要求相等，但必须记录分布。不要一开始比较“SIFT brute-force”与“ALIKED LightGlue + Global Mapper”，那只能作为集成 smoke，不能回答哪一项有效。

---

## 8. API、前端和 CLI 产品化设计

### 8.1 API

第一批只新增：

```text
sfm_feature_profile = sift_v1 | aliked_n16rot_v1
```

- 默认 `sift_v1`，保持现有行为；
- 仅在包含普通 COLMAP feature stage 的 backend/output 组合中生效；
- 未安装 ALIKED 模型时 Job 在排队前返回明确原因和 setup command；
- request、manifest metrics、run.log、SfM diagnostics 同时记录 requested/effective profile；
- 不公开 `min_score`、模型路径、descriptor 维度等 raw 参数。

第二批已新增：

```text
sfm_local_matcher = bruteforce | lightglue
```

由后端做 descriptor compatibility 映射：

```text
sift_v1 + bruteforce          → SIFT_BRUTEFORCE
sift_v1 + lightglue           → SIFT_LIGHTGLUE
aliked_n16rot_v1 + bruteforce → ALIKED_BRUTEFORCE
aliked_n16rot_v1 + lightglue  → ALIKED_LIGHTGLUE
```

第三批已将现有 `colmap_matcher` 的产品语义迁移为 `sfm_pairing`，旧字段保留兼容期。

第四批已新增：

```text
sfm_geometric_verification = default_v1 | guided_v1
```

两者都显式保持几何验证开启；Guided 只切换 `FeatureMatching.guided_matching=1`，其余 RANSAC/`TwoViewGeometry` raw 参数沿用同一 COLMAP build 默认值且不对外开放。

第五批已新增：

```text
sfm_camera_calibration = shared_opencv_v1 | shared_simple_radial_v1 | auto_grouped_simple_radial_v1
```

缺省由 JobStore 按 backend 保持真实历史行为；`auto_grouped_simple_radial_v1` 只允许 multi-image，`colmap_vggt + shared_opencv_v1` 明确不可用。request/queued manifest 记录 requested，completed effective 来自严格校验的 raw-camera 诊断，不以请求值臆测。

### 8.2 前端

在“几何实验”折叠区按流水线显示，而不是把所有参数平铺：

```text
1 特征提取      SIFT v1（默认） / ALIKED N16Rot（实验）
2 局部匹配      Brute-force（默认） / LightGlue（实验）
3 图像对策略    Exhaustive / Sequential + Loop / Vocab Tree
4 两视图几何    Default v1（默认） / Guided v1（实验）
5 相机标定      Shared OPENCV / Shared SIMPLE_RADIAL / Auto-grouped
6 SfM 求解       Incremental / Global（后续）
```

要求：

- 当前已启用第 1、2、3、4、5 项；SfM 求解仍保持 incremental；
- 每个实验项显示服务器 availability、缺失模型和 setup command；
- 结果页显示 requested/effective 值，历史 Job 缺字段时解释为 SIFT + brute-force + 当时 pairing + default verification + 当时 Project shared OPENCV + incremental；
- SfM inspector 的关键点/pair canvas 继续复用，不新增第二套 Viewer；
- 进度显示阶段、计数和 elapsed，不把固定百分比当作真实内部进度。

### 8.3 CLI

第一批 runner 建议接受 profile，而不是向 Job 暴露原始 COLMAP enum：

```bash
uv run python scripts/run_colmap_sparse.py \
  --image-dir INPUT \
  --output-dir OUTPUT \
  --feature-profile aliked_n16rot_v1 \
  --local-matcher lightglue \
  --geometric-verification guided_v1 \
  --camera-calibration auto_grouped_simple_radial_v1
```

内部再展开为固定 COLMAP 参数和本地模型路径。`run_vggt_ba_sparse.py`、`run_colmap_vggt_dense.py` 使用同一组 resolver，避免三处命令漂移。dense runner 在 `--colmap-model-dir` 复用既有文本模型时不允许声称切换 Guided 或 camera profile，因为该路径不会执行 matching/ImageReader；VGGT-BA 明确拒绝 auto-grouped。

---

## 9. 分阶段实施顺序

### Phase 0：最小前置审计（随 Phase 1 一起完成）

目标：在改变算法前保证 provenance 不说错。

- 把“pairing”和“local matcher”在代码、日志、诊断和 UI 中分开命名；
- 保持旧 `colmap_matcher` API 兼容；
- `/api/backends` 报告 learned feature capability；
- setup 脚本固定 ALIKED N16Rot extractor 与 ALIKED brute-force ONNX 的 URL、大小和 SHA；
- 禁止运行时下载和静默回退 SIFT。

验证：同一旧默认 Job 命令仍是 SIFT/SIFT_BRUTEFORCE，manifest 仅增加准确 provenance，不改变几何结果。

### Phase 1：特征提取器 A/B（代码已接入，待真实证据）

实现状态（2026-09-02）：`sift_v1|aliked_n16rot_v1` 已接入共享 resolver、三个 COLMAP runner、API/JobStore/backend capability、前端选择器、manifest 与 SfM diagnostics schema 2。默认仍为 SIFT；本机 ALIKED extractor/brute-force 资产已由用户显式安装并通过大小/SHA 校验，但尚无当前 profile 的真实 geometry A/B 结论。

冻结两个 profile：

```text
sift_v1:
  FeatureExtraction.type=SIFT
  SiftExtraction.max_num_features=8192
  derived local matcher=SIFT_BRUTEFORCE

aliked_n16rot_v1:
  FeatureExtraction.type=ALIKED_N16ROT
  AlikedExtraction.max_num_features=8192
  AlikedExtraction.min_score=0.2
  AlikedExtraction.n16rot_model_path=<local verified asset>
  derived local matcher=ALIKED_BRUTEFORCE
  AlikedMatching.bruteforce_model_path=<local verified asset>
```

接入范围：

- `scripts/run_colmap_sparse.py`；
- 普通 COLMAP Project Gaussian 路径；
- `run_vggt_ba_sparse.py` 的最终 COLMAP database stage；
- `run_colmap_vggt_dense.py`；
- 直接 `colmap` point-cloud/mesh adapter；
- API、JobStore、`/api/backends`、前端、manifest、SfM diagnostics；
- 单元测试 + 小型 geometry-only smoke，不启动 3DGS 训练。

注意：VGGT-BA 窗口 tracker 自己的 ALIKED 配置不是这个 selector；selector 只控制其最终 COLMAP database stage，诊断要分别命名，避免把两者混为一个算法。

### Phase 1.5：特征快照与重放

- 特征完成后冻结数据库副本和 record；
- matcher arm 从副本 copy，不可写 hardlink；
- 校验 input set、feature profile/model hash、COLMAP build；
- 先提供可信 CLI replay，再决定是否增加“从历史 Job 派生几何实验”的 API 生命周期。

### Phase 2：局部 matcher A/B（代码已接入，待真实证据）

实现状态（2026-09-02）：`sfm_local_matcher=bruteforce|lightglue` 已独立接入共享 resolver、三个 COLMAP runner、API/JobStore/backend nested capability、前端选择器、manifest 与现有 SfM diagnostics schema 2。默认仍为 brute-force；SIFT/ALIKED LightGlue ONNX 由同一 dry-run setup 脚本按大小/SHA 安装，代码任务不自动下载，尚无当前 8192 点 profile 的真实 geometry A/B 结论。历史 2026-08-13 的 2048 点/1280px 结果保留为风险证据：SIFT-LightGlue 几乎断图，ALIKED-LightGlue 能完成但匹配成本高，不能直接推广到当前配置。

- SIFT brute-force ↔ SIFT LightGlue；
- ALIKED brute-force ↔ ALIKED LightGlue；
- pairing 和 Mapper 保持固定；standard_v2 新增帧的 recovery extraction/`matches_importer` 复用同一 feature/local-matcher options；
- LightGlue min score 第一版使用 COLMAP 固定默认 `0.1`，不放到前端；
- 记录 COLMAP build、GPU 请求/index、模型 hash、匹配阶段 wall time和 inlier 分布；不伪造 COLMAP 未报告的 ONNX provider/per-pair 时间。

### Phase 3：pairing / retrieval（代码已接入，待真实证据）

实现状态（2026-09-02）：稳定字段 `sfm_pairing=exhaustive|sequential_loop|vocab_tree` 已接入共享 resolver、三个 COLMAP runner、API/JobStore、backend nested capability、前端、manifest 与 SfM diagnostics schema 2。新产品 Job 默认仍为 `exhaustive`；旧 `colmap_matcher=exhaustive|sequential` 保留并映射到 `exhaustive|sequential_loop`，冲突请求失败。video 支持 exhaustive ↔ sequential+loop，still/multi-image 支持 exhaustive ↔ descriptor-compatible vocab-tree。官方 SIFT 256K tree 与 ALIKED N16Rot 64K tree 已按 URL/大小/SHA 固定在 dry-run setup 中，但代码任务不自动安装 ALIKED tree，也尚无当前 feature/local-matcher profile 的真实 pairing A/B 结论。

- `sequential_loop` 固定为有序视频的 temporal overlap + descriptor-compatible vocab-tree loop detection；
- `vocab_tree` 固定为无序多图 retrieval；
- SIFT tree 不可用于 ALIKED，缺失/损坏资产不静默回退 exhaustive；
- standard_v2 recovery 继续使用独立、显式记录的 bounded temporal pair list，不伪装成重跑初始 pairing；
- 如 COLMAP vocab 对无序大图不足，再评估 HLoc retrieval，不先引入。

### Phase 4：两视图几何与 view graph（代码已接入，待真实证据）

实现状态（2026-09-03）：稳定字段 `sfm_geometric_verification=default_v1|guided_v1` 已接入共享 resolver/capability、三个 COLMAP runner、standard-v2 `matches_importer` recovery、API/JobStore、前端和 manifest。两者均显式设置 `skip_geometric_verification=0`；`guided_v1` 唯一算法差异是 `guided_matching=1`，未开放或修改 RANSAC raw 阈值。默认仍为 `default_v1`。

Phase 4 当时引入 schema 3；Phase 5 的当前 schema 4 保留同一几何语义并增加相机 provenance。verified edge 严格来自非空 `two_view_geometries`，View Graph 汇总 degree/component/孤立节点、候选保留/Guided 新增 correspondence 与视频 soft-gap bridge evidence。schema 1/2/3 继续可读，已有 Job 可用只读 analyzer 汇总，不写回 immutable attempt。

- 真实 A/B 固定 feature、local matcher、pairing、相机、Mapper、BA 和输入，只改变 `default_v1 ↔ guided_v1`；
- matching wall time 是 local matching + geometric verification 合计，不能伪称独立 RANSAC 耗时；
- 代码接入不构成质量提升或默认推广证据。

### Phase 5：相机标定 profile（代码已接入，待真实证据）

实现状态（2026-09-03）：稳定字段 `sfm_camera_calibration=shared_opencv_v1|shared_simple_radial_v1|auto_grouped_simple_radial_v1` 已接入共享 resolver、三个 COLMAP runner、API/JobStore、backend 顶层 capability、前端、manifest 和 SfM diagnostics schema 4。默认按 backend 保留历史行为：Project ordinary/VGGT-BA 为 shared OPENCV，direct COLMAP/COLMAP+VGGT 为 shared SIMPLE_RADIAL；没有默认推广。

- auto-grouped 采用项目自有焦距感知分组，而不是 COLMAP 原生 Auto；可靠完整证据相同才共享，缺失/无效证据每图独立；
- 分组只使用 Make/Model、可选 LensModel、焦距/35mm equivalent、解码尺寸和 Orientation，不读取/保存 GPS、序列号或完整 EXIF；
- `colmap_vggt` 拒绝 OPENCV，VGGT-BA/video 拒绝 auto-grouped，standard-v2 新帧继承 existing camera；
- raw sparse camera sidecar 记录 focal/distortion/registration/track/reprojection；COLMAP 默认 focal ratio/extra-param边界只产生 soft warning，provenance/model/assignment 漂移才是合同错误；
- 真实 A/B 固定 feature、local matcher、pairing、geometric verification、Mapper/BA、输入和 trainer/split，只改变 camera profile；代码接入不构成质量提升或默认推广证据。

### Phase 6：SfM 求解器

- incremental ↔ `view_graph_calibrator + global_mapper`；
- 先 still/短视频，再长视频；
- standard_v2 expansion/recovery 仍由 incremental 路径拥有，Global 不静默套用或替代。

### Phase 7：三角化与 BA 性能

- GPU BA；
- bounded BA cadence；
- two-view tracks；
- 只在前面 view graph 稳定后做，避免用 BA 修复错误匹配。

### Phase 8：端到端 geometry→3DGS

- 使用同一图像、split、trainer/config 比较 geometry；
- 先看 raw sparse/registration，再看 raw Gaussian，最后才看相同后处理；
- Graphdeco/Project/MCMC 的默认身份不因 geometry 实验改变。

---

## 10. 第一批代码成功标准

Phase 1 完成必须满足：

1. 旧请求未指定新字段时，生成的 COLMAP 命令与现有 SIFT baseline 等价。
2. 显式 `aliked_n16rot_v1` 时，命令使用本地 hash-verified ALIKED extractor 和 ALIKED brute-force 模型，不访问网络。
3. descriptor/local matcher 不兼容在 Job 创建或 runner 参数校验时失败，不进入 COLMAP 深处崩溃。
4. API OpenAPI enum、JobStore、adapter、CLI、frontend selector 和结果 provenance 一致。
5. `sfm_diagnostics` 不再把 ALIKED 写成 SIFT，也不再把 exhaustive/sequential 写成 local matcher。
6. 前端仍复用现有 SfM inspector，能显示 ALIKED keypoint 与 verified pair。
7. stage timing、keypoint/match/inlier/registration 指标可比较；不添加伪精确进度。
8. 自动测试覆盖默认保持、ALIKED 命令、缺模型、非法 profile、API forwarding、旧 manifest fallback 和前端 selector。
9. 只运行 geometry-only 小 smoke；未经用户明确授权，不自动启动远端 Job 或完整 3DGS 训练。
10. 没有证据前，SIFT、incremental Mapper、现有 pairing 和 Graphdeco 默认全部保持不变。

---

## 11. 明确不在第一批做的事情

- 不同时接入 SuperPoint、DISK、XFeat、LoFTR、RoMa 和 MASt3R-SfM。
- 不新增一套 HLoc Job pipeline。
- 第一批 SfM Profile 接入本身不改变视频默认；其后独立的 2026-09-03 用户决策已将 `standard_v2` 设为新视频 Job 默认，并保留显式 v1。
- 不开放几十个 SIFT/ALIKED/LightGlue/RANSAC/BA raw 参数。
- 不根据 sparse point 数单指标推广算法。
- 不运行时下载模型，不静默 fallback，不使用 Test 调参。
- 不宣称相机或点云具有 metric scale。

---

## 12. 资料与依据

### COLMAP / GLOMAP

- [COLMAP 4.0.0 release：ALIKED、LightGlue、Global Mapper、性能改进](https://github.com/colmap/colmap/releases/tag/4.0.0)
- [COLMAP CLI：feature/pairing/mapper 命令与大规模重建建议](https://colmap.github.io/cli.html)
- [COLMAP FAQ：相机模型、自标定、guided matching、global mapper 注意事项](https://colmap.github.io/faq.html)
- [COLMAP 相机模型与自标定建议](https://colmap.github.io/cameras.html)
- [COLMAP 4.0.0 ImageReader 相机共享实现](https://github.com/colmap/colmap/blob/4.0.0/src/colmap/controllers/image_reader.cc)
- [GLOMAP repository：global SfM 与 COLMAP database 接口](https://github.com/colmap/glomap)

### 局部特征与匹配

- [ALIKED paper](https://arxiv.org/abs/2304.03608)
- [ALIKED repository（BSD-3-Clause；权重需单独确认）](https://github.com/Shiaoming/ALIKED)
- [LightGlue repository / paper / Apache-2.0 权重说明](https://github.com/cvg/LightGlue)
- [Hierarchical-Localization（HLoc）](https://github.com/cvg/Hierarchical-Localization)
- [XFeat / Accelerated Features](https://github.com/verlab/accelerated_features)

### 证据边界

论文和仓库中的精度、FPS、10–100× 加速均为作者在其数据与硬件上的报告，不是本项目结论。候选是否进入默认路径，只能由冻结输入、单因素 geometry A/B、资源记录和后续同配置 3DGS Validation 证据决定。
