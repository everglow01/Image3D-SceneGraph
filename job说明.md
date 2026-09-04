# Job 资产、目录结构与保留策略

> 适用范围：当前仓库实现（审查日期：2026-09-04）。
> Job 根目录：`outputs/jobs/{job_id}/`。
> 权威接口：`manifest.json`；本文中的目录树是实现说明，不替代 `docs/manifest-schema.md`。

## 0. 先读这五条

1. **前端和外部消费者只能把 `manifest.assets` 中存在的角色视为稳定资产。** 某个文件物理存在，不代表它有稳定路径或兼容性承诺。
2. **`manifest.inputs` 与 `manifest.assets` 不同。** `inputs` 是原始上传文件清单；`assets` 是成功处理后公开的产物角色。
3. **成功 Job 不会保留整个运行 workspace。** Worker 只发布 `frames/`、`geometry/`、`gaussian/`、`diagnostics/`、`semantic/`、`scene_graph/`、可选 `navigation/` 和 `logs/run.log`，随后删除剩余 workspace。
4. **失败或取消不等于删除中间数据。** 正常捕获到的失败/取消会把 workspace 保留为 `lifecycle/attempts/{attempt_id}/partial/`，但这些文件不会写入 `manifest.assets`。
5. **所有重建坐标目前都是归一化任意单位。** 除非未来有经过独立评估的尺度恢复证据，否则不能把距离、眼高、移动速度或碰撞尺寸称为米制或真实尺寸。

---

## 1. 三种“资产”概念

| 类别 | 在哪里声明 | 是否稳定 | 典型例子 |
| --- | --- | --- | --- |
| 原始输入 | `manifest.inputs[]` | 输入合同稳定 | `input/images/room/001.jpg`、`input/video.mp4` |
| 稳定输出资产 | `manifest.assets.{role}` | **是；角色名稳定，实际路径以 manifest 为准** | `scene_splat`、`point_cloud_aligned`、`sfm_diagnostics` |
| 内部保留文件 | 只存在于磁盘目录 | 否 | `frames/progress.json`、Validation 预览、`gaussian/export/.../bundle.json` |
| 中间工作文件 | workspace 或 failed partial | 否 | `colmap/database.db`、原始 sparse model、undistorted 图像副本 |

重要推论：

- 不要通过猜测 `gaussian/export/train-001/scene.ply` 来加载结果，应读取 `manifest.assets.scene_splat`。
- 不要因为 `diagnostics/` 下存在某个 JSON 就把它当成稳定 API；只有对应角色出现在 `assets` 中时才可靠。
- 同一稳定角色在历史 Job、不同 trainer 或未来 schema 中可以指向不同路径。
- Job 成功后仍可能保留没有 asset role 的审计文件；它们可供人工排查，但前端不应硬编码使用。

---

## 2. 当前实现的 Job 超集目录树

下面是**可能出现文件的超集**，不是每个 Job 都会同时包含所有目录。标记含义：

- `[通用]`：所有新 lifecycle Job 都会创建，内容可能为空；
- `[视频]`：仅视频 Project 3DGS；
- `[高斯]`：仅 `project_3dgs + gaussian_splat`；
- `[可选]`：取决于后端、输出类型或后处理；
- `[失败]`：失败、取消或中断证据；
- `[内部]`：物理保留，但不是稳定 asset role。

```text
outputs/jobs/
├── {job_id}.zip                              # 按需生成的整个 Job 下载包，不在 Job 目录内
└── {job_id}/
    ├── manifest.json                         # [通用] 前后端稳定接口、状态、inputs、assets、metrics
    ├── request.json                          # [通用][内部] 持久化的规范化请求，retry 复用
    │
    ├── input/                                # [通用] 原始上传，生命周期内保留
    │   ├── images/                           # image / multi_image / panorama 输入
    │   │   └── ...                           # 清理过路径并去重后的原始文件
    │   └── {uploaded-video}.{ext}             # video 输入直接位于 input/ 下
    │
    ├── frames/                               # [视频；非视频 Job 中通常为空]
    │   ├── selected/
    │   │   └── *.jpg                         # 实际进入几何阶段的物理摆正关键帧
    │   ├── selection.json                    # 最终候选评分、拒绝原因、选帧及哈希
    │   └── progress.json                     # [内部] 视频抽帧阶段进度
    │
    ├── geometry/                             # 几何和通用后处理结果
    │   ├── points.ply                        # raw 点云；Project 高斯中是 accepted SfM sparse cloud
    │   ├── points_aligned.ply                # [可选] dominant-plane 对齐后的独立点云
    │   ├── cameras.json                      # [可选] 相机内外参与注册图像
    │   ├── mesh.glb                          # [mesh 输出] baseline 网格
    │   ├── mesh_{method}_{id}.glb            # [可选] API 生成的 A/B mesh variant
    │   └── depth/                            # 当前预创建目录；多数路径不发布稳定深度资产
    │
    ├── diagnostics/                          # 诊断证据；目录整体发布
    │   ├── alignment.json                    # 点云对齐变换、平面和 inlier 证据
    │   ├── mesh.json                         # baseline meshing 诊断
    │   ├── mesh_{method}_{id}.json            # mesh variant 诊断
    │   ├── sfm_frontend_contract.json        # [COLMAP][内部] Mapper 前冻结的前端策略合同
    │   ├── sfm_camera_calibration.json       # COLMAP 原始 sparse camera 校准证据
    │   ├── sfm_pose_health.json              # [高斯] 进入 undistort/训练前的 pose gate
    │   ├── sfm_pose_recovery.json            # [普通 COLMAP 高斯] bounded recovery 证据
    │   ├── colmap_timing.json                # COLMAP 子阶段、模型、DB 哈希和时间
    │   ├── sfm/                              # [高斯；fail-soft] 前端关键点/匹配浏览数据
    │   │   ├── manifest.json                 # schema 4，总索引、坐标和 run provenance
    │   │   └── runs/{run_id}/
    │   │       ├── features/
    │   │       │   ├── index.json.gz         # 图像到 feature shard 的索引
    │   │       │   └── shard-00000.json.gz   # 每 shard 最多 32 张图的关键点
    │   │       └── pairs/
    │   │           ├── index.json.gz         # 所有实际测试/保留图对及完整计数
    │   │           └── shards/
    │   │               └── shard-00000.json.gz # 每 shard 最多 256 对的内点/外点坐标
    │   ├── video_probe.json                  # [视频] 规范化 ffprobe、方向和源哈希
    │   ├── video_keyframes.jpg               # [视频] 仅展示用 contact sheet
    │   ├── video_keyframe_timing.json         # [视频] probe/analyze/materialize 时间
    │   ├── video_registration.json            # [视频] 最终注册率、时间覆盖和 gap
    │   ├── video_initial_registration_expansion.json # [COLMAP standard_v2]
    │   ├── video_registration_recovery.json   # [standard_v2] 最多两轮局部恢复证据
    │   ├── vggt_ba.json                      # [实验 VGGT-BA] 主诊断
    │   ├── vggt_ba_initialization.json       # [有效源仍为 VGGT-BA] Train-supported seed 统计
    │   ├── fusion.json                       # [COLMAP+VGGT] 融合诊断
    │   ├── visibility_graph.json              # [COLMAP+VGGT] 可见性图
    │   ├── scale_disagreement.json            # [COLMAP+VGGT] 尺度分歧
    │   ├── consistency.json                   # [COLMAP+VGGT] 一致性检查
    │   ├── vggt_groups.json                   # [COLMAP+VGGT][内部] 分组记录
    │   ├── depth_scale_graph.json             # [COLMAP+VGGT][内部] 深度尺度图
    │   ├── vggt_window_predictions.json       # [COLMAP+VGGT][内部] window 元数据
    │   └── vggt_window_predictions/           # [COLMAP+VGGT][可选内部] window 预测文件
    │
    ├── gaussian/                             # [高斯] 三种 trainer 的共同交付根
    │   ├── preparation/{trainer_attempt_id}/
    │   │   ├── dataset.json                  # 带 initialization 的有效 dataset contract
    │   │   ├── effective_config.json          # 完整 resolved config 与哈希
    │   │   ├── geometry_readiness.json        # CUDA 前几何就绪检查
    │   │   ├── initialization/
    │   │   │   ├── sparse.npz                # 冻结 Gaussian 初始化张量
    │   │   │   └── sparse.json               # 初始化选择/哈希/拒绝统计
    │   │   └── graphdeco-dataset/             # [Graphdeco] 外部 trainer 的隔离数据镜像
    │   │
    │   ├── attempts/{trainer_attempt_id}/
    │   │   ├── attempt.json                  # [Project/MCMC] trainer attempt provenance
    │   │   ├── artifacts/
    │   │   │   ├── model.pt                  # Validation 选中的项目格式 raw snapshot
    │   │   │   ├── result.json               # 训练结果、资源、checkpoint 引用
    │   │   │   ├── progress.jsonl            # 迭代/Validation/拓扑事件
    │   │   │   └── validation/
    │   │   │       └── iteration_NNNNNNNNN/  # [Project/MCMC][内部] cadence 预览 PNG
    │   │   └── checkpoints/                  # [Project/MCMC]
    │   │       └── iteration_NNNNNNNNN/      # 成功后同 attempt 只保留 terminal checkpoint
    │   │           ├── checkpoint.json
    │   │           ├── model.bin
    │   │           ├── optimizer.bin
    │   │           ├── scheduler.bin
    │   │           ├── densification.bin
    │   │           ├── rng.bin
    │   │           └── metrics.json
    │   │
    │   ├── replay/                           # 自包含、哈希绑定的冻结训练输入
    │   │   ├── dataset.json
    │   │   ├── replay.json
    │   │   ├── geometry/cameras.json
    │   │   ├── initialization/{sparse|dense}.{npz,json}
    │   │   └── colmap/undistorted/images/... # 仅已注册的 undistorted 图像，保留原相对路径
    │   │
    │   ├── native/{trainer_attempt_id}/graphdeco/ # [Graphdeco] 原生 checkout 输出、PLY、命令和日志
    │   ├── sor/{trainer_attempt_id}/         # [默认 fail-soft SOR]
    │   │   ├── filtered-model.pt
    │   │   ├── filter-record.json
    │   │   └── filter-mask.npz
    │   ├── evaluation/{trainer_attempt_id}/validation/
    │   │   ├── evaluation.json               # 稳定 Validation 评估角色指向此处
    │   │   ├── metrics.jsonl
    │   │   ├── record.json
    │   │   └── previews/*.png
    │   ├── export/{trainer_attempt_id}/
    │   │   ├── canonical.ply                 # 项目确定性 canonical PLY
    │   │   ├── scene.ply                     # 浏览器 INRIA-v1 derivative
    │   │   ├── export.json                   # 坐标、哈希、SH、scene bounds 等
    │   │   ├── camera_path.json              # Validation 相机路径描述
    │   │   ├── result.zip                    # 单次 Gaussian 交付包
    │   │   ├── bundle.json                   # result.zip 的外部哈希/字节数
    │   │   ├── .dataset.json                 # [内部] 写入 result.zip 的合同副本
    │   │   └── .effective_config.json         # [内部] 写入 result.zip 的配置副本
    │   ├── postprocess/{trainer_attempt_id}/ # [实验 VGGT visibility]
    │   │   ├── filtered-model.pt
    │   │   ├── diagnostics.json
    │   │   ├── filter-mask.npz
    │   │   └── result.json
    │   ├── evaluation/{trainer_attempt_id}/validation-vggt-filtered/...
    │   └── export/{trainer_attempt_id}-vggt-filtered/...
    │
    ├── semantic/
    │   └── masks/                            # 当前预留；语义系统仍是 placeholder
    ├── scene_graph/
    │   └── scene.json                        # 当前 mock scene graph，不代表真实语义完成
    │
    ├── navigation/                           # [高斯可选；完整三件套原子发布]
    │   ├── collision.glb                     # 隐形低模碰撞网格，不是展示 mesh
    │   ├── navigation.json                   # 归一化边界、spawn、player 合同
    │   └── diagnostics.json                  # Train-only 生成和质量证据
    │
    ├── logs/
    │   ├── run.log                           # 最终统一运行日志；稳定 asset role
    │   ├── attempt-001.log                   # [内部] Job attempt 状态日志
    │   └── attempt-002.log                   # [retry 时可能存在]
    │
    └── lifecycle/
        ├── attempts/
        │   └── attempt-NNN/
        │       ├── partial/                  # [失败/取消] 捕获后保留的完整未发布 workspace
        │       ├── partial_published/         # [中断] 已移到稳定路径但未完成 manifest 的隔离数据
        │       └── workspace/                # [硬中断可能残留] 尚未来得及归档的 workspace
        └── navigation/
            ├── attempt-NNN/
            │   ├── partial/                  # navigation 失败/取消
            │   └── partial_published/         # rename 后、manifest 前中断
            └── invalid_published[-N]/         # 校验失败的旧 navigation 隔离目录
```

### 2.1 这个目录树与早期规划示意的差异

`codex.md` 早期示意曾列出 Job 根级 `checkpoints/`、`evaluation/` 和 `exports/`。**当前实现不这样落盘**：

- checkpoint 在 `gaussian/attempts/{trainer_attempt_id}/checkpoints/`；
- evaluation 在 `gaussian/evaluation/{trainer_attempt_id}/`；
- export 在 `gaussian/export/{trainer_attempt_id}/`；
- 外层 `lifecycle/attempts/{attempt_id}` 是 Job worker 生命周期，不是 Gaussian trainer checkpoint attempt。

普通前端 Job 中，外层 ID 通常是 `attempt-001`，内层 trainer ID 默认是 `train-001`。两者不能混用。

---

## 3. 根目录和文件夹分别做什么

| 路径 | 用途 | 成功后 | 失败/取消后 |
| --- | --- | --- | --- |
| `manifest.json` | 唯一稳定索引；状态、请求身份、inputs、assets、metrics | 保留并写为 `done` | 保留；`assets` 清空，记录结构化 error |
| `request.json` | Worker/retry 的持久化规范化请求 | 保留，但不是 asset | 保留 |
| `input/` | 原始上传真源 | 永久保留 | 永久保留 |
| `frames/` | 视频最终选帧与选择证据 | 整目录发布 | 留在 partial/workspace |
| `geometry/` | raw/aligned 点云、相机和 mesh | 整目录发布 | 留在 partial；中断后可能进 `partial_published` |
| `diagnostics/` | 各阶段审计和前端诊断 shards | 整目录发布 | 留在 partial；不进入 assets |
| `gaussian/` | trainer、模型、评估、导出和 replay | 整目录发布 | 留在 partial |
| `semantic/` | 未来语义 masks 的预留根 | 当前通常为空但会发布 | 留在 partial |
| `scene_graph/` | 当前 mock 场景图 | 发布 | 留在 partial |
| `navigation/` | Walk 模式的完整碰撞/边界三件套 | 单独原子发布 | 移至 navigation lifecycle partial |
| `logs/` | 最终日志及每次 worker attempt 状态 | `run.log` + attempt logs 保留 | attempt logs 保留；workspace 内其他日志随 partial 保留 |
| `lifecycle/` | retry、失败、取消和中断现场 | 始终保留 | 始终保留，可能很大 |

### 3.1 `input/` 为什么会看起来重复占空间

运行开始时，Worker 会把 Job 根级 `input/` 复制到当前 attempt 的 disposable workspace：

```text
input/...                                      # 永久原始输入
lifecycle/attempts/attempt-NNN/workspace/input/... # 本次运行副本
```

- 一次成功：workspace 副本随 workspace 删除，只保留根级原始输入。
- 运行失败/取消：副本会随 `partial/` 保留，因此同一个视频或图片可能占两份空间。
- retry：再次创建新的 workspace 副本；旧 attempt 的 partial 不会被新 attempt 当作成功资产或 checkpoint 读取。

---

## 4. `manifest.assets` 全部稳定角色

下表列的是当前合同和当前 adapter 使用的稳定角色。**典型路径只是当前实现，读取时仍应以 manifest 中的值为准。**

### 4.1 通用几何、网格和场景角色

| 角色 | 当前典型路径 | 产生条件 | 用途 |
| --- | --- | --- | --- |
| `point_cloud` | `geometry/points.ply` | `mock`、`vggt`、`colmap`、`colmap_vggt` 的 point-cloud/mesh 路径 | 后端原始点云；也是 alignment/mesh 输入 |
| `sfm_sparse_point_cloud` | `geometry/points.ply` | 成功的 Project Gaussian | final accepted COLMAP/VGGT-BA sparse PLY；独立于通用 `point_cloud` 角色 |
| `point_cloud_aligned` | `geometry/points_aligned.ply` | 任一 raw 点云成功通过 dominant-plane RANSAC 对齐 | 默认稀疏几何查看和优先 meshing 输入；不覆盖 raw |
| `cameras` | `geometry/cameras.json` | 后端产生相机时 | 相机内参、外参、注册图像和 viewer 轨迹 |
| `alignment_diagnostics` | `diagnostics/alignment.json` | alignment 成功 | raw 到 aligned 的变换、平面和 inlier 证据 |
| `mesh` | `geometry/mesh.glb` | `output_type=mesh` | 用户可见 baseline 表面网格 |
| `mesh_diagnostics` | `diagnostics/mesh.json` | baseline mesh 成功 | 方法、参数、点/面数和清理统计 |
| `scene_graph` | `scene_graph/scene.json` | 所有成功 Job | 当前仍是 mock placeholder；不可解读为真实语义图 |
| `log` | `logs/run.log` | 所有成功 Job | adapter、有效参数、子进程摘要和后处理日志 |

注意：Project Gaussian 的 `geometry/points.ply` 会以 `sfm_sparse_point_cloud` 公开，通用后处理仍会从它生成 `point_cloud_aligned`。因此 Gaussian Job 可以有 `sfm_sparse_point_cloud + point_cloud_aligned`，但没有 `point_cloud`，这是正常的。

### 4.2 SfM、COLMAP+VGGT 和 VGGT-BA 诊断角色

| 角色 | 当前典型路径 | 产生条件 | 用途 |
| --- | --- | --- | --- |
| `sfm_diagnostics` | `diagnostics/sfm/manifest.json` | Project Gaussian；DB 导出成功，fail-soft | 全图关键点、实际 tested-pair 邻接、内点/外点及 detector/matcher/run provenance |
| `sfm_camera_calibration_diagnostics` | `diagnostics/sfm_camera_calibration.json` | 当前所有 COLMAP-backed reconstruction | raw sparse model 的相机模型、共享策略、焦距/畸变和软 warning |
| `sfm_pose_health` | `diagnostics/sfm_pose_health.json` | 每个新 Project Gaussian 几何结果 | 训练前 pose hard gate；成功资产必须记录 `passed` |
| `sfm_pose_recovery` | `diagnostics/sfm_pose_recovery.json` | Project Gaussian 且 geometry source 为普通 COLMAP | incremental/global/core-repair 选择和同 matches 恢复 provenance |
| `fusion_diagnostics` | `diagnostics/fusion.json` | `colmap_vggt` | 稀疏几何与 VGGT dense point 融合统计 |
| `visibility_graph` | `diagnostics/visibility_graph.json` | `colmap_vggt` | window/frame 可见性支持图 |
| `scale_disagreement_diagnostics` | `diagnostics/scale_disagreement.json` | `colmap_vggt` | 局部深度/几何尺度分歧 |
| `consistency_diagnostics` | `diagnostics/consistency.json` | `colmap_vggt` | multi-view consistency 支持和拒绝统计 |
| `vggt_ba_diagnostics` | `diagnostics/vggt_ba.json` | 请求实验 `gaussian_geometry_source=vggt_ba`，包括允许的 COLMAP fallback | window、pose graph、fallback 与有效 geometry source |
| `vggt_ba_window_graph` | `vggt_ba/window_graph.json` | 合同意图为所有完成的 VGGT-BA attempt | window graph 边、连通性和非局部证据；**见第 11.5 节 A 项** |
| `vggt_ba_initialization_diagnostics` | `diagnostics/vggt_ba_initialization.json` | 最终有效 geometry source 仍是 VGGT-BA | 仅 Train-supported sparse initialization 的接收/拒绝/recolor 统计 |

`sfm_diagnostics` 是 fail-soft：导出失败不会让 Gaussian training 失败。此时 manifest 不含该角色，只记录 `sfm_diagnostics_status=unavailable` 和原因。历史 Job 也可能完全没有这组字段。

### 4.3 视频角色

| 角色 | 当前典型路径 | 产生条件 | 用途 |
| --- | --- | --- | --- |
| `video_probe` | `diagnostics/video_probe.json` | 成功视频 Job | 源 hash、时长、源/显示尺寸和 quarter-turn；不公开位置元数据 |
| `video_frame_selection` | `frames/selection.json` | 成功视频 Job | 6 fps 候选评分、拒绝理由、最终选择、source PTS、输出 JPEG hash |
| `video_keyframe_contact_sheet` | `diagnostics/video_keyframes.jpg` | 成功视频 Job | 快速人工查看；仅展示，不是训练输入合同 |
| `video_keyframe_timing` | `diagnostics/video_keyframe_timing.json` | 成功视频 Job | probe、分析、materialize 和总耗时 |
| `video_registration_diagnostics` | `diagnostics/video_registration.json` | 成功视频 Job | 最终注册率、时间覆盖、最大 gap 和 gap violations |
| `video_initial_registration_expansion` | `diagnostics/video_initial_registration_expansion.json` | 普通 COLMAP + `standard_v2` | seed Mapper 后最多两次 registrator/triangulator 扩展证据 |
| `video_registration_recovery` | `diagnostics/video_registration_recovery.json` | `standard_v2` | 最多两轮 bounded gap recovery、局部图对、保留率和 final BA |
| `colmap_timing` | `diagnostics/colmap_timing.json` | 普通 COLMAP + `standard_v2` 作为稳定角色 | 各 COLMAP 子阶段时间、seed、Mapper 和有效数据库哈希 |

`standard_v1` 历史 Job 不会拥有 v2-only 三个角色。`colmap_timing.json` 在其他 COLMAP 路径也可能物理存在，但只有 manifest 显式列出时才是稳定资产。

### 4.4 Gaussian 训练、Validation 和导出角色

`{trainer_attempt_id}` 在普通集成 Job 中通常为 `train-001`。

| 角色 | 当前典型路径 | 内容和用途 |
| --- | --- | --- |
| `gaussian_raw_model` | `gaussian/attempts/{id}/artifacts/model.pt` | trainer 的 immutable Validation-selected 项目格式 snapshot，发生 SOR 之前 |
| `gaussian_model` | SOR 成功时 `gaussian/sor/{id}/filtered-model.pt`；否则常与 raw 相同 | 进入公共 Validation、export 和 navigation 的实际模型 |
| `gaussian_training_result` | `gaussian/attempts/{id}/artifacts/result.json` | final iteration、selected iteration、loss、资源、strategy/cap、checkpoint 引用 |
| `gaussian_progress` | `gaussian/attempts/{id}/artifacts/progress.jsonl` | 逐迭代 loss、拓扑变化、Validation 和可选 recovery-prune 事件 |
| `gaussian_dataset` | `gaussian/preparation/{id}/dataset.json` | 当前 attempt 的有效 dataset/normalization/splits/initialization 合同 |
| `gaussian_effective_config` | `gaussian/preparation/{id}/effective_config.json` | 完整 resolved config 及 hash；navigation 也校验此资产 |
| `gaussian_replay_dataset` | `gaussian/replay/dataset.json` | 可离开已删除 COLMAP workspace 使用的冻结 dataset contract |
| `gaussian_replay_record` | `gaussian/replay/replay.json` | replay 中相机、图像、初始化的 hash/count/bytes 绑定 |
| `gaussian_evaluation` | `gaussian/evaluation/{id}/validation/evaluation.json` | 公共 Validation 评估；普通前端 Job 不加载 Test |
| `gaussian_export_metadata` | `gaussian/export/{id}/export.json` | 坐标、任意单位、模型/数据/配置/评估 hash、SH 布局和 viewer bounds |
| `gaussian_canonical` | `gaussian/export/{id}/canonical.ply` | 项目自有、确定性 binary little-endian canonical PLY |
| `scene_splat` | `gaussian/export/{id}/scene.ply` | 浏览器稳定 Gaussian derivative；当前字节可与 canonical 相同，但角色不可互换 |
| `gaussian_camera_path` | `gaussian/export/{id}/camera_path.json` | Validation camera keyframes；用于初始 orbit pivot/up，不是渲染视频 |
| `gaussian_bundle` | `gaussian/export/{id}/result.zip` | 单个 Gaussian 结果的确定性交付包，见第 9 节 |
| `gaussian_test_evaluation` | 无普通 Job 固定路径 | 仅单独授权的 frozen-candidate Test 评估后出现；默认缺失 |
| `gaussian_test_decision` | 无普通 Job 固定路径 | 与上项配套的终局 Test 判定；默认缺失 |

#### `gaussian_raw_model` 与 `gaussian_model` 是否重复

- SOR 关闭或 fail-soft 失败：两个角色通常指向同一个 raw `model.pt`。
- SOR 成功：raw 仍保留；`gaussian_model` 改指 `sor/.../filtered-model.pt`。
- `scene_splat` 始终由 `gaussian_model` 导出，因此默认 SOR 成功时浏览器看到的是清理后模型。
- SOR 不会原地覆盖或删除 raw model；这是为了可审计和回退。

### 4.5 Gaussian SOR 与实验 visibility derivative

| 角色 | 当前典型路径 | 产生条件和用途 |
| --- | --- | --- |
| `gaussian_sor_filter_record` | `gaussian/sor/{id}/filter-record.json` | SOR 成功；参数、源/结果 hash、输入/保留/删除数 |
| `gaussian_sor_filter_mask` | `gaussian/sor/{id}/filter-mask.npz` | 与 Gaussian 行对齐的 SOR keep mask |
| `gaussian_vggt_filtered_model` | `gaussian/postprocess/{id}/filtered-model.pt` | 实验 `vggt_visibility_v1` 完整成功 |
| `gaussian_vggt_filter_diagnostics` | `gaussian/postprocess/{id}/diagnostics.json` | Train-only depth/support/free-space 判据与统计 |
| `gaussian_vggt_filter_mask` | `gaussian/postprocess/{id}/filter-mask.npz` | row-aligned keep/reason/support arrays |
| `gaussian_vggt_filtered_evaluation` | `gaussian/evaluation/{id}/validation-vggt-filtered/evaluation.json` | filtered derivative 的独立 Validation |
| `gaussian_vggt_filtered_export_metadata` | `gaussian/export/{id}-vggt-filtered/export.json` | filtered derivative 的独立 export/hash 记录 |
| `gaussian_vggt_filtered_canonical` | `gaussian/export/{id}-vggt-filtered/canonical.ply` | filtered canonical PLY |
| `scene_splat_vggt_filtered` | `gaussian/export/{id}-vggt-filtered/scene.ply` | Viewer A/B 的 filtered derivative |
| `gaussian_vggt_filtered_bundle` | `gaussian/export/{id}-vggt-filtered/result.zip` | filtered derivative 的独立 bundle |

八个 VGGT filtered 角色必须在 filter、Validation 和 export 全部成功后一起公开。任一步失败都不公开 partial filtered roles，Original `scene_splat` 仍然有效。

### 4.6 Navigation 角色

| 角色 | 固定当前路径 | 用途 |
| --- | --- | --- |
| `collision_mesh` | `navigation/collision.glb` | 只供本地物理 Octree 使用的低模隐形碰撞网格 |
| `navigation` | `navigation/navigation.json` | normalized/arbitrary-unit 边界、spawn、player 和 source hash 合同 |
| `navigation_diagnostics` | `navigation/diagnostics.json` | Train-only、拓扑、三角形/字节/耗时和完整性证据 |

三者必须一起通过路径 containment、source/model/config/export hash、split isolation、schema、GLB、三角形数、体积和耗时校验，才会一次目录 rename 发布。Navigation 失败是 fail-soft：Gaussian Job 仍为 `done`，Orbit 和 `scene_splat` 仍可用。

### 4.7 Mesh variants 不是普通 asset role

`POST /api/jobs/{job_id}/mesh-variants` 会写：

```text
geometry/mesh_{method}_{random-id}.glb
diagnostics/mesh_{method}_{random-id}.json
```

它们登记在 manifest 顶层 `mesh_variants[]`，每项含：

- `id`、`label`、`method`；
- `mesh_asset`、`diagnostics_asset`、`source_asset`；
- `options`、`metrics`、`created_at`。

已有 baseline `mesh`/`mesh_diagnostics` 可在读取 manifest 时被重新派生为 `mesh_variants[id=baseline]`。因此 mesh variant 路径不应被误认为新的 `assets` 键。

---

## 5. 不同 backend / output_type 实际会有什么

| 请求组合 | 主要稳定资产 | 额外行为 |
| --- | --- | --- |
| `mock + point_cloud` | `point_cloud` | 再尝试 aligned；用于轻量 smoke |
| `vggt + point_cloud` | `point_cloud`, `cameras` | 再尝试 aligned |
| `vggt + mesh` | 上述 + `mesh`, `mesh_diagnostics` | 优先从 aligned 点云建 mesh |
| `colmap + point_cloud` | `point_cloud`, `cameras`, `sfm_camera_calibration_diagnostics` | 根级 `colmap/` 工作区成功后删除 |
| `colmap + mesh` | 上述 + `mesh`, `mesh_diagnostics` | 同样不保留成功 COLMAP DB |
| `colmap_vggt + point_cloud` | point cloud/cameras/calibration + fusion/visibility/scale/consistency | `colmap_vggt/` 工作区删除，diagnostics 发布 |
| `colmap_vggt + mesh` | 上述 + mesh 两角色 | 使用 dense fused point cloud 建 mesh |
| `project_3dgs + gaussian_splat` | SfM sparse/cameras/diagnostics + Gaussian 全套 + scene graph/log | multi-image 或 bounded video；随后 fail-soft navigation |

所有含 raw point cloud 的成功路径都会尝试 alignment。Alignment 失败不会让点云 Job 失败，只是不添加 `point_cloud_aligned` 和 `alignment_diagnostics`。

当前未实现成功资产的组合：

- `dust3r`、`mast3r`；
- panorama geometry；
- `project_3dgs` 的非 `gaussian_splat` 输出；
- video 与非 Project Gaussian backend 的组合。

---

## 6. Project、MCMC 与 Graphdeco 的目录差异

三者都通过 `project_3dgs + gaussian_splat`，并共享接受后的 SfM、dataset contract、项目格式 `model.pt`、SOR、公共 Validation、export、manifest 和 Viewer。

| 内容 | Project | MCMC | Graphdeco |
| --- | --- | --- | --- |
| 训练实现 | 项目 native + gsplat DefaultStrategy | 项目 native + gsplat MCMCStrategy | 隔离 external Graphdeco checkout |
| `gaussian/attempts/{id}/attempt.json` | 有 | 有 | 当前 external wrapper 不创建同一 native attempt descriptor |
| 项目 checkpoint | 有，成功保留一个 terminal checkpoint | 有，成功保留一个 terminal checkpoint | 无项目 optimizer checkpoint；`result.json` 指向 native PLY hash |
| cadence Validation 预览 | `artifacts/validation/iteration_*/` | 同 Project | 原生 trainer 自己的输出位于 `native/`，随后只走公共 Validation |
| 额外目录 | 无 trainer-native 镜像 | 无 trainer-native 镜像 | `preparation/.../graphdeco-dataset/` 和 `native/.../graphdeco/` |
| Gaussian cap | 默认无 hard cap | 全 rank 合计 3,000,000 | 由固定外部方法和配置记录，不伪装成 native strategy |
| frozen replay 重跑 | 支持 | 支持 | `--initialization frozen` 不支持 Graphdeco |

当前 adapter 对完成的三种 trainer 都要求并公开 replay dataset/record；native Project/MCMC 是该 replay 的直接冻结重跑消费者。Graphdeco 的原生环境和 checkout 在仓库 `external/`，不复制进 Job；它在 Job 内只保留这次运行的数据镜像和原生产物。

Graphdeco 仍是 research/evaluation-only，受其自定义许可约束。目录存在不改变许可边界。

---

## 7. SfM frontend diagnostics 具体包含什么

`sfm_diagnostics` 当前是 schema 4 主 manifest；feature shard 仍是 schema 1，pair shard 是 schema 2。

### 7.1 主 manifest

记录：

- normalized OpenCV camera axes 与 arbitrary-unit 语义；
- dataset hash、default run、一个或多个 detector/matcher run；
- 总图像、关键点、图对、tentative match、candidate inlier、Guided-added inlier、final inlier 和 outlier 计数；
- 每张已提取 feature 的图片，包括未注册图；
- 图片 job-relative path/hash/尺寸、COLMAP image/camera ID、可选视频时间；
- 最终 registered/split 状态；注册图才有 normalized center/forward/up 和 FOV；
- feature/local matcher/pairing/geometric verification/camera calibration/Mapper/COLMAP build provenance；
- verified view graph 的连通分量、degree、isolated node、视频 edge span 和 gap bridge 证据。

### 7.2 Feature shards

- 每个 shard 最多 32 张图；
- 只保留原始 upright feature-input pixel 中的 `(x, y)`；
- 坐标保留到 0.01 pixel；
- 不保留 descriptor 向量。

### 7.3 Pair shards

- pair index 是实际 COLMAP match/geometry tables 中保留过的图对，不是前端猜测邻居；
- 每个 detail shard 最多 256 对；
- 保留 candidate、candidate inlier、Guided-added inlier、final inlier 和 rejected tentative outlier 计数；
- detail 中 `inliers` 是所有最终 verified correspondence；`outliers` 只含被拒绝的 tentative candidates；
- index 中缺失某对表示它不在保留的 match/geometry tables；存在但计数为零表示测试/记录过但没有保留 correspondence。

### 7.4 明确不导出的内容

- SIFT/ALIKED descriptor 向量；
- 原始或 recovery 后的 mutable COLMAP database；
- 最终 SfM 3D track 的逐 observation 浏览数据；
- 像素 mosaic/stitching；
- failed/replaced attempt 与 accepted attempt 的混合数据。

---

## 8. 视频资产：候选帧、选中帧和原视频

### 8.1 永久保留

- `input/{video}`：原始视频；retry 从这里确定性重新抽帧。
- `frames/selected/*.jpg`：最终 geometry 实际使用的选中帧。
- `frames/selection.json`：候选评分/拒绝原因和最终 selection provenance。
- 四个基础视频 diagnostics，以及 v2 的 expansion/recovery/timing 证据。

### 8.2 不保留为图片

6 fps 分析阶段最多 3,636 个候选。候选 RGB 帧用于评分，但**不会把所有候选帧逐张保存**。只有最终选中的帧才 materialize 到 `frames/selected/`。未选候选仍以结构化 metadata 形式记录在 `selection.json`，而不是以 JPEG 形式保留。

### 8.3 v2 recovery 新增帧

被接受的 recovery frames 会原子加入：

- `frames/selected/`；
- 同一个 `frames/selection.json`；
- 最终 SfM/registration/dataset 证据。

因此 `video_selected_count` 描述 final geometry input，不只是初始 selector 输出。

---

## 9. 两种 ZIP 完全不同

### 9.1 Gaussian `result.zip`

路径：

```text
gaussian/export/{trainer_attempt_id}/result.zip
```

稳定角色：`gaussian_bundle`。

确定性包含：

```text
gaussian/canonical.ply
gaussian/scene.ply
gaussian/export.json
gaussian/camera_path.json
contracts/dataset.json
contracts/effective_config.json
evaluation/evaluation.json
postprocess/diagnostics.json      # 仅 SOR/filtered export 提供时
postprocess/filter-mask.npz       # 同上
```

不包含：

- optimizer、scheduler、densification 或 RNG state；
- full checkpoint；
- COLMAP database/descriptors/matches；
- 绝对本机路径；
- 原始上传视频/图片；
- navigation。

`bundle.json` 位于 ZIP 外，记录 ZIP 的 SHA-256 和字节数，以避免自引用 hash；它物理保留，但没有独立 asset role。

### 9.2 整个 Job 下载 ZIP

路径：

```text
outputs/jobs/{job_id}.zip
```

它由下载接口按需重建，位于 Job 目录外，不写入 `manifest.assets`。当前实现遍历整个 Job 目录并加入所有文件，唯一特殊排除是：

- `navigation/`；
- `lifecycle/navigation/`；
- ZIP 内 `manifest.json` 的 navigation 三个 roles、navigation 状态字段和 `navigation_*` metrics。

因此 Job ZIP 通常包括原始输入、稳定输出、request、logs，以及存在时的 `lifecycle/attempts/.../partial`。这一点与精简的 Gaussian `result.zip` 完全不同。

---

## 10. 哪些数据成功后会删除、覆盖或不发布

### 10.1 Worker 成功发布时整体删除

运行发生在：

```text
lifecycle/attempts/{attempt_id}/workspace/
```

成功时只把白名单目录移到 Job 根，再删除剩余 workspace。以下主要数据因此不会出现在普通成功 Job 的稳定根目录：

| 被删除/不发布内容 | 典型路径（均在 workspace 内） | 为什么可以删除 / 替代证据 |
| --- | --- | --- |
| workspace 输入副本 | `input/...` | 根级 `input/` 原件仍在 |
| COLMAP SQLite DB | `colmap/database.db` 或 recovery DB | 前端需要的关键点/图对已导出为 gzip shards；DB 可变且包含 descriptors |
| COLMAP feature descriptors/matches | DB 内表 | 明确排除，不作为产品资产 |
| raw sparse binary models | `colmap/sparse/` | final PLY、cameras、pose/calibration diagnostics 已发布 |
| raw/final sparse text conversions | `colmap/sparse_raw_txt/`、`colmap/sparse_txt/` | 属于转换和 gate 中间输入 |
| undistorted COLMAP dataset | `colmap/undistorted/images/`、`sparse/`、`sparse_txt/` | 训练所需冻结副本已进入 `gaussian/replay/`；普通 geometry Job 不需要训练 replay |
| camera group staging | `colmap/camera_groups/` | 分组证据已进入 calibration diagnostics |
| v2 seed/expansion/recovery models | `colmap/v2-mapper-seed.txt`、`v2_initial_expansion/` 及 recovery dirs | 最终模型与 JSON diagnostics 取代过程目录 |
| COLMAP 进度 | `colmap/progress.json` | 仅运行期 UI polling |
| COLMAP+VGGT 工作区 | `colmap_vggt/` | final point cloud/cameras 和 diagnostics 已发布 |
| VGGT-BA 工作区 | `vggt_ba/` | 设计上应由 diagnostics 替代；但 window graph 当前有发布缺口，见第 11.5 节 A 项 |
| VGGT-BA 进度和 Train seed text | `vggt_ba/progress.json`、`train_points3D.txt` | 运行期/初始化中间输入 |
| workspace 根级 dataset/config | `dataset.json`、`gaussian_config.json` | 有效版本已复制到 `gaussian/preparation/{id}/` |
| workspace 中其他根级日志 | 例如 `logs/vggt_ba.log` | publisher 只移动最终 `logs/run.log` |

### 10.2 Trainer 内部会覆盖或删除

| 内容 | 当前行为 |
| --- | --- |
| 最佳 Validation candidate snapshot | 使用隐藏 `.best-model*.pt` 覆盖写；成功时复制/合并为 `artifacts/model.pt` 后删除隐藏 candidate |
| 较早 committed full checkpoints | final/最新 checkpoint 成功后，同一 trainer attempt 仅保留指定 iteration，其他 committed checkpoint 目录删除 |
| checkpoint 临时目录 | 正常异常处理会清理；硬崩溃可能在 failed workspace 留下隐藏 temp，但它不被 loader 当作 committed checkpoint |
| replay 临时目录 | 在 `.replay-*` 构建后原子 rename；捕获异常时清理 temp |
| SfM diagnostics 临时目录 | 成功原子 rename 为 `diagnostics/sfm/`；导出失败会清理 `.sfm.tmp`，并省略角色 |
| 旧 `result.zip` | Gaussian export 目录不可覆盖；Job 总 ZIP 则每次请求会先删除同名旧 ZIP 再重建 |

### 10.3 不会因成功导出而删除

以下常被误认为“导出后没用了”，但当前实现会保留：

- `gaussian_raw_model`；
- SOR 后的 `gaussian_model`；
- native Project/MCMC terminal full checkpoint；
- Graphdeco 原生输出目录；
- Gaussian preparation、initialization 和 replay；
- cadence Validation 预览；
- standalone Validation 的 JSONL、record 和 previews；
- canonical PLY 与 browser PLY；
- export dotfiles 和 `bundle.json`；
- SOR / VGGT visibility masks 与 records；
- 原始上传；
- 视频选中关键帧；
- lifecycle 下历史 failed partial。

因此一个成功 Gaussian Job 仍可能很大；“已生成 `scene.ply`”不代表 trainer、checkpoint、replay 或原始输入已自动清理。

### 10.4 fail-soft 子阶段失败时的残留

SOR、SfM frontend export、VGGT visibility 和 navigation 的失败不一定让主 Job 失败：

- 没通过完整性检查的角色不会进入 `assets`；
- 已写出的 partial 文件可能仍物理存在于发布目录或 lifecycle partial；
- 这些文件没有稳定消费保证；
- 不应通过扫描目录把它们重新当成成功结果。

---

## 11. 失败、取消、中断与 retry 的保留规则

### 11.1 queued 时取消

尚未创建 workspace。保留：

- 原始 `input/`；
- `request.json`；
- cancelled `manifest.json`；
- 对应 `logs/attempt-NNN.log`。

不会有成功 output assets。

### 11.2 running 时正常失败或取消

Worker 捕获错误后：

```text
lifecycle/attempts/{attempt_id}/workspace/
    -> lifecycle/attempts/{attempt_id}/partial/
```

然后 terminal manifest：

- `status=failed` 或 `cancelled`；
- `assets={}`；
- `error={code,message}`；
- 原始根级输入不删除。

`partial/` 可包含输入副本、COLMAP database、descriptor/match tables、undistorted images、半成品 Gaussian/checkpoint/export 等。它是本地诊断现场，不是成功接口。

### 11.3 Worker/进程硬中断

API/worker 重启发现旧 `running`/`exporting` Job 时，会显式标为 `worker_interrupted`，不会猜测为成功：

- 已经移动到稳定根但还没有完成 `done` manifest 的输出，会隔离到 `partial_published/`；
- 尚未发布的 workspace 当前可能仍以 `workspace/` 名称留在 attempt 下；
- manifest assets 清空。

Navigation 中断采用同类规则，workspace 改为 `partial/`，已经 rename 的 `navigation/` 改为 `partial_published/`。

### 11.4 retry

- 仅 `failed`/`cancelled` 且具有 lifecycle schema 的 Job 可 retry；
- 最多三个 outer Job attempts；
- retry 创建新的 immutable `attempt-NNN`；
- 从根级原始输入和 `request.json` 重新开始；
- **不加载旧 partial checkpoint，不合并旧 diagnostics，不复用旧成功角色**；
- 旧 partial 保留，因此磁盘体积会累积；
- 新 attempt 成功后，稳定输出来自这一个 accepted attempt。

### 11.5 当前实现需要特别注意的两个发布边界

#### A. `vggt_ba_window_graph` manifest 悬空风险

Adapter 当前登记：

```text
vggt_ba_window_graph -> vggt_ba/window_graph.json
```

但主 workspace 成功发布白名单**不包含 `vggt_ba/`**，随后剩余 workspace 会被删除。结果是：成功 VGGT-BA Job 的 manifest 可能引用已经删除的 `vggt_ba/window_graph.json`。

在修复发布路径或把图复制到 `diagnostics/` 前：

- `vggt_ba_diagnostics=diagnostics/vggt_ba.json` 可可靠保留；
- 不应假设 `vggt_ba_window_graph` URL 一定可下载；
- 这属于当前实现缺口，不是正常清理语义。

#### B. Job 总 ZIP 会包含 main lifecycle partial

合同层面 failed partial 不进入 `manifest.assets`，但当前 `build_zip()` 只排除 navigation 目录，不排除 `lifecycle/attempts/`。因此有失败/retry 历史的 Job 下载 ZIP 可能包含：

- duplicated input；
- COLMAP database；
- descriptors/matches；
- 半成品 checkpoint/export；
- 其他失败现场。

这与“稳定资产不发布 DB/descriptors”是不同层面的当前行为。若 Job ZIP 被当作外部交付物，应先修订 ZIP 排除策略；在此之前，精简且边界清晰的外部 Gaussian 交付应优先使用 `gaussian_bundle`。

---

## 12. Navigation 与普通 Job ZIP 的特殊关系

Navigation 是 Gaussian Job 完成后的独立 fail-soft lifecycle：

```text
lifecycle/navigation/attempt-NNN/workspace/
    --完整校验并 rename--> navigation/
```

特殊规则：

- 生成失败不会把主 Job 从 `done` 改为 `failed`；
- 三个稳定角色必须一起存在；
- 旧成功 Gaussian Job 可通过专用 API 幂等补生成；
- 一旦 `available`，再次请求不会覆盖；
- 校验失败的旧 published navigation 会移入 `lifecycle/navigation/invalid_published[-N]/`；
- navigation 和它的 lifecycle 全部排除在 Job 总 ZIP 外；
- 前端 Walk 只有在三件套存在且客户端复验通过时启用，Orbit 始终独立可用。

`collision.glb` 不是 `geometry/mesh.glb`：前者是隐藏物理代理，后者是用户可见表面网格。

---

## 13. 坐标系、对齐和尺度

### 13.1 Raw 与 aligned

- raw 点云保留 backend 原坐标系；Project sparse 通常标记为 `colmap_world`。
- `points_aligned.ply` 是独立 derivative，不覆盖 raw。
- `alignment.json` 记录变换和 dominant-plane 证据。
- 平面法向存在 `+Z/-Z` 符号歧义；前端结合变换后的 camera-up 选择初始显示方向。
- Viewer 的 X/Y/Z flip 是显示变换，不会改写 PLY、camera 或 manifest。

### 13.2 Gaussian normalized frame

Gaussian dataset 把 accepted world geometry 变换到 normalized frame。Export 记录：

- `coordinate_frame=normalized`；
- `world_units=arbitrary`；
- `world_from_normalized`；
- OpenCV camera axes；
- scene center/radius 和 camera path。

当前视角查询最近输入图时，Viewer 必须把自动摆正后的查询相机逆变换回 diagnostics 使用的 normalized frame；不能拿显示坐标直接与记录相机比较。

### 13.3 禁止的尺度声明

以下量均是 scene-relative，而不是米：

- 相机间距；
- Gaussian scene bounds；
- navigation eye height `H`、capsule、速度和边界；
- collision mesh 尺寸；
- alignment 后的坐标。

只有未来独立、可审计的 scale recovery 评估才能改变该口径。

---

## 14. 历史 Job 兼容性

旧 Job 可能缺少：

- lifecycle schema、attempt history 和 retry 能力；
- `sfm_diagnostics` 及其所有计数；
- `sfm_pose_health` / `sfm_pose_recovery`；
- camera calibration provenance；
- `gaussian_replay_*`；
- SOR 和 VGGT visibility 状态/资产；
- v2 video diagnostics；
- navigation 字段；
- `mesh_variants`；
- 当前 trainer identity、strategy 或 cap。

读取原则：

1. 缺字段表示“历史 Job 未记录”，不是自动补成当前默认值。
2. 旧 Job 有 raw/aligned PLY 时仍可查看几何。
3. 没有 `sfm_diagnostics` 时，前端应明确显示“诊断不可用”，不能报错，也不能从最近三图猜匹配。
4. `get_manifest()` 可以根据磁盘上已经存在的 alignment、navigation 和 mesh variant 文件重新浮现对应当前资产，但不会重跑 reconstruction。
5. 不要用一个历史 Job 的实际目录推断所有新 Job 都不会有 replay 或新 diagnostics。

---

## 15. 磁盘清理时应如何判断可删性

本文描述行为，不自动授权删除。人工清理前建议按以下优先级判断：

1. **必须保留**：`manifest.json`、`request.json`、根级 `input/`、manifest 当前引用的所有文件。
2. **为可复现训练保留**：`gaussian/replay/`、有效 config/dataset、raw/model、evaluation/export、需要 resume 时的 terminal checkpoint。
3. **仅本地排错需要**：`lifecycle/attempts/*/partial`、`partial_published`、Graphdeco native logs、cadence previews、未登记 diagnostics。
4. **可重新生成但代价高**：selected video frames、COLMAP/SfM/Gaussian outputs；删除前确认是否愿意重新跑 geometry/GPU training。
5. **外部交付优先选择**：只需要 Gaussian Viewer 结果时用 `gaussian_bundle`，不要默认发送包含原始输入和 partial 的 Job 总 ZIP。

任何清理工具都应先解析 manifest 引用和 hardlink 情况，再按明确策略处理；不能仅按文件名或目录名批量删除。

---

## 16. 代码与合同依据

本文以以下当前文件为准：

- `docs/manifest-schema.md`：稳定 manifest、assets、SfM/video/VGGT-BA/navigation 合同；
- `docs/gaussian-trainer-contract.md`：初始化、replay、trainer 和模型资产；
- `docs/gaussian-checkpoint-contract.md`：checkpoint 原子性与 retention；
- `docs/stage2d-contract.md`：Validation、canonical export 和 Gaussian bundle；
- `src/image3d_scenegraph/jobs.py`：Job lifecycle、workspace publication、partial、ZIP、mesh variants、navigation；
- `src/image3d_scenegraph/geometry/adapters.py`：backend 资产角色与 Project Gaussian 主链路；
- `src/image3d_scenegraph/geometry/colmap_diagnostics.py`：SfM gzip shard 导出；
- `src/image3d_scenegraph/video/keyframes.py`：视频候选与 materialized selection；
- `src/image3d_scenegraph/gaussian/replay.py`：冻结 replay 组成；
- `src/image3d_scenegraph/gaussian/trainer.py`：native model、Validation candidate 和 checkpoint 清理；
- `src/image3d_scenegraph/gaussian/external_trainer.py`：Graphdeco 数据镜像、native output 和导入；
- `src/image3d_scenegraph/gaussian/export.py`：canonical/browser PLY 与 deterministic `result.zip`。

若代码与本文以后发生偏差，应先以 `manifest.json` 和冻结合同判断消费者行为，再同步更新本文。
