# 简历亮点梳理：3D看房师（免标定视频/图像 → 可漫游 3D 重建）

> 日期：2026-09-16
> 状态：简历/面试素材梳理，不涉及代码改动
> 范围：面向"计算机视觉算法工程师（深度学习方向）"实习岗位，从本仓库现有已实现能力中提炼可写亮点、量化事实与风险红线
> 约束：坐标为归一化任意单位，不宣称公制精度；`scene_graph` / `semantics` / `physics` 仍为 placeholder，不得写成已实现；质量结论主要是单场景单 seed，不得外推为通用

---

## 1. 定位判断

项目原名 **Image3D-SceneGraph**，但 `src/image3d_scenegraph/{semantics,physics,scene_graph}/` 基本是空 `__init__.py`，`scene_graph/scene.json` 是 mock。若简历写"语义场景图"，一追问即穿。

改用 **"3D看房师"** 是更诚实的包装：它把叙事从**未实现的语义理解**转移到**已实现的可漫游看房**。支撑证据是 `scripts/build_gaussian_navigation.py`（1,547 行）：渲染 Train 视角深度 → 多视深度一致性滤波 → Open3D `ScalableTSDFVolume` 融合碰撞网格 → 地板/支撑/障碍掩码相交 + 形态学开运算 + 连通域分析 → 可行走区域与出生点合法性校验；前端具备 pointer-lock 行走。导航作为独立 fail-soft 生命周期阶段，失败不影响 Gaussian 主任务 `done`。

**对 DL 方向岗位需诚实的一点**：本项目的深度学习成分是"**使用**预训练前馈模型 + **做**可微渲染优化"，不是从零训网络。VGGT / DINOv2 / ALIKED / LightGlue / gsplat 均为外部权重。DL 相关性来自：可微光栅化与 3DGS 优化、前馈几何先验（VGGT 点图）与传统 SfM 的融合、MCMC 采样式致密化策略、双卡分布式训练、混合精度与字节上限惰性解码。"你自己训过网络吗"是必问题，建议用前序 Mono3D-Grounding（TensorRT 单目度量深度实时化）补这块。

---

## 2. 简历文案（精简版，可直接粘贴）

> **3D看房师 | 免标定视频/图像 → 可漫游 3D 场景重建系统**

1. **独立设计并实现免标定 3D 重建全链路**：视频解封装 → 质量感知选帧 → COLMAP SfM → 3D Gaussian Splatting 训练/评估/导出 → 浏览器第一人称漫游。自研 3DGS trainer（基于 gsplat 可微光栅化），**不依赖 Nerfstudio 等外部训练平台**；约 45k 行 Python + 20k 行测试（76 个测试文件），前端 9.5k 行 TS/TSX。

2. **提出 VGGT 前馈几何 + 传统 SfM 的混合初始化**：8 帧滑窗（4 帧重叠）VGGT 点图 → 按"≥32 个可靠观测"做强/弱相机分类并将弱相机排除出局部 BA 图证据 → 每个相邻断连做一次确定性恢复窗口 → 局部 BA + 鲁棒全局 Sim(3) 对齐 → 取满足 12 相机 / 70% 支撑 / 80% 时序覆盖的最大可靠分量，作为 COLMAP 三角化、遗漏图像注册与全局 BA 的种子。**仅在三类已分类的晚期几何质量失败时才回退普通 Mapper**，requested/effective 来源与回退原因全程可审计。

3. **用 View Graph 连通性诊断定位 4K 长视频重建失败的真正瓶颈**：对 601 s / 3840×2160 / HEVC / 11.3 GB 视频（6 fps 候选 3,608 帧 → 入选 2,787 帧，含 2,405 基线 + 382 运动自适应），在 ≥15 内点的标定边上做分量分析，发现**全帧图最大连通分量 2,655/2,787，而 1,000 帧种子诱导子图仅 457 且含 191 个孤立点**——判定瓶颈在 Mapper 种子选择而非素材质量。种子预算提到 2,500 后，注册相机由此前 Incremental 最高 **60/1,000** 提升到 **1,251/2,500（50.04%）**，稀疏点 146,321。

4. **设计位姿健康门禁与有界恢复**：在进入去畸变、数据切分、初始化、CUDA 之前，用**时序灾难步长、空点云、加权正深度占比、共视分量（并查集）**四道相互独立的守卫拦截坏几何；失败后依次尝试"数据库副本上的全局 Mapper 恢复"与"剔除 ≤10% 灾难位姿分支 + 点滤波 + BA 重跑"的视频专用修复，并**显式标记为 `recovered` 而非当作干净通过**；恢复过程永不静默切换特征/匹配器/配对策略/相机模型/训练器。

5. **用等预算对照实验量化交付期 fine-tune 的真实收益**：固定拓扑 2,000 步 / 4,000 相机采样，对 Selection / Train-only / Train+Validation 三个候选在 **171 个隔离 Test 视图**上各评一次，2,999,577 个高斯保持不变；Test PSNR **24.08 → 24.40 → 24.54 dB**（配对差 +0.323 / +0.458 dB）。同时明确区分"拟合集上 MCMC +1.79 dB"与"Test 上仅 +0.46 dB"，**拒绝把 in-sample 收益当泛化收益**；第三组配对差 PSNR 中位数仅 +0.0099 dB、p10 −0.154 dB、最小 −0.478 dB，如实记录为"收益小且分布不均"。

6. **从 3DGS 场景生成第一人称可漫游导航**：渲染 Train 视角深度 → 多视一致性滤波 → ScalableTSDF 融合碰撞网格 → 地板/支撑/障碍掩码相交 + 形态学开运算 + 连通域 → 可行走掩码与出生点校验；前端 pointer-lock 行走，取消/重试/网格变体等变更操作对比较结果服务端拒绝。

7. **建立门槛驱动（gate-driven）的评估方法论**：预声明 gate 与阈值、单因子消融、**Test 视图全程隔离（从不加载 Test RGB）**、effective config 哈希绑定、`gaussian/replay/` 冻结几何与 3NN 初始化以支持跨 trainer 无重跑复现；Stage 1 因子实验因跨场景证据不一致而**主动不 promote 任何候选**。

---

## 3. 算法侧备选弹药（按岗位挑用）

| 亮点 | 可写的量化事实 |
|---|---|
| **Morton 分层点预算采样**（`spatial_balanced`） | 相同点数、无 GT 参与；ETH3D `terrains` 1/2/5 cm F1 **+0.006265/+0.005373/+0.002104**，`delivery_area` **+0.002567/+0.005658/+0.005174**；`pipes` 未触及 2M 上限故不变；与另两个实验因子组合时仍为正 |
| **VGGT 可见性浮点清理**（`vggt_visibility_v1`） | ≤64 Train 视图 + 尺度对齐 VGGT 深度，保守剔除多视自由空间 floater 与出采集包络的超大高斯；Original 不可变 + 派生 A/B，fail-soft；不填洞、不造墙、不重训、不用 Validation/Test 推导掩码 |
| **视频选帧 v2 与注册扩展** | 6 fps 候选（≤5,436）→ 4 fps 均匀基线 → 光流运动自适应补帧至 5 fps → 质量拒绝 → 最多两轮局部 `image_registrator` + **非清空** `point_triangulator` 扩展（每轮 ≤25% 初始帧、累计 ≤50%）；首轮可喂给"零新候选传播轮"；12 帧 / 70% 注册率 / 80% 时序覆盖三道产品门禁 + >2 s 间隙软告警 |
| **学习特征归因实验** | 2×2 因子（SIFT/ALIKED × Brute-force/LightGlue）冻结同一 `sfm_frontend_contract_v1` 证据后只比未恢复的主结果；2026-08-13 ETH3D 2048 关键点/1280px 上 SIFT-LightGlue 近乎断连、ALIKED-LightGlue 可完成但匹配耗时显著更高 |
| **渲染器 A/B 验收**（Spark 2.2.0 vs GaussianSplats3D 0.4.7） | 同一 2,999,577 高斯 PLY、冻结相机、真实 NVIDIA L2 / Vulkan（Chrome 153，非 SwiftShader）；浏览器-原生 **MAE 0.03549→0.02005（−43.5%）**、参考 **PSNR 22.370→23.775 dB（+1.406）**；每臂 720 次 GPU query 全部有效。如实报告局限：**SSIM 仅 6/12 视图改善**（最差 −0.012655）、**GPU query P95 反而更高（7.261 vs 6.137 ms）**、冷加载更慢（19.983 vs 10.717 s）、rAF P95 两者都约 16.7–16.8 ms 不构成帧率优势 |
| **MCMC vs Project v7 同 replay 对比** | 双 L2、2,000 固定拓扑更新 / 4,000 分布式相机采样（1,376 Train + 179 Validation）；拟合后 MCMC 领先 **+0.4324 dB / +0.01598 SSIM**，179 视图中赢 148 PSNR；final-fit 253.0 s vs 379.8 s，每卡峰值预留 2.124 GB vs 3.725 GB；全局高斯上限冻结 3,000,000，与 `recovery_prune=on` 互斥 |
| **6 视图定向对比**（Spark 早期证据） | 同一 PLY、六个冻结 Validation 相机：浏览器-原生 MAE 0.03950→0.02621（−33.6%）、SSIM 0.96653→0.98416、PSNR 25.975→30.879；对参考 PNG 的 PSNR 20.252→21.037（+0.785 dB），六视图 PSNR 全升、SSIM 五升一降；317 视图 MAE 仅改善 3.86%，说明残差来自源模型而非渲染器 |

---

## 4. 工程侧亮点（偏工程/MLOps 岗位更值钱）

- **Manifest 即契约**：前后端只通过 `outputs/jobs/{job_id}/manifest.json` 通信，前端不知模型内部（开发规则 5）；资产带角色/相对路径/哈希/坐标系/单位状态，**只发布完整资产**，部分 checkpoint/导出永不冒充成功。
- **作业生命周期**：文件系统租约的串行 GPU worker、不可变 attempt、重启恢复（`running` → `failed` 且**不误判为成功**）、幂等取消、有界重试（第三次触发 retry limit）、运行中取消保留部分工作区、8 MiB 分块流式上传（16 GiB 上限，溢出即删部分文件），大视频不在应用内存里整体持有。
- **子进程与依赖治理**：adapters 一律 shell out 到 `scripts/`（每个脚本可独立 CLI 运行）；自动剥离 `LD_LIBRARY_PATH` 防止链到系统 cuDNN（可用 `IMAGE3D_VGGT_KEEP_LD_LIBRARY_PATH=1` 覆盖）；COLMAP 二进制三级解析顺序且不覆盖 `/usr/bin/colmap`；gsplat 锁定 tag/commit/license（`1.5.3+pt23cu121`，`937e2991`，Apache-2.0）；Graphdeco 需显式 `--accept-research-license` 且跑在独立 pin 环境；**Job 运行时从不下载权重**；`GET /api/backends` 分项报告能力，缺 VGGT-BA 依赖不会连带禁用普通 COLMAP 路径。
- **8 GB 单卡约束下的取舍**：VGGT 分组重叠 + backbone 自动半精度 / 相机与深度头保持 fp32；`TrainingViews` 惰性解码 + 每 rank 512 MiB 字节上限缓存，才撑住 4K UHD 训练（单帧实测 JPEG95/4:4:4 为 1.54–2.06 MB、PNG 为 5.72–7.46 MB）。
- **测试与验证规模**：后端 665 个测试（CPU/mock/编解码级），前端 63 个 CPU/mock 测试 + 13 个渲染审计测试；改动文件 Ruff、生产构建与 diff 检查常态化。
- **可复现性基础设施**：`gaussian/replay/` 硬链接注册图像/相机/dataset 契约/冻结初始化（排除 COLMAP 数据库与匹配），可跨 Project/MCMC 重放而不重跑几何或 3NN 初始化；`--prepare-only` 与适配器 stop 边界可只冻结几何；所有实验有独立 lock/once/exit/dependency 守卫。

---

## 5. 风险红线：不要写

| 别写 | 原因 |
|---|---|
| ❌ 语义 3D 场景图 / 物体级语义 / 物理一致性检查 | `scene_graph` 是 placeholder，`physics/`、`semantics/` 为空目录，scene.json 是 mock |
| ❌ 厘米级精度 / 公制尺度恢复 | 坐标是归一化任意单位，尺度恢复未评估；ETH3D 结果只能称"GT 相机 Sim(3) 对齐后的几何质量"，评估器从不用 ICP 贴合激光扫描 |
| ❌ 自研 3DGS 算法 / 新损失函数 | 训练数学是标准 `L1 + λ·D-SSIM`、指数 LR 调度、SH 阶次调度、PSNR；贡献在**编排、契约、评估与密度控制策略**，不在发明损失 |
| ❌ 把 final-fit 的 +1.79 dB 当泛化收益 | 那是**拟合集**数字；Test 上仅 +0.458 dB，第三组配对差中位数 +0.0099 dB、p10 −0.154 dB |
| ❌ "提升重建精度 X%"的泛指 | Stage 1 的 per_frame / adaptive_two / spatial_balanced 跨场景证据**混合**，G1 无候选被 promote；干净的正收益只有 Morton 采样的 F1 与 final-fit 的 Test PSNR |
| ❌ 跨场景 / 通用结论 | 主要质量结果为**单场景单 seed**（num4_room，171 Test 视图，2,999,577 高斯）；LPIPS 因无获批本地预训练主干而**从未运行**，属已审计的 not-run 项 |
| ❌ 实时 SLAM / 长时序无漂移建图 | 视频是**有界离线重建**；未验证的非局部几何只标 `open_trajectory_unverified`，不支持回环或无漂移声明 |
| ❌ Spark 已通过生产验收 | 只通过**同场景有界研究验收**，非跨场景/通用生产验收；无头受控全排序调度、单一本地路径、缩略图而非运动视频真值、SSIM 回退与未测真实资源峰值仍是明确局限 |

---

## 6. 面试前最值得补的三件事（按性价比排序）

1. **给 `scene_graph` 做一个最小可用版本**：哪怕只基于已有 `points_aligned.ply` 的 RANSAC 主平面做"地板/墙面/竖直性"检查 + `on` / `near` 两条关系。理由：项目名与"看房"叙事都会被追问语义，有可跑的最小版本即可答"roadmap 第一步已落地平面与支撑关系"。现有 `analyze_gravity_axis.py`、`analyze_manhattan_frames.py`、`align_pointcloud.py` 已有一半零件。
2. **换一个房间再跑一次 Test 评估**：让 final-fit / MCMC 结论获得第二个场景支撑，这是当前最硬短板（单场景单 seed）。
3. **准备一张 pipeline 图 + 一页数字卡片**：2,787 帧 / 2,655 连通分量 / 1,251 注册 / 2,999,577 高斯 / 171 Test 视图 / +0.458 dB / MAE −43.5%。面试时能随口报数比简历写得多更有说服力。

---

## 7. 一分钟口述版（电梯陈述）

> "我做了一个免标定的 3D 看房系统：用户只上传一段房间视频或几张照片，不给任何相机参数，系统自己恢复相机位姿和几何，训一个 3D Gaussian Splatting 场景，最后能在浏览器里第一人称走进去看房。
>
> 我自己写了 3DGS 的 trainer，没有用 Nerfstudio 这类现成平台，所以数据切分、坐标契约、checkpoint、评估隔离、导出全是我自己的。几何上我做了两件事：一是把 VGGT 的前馈点图和 COLMAP 的传统 SfM 混起来——滑窗 VGGT 出局部几何，做强弱相机分类和局部 BA，再用鲁棒 Sim(3) 拼成全局，取最大可靠分量去给 COLMAP 当种子；二是加了一套位姿健康门禁，在进 CUDA 之前用四道独立守卫拦掉坏几何。
>
> 最有意思的一次是排查一个 4K、10 分钟、11 GB 的视频重建失败。我没有去调参，而是先把 View Graph 的连通性打出来，发现全帧图最大连通分量有 2,655 帧，但 1,000 帧的种子子图只有 457 帧还带 191 个孤立点——瓶颈在种子选择，不在素材。把种子提到 2,500 之后，注册相机从 60 涨到 1,251。
>
> 另外我在评估上比较克制：Test 视图全程隔离从不加载，交付前那一步 fine-tune 我做了等预算三臂对照，拟合集上 MCMC 涨了 1.79 dB，但 Test 上只有 0.46 dB，我把这两个数字分开报，不拿 in-sample 当泛化。"
