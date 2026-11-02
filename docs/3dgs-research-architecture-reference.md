# 3DGS 前沿论文、架构启发与实验设计参考

> 日期：2026-09-17  
> 状态：研究参考；尚未构成实现批准、默认变更或 `codex.md` 决策  
> 面向：负责架构设计、代码实现、实验冻结、结果审计与技术写作的 Agent  
> 范围：AirSplat、SToRe3D，以及与本项目高度相关的 CVPR / ICCV / NeurIPS / IJCV 前沿工作  
> 项目：Image3D-SceneGraph（“3D看房师”）  
> 核心边界：输入默认无内参、外参、深度和位姿；坐标为归一化任意单位；不得宣称公制精度

---

## 0. 如何使用本文档

本文档有三个用途：

1. **架构设计**：理解论文解决的真实问题、适用条件和与现有模块的接口位置；
2. **编写代码**：避免把不兼容的论文范式硬塞入当前训练器，并优先复用现有数据、诊断与 replay 合同；
3. **创建实验**：将论文启发转成单变量、可归因、Test 隔离、失败可审计的项目实验。

### 0.1 权威顺序

- `codex.md` 仍是项目 plan of record，尤其 §15 和 §17；本文档不能自行改变项目方向或默认值。
- `docs/manifest-schema.md`、`gaussian-trainer-contract.md`、`gaussian-checkpoint-contract.md`、`stage2a/stage2d-contract.md` 仍是绑定合同。
- 本文档提出的“候选方向”只有在用户明确批准、写入 `codex.md` 决策日志并冻结实验合同后，才能进入实现或远端实验。

### 0.2 证据等级

| 等级 | 含义 | 可用于什么 |
|---|---|---|
| **A：逐页精读** | 已阅读本地完整正文、补充材料、公式、表格和参考文献 | 可据此提出方法级实验，但仍需代码/许可证审计 |
| **B：主来源核查** | 已通过会议/期刊正文、arXiv 全文索引、官方代码和实验表完成第一轮核查 | 可用于架构筛选和设计候选，实施前仍需下载全文逐页复核 |
| **C：脉络候选** | 仅确认论文定位与主要贡献 | 仅供继续检索，不得作为实现依据 |

本轮两篇用户提供的本地论文属于 A 级：

- `/home/owen/下载/2603.25129v2.pdf`：AirSplat，正文 15 页 + 补充材料 10 页；
- `/home/owen/下载/2605.14110v1.pdf`：SToRe3D，正文 8 页 + 补充材料 4 页。

其他重点论文当前为 B 级，来源列在文末。Agent 若要实现其中任一方法，必须先把对应 PDF 固化到本地并逐页复核。

### 0.3 不得绕过的项目规则

任何受本文档启发的改动都必须遵守：

- 不使用 Test 视图选择算法、调阈值、选 checkpoint 或决定默认；
- 不把拟合集指标称为泛化收益；
- 原始 Gaussian 模型和已完成 Job 资产不可变，实验输出使用独立派生资产或 replay 根；
- 一次只改变一个可归因的 leaf；不要同时更换几何、损失、剪枝、训练预算和渲染器；
- 新模型/权重不得在 Job 运行时下载；依赖、许可证、哈希和可用性必须预先审计；
- 失败必须保留原因、配置、日志和中间证据；不得静默回退或将 recovered 结果冒充 clean pass；
- 前端只读 manifest/asset 合同，不得了解训练器内部；
- 视频仍是有界离线重建，不得描述为实时 SLAM 或无漂移长时建图；
- 语义、物理和 scene graph 当前仍是 placeholder，本文档中的语义方向是 roadmap，不是已实现能力。

---

## 1. 当前项目基线：论文思想应接在哪里

```text
无标定图片 / 视频
        │
        ├─ 视频：6 fps 候选 → 4 fps 均匀基线 → 运动自适应补帧至 5 fps
        │          → 质量拒绝 → 有界增量注册/三角化传播
        │
        ▼
几何前端
  A. ordinary COLMAP（SIFT/ALIKED × BF/LightGlue × pairing × Mapper）
  B. experimental VGGT-BA（窗口点图 + tracks + local BA + robust global Sim(3)）
        │
        ▼
SfM pose health + 12 帧 / 70% 注册 / 80% 时序覆盖门禁
        │
        ▼
去畸变、归一化任意单位、Train/Validation/Test 冻结切分
        │
        ▼
3DGS 初始化（稀疏点或冻结 replay 中的初始化）
        │
        ├─ Project v7（默认）
        ├─ MCMC（实验，3M 全局上限）
        └─ Graphdeco（Official 对照）
        │
        ▼
Validation 选择 → 可选 SOR / final-fit 派生 → export → viewer
        │
        ├─ vggt_visibility_v1（Train-only、冻结模型、fail-soft 派生）
        └─ navigation（Train 深度 → 多视滤波 → TSDF → 碰撞/可行走区）
```

因此论文思路需先归类：

- **几何前端类**：作用于 COLMAP/VGGT-BA 之前或内部；
- **相机—几何联合优化类**：可能跨越现有严格边界，风险较高；
- **3DGS 训练正则类**：作用于 trainer 或固定拓扑 final-fit；
- **冻结模型后处理类**：作用于 Validation 选择后的派生模型；
- **前馈重建类**：需要跨场景训练，短期不能直接替换当前 per-scene trainer；
- **语义/空间智能类**：面向未来 Stage 4，而非当前已实现功能。

---

## 2. 论文地图与项目适配总览

| 论文 | Venue | 主问题 | 范式 | 对当前项目的直接价值 | 当前证据 |
|---|---|---|---|---:|---|
| AirSplat | arXiv 2026，KAIST | 前馈 pose-free 3DGS 的位姿—几何错位与 floater | 跨场景训练 | ROM 高；SCPA 当前低 | A |
| SToRe3D | arXiv 2026，Toronto/Zoox | 多视 3D 检测的 token/query 联合稀疏 | 端到端检测 | 方法论中等，技术直接性低 | A |
| LongSplat | ICCV 2025 | 无位姿长视频 3DGS | 增量优化 | 极高 | B |
| Deep Gaussian from Motion / ZeroGS | NeurIPS 2025 | 3D 几何基础模型驱动的无位姿注册与 3DGS | 增量前馈+优化 | 极高 | B |
| NopeRoomGS | NeurIPS 2025 | 无外参室内 3DGS | 局部—全局优化 | 高；但假设已知内参 | B |
| SaLon3R | IJCV 2026 | 长序列前馈重建、紧凑高斯 | 跨场景前馈 | 中长期极高 | B |
| AnchorSplat | CVPR 2026 | 3D anchor 对齐的紧凑前馈高斯 | 跨场景前馈 | 中长期高 | B |
| TokenSplat | CVPR 2026 | token 对齐替代逐像素高斯 | 跨场景前馈 | 中长期高 | B |
| Off The Grid | CVPR 2026 | 亚像素、自适应密度 Gaussian 预测 | 跨场景前馈 | 中长期中高 | B |
| C3G | CVPR 2026 | 用 2K query 高斯学习紧凑 3D 表示 | 跨场景前馈 | 语义/紧凑表示高 | B |
| VCR-GauS | NeurIPS 2024 | 深度—法线一致的高斯表面重建 | 场景级优化 | 几何/导航高 | B |
| PUP 3D-GS | CVPR 2025 | 基于 Fisher/Hessian 的不确定性剪枝 | 后训练剪枝 | 中高 | B |
| UGS-Loc | CVPR 2026 | 位姿先验与 3DGS 几何不确定下的定位细化 | 定位后处理 | 诊断/不确定性中等 | B |
| Uni3R | CVPR 2026 Highlight | 无位姿重建与开放词汇语义统一 | 跨场景前馈 | Stage 4 高 | B |
| UniSplat | CVPR 2026 | 面向空间智能的统一相机/几何/语义高斯 | 跨场景前馈 | world model 高 | B |

### 2.1 AirSplat 的直接学术脉络（C 级背景）

实现 AirSplat 或比较其定位前，应继续复核以下工作：

- SPFSplat / “No Pose at All”（ICCV 2025）：self-supervised pose-free 3DGS；
- FreeSplatter（ICCV 2025）：共同参考坐标中的 pose-free pixel-aligned Gaussians；
- FLARE（CVPR 2025）：免标定稀疏视图的几何、外观与相机前馈估计；
- SelfSplat（CVPR 2025）：无 pose / 无 3D prior 的 generalizable 3DGS；
- AnySplat、NoPoSplat、WorldMirror、DA3：AirSplat 的主要前馈比较基线。

这些论文是理解前馈 3DGS 的基线，不等于适合当前 per-scene trainer。

---

# 第一部分：两篇本地完整论文

## 3. AirSplat：Alignment and Rating for Robust Feed-Forward 3D Gaussian Splatting

### 3.1 论文定位

作者：Minh-Quan Viet Bui、Jaeho Moon、Munchurl Kim（KAIST VIC Lab）；arXiv:2603.25129v2。

> 📄 原文 PDF：[arXiv 2603.25129](https://arxiv.org/pdf/2603.25129) ｜ 本地：`/home/owen/下载/2603.25129v2.pdf`

AirSplat 训练的是 **3D-VS-VFM**（具有视图合成头的 3D Vision Foundation Model），不是经典单场景 3DGS。输入若干无标定 context 图像，网络一次前向预测：

- 相机内外参；
- 点图/深度等基础几何；
- 像素对齐高斯：
  \[
  g_i=(\mu_i,\Sigma_i,\alpha_i,c_i)
  \]
- 未见视角图像。

训练时冻结主 encoder、深度头和位姿头，只优化 Gaussian prediction head，以保留基础模型零样本几何。

### 3.2 两个被识别的问题

#### 问题 A：pose-geometry discrepancy

在 context-target 训练中：

- 场景几何只由 context 决定；
- target 位姿由 context+target 联合决定。

两条分支的信息域不同，可能形成潜在坐标错位。像素监督会错误地让高斯形变去补偿相机偏差，导致细杆、边缘、高频纹理发糊。

#### 问题 B：局部跨视图不一致

逐像素生成高斯时，不同 context 视图可能为同一表面生成多组轻微错位的 primitives，形成 floaters、重影和视角不稳定。

### 3.3 SCPA：Self-Consistent Pose Alignment

第一次预测目标位姿并渲染：

\[
\hat I_t^{(1)}=R(G,\hat P_t^{(1)})
\]

把 context 与该合成目标图再次送入冻结位姿网络：

\[
\hat P_t^{(2)}=f_\phi(I_{ctx},\hat I_t^{(1)})
\]

计算两次预测间的 SE(3) 漂移：

\[
\xi_t=
\log\left(\hat P_t^{(2)}(\hat P_t^{(1)})^{-1}\right)^\vee
\]

沿李群流形反向抵消：

\[
\hat P_t^{align}=\exp(-\xi_t^\wedge)\hat P_t^{(1)}
\]

为了防止第二次位姿预测失败，使用 minimum-error supervision：

\[
\mathcal L_{scpa}=\sum_t \min\left(
\mathcal L_{rec}(\hat I_t^{align},I_t),
\mathcal L_{rec}(\hat I_t^{(1)},I_t)
\right)
\]

\[
\mathcal L_{rec}=\|\hat I-I\|_2^2+\lambda_s LPIPS(\hat I,I)
\]

**适用边界**：SCPA 针对“位姿头和高斯头在不对称输入下产生的 gauge 错位”。当前项目的 Gaussian trainer 使用同一 SfM 模型的冻结相机，没有这个误差来源，因此不得直接接入主 trainer。

### 3.4 ROM：Rating-based Opacity Matching

对某个来自视图 \(v\) 的高斯中心 \(\mu_i\)，投影到相邻视图 \(v'\)，读取对应像素处的 3D 均值，计算尺度归一化误差：

\[
\epsilon_i=
\frac{\|\mu_i-\Pi_{v\rightarrow v'}(\mu_i)\|_2}
{\operatorname{median}(D_i)+\eta}
\]

轻量 sparse-view 3DGS 教师计算自己的误差 \(\tilde\epsilon_j\)。学生只对“比教师自身更差”的超额误差负责：

\[
E_i^{geo}=\max(0,\epsilon_i-\tilde\epsilon_j)
\]

连续评级：

\[
\bar y_i=\exp(-\lambda\cdot sg[|E_i^{geo}|])
\]

其中 \(\lambda=5\)，`sg` 为 stop-gradient。

关键设计：评级不是 opacity 的对称目标，而是**上界**：

\[
\mathcal L_{opa}=
\frac1N\sum_i\left[\max(0,\alpha_i-\bar y_i)\right]^2
\]

- 好高斯 \(\bar y_i=1\)：ROM 不干预 opacity；
- 坏高斯 \(\bar y_i\to0\)：仅当其可见性超过可信上界时压低 opacity；
- 避免把所有几何正确高斯强推向 \(\alpha=1\)，从而避免“实体化”伪影。

轻度错位由空间正则修复：

\[
\mathcal L_{geo}=\frac1N\sum_i sg[\alpha_i]\min(\epsilon_i,\tau)
\]

其中 \(\tau=2.0\)。opacity 只作 detached 权重，高可见结构优先被拉回跨视图共识，严重错误由 \(L_{opa}\) 隐去。

总损失：

\[
\mathcal L_{total}=\mathcal L_{scpa}
+\lambda_{geo}\mathcal L_{geo}
+\lambda_{opa}\mathcal L_{opa}
\]

论文取 \(\lambda_{geo}=0.1\)、\(\lambda_{opa}=1.0\)、LPIPS 权重 0.1。

### 3.5 主要证据

- 基线 DA3-GIANT；4×A100 40GB；100k 迭代；约 4.5 天；
- RE10K 12 视图：DA3 20.78/0.715/0.250 → AirSplat 23.08/0.799/0.190；
- DL3DV 12 视图：22.50/0.747/0.207，相对 DA3 +1.76 dB，LPIPS 下降约 14.4%；
- 跨数据集 ACID 16 视图：25.96/0.796/0.188；
- SPFSplat 21.32→22.76、WorldMirror 21.08→23.66、DA3 21.06→23.77（RE10K 24 视图）；
- 每步训练 2.35→3.89 秒，约 +65%，推理无额外模块；
- 只微调 Gaussian head，在 7-Scenes 点图评估中保住 DA3 基础几何；对照 AnySplat 式联合微调会损伤 VGGT 零样本几何。

### 3.6 对本项目的可借鉴结论

**可直接研究：**

1. 教师自身跨视误差作为基线，而不是把教师深度当绝对真值；
2. 单侧 opacity upper-bound；
3. “轻微错误移动、严重错误隐去”的双路径；
4. 冻结可靠几何能力，只优化高斯表达。

**不可直接搬运：**

- SCPA 到当前固定相机 trainer；
- 100k 跨场景训练框架到 per-job 训练；
- 逐像素高斯对应关系到当前自由全局高斯（需要重新定义每个高斯的 rating）；
- 只凭 NVS 指标宣称几何改善。

**最合理的项目转化**：先把现有 `vggt_visibility_v1` 的二值支持/矛盾统计扩展为 Train-only 连续评级诊断；信号成立后，再做固定拓扑、Train-only、opacity-only 派生微调。

---

## 4. SToRe3D：Sparse Token Relevance in ViTs for Efficient Multi-View 3D Object Detection

### 4.1 论文定位

作者：Sandro Papais（University of Toronto / Zoox 实习）、Lezhou Feng、Charles Cossette、Lingting Ge（Zoox）；arXiv:2605.14110v1。

> 📄 原文 PDF：[arXiv 2605.14110](https://arxiv.org/pdf/2605.14110) ｜ 本地：`/home/owen/下载/2605.14110v1.pdf`

它是自动驾驶多视角 3D 检测效率论文，不是重建或 3DGS 论文。其直接技术不可接入本项目，但三种方法论有价值：

- 计算预算应对齐下游任务相关性；
- 删除应尽量可复活，而非不可逆硬剪；
- 在决定剪什么之前先做模块敏感度 profiling。

### 4.2 规划相关性

在 \(H=5s\) 时间域内，对 ego 与 agent 的运动框取扫掠凸包：

\[
S_i(H)=conv\left(\bigcup_{\tau\in[0,H]}B_i(\tau)\right)
\]

若 agent 与 ego 走廊最近距离小于阈值即为 relevant：

\[
y_i^{rel}=1(dist(S_i(H),S_{ego}(H))\le d_{min})
\]

论文用 nuScenes 最近距离分布第 10 百分位，约 \(d_{RM}=1.2m\)，平均每帧约 3 个相关 agent。

### 4.3 2D token 与 3D query 互门控

3D query 在图像 token 上下文中评分：

\[
c_j^{qry}=CrossAttn_{def}(q_j,X_i)\oplus e_t
\]

\[
r_j^{qry}=\sigma(u^T\phi([q_j|c_j^{qry}]))
\]

图像 token 的相关性来自 top-K 相关 query 的注意力聚合：

\[
r_i^{img}=\frac1K\sum_{j\in\mathcal K^{qry}}A_{j\rightarrow i}
\]

因此 token 是否重要由 3D query 决定，query 是否重要又由图像上下文与未来走廊决定。

### 4.4 Store–Reactivate

每阶段保留 top-k 活跃 token/query：

\[
\mathcal K_\ell=TopK(r_\ell,\lfloor\rho_\ell N_\ell\rfloor)
\]

其余不删除，而是写入 buffer：

\[
S_\ell^{img}\leftarrow X_\ell[\bar K_\ell^{img}],\quad
S_\ell^{qry}\leftarrow Q_\ell[\bar K_\ell^{qry}]
\]

在 backbone/decoder 最后一层注意力前复活，以更新后的上下文补全信息。训练从 dense 线性 warm up 到目标稀疏度，避免早期塌缩。

### 4.5 主要证据

- ViT-B、1/10 保留率：0.479 mAP / 0.571 NDS / 17.7 FPS；
- ViT-L、1/10：0.521 mAP / 0.607 NDS / 5.2 FPS，dense StreamPETR 为 0.521/0.608/2.2 FPS；
- TKR=0.5 时 NDS 0.618，甚至比 dense 0.614 高；随机剪枝为 0.567；
- TKR=0.1 时仅损失 1.1% NDS、2.3% mAP，吞吐约 2.42×；
- store–reactivate 优于 hard remove；线性稀疏 warm-up 优于直接 fine-tune；
- buffer 约 20 MB，换约 397 GFLOPs（约 28%）计算下降。

### 4.6 对项目的转化边界

**可借鉴的方法论：**

1. 视频候选帧当前本来就有“candidate pool → selected → recovery candidate”，可继续保持未选帧可复活，而非物理删除；
2. 在修改 keyframe、Mapper seed 或 Gaussian pruning 前，先测算：
   - 哪些帧贡献 view-graph 连通、三角化与时序覆盖；
   - 哪些高斯贡献可见像素、几何表面、navigation collision；
3. 指标应对齐产品目标：看房导航更关心可行走区域、墙地边界与碰撞几何，而不是场景所有像素同权。

**不得照搬：**

- DETR query 结构、未来轨迹标签、nuScenes-R 指标；
- 把帧选择简单类比为 token pruning；
- 未 profiling 就增加神经 relevance head。

---

# 第二部分：长视频、位姿与几何

## 5. LongSplat（ICCV 2025）

> 📄 原文 PDF：[CVF Open Access](https://openaccess.thecvf.com/content/ICCV2025/papers/Lin_LongSplat_Robust_Unposed_3D_Gaussian_Splatting_for_Casual_Long_Videos_ICCV_2025_paper.pdf) ｜ [arXiv 2508.14041](https://arxiv.org/pdf/2508.14041)

### 5.1 方法

- MASt3R 从三个初始帧估计位姿、焦距、深度和稠密点云；
- 稠密点云体素化为自适应八叉树 anchors：密集区域递归细分，稀疏区域剪掉，重叠 anchors 合并；
- 每个新帧通过 MASt3R correspondences → 2D–3D → PnP/RANSAC → 3DGS 光度位姿细化；
- MASt3R 深度与当前渲染深度对齐尺度，识别未观察区域并生长 anchors；
- 可见性自适应 local window 优化与 global optimization 交替；PnP 失败先做全局重优化再重试；
- 最后全部相机与高斯全局 refinement。

目标结合 RGB、尺度对齐深度与 2D correspondence reprojection。

### 5.2 证据

- Free：27.88 PSNR / 0.85 SSIM / 0.17 LPIPS；
- 自适应八叉树 101 MB，对照稠密逐像素反投影 799 MB；
- 281.71 FPS；约一小时 / RTX 4090；
- 去掉位姿估计或 global optimization，PSNR 分别降至 20.19 和 20.50；
- 官方 `NVlabs/LongSplat` 可复现，自述局限为静态场景、固定内参、动态物体与变焦不支持。

### 5.3 项目启发

- 当前 VGGT-BA 只为 COLMAP seed 提供几何；LongSplat 展示了“局部 Gaussian map 同时作为新帧定位地图”的另一条路线；
- 可见性自适应窗口比固定帧数更合理，可用于研究当前局部 BA 窗口是否应由 track/view-graph 支撑而不是纯时间长度决定；
- 深度尺度对齐只作为几何先验，不可冒充公制尺度；
- 直接接入会跨越现有“先几何、后高斯”合同，短期只做算法对照，不应修改默认路径。

---

## 6. Deep Gaussian from Motion / ZeroGS（NeurIPS 2025）

> 📄 原文 PDF：[NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/1c2b1c8f7d317719a9ce32dd7386ba35-Paper-Conference.pdf) ｜ [arXiv 2411.15779（ZeroGS 版）](https://arxiv.org/pdf/2411.15779)

### 6.1 方法

- 基于 Spann3R/DUSt3R 的网络预测共同世界坐标中的稠密点图和每像素 Gaussian 参数；
- NetVLAD 图像相似图选择最大度 seed，固定其位姿为单位矩阵；
- 新图的预测全局点提供 2D–3D 对应，经 DSAC/RANSAC + PnP 得粗位姿；
- 多视 tracks 经过并查集合并与歧义去除；
- 核心 point-to-camera-ray consistency：

\[
\min_{X_i,C_k}\sum_{i,k}\rho\left(
\|d_{i,k}\nu_{i,k}-(X_i-C_k)\|_2
\right)
\]

- 固定 seed pose 消除 gauge；固定一个相邻尺度消除 scale ambiguity；
- 增量过程中旧相机固定，只优化新注册相机；最终做一次全局 ray consistency；
- 最后切换显式 3DGS refinement。

### 6.2 证据与局限

LLFF 八场景平均粗→细：

- 旋转约 10.20°→0.17°；
- 平移约 0.230→0.0053。

约 21 GB 显存 / RTX 4090；LLFF 约 25 分钟，Mip-NeRF360 / Tanks and Temples 约 2 小时。粗初始化严重错误时仍会失败；公开仓库的完整端到端实现状态需重新审计。

### 6.3 项目启发

这是当前 VGGT-BA 最值得优先研究的残差形式。现有窗口 Sim(3) 对齐回答“窗口如何拼”；ray consistency 回答“每个共享 track 的 3D 点是否真落在所有相机射线上”。候选实现必须：

- 只消费 Train/几何证据；
- 保持任意尺度语义；
- 与现有 PyCOLMAP BA 做单 leaf A/B，不能同时改窗口、track 过滤和 loss；
- 先离线读取 frozen window graph 计算残差分布，不立即写回相机。

---

## 7. NopeRoomGS（NeurIPS 2025）

> 📄 原文 PDF：[NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/464012c83279e19be4cd42c25f341c92-Paper-Conference.pdf)

### 7.1 方法

输入无外参 RGB 序列，但**已知内参**：

- 重叠局部 clips；
- 单目深度初始化 + CoTracker 准稠密轨迹 + 多帧重投影；
- 强单目深度模型提供尺度—平移不变先验；
- 局部点云下采样为局部高斯；
- progressive global fusion；
- 全局 photometric、depth-alignment 和 piecewise-planarity pose loss；
- 高斯固定时优化相机，相机固定时优化高斯，避免梯度互相补偿。

### 7.2 证据

Replica：0.148 RPEt / 0.126 RPEr / 0.009 ATE，32.14 PSNR / 0.90 SSIM / 0.09 LPIPS；ScanNet：25.38/0.72/0.20。

消融：

- 无局部 neural representation：PSNR 14.82；
- 无分段平面：30.34；
- 无交替优化：28.28；
- 无深度对齐：31.42，ATE 明显恶化。

### 7.3 项目启发

- 平面先验非常适合室内墙/地/顶/桌面，但需图像边缘或几何边界防止跨物体错误传播；
- 现有 `align_pointcloud.py`、`analyze_gravity_axis.py`、`analyze_manhattan_frames.py` 与 navigation TSDF 可提供只读平面证据；
- 若未来做 pose/Gaussian joint refinement，优先交替优化，不要同一步共同反传；
- 已知内参假设与项目不一致，不能把其结果当成完全免标定证据。

---

## 8. UGS-Loc：不确定性感知 3DGS 位姿细化（CVPR 2026）

论文：Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty。

> 📄 原文 PDF：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/papers/Kong_Rethinking_Pose_Refinement_in_3D_Gaussian_Splatting_under_Pose_Prior_CVPR_2026_paper.pdf) ｜ [arXiv 2603.16538](https://arxiv.org/pdf/2603.16538)

它不是重建方法，而是基于已训练 3DGS 的定位 refinement：

- 用多个 pose particles 表示先验不确定性；
- 每个 particle 渲染、匹配、PnP 预校正；
- Fisher 信息估计 Gaussian 几何不确定性并渲染成 depth uncertainty map；
- PnP 采样权重：

\[
s_i=e^{-\beta\bar U(r_i)}+\epsilon
\]

低几何不确定区域优先进入 PnP；particle 权重同时考虑匹配置信与几何可靠性。

**项目价值**：不是接入当前重建，而是提示任何“用 3DGS 渲染深度反过来修位姿”的方案都必须考虑 Gaussian 几何本身不确定，不能默认渲染深度是真值。其 Monte Carlo 多假设成本较高，短期更适合作为诊断思想。

---

# 第三部分：前馈紧凑 3DGS

## 9. SaLon3R（IJCV 2026）

> 📄 原文 PDF：[Springer/IJCV](https://link.springer.com/content/pdf/10.1007/s11263-026-02965-2.pdf) ｜ [arXiv 2510.15072](https://arxiv.org/pdf/2510.15072)

### 9.1 方法

支持 VGGT offline 与 CUT3R online：

1. backbone 输出相机、世界点图、Gaussian latent 与 saliency；
2. saliency-dependent voxel scaling：高复杂区域细量化，平坦区粗量化；
3. 同一体素点按 saliency 加权融合成 anchor；
4. Point Transformer V3 五阶段 U-Net，Z-order/Hilbert 序列化与局部 patch attention；
5. 每 anchor 最多生成 4 个 residual Gaussians；
6. refined saliency 调制 opacity，并通过 mask 剪除冗余；
7. online 模式维护全局 Gaussian buffer，只取当前视锥邻近部分局部更新再合并。

训练使用 RGB MSE+LPIPS、置信掩码深度/深度梯度一致性、边缘感知深度平滑；无需 GT pose/depth。

### 9.2 证据

ScanNet 100 views offline：21.05 PSNR / 0.761 SSIM / 0.033 AbsRel / 3.572M Gaussians / 0.09 s per frame；AnySplat 20.47/0.697/0.367/9.686M。论文报告整体减少 50%–90% 冗余，online 超过 10 FPS。

### 9.3 项目启发

- 长期可将 VGGT 从“几何 seed”升级成“点图+latent→稀疏 anchor→粗高斯初始化器”；
- 当前无需训练网络，也可先研究 saliency-aware spatial budget 是否能替代统一随机/Morton 初始化预算；
- online global buffer 类似 SToRe3D 的 store–reactivate，但属于复杂新架构，当前 Stage 5 bounded offline 路线不应提前扩成实时系统。

---

## 10. AnchorSplat（CVPR 2026）

> 📄 原文 PDF：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_AnchorSplat_Feed-Forward_3D_Gaussian_Splatting_With_3D_Geometric_Priors_CVPR_2026_paper.pdf) ｜ [arXiv 2604.07053](https://arxiv.org/pdf/2604.07053)

### 10.1 方法

- 预训练 MVS 输出相机和深度；
- 反投影点通过体素尺寸与 FPS 形成稀疏 3D anchors；
- RGB、depth、camera-ray 特征投影到 anchors；
- Transformer decoder 每 anchor 生成 4 个高斯；
- Gaussian Refiner 把图像渲染误差反投影回高斯属性做第二阶段更新。

### 10.2 证据

ScanNet++ 32 源视图：21.48 PSNR / 0.79 SSIM / 0.066 AbsRel / 247,153 Gaussians / 5.52 s。AnySplat：20.20/0.73/0.160/5,550,940/6.83 s。约 22× 更少高斯；视图数从 32 增到 64，Gaussian count 不变。LPIPS 0.38，差于 AnySplat 的 0.32，不能称全指标领先。

### 10.3 项目启发

最自然的长期接口是：

```text
COLMAP / VGGT-BA 几何
        ↓
空间 anchors + 跨视特征
        ↓
少量粗 Gaussians
        ↓
现有 Project/MCMC refinement
```

官方实现以 Ascend NPU 为先，checkpoint 发布状态需再次核查；不可把它直接加入 base dependency。

---

## 11. TokenSplat（CVPR 2026）

> 📄 原文 PDF：[arXiv 2603.00697](https://arxiv.org/pdf/2603.00697) ｜ CVF 页面：[HTML](https://openaccess.thecvf.com/content/CVPR2026/html/Li_TokenSplat_Token-aligned_3D_Gaussian_Splatting_for_Feed-forward_Pose-free_Reconstruction_CVPR_2026_paper.html)

### 11.1 方法

- 预测 coarse token position 与 fusion confidence；
- 跨视图对应 token 先聚合，再生成高斯；
- one-to-many decoder 使 Gaussian count 与图像分辨率解耦；
- camera tokens 与 scene tokens 通过 asymmetric dual-flow decoder 限制信息方向，降低 pose 与 appearance 相互污染。

### 11.2 证据

RE10K 8 views：26.15/0.858/0.135；换成 pixel-aligned head 后 25.33/0.832/0.159；移除 asymmetric decoder，旋转误差 0.458→0.504。ScanNet 28 views：26.87 PSNR，长输入扩展优于 AnySplat。

### 11.3 项目启发

AirSplat ROM 是“生成后评级清理”；TokenSplat 是“生成前跨视融合防重复”。若未来训练前馈高斯头，优先避免 one-pixel-one-Gaussian 设计，而不是依赖后续大量剪枝。当前未找到明确官方代码，实施前不可依据第三方摘要重写。

---

## 12. Off The Grid（CVPR 2026）

> 📄 原文 PDF：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/papers/Moreau_Off_The_Grid_Detection_of_Primitives_for_Feed-Forward_3D_Gaussian_CVPR_2026_paper.pdf) ｜ [arXiv 2512.15508](https://arxiv.org/pdf/2512.15508)

### 12.1 方法

- VGGT 输出 depth/confidence/camera/latent；
- 每个 14×14 patch 预测多张 heatmap；
- spatial softmax + DSNT/soft-argmax 得亚像素浮点位置；
- patch Shannon entropy 决定 16/32/64 primitives；
- 双线性采样 RGB/depth/feature，反投影后 MLP 预测 scale/rotation/opacity/confidence；
- opacity×confidence 低的基元在推理时剪除。

### 12.2 证据

每像素 primitive：0.1431，对照 AnySplat 0.8141、pixel-aligned 1.0；aggregate NVS 为 21.21/0.6470/0.3532，高于 AnySplat 17.71/0.5075/0.3937。自适应密度与 confidence pruning 均有独立消融收益。

### 12.3 项目启发

- 初始化预算不应均匀分给每个像素/点；
- 可在不训练新网络的前提下，先用现有图像梯度、track density、SfM reprojection support 研究预算分配；
- 论文采用的固定 opacity×confidence 阈值属于其模型设置，不能复制到项目现有 opacity 语义。

---

## 13. C3G（CVPR 2026）

> 📄 原文 PDF：[arXiv 2512.04021](https://arxiv.org/pdf/2512.04021) ｜ CVF 页面：[HTML](https://openaccess.thecvf.com/content/CVPR2026/html/An_Learning_Compact_3D_Representations_from_Feed-Forward_Novel_View_Synthesis_CVPR_2026_paper.html)

### 13.1 方法

- VGGT 编码 unposed multi-view；
- 2,048 个 learnable Gaussian queries 与图像 token 经两层全 self-attention；
- 每个 query 预测一个全局 Gaussian；
- progressive low-pass filtering 在训练早期放大 projected Gaussian，逐步收缩，避免稀疏梯度坍塌；
- C3G-F 冻结 Gaussian decoder 学得的 attention 结构，只训练 value projection，把任意 2D 特征聚合到 3D。

### 13.2 证据

RE10K 24 views：约 2K Gaussians、23.797 PSNR；AnySplat 1.5M–3.31M、24.105 PSNR。短时 test-time optimization 后 C3G 约 26K–28K Gaussians、29.987 PSNR。24 视图推理 0.466 s、4,372 MiB、3,083 FPS；训练需 8×H100、450k steps。

### 13.3 项目启发

- query-based global Gaussian representation 对语义特征抬升和 scene graph 更有价值，而不只是压缩；
- 2K Gaussian 的 NVS 模型不等于适合高保真看房交付；其短时优化后数量上升；
- 可将 C3G-F 作为未来 2D 语义证据跨视融合的研究参考，不应现在训练同规模模型。

---

# 第四部分：几何一致性、剪枝与表面质量

## 14. VCR-GauS（NeurIPS 2024）

> 📄 原文 PDF：[NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/fc9f83d9925e6885e8f1ae1e17b3c44b-Paper-Conference.pdf) ｜ [arXiv 2406.05774](https://arxiv.org/pdf/2406.05774)

### 14.1 方法

它指出直接监督 Gaussian normal 主要改变旋转，难以约束位置和尺度；单目 pseudo-normal 又可能跨视不一致。

- 将高斯压平为 planar primitives；
- 求 camera ray 与 Gaussian plane 的交点深度；
- 由邻近重建 3D depth points 导出 normal，使 normal loss 能回传到 position/rotation/scale；
- 用跨视 pseudo-normal 一致性得到 confidence map；
- 只 densify 首个表面交点相关的大高斯；
- 大高斯沿最大尺度轴确定性一分为二，减少随机 split 聚团与凸起。

### 14.2 证据

Tanks and Temples mean F1：3DGS 0.09、2DGS 0.30、VCR-GauS 0.40；训练约 53 分钟，145 FPS。无 D-Normal 为 0.30，无 confidence weighting 为 0.36，无专门 densification/splitting 为 0.33。

### 14.3 项目启发

- 若目标是 navigation mesh / surface fidelity，不能只优化 RGB PSNR；
- 可将现有 TSDF/navigation 资产用于评估表面，而不是直接训练；
- “置信度加权伪几何”与 AirSplat“教师自身误差基线”共同表明：外部几何教师不可无条件当真；
- 替换当前 3D Gaussian 为 planar surfels 会跨模型/renderer/checkpoint/export 合同，短期不做。

---

## 15. PUP 3D-GS（CVPR 2025）

> 📄 原文 PDF：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Hanson_PUP_3D-GS_Principled_Uncertainty_Pruning_for_3D_Gaussian_Splatting_CVPR_2025_paper.pdf) ｜ [arXiv 2406.10219](https://arxiv.org/pdf/2406.10219)

### 15.1 方法

对已收敛模型的 L2 重建损失，用 Gauss–Newton/Fisher 近似：

\[
H\approx\sum_\phi J_\phi J_\phi^T
\]

每个高斯只保留 position+scale 的 6×6 block：

\[
H_i=\sum_{\phi,p}J_{i,\phi,p}J_{i,\phi,p}^T
\]

重要度：

\[
U_i=\log|H_i|
\]

高 \(U_i\) 表示位置/尺度轻微变化会显著影响训练视图，因而约束强、重要；低值表示冗余或不确定。4×4 patch 近似 Fisher；两轮剪枝 80%→5k refine→再剪剩余 50%→5k refine，总计 90%。

### 15.2 证据

Mip-NeRF360：27.47 dB / 64 FPS / 746 MB → 26.67 dB / 205 FPS / 74.65 MB；平均加速约 3.56×。90% 一次性剪枝差于两阶段。限制：依赖已收敛假设；约需 36N Fisher 数值，3M 高斯上的实际峰值必须先 profile。

### 15.3 项目启发

PUP 与 ROM 回答不同问题：

- ROM：几何是否比教师的自一致性还差；
- PUP：Gaussian 是否被训练视图充分约束、是否对渲染敏感。

未来优先级可组合为二维决策，而不是加权成一个未经验证的总分：

| 几何一致性 | Fisher 重要度 | 建议解释 |
|---|---|---|
| 好 | 高 | 核心结构，保留 |
| 好 | 低 | 可能冗余，压缩候选 |
| 差 | 高 | 可能是重要结构/教师错误，禁止直接删除，需人工/额外证据 |
| 差 | 低 | 最强清理候选 |

第一实验只能计算两个独立分布/象限，不得立即发明综合权重。

---

# 第五部分：语义与空间智能

## 16. Uni3R（CVPR 2026 Highlight）

> 📄 原文 PDF：[arXiv 2508.03643](https://arxiv.org/pdf/2508.03643) ｜ CVF 页面：[HTML](https://openaccess.thecvf.com/content/CVPR2026/html/Sun_Uni3R_Unified_3D_Reconstruction_and_Semantic_Understanding_via_Generalizable_Gaussian_CVPR_2026_paper.html)

### 16.1 方法

输入任意数量 unposed 图像及内参，通过 DINOv2 + 24 层交替 intra-frame / cross-frame Transformer，像素对齐输出：

- position / scale / rotation；
- opacity / RGB；
- 高维 semantic feature。

语义特征用 autoencoder 压缩，随 Gaussian rasterization 后解码回 CLIP-compatible 空间，通过文本类别余弦相似度实现开放词汇分割。

总损失：

\[
L_{total}=L_{rgb}+0.02L_{sem}+0.005L_{geo}
\]

- RGB：L1+LPIPS；
- semantic：冻结 LSeg 教师特征蒸馏；
- geometry：冻结 VGGT 点图，保留 top 90% confidence，经 Umeyama 对齐后做 one-way Chamfer。

### 16.2 证据与边界

ScanNet：0.162 s、target mIoU 55.84%、PSNR 25.53、SSIM 0.8727、LPIPS 0.1380。移除 semantic loss，mIoU 约 54.84%→1.83%；移除 geometry loss，depth relative error 3.9→5.6。

它要求相机内参输入，因此不完全符合项目 calibration-free 边界。官方代码可用，但模型/许可证/权重仍需独立审计。

### 16.3 项目启发

语义可以成为 Gaussian 的一等属性，通过同一 visibility/rasterization 完成跨视融合，而不一定在最终点云上贴标签。但 Stage 4 首版仍应走最小路径：2D evidence→现有几何 lift→跨视关联→版本化 scene graph；不要因为 Uni3R 存在就提前训练统一 foundation model。

---

## 17. UniSplat（CVPR 2026）

> 📄 原文 PDF：[arXiv 2604.10573](https://arxiv.org/pdf/2604.10573) ｜ CVF 页面：[HTML](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_Learning_3D_Representations_for_Spatial_Intelligence_from_Unposed_Multi-View_Images_CVPR_2026_paper.html)

### 17.1 方法

统一输出 camera、point map、semantic Gaussians 和 appearance Gaussians：

- encoder 随机 mask image patches；
- coarse camera/Gaussian 生成 rendered importance map；
- decoder 再 preferentially mask 几何重要区域，迫使跨视 3D 推理；
- 256 coarse Gaussian tokens→每个生成约 10 semantic Gaussians→再细化为 appearance Gaussians；
- 点图投影 appearance 与 RGB Gaussian render 对齐；semantic point feature 与 semantic Gaussian render 对齐；
- 损失包括 RGB、LSeg semantic distillation、VGGT point/camera distillation 与 pose-conditioned recalibration。

### 17.2 证据

ScanNet：0.041 s、target mIoU 0.5625、PSNR 25.65、SSIM 0.8782、LPIPS 0.1353。移除 recalibration 后 PSNR 24.35、AbsRel 3.52；移除几何蒸馏后 PSNR 24.23、AbsRel 3.59。还在 embodied control、3D detection、semantic occupancy 和 3D grounding 上评估 frozen encoder。

### 17.3 项目启发

UniSplat 最值得研究的不是其 NVS 分数，而是把 camera/geometry/appearance/semantics 锁在同一坐标系的 **recalibration loss**。未来 scene graph 若要避免“2D label 对但 3D 锚点错”，可借鉴投影证据与渲染证据互相校验的原则。

---

# 第六部分：从论文到本项目的候选架构

## 18. 三层演进，而不是一次重写

### 18.1 近期：强化现有显式几何与场景级 3DGS

```text
现有 selected frames / tracks / VGGT windows
                │
                ├─ RS-1：point-to-camera-ray 只读审计
                ├─ RS-2：室内 piecewise-planarity 只读审计
                └─ RS-3：连续几何 rating 只读审计
                         │
                         ▼
         只有审计信号通过预声明区分度 gate
                         │
                ├─ RS-4：ROM opacity-only 固定拓扑派生
                └─ RS-5：Fisher importance 后训练审计
```

不改变：普通 COLMAP 默认、VGGT-BA research-only 身份、12/70%/80% gate、Test 隔离、Original stable asset。

### 18.2 中期：局部—全局视频重建研究分支

参考 LongSplat / NopeRoomGS：

```text
视频 clips
  ↓
局部相机 + 点图 + tracks
  ↓
局部 BA / ray consistency
  ↓
局部几何或 Gaussian submap
  ↓
全局 pose graph / Sim(3)
  ↓
最终统一 Project/MCMC refinement
```

该方向会改变 Stage 5 架构，必须先写新合同；不得把它混入 `standard_v2` 的局部恢复小改动中。

### 18.3 长期：前馈粗模型 + 场景级 refinement + 语义

参考 SaLon3R / AnchorSplat / TokenSplat / C3G / Uni3R / UniSplat：

```text
无标定多视图
   ↓
冻结或轻量微调 VGGT/3DVFM
   ↓
跨视 token / 3D anchor / global queries
   ↓
紧凑粗 Gaussian + camera + point map
   ↓
现有 readiness / split / Validation / replay
   ↓
Project/MCMC 短时 refinement
   ↓
语义 feature lift + scene graph
```

这不是当前功能，且需要跨场景数据、多卡训练、许可与发布权重治理。

---

# 第七部分：建议创建的有界实验

以下实验编号只属于本文档，不能与 codex Gate G1/G2 等混用。

## 19. RS-1：Point-to-Camera-Ray 只读残差审计

**来源**：Deep Gaussian from Motion。

**假设**：当前 VGGT-BA 失败窗口/弱相机在 point-to-ray residual 上可与健康窗口分离；若不能分离，增加该优化项没有依据。

**输入**：一个冻结 `vggt_ba` preparation 的：

- window cameras；
- local/global point maps；
- tracks 与 observation confidence；
- 当前 Sim(3) 图结果；
- 不读取 Validation/Test RGB。

**实现边界**：只新增分析脚本，读取证据并输出 JSON；不写相机、不跑 BA、不训练。

**输出**：

- 每 observation/track/window 的 robust residual 分布；
- residual 与 strong/weak 分类、断连、pose health、后续 COLMAP 注册成功的相关性；
- gauge/scale 处理记录；
- 输入哈希。

**继续条件**：预先冻结“健康与失败是否具有稳定区分度”的统计 gate；不得看 Test 决定。

**潜在代码位置**：优先复用 `geometry/vggt_ba.py` 的窗口、track、camera 数据结构，新增 `scripts/analyze_vggt_ray_consistency.py`；不要把审计逻辑塞进 runner。

---

## 20. RS-2：室内分段平面一致性审计

**来源**：NopeRoomGS、VCR-GauS。

**假设**：墙/地/顶等大平面上的跨帧不一致能解释 navigation mesh 破洞或位姿局部波动。

**输入**：已有 aligned sparse/dense point cloud、Train cameras、rendered Train depth、navigation TSDF diagnostics。

**第一阶段只读**：

- 从已存在的 dominant-plane / Manhattan / gravity 工具复用结果；
- 计算相邻 Train 视图中局部 plane normal 与 offset 的一致性；
- 使用 RGB/depth edge 阻断跨边界区域；
- 不把平面当硬门禁，不修相机。

**指标**：normal angular residual、plane offset residual（任意单位）、边界泄漏率、与 collision mesh connectedness/walkable mask 的关系。

**禁止**：宣称墙面距离为米；对非平面物体施加强约束；同时更改 alignment 与 TSDF 参数。

---

## 21. RS-3：AirSplat 连续 rating audit

**来源**：AirSplat ROM。

**假设**：教师相对误差 \(E_i^{geo}\) 比现有二值 `support==0 && contradictions>=2` 更能区分 floater 与有效薄结构。

**输入**：`vggt_visibility_v1` 已有 `DepthEvidence`、冻结 Gaussian model、Train-only VGGT depth/confidence/scale alignment。

**步骤**：

1. 计算每个 Gaussian 对各相邻 Train 视图的连续深度/3D 不一致；
2. 计算 VGGT 教师在对应视图对上的自身不一致；
3. 得到超额误差和连续 rating；
4. 不修改 opacity，只输出分布和可视化 PLY 属性；
5. 与现有 hard filter 原因、floater census、SfM surface distance、screen size、navigation mesh support 交叉分析。

**关键原则**：教师自误差必须进入基线；不得简化为 `|gaussian_depth - vggt_depth|`。

**继续条件**：rating 必须在至少两个冻结场景上显示可重复区分度；单场景视觉案例不足以进入训练。

---

## 22. RS-4：固定拓扑 ROM opacity-only 派生

**前置**：RS-3 通过；用户单独批准实现和训练。

**假设**：单侧 opacity upper-bound 在不改变 topology 的条件下，可降低自由空间 floater 且不损伤 held-out Validation 与 navigation。

**臂**：

1. Original Selection/SOR；
2. 当前 `vggt_visibility_v1` hard derivative；
3. ROM opacity-only fixed-topology Train-only derivative；
4. 只有 3 通过后才增加 ROM+position regularization，不能同批起跑。

**冻结**：同一 replay、模型、Train/Validation split、相机、步数、采样预算、renderer、loss 基线；无 clone/split/densify/reset/recovery-prune，以免归因混乱。

**监督**：Train RGB + 冻结 rating；Validation 仅评估，不参与优化；Test 不加载。

**报告**：

- raw/display PSNR/SSIM；
- floater census 与连续 contradiction；
- opacity distribution、有效可见数、若未物理 prune 则明确 count 未下降；
- 细杆/门框/墙地边缘的固定 ROI；
- navigation build/connectedness/spawn/walkable；
- time/memory/model hash。

**失败条件**：系统性孔洞、薄结构消失、navigation 失败、Validation 明显回退、rating/teacher 输入不完整；失败不得静默回 Original 并报告成功。

---

## 23. RS-5：Fisher importance audit，不立即剪枝

**来源**：PUP 3D-GS、UGS-Loc。

**假设**：position+scale Fisher sensitivity 能补充 opacity 与几何 rating，区分“可见但冗余”与“重要薄结构”。

**阶段 1**：在较小冻结场景或 Gaussian 子集上 profile 计算成本与显存；输出 logdet 分布和异常数值，不剪模型。

**阶段 2**：与 ROM rating 做四象限分析，保持两个原始分数，不发明加权总分。

**阶段 3（另行批准）**：若资源与区分度通过，再做 post-hoc 分阶段 prune/refine；Original 不可变。

**风险**：3M Gaussians × 36 block values 仅原始数值就很大，且需要 raster Jacobian；8GB GPU 上不得假定可行。优先离线/分块/较小场景证明。

---

## 24. RS-6：SToRe3D 式“可复活候选”审计

**来源**：SToRe3D。

**目标**：验证当前视频 pipeline 的被舍弃候选中，哪些帧在后续 view graph / gap recovery 中有价值，而不是训练 relevance 网络。

**设计**：

- 将初始未选 candidate 保留为 immutable pool；现有实现已经接近这一原则；
- 对被 recovery 激活的帧统计：清晰度、运动、track degree、跨 gap bridge、注册贡献；
- 分别 profile extraction、matching、Mapper、registration、triangulation 的成本敏感度；
- 判断瓶颈来自 frame 数、pair 数、query/track 数还是 BA，而非笼统称“帧太多”。

**禁止**：直接增加 learned relevance head；使用最终 Test quality 反推帧选择。

---

## 25. RS-7：前馈 anchor initializer 可行性（仅研究设计）

**来源**：AnchorSplat、SaLon3R、TokenSplat、Off The Grid、C3G。

**短期问题**：在不训练新 foundation model 的前提下，现有 COLMAP/VGGT 点与跨视特征是否足以构造更紧凑、更均匀的 3DGS 初始化？

**可先做的非学习对照**：

- sparse 3NN baseline；
- Morton-stratified points；
- voxel/FPS anchors；
- track support / image-gradient / reprojection-error 分层预算。

每臂保持相同初始化点数、trainer、split、预算和随机种子，只比较收敛、Validation、Gaussian growth、资源与 geometry readiness。不要同时实现 Transformer decoder。

---

## 26. RS-8：语义 Gaussian 最小 smoke（Stage 4 前置研究）

**来源**：Uni3R、UniSplat、C3G-F。

**最小目标**：不是训练统一模型，而是验证“2D semantic evidence 经现有 visibility 投影到 Gaussian/point 资产后，能否稳定跨视融合”。

**约束**：

- 只用冻结 2D evidence 与已有几何；
- 每个语义结论保留 view/image/mask/score provenance；
- 不把 VLM/LSeg proposal 当事实；
- 不改变 Gaussian renderer/export 格式，首版可使用 sidecar asset；
- scene graph schema 落地前不做 frontend feature tour。

---

# 第八部分：Agent 实施规则

## 27. 架构 Agent

在提出方案时必须回答：

1. 论文范式是跨场景前馈训练、单场景优化、后处理还是定位？
2. 它假设已知内参/外参/深度吗？与本项目 calibration-free 边界是否冲突？
3. 它应接在 geometry、trainer、final-fit、postprocess、navigation 还是 semantics？
4. 是否改变 manifest/checkpoint/export/renderer 坐标合同？
5. 是否需要新权重、训练数据、许可证或 GPU 资源？
6. 能否先做只读诊断，而不是直接改默认？
7. 最小单 leaf 实验是什么？
8. 失败时哪些证据必须保留？

不得仅凭论文 SOTA 数字建议“整合”。

## 28. 编码 Agent

实施前：

- 读取 `codex.md` 当前 §17；
- 搜索现有 `scripts/analyze_*`、`evaluate_*`、replay、DepthEvidence、pose-health 与 navigation 工具，优先复用；
- 若论文代码要引入，先审计许可证、commit、依赖、权重、CUDA/PyTorch 兼容和 runtime download；
- 新研究逻辑优先做独立 CLI / analyzer；没有通过只读证据 gate 前，不进 Job 默认路径；
- 任何非平凡逻辑至少留一个最小可运行测试；
- 不修改历史 evidence schema，新增 schema/version 或新资产角色；
- 不把研究参数暴露给普通 frontend 用户。

## 29. 实验 Agent

必须先冻结：

- hypothesis；
- input/replay hash；
- single changed factor；
- Train/Validation/Test 使用边界；
- seed、budget、hardware、code SHA；
- success/failure gates；
- output root 与不可覆盖规则；
- 是否允许 fallback；默认不允许将 fallback 计为该方法成功。

报告至少包含：

- paired per-view/per-frame 分布，而非只报均值；
- improved/regressed count；
- 资源峰值与失败；
- 几何质量和外观质量分开；
- 任意单位声明；
- 单场景/单 seed/特定硬件限制；
- not-run 指标（如当前 LPIPS trunk 未获批）不得伪装为缺失无关。

## 30. 写作 Agent

- 可写公开论文原理和项目公开架构；
- 不公开项目私有 3DGS 核心实现细节、实时渲染内部逻辑、私有阈值、未公开打分权重和敏感消融资产；
- 不把 research-only option 描述为产品默认；
- 不把 recovered/fallback 结果当 clean method evidence；
- 不把 RGB NVS 改善等同于几何准确；
- 不把 GT-camera Sim(3) aligned ETH3D 结果写成原生公制尺度恢复。

---

# 第九部分：明确不要做什么

1. **不要把 AirSplat 整套塞进当前 Project trainer。** 它训练跨场景 Gaussian prediction head，当前 trainer 优化场景参数，问题定义不同。
2. **不要在第一轮同时实现 ROM 的 opacity 与 position 正则。** 先证明 rating，再 opacity-only，再单独位置项。
3. **不要把 VGGT depth 直接当真值。** 必须包含 confidence、scale alignment、教师自身跨视误差或 fail-soft 边界。
4. **不要同时优化相机和高斯而没有交替阶段/约束。** NopeRoomGS 与多篇 pose-free 工作均显示二者会互相补偿。
5. **不要因 PUP 可剪 90% 就在 3M 模型上直接跑。** 先做显存/时间 profiling 和小规模审计。
6. **不要用 PSNR 单独决定几何算法。** 至少同时检查 pose health、depth/point consistency、surface/navigation 与薄结构。
7. **不要把前馈模型的百万级多卡训练加入短期 roadmap。** AnchorSplat/C3G/SaLon3R/UniSplat 属中长期研究。
8. **不要把 SToRe3D 的驾驶 relevance label 照搬到看房。** 只借鉴下游相关性、store–reactivate 和敏感度 profiling。
9. **不要因为论文写 pose-free 就默认 calibration-free。** NopeRoomGS、Uni3R 等仍假设内参。
10. **不要未经用户批准启动远端 GPU 训练或 Test 消费。** 研究文档不是执行授权。

---

# 第十部分：推荐阅读与实施顺序

## 31. 逐页精读优先级

1. **LongSplat**：直接对应长视频、local/global、pose drift 与紧凑高斯；
2. **Deep Gaussian from Motion**：直接对应 VGGT-BA、tracks、ray consistency；
3. **NopeRoomGS**：直接对应室内平面与 pose/Gaussian 交替；
4. **VCR-GauS**：对应表面质量与 navigation mesh；
5. **PUP 3D-GS**：对应后训练重要性与压缩；
6. **SaLon3R / AnchorSplat**：对应中长期前馈 initializer；
7. **Uni3R / UniSplat / C3G**：对应 Stage 4 semantics 与 world model。

## 32. 项目推进优先级

```text
先诊断，不改模型：
RS-1 ray consistency
  ↓
RS-2 indoor planarity
  ↓
RS-3 continuous rating

有证据后再做派生：
RS-4 ROM opacity-only
  ↓
RS-5 Fisher audit/pruning

另立合同的中长期路线：
LongSplat-style local/global
  ↓
Anchor/Token/Query feed-forward initializer
  ↓
semantic Gaussian / scene graph
```

短期最有希望形成项目原创组合的是：

```text
VGGT / tracks
  → point-to-camera-ray consistency
  → 室内局部平面审计
  → Project/MCMC
  → AirSplat teacher-relative rating
  → 单侧 opacity ceiling
  → Fisher importance 二维审计
  → Validation + navigation 双重验收
```

这些信号分别回答不同问题，不应过早压成一个综合分数：

- ray consistency：点与相机是否自洽；
- planarity：室内大结构是否稳定；
- ROM：Gaussian 是否比教师自身一致性更差；
- Fisher：Gaussian 是否被视图充分约束、对渲染是否重要；
- Validation：外观是否退化；
- navigation：清理后物理可行走结构是否仍完整。

---

# 来源

## PDF 直链速查

| 论文 | 原文 PDF |
|---|---|
| AirSplat | [arXiv PDF](https://arxiv.org/pdf/2603.25129)（本地 `/home/owen/下载/2603.25129v2.pdf`） |
| SToRe3D | [arXiv PDF](https://arxiv.org/pdf/2605.14110)（本地 `/home/owen/下载/2605.14110v1.pdf`） |
| LongSplat | [CVF PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Lin_LongSplat_Robust_Unposed_3D_Gaussian_Splatting_for_Casual_Long_Videos_ICCV_2025_paper.pdf) ｜ [arXiv](https://arxiv.org/pdf/2508.14041) |
| Deep Gaussian from Motion | [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/1c2b1c8f7d317719a9ce32dd7386ba35-Paper-Conference.pdf) ｜ [arXiv/ZeroGS](https://arxiv.org/pdf/2411.15779) |
| NopeRoomGS | [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/464012c83279e19be4cd42c25f341c92-Paper-Conference.pdf) |
| UGS-Loc | [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Kong_Rethinking_Pose_Refinement_in_3D_Gaussian_Splatting_under_Pose_Prior_CVPR_2026_paper.pdf) ｜ [arXiv](https://arxiv.org/pdf/2603.16538) |
| SaLon3R | [Springer PDF](https://link.springer.com/content/pdf/10.1007/s11263-026-02965-2.pdf) ｜ [arXiv](https://arxiv.org/pdf/2510.15072) |
| AnchorSplat | [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_AnchorSplat_Feed-Forward_3D_Gaussian_Splatting_With_3D_Geometric_Priors_CVPR_2026_paper.pdf) ｜ [arXiv](https://arxiv.org/pdf/2604.07053) |
| TokenSplat | [arXiv PDF](https://arxiv.org/pdf/2603.00697) |
| Off The Grid | [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Moreau_Off_The_Grid_Detection_of_Primitives_for_Feed-Forward_3D_Gaussian_CVPR_2026_paper.pdf) ｜ [arXiv](https://arxiv.org/pdf/2512.15508) |
| C3G | [arXiv PDF](https://arxiv.org/pdf/2512.04021) |
| VCR-GauS | [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/fc9f83d9925e6885e8f1ae1e17b3c44b-Paper-Conference.pdf) ｜ [arXiv](https://arxiv.org/pdf/2406.05774) |
| PUP 3D-GS | [CVF PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Hanson_PUP_3D-GS_Principled_Uncertainty_Pruning_for_3D_Gaussian_Splatting_CVPR_2025_paper.pdf) ｜ [arXiv](https://arxiv.org/pdf/2406.10219) |
| Uni3R | [arXiv PDF](https://arxiv.org/pdf/2508.03643) |
| UniSplat | [arXiv PDF](https://arxiv.org/pdf/2604.10573) |

注：CVF / NeurIPS / Springer 链接是检索中确认过的正式出版方 PDF；arXiv PDF 采用 `arxiv.org/pdf/<编号>` 标准格式。TokenSplat、C3G、Uni3R、UniSplat 未见直接确认的 CVF PDF 命名，故以 arXiv PDF 为主、CVF 页面为辅。

## 本地完整论文

- [AirSplat: Alignment and Rating for Robust Feed-Forward 3D Gaussian Splatting](https://arxiv.org/abs/2603.25129)
- [SToRe3D: Sparse Token Relevance in ViTs for Efficient Multi-View 3D Object Detection](https://arxiv.org/abs/2605.14110)

## 长视频、位姿与几何

- [LongSplat — ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Lin_LongSplat_Robust_Unposed_3D_Gaussian_Splatting_for_Casual_Long_Videos_ICCV_2025_paper.html)
- [LongSplat — arXiv full text](https://arxiv.org/html/2508.14041)
- [LongSplat official code](https://github.com/NVlabs/LongSplat)
- [Deep Gaussian from Motion — NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1c2b1c8f7d317719a9ce32dd7386ba35-Abstract-Conference.html)
- [ZeroGS / Deep Gaussian from Motion — arXiv full text](https://arxiv.org/html/2411.15779v1)
- [NopeRoomGS — NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/464012c83279e19be4cd42c25f341c92-Abstract-Conference.html)
- [NopeRoomGS paper PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/464012c83279e19be4cd42c25f341c92-Paper-Conference.pdf)
- [UGS-Loc — CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Kong_Rethinking_Pose_Refinement_in_3D_Gaussian_Splatting_under_Pose_Prior_CVPR_2026_paper.html)
- [UGS-Loc full text](https://arxiv.org/html/2603.16538)
- [UGS-Loc official code](https://github.com/kmk97/UGSLoc)

## 前馈紧凑 3DGS

- [SaLon3R — IJCV 2026](https://link.springer.com/article/10.1007/s11263-026-02965-2)
- [SaLon3R PDF](https://link.springer.com/content/pdf/10.1007/s11263-026-02965-2.pdf)
- [AnchorSplat — CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_AnchorSplat_Feed-Forward_3D_Gaussian_Splatting_With_3D_Geometric_Priors_CVPR_2026_paper.html)
- [AnchorSplat full text](https://arxiv.org/html/2604.07053)
- [AnchorSplat official code](https://github.com/Zhang-Xiaoxue/AnchorSplat)
- [TokenSplat — CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_TokenSplat_Token-aligned_3D_Gaussian_Splatting_for_Feed-forward_Pose-free_Reconstruction_CVPR_2026_paper.html)
- [TokenSplat full text](https://arxiv.org/html/2603.00697v1)
- [Off The Grid — CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Moreau_Off_The_Grid_Detection_of_Primitives_for_Feed-Forward_3D_Gaussian_CVPR_2026_paper.html)
- [Off The Grid full text](https://arxiv.org/html/2512.15508)
- [C3G — CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/An_Learning_Compact_3D_Representations_from_Feed-Forward_Novel_View_Synthesis_CVPR_2026_paper.html)
- [C3G full text](https://arxiv.org/html/2512.04021v2)
- [C3G official code](https://github.com/cvlab-kaist/C3G)

## 几何一致性与剪枝

- [VCR-GauS — NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fc9f83d9925e6885e8f1ae1e17b3c44b-Abstract-Conference.html)
- [VCR-GauS official code](https://github.com/HLinChen/VCR-GauS)
- [PUP 3D-GS — CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Hanson_PUP_3D-GS_Principled_Uncertainty_Pruning_for_3D_Gaussian_Splatting_CVPR_2025_paper.html)
- [PUP 3D-GS full text](https://arxiv.org/html/2406.10219v3)
- [PUP 3D-GS official code](https://github.com/j-alex-hanson/gaussian-splatting-pup)

## 语义与空间智能

- [Uni3R — CVPR 2026 Highlight](https://openaccess.thecvf.com/content/CVPR2026/html/Sun_Uni3R_Unified_3D_Reconstruction_and_Semantic_Understanding_via_Generalizable_Gaussian_CVPR_2026_paper.html)
- [Uni3R full text](https://arxiv.org/html/2508.03643)
- [Uni3R official code](https://github.com/HorizonRobotics/Uni3R)
- [UniSplat — CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_Learning_3D_Representations_for_Spatial_Intelligence_from_Unposed_Multi-View_Images_CVPR_2026_paper.html)
- [UniSplat full text](https://arxiv.org/html/2604.10573v1)

---

## 最终结论

AirSplat 不应被理解为“又一个可直接接入的 3DGS trainer”，而应被放进一条更完整的研究脉络：

- LongSplat / DeepGfM / NopeRoomGS 解决无位姿视频的显式几何与局部—全局优化；
- AirSplat 解决前馈高斯的位姿—几何错位和局部跨视不一致；
- VCR-GauS / PUP 分别补充表面几何与不确定性重要度；
- SaLon3R / AnchorSplat / TokenSplat / Off The Grid / C3G 探索紧凑前馈高斯的不同表示；
- Uni3R / UniSplat 把几何、外观和语义推向统一空间表示。

对当前项目最安全、最有信息增益的下一步不是重写架构，而是依次建立：

1. point-to-camera-ray 只读审计；
2. 室内分段平面一致性审计；
3. AirSplat teacher-relative continuous rating；
4. 只有前述信号通过后，才做固定拓扑 ROM opacity-only 派生；
5. 再以 Fisher importance 作为正交证据，而不是替代几何判断。

这条路线保持现有产品闭环、manifest、replay、Test 隔离和任意尺度诚实边界，同时把前沿论文转成可逐步验证的项目改进，而不是一次不可归因的大型集成。
