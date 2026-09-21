# 视频重建质量优化执行计划

> 状态（2026-09-21）：已批准计划已保存，实施暂缓。用户将“训练后手动修剪 3DGS”与“服务器端交互渲染”列为更高优先级；两项基本完成后再恢复本计划。Q0/Q1 代码尚未编写，未启动远端诊断、渲染或训练。以下保留批准时的完整计划内容；恢复前须重新核对代码、输入与资源状态。

## Context：问题、依据与交付目标

当前新拍视频已完成普通 COLMAP 几何、Project v7/MCMC 双臂主训练、SOR、2k Train-only、Validation 和浏览器导出。用户对椅背、椅脚、窗帘、地毯以及局部雾化的实际观感仍不满意。本阶段目标不是再接一个训练器或提高一项均值，而是在现有 COLMAP＋gsplat 架构中，取得固定问题区域上可辨认、可复核、资源可接受的改善。

依据：
- `docs/920论文计划.md`：一次有界归因，只选择一个结构或细节候选；先质量，后清理和压缩。
- `docs/3dgs-research-architecture-reference.md`：论文适用边界、单变量实验、数据隔离和 RS 研究候选。
- `codex.md` §13、§15、§17，以及现有 trainer/checkpoint/manifest 合同。
- 上一轮论文审读和四个既有 Validation 视角的观察，不重新包装成根因已证实。

文档状态：首次读取 `920论文计划.md` 时为空，复查已读到内容；当前正文有明显截断，包括视角号、论文数字和部分结论。采用可确认的主张，不从残句补造数值。实施文档落地时保留原文件，不覆盖或自动“修复”它；完整计划另存 `docs/video-quality-improvement-plan.md`。本计划阶段只写当前计划文件。

**推荐路线：冻结覆盖样本、困难样本和新视角路径 → 一次有界归因 → 一个候选 A/B → 静态与 NVS 验收 → 停止并决策。** 几何监督为有证据时的优先方向；只有归因更支持细节增密时才走替代分支，两者不同时实现或起跑。

用户对首稿的修订要求：不能只看四张已有图片；必须增加参考样本，主动验证基线效果差的视角，并检验不与已有输入相机重合的 NVS 新视角与连续运动。本版据此扩大样本和 NVS 验收，但仍保持一次归因、一个训练候选的范围。

本计划不是远端执行授权：当前不训练、不渲染完整模型、不下载权重、不同步代码、不发布新模型、不消费 Test，也不修改产品默认。后续批准实施代码与批准远端运行须分别明确，执行许可须绑定主机、输出目录、阶段和预算。

## 1. 已知基线与尚未知事项

### 1.1 冻结输入身份

远端项目：`i-94B8D131:/usr/local/3dgs_new/Image3D-SceneGraph`。

既有实验根：
`outputs/experiments/num4-retake-1080-seed3500-train-only-v1`。

- 视频：`VID20260917094108.mp4`，约 848.447 秒，原生 1080p；去畸变图像为 1078×1920。
- 几何：普通 COLMAP，3780/3955 注册，约 95.58%，时序覆盖 1.0，pose health 通过；4 个 >2 秒缺口仍为软警告。
- Train/Validation/Test：3016/377/387；本次 Test 未用于 Gaussian 训练或评价。
- protocol SHA256：`c3061cb25911e2faa96e87783ad2b3569a9d5c4a3084830889f05ae6a5e4af69`。
- dataset hash：`ca7c8624836ddb54f51bb8ba02d85b5fa4d94d5126355fe82816d5fae1c2ea36`。
- 训练代码：`0cbf0a06016550a8c6062787bf3ee304dd1b5472`；展示支持后续为 `51a6a9d`，不能混称同一版本。

最终 held-out Validation 的历史核验结果：

| 模型 | Gaussian 数 | raw PSNR | raw SSIM |
|---|---:|---:|---:|
| Project v7，30k＋SOR＋2k Train-only | 1,483,682 | 22.167029 | 0.807675 |
| MCMC，同预算流程 | 2,999,158 | 22.546477 | 0.823115 |

MCMC PSNR 在 257/377 视角改善、120 视角退化；这不是等 Gaussian 数量对照，也不能单独归因于 recovery-prune、容量或初始化。

### 1.2 当前可成立的判断

- 2907 窗边椅子：MCMC 比 Project 明显减轻雾化，说明方法层面有改进空间。
- 3670 低机位椅脚/地毯、910 桌面/窗帘：两个模型仍有共同缺陷。
- 1728 墙角相对正常，适合作为保护区域。
- 原生预览已有缺陷，不能全部解释为浏览器问题；但新模型完整浏览器渲染与性能尚未验收。
- 这些四视角是已查看过的开发样本，不是独立确认集。

### 1.3 不应提前下结论

注册率和 pose health 通过不等于局部相机已精确到像素。稀疏重投影小也不等于弱纹理区域的稠密表面正确。60000/3016≈19.9 次采样、增密前约 10 次采样只能说明预算，不证明欠拟合；末期指标平台也不能证明早期拓扑充分。手机滚动快门、防抖、曝光变化仅是待证实因素。

既有 PSNR/SSIM 评价属于共享 SfM 几何下的 held-out RGB 监督评价，不是完全未见几何、跨场景泛化或公制精度。新增任意相机 NVS 是另一类无参考观感/稳定性证据，不能与这些有参考指标混称。

## 2. 将论文转成工程取舍

| 论文方向 | 本阶段采用什么 | 不做什么 |
|---|---|---|
| LongSplat / DeepGfM | 先辨别局部几何误差，保留独立几何约束的思想 | 不替换已成功的 COLMAP，不把可靠 COLMAP 上小幅 pose 优化收益夸大 |
| NopeRoomGS | 大平面与边界保护；未来若修相机，采用受约束交替优化 | 不假装已知内参，不借目标 RGB 对齐 Validation 相机来提高指标 |
| VCR-GauS | 置信度、尺度对齐、跨视一致性；先一个深度项 | 不同时增加法线/平面/新分裂规则；普通 expected depth 不冒充原文交点深度 |
| AbsGS（补充候选） | 仅在细节归因成立时研究绝对梯度统计 | 不把开一个 bool 称为完整复现，不顺带改预算和剪枝 |
| AirSplat | 后续先验证 teacher-relative rating | 当前自由高斯没有逐像素教师对应，不直接抄 ROM；不先靠压透明掩盖结构问题 |
| PUP | 质量稳定后再做重要性/压缩资源审计 | 不把 Fisher 当正确性，不直接剪 90% |
| 前馈、SToRe3D、语义论文 | 保留紧凑预算与可恢复选择等参考 | 不接跨场景训练系统、不做新语义/导航主线 |

原文档 RS-1 专指 VGGT-BA 窗口与 tracks；本次普通 COLMAP 没有相同的窗口证据，因此 RS-1 不作为强制第一项。RS-2 仅在大平面问题被证实时借用；RS-3/4/5 后移。它们不是必须全部做完的串行任务。

## 3. 阶段 Q0：冻结质量目标、困难样本与 NVS 相机

**预计开发/整理量：0.5–1 个工作日；此阶段先读已有评价/图像/相机，不做新 GPU 运算。**

### 3.1 核验并绑定来源

1. 只读核验既有 protocol、replay、两臂 complete、SOR、Selection、Train-only、export 的哈希和 lineage；路径从记录解析，不靠猜测目录。
2. 建立 `quality-protocol.json`，冻结输入身份、代码、模型、样本、ROI、相机路径、尺度、预算、通过条件和禁止项。
3. 377 个 Validation 的已有 per-view 指标全部纳入排序和总体统计，不先看几张截图再猜差视角。

### 3.2 冻结 80 个 Validation 视角，既覆盖场景也主动找差例

按下列顺序选取，跨层去重；每个排序以数值化 image ID 作稳定 tie-break，重复时在该层排序中顺延，保证最终 80 个不同视角：

| 样本层 | 数量 | 固定选择规则 |
|---|---:|---|
| 已知锚点 | 4 | 2907、3670、910、1728，保留历史问题与正常对照 |
| 全视频覆盖 | 24 | 按时间分成 24 个等时段，选各段靠近中点的未选 Validation；空段按邻近未选候选补齐并记录 |
| 基线困难集 | 32 | Project raw PSNR 最差 8、Project raw SSIM 最差 8、MCMC raw PSNR 最差 8、MCMC raw SSIM 最差 8；每组跳过已选 ID 顺延 |
| 方法分歧集 | 12 | MCMC−Project PSNR 差最大的正向 6 和负向 6，检查一方改善、另一方失效的区域 |
| 固定随机检查 | 8 | 从剩余 Validation 用单独冻结的诊断 seed 采样；不消耗或改变训练 RNG |

- 排序来自**本轮候选实现前的历史基线**，不是新候选跑完后挑赢的图。保存样本层、原始排名、选入原因和来源评价 hash。
- 80 视角是有意包含困难例的开发诊断集，不把其平均数冒充总体质量；总体结论仍用全部 377 视角。
- 32 个困难视角不是同一个对象的 32 个相邻截图。全部保留用于尾部统计；另按时间/共同对象聚为至少 6 个不同的困难区域组，用于几何归因。若确实没有 6 个不同组，如实记录，不人为制造独立案例。
- 增加至少 4 个不同位置的正常保护区域，其中包括 1728；优先在 80 视角中的覆盖层和随机层选择，按原始参考及两个基线共同表现冻结，候选后不能换样本。若找不到足够正常区域，记录基线本身缺少保护参照，不将相对不差的区域伪标为正常；须在候选起跑前明确实际组数及相应验收口径。
- 为困难/正常组标出原生图上的归一化 ROI、对象目标和排除原因；80 张同时提供整图与局部原生裁剪，缩略图只用于索引。

### 3.3 冻结最多 64 个 Train 视角，用于区分欠拟合与缺覆盖

- 先给上述至少 6 个困难组和 4 个保护组各找最多 4 个实际观察同一对象的 Train，去重后最多 40 个。
- 再补全视频时间/相机方向覆盖，合计最多 64 个；不是只取 Validation 前后相邻帧。
- 候选依靠共享 tracks、相机方向和时间生成，再确认确实看见同一实物。没有足够有效视角时记录欠覆盖，不硬凑 64。
- 多视角 ROI 对应同一对象，不把相同像素坐标当同一物体。Train 图和渲染只用于拟合诊断，不能称为 held-out NVS。

### 3.4 单独冻结真正的新相机，而非只重放输入视角

生成 `nvs-cameras.json`，保持内参/分辨率一致，记录 camera pose、视线、路径类型、来源锚点和到最近输入相机的位置/角度差。

固定 6 条路径，共 352 个新 pose：
- **4 条有较好观测支持的插值/侧移路径，每条 64 pose，共 256**：根据 80 视角诊断清单覆盖 3 个不同困难区域和 1 个正常保护区；在确有新增困难区域时，至少一条覆盖本轮新发现的差区域，不能全部锁在旧四视角附近。椅子、低机位细结构、桌面/窗帘是候选，不是必须固定的全部对象。路径覆盖平移与转头，不只原地旋转。
- **2 条轻度外推压力路径，每条 48 pose，共 96**：在局部观察范围边缘侧移或改变俯仰，检查遮挡后露出的表面；不把穿墙或完全未拍区域的空白当常规失败。

路径仅依据冻结相机、原始观测与只读几何生成；位置尺度用局部相机基线描述，不使用“米”。旋转用一致的插值约定，保存每帧显式矩阵，不依赖浏览器帧率积分。

所有正式新 pose 都不能与任何输入相机矩阵完全重合；路径端点若需要既有相机做校验，额外标为 anchor，不计入 352 个新视角。用于代表性验收的 pose 至少满足相对最近输入相机平移 ≥局部中位基线的 5%，或视线夹角 ≥2°，避免用几乎重复的相机冒充新视角。对退化局部基线记录并换用明确的角度条件，不臆造平移尺度。

相机空间内插并不保证物体被充分观察，因此路径还记录近邻 Train 数、视线支持以及只读几何显示的可能碰撞/遮挡。不能确认安全/观测支持的片段标为压力测试，不混进常规 NVS 通过率。

### 3.5 输出与通过条件

- `quality-protocol.json`：80 Validation、最多 64 Train、区域分组、ROI、指标口径和选择原因。
- `nvs-cameras.json`：352 新 pose、路径类型、支撑说明及哈希。
- `baseline-summary.json`：既有全 Validation 配对统计、模型数、资源记录。
- 所有有参考图样本均为 Train/Validation；Test 图像不得读取。新相机生成不使用 Test RGB，保持原数据划分。
- 任一哈希漂移、源模型缺失、Test ID 混入或角色不符立即停止，不创建替代身份。
- 旧模型只读，不重建几何、不重划分数据、不要求重新拍完整视频。

## 4. 阶段 Q1：一次有界、扩大样本的质量归因

**预计开发量：1–2 个工作日。远端授权后最多 4 小时墙钟。** 每个历史模型最多 144 个有参考静态视角（80 Validation＋64 Train）和 352 个无参考新 pose；两个模型顺序加载，最多 992 次 pose 渲染。所需 RGB/alpha/depth 尽量单次调用取得，不额外重复逐通道渲染。先复用同分辨率且 hash 匹配的已有 PNG，只补缺失证据。

### 4.1 复用与最小新增

新增一个独立研究入口 `scripts/analyze_gaussian_quality_regions.py`，支持先冻结样本、再生成受限诊断、最后只读汇总；不搭建通用实验平台。

直接复用：
- `gaussian.runtime` 的 `load_training_views`、`load_evaluation_views` 和相机变换；在读取像素前验证允许的 split/ID。
- `gaussian.render.render_gaussians` 的 RGB/RGB+ED。
- `gaussian.training_math` 的 PSNR、SSIM，`scripts/audit_gaussian_render_consistency.py` 的图像差分定义。
- 已有 COLMAP parser、pose-health 和 view-graph 诊断。
- `scripts/analyze_gaussian_floaters.py` 的分布口径，仅在已有点云与坐标变换可绑定时使用。

注意：现有 render-consistency CLI 固定 Validation、最长边 1280，不能原样用来声称完成本次 1920 的 Train/Validation 诊断；复用底层函数而非静默改变旧 audit profile。Floater census 中的相机距离/稀疏点距离只是代理，不是自由空间真值，旧场景阈值不可直接迁移。

### 4.2 每个区域只查五类证据

1. **图像有效性**：原生 ROI 清晰度、饱和/过曝比例、明显动态与反光；同物体跨视外观是否冲突。不靠设备类型推断滚动快门。
2. **Train/Validation 拟合**：参考、Project、MCMC、绝对误差图；raw/display 指标分别报告。只在诊断中比较选定 Train 子集，禁止称其为完整 Train 均值。
3. **实际观测支持**：覆盖该对象的 Train 数、观察方向差、tracks 支持、重投影 median/p90、边缘是否一致错位。读取旧几何是共享 SfM 证据；不得把其中利用过其他帧的点称为独立 Train-only 重建。
4. **训练时序**：解析已有 `progress.jsonl` 的 `batch_view_ids`、增密、reset、recovery-prune 和 Validation；统计实际采样次数及增密前后覆盖。只读历史日志；不为补日志重跑训练。不可从净增长量推断没有记录的 split/clone 分量。
5. **几何/遮挡代理**：在需要时渲染 alpha/expected depth；看局部雾层、跨视不稳定和大投影高斯。Gaussian expected depth 不作真值，不能凭它单独裁定位姿或教师正确。

### 4.3 输出唯一下一步，禁止无界审计

交付 `diagnosis.json` 和一份简短归因说明，给出主要瓶颈、两项相互补充的证据、反证、不确定项、允许进入的候选。

| 观察 | 决策 |
|---|---|
| 图像清楚、Train 也糊、支持足够，几何边缘无一致错位，细节生长不足信号成立 | 进入 Q2-D 细节候选 |
| 相机没有明显系统错位，但弱纹理/表面/前景遮挡不稳定 | 优先进入 Q2-G 的教师证据准备；只有其深度质量/覆盖 gate 通过才允许训练，不要求 Q1 已经生成尚不存在的教师深度 |
| Train 清楚而 Validation 弱，主要缺有效 Train 观察 | 停止算法 A/B，交付具体欠覆盖区域与定点补采建议；不自动改 split/关键帧/重拍整场 |
| 多视一致边缘错位，或曝光/动态冲突主导 | 停止本计划训练分支，另立相机/成像或外观单变量合同；不偷偷给当前 A/B 增加 pose/exposure 参数 |
| 在有界样本内仍无法区分 | 结论写“证据不足”，结束这一轮；不自动扩大到更多论文和诊断 |

只有一种分支进入实现。若两类都有信号，先选对困难区域组及新视角缺陷解释力更强的一项；证据相近时优先几何监督，但其教师质量门槛仍必须通过。单个好看例子不足以支持候选。

Q1 同时检查 6 条基线 NVS 路径，每条固定取 8 个关键帧，共 48 个用于原生尺寸细看，并保留整段序列。另允许从完整基线序列中预先标出最多 12 个最差新视角，写明雾层、重影、结构消失或视角跳变等原因，形成冻结 NVS 困难集；必须在候选运行前完成，之后不增删以迁就结果。最终归因须至少引用一个有参考视角组和一个新视角片段；若两者结论不同，分别陈述，不能用静态指标覆盖 NVS 失败。

## 5. 阶段 Q2-G：高置信度稀疏深度监督（推荐的条件分支）

**仅在 Q1 支持几何问题时执行。预计开发量 2–4 个工作日；不同时实现 Q2-D。**

### 5.1 先建立可复用的 Train 深度证据

- 首选项目已安装且已审计的 VGGT；核对代码、权重哈希与许可证。缺失则阻塞，不自动换教师或下载。
- 从 Train 确定最多 256 个覆盖全场景的视角，优先复用 `select_train_images` 的空间选择思想；冻结列表，不依据候选 Validation 调整。
- 先处理其中冻结的 64 个试运行视角；信号/资源通过才补齐到最多 256 个，不做多档网格搜索。
- 固定小窗口 4 帧、重叠 2 帧，使用明确记录的已有预处理和精度；不在 OOM 后自动调小/改精度重试。此预算是新研究 profile，不修改已有 filter 的 8/4 默认。
- 复用 `run_colmap_vggt_dense.py` 中 `run_vggt_depth_batches`、`estimate_depth_scale`、`build_vggt_image_transform`。仅整理真正需要复用的证据构建代码，不顺便重构整个 VGGT runner。
- 新增 `scripts/prepare_gaussian_depth_evidence.py`，只保存深度/confidence/有效 mask/图像变换和来源 sidecar，不调用现有 filter CLI 生成剪枝模型。

必须正确处理：
1. 只把 Train RGB 送入教师；尺度拟合限制在 Train 观测。既有 SfM 点本身可能由共享几何求得，这一局限显式保留。
2. 教师深度先对齐 COLMAP 任意单位，再转成 renderer 使用的 normalized camera-Z。现有 `DepthEvidence.camera_from_normalized` 为配合原始深度保留 similarity scale，不能直接拿来和 renderer 的 rigid transform 深度相减。
3. 所有 resize、padding、去畸变、K 缩放必须通过合成投影测试；不得把教师预处理像素索引直接当原生训练索引。
4. 复用置信阈值原则，记录其为相对分数而非校准概率；做跨 Train 视图投影及遮挡判别。落在另一表面后的点视为遮挡，不直接当矛盾；无共同可见支持不强行评分。
5. 高反光、玻璃、网面、轮廓边界和不可信深度跳变不施加强监督。第一版允许冻结小量人工排除 ROI，但要记录 provenance，不能看新结果后追加。

以下是**拟议工程门槛，不是论文保证或已测数据**，须在教师运行前冻结：
- 每个接受视角至少 20 个有效尺度观测，scale log-MAD ≤0.15；
- 可信共同可见像素的跨视 absolute log-depth residual ≤0.10，并通过遮挡判别；
- 最终至少 64 个可用 Train 视角；被选定要修复的每个区域至少 2 个不同观察方向的可信视角，各自 ROI 中有效监督像素比例 ≥10%；
- 报告全部拒绝原因和未覆盖区域；门槛不通过则该深度候选不启动。不能为凑数放宽门槛或将 missing target 设成零深度。

`depth-evidence.json` 绑定 dataset、Train ID、教师版本、权重、scale、mask、尺寸、坐标、每个 NPZ 哈希；先冻结，再训练。缓存有限，不把 256 张深度全搬上 GPU。

### 5.2 第一版只增加一个损失

在现有 L1＋SSIM 上增加置信度加权 log-depth L1：

`L = L_rgb + lambda(t) * sum(w * |log(D_render) - log(D_teacher)|) / sum(w)`。

- 使用现有 `RGB+ED` expected depth；明确这是借鉴几何监督原则，不是 VCR-GauS 完整复现。
- `w` 来源于冻结教师有效性、置信度、跨视一致性和边界 mask；teacher、scale、mask 均无梯度。
- 仅对正有限深度和足够覆盖的像素计算；初版 alpha 有效阈值拟定为 0.95，mask detached，同时单独记录 alpha 和有效像素下降，防止通过降低可见性逃避监督。
- 无可用像素的单帧该项为零且记录；整场有效监督不足按前述 gate 失败，不能把它报告为深度实验成功。
- 首轮 `lambda_max=0.01`，前 1000 updates 为零，1000–3000 线性增加，此后固定到主训练结束。以上是待冻结的首个工程起点，不从论文直接照搬、不承诺最优、不跑多权重搜索。
- 教师分辨率只限制辅助项：通过确定的投影/采样变换计算深度损失；RGB 主损失和评价仍为原生最长边 1920，不降低正式画质分辨率。
- 仍从全部 3016 Train 采样，仅其中有证据的最多 256 视角附加深度项。不得为增加监督频率重采样；因此这是“稀疏 Train 深度监督”，不是全视角监督。
- 不增加 normal、plane、opacity ceiling、pose update、颜色校准、补洞或新的初始化。

### 5.3 代码与合同

主要改动限定于：
- `gaussian/config.py`：为选中的分支增加显式内部研究 profile/schema-11，深度项参数及 evidence hash（或绝对梯度开关）纳入有效配置；旧 public profile 仍产生原 schema-10 内容。校验器按版本检查准确字段集合，`resolved_config_record` 按实际 effective schema 记版本，不仅修改全局常量后使旧配置失效。保持旧 schema-10 文件和哈希可验证，不原地迁移历史文件，不把新项塞进 public 默认。
- `gaussian/training_math.py`：一个可 CPU 测试的 masked log-depth loss。
- `gaussian/trainer.py`：主训练获取对应证据、请求 RGB+ED、组合 loss、记录分项/有效像素；两卡 loss 缩放和同步保持一致。
- 必要时新增一个窄职责 `gaussian/depth_supervision.py`，只负责 sidecar 校验、坐标采样和有界缓存，不创建通用监督插件系统。
- `scripts/run_gaussian_training.py`：在 CUDA 前验证研究 profile、证据绑定和 Train-only 身份。
- 相关 config/trainer/render/checkpoint 合同与测试。

旧 `train_only_control_v1` 的 2k 固定拓扑流程保持原数值含义：两臂都只做既有 RGB 补拟合，不把深度项带入此阶段。新研究信息记录为主训练配置与 lineage；源模型配置和评价必须一致，不伪造旧 hash。报告主训练/SOR 后与 2k 后两组效果，若深度收益被补拟合抹去，要明确失败，不能临时取消补拟合救结果。

## 6. 阶段 Q2-D：针对性细节增密（与 Q2-G 互斥）

**仅在 Q1 更支持细节问题时执行。预计开发量 2–4 个工作日。**

选定第一个候选为 **gsplat 原生绝对梯度增密统计对照**，而非直接延长训练或提高 cap。

1. 固定安装的 gsplat 版本与源码，核查 `DefaultStrategy`、rasterization 的 absgrad、packed/distributed 限制以及阈值单位。本轮已核对项目调用点，但未核验远端实际安装的 gsplat 源码；双卡兼容性是前置阻塞检查，不是已具备的能力。
2. CPU 合成测试证明正负像素梯度抵消时绝对梯度统计可区分；远端双卡 smoke 验证梯度 metadata 存在、各 rank 更新有效、没有重复计数或漏算。
3. 候选只有 `densification.gradient_statistic=absolute` 这一行为差异；A 为 signed。先保留相同数值阈值 0.0002、15k 停止时点及全部其他规则。这是同阈值的统计切换试验，不声称等触发率或等 Gaussian 数。
4. 同时接通 rasterizer 的 `gradient_statistics` 和 strategy 的 absgrad；只改其中一处不能构成有效实验。
5. 原生 gsplat 若对 clone/split 都使用该统计，报告为“gsplat absgrad 候选”，不冒充论文中 split 使用 AbsGrad、clone 保留原规则的完整 AbsGS。
6. 若当前 gsplat 不支持所需双卡路径，停止此候选并报告阻塞；不自动升级 gsplat、退回单卡、加入自定义 CUDA 或改变比较预算。
7. 不在第一轮联动改阈值。增密爆炸或无效也是这一个候选的结果；后续阈值校准须另立有限实验，不重跑到成功为止。

主要改动：`config.py`、`trainer.py:_build_strategy/_render_visible_training_view`、render 参数传递和对应测试；旧 public 默认仍 `absgrad=False`。

特别约束：当前 `densification.end_iteration` 还控制 opacity reset、recovery-prune 和部分 screen pruning。第一轮不改它；未来如要单测延长生长，必须先定义独立时序并证明不混改剪枝，不能仅因 config diff 是一个 leaf 就宣称算法只有一个机制变化。

## 7. 阶段 Q3：一轮正式单变量 A/B

### 7.1 对照对象与固定项

首次只在 **Project v7** 上做 A/B：它是产品默认、已有清楚问题区域且更易归因。既有 MCMC 作为只读质量参照，不作为新候选的因果 control；不同时扩成 Project/MCMC × 两算法四臂。

- A：新代码下研究 profile 的关闭态；B：同代码同 profile，只打开 Q2 选定的唯一算法因素。
- 深度分支 A/B 都验证同一 evidence manifest，并走相同 RGB+ED 路径；A 的深度权重为零。先验证此路径 RGB 与旧 RGB-only 一致到预声明容差，否则不开始正式试验。
- 同一个 frozen replay、初始化、相机、3016/377 划分、seed 20260729、1920 分辨率、两卡/相同 rank 数、相同相机序列。
- 每臂 30000 updates / 60000 camera samples；相同 Validation 日程、Selection、SOR(k=30/std=2/opacity band<0.05)、2000 updates / 4000 samples RGB-only Train-only、最终全 Validation、导出。
- 保留现有 checkpoint 选择规则，同时报告 30k 最终诊断；不能挑中间某步替换交付候选后仍称相同流程。
- 新 A 必须重跑以控制代码/环境差异；历史 Project 和 MCMC不充当唯一控制。若新 A 相对历史 Project 明显漂移，先解释差异，不用 B 的收益掩盖控制不一致。

### 7.2 不复用旧 runner 的运行身份

`scripts/run_video_4k_comparison.py:load_protocol` 强制旧 code SHA，预算和双 trainer 也写死。不能取消其校验、修改旧 protocol，或把新算法塞入旧实验根。

新增薄研究 runner：`scripts/run_gaussian_quality_ab.py`，只负责编写/验证新 protocol、串行调用现有 train/SOR/evaluate/final-fit/export CLI 和生成配对报告。复用现有 replay/config/provenance/不可覆盖机制，不复制一套 trainer 或几何 preparation。

新 protocol 分别记录：
- `source_protocol_sha256`、旧几何/初始化生产代码；
- 本次 `execution_code_sha`、环境与依赖指纹；
- 关闭态/候选态完整配置及算法差异；
- evidence hash（若有）、训练预算、seed、两卡身份、质量和资源 gate。

保持初始模型、几何与源文件不可变；新 Gaussian、PNG、模型、日志均写独立目录。

### 7.3 资源与失败边界

以下为拟议上限，授权前写死到 protocol：
- Q1：≤4 小时墙钟，每个历史模型最多 144 个有参考静态视角＋352 个新 pose，顺序加载两个模型；复用已存在且身份一致的渲染，不以增加样本为由重跑训练。
- Q4：正式 A/B 完成后的补充 Train/NVS 渲染与比较 ≤4 小时墙钟；每臂最多 64 个诊断 Train＋352 个新 pose。377 Validation 优先复用本轮完整评价及预览，不重复生成。
- Q2-G 教师：试运行及补齐合计 ≤2 小时墙钟、最多 256 Train 视角；单独阶段，不与 Gaussian 训练并跑。
- 数值 smoke：每臂最多 100 updates，≤30 分钟墙钟；不据此选择质量赢家。
- Q3：两臂串行，每臂完整链 ≤6 小时墙钟；一个正式 A/B 对，不自动增加 seed、候选或重试。
- Project count >3M 时同步报错停止，作为实验资源安全线，不做自动删高斯来维持容量。
- CUDA reserved peak <设备总显存的 90%，host available ≥8 GiB；启动时磁盘 free ≥max(40 GiB, 预计新增资产×1.25＋8 GiB)，主训练/导出运行中低于 8 GiB 停止。
- 不为降低资源自动降分辨率、换单卡、跳 Validation、清旧模型或改变 profile。
- 预估阶段输出量必须含 checkpoints、Adam/strategy 状态、预览和两个 export，不只计算 PLY；仍沿用已有 checkpoint 保留机制。

运行方式维持 dashboard 主任务＋只读 monitor、`flock`、`once`、独立退出码。任一失败先查任务与输出状态，保留失败根，不盲重试。原始模型、前后端和已有展示任务都不停止。

## 8. 阶段 Q4：验收规则——画质先于指标均值

以下为本轮**建议冻结门槛**，不是已验证的收益承诺。Q0 冻结后，不因候选结果不理想再改 ROI、指标或阈值。

### 8.1 先验收完整性

- 377/377 Validation 成功；failed_views 必须为空，不丢弃差视角。
- 无 Test RGB 加载/优化/相机对齐；Teacher 只用 Train。
- 输入、源模型哈希不变；dataset/config/code/evaluation/export lineage 一致。
- 未发生 silent fallback、OOM 后偷偷降配、NaN 或未声明训练阶段。

### 8.2 扩展静态样本、困难尾部与整体质量

1. 冻结的困难区域组中至少一半（向上取整，且不少于 3 组）达到：多视 ROI 的 display MAE 中位数下降 ≥10%，并且可辨认结构没有因模糊化/抹平而消失。各组等权，不能让同一房间角落的大量相邻图淹没其他区域。
2. 任一困难区域组的同口径 MAE 退化不得超过 5%；至少 4 个保护区域逐组要求 PSNR 退化 ≤0.1 dB、SSIM 退化 ≤0.002，且无新增结构缺失。
3. 单独报告冻结的 32 个基线困难视角：paired PSNR median ≥0、至少 20/32 视角 PSNR 改善；此外对两臂完整 377 视角分别排序，计算各自最差 38 个视角（10% 向上取整）的质量均值，PSNR 和 SSIM 各自按自身排序，B 的两个尾部均值均不低于 A。前者追踪旧问题，后者防止把缺陷转移到别处。
4. 全 377 视角 raw mean PSNR 退化 ≤0.1 dB、raw mean SSIM 退化 ≤0.002；paired PSNR median ≥0。raw/display 两组都完整报告。
5. 对 80 视角按样本层分别报告，保留完整整图/裁剪/误差图；不得把刻意过采样差例的 80 视角平均数称为全场景平均数。Train 64 视角单列，不与 Validation 混合。
6. 报告全部 377 个每视角差值、改善/退化数、p10/median/p90、两臂各自最差 10 个视角及原生图，不只报告平均数。
7. 按时间组汇总，避免把相邻帧当独立样本夸大统计显著性；若使用 bootstrap，以时间组而非单帧抽样，置信区间只作描述。
8. 不把 MAE 降低自动等同于结构正确。必须同时按冻结对象清单检查椅脚分离、椅背保留、桌沿和窗帘边界；没有真实改善则不晋级。

### 8.3 NVS 单独验收，不由静态分数代替

**区分两类 NVS 证据：**
- 有实拍参考的 held-out NVS：全部 377 Validation。这些 RGB 没用于 Gaussian 拟合，可计算 PSNR/SSIM，但相机来自共享 SfM，不能称完全未知相机或独立几何。
- 无实拍参考的新相机 NVS：Q0 冻结的 352 个新 pose。位置/方向不复制输入视角，用来检验连续视角中的结构、遮挡与稳定性；没有真值，不计算或编造“新路径 PSNR”。

**执行与展示：**
1. A/B 使用完全相同的 `nvs-cameras.json`、内参、原生分辨率、背景、SH 和 renderer；所有矩阵写入结果，不逐臂调色、opacity、裁剪或曝光。
2. 保存 6 条完整、固定帧率的 A/B 同步视频；帧率只是播放参数，不当 GPU 性能指标。每条 8 个关键帧及最多 12 个冻结困难新视角保存无损 PNG，视频帧可按冻结编码参数压缩以限制磁盘；诊断差异不能仅凭有损视频计算。
3. 逐片段记录前景雾层/遮挡、物体边界和薄结构连续性、孔洞、重复表面、突然出现/消失、亮度闪烁。分别记录基线缺陷、新增缺陷和改善，保留对应帧号。
4. 原生深度/alpha 可辅助解释，但不是几何真值；若计算 depth warp、alpha coverage 或时间变化代理，必须处理遮挡、报告有效比例，且不能把正常视差/反射变化直接叫 flicker。
5. 报告完整序列和所有冻结困难帧，不以漂亮关键帧替代其他片段。

**拟议 NVS 通过条件：**
- 4 条观测支持较好的路径均不能新增严重遮挡、物体缺失/重复或持续跳变。若基线有问题的支持路径数为 m，至少 min(2,m) 条出现清楚可定位的改善；m 在 Q1 冻结。若 m=0，只能验收 NVS 无新增缺陷，不能声称 NVS 质量提升。
- 冻结困难新视角中至少一半（向上取整）的主缺陷减轻，且没有以丢失薄结构/扩大孔洞换取“干净”；采用同尺度并排或匿名 A/B 核验，逐帧留判定说明。该集合若为空，记录“未发现该类基线困难新视角”，不虚造坏例，不以空集合通过宣称改善。
- 2 条轻度外推压力路径单独报告，不要求补出从未观察过的背面。受已知表面支持的部分不可明显变差；未观测区域的失败保留并限制能力声明，不计为可由训练保证修复的缺陷。
- 若静态指标通过但受支持的新视角仍明显失败，整体结论是混合/失败，不晋级。
- 自动指标只能发现异常，用户对完整 NVS A/B 的观看确认是最终主观验收的一部分，不能由代理宣称已达到其预期。

**浏览器交付：**
- 算法比较先用原生 renderer；通过后用同一组固定新相机在浏览器检查对应 PLY，再做自由移动，避免把 renderer 变化掺入训练效果。
- 复用现有查看器，不改变 legacy 默认。已有相机路径格式如不能表达新姿态，先复用 renderer audit 的固定相机驱动，不为研究新建一套观看产品。
- 新派生若无法诚实匹配现有 manifest stage/metric role，先补最小版本化角色再发布，不能冒充旧训练身份。
- 本次研究和新主训练 profile 是否可直接复用现有 Train-only 发布器，要验证其固定实验 profile/双 trainer 校验；不能关闭安全校验来发布 Project A/B。必要时用独立薄发布入口，仍原子新建、contained assets、只读。

### 8.4 资源与结论

除不超过绝对安全线，还要求 B/A 完整链墙钟 ≤1.30、峰值显存 ≤1.25；Gaussian 数、主训练及额外教师耗时分别列出。教师准备不得藏在训练计时之外，分别报告训练比和端到端比。

结论只允许：
- **通过本场景候选门槛**：质量、完整性、运动和资源均通过，可申请第二场景验证；不自动改默认。
- **混合结果**：局部改善但保护区、运动或资源失败，保留研究结果，不发布成默认改进。
- **失败/无效**：没有关心区域改善或数据/运行不完整，停止该候选，不继续堆第二个论文模块。

技术报告可完成自动与人工观察项；“达到用户预期”的最终主观验收必须由用户看固定 A/B 后确认，不能由代理替用户宣布。

## 9. 测试与端到端验证

### 本地仅 CPU/mock/合同测试

至少覆盖：
- Test ID 在解码/模型加载前被拒绝；Val 不进入 teacher 或训练 sampler。
- 80 视角各层选样的稳定排序、去重顺延、空时间段处理、固定随机 seed；候选结果不得参与选样，训练 RNG 不受诊断影响。
- NVS 相机矩阵的刚体性、K/尺寸一致、352 pose 数量、锚点排除、最近输入相机距离/角度、路径 hash；A/B 相机逐帧一致，插值与压力测试身份不可混淆。
- 全 Validation、困难样本层、Train 子集和无真值 NVS 报告分开；无真值路径拒绝生成 ground-truth PSNR 字段。
- 源哈希漂移、证据缺失、错误单位、越界路径、重复目标目录拒绝。
- scale、旋转、K、padding、resize 与 camera-Z 转换的合成投影测试。
- depth loss 的 NaN/zero-depth/空 mask、安全梯度；teacher 和 mask 无梯度。
- 关闭态旧 profile 行为、旧 config hash、Train-only record 语义与 public 默认不变。
- 两 rank 全局 loss、camera_samples、count guard 和证据 ID 的 mock 测试。
- 若走 Q2-D，增加绝对梯度抵消示例及 rasterizer/strategy 同步开关测试。
- 新 runner 的阶段失败不继续、源 root 不可写、只允许约定的一个算法差异；新配置不能恢复旧不兼容 checkpoint。
- 评价/导出保留 Train-only 与新主训练配置真实 lineage，拒绝错角色和伪造 record。

建议回归命令（新增测试名在实现时确定）：

```bash
uv run pytest tests/test_gaussian_config.py tests/test_gaussian_render.py tests/test_gaussian_trainer.py tests/test_gaussian_replay.py tests/test_gaussian_checkpoint.py tests/test_gaussian_final_fit_comparison.py tests/test_video_4k_comparison.py
uv run ruff check <本次改动的Python文件>
git diff --check
```

不在本机运行完整模型训练或 GPU 质量评价；不因测试命令隐式安装新 GPU/模型依赖。只有确实改前端合同才追加相关前端测试和 `npm run build`。

### 远端先 smoke，再正式跑

1. 确认 hostname/pwd、代码 SHA、GPU 和磁盘，确认没有冲突活动任务。
2. 已审计环境中检查依赖 API，禁止 runtime download。
3. 远端授权范围内做 RGB/RGB+ED RGB 一致性、有限梯度、双卡 loss/metadata、源哈希、取消失败保留等 smoke。
4. 零权重路径在合成输入上 RGB 差异拟定 `max_abs≤1e-6`；实际 kernel 环境若无法满足，先解释并冻结合理容差，不能跑完正式 A/B 后补改标准。
5. smoke 只证明可运行；通过后仍按已授权的阶段范围决定是否可以进入正式双臂。
6. 运行完整 Q3，之后用独立只读汇总重新计算均值和配对差值，再验收 Q4。

## 10. 产物、提交与授权边界

### 建议目录（批准执行前须绑定绝对路径）

在远端项目下新建，均不能覆盖既有目录：

```text
outputs/experiments/num4-quality-attribution-v1/
  quality-protocol.json
  nvs-cameras.json
  baseline-summary.json
  diagnosis.json
  regions/
  nvs/                         # 两个历史模型的完整路径、关键帧与困难新视角清单

outputs/experiments/num4-quality-depth-v1/       # 若选 Q2-G
  evidence/
  smoke/
  protocol.json
  control/
  candidate/
  comparison/

outputs/experiments/num4-quality-absgrad-v1/     # 仅在改选 Q2-D 时创建
```

只创建被选中的实验根；每个远端执行根配独立 `-dispatch/`，失败重试另起明确授权的后缀，不删除旧 once/退出记录。比较结论包括输入身份、论文借鉴范围、not-run 指标、失败/反证和下一阶段准入。

### 代码交付分两段

1. **诊断交付**：计划文档、Q0/Q1 最小分析脚本、测试；在 `codex.md` §17 写明确批准的范围。先看诊断结论，不预先写两套训练候选。
2. **一个候选交付**：只有 Q1 通过后，提交选定分支、合同/回归测试和薄 A/B runner；再冻结实际执行命令、SHA 与 protocol。

源码仅通过 Git 同步，不上传源码包。main 作为集成基线；确有提交需要时只使用一个工作分支。不得顺手提交当前已有 `codex.md` 历史未提交段或用户未跟踪文档。commit/push/pull、远端阶段运行、发布新比较分别按明确授权执行。

**完成本计划后的第一项可实施工作仅为 Q0＋Q1 诊断代码与合同，不是启动所有阶段。** 首轮累计开发估计 4–7 个工作日，取决于选中的一个分支；时间为工作量估计而非交付承诺。远端预算独立核算，最多一次正式 A/B。任何分支 gate 失败，停止并提交结果，不自动转投另一分支。

## 11. 后续阶段准入，不列为当前实现任务

- **第二场景确认**：只有 Q4 通过，才用一个另行冻结且 Test 仍隔离的场景复验同参数；不按第二场景结果继续调参后称独立确认。Project 默认是否改变仍需明确决策。
- **AirSplat 启发的 rating/ROM**：结构/细节已有改善且残余遮挡 primitive 有证据时再启动；RS-3 至少两场景可重复区分度通过，才考虑 RS-4。opacity-only 是有孔洞风险的受限实验，不是预定赢家。
- **PUP 压缩**：质量被接受后先小规模资源 profile，再分别看几何 rating 与 Fisher；不直接发明综合分数、不默认可剪 90%。
- **法线/平面、受约束相机细化**：各自要新证据与单变量合同，不能搭便车加入深度实验。
- **LongSplat 式 local/global、前馈 initializer、语义/导航**：另立架构合同，本轮不做。

最终成功标准：交付一项在用户关心区域确有改善、保护区域和短路径不过度退化、代价可解释的候选；若没有，则交付一次有边界、可归因的否定结论，而不是一串无法判断贡献的论文集成。
