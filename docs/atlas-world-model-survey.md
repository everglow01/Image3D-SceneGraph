# World Labs Atlas 世界模型调研报告

> 日期：2026-09-02
> 状态：调研完成，未进入实施；Atlas 处于 early access、不开源、仅对选定伙伴开放，当前无 API 可接入
> 范围：Atlas 的功能、技术细节，与本项目（免标定重建）的对齐点、可借鉴/融入判断，以及 world model 研发方向
> 约束：坐标仍是归一化任意单位、不宣称公制精度；不追生成式外推，以免拆掉"重建 = 不撒谎"的护城河

## 1. 结论摘要

Atlas 是 World Labs 在 2026-09-01 发布的"omni world model"（全能世界模型），核心思路是用一个共享的**空间上下文（spatial context）** 把 text / image / video / 3D 四件事统一在一个模型里，并把"相机位姿 + 参考图 + 已生成视图"作为条件信号，驱动一个自回归扩散网络直接吐图像和深度。

对它最关键的判断只有一句：**它是"生成"，我们是"重建"。** 它脑补未见视角，我们忠实还原输入；两条路线在"相机控制下的多视图一致性"上撞了，但价值观相反。

- **能借鉴的**是信号组织层面的三样东西（位姿条件化、image+深度双通道、尺度锚），详见 §5。
- **不能融入的**是 Atlas 模型本身——不开源、无 API，且生成式补洞会 hallucinate，亲手拆掉本项目"不撒谎"的壁垒，详见 §6。
- **继续 world model 研发**的正确姿势是走方向 A（相机轨迹渲染，已有 90%）和方向 C（语义 scene graph + 物理一致性，独有），而不是复刻 Atlas 的生成式外推，详见 §7。

## 2. Atlas 是什么

定位：一个模型同时干 text / image / video / 3D 四件事，靠一个共享的"空间上下文"串起来。目前 early access、只对选定伙伴开放、不开源，官方明确它会成为更早发布的 Marble 的底层模型。

能力清单（交叉多个来源确认）：

- **pixel-perfect camera control** —— 头号卖点：给 1–6 张参考图 + 用户指定的相机轨迹，生成新视角，且新视角与参考图内容对齐。
- **camera-controlled video** —— 最长 1 分钟、1440p 的视频生成，相机路径由人设计。
- **3D 输出** —— 可导出**点云**或**3D Gaussian splat**。
- **单图外推 / 少图重建** —— 一两张照片 → 生成未见过的角度。
- **robot-view** —— 模拟机器人在任意位姿会看到的画面。

## 3. 技术细节（实的 vs 营销）

有出处的实情：

1. **架构 = autoregressive diffusion model**（Ben Mildenhall 原话，NeRF 作者本人现 World Labs）。即自回归 + 扩散的混合：按因果顺序逐段生成（类似 LLM 出 token），每段内部用 diffusion 去噪。这把它与纯自回归视频模型（Genie 2 一类）和纯扩散视频模型都区分开。
2. **原生 2D + 3D 双通道** —— 模型直接同时处理 2D 图像帧与 3D 深度图/点图，比"先出图、再事后提几何"更根本。
3. **核心机制 = spatial context（空间上下文）** —— 生成时把三样东西条件化进网络：①输入参考图、②自己之前生成的视图、③每一张对应的相机位姿。几何一致性不是靠显式 BA，而是被这个空间上下文"学"出来的。

偏营销、不可照单全收的部分：

- "beats specialized 3D models with one omni model" 是自家通稿与二手媒体说辞，无公开第三方 benchmark。
- 具体 token 化方式、训练数据、空间上下文如何编码、参数量均未公开，属私有资产。

一句话概括技术真相：**"相机位姿 + 参考图 + 已出视图"作为条件信号，驱动一个自回归扩散网络直接吐图像与深度，几何一致性靠训练学出来，而不是靠 SfM 显式解出来。**

## 4. 与项目的关系：重建 vs 生成

| 维度 | 本项目 (Image3D-SceneGraph) | Atlas |
|---|---|---|
| 本质 | 免标定**重建** | 生成式 world model |
| 输入 | 真实照片 / 视频 / 全景 | 1–6 张参考图 + 相机轨迹 |
| 几何 | 显式 SfM（COLMAP / VGGT+BA / SIFT） | 隐式，靠训练学出 |
| 外观 | 3DGS 拟合真实输入 | diffusion 生成/外推 |
| 相机 | SfM 恢复（尺度任意单位） | 用户显式给定（天然带尺度） |
| 价值观 | **忠实于输入、不撒谎** | **会脑补（hallucinate）** |

两条路线在"相机控制下的多视图一致性"上是同一类问题，但解决哲学相反：我们**显式解几何再贴外观**，它**隐式学一致性再出像素**。这决定了 Atlas 对本项目的价值是"偷信号组织"，而不是"抄范式"。

## 5. 可借鉴设计（有具体价值）

1. **camera-pose-conditioning（位姿条件化）。**
   本项目的 navigation 资产、collision mesh、3DGS render 已经在做"给位姿出视图"。Atlas 的启示是：把"位姿 + 参考图"作为**显式条件**喂进去，可以在 3DGS 覆盖空洞处（白墙、玻璃、天空、被 temporal split 丢弃的 holdout 段）做外推补洞。这条路只差一步——`gaussian/replay` 已能 freeze rerun，补洞可作为 fail-soft 增强而非推翻现有管线。

2. **image + depth 原生多任务。**
   Atlas 把 2D 图像与 3D 深度当一等公民同时建模，与本项目 VGGT（点图 = 密集深度）同一哲学；`gaussian_geometry_source=vggt_ba` 分支（窗口化 VGGT + 局部 BA）已在方向上用深度当中间表示。

3. **尺度锚（scale anchor）。**
   Atlas 因相机位姿是显式条件而天然带尺度，恰好反衬本项目 codex 里反复强调的"任意单位、不宣称公制精度"痛点——只要拿到**哪怕一个已知距离做锚**，任意单位就能升格。这个锚点思路可直接进 roadmap，但须遵循"不得在 Job 运行时下载权重、不得用 Test 选择算法"的既有约束。

## 6. 不能 / 不应融入的判断

- **Atlas 本身融不进来**：不开源、early access、无公开 API，这不是"能否拿到"的问题，而是"压根没接口可调"。不在其上花工程力气。
- **更应警惕被范式带偏**：本项目壁垒恰恰是"几何自洽 + 忠实输入 + 任意单位也不撒谎"。生成模型在新视角上会 hallucinate（脑补出不存在的家具），而重建的价值就是"不脑补"。若模仿 Atlas 做生成式补洞，会亲手拆掉差异化护城河。因此 §5 里的补洞只能限定为"几何来源的 fail-soft 增强"，不得升格为主路径。

## 7. world model 研发方向（三条路）

一个真正的 world model 需要三样：真实外观、真实几何、对世界的可查询理解。前两样本项目已 90% 具备，缺的是第三样，以及"从部分观察推断 + 时间演进预测"。

**方向 A —— 重建 → 导航 → 相机轨迹渲染（最贴现有资产）**
已具备 3DGS render + navigation + gaussian replay。下一步是"给定相机位姿序列，实时渲染任意轨迹"——本质上是一个小号的 camera-controlled rendering，且 90% 已在代码里。这是性价比最高、最不 speculative 的一步。

**方向 B —— 前馈外推补洞，作为几何的 fail-soft 增强**
用相机位姿 + 稀疏覆盖当条件，补 SfM 覆盖不到的区域。对接已有 VGGT（前馈）+ 3DGS（微调）分工，是 VGGT 博客的自然延伸，不是新起炉灶。

**方向 C —— 语义 scene graph → 物理一致性 → 可查询的世界描述（差异化）**
对应本项目还空着的 `scene_graph`。这才是 Atlas 永远做不到、而本项目能做的"world model"：Atlas 会渲染一把椅子，但不知道"椅子在地板上、地板承重、椅子不会穿墙"。一个不靠像素、靠关系描述、能回答"桌子和墙的相对位置、能否走过去"的世界，才是标准意义的世界模型，也是相对整条生成路线最硬的壁垒。

## 8. 与博客第三篇的关系

Atlas 这条线适合放进博客第三篇（发散思维 / 不涉机密）：

- **"生成 vs 重建"的价值观对立**——可公开讲，且有洞察。
- **尺度锚**——呼应本项目"任意单位"这一贯穿主线，是现成的公开素材。
- 注意机密边界：第三篇仍不得披露实时渲染工程细节、recovery_prune 机制、消融数字、gate 阈值等私有资产。

---

## 来源

- [Atlas: A World Model for Spatial Intelligence — World Labs](https://www.worldlabs.ai/blog/atlas)
- [Ben Mildenhall: autoregressive diffusion model](https://x.com/BenMildenhall/status/2094847509466358124)
- [World Labs Launches Atlas（1440p / 3D worlds / robot views）](https://superpowerdaily.com/posts/world-labs-launches-atlas-for-1440p-camera-controlled-video-3d-worlds-and-robot-views)
- [Atlas Beats Specialized 3D Models With One Omni Model — AlphaSignal](https://alphasignal.ai/news/world-labs-atlas-beats-specialized-3d-models-with-one-omni-model)
- [Marble: A Multimodal World Model — World Labs](https://www.worldlabs.ai/blog/marble-world-model)
- [Streaming 3DGS worlds on the web（Spark 2.0）](https://www.worldlabs.ai/blog/spark-2.0)
- [A Functional Taxonomy of World Models — World Labs](https://www.worldlabs.ai/blog/taxonomy-of-world-models)