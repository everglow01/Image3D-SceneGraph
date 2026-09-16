# Spark 独立渲染审计原型

## 范围

`node scripts/audit_gaussian_browser.mjs --renderer spark` 是研究专用入口，profile 为 `fixed_camera_spark_v1`，输出 `spark.json` 和 `spark-controlled-sh*-<image_id>.png`。省略 `--renderer` 仍运行原来的 GaussianSplats3D 审计。页面接入现提供显式 Spark 切换，旧查看器仍为默认；导出、训练均不变（见下方“页面接入”）。

Spark 精确锁定为运行依赖 `@sparkjsdev/spark@2.2.0`（MIT），由 `frontend/package-lock.json` 固定发行包；页面仅在显式选择 Spark 后按需加载独立代码块。只使用已安装的本地模块和输入 PLY，不上传模型到第三方、不在浏览器中访问 CDN。npm 发行包已包含 WASM/Worker，不要求运行时安装 Rust。远端安装、执行与部署仍需分别授权；源码仅通过 Git 同步。

## 冻结设置

- 直接读取现有 binary-little-endian Gaussian PLY，不转换为 SPZ/SOG，不生成 LoD。
- `SplatMesh.extSplats=true`；源和渲染器均禁用 LoD；中间累积使用 `accumExtSplats=true`。
- **非无损**：位置保留 float32，但颜色、不透明度和 log-scale 为半精度，旋转和 SH 仍有编码量化。
- 禁止自动更新；固定相机后显式 `await spark.update()`，检查排序完成、映射版本和累积数量，再渲染与读回像素。
- Z-depth 排序；`preBlurAmount=0.3`、`blurAmount=0`，对应项目当前 gsplat classic / eps2d=0.3 的约定，不宣称相同光栅算法。
- `minAlpha=1/255`、`maxStdDev=3`、`focalAdjustment=1`；`maxPixelRadius=1e10` 避免默认 512px 半径截断成为未记录的变量。相机 near/far 取冻结审计记录。
- 黑背景、像素比 1、关闭 Three.js tone mapping。记录 output color space 和 Spark 的 `encodeLinear`，不通过手工 gamma 或后期锐化拟合参考图。
- `active_count` 允许少于源高斯数量，因为存在可见性剔除；源与累积数量必须等于冻结的 `model_count`，否则失败。

这些设置是预先声明的候选配置，不是在数据上调出来的优胜配置。不能从当前设置推导像素等价或画质提升。

## 输入和结果约束

真实模型只接受 `audit_gaussian_render_consistency.py` 产生的完整 `frozen_render_consistency_v1`：`status=completed`、`split=validation`、`test_rgb=not_loaded`、`source_unchanged=true`、SH3、有效且唯一的相机。该原型拒绝非零 skew，不静默近似。

在启动 Chrome 前检查 PLY SHA256；运行后再次检查。记录实际读取的审计 JSON 哈希、模型来源哈希、实际模块哈希、Spark/Three.js 版本、相机矩阵、源与实际 SH 阶数、源/累积/可见数量、首个解码高斯、设备信息、加载与单次更新/读回耗时。输入不改写，已有输出目录不覆盖。仅读取 PLY 和冻结相机 JSON，不读取数据集 RGB 或 Test。

首次实现仅提供捕获与完整性检查，**不自动计算对原生/参考图的质量提升、不做连续路径或生产性能验收**。单次 `render_readback_ms` 包含读回开销，不是 FPS；Three.js 纹理计数也不等于显存。全模型峰值内存、真实硬件性能和切换释放测试属于后续获授权验证。

## 验证顺序

以下浏览器命令仅在获授权的验证机器运行。本地开发可先运行不启动浏览器的检查：

```bash
node --check scripts/audit_gaussian_browser.mjs
node --check scripts/gaussian_spark_audit.mjs
node --test tests/test_gaussian_spark_audit.mjs
```

### 1. 六高斯合成自检

```bash
node scripts/audit_gaussian_browser.mjs --renderer spark --self-test \
  --output outputs/analysis/spark-selftest-v1
```

依次捕获 SH0/SH2/SH3。合成 PLY 的低阶系数为零，蓝色通道包含非零三阶系数；检查 CV 投影位置、Y 方向、基础灰度、SH0/SH2 一致性，以及 SH3 必须产生可测像素变化。不能仅凭 `effective_sh=3` 判通过。

必须先确认 `spark.json` 中 `status=completed` 且 `self_test_result.status=passed`，再执行真实模型审计。`--software` 可用于诊断，但结果不得用作目标 GPU 性能证据。驱动脚本保留 20 分钟超时；完整模型执行还应由验证环境配置内存上限，超限应停止，不自动抽稀或降阶。

### 2. 冻结模型捕获

下面路径是项目中已有的本地证据布局；远端执行前需核对实际路径和文件哈希，不能假定远端同名文件存在。

```bash
node scripts/audit_gaussian_browser.mjs --renderer spark \
  --audit outputs/analysis/mcmc-render-audit-20260910/native-roundtrip-audit.json \
  --ply outputs/analysis/mcmc-render-audit-20260910/scene.ply \
  --image-ids 1108,1718,317,425,508,1684 --sh-probe 317 \
  --output outputs/analysis/spark-six-view-v1
```

所有选定视角捕获 SH3，探针视角另捕获 SH0/SH2。探针必须包含在选定视角中；默认优先 317，否则采用选定的首个视角。Spark 仅支持 `controlled` 模式，不冒充已有 `product` 截图。

Spark 退出码：0 为捕获/完整性检查完成（不是质量提升 PASS）；1 为输入、自检、捕获、释放或完整性失败。任一步失败立即停止，不继续尝试较低 SH 阶数；已有截图与失败记录保留。原有 legacy 审计的部分失败退出码 2 保持不变。下一次重试必须使用新输出目录。

## 独立视角与硬件移动验收

用户授权后可用 `freeze_spark_acceptance.py --dataset ... --output protocol.json`，在任何新候选渲染前冻结 12 个 Validation 视角：去掉前六视角，按数值 ID 排序等距选取，不读取质量分数。它们只是相对前六视角独立，仍是同场景开发 Validation，不是新的 Test。

先用既有 `audit_gaussian_render_consistency.py` 在远端生成这 12 个相机的原生/参考 PNG，然后两臂运行 `audit_gaussian_browser.mjs --renderer spark|legacy --hardware --motion --motion-base <首个冻结ID> --modes controlled`，传入该原生 `audit.json`、同一 PLY 和冻结 IDs。`--hardware` 固定 ANGLE Vulkan、禁止软件光栅回退，并要求每次捕获实际身份为 NVIDIA L2；不修改驱动、权限或关闭沙箱。NVIDIA ICD 通过本次进程的 `VK_ICD_FILENAMES` 指定，不修改系统配置。

连续路径在首个冻结相机附近作相机局部横向 ±0.01 arbitrary units 与 ±3° yaw 的正弦闭环。每轮60帧预热、240帧测量，共3轮；路径位移按帧索引固定，不宣称两臂具有相同墙钟移动速度。静态显示配置不变，交互测量不等待每次排序，Spark 同时只提交一个更新，旧库采用异步全量强制排序；这是受控全量排序负载，不是生产查看器的裁剪与排序触发策略。记录 rAF 间隔、CPU提交耗时、可用且非disjoint的GPU timer query；无截图/读回进入性能轮次。GPU query包含本次同步提交的离屏生成与绘制，不是整个异步排序的GPU耗时。rAF是headless吞吐/调度证据，不是带显示器的端到端延迟。

另一个不计入性能的240帧截图轮次每20帧及终点取样（13帧），生成320px宽缩略图；再在完全相同的相机姿态等待排序并渲染，得到settled参考。live-vs-settled残差衡量采样点的排序/更新延迟，不是对真实移动视频的重建准确率，也不能排除采样点之间的闪烁。预先冻结质量门槛沿用MAE改善≥10%、参考PSNR下降≤0.1dB/SSIM下降≤.002；性能P95及加载比≤1.2；采样运动最大MAE≤.01且比旧库增加≤.002。`compare_spark_acceptance.py --root ...`汇总指标并保留逐视角与逐样本结果，绝不自动推广默认。

## 页面接入

高斯页面工具栏提供“旧查看器（默认） / Spark（实验）”。不改变 manifest、模型变体或资产 URL；选择只影响当前页面，刷新后恢复旧查看器。切换会重新加载当前模型，保留相机位置、朝向、目标点、FOV 和 zoom；更换模型或对齐来源时不沿用旧相机。漫游中先按 Esc 退出再切换。

两臂复用已有环绕控制器、预设视角、SfM 摆正与输入视图检查、导航碰撞和漫游设置。Spark 使用扩展源/累积编码、禁用 LoD、Z-depth 排序及已审计的 blur 参数；不透明度滑块只保留在旧查看器中。页面相机、窗口尺寸、裁剪和导航由交互控制，不冒充固定相机审计截图；此前 718×1277 的 headless 验收不是任意页面分辨率的性能保证。

页面等待上一实例释放后才创建下一实例。Spark 下载可取消，解码和排序结束后释放纹理、worker、controls、ResizeObserver、动画帧及 WebGL 上下文；旧库使用页面持有的外部 renderer，避免其销毁逻辑错误移除 React 容器。加载、SH 身份、渲染或上下文错误会显示，不自动改用旧库或降 SH；用户仍能手动切回旧查看器。

本地检查：`npm --prefix frontend test`、`npm --prefix frontend run build`，以及原有 Node 审计测试。新增测试使用模拟浏览器/渲染器，覆盖串行释放、快速切换、相机保留和跨模型隔离、解码/排序中的取消、错误及资源释放；不等同真实 GPU 页面验收。远端完整模型页面的来回切换、摆正/预设视角、缩放窗口、输入视图检查、漫游与长时间资源释放仍需部署后单独验证。

## 实现验证边界

CPU 单元测试覆盖输入门槛、SH3 非零合成数据、自检判定、模拟渲染器下的相机转换/排序/数量/降阶/WebGL/空帧拒绝，以及 CLI 在读取 Test 输入时于打开 PLY 和启动浏览器前失败。模拟测试不编译 GPU shader，不代表合成浏览器自检或约 300 万高斯加载已通过。
