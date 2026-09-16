# Spark 独立渲染审计原型

## 范围

`node scripts/audit_gaussian_browser.mjs --renderer spark` 是研究专用入口，profile 为 `fixed_camera_spark_v1`，输出 `spark.json` 和 `spark-controlled-sh*-<image_id>.png`。省略 `--renderer` 仍运行原来的 GaussianSplats3D 审计；生产 `GaussianSplatViewer.tsx`、导出、训练和默认查看器均不变。

Spark 精确锁定为开发依赖 `@sparkjsdev/spark@2.2.0`（MIT），由 `frontend/package-lock.json` 固定发行包。只使用已安装的本地模块和输入 PLY，不上传模型到第三方、不在浏览器中访问 CDN。npm 发行包已包含 WASM/Worker，不要求运行时安装 Rust。远端安装、执行与部署仍需分别授权；源码仅通过 Git 同步。

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

## 实现验证边界

CPU 单元测试覆盖输入门槛、SH3 非零合成数据、自检判定、模拟渲染器下的相机转换/排序/数量/降阶/WebGL/空帧拒绝，以及 CLI 在读取 Test 输入时于打开 PLY 和启动浏览器前失败。模拟测试不编译 GPU shader，不代表合成浏览器自检或约 300 万高斯加载已通过。
