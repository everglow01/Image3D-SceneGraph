# 本地 Gaussian 编辑 P0

日期：2026-09-24。状态：独立技术探针开发中，**运行准入未通过、没有产品入口**。

## 范围与身份

只验证固定 Spark 2.2.0 的非 LOD、SH3、ExtSplats 路径；不训练、导出或改写模型，不启动云渲染／本机Python服务。产品仍用原查看器入口。

`frontend/src/localGaussianP0Source.ts` 提供紧凑的源序几何快照（xyz、scale、xyzw、opacity），保留source SHA和count；不复制SH。源ID约定为原PLY行号，而非排序位置。`requireP0Mesh` 拒绝LOD、paged、已有修改器、RGBA覆盖和非SH3路径。

已读安装包 `SplatLoader.loadInternal`／内嵌Worker：`lod=false` 使用 `decode_to_extsplats`，排序不在此JavaScript加载层执行。实际解码在WASM内，因此**没有把“JS层未排序”当作源行顺序证明**。

17行合成PLY含非空间顺序的唯一中心、不同尺度／旋转／alpha及非零SH3，还含低于显示阈值的首行。`verifyP0FixtureRows` 比较每一行全部11项几何属性，在明确的Ext量化容差内检查，而非只看count。合成通过只证明该夹具；真实源映射及重新排序／隐藏后的身份仍须独立浏览探针验证，不允许据此开放产品删除。

## 仅透明度探针

`localGaussianP0Mask.ts::P0AlphaMask` 在 object modifier 中按原始source index查询R8UI packed mask，只将隐藏项的alpha置零，不覆盖RGB或SH。纹理更新同时触发Spark generator失效，以免沿用旧accumulator；撤销仅恢复mask。独占未修改mesh，拒绝LOD／已有修改器；释放恢复原管线、清理纹理。调用方须串行等待在途Spark更新后再修改／释放，此类不负责渲染调度。

测试使用安装包的真实Dyno着色器构造器检查采样器绑定和仅alpha写入，以及纹理副本、撤销、拒绝非法管线和幂等释放；没有把shader字符串检查称为GPU编译／像素验收。产品查看器尚未接入本探针。

## 有界 Worker 表层选择

`localGaussianP0Selection.ts` 固定到当前Spark渲染配置：无斜切透视、刚体模型相机矩阵、preBlur=0.3、blur=0、maxStdDev=3、clipXY=1.4、focalAdjustment=1、falloff=1、alpha阈值1/255，不支持2DGS、景深或LOD。使用投影椭圆与前向透明贡献，而非中心点或期望深度；输出ID仍为输入几何的行号。JS双精度投影与GPU浮点及accumulator再次量化的差异须实测，不宣称原生像素完全一致。

首个技术探针只接受多边形包围矩形面积≤16,384实际绘图像素，最多100,000候选、8,000,000候选／贡献访问、2秒；超过任何预算整次失败、不应用部分结果。每批主动让出Worker事件循环，以处理取消／换源；失败不降采样或退回穿透。16×16瓦片的按深度贡献遍历复用Python `FrontLayer` 的语义，弱于5%前层可能跳过寻找主表层；5%–50%的半透明前景弃权，达到50%才形成可信选集。这不是物体分割，也不保证背景绝不受影响。

`gaussianSelection.worker.ts` 不在产品中自动启动。控制器拒绝旧model generation／重复sequence，换源和取消阻断在途结果；返回camera generation供调用方在实际应用前再次核对。共享夹具 `tests/fixtures/gaussian_front_layers.json` 同时供Python与TS回归使用。

## 检查边界

新增Node测试只覆盖夹具结构、逐行检查器和packed little-endian掩码。真实Spark解码和WebGL必须另外执行浏览探针，mock通过不能替代它。依用户持续规则，本机当前仅执行静态TypeScript／语法／diff检查；单元测试、真实GPU和远端执行没有本批授权，不称通过。
