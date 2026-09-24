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

## 检查边界

新增Node测试只覆盖夹具结构、逐行检查器和packed little-endian掩码。真实Spark解码和WebGL必须另外执行浏览探针，mock通过不能替代它。依用户持续规则，本机当前仅执行静态TypeScript／语法／diff检查；单元测试、真实GPU和远端执行没有本批授权，不称通过。
