# 本地 Gaussian 编辑 P0

日期：2026-09-24。状态：**P0技术准入通过**——固定Spark2.2.0／SH3／非LOD路径的源行表示、GPU掩码索引、遮挡选择、真实跨视角保留行参照和可逆隐藏通过；相关180项回归及构建通过。**导航短测P95=100ms，用户已明确当前足够，暂不要求50ms目标；P5完整产品验收尚未完成。** 本轮未进入P1、没有产品入口，未推送或部署。

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

## 合成浏览准入入口

`localGaussianP0Probe.ts::runLocalGaussianP0Probe(emptyMount, signal?)` 仅显式调用时运行；导入模块不启动渲染或Worker。它要求真实NVIDIA RTX4060 WebGL（拒绝软件回退）及现有localhost／HTTPS的Web Crypto；不通过浏览器安全开关绕过。

探针只生成17行小型合成PLY，不请求真实Job／云会话、不写模型／保存接口。两个相机／模型姿态分别检查：

1. 实际Spark解码后逐行核对11项几何属性，包含不显示的弱首行；
2. SH3与SH0实际画面有响应差异；
3. 全保留mask对原画面无额外>1/255通道差；
4. 原始第12行单独显示与从原PLY直接提取该行重新解码的画面一致，核对GPU source ID而非只看总画面变化；
5. Worker选择→隐藏→全隐藏→撤销，确认画面变化／全黑／恢复及源几何与SH数组哈希不变；
6. 重排后再次核对源行，结束释放mask、两个模型、Spark、WebGL和Worker。

返回JSON的 `status=passed` **仅指这个合成探针**，仍明确 `realModelAcceptance=not_run`、`productionEditor=not_enabled`。失败抛出原因，调用者必须保留失败记录，不能当作通过。像素采集包含readback，不是正式性能基准。

获单独授权后，可在既有Vite开发页中显式导入 `/src/localGaussianP0Probe.ts`，给它一个单独创建的空div并await调用，保存返回的JSON；禁止占用产品画布或关闭用户窗口。生产构建没有该实验按钮，也不保证开发模块URL存在。真实3M源身份、视角边界及500ms/2s预算仍须独立验收，不能用17行结果代替。

## 回归执行记录

用户另行明确授权本机回归与本机4060 P0验收后，本轮运行：

```bash
npm --prefix frontend test
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_gaussian_visible_selection.py tests/test_gaussian_editing.py tests/test_gaussian_editor_api.py tests/test_gaussian_export.py
node --test tests/test_gaussian_browser_loading.mjs tests/test_gaussian_browser_motion.mjs tests/test_gaussian_spark_audit.mjs
npm --prefix frontend run build
```

最终前端 **107/107**（本批共16项新增检查）、相关Python **54/54**、Node **15/15**，合计176项；生产构建通过。Python实际使用独立basetemp并限制CPU线程，未使用CUDA。保留既有Starlette/httpx弃用及Vite大chunk警告。完整日志在 `outputs/analysis/local-gaussian-p0-20260924-v1/final-regression/`。

## RTX4060实测结果与失败保留

本轮只绑定loopback启动临时只读审计HTTP服务及独立Chrome profile，不启动产品API、CUDA训练或远端服务，不读取真实数据集RGB／Test。实际renderer是 `ANGLE (NVIDIA, Vulkan 1.3.242 (NVIDIA NVIDIA GeForce RTX 4060 Laptop GPU (0x000028E0)), NVIDIA)`。七份自建临时profile均确认退出后清理；源模型、所有日志、失败JSON及截图保留。

- **合成v1失败**：第二测试相机把固定目标投影到x=266.33，超出256像素画布；选择器正确拒绝非法多边形。修正相机方向并增加视口回归（717cd84），没有放宽选择校验。v2及最终v3通过：两个视角均选中原始ID12，全保留／撤销恢复最大通道差0，源几何与SH数组不变。
- **真实源**：本机既有 `outputs/analysis/mcmc-render-audit-20260910/scene.ply`，2,999,577高斯，743,896,684字节，SHA256 `59cf39a46948ef447d35b299d77ffc14878a9e902a3e7460df0700a0b1df3a38`。这是与历史Spark审核相机同源的本机模型，不混用后来的c8b793…模型。冻结描述见本轮 `real-protocol.json`：相机1022/1111/1192，1280×720、DPR1，每视角三个128×128 ROI；只读取相机描述，不读RGB。
- **真实v1参考检查失败**：原始四元数与Ext量化结果在一行相差0.002385，不能套用合成夹具的0.002容差来认定行错乱。后续将属性差异作为量化诊断单独报告，未扩大容差后宣布身份通过。全量2,999,577行中心逐项完全匹配，但三对中心重复且属性近似，身份仍标为incomplete，不算已证实错误重排，也不算精确身份准入通过。
- **真实v2九个选择全部超时**：Chrome Worker单独诊断512次`setTimeout(0)`让出耗时2086ms，MessageChannel让出1.9ms。嵌套timer的4ms限速本身就超过3M扫描预算。144c5ac改为MessageChannel并在成功／失败／取消时关闭端口；2秒预算和选择算法不变。
- **真实v3选择速度样本通过**：九个固定ROI全部完成，Worker内部212.2–443.0ms（9样本nearest-rank P95=443.0ms），无超时。选中数量为33/268/520、3/31/0、976/591/251；零命中是保守结果，不扩大选择。八次有命中的隐藏＋渲染＋读回为64.2–79.4ms；撤销后全部逐像素恢复。三视角全保留mask与原图最大通道差0。此单场景样本不是跨场景性能保证或正式页面端到端测量。
- 每次真实执行前后源SHA一致，无模型保存／导出；本机加载约4.2s是本地loopback读文件，不代表远端网络加载优化。

## 重复行补充验证（2026-09-24，第二批）

`verifyP0EncodedRow` 将指定渲染槽位与从原 PLY **独立提取、单行解码**的参照比较：两组 Ext 几何／RGBA、SH1、SH2、SH3a、SH3b，共24个uint32逐字一致，不使用浮点容差。缺失数组、越界、非单行参照均拒绝。源ID定义仍是原PLY行号，渲染槽位只持有该原行的量化表示；导出必须回读原行，不能反推未量化属性。

第二批 `identity-v2` 六行检查全部通过；其中 `[2216493,2976869]` 可由完整编码区分，另外两对连全部SH编码都相同。这证明对应槽位具有各自原行应有的渲染表示，不声称恢复了量化前属性或观察到了WASM内部不可观测的相同行排列。不能以编码相同合并、去重或更换源ID；这两行仍分别占有独立mask位。完整GPU索引、排序后与跨视角检查继续执行，尚不据此单独宣告P0通过。新增回归逐一翻转24个编码字确认拒绝错误槽位；前端108项及构建通过。证据目录 `outputs/analysis/local-gaussian-p0-20260924-v2/`。

新增显式 `runLocalGaussianP0OcclusionProbe`，固定八类已知源ID夹具：前后遮挡、弱雾、两类半透明弃权、大高斯跨选区边、近裁剪、旋转轴向、各向异性边缘。第二批 `occlusion-v1` 在真实RTX4060上通过，八类Worker输出均等于预先给定ID；每类在两个视角将mask删除画面与**从原夹具保留行重新解码的独立模型**对照，16次最大通道差均≤1，撤销恢复亦≤1。未将GPU通过推广为真实场景物体分割或背景像素绝对保护。前端109项及构建通过；导入模块不启动GPU或产品入口。

## 可复现的显式审计

`scripts/audit_local_gaussian_p0.mjs` 仅在命令行显式执行时启动自建Chrome及loopback只读服务；固定路由不访问产品API。支持 `synthetic`、`occlusion`、`real`；真实模式必须传入先冻结的协议JSON，源SHA、字节数和62-float布局须匹配，输出目录必须不存在。原行／保留行参照只在本机流式传输，不写出派生PLY，不覆盖历史证据。脚本退出关闭自己的Chrome和HTTP服务，profile按验证退出后的清理记录处理。

```bash
node scripts/audit_local_gaussian_p0.mjs outputs/analysis/p0-new-synthetic synthetic
node scripts/audit_local_gaussian_p0.mjs outputs/analysis/p0-new-occlusion occlusion
node scripts/audit_local_gaussian_p0.mjs outputs/analysis/p0-new-real real outputs/analysis/local-gaussian-p0-20260924-v2/protocol.json
```

第二批真实协议在运行前固定：同源／同3相机／9个128×128 ROI，选择**端到端**P95≤500ms、硬预算2秒，已知选集隐藏含读回≤150ms；三个选集分别在三视角对照原PLY保留行重新解码结果，像素容差1；六条重复源行各两个近景视角核对GPU掩码索引，并用白底防止黑色高斯空图通过；所有几何及SH编码数组前后哈希必须相同。额外运行ABBA顺序的导航短测，以及两分钟、100次隐藏／撤销，采样JS heap、GPU资源数量和整机显存／可用RAM；整机采样不归因于本浏览器，也不是精确峰值。

第二批真实审计v1/v2在返回包含多张PNG的巨大CDP消息时断线，保留 `status=failed`，不计为GPU验收通过。阶段日志确认v2已运行到资源检查110秒；不能将 `DevTools关闭` 误称为触发10分钟超时。独立无模型传输复现：1MiB成功、4MiB整条消息断开；改为 `readP0String` 每次最多64KiB、总字符串上限16MiB，报告与截图分别读取后，4MiB和16MiB均通过。两项Node回归覆盖分块大小、重组、超限、截断和浏览器异常；不改选择预算或渲染门槛。

阶段边界以原实施计划为准：P0是源ID／仅alpha／选择正确性及最小真实技术验证；导航50ms／20%目标可提前诊断，但两分钟库探针不替代P5的20分钟、双模型、正式UI与持久化完整验收。原报告把这些未完成项一并列为“P0阻塞”不够准确；即使P0技术项通过，也不自动启用入口或宣布P5通过。

## 首轮未通过的门槛（历史记录）

1. 重复中心行 `[793798,1440771]`、`[2216493,2976869]`、`[1615456,1701443]` 不能仅凭位置和近似量化属性完成一对一源行证明；须补显式来源身份／重复高斯验证，不允许把count与位置匹配当完整证明。
2. 同视角可逆隐藏不能证明多视角背景安全。真实删除后画面有框外变化，当前样本未完成跨视角误删验收。
3. 连续导航P95／编辑额外开销、长时资源和产品交互尚未验收；本轮未进入P1/P5完整产品验收。

因此第一批真实v3保留 `status=failed`（当时身份门槛未闭合），第一批结论为**部分验证通过、P0仍阻塞**。第一批 `summary.json`、失败v1/v2不覆盖；第二批补证结果如下。

## 第二批最终结果

证据：`outputs/analysis/local-gaussian-p0-20260924-v2/summary.json`；完整真实报告 `real-v4/result.json`，12张PNG同目录；`real-v3`首次成功记录保留，v4补强为在首次GPU准备后及全部操作后重新取得**当前实际数组**计算哈希，不只比较旧数组引用。最终合成／遮挡报告 `synthetic-final/result.json`、`occlusion-final/result.json`。

- **源身份**：同一59cf39…模型全部2,999,577行中心一致；六条重复源行的24字独立编码逐项一致；六行各两个近景视角的真实GPU单行mask与独立原行模型最大像素差均0。编码相同的原行仍有两个独立源ID，不声称反演量化前属性。所有几何／颜色／SH编码数组前后SHA相同，磁盘原PLY SHA相同。
- **选择与即时隐藏**：九个固定ROI输出33/268/520、3/31/0、976/591/251；Worker内部187.9–353.9ms；包含消息往返的端到端211.3–380.1ms，九样本P95=380.1ms，满足500ms目标和2秒硬预算。八次非空隐藏＋渲染＋读回96.9–113.2ms，满足150ms目标；全保留通过≤1通道差检查，八次选后撤销最大通道差均0。
- **真实跨视角**：三个固定选集（33/268/520行）各在三个冻结视角对照独立保留原行重新解码模型，九次最大通道差均0，跨视角撤销亦恢复。第三组选集在1192视角仍改变17,638个“最大通道差>8”的像素，这是删除这些高斯的真实跨视角影响，不是额外源ID错删。保存基线和删除截图；没有语义物体／悬浮物标签，不能将此宣称为任意背景绝对保护、精确去雾或物体分割验收。
- **短时稳定性**：120.018秒、100次隐藏／撤销均通过≤1/255通道差恢复检查；无context丢失；13个采样中纹理15、几何2、program3均稳定。JS heap约365.6–400.1MiB浮动；整机显存采样最高580MiB、稳定阶段574MiB，不是独占浏览器精确峰值。两分钟不替代二十分钟验收，也不能证明永久无泄漏。
- **导航诊断与用户决定**：ABBA顺序同路径短测，纯Spark／全保留编辑mask的帧间隔P95分别99.9／100ms，比值约1.001，额外开销样本达标，原冻结协议的50ms绝对目标未达到。2026-09-24用户明确“暂时不用管导航的间隔目标，100ms已经足够”，因此当前导航表现按用户决定接受，不再作为P0收尾阻塞，也不在本轮追加性能优化。原协议与报告中的 `targetPassed=false` 保持不变，避免把历史未达标重标为通过；这是要求暂缓，不是性能改善。探针每帧同步等待Spark排序再提交渲染，与未来正式页面的异步调度不等价，不能换算为正式页面平均FPS。
- **最终回归**：前端109、Python54、Node17，共180项通过，构建通过。仅既有Starlette弃用和Vite大chunk警告。合成17行探针、八类遮挡／双视角探针均以最终审计器再次通过。
- **清理与证据保留**：第二批11份自建Chrome profile均确认没有引用该目录的存活进程后清理，记录见 `cleanup.json`。保留全部成功／失败报告、截图、冻结协议和原模型，不清理用户浏览器目录。

结论仅为原计划P0的技术准入完成。P1–P5尚未实施／完成；产品入口、保存与导出协议、正式UI、双模型与20分钟资源验收仍保持原阶段边界。本轮不训练、不读取数据集RGB/Test，不保存或落盘导出模型，不修改云服务或生产默认，不推送。
