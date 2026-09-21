# 3DGS 修剪与云端交互渲染实施计划

> 实施状态（2026-09-21 续建）：本地编辑数据核心、API/会话、独立修剪界面、WebRTC接线和产品GPU租约已实现，CPU/mock回归及前端构建通过。**用户已确认8082/TCP和8082/UDP开放。** 可选cloud依赖已锁定但未安装；远端Git同步、同机认证TURN部署、真实CUDA/浏览器媒体连通和性能验收均未执行。这里的实现状态不等于已上线或E1–E3远端验收完成。真实接口、配置和限制以 `gaussian-editor-contract.md`、`cloud-gaussian-rendering.md` 为准；以下保留总体分阶段计划。

## Context：当前最高优先级

用户要求优先交付两项能力：训练、导出完成后，手动修剪 3DGS 中的白雾、漂浮物和家具发散白丝；把重型高斯渲染移到服务器，浏览器只接收画面并提供交互。此前论文工程化计划已保存于 `docs/video-quality-improvement-plan.md` 并暂停，Q0/Q1 尚未实现；两项新能力基本完成后，再根据用户指示恢复论文工作。

**用户已确认：**
- 首版只做修剪清理，不做家具移动/旋转/缩放、复制、补洞、颜色或材质编辑。
- 包含选中预览、范围限制、删除/隐藏、撤销/重做、保留 Original、另存及导出。
- 仅公司内网/VPN，先保证单人活跃会话流畅，不做多人同时编辑。
- 沿用 `i-94B8D131:/usr/local/3dgs_new/Image3D-SceneGraph`。
- 接受运动时自适应分辨率、停稳和编辑时高清。
- 编辑/云观看与训练互斥排队，不抢占、不自动停止已有任务。
- 管理员已开放8082/TCP和8082/UDP；**开放端口不等于已授权部署TURN，不擅自增加端口，也不把正式媒体方案改为WebSocket图像流。**

本计划及续建批准允许实现本地代码与CPU/mock测试；安装远端依赖、运行GPU验证、启动服务、同步代码和部署仍须明确范围授权。8082两协议已确认，但同机TURN/relay实际部署及最终连通验收仍是独立待办，不宣布云产品已上线。

## 1. 现场事实与现有代码

### 1.1 2026-09-21 只读检查

- 远端代码：`51a6a9d4daec7339dd4b00756f47a92392e26961`。
- GPU：2× NVIDIA L2，各 23034 MiB；检查时没有 CUDA compute 进程，各约 3 MiB 使用量。
- CPU：12 个逻辑 CPU；项目文件系统剩余约 22 GiB。不得自动清理历史实验以腾空间。
- Torch 2.3.1+cu121、gsplat 1.5.3+pt23cu121；av/aiortc/websockets 未安装。
- FFmpeg 4.4.2 声明 libx264 和 h264_nvenc；只证明编译器列表存在，未运行 NVENC/吞吐试验。
- Vite 8081 代理 `/api` 到 127.0.0.1:8000；当前没有云渲染会话服务。监听端口清单不等于防火墙允许范围。

### 1.2 可复用边界

- 前端 React/TypeScript/Three.js；`frontend/src/GaussianSplatViewer.tsx` 已有相机、预设、upright、orbit/walk、诊断面板，以及 legacy/Spark 两个本地高斯引擎。
- 后端 `backend/main.py` 是 FastAPI；`JobStore` 和 `LocalJobWorker` 已有文件生命周期与单 worker 租约，但没有训练/实时渲染的共同 GPU 租约。
- `gaussian/render.py:render_gaussians` 可输出 RGB/RGB+ED；服务器优先复用原生 gsplat，不用服务器上的完整浏览器截图充当正式渲染架构。
- `gaussian/export.py` 已有 `read_gaussian_ply`、`write_binary_ply` 和标准 PLY 字段；`gaussian/importer.py` 可导入标准 PLY。
- `gaussian_comparison` 当前是只读容器；已发布 PLY 可能是原实验的硬链接。**绝不能原地修改这些文件或为了编辑解禁源 Job。**
- 模型渲染无需加载训练/Validation/Test RGB；本项目编辑与云观看不运行训练或评价。

## 2. 推荐技术栈与进程架构

### 2.1 首版选型

| 层 | 采用 | 不采用 |
|---|---|---|
| 交互界面 | 现有 React＋TypeScript；Three.js 只做相机数学，Canvas/SVG 做二维选择和覆盖层 | 不把百万高斯下载到浏览器，不引入完整 DCC 平台 |
| API/会话控制 | 现有 FastAPI，同源 HTTP API | 不另建 Node 后端或独立公开控制端口 |
| 渲染 | 独立 Python 子进程，Torch inference mode＋现有 gsplat | 不在 Uvicorn 事件循环执行 CUDA，不用训练 worker 承载无限会话 |
| 实时媒体 | 优先 WebRTC，aiortc＋PyAV，浏览器 `<video>`；无音频 | 不采用高延迟 HLS，不承诺 WebCodecs 在当前 HTTP 下可用 |
| 高频相机输入 | WebRTC DataChannel，带序号、最新状态优先 | 不逐鼠标事件发阻塞 HTTP 请求，不积压相机历史 |
| 编辑/保存 | 有序、幂等 HTTP 命令＋编辑 revision | 不用不可靠 DataChannel 提交删除/保存 |
| 持久化 | 本地文件式编辑文档、不可变操作快照、原子发布 | 单人阶段不引入 Redis、数据库、对象存储或多人协作框架 |

aiortc 1.15.0 的 PyPI 元数据支持 Python ≥3.10，依赖 av ≥14,<18；PyAV 16.1.0 有 Python 3.10 Linux x86_64 wheel。将二者作为依赖审计起点，实施时精确锁定及核验传递依赖、许可证/二进制兼容，不采用不兼容的最新 PyAV 18。现有 Torch/CUDA/gsplat 不升级；新增 `cloud` 可选依赖组，不强加给 base/mock 安装。

**编码诚实边界：**aiortc 标准 H.264 编码器采用 libx264 软件编码，不会自动用 NVENC。首版先以一个会话验证该标准链路。若达不到门槛，再单独评估硬件编码；不第一天实现 FFmpeg packet passthrough、定制 RTP/关键帧控制和零拷贝。FFmpeg encoder 名称存在不能当作硬件编码已经可用。

### 2.2 数据流

```text
浏览器：相机输入 / 选择工具 / 视频解码
  ├─ 同源 HTTP：会话、SDP、固定高清帧、编辑、保存、导出
  ├─ WebRTC DataChannel：最新相机状态
  └─ WebRTC video：服务器渲染的连续画面
              │
FastAPI：来源校验、会话凭证、revision、资源仲裁、持久化
              │ 有界本机 IPC
单会话渲染进程：只读加载源高斯 → 应用可见 mask → gsplat → RGB
              │
标准 aiortc/PyAV 编码与传输
```

- 首版一个活跃媒体会话、一个加载模型、一个渲染进程。Project/MCMC/编辑版本切换先释放前一个，不同时驻留三个大模型。
- 不搞渲染集群、SFU 或通用插件系统。IPC 用标准库有界队列/管道；未测到瓶颈前不引入共享内存池或 GPU IPC。
- 相机队列只保留最新状态，编码/发送最多少量有界在途帧；过时的相机与未编码帧可丢弃，编辑命令不可丢弃。
- 会话创建后只加载一次模型；帧循环只推理、不构建优化器、不反向传播。
- 模型加载、保存和导出等慢操作返回进度，不阻塞 API 健康检查。

## 3. 8082已确认，部署与连通待验收

- 信令和编辑API继续通过8081 → 8000同源代理，不新增公开控制监听。
- 浏览器使用管理员批准的同机认证TURN，客户端入口8082/TCP与8082/UDP，强制relay；服务器aiortc保持 `iceServers=[]`。不使用公网STUN/TURN。
- 后端读取受保护部署配置，生成1小时有效的TURN凭证，只返回限时凭证而不返回共享密钥。
- aiortc1.15.0没有固定ICE媒体端口配置。TURN的listening port不等于relay endpoints，管理员须核验同机relay/peer路由、防火墙、允许peer IP和中继端口范围。端口开放本身不是已部署/必定可达的证据。
- 未获部署授权，不安装TURN、启动ICE监听、修改防火墙或重启服务。缺配置时明确不可用，不自动退回图像流。
- WebCodecs依赖secure context，不把当前内网HTTP当成默认可用前提，不关闭浏览器安全策略绕过。

精确环境变量及当前媒体合同见 `cloud-gaussian-rendering.md`。

## 4. 修剪交互：先选准，再删，始终能恢复

### 4.1 用户操作闭环

1. 在历史模型/比较中选择明确的 Project、MCMC 或编辑版本，进入“云端查看”。
2. 自由旋转、平移、缩放；运动时自适应分辨率，停稳时提高画质。
3. 点“进入修剪”：服务器生成固定高清 PNG，前端确认该图已经显示后才启用圈选。此时暂停相机移动，避免画面与操作坐标错位。
4. 矩形/套索圈选，并使用近远深度范围；或使用三维轴对齐盒裁剪。提供增加选择、减去选择、清空选择的明确范围说明。
5. 服务器返回选中数、高亮预览、选中项单独显示及删除后预览；不立即修改保存版本。
6. 用户确认删除，生成一个可撤销操作；恢复交互观看，从其他角度检查。
7. 撤销/重做；保存编辑版本；按需导出新 PLY/元数据/ZIP。Original 始终可重新打开。

首版“删除”是从当前编辑可见集合移除，不修改源高斯属性；“隐藏/隔离选中项”用于检查，不冒充已保存删除。预览不产生持久编辑 revision，确认操作才产生。

### 4.2 二维选择的准确语义

半透明 3DGS 不是三角网格，没有天然唯一的“鼠标点到的表面”。首版不承诺自动识别白雾或完美可见面 picking：
- 选择依据为固定相机下的投影区域＋显式 camera-Z 区间，或显式三维盒范围。
- 默认采用高斯中心命中，行为清楚可复现；对白丝/大高斯提供“覆盖范围候选”模式，利用投影 bounds 找与区域重叠的候选，并明确它是保守候选，可能多选，必须预览确认。
- 高斯延伸部分被圈到但中心在远处时，不静默声称中心模式已选中它；引导改用覆盖范围或三维盒、隔离检查。
- 不把 expected depth 当绝对表面，不能据此宣称后台家具绝不会被选中。深度限制、预览和另一视角检查共同降低误删。
- 大比例删除增加二次确认；清空全部在首版拒绝保存/导出，避免违反现有非空模型合同，仍允许清空选择。
- 参数有限：polygon 点数/面积、范围、索引数量、数值有限性、request bytes 均设上限；客户端不能上传任意 mask 文件或任意路径。

选择算法先复用 NumPy/Torch 投影与现有 render metadata，不为第一版修改 CUDA rasterizer 生成复杂逐像素 Gaussian ID buffer。合成遮挡/大椭球测试必须明确展示选择语义的局限，而不是用简单点云测试宣称支持了完整高斯 picking。

### 4.3 固定帧票据防错删

`freeze_frame` 返回 PNG 及 opaque frame ticket，服务器保存以下绑定：
- source PLY hash、edit document ID、revision；
- camera_seq、完整相机矩阵/K、像素尺寸、近远裁剪、显示变换；
- 帧 ID、失效状态与有限生命周期。

前端用 PNG 实际像素区域映射选择，不用 CSS 容器尺寸直接计算；处理 letterbox、DPR、页面缩放和 resize。选择、预览、确认均引用票据及 expected revision。相机变更、模型切换、viewport 变更或其他编辑使旧票据失效，返回明确冲突，不能自动套用到新画面。

这样无需猜测视频 RTP 时间戳对应哪次鼠标输入，也不会在陈旧视频帧上执行当前相机的删除操作。

## 5. 编辑文档、来源与导出合同

### 5.1 源选择

服务器只接受 `job_id + variant_id/asset_role` 或已注册的编辑版本 ID，通过 manifest 和 `JobStore.get_asset_path` 解析 contained assets。禁止客户端传任意 PT/PLY 路径。

优先从用户正在看的标准导出 PLY 建立编辑源；校验 browser hash、Gaussian count、SH、坐标与 metadata。一份 PLY 的行号作为稳定 base Gaussian ID，绑定该源 hash，不能用 renderer 每帧排序位置作为 ID。

支持普通完成 Job 及 `gaussian_comparison` 的已发布模型，但编辑副本独立于源容器。源资产仍只读，即使是 hardlink 也不得打开写入。

### 5.2 单独的编辑文档，不伪造训练 Job

建议根目录：`outputs/edits/{edit_id}/`，由后端生成 ID。

```text
edit.json                  # 源 Job/variant/PLY hash、schema、归一化坐标、当前游标
operations/                # 不可变的 mask 快照与操作记录
versions/v000001/           # 保存的版本，后续不可覆盖
  edit-manifest.json
  visible-mask.npz
  export/                  # 只有显式导出时才创建
    scene.ply
    export.json
    bundle.zip
```

- 几百万高斯的 bool mask 可 packbits；3M 约 0.36 MiB/快照，不为每次点击复制 700 MiB PLY。
- 优先采用简单的完整 packed mask 操作快照，不设计复杂增量数据库。支持至少最近 100 次编辑的撤销/重做；具体保留上限写入合同并显示，不能静默删除保存版本。
- 新操作的文件原子发布后才更新 revision/游标并回 ACK。重复 operation ID 返回同一结果；相同 ID 不同参数拒绝；expected revision 冲突拒绝。
- Undo 后再编辑会形成新的线性分支，旧已保存版本仍保留，不偷偷覆盖其文件。
- 断线释放 GPU，但已确认的编辑保留；重新进入从已持久化状态恢复。未确认预览可以丢弃，并清楚区分。
- 输出根现有目录不覆盖；磁盘不足时拒绝新保存/导出，已提交状态和 Original 保留。

### 5.3 导出必须诚实

- 按 visible mask 从原始 PLY 的行中确定性过滤，保留未删行的数值、顺序、SH3 和坐标，不做优化/重采样/自动修洞。
- 复用 `write_binary_ply` 和 ZIP 基础工具；编辑专用 export record 包含 source/parent/revision/mask/output hashes、删除数量与编辑 profile。
- 不能把修改后 PLY 塞入旧 `export_gaussians` 并伪造 evaluation hash。源训练 PSNR/SSIM 仅作为“原模型历史指标”显示；编辑模型标 `manual_edit_not_evaluated`。
- 源 navigation/collision/scene graph 不自动继承为编辑后有效。首版云编辑为 orbit 观看，不宣称支持编辑后的碰撞 Walk；原 Job 的功能不删除。
- 编辑 API 返回版本化 `edit-manifest`，前端只读资产与能力，不理解 tensor 格式。编辑版本在源模型旁提供独立版本选择，不冒充另一次训练成功。

## 6. 渲染会话、坐标与 GPU 调度

### 6.1 会话状态

`idle → loading → viewing → editing_frozen → viewing → closing/closed`；失败有独立 error/recoverable 状态。

- 模型加载、close、切换和后台进程退出幂等；迟到帧不得重新覆盖新模型画面。
- 每个画面绑定 session/model revision；编辑 ACK 后展示的第一帧必须是新 revision，防止“删了又出现”的旧帧回放。
- heartbeat、闲置时限、断线宽限有明确记录。初值建议：heartbeat 5 秒、断线宽限 30 秒、无实际交互 10 分钟释放 GPU；空 heartbeat 不延长实际闲置时间。释放前保存已确认操作，不自动删除持久版本。
- 网页隐藏/暂停时降低或停止持续渲染；可恢复时请求当前完整帧。

### 6.2 坐标和画质

- 渲染全程沿用 normalized arbitrary units，保留 SH3；不借机改变训练或模型单位。
- Three 相机的 Y-up/−Z-forward 与 native camera 的 +Z-forward 转换明确封装；upright 只作显示变换，反向映射选择到原始归一化坐标。
- 用固定 camera-path 与合成轴/点/椭球验证 server/client 的 K、pose、投影、裁剪、up 和 letterbox，不靠肉眼猜方向。
- 建议运动最大边 1280、目标 30fps；静止/固定编辑帧最大边 1920，按 viewport aspect 生成。自适应只在声明的分辨率档位内，不自动降低 SH、删高斯或改 opacity。
- 服务器画面先按现有 display profile clamp/量化，再编码；编辑固定帧为无损 PNG，不在有损视频上精细判断删除结果。
- 不承诺浏览器零 GPU 使用：视频解码/显示可能由本地硬件完成；承诺的是不在浏览器加载、排序和 rasterize 百万高斯。

### 6.3 互斥而不抢占

现有 `.worker.lock` 只防第二个 Job worker，不能保护云渲染；新增共享 GPU 资源租约：
- 所有产品 GPU Job（含几何/导航）在执行前取得租约；云会话在加载模型前取得同一租约。
- 已有训练运行：云请求返回“训练占用，等待”，不杀进程、不启动半个模型。
- 云会话活跃：新训练保持 queued；用户关闭或会话到期后释放，再由 worker 执行。
- 前台本次模式不拆 GPU 给训练，不修改双卡训练参数。首版即使渲染只用 GPU0，也把产品训练整体隔离，以免双卡训练冲突。
- 租约必须覆盖真正 CUDA 子进程生命周期：进程异常、API退出或子进程未退完不能过早释放。用 OS 文件锁/子进程持有方式及退出回收测试保证，不仅写一个 owner JSON。
- 检测到非产品的外部 GPU 任务时保守拒绝启动；不自动停止它。未来研究/dashboard 任务须使用同一租约入口；不能声称能强制约束任意 root 用户手动启动的 CUDA 程序。

### 6.4 内网安全与资源上限

- 会话使用短期随机凭证，绑定当前会话及允许源；HTTP mutation 检查同源、CSRF/自定义头、revision；不在 URL query、日志或模型 metadata 中写凭证。
- 会话凭证是隔离机制，不冒充账户登录。首版继承受信内网产品边界，不对公网提供安全性承诺。
- DataChannel 的消息有大小/速率上限，相机参数必须 finite、合法刚体/K/视锥；禁止任意命令、路径或代码。
- 首次目标模型上限 3M Gaussians；拒绝超限，不偷偷截断。加载前检查模型完整性、预计内存、源大小和磁盘余量。
- 首版仅一活跃会话，其他请求明确 busy，不能从另一浏览器抢走编辑会话。
- GPU OOM 在渲染子进程隔离，回收 CUDA 后报告失败，API 和已保存编辑继续可用；不以无限降画质或重启后端掩盖问题。
- 22 GiB 剩余磁盘下，保存 mask 很小，但导出会产生 PLY/ZIP。每次导出按源体积、临时文件和安全余量预检，禁止自动清旧实验。

## 7. 关键文件与最小改动范围

不重写 `jobs.py` 或整个查看器，新增窄职责模块：

- `src/image3d_scenegraph/gaussian/editing.py`：选择、稳定 ID、mask、undo/redo 记录与编辑导出验证；纯数据部分 CPU 可测。
- `src/image3d_scenegraph/gaussian/cloud_render.py`：冻结模型加载、相机验证、进程/帧生命周期；复用现有 renderer。
- `src/image3d_scenegraph/gaussian/cloud_media.py`：唯一的 WebRTC 会话实现，依赖延迟导入，不搭建多 transport 插件层。
- `src/image3d_scenegraph/gpu_lease.py`：训练与渲染共用的最小文件租约。
- `backend/gaussian_editor.py`：专用 APIRouter；`backend/main.py` 只接入生命周期和路由。复用 `GET /api/backends` 的能力报告方式，区分编辑核心可用、cloud 依赖缺失、网络待配置、GPU忙碌等状态；缺少可选依赖不能使已有 mock/训练 API 启动失败。
- `worker.py`/必要的执行边界：在开始实际任务前取 GPU 租约，不改训练参数。
- `frontend/src/CloudGaussianViewer.tsx`：video、相机输入、连接/错误状态、固定帧修剪工具。已有 `GaussianSplatViewer.tsx` 仅增加明确的云/本地入口和必要共享相机工具，不把 3 种 engine 混成上千行新条件。
- `frontend/src/App.tsx`：来源、编辑文档/版本选择和指标身份区分。
- `pyproject.toml`/`uv.lock`：可选 cloud 依赖；没有新增 WebSocket 协议用途时不顺手装 websockets。
- `docs/gaussian-editor-contract.md`、`docs/cloud-gaussian-rendering.md`、必要的 manifest 文档及 `codex.md`：版本/来源/安全/资源和运行方式。

测试分为编辑数学/存储、会话API/租约、前端输入与 stale frame 三组；不为单一实现增加通用服务框架。

## 8. API 草案（实施前冻结到合同）

- `POST /api/gaussian-edits`：从受控源创建编辑文档，不加载 CUDA。
- `GET /api/gaussian-edits/{id}`：读取文档、版本、能力和当前 revision。
- `POST /api/gaussian-render-sessions`：为受控源/编辑版本请求单会话 GPU 租约。
- `POST /api/gaussian-render-sessions/{id}/offer`：SDP 协商，媒体网络确认后启用。
- `POST .../{id}/freeze-frame`：取得绑定 revision/camera 的高清 PNG 与票据；响应身份通过安全元数据而非任意文件路径给前端。
- `POST /api/gaussian-edits/{id}/selection`：票据＋圈选/体积＋深度区间，返回 selection token、数量和预览。
- `POST .../{id}/operations`：selection token＋expected revision＋operation ID，确认删除/undo/redo。
- `POST .../{id}/versions`：保存不可变版本。
- `POST .../{id}/versions/{version}/export`：有界异步导出，状态/资产用受控 GET 查询和下载。
- `DELETE /api/gaussian-render-sessions/{id}`：仅释放自己的会话，不删除编辑文档、源模型或历史版本。

所有动作以源/revision/lease 验证为先；导出和保存状态不能用“返回200”冒充全部完成。首版不提供任意文件上传编辑或自动剪枝接口。

## 9. 分阶段交付与阻塞处理

### E0：合同和可运行性证明，先验证最难链路

本地先完成合成相机/选择/来源/租约测试、依赖审计、最小云会话骨架。远端获得精确授权后：
- 固定现有 ~1.48M、~3M PLY，只读加载；同一视角测试原生渲染耗时、RGB到CPU/编码开销及内存；输出独立证据，不训练。
- 网络未批准时，不绑定 WebRTC UDP；只能在获准的有界试验中测渲染/编码，不称端到端串流通过。
- 标准软件 H264 是否足够由此决定；不承诺 NVENC，也不在失败后擅自升级依赖。

**停止条件：**源/坐标错误、资源不足、渲染或编码达不到可用目标，先报告并修主因，不继续堆编辑UI。

### E1：云端浏览闭环

管理员确认网络后，单用户可在现有页面创建会话、接收视频、旋转平移缩放、调整窗口、关闭/重连。加载/等待训练/网络失败状态完整。云模式浏览器不请求大 PLY。

通过连续移动和10分钟会话、退出释放及与训练互斥后，进入精细编辑集成；编辑纯逻辑可在等待网络时先完成，但不得声称媒体体验已验收。

### E2：修剪闭环

固定高清帧、矩形/套索＋深度、三维盒、预览、确认删除、undo/redo、断线恢复、保存版本、原模型对照。优先让“选对并安全删掉白雾/白丝”可用，不做整套3D建模工具。

### E3：导出与产品验收

编辑模型可导出独立PLY，重新打开后效果与保存版本一致；原始文件哈希不变。真实浏览器、3M模型、内网链路综合验收后再决定云模式默认入口；legacy/Spark保留为显式选择，云失败绝不静默下载PLY转本地GPU。

**待验收项：同机认证TURN部署、relay/peer通路和实际媒体表现。** 8082两种协议已经确认开放；不以别的协议绕过所选方案，也不把后端能生成PNG当成云交互完成。

## 10. 验收标准与测试

### 10.1 必须通过的功能/安全测试

- 已知点/椭球、多层遮挡、长高斯、画外中心、不同深度、upright、DPR/letterbox 的选择结果符合声明语义。
- 陈旧票据、模型切换、resize、乱序命令、重复操作 ID、revision 冲突不会误删。
- 连续删除/撤销/重做恢复相同 mask；断开重进可恢复已确认操作；保存/导出后的 count 与mask严格一致。
- 源 PLY（包括 hardlink）前后 hash 不变；所有派生行的原属性不被意外重算。
- 空模型、越界 ID、超大 polygon/消息、非法相机、path traversal、跨会话 token、越权源被拒绝。
- 训练/导航与渲染的租约互斥，排队而不抢占；渲染子进程异常退出不影响 API，GPU内存释放后再允许下一任务。
- edited 模型没有冒用旧评价指标、旧碰撞有效性或训练完成身份；不读取 Test RGB。
- 云模式无大 PLY 下载、无本地 splat engine 初始化；视频解码/轻量UI仍允许使用浏览器硬件。

### 10.2 初始性能目标，不是已实现保证

在同一内网、桌面 Chrome/Edge、~3M模型、单会话下，冻结 viewport/相机短路径和统计方式：
- 运动：最大边1280档位目标30fps；实测呈现帧率目标中位≥25fps、P95 frame interval≤66.7ms。
- 相机输入到对应画面呈现的 P95 目标≤200ms；以带序号的受控视觉响应试验和浏览器呈现回调测量，不把服务器 render耗时冒充端到端延迟。无法可靠映射时报告该项未验收。
- 停稳后高清帧最大边1920，目标≤500ms得到清晰画面；选择 preview 目标≤1秒，确认删除后的反馈目标≤1秒。
- 无长期增长的帧队列；连续10分钟及20次连接/关闭/模型切换无持续内存增长或残留进程。
- Original与未编辑云模型使用相同原生renderer固定相机比较；无损帧先验收坐标/SH/颜色，再单列有损视频压缩偏差。
- 不给未知VPN延迟、所有浏览器或任意大于3M模型承诺上述性能。

不达标则明确记录瓶颈在渲染、CPU拷贝、编码、网络还是浏览器；只优化主瓶颈，不静默降SH/删高斯/取消保护校验。

### 10.3 验证方式

本地：CPU数学、文件原子性、API/mock进程/租约、前端模拟视频与输入状态、TypeScript/Vite build、Ruff、diff检查。不在本机做完整场景GPU验收。

远端：通过批准后的 Git 同步及dashboard独立任务运行有界验收，先确认 hostname/pwd 和当前任务、GPU、磁盘；不自动重启现有服务。部署会影响后端/前端时，列明精确服务与步骤单独授权；不改面板认证。

最终交付物包括可用功能、合同/启动说明、实际性能与失败证据、管理员确认的网络范围，以及仍未完成项。两项基本完成后再恢复保存的论文计划。

## 参考核查

- aiortc 1.15.0 元数据： https://pypi.org/pypi/aiortc/json
- PyAV 16.1.0 Python 3.10 wheel： https://pypi.org/pypi/av/16.1.0/json
- aiortc RTCConfiguration： https://github.com/aiortc/aiortc/blob/1.15.0/src/aiortc/rtcconfiguration.py
- aiortc H264 encoder： https://github.com/aiortc/aiortc/blob/1.15.0/src/aiortc/codecs/h264.py （已核查固定tag的 `av.CodecContext.create("libx264", "w")`）
- WebCodecs安全上下文： https://developer.mozilla.org/en-US/docs/Web/API/VideoDecoder
