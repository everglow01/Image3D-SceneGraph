# 云端 Gaussian 渲染合同

日期：2026-09-21。**本地实现已完成，并已在 i-94B8D131 部署认证 TURN 与云编辑后端；两份完整模型的服务器内部 CUDA／严格 relay 视频冒烟通过。SDP 跨网浏览器、持续交互与性能目标仍未验收。**

总体设计见 [gaussian-editor-cloud-plan.md](gaussian-editor-cloud-plan.md)，编辑接口见 [gaussian-editor-contract.md](gaussian-editor-contract.md)。论文计划继续暂停。

## 当前实现

- `cloud_render.py`：有限刚体 `camera_from_normalized`、pinhole K；16–1920 像素边长、总像素≤1920×1080。相机矩阵复制并只读。固定帧票据绑定源 SHA256、revision、camera_seq、相机 digest，5 分钟有效。
- native renderer 在独立 spawn 子进程运行；最多 3M 高斯，标准 SH3 PLY 只读加载一次。子进程持有 GPU 文件锁直到进程退出。复用既有 importer/model/render，不加载数据集 RGB，不执行训练或评价。
- `editor_session.py`：单活跃会话，loading/viewing/editing_frozen/closing/closed/error；加载、渲染、选择、保存、导出通过受控线程执行，CUDA 只在子进程。关闭等待加载和在途调用，取消 HTTP 不能提前释放线程使用的资源。已有会话（包括关闭中）不被新会话抢占。
- 每次最多一个编辑动作等候当前渲染完成；额外动作返回冲突，而非无界排队。相机输入只保留最新序号。可见 mask 变化时才传给子进程、重建活跃模型，不逐视频帧重建。
- heartbeat 每 5 秒；超过 30 秒未收到受控 HTTP 请求，或超过 10 分钟没有相机/编辑交互，释放会话。heartbeat 本身不延长实际交互闲置时间。媒体失败/断开有 30 秒宽限。已确认的文件式编辑不删除。
- `cloud_media.py`：延迟导入 aiortc/PyAV，不影响 base API 启动；视频只读，无摄像头或麦克风权限。DataChannel `camera` 为可靠、非顺序交付，序号拒绝旧相机；消息≤8192 字符，突发输入合并保留最高序号，以≤60Hz应用最新相机（不是直接丢弃限速窗口内的消息），浏览器约29Hz且只发相机变化。发送缓冲有界；页面隐藏暂停持续渲染。
- 视频目标30fps；运动最长边1280，停稳0.3秒后最长边1920（同时受总像素限制）。这是代码调度策略，不是实测帧率/500ms高清承诺。aiortc 默认协商软件编码，H264 路径用 libx264；没有接入 NVENC。
- 固定帧/预览采用请求触发的无损 PNG；不是 HTTP 连续图像流。编辑后先显示新 revision PNG，恢复观看关闭旧 PeerConnection，保留 PNG 到新连接实际呈现首帧，避免旧视频盖回删除结果。DataChannel 的渲染序号通知不是浏览器已呈现证明。

## GPU 互斥

启用 `IMAGE3D_CLOUD_ENABLED=1` 后，产品 worker 执行 Job/导航前取得与渲染器相同的 `output_root.resolve().parent / ".gpu.lock"`。云会话持锁时任务保持 queued；worker 每5秒重试并保留 `gpu_wait_reason`，不改训练参数、不抢占。

`execution.run_cancellable_command` 用 ContextVar 中的 lease FD 和 `pass_fds` 传给直接子进程。锁只关闭 FD、不显式 LOCK_UN，父进程退出不会解除活跃直接子进程继承的锁。产品 `execute_job(cancellable=True)` 和导航路径都走此边界。原 `.worker.lock` 仍只防重复 worker，不充当 GPU 租约。

worker 和渲染子进程取得锁后，先用只读 `nvidia-smi --query-compute-apps=pid` 探测外部计算进程；探测失败或有进程则保守拒绝启动。渲染请求会显示加载失败/资源占用，需要关闭后重试；不会偷偷启动半个模型等待。关闭云功能时，原 worker 不增加 GPU 探测或租约副作用。

OS 锁不会约束任意 root 手动 CUDA 程序，也不能强制让所有孙进程继承 FD；外部进程探测只是补充防线，不是强隔离保证。首次启用须保证旧 worker/研究任务已自然结束或按独立授权迁移，不能假设旧进程已经遵守新租约。

## 8082 与配置

**用户已确认管理员开放 8082/TCP 和 8082/UDP。** 剩余是部署与实测，不再是“协议未知”。

推荐同机认证 TURN：客户端通过8082接入，浏览器 `iceTransportPolicy=relay`；服务器 aiortc 使用 `iceServers=[]`，不调用公网 STUN/TURN。TURN listening port 不等于 UDP relay endpoints；管理员仍须配置同机 relay/peer 可达性、允许的 peer IP、受限 relay 范围以及内网/VPN来源策略。标准 aiortc 没有固定 ICE UDP 端口配置，不能只开放8082就宣称媒体必通。

后续获准部署时，后端进程配置：

| 配置 | 含义 |
|---|---|
| `IMAGE3D_CLOUD_ENABLED=1` | 显式启用云会话和产品共享 GPU 租约，默认关闭 |
| `IMAGE3D_EDITOR_ORIGINS` | 允许的前端 Origin，逗号分隔、含协议和端口，例如实际内网页面地址；默认仅 localhost/127.0.0.1:8081 |
| `IMAGE3D_TURN_HOST` | 管理员批准的同机 TURN 内网 IP/主机名；客户端端口固定8082 |
| `IMAGE3D_TURN_SECRET` | 与 TURN `use-auth-secret` 配对的≥32字符共享密钥；通过受保护部署配置注入，禁止写入源码、命令行或日志 |

HTTP mutation 只匹配显式 Origin allowlist，不信任任意 Host/转发头推导来源；兼容现有 Vite `changeOrigin:true` 代理，不修改原代理。请求还必须带 `X-Image3D-Editor: 1`。

浏览器只收到有效期1小时的 HMAC TURN 限时凭证，永远不收到共享密钥；当前会话凭证只在内存和 `X-Editor-Token` 请求头，不放 URL 或 browser storage。只读编辑文档/派生下载沿用已有内网资产边界，不冒充账户登录。部署必须阻止代理记录凭证请求头、完整 SDP 或响应体。

`pyproject.toml` / `uv.lock` 的可选 `cloud` 组锁定 aiortc1.15.0、PyAV16.1.0。远端已从获准 PyPI 官方源安装这两个包及锁定的10项媒体依赖；本地未安装。**现场完整 `uv sync` 的 dry-run 会替换两项既有 NVIDIA 库，故没有执行**；实际使用 `uv pip install --no-deps` 只新增12个锁定媒体包，保留现场 Torch2.3.1+cu121、gsplat1.5.3+pt23cu121、nvidia-cublas-cu12 12.9.2.10 和 nvidia-cuda-nvrtc-cu12 12.9.86。未来同步必须先 dry-run，不把锁文件未改 Torch 当成现场环境不会变化。媒体已实际导入并解码，但这不是所有编码器、ABI或许可证分发场景的验收；没有接入 NVENC。

## 已部署现场与服务器内部证据

用户已明确授权目标实例 `i-94B8D131`、项目 `/usr/local/3dgs_new/Image3D-SceneGraph`、依赖来源、受保护配置、项目后端重启及单GPU有界测试。源码只经 Git 同步；主功能提交 `f3a41aa`，基础relay冒烟脚本提交 `82d7113`，扩展验收脚本提交 `9ce320f`。

- `image3d-cloud.service`：沿用原 `127.0.0.1:8000`，保留原后端白名单配置（实际为 HOME、PATH、IMAGE3D_OUTPUT_ROOT），再加载 `/etc/image3d-cloud/backend.env`。配置目录0700、文件0600，密钥不进入仓库或日志。旧后端在确认无排队/运行任务与CUDA进程后优雅退出；未重启GPU面板。现有Vite仍在8081，代理能力接口返回 `cloud_available=true`。
- `image3d-turn.service`：DynamicUser独立运行，同机10.186.96.23的8082 TCP/UDP；relay范围49160–49179，peer白名单仅本机IP，其余地址拒绝。使用systemd `LoadCredential` 读取TURN配置；systemd249用 `${CREDENTIALS_DIRECTORY}`，不支持此处的 `%d`。
- coturn4.5.2-3.1~ubuntu22.04.1及4项必要运行库来自获准 Ubuntu 阿里云镜像，逐包SHA256与APT元数据匹配后隔离展开到 `external/coturn-cloud-v1`。首次APT下载遇过期索引404；刷新仅Ubuntu索引后发现常规安装会升级SQLite，故未执行系统包安装。现有SQLite库仍为3.37.2-2ubuntu0.7。默认 `coturn.service` 保持runtime mask，只有专用认证实例运行。没有修改防火墙。
- 冒烟目录：`outputs/experiments/cloud-editor-smoke-v1/`。run-v1误选直连，被严格断言判失败；run-v2的测试客户端提前关闭未使用host协议，向共享ICE接收队列注入EOF，造成超时。两次失败目录与记录保留，未作为通过证据。最终脚本将host协议从候选配对中移除、退出时才关闭，并同时断言SDP候选和最终nominated pair为relay。
- run-v3执行任务 `20260921-150846-9d25`、只读监控 `20260921-150846-e185` 均exit0。Project **1,483,682**与MCMC **2,999,158**高斯串行在GPU0完成真实加载、固定PNG、选区隔离预览、删除、撤销及保存；每份模型分别通过TURN/UDP和TURN/TCP解码5帧640×360视频，共20帧，DataChannel相机序号有推进。这里只用了相机路径与PLY，没有数据集RGB、训练或Test消费。
- 两份源PLY在测试前后以及独立复核中的SHA256一致。结束后无CUDA compute进程，两卡各约3MiB，共享GPU租约可重新取得。服务器103项Python CPU/mock回归、69项前端测试和生产构建通过，保留Starlette弃用和Vite大包警告。
- 报告：`run-v3/result.json`，SHA256 `8feb43b88dd3c8baa9139bb81065d79b8735e5d8d590c93fa553365e0f52ed74`。小报告与两张固定帧已下载至本地 `outputs/analysis/cloud-editor-smoke-v1/run-v3/`，未复制模型。可复用脚本：`scripts/smoke_cloud_gaussian.py`；它拒绝覆盖已有输出目录，不自动启动训练。

- 在新的run-v4目录完成扩展验收：执行 `20260921-151718-e8d3`、监控 `20260921-151719-7c05` 均exit0，用时约141秒。两模型均通过真实渲染器持锁时worker不执行内存中的排队探针、删除→撤销→重做→保存→PLY/ZIP导出→最终撤销恢复、1920×1080固定帧，以及再次各两种传输的20帧严格relay视频。没有提交真实训练任务。导出版本均为v00000003，各移除原始第0行的一个高斯；导出数分别1,483,681和2,999,157。ZIP所有条目CRC通过；独立流式对比证明导出PLY的全部属性和行序与源文件去掉第0行后的二进制数据逐字节一致。编辑结果保持未评价和不继承导航身份。
- 最终报告：`run-v4/result.json`，SHA256 `c9c03b8f06b71bca80d434e7bd17596620314704ef23510927ab988885c131e8`；本地报告及两张1080高清帧位于 `outputs/analysis/cloud-editor-smoke-v1/run-v4/`。原文件哈希独立复核未变，GPU进程已退出、共享锁可再次取得，后端与TURN仍active。完整导出保留在独立实验目录，服务器剩余约19GiB；没有删除旧实验或测试证据，下一次训练仍须重新检查其磁盘门禁。

## 剩余验收边界

用户当前经公司SDP远程办公，不是直连内网。本机访问8082的TCP/UDP实测超时；**服务器内部认证relay已通过，但不证明SDP路径获准或浏览器可连**。用户正向管理员确认SDP权限；无需据此修改服务器防火墙、改公网暴露或改为HTTP图像流。

每轮20帧是同机aiortc客户端的功能冒烟，**不是浏览器端到端验收，也不是帧率或时延基准**；两模型加载约16–17/32秒是单次观测，不代表稳定加载性能。真实训练运行/多轮排队、崩溃回收压力、1920高清持续视频、10分钟浏览器运行及20次重连仍未完成；已验证真实渲染锁阻断探针、完整模型编辑导出与高清固定帧，不能据此扩大为全部压力/性能目标通过。后续测试仍需明确GPU/时间预算并使用新输出目录。

剩余项目：
1. 完整模型显存峰值、跨视角 SH/坐标/颜色的定量一致性；现有固定帧只证明实际输出，不是质量基准。
2. 真正的渲染、CPU拷贝、编码、网络与浏览器呈现分段测量；不要把 render 耗时当端到端时延。
3. 训练/导航排队、外部进程保守阻断、关闭/崩溃后的 GPU 释放。
4. 浏览器10分钟运行、20次切换/重连，无PLY下载；运动中位≥25fps、P95帧间隔≤66.7ms、输入到呈现P95≤200ms均仍是待验目标。
