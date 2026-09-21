# 云端 Gaussian 渲染合同

日期：2026-09-21。**本地 API、编辑界面、WebRTC 接线和产品 GPU 租约已实现并做 CPU/mock 验证；远端未部署，真实媒体/CUDA/性能未验收。**

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

`pyproject.toml` / `uv.lock` 已添加可选 `cloud` 组：aiortc1.15.0、PyAV16.1.0。锁文件解析成功，未改变 Torch2.3.1/gsplat1.5.3，未安装这些新依赖到远端或本地运行环境。获授权后可使用项目文档既有 uv 流程加 `--extra cloud --extra gpu --inexact --locked`，但依赖解析不等于实际 wheel 编码/ABI/许可证分发验收。H264/libx264 与媒体二进制的分发条件仍需部署方确认。

## 本地证据与剩余验收

本地相关 Python 回归、前端 Node/mock 测试、TypeScript/Vite build、Ruff、diff 检查通过。测试用合成小 PLY、假 renderer/media，不绑定真实 ICE 端口，不执行 CUDA；前端 hook 模拟不是浏览器截图或真实视频验收。详细结果记录在 `codex.md` §17。

未执行：Git 提交/推送/远端 pull、安装和启动 TURN、修改防火墙、重启现有服务、真实模型加载/渲染/软件编码、浏览器 WebRTC 连通与性能测试。

后续远端验收须单独明确实例、目录、源 PLY hash、时间/GPU预算和允许的服务操作；先经 Git 同步代码，禁止源码包。验证内容：
1. 1.48M/3M 模型导入与显存、固定相机 SH/坐标/颜色一致性。
2. 真正的渲染、CPU拷贝、编码、网络与浏览器呈现分段测量；不要把 render 耗时当端到端时延。
3. 训练/导航排队、外部进程保守阻断、关闭/崩溃后的 GPU 释放。
4. 浏览器10分钟运行、20次切换/重连，无PLY下载；运动中位≥25fps、P95帧间隔≤66.7ms、输入到呈现P95≤200ms均仍是待验目标。
