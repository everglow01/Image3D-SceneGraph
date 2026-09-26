# Gaussian 手动修剪数据合同

2026-09-26 本地编辑P4补充：实验入口接入正式查看器，旧viewer/K2明确转原始SH3 Spark，同源进出不重载；App切Job/variant/几何、切云端/renderer及退出共用未保存门禁，取消保留本地内容，确认后等待旧授权/Worker/显示释放。编辑禁用Walk，原模型与旧云安全边界不变。仅代码与静态检查完成，运行验收仍未执行，P5未开始。见 [本地P4](gaussian-local-editor-p4.md)。

2026-09-24 本地编辑P3补充：独立组件已接保存快照／版本、原请求重试、严格源绑定草稿、历史只读mask及CPU导出；保存中继续修改不被旧ACK标为已保存，冲突不自动覆盖，保护只随草稿保存。代码与静态检查完成，运行回归未执行；正式入口与App层离开保护仍属P4。见 [本地P3](gaussian-local-editor-p3.md)。

2026-09-24 本地编辑P2补充：新增独立源绑定二进制快照、短期单文档授权、云／本地共享文档锁及无GPU版本／导出通道；旧云HTTP继续禁止客户端mask，票据／预览门禁不变。仅代码与静态检查完成，回归未运行；前端保存、草稿与正式入口留在P3/P4。具体协议与部署边界见 [本地P2](gaussian-local-editor-p2.md)。

2026-09-24 本地编辑P1补充：在已完成的P0技术准入之上实现独立内存选集／历史、保护／隔离、Worker深度／穿透／源轴盒选、显示高亮与交互组件。仅静态检查，新增回归与GPU运行验收未执行；未接正式入口或保存API。旧云端票据、预览门禁与客户端mask禁入保持不变。本地操作语义另见 [本地P1](gaussian-local-editor-p1.md)。

2026-09-24 本地编辑P0补充：新增的独立源ID／仅alpha／Worker选择探针不构成产品编辑接口，未通过真实GPU准入前不得接入删除／保存。旧云票据、预览和客户端mask禁入限制不变。见 [本地P0](gaussian-local-editor-p0.md)。

日期：2026-09-22。状态：云修剪修复已部署，本地CPU/mock及浏览器DOM回归、两模型服务端真实gsplat GPU选择/保护/预览/删除撤销验收通过；**新版真实产品浏览器交互、跨视角背景质量与端到端性能尚未验收**，不能视为完整生产可用性通过。部署历史见 [部署证据](cloud-gaussian-rendering.md)。

完整批准计划：[gaussian-editor-cloud-plan.md](gaussian-editor-cloud-plan.md)。论文优化计划保持暂缓。

## 当前实现与未实现

`gaussian/editing.py` 提供 CPU 选择数学、受控源解析、文件式编辑文档、删除/撤销/重做、不可变版本与派生 PLY/ZIP 导出。`editor_session.py` 和 `backend/gaussian_editor.py` 已将其接到 HTTP、会话凭证及 renderer 接口；`CloudGaussianViewer.tsx` 提供独立界面。`apply(selected=...)` 仍是内部函数，客户端不能直接上传 mask/索引，HTTP 路由验证固定帧和服务器生成的 selection token。生产后端和同机认证TURN已部署，但浏览器经SDP的真实交互仍待验收。

没有训练、自动修洞、颜色编辑、语义分割或 Test RGB 消费。

## 来源与坐标

- 只从完成的 Gaussian Job/比较结果解析 manifest 中的资产；支持明确 variant ID，或 `scene_splat` / `scene_splat_vggt_filtered` 角色。
- 绑定 Job、variant/role、源 PLY 和 metadata 的 SHA256、Gaussian count、world_from_normalized。
- 当前只接收 normalized/arbitrary、SH3 标准导出，最多 3,000,000 Gaussians。
- 源 PLY 行号是 base Gaussian ID。全部 mask 长度等于原始 count；renderer 排序号和导出后行号不是源 ID。
- 源文件始终只读，即使发布 PLY 与实验原件是硬链接也不覆盖。
- 三维盒和 means 都在 normalized 空间；二维选择使用 rigid `camera_from_normalized` 的 +Z camera depth。前端使用 `diag(1,-1,-1,1) × Three.matrixWorldInverse × upright`，明确转换 Three 的−Z/Y-up到native +Z/Y-down；投影一致性已有CPU测试。三维盒输入保持原始normalized轴，不冒充摆正后的显示轴。

## 选择语义

- `select_box`：按 Gaussian 中心是否在闭区间轴对齐盒中选择。
- `select_polygon` 仍用于显式“穿透”或“指定深度薄层”：3–128 个像素顶点，原生固定帧尺寸内、面积≥1px²；`0.01 ≤ near < far ≤ 1e6`。按投影中心命中，不做遮挡检测；深度是 normalized arbitrary units，非米。
- 新默认“仅可见贡献”在渲染子进程使用gsplat投影/瓦片贡献索引和前向alpha透射率，寻找首个足够不透明的深度层：前层累计权重低于5%视作弱悬浮贡献并继续寻找主表层，达5%但低于50%则放弃该像素，不自动选后方；达到50%才视作可信。若用户要去掉这种弱悬浮项，应改用明确深度薄层并仔细预览，**不能把默认模式理解为背景绝不被选中**。必须在同一固定相机/版本下完成，失败/超时不回退穿透；源行号仍是唯一持久ID。CPU参考算法通过合成遮挡/半透明测试；真实gsplat CUDA首轮因过早锁定弱前层零命中，随后修复的v3在两完整模型单视角分别选中448/6,422个高斯并完成预览、删除/撤销。框外预览像素仍有变化，不能据此宣称精确可见拾取或背景零误删。
- 仅可见贡献不等于语义对象：同一个高斯可能横跨前景和背景、框内也可能有其他可见物；保守高斯选择依然必须预览及多角度复核。点击可信表层可取深度锚点，再手工调前后厚度；失败时提示改位置/手动设置。
- 可选 `radii` 是归一化单位的非负包围球半径；仅显式穿透模式可开启“扩大覆盖候选（可能多选背景）”，按包围盒保守相交，不是表面拾取。
- 当前会话“保护选中项”保存服务器生成的源ID mask，删除/预览排除受保护项；保护变更使旧选区及预览失效，跨重连保留、关闭会话后清除。不是永久物体锁，也不改变编辑schema。
- 自动叠色是图像合成参考，**不能单凭叠色删除**；服务器要求隔离或删除后预览才允许提交。保留50%确认、禁止清空模型、幂等ACK和撤销。
- 相机停稳后准备无损PNG、票据与camera_seq绑定，浏览器只在已呈现图片上执行 `freeze-prepared`，不在点击时重新采样滞后的WebRTC画面。导航更新/源/revision变更使票据失效；纯CSS缩放或窗口resize只更新图片与SVG的letterbox映射，不使同一固定图失效。
- API若未提供mode沿用旧穿透语义；新UI明确发送 `visible|depth|through`。默认可见模式已通过服务端真实GPU短时测试；实际产品浏览器交互及跨视角安全仍须独立验收。

## 编辑存储 schema 1

默认根为 JobStore output_root 的同级 `edits/`；构造 store 不创建文件。创建文档才写新目录：

```text
{edit_id}/
  edit.json
  .edit.lock
  operations/{uuid}.bin
  versions/v00000001/
    edit-manifest.json
    visible-mask.bin
    export/scene.ply
    export/export.json
    export/bundle.zip
```

mask 使用 `numpy.packbits(..., bitorder="little")`，长度固定为 `ceil(source_count/8)`，未用 padding bits 为0。选择原始二进制而非 NPZ 是为避免压缩/解压与格式歧义；文件只保存 bool 可见集合，不包含对象或pickle。3M mask 为375,000 bytes。

`edit.json` 记录源、单调递增 revision、history/cursor、幂等请求摘要。删除后 history 最多保留101个可用状态，即最近100次删除的撤销窗口；undo/redo本身增加revision。每文档最多1000个已确认操作，达到后明确报错，保存版本仍可读取。旧不可变mask文件和已保存版本不自动删除。

- `operation_id` 支持1–80个字母、数字、连字符、下划线。
- 相同ID且相同请求返回原ACK；相同ID不同参数拒绝；旧 expected_revision 拒绝。HTTP完整请求的SHA256以可选 `api_request_sha256` 字段和ACK原子持久化，重连到同一文档后也能重放已经确认的同一请求；不持久化会话凭证或原始票据。重放返回原ACK不代表它是最新revision，当前状态仍以文档/会话查询为准。
- 删除必须影响至少一个但不是全部可见Gaussian；删除超过当前可见数50%须明确 `confirm_large=True`。
- 确认操作在文件锁下先写不可变mask并fsync，再原子替换 `edit.json`。提交前失败的孤立mask不算已应用操作，保留供排障；不在异常处理中删除原始资料。
- Undo 后新删除截断当前可重做分支，但旧已保存版本仍不可变。
- `save_version` 用revision确定版本号，同revision重复保存复用原版本。
- 新文档、操作目录、版本和mask路径限制在编辑根内，拒绝符号链接逃逸。

## 派生导出

内部同步导出函数已通过一个有界后台导出任务接入API，不在ASGI事件循环执行；同版本重复导出复用已发布结果，不覆盖。失败的staging留作证据。下载用只读FileResponse，不在浏览器先把整个ZIP载入内存。

1. 验证版本、source、mask/count和源文件哈希。
2. 预检空闲磁盘至少 `3×源PLY大小＋1GiB`；不足报错，不清旧产物。
3. 从原始62列PLY中按mask筛行，不重新估计位置、旋转、SH或颜色。
4. 在独立staging写PLY、编辑专用export record和ZIP；源哈希再次一致后原子发布。
5. 已存在export拒绝覆盖；失败staging保留，不能作为成功资产显示。

`profile=manual_trim_export_v1`，包含源/可见mask/output哈希、保留/删除数量、原坐标变换。

- `quality_role=manual_edit_not_evaluated`：不复制旧PSNR、SSIM或evaluation hash为新模型质量。
- `navigation_status=not_inherited`：不宣称编辑后的碰撞或Walk有效。
- 编辑export record不是普通训练export schema的替身。前端通过独立编辑文档/版本列表显示，原Job manifest不修改；已保存版本可只读重开，继续修剪使用当前文档，导出成功的版本可在不占GPU时下载。

## HTTP 接口（本地已实现）

前缀 `/api`。所有修改请求要求显式Origin allowlist和 `X-Image3D-Editor: 1`；JSON严格拒绝未知字段，body≤96KiB。会话接口另要求 `X-Editor-Token`，由创建响应获得且只保存在页面内存。部署配置见 `cloud-gaussian-rendering.md`。

| 方法与路径 | 用途 |
|---|---|
| `GET /gaussian-editor/capabilities` | 核心/媒体能力及不可用原因 |
| `POST /gaussian-edits` | `{job_id, variant_id?, asset_role?}` 创建独立文档，不启动CUDA |
| `GET /gaussian-edits?job_id=...` | 列出源Job的文档，界面再按variant/role过滤 |
| `GET /gaussian-edits/{edit}` | revision、可见数、undo/redo、版本及exported状态；不公开内部请求注册表 |
| `POST /gaussian-render-sessions` | `{edit_id, version?}`；202返回loading及会话凭证，已有会话409 |
| `GET /gaussian-render-sessions/{session}` | 状态/错误/当前revision，同时刷新heartbeat |
| `DELETE /gaussian-render-sessions/{session}` | 等待本会话关闭；最近关闭会话的同凭证重试幂等 |
| `POST .../{session}/prepare-frame` | viewing时准备静止高清PNG和票据；相机更新/晚到响应作废，不中断视频 |
| `POST .../{session}/freeze-prepared` | 仅锁定已准备且已显示的票据，进入editing_frozen，不重新渲染/传输PNG |
| `POST .../{session}/freeze-frame` | 兼容旧固定/编辑后重新渲染请求，产品首次冻结改走prepare/freeze-prepared |
| `POST .../{session}/depth-pick` | 基于当前固定相机的可信前层取目标深度；不确定时明确失败 |
| `POST .../{session}/protection` | 服务器选集加入/移除/清空会话保护，旧选集/预览立即失效 |
| `POST .../{session}/invalidate-frame` | 相机/版本切换可使旧票据/选择失效；单纯CSS resize不再调用 |
| `POST .../{session}/selection` | 票据、expected_revision、polygon/box/clear、`mode=visible|depth|through`、depth_range、coverage、combine；返回selection_token、选中数与可删除数 |
| `POST .../{session}/preview` | 票据/revision/selection_token及highlight/isolated/after_delete，返回PNG |
| `POST .../{session}/operations` | 票据/revision、operation_id、delete/undo/redo、selection_token、confirm_large；返回已提交ACK |
| `POST .../{session}/versions` | expected_revision，保存不可变版本 |
| `POST .../{session}/exports/{version}` | 202启动后台导出，最多一个在途任务 |
| `GET /gaussian-edits/{edit}/versions/{version}/export` | 导出状态；进程重启后按已发布文件恢复成功身份 |
| `GET /gaussian-edits/{edit}/versions/{version}/assets/{name}` | 只允许scene.ply/export.json/bundle.zip，路径和symlink检查 |
| `GET .../{session}/ice` | 限时TURN凭证；不返回共享密钥 |
| `POST .../{session}/offer` | SDP offer/answer，超时关闭本PeerConnection，不回退协议 |
| `POST .../{session}/resume` | 使票据失效、关闭旧媒体；客户端用新PeerConnection重连 |

selection和operations归在session路径而不是直接挂在edit路径，身份只由该session绑定的edit决定，客户端不能另传一个目标文档绕开校验。删除的ACK与刷新图像分开：已提交删除不会因下一帧失败重做；页面恢复读取已提交状态。

当前是受信内网、单活跃会话产品边界，不是多租户账户鉴权；只读列表/下载与现有Job资产一样对可访问内网API的用户开放。

## 本地验证

```bash
uv run --no-sync pytest tests/test_gaussian_editing.py tests/test_gaussian_editor_api.py tests/test_gpu_lease.py -q
npm --prefix frontend test
npm --prefix frontend run build
```

测试仅使用合成小PLY、CPU选择/文件操作和mock子进程；覆盖硬链接源不变、属性保真、100次撤销、分支、幂等、陈旧revision、原子提交失败重试、路径和mask损坏。另有双模型单卡真实GPU服务端v3验收：独立文档、固定图、32×32可见/穿透选择、会话保护、隔离/删除后预览、删除后撤销；源PLY哈希不变，见 [云渲染证据](cloud-gaussian-rendering.md)。这不是产品浏览器端到端、真实3M持续性能或多视角背景零误删验收。
