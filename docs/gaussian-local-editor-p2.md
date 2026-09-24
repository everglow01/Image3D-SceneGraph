# 本地 Gaussian 编辑 P2

日期：2026-09-24。状态：**保存协议代码与静态检查完成，运行回归未执行。** P1 的运行验收仍未执行；P0 历史证据不覆盖本协议。本阶段只接后端 CPU 文档通道，不连接前端保存按钮、草稿恢复或正式页面入口。

## 协议边界

旧云端 `selection`／`operations` 仍拒绝客户端 mask，固定帧、服务器选集和预览门禁保持不变。新增本地通道允许客户端提交完整可见 mask，服务端只验证源身份、结构、revision 和删除门禁，**不证明客户端选择算法正确，也不验证临时保护集合**。保护不写入已保存修剪版本。

所有地址以 `/api/gaussian-edits/{edit_id}` 为前缀；文档通过既有 `POST /api/gaussian-edits` 创建，源身份通过既有文档 GET 获得。客户端不能提交源文件路径。

| 方法与后缀 | 内容与用途 |
|---|---|
| `POST /local-authorization` | JSON `{ply_sha256, metadata_sha256, gaussian_count}`；返回 201、单文档 token、相同 identity、当前 revision／visible_count、剩余授权秒数 |
| `DELETE /local-authorization` | 关闭授权，等待正在执行的 CPU 操作结束再释放文档锁 |
| `GET /local-mask` | 返回 `application/octet-stream` packed mask 和权威 revision／源身份响应头；P3 增补可选 `version=v00000001`，只读获取不可变版本 mask |
| `PUT /local-mask` | 二进制完整可见 mask；控制字段为下述 query 参数；返回已提交 ACK |
| `POST /local-versions` | JSON `{expected_revision}`；保存或复用该 revision 的不可变版本 |
| `POST /local-exports/{version}` | 返回 202，复用云端单在途 CPU 导出协调器 |

导出状态及文件下载继续使用既有 `/versions/{version}/export` 与 `/versions/{version}/assets/{scene.ply,export.json,bundle.zip}`，不另建下载或导出系统。

### 授权与身份

- 修改请求继续要求 `IMAGE3D_EDITOR_ORIGINS` 明确 allowlist 和 `X-Image3D-Editor: 1`。除首次授权外，上述接口还要求 `X-Editor-Token`；GET mask 同样需要 token。token 不进入 URL、持久文档或草稿；响应均为 `Cache-Control: no-store`。
- 每次授权固定有效 600 秒，不因读取自动续期；每个服务实例最多 64 个本地授权。不占项目 `.gpu.lock`，不创建 renderer，不调用 GPU 空闲探测，不检查 CUDA／aiortc／TURN 可用性。
- 授权绑定文档以及 PLY SHA256、metadata SHA256、源 count。打开、读取、提交、版本保存和导出启动均从文档解析原始源并复核冻结身份；不信任客户端路径，不重新构造模型。
- 失效后需重新授权并核对当前 revision。授权响应丢失时，未知 token 不允许抢占；可等待该短期授权到期。关闭后重复 DELETE 不保留旧 token 的成功 ACK，可能返回 403；已提交快照的幂等记录则持久保留。
- 这是现有受信内网、单服务产品边界，不是多租户账户鉴权；取得可读源身份不是证明用户拥有该模型。已有只读文档／导出下载权限边界不变。

### 二进制提交

`PUT /local-mask` query 参数：

- `ply_sha256`、`metadata_sha256`：各 64 位小写十六进制，必须与授权一致。
- `gaussian_count`：1–3,000,000，与授权一致。
- `expected_revision`：非负整数，不超过 JavaScript 安全整数上限。
- `operation_id`：1–80 个字母、数字、下划线或连字符。
- `confirm_large`：默认 false；确认本次实际隐藏当前可见项的比例超过 50%。

拒绝未知 query 字段。query 中的整数／布尔通过字段级转换解析；JSON 控制接口继续严格验证类型并拒绝未知字段。

仅该 PUT 路由允许最大 **512 KiB**，其余修改请求仍最多 **96 KiB**。必须使用精确 `Content-Type: application/octet-stream`，不接收 JSON 索引数组或压缩 body。逐个读取实际流块，先检查累积长度再追加，不依赖 `Content-Length`。长度必须为 `ceil(count/8)`，编码为 little-endian packbits，padding 位必须为零；3M 时为 375,000 字节。

GET mask 通过同一个存储文件锁读取 revision 与数据，返回：

- `X-Edit-Revision`
- `X-Visible-Count`
- `X-Source-Sha256`
- `X-Metadata-Sha256`
- `X-Gaussian-Count`

这些响应头用于同源前端读取；本阶段不放宽 CORS。客户端不能将旧提交的 ACK revision 当作当前权威 revision。

## 文档互斥与生命周期

每份文档新增 `.writer.lock`，本地授权与云渲染会话均通过同一 `FileLease` 获取并持续持有；提交时在有效授权和操作锁内执行。同文档的本地／本地、云／本地打开冲突返回 409，不终止旧持有者；不同文档可以共存。为避免旧云端已加载状态过时，云端历史只读版本会话也保守持有此锁。

- 云会话从开始加载到完成关闭均持锁，包括错误待关闭状态；不会在关闭 renderer 前释放。
- 本地每文档最多一个在途操作，其余请求返回 409，不新增无限队列。过期后拒绝新操作；已开始的 CPU 写入继续完成，完成前保留授权占用和文件锁。
- 定时清理每 5 秒处理已过期且空闲的授权，打开新授权时也会清理本服务过期项。关闭等待操作结束；HTTP 取消不取消底层线程，重复取消也不能提前释放锁。
- 文件锁可协调共享编辑目录的升级后服务进程；token／过期清理仍是进程内状态，重启后需重新授权。不是分布式租约。旧版服务或绕过协调器直接调用存储的脚本不会自动遵守新 `.writer.lock`，不得与新服务混跑同一文档。
- 导出仍是当前 `EditorSessions` 内云／本地共用的一个在途任务，同版本复用状态，不同文档／版本返回 409。未扩展成跨服务实例的全局导出调度；生产仍按现有单服务部署边界使用。

## 快照与版本语义

`GaussianEditStore.submit_snapshot` 是独立于旧 `apply(delete/undo/redo)` 的入口。源范围内的行可以重新变为可见，因此“保存后本地撤销，再保存”产生新 revision，不回退旧版本。

- 在既有 `.edit.lock` 下读取当前状态，验证源、长度、padding 并重新统计数量。禁止全隐藏；`当前visible ∩ 非新visible` 超过当前可见数 50% 时要求确认，不能用同时恢复其他行抵消实际删除量。
- 快照追加不可变 mask，截断 redo 分支，保留最多 101 个历史状态；revision 单调增加。每文档最多 1000 个已确认操作，达到上限后仍可重放原 ACK 和保存版本。
- 与当前 mask 相同的新 operation_id 也记为一个快照 revision；调用方应避免无意义重复保存。相同 operation_id、相同规范化控制字段和 mask 哈希返回原 ACK，不另占操作；不同请求复用该 ID 返回 409。
- 幂等摘要含 `local_snapshot_v1`、源身份、expected_revision、confirm_large 和二进制 mask SHA；token 不写盘。重新授权不影响已保存的同请求重放。
- 新 mask fsync 后原子替换 `edit.json`，ACK 与请求摘要一起提交。提交前失败保留孤立文件作为证据，重试同请求不会把它误认为已提交。CAS 冲突不合并、不覆盖；浏览器保留本地状态和恢复逻辑属于 P3。
- 保存与导出复用现有不可变版本、磁盘空间预检、独立 staging、源前后哈希和原始 62 列保真筛行，不从 GPU 数据重建。质量／导航继续为 `manual_edit_not_evaluated`、`not_inherited`，原 Job manifest 和源硬链接不改写。

## 检查与验收状态

仅执行修改 Python 文件的 AST、Ruff 及 `git diff --check` 静态检查。没有执行 pytest、前端回归、生产构建、真实模型 GPU 或远端操作，没有 push／pull／部署或安装依赖。

回归源码扩展于 `tests/test_gaussian_editing.py`、`tests/test_gaussian_editor_api.py`：源绑定、padding／长度／全删／大比例、恢复与旧版本不变、CAS／重放／原子失败、历史与操作预算、CPU-only 保存导出及保留行逐字节、跨文档 token、body 实际上限、旧云 mask 禁入、过期／加载关闭竞争、重复取消时的锁占用及共用导出预算。**这些用例尚未运行，不能宣称通过。**

后续 [P3](gaussian-local-editor-p3.md) 已实现独立组件的前端保存／导出、ACK 丢失恢复和草稿代码，仍未运行验收；本文的 P2 历史验证状态不变。P4 才接正式页面与全局离开保护；P5 完整验收仍独立。当前 P1／P2 运行验收也仍待执行。
