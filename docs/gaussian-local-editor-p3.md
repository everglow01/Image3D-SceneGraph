# 本地 Gaussian 编辑 P3

日期：2026-09-24。状态：**前端持久化闭环代码与静态检查完成，运行回归／真实浏览验收未执行。** P1／P2 的运行验收仍未执行。P3 仍是独立组件，不是已接入正式页面的产品入口，不改变默认查看器或生产服务。

## 接入边界

`LocalGaussianEditor` 继续复用已加载的 `SparkPageViewer`，新增可选的 `documentBinding: {editId, metadataSha256}`，源 PLY SHA 与 count 来自现有 P0 源行校验。提供绑定后自动取得 P2 单文档授权并读回权威 mask，不出现手动“连接会话”步骤；没有绑定时保留仅内存编辑模式。

本阶段只消费既有编辑文档。Job／variant／role 的文档创建与选择、正式挂载、App 层切源与退出保护属于 P4，尚未接通。不得把 `sourceSha256` 或源身份相同视为允许忽略全局未保存保护。

- 初次读回基线期间暂停编辑输入，防止用延迟响应覆盖修改；失败后明确报告并允许本地编辑。请求最长等待 60 秒，不无限等待。
- 完成基线读取后的选择、隐藏、撤销、重做与保护不依赖网络。保存中继续编辑不被禁用；每份文档只允许一个在途网络动作。
- 组件换源／卸载串行等待旧初始化、在途保存、授权释放和 Spark 编辑适配清理；迟到结果不进入新编辑器。释放失败仍清理 Worker／显示资源，文档授权最终按服务端期限回收。
- token 仅存在控制器内存和请求头，不进入 URL、草稿、日志或浏览器存储。不创建云 renderer，不访问 render-session／ICE／offer，不占远端 GPU 租约。

## 保存与确认

`localGaussianEditing.ts` 区分用于交互失效的 revision、可见修改代次和内容代次。选集、相机与临时预览不被误当作可见内容修改；保护有本地历史，但不进入服务端修剪版本。

`localGaussianPersistence.ts` 持有服务器基线、已确认本地代次和至多一份待确认快照：

1. 点击保存时复制当前 visible，并冻结代次。实际隐藏服务器当前可见项超过 50% 时单独确认；不能由同时恢复其他行抵消该数量。
2. 核对授权和权威源／revision／mask；失效后重新授权也不跳过基线检查。检测到不同服务器状态时停止同步，保留本地修改，不自动合并、并集或覆盖。
3. mask 有变化才发二进制 PUT，使用固定 operation ID、expected revision 和确认参数。与服务器基线相同则直接保存／复用版本，不制造空操作 revision。
4. 获得快照 ACK 后调用 `local-versions`，随后再次核对权威 mask。只把提交时的可见代次标为已确认，不覆盖当前本地 mask，不清空 undo／redo。
5. 快照 ACK、版本响应或后续读回丢失时，保留原请求并显示“重试原保存”。重试不会把较新修改夹入旧操作；快照已确认时只继续版本／核对步骤。暂时的 409 也可能是在途超时操作，保留请求供再次核对，而不是另造请求。

保存后撤销再保存产生新的服务器 revision／版本，原不可变版本不变。页面刷新后重新打开文档读回权威 mask；上一页的临时撤销历史不恢复。

“当前可见状态已持久化”只描述可见 mask，不代表永久保存了保护锁。未下载的保护状态仍触发离页提醒。保存取消不会冒报上一个版本为此次成功。

## 草稿文件

手动“下载当前编辑草稿”生成 JSON；“导入草稿”在明确确认后替换本地 visible／protected，清空临时选集和历史。不会写入服务器；保存必须另行点击。

`image3d_local_draft_v1` 包含且仅允许以下顶层字段：

- `schema`：上述固定标识。
- `edit_id`：原编辑文档 ID；不能把同源但不同文档的 revision 混用。
- `source`：PLY SHA256、metadata SHA256、gaussian count，三个字段严格匹配当前源。
- `base_revision`：已知服务器基线 revision；尚未取得基线时为 null，不能据此自动覆盖服务器。
- `visible`、`protected`：规范 base64 的 little-endian packed mask；精确字节数、padding 为零，visible 非空。
- `camera`：同源标识下的 position／target／up／fov，或 null；有限数值、合法视场及非退化方向。相机是显示状态，不冒充源几何变换。

文件上限 2 MiB，先查 File.size 再读取文本；拒绝未知字段、错文档／错源、损坏编码、长度／padding、全隐藏和退化相机。没有 PLY、token、服务器路径或完整撤销历史。3M 的两个 packed mask 合计 750,000 字节，base64 约 1 MB，不复制完整模型。

- 网络不可达时，有效草稿可先恢复为本地状态，随后必须“重新核对草稿基线”才可保存。服务器 revision 不一致就停止同步，不能静默覆盖。
- 首次基线未知的草稿允许本地恢复与查看，但不能自动补一个新 revision 进行覆盖。须保留文件，重新核对／处理来源后再决定迁移；本阶段不做跨文档草稿迁移或冲突合并。
- 导入读取／网络等待期间若本地内容已变化，拒绝应用该导入，不覆盖新修改。
- 当前存在待确认保存时禁止导入，先用原请求确认结果。草稿不携带待确认网络请求；若保存未确认就刷新，重新打开按服务器权威状态恢复，较新草稿若基线已过期保持冲突，不伪装成已同步。
- 不使用 localStorage／IndexedDB 自动备份。浏览器原生 beforeunload 仅尽力提醒，不保证拦住崩溃、关机或所有导航。下载按钮只表示已请求浏览器下载，用户仍需确认文件确已保存。

## 历史只读与导出

P3 对 P2 增补 `GET /local-mask?version=v00000001`：仍需同文档 token，沿用源校验、文件锁及 `version_visible` 的长度／哈希／数量／padding 校验。响应头 revision 属于所请求不可变版本；无 version 时保持当前文档语义。不修改当前 revision，不增加 GPU 依赖。

历史查看把版本 mask 仅送入现有 Spark 显示适配，不替换当前编辑状态或历史。选择／删除／保护／快捷键及保存禁用，导航仍可用；回到当前编辑时恢复原本地状态。迟到选择结果作废，不下载／重解析 PLY、不创建 WebGL context。

导出按钮只导出选中的已保存版本，不隐式保存较新修改。启动和状态查询复用 P2 CPU 通道及云／本地共享的单在途导出预算；状态约每 1.5 秒查询，网络失败后停自动查询并提供手动检查／重试。只有完成的版本显示 PLY／ZIP 下载链接，直接走已有只读文件响应，不在浏览器完整缓冲 ZIP。

原始 62 列保真筛行、相对行序、空间预检、独立 staging、原子发布和源 SHA 校验均复用既有实现；本轮没有运行逐字节导出验收。质量／导航标识仍为 `manual_edit_not_evaluated`／`not_inherited`。

## 检查与剩余边界

本轮仅执行产品和变更回归源码的 TypeScript 类型检查、Node 类型擦除语法检查、修改 Python 的 AST／Ruff 与 Git diff 检查。独立回归类型检查需沿用项目 bundler 解析，并显式包含现有 Node 类型和 `frontend/src/types/gaussian-splats-3d.d.ts`；未安装依赖。

新增／扩展回归源码：

- `frontend/tests/localGaussianPersistence.test.ts`：保存中继续修改、undo 后新版本、重复保存、ACK／版本响应丢失、过期授权、CAS、离线草稿与严格结构、坏响应、只读 mask、导出和关闭期间授权。
- `frontend/tests/localGaussianSavePanel.test.ts`：保存等待时下载草稿、未保存提示、保护离页提醒、只读返回、导出链接和文件上限。
- `frontend/tests/localGaussianEditor.test.ts`、`localGaussianInteraction.test.ts`：绑定／不绑定生命周期、串行授权释放、历史显示不改当前状态、快捷键和旧选择拒绝。
- `tests/test_gaussian_editor_api.py`：历史 mask 只读、token／源身份、非法／不存在版本与当前 revision 不变。

**上述回归未执行，不能宣称通过。** 未运行构建、真实模型、GPU、远端操作，未 push／pull／部署；用户原有文档保留。P4 的正式入口／源切换／全局离开保护与 P5 的双模型、20 分钟资源和产品实机验收尚未开始；原有约 100 ms 导航表现不在本轮优化范围。
