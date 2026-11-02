# Windows + WSL2 开发环境迁移手册

适用项目：Image3D-SceneGraph。编写日期：2026-11-02。

**目标：Windows负责桌面和浏览器，WSL2 Ubuntu 22.04负责开发，现有远端服务器继续负责训练。Ubuntu、项目、开发依赖和交换文件的主要存储放在D盘。**

本文是操作指南，尚未在目标Windows上执行。不要把“迁移本机”理解为重新部署服务器或重新训练。

## 0. 约定与最终布局

以下假定Windows为更新后的Windows 11、CPU为x86-64、D盘为本地SSD。使用其他盘符时统一替换D，不要只改部分路径。Windows 10需先确认可安装最新版WSL；ARM机器不能直接照搬本文的x86依赖环境。

| 内容 | 位置 |
|---|---|
| Ubuntu发行版虚拟磁盘 | `D:\WSL\Ubuntu2204\ext4.vhdx` |
| WSL交换文件 | `D:\WSL\wsl-swap.vhdx` |
| 项目在Linux中的路径 | `/home/owen/Image3D-SceneGraph` |
| Python虚拟环境 | 项目中的`.venv/` |
| 前端依赖 | 项目中的`frontend/node_modules/` |
| Windows侧备份 | `D:\Migration-backup\Image3D-20261102\` |
| 远端项目 | `i-94B8D131:/usr/local/3dgs_new/Image3D-SceneGraph`，不迁移 |

`/home/owen`等Linux目录实际存储在D盘的虚拟磁盘内，不是在C盘另存一份。WSL的Windows组件、Windows应用和部分系统临时文件仍会占C盘，不能做到系统盘零占用。

建议D盘为新开发环境留出至少30–50GB可用空间；这是余量建议，不是固定占用。另计备份和以后下载的本地资产。虚拟磁盘按使用量增长，显示的最大容量不等于已经占满；删除Linux文件后也不保证宿主文件立即缩小。

**原则：先备份，再恢复和验收，最后才决定是否停用旧Ubuntu。不要提前格式化旧盘。** 如果D盘与旧Ubuntu位于同一块将被重新分区的物理硬盘，备份必须再有一份在不会被格式化的外置介质上。

### 0.1 环境与权重：分别从哪里获取

WSL安装快，是因为它主要安装基础Ubuntu系统，并不包含本项目的Python依赖、CUDA、模型权重或数据集。**迁移不需要统一采用一种下载/拷贝方式：环境重装，代码和个人工作恢复，权重按需复用，远端训练环境不动。**

| 内容 | 推荐来源/方式 | 本次是否需要 |
|---|---|---|
| 项目代码 | Git；本手册用含`.git`的本机工作区备份保留完整现场 | 需要；未提交修改和个人文件不能只靠clone |
| Python、uv、Node | 官方安装渠道，沿用本手册指定版本 | 需要，在WSL中重新安装 |
| Python项目依赖 | 按`uv.lock`及项目包源配置安装，通常来自PyPI或明确指定的发布源 | 需要基础依赖，不手工逐个从GitHub下载 |
| 前端依赖 | 按`package-lock.json`用`npm ci`安装 | 需要，不复制旧`node_modules` |
| CUDA、Torch、gsplat、COLMAP | 保持远端现有环境；确需本机计算时再按项目兼容要求配置 | 当前不需要在本机安装 |
| 预训练模型权重 | 优先拷贝已有可信且版本正确的文件；缺失时再从官方指定渠道下载 | 本机仅开发、远端计算时不需要 |
| 已训练模型/展示资产 | 从已有本机备份或远端按需取得指定资产 | 本机需要查看结果时才取 |
| 完整训练checkpoint | 默认保留远端，只有确需恢复训练时才复制完整组件与谱系 | 当前不需要搬回本机 |
| 本机独有的数据、报告、实验记录 | U盘/移动硬盘安全备份恢复 | 必须保留，不能指望重新下载 |
| 远端访问凭据 | 加密备份恢复或重新配置 | 需要，但不能放Git、公开下载地址或聊天 |

**为什么环境重装、权重却可拷贝？** `.venv`、`node_modules`和`external/`中的编译产物可能包含绝对路径、解释器链接、原生扩展和特定动态库依赖；即使同为Ubuntu 22.04，也不能保证直接搬过去可用。权重通常是数据文件，只要模型实现、权重版本和加载格式匹配，不必因换电脑重新下载。不要将`.engine`等依赖硬件/运行时的编译产物当成通用权重直接复用。

### 0.2 按本机用途决定需要哪些模型文件

- **只写代码、调前端、操作远端训练：** 不迁入预训练权重，不安装本机GPU环境。真实模型后端显示不可用是预期状态，项目不会因为启动网页就自动下载权重。
- **在本机查看训练好的Gaussian模型：** 按查看方式取得必要的PLY/浏览器模型资产及元数据即可，不必下载原始数据集、预训练权重、Adam状态或完整训练checkpoint。
- **以后在WSL运行模型推理或训练：** 再配置对应的GPU依赖，并只取得该模型所需权重。顺序是：旧本机已有有效文件→移动硬盘拷贝；只有远端有→按需下载；没有可信可用副本→官方重新下载。

“官方渠道”不等于“全部从GitHub下载”：可能是模型作者的GitHub Release、官方Hugging Face仓库或指定下载地址。按项目安装脚本/文档选择匹配版本，遵守访问和许可要求，不以不明网盘同名文件替代。不要因为迁移而更新远端Torch/CUDA或重新执行整套依赖安装。

### 0.3 如何选择传输方式和确认完整性

- **U盘/移动硬盘：** 适合已有大文件与唯一的个人数据；大批量文件建议移动SSD，介质需支持大于4GB的单文件。保留原副本，恢复成功前不删除。
- **远端下载：** 只拿本机要用的指定结果或权重，不整批下载远端项目、`.venv`和全部checkpoint。恢复训练所需checkpoint不能只复制一个模型参数文件，应保留完整组件、分片和项目要求的谱系。
- **官方重新下载：** 适合重建依赖，或补齐缺失、损坏、来源不明的权重；已有可信且版本正确的数GB权重无需重复下载。

拷贝重要权重或大型备份后，可在两端分别运行`sha256sum 文件名`，比较摘要是否相同。如果官方提供摘要，也应与官方值核对。相同摘要只能证明字节一致，不能单独证明文件来源可信；尤其不要加载来源不明的可执行反序列化格式。

**第一轮迁移的最小目标：恢复代码、个人文件、规则和实验记录，重建基础Python/Node依赖，恢复gp，验证页面、mock接口和远端访问。可以不新增下载任何模型权重。** 大数据先备份在外置盘，是否恢复进WSL由后续实际用途决定；“暂不恢复”不是“不备份”。

## 1. 旧Ubuntu：盘点并备份

### 1.1 暂停本机写入，记录Git状态

关闭本机正在修改仓库/输出目录的开发服务和下载任务，保存编辑器内容。不要停止远端服务或训练。

在**旧Ubuntu终端**执行。先将下面`BACKUP`改成已挂载、空间足够的外置盘上的新备份目录，父目录须已存在，不能设在项目里面。该节命令在同一个Bash终端顺序执行；新开终端时先重新设置`PROJECT`和`BACKUP`：

```bash
PROJECT="$HOME/Image3D-SceneGraph"
BACKUP="/media/owen/你的备份盘/Image3D-20261102"

mkdir "$BACKUP" &&
cd "$PROJECT" &&
git status --short &&
git worktree list &&
git rev-parse HEAD > "$BACKUP/git-head.txt" &&
git status --porcelain=v1 -uall > "$BACKUP/git-status.txt"
```

若有项目目录以外的其他worktree，其中的工作也要单独备份。不要执行`git clean`、`reset --hard`或全量stash来“清理后迁移”。

本机检查时HEAD为`8792816`；实际迁移以保存的`git-head.txt`为准。当前未提交的`codex.md`、`docs/cloud-gaussian-rendering.md`以及个人报告/博客/草稿都必须保留，不能只clone后以为迁移完成。

### 1.2 备份工作区，包括Git和未提交文件

继续在同一个旧Ubuntu终端执行：

```bash
test ! -e "$BACKUP/project-workspace.tar.gz" &&
tar -czf "$BACKUP/project-workspace.tar.gz" \
  --exclude='./.venv' \
  --exclude='./frontend/node_modules' \
  --exclude='./frontend/dist' \
  --exclude='./outputs' \
  --exclude='./data' \
  --exclude='./datasets' \
  --exclude='./external' \
  --exclude='./checkpoints' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='.ruff_cache' \
  --exclude='.uv-cache' \
  -C "$PROJECT" .
```

任何命令报错都先停下，不把失败或“文件在读取时发生变化”的备份当成完整备份。

这个包保留`.git`、暂存区、已跟踪文件的未提交修改、未跟踪个人文档、项目`CLAUDE.md`、`.claude/`和`experiment/`，不需要为了迁移强行提交个人文件。环境和依赖稍后重建。

**这是本机工作站的离线备份/恢复，不用于上传代码到GPU服务器。后续远端代码同步仍只用Git。** 工作区可能含私有配置，备份只放自己的受控/加密介质，不公开上传。

### 1.3 大文件单独备份，不默认全部装进新WSL

2026-11-02本机目录体积约为：

| 目录 | 体积 | 处理 |
|---|---:|---|
| `.venv` | 6.7GB | 不迁移，重建 |
| `frontend/node_modules` | 185MB | 不迁移，重建 |
| `experiment` | 49MB | 已包含在工作区包，保留本地实验记录与证据 |
| `outputs` | 28GB | 单独备份；其中`outputs/analysis`约5.1GB |
| `data` | 5.8GB | 单独备份，按需恢复 |
| `external` | 4.8GB | 单独备份，开发机默认不恢复编译环境 |
| `checkpoints` | 4.7GB | 单独备份，默认不装回开发机 |

**不要假定本地这些文件在服务器上都有副本。** 若准备重装/删除旧Ubuntu，先备份被排除的数据目录。当前这台机器可用以下命令生成约43GB量级的独立包，备份盘应另有余量：

```bash
test ! -e "$BACKUP/large-local-data.tar" &&
tar -cf "$BACKUP/large-local-data.tar" \
  -C "$PROJECT" outputs data external checkpoints
```

如果实际存在额外的`datasets/`或其他唯一数据，也需纳入备份。这个大包可以长期留在外置盘，不必立即解压进WSL。

### 1.4 安全备份用户规则、工具与远端凭据

项目`.claude/`在工作区包里，但用户目录里的全局规则、技能和记忆不在其中。以下只备份指定配置，不复制整个用户目录、旧虚拟环境或登录缓存。

下面使用GnuPG加密，密码通过交互输入，不放命令行。先确认`command -v gpg`可用：

```bash
umask 077
set -o pipefail
paths=(
  .claude/CLAUDE.md
  .claude/settings.json
  .claude/skills
  .claude/projects/-home-owen-Image3D-SceneGraph/memory
  .local/bin/gp
  .config/gpu-panel/access
)
existing=()
for path in "${paths[@]}"; do
  if [ -e "$HOME/$path" ]; then existing+=("$path"); fi
done

test ! -e "$BACKUP/developer-config.tar.gz.gpg" &&
tar -czf - -C "$HOME" "${existing[@]}" |
  gpg --symmetric --cipher-algo AES256 \
    --output "$BACKUP/developer-config.tar.gz.gpg"
```

保存好加密口令。若没有GnuPG，先用现有的可靠加密备份工具完成同等备份，不要把GPU面板凭据明文放到共享盘。

- 需要完整历史会话时，另外安全备份对应`.claude/projects/`目录；上面只保留项目记忆，不含全部会话。
- 开发助手、插件在新环境重新安装和登录，不复制认证缓存或插件二进制缓存。已有全局/项目设置中的插件、hook、代理和绝对路径，在启用前检查。
- Git身份与认证也需恢复：沿用原姓名/邮箱，在新WSL重新配置登录或SSH密钥访问。不要把私钥或访问令牌加入仓库。

最后列出压缩包确认能读取，并在新环境恢复时解密验证配置包；不要在尚未验证备份时删除原系统：

```bash
tar -tzf "$BACKUP/project-workspace.tar.gz" >/dev/null
tar -tf "$BACKUP/large-local-data.tar" >/dev/null
```

把备份安全转移到Windows的`D:\Migration-backup\Image3D-20261102\`，或继续从外置盘读取。若介质是FAT32，它不能存放超过4GB的单文件，应使用支持大文件的介质；不要因此临时格式化有数据的盘。

## 2. Windows：安装WSL并把Ubuntu放到D盘

以下命令全部在**Windows管理员PowerShell**执行，不在旧Ubuntu终端执行。

### 2.1 安装运行组件，不自动安装默认Ubuntu

```powershell
wsl --install --no-distribution
```

提示重启时先重启电脑。之后执行：

```powershell
wsl --update
wsl --set-default-version 2
```

如果报告虚拟化不可用，检查任务管理器“性能→CPU→虚拟化”，再按电脑厂商指引开启VT-x/AMD-V。不要为安装WSL关闭防火墙或安全软件。

### 2.2 安装到指定位置

确认`D:\WSL\Ubuntu2204`没有已有发行版数据，再执行：

```powershell
New-Item -ItemType Directory -Force D:\WSL
wsl --install -d Ubuntu-22.04 --location D:\WSL\Ubuntu2204
```

首次启动创建Linux用户，建议继续使用`owen`，使项目路径与旧机器一致。Linux密码可以与Windows不同。

```powershell
wsl -l -v
```

确认`Ubuntu-22.04`对应`VERSION 2`。如果需要手动进入：

```powershell
wsl -d Ubuntu-22.04
```

若`--location`不被识别，先检查`wsl --version`和更新结果，不退回默认安装后手工拖动虚拟磁盘。若已经在C盘安装过同名Ubuntu，先确认里面是否有数据，再决定迁移现有发行版；**不要使用`wsl --unregister`来腾名字，它会删除该发行版的数据。**

## 3. Windows：把WSL交换文件放到D盘

在Windows用户目录创建或编辑`%UserProfile%\.wslconfig`，注意不是`.wslconfig.txt`：

```ini
[wsl2]
swapFile=D:\\WSL\\wsl-swap.vhdx
```

若已有`[wsl2]`，只补/修改其中的`swapFile`，不覆盖其他配置。不需要顺便设置内存硬限制。

保存所有WSL工作、退出进程后，在PowerShell执行：

```powershell
wsl --shutdown
```

再进入Ubuntu。该命令会关闭所有WSL发行版，不应在有本地任务运行时直接执行。Windows组件、下载临时文件及Windows应用仍可能占用C盘；这一步解决的是交换文件位置。

## 4. 新Ubuntu：恢复项目和配置

从这节开始，除特别说明外，都在**新WSL Ubuntu终端**操作。

### 4.1 安装基础工具

```bash
sudo apt update
sudo apt install -y git curl ca-certificates xz-utils gnupg \
  libgomp1 libgl1 libglib2.0-0
```

后三个库用于常见CPU图像/几何依赖的运行环境。此处不安装CUDA、Torch、gsplat、COLMAP或Docker，不升级远端依赖。

### 4.2 将工作区恢复到Linux文件系统

```bash
BACKUP=/mnt/d/Migration-backup/Image3D-20261102
PROJECT="$HOME/Image3D-SceneGraph"

test ! -e "$PROJECT" &&
mkdir "$PROJECT" &&
tar -xzf "$BACKUP/project-workspace.tar.gz" -C "$PROJECT"
cd "$PROJECT"
git rev-parse HEAD
cat "$BACKUP/git-head.txt"
git status --short
```

两份HEAD应一致，原有未提交修改和个人文件应仍在。用备份中的`git-status.txt`对照，不以工作区“干净”为验收要求。

**如果目录已经存在，停下来检查，不把备份覆盖到已有仓库上。** 此方式已保留完整`.git`，无需再clone一次，也不应重新初始化Git。

项目放在`/home/owen/Image3D-SceneGraph`，实际占用D盘的`ext4.vhdx`。不要将主工作区放在`/mnt/c`或`/mnt/d`的Windows普通文件夹内；备份包从`/mnt/d`读取则没有问题。

### 4.3 恢复指定开发配置

先解密到新的隔离目录，不直接覆盖现有用户设置：

```bash
CONFIG_STAGE="$HOME/migration-config-20261102"
set -o pipefail
test ! -e "$CONFIG_STAGE" &&
mkdir -m 700 "$CONFIG_STAGE" &&
gpg --decrypt "$BACKUP/developer-config.tar.gz.gpg" |
  tar -xzf - -C "$CONFIG_STAGE"
```

仅从这个自己制作的备份恢复配置，不执行来源不明的脚本。随后：

1. 将暂存目录中的`.claude/CLAUDE.md`、自定义`skills/`和项目`memory/`恢复到新用户目录对应位置。目标已存在时先比较合并，不覆盖新设置。
2. 全局`settings.json`先检查插件、hook和代理路径；重新安装所需插件后再合并，不直接复制整个旧运行环境。
3. 恢复`gp`和凭据，以下命令在目标尚不存在时使用：

```bash
mkdir -p "$HOME/.local/bin" "$HOME/.config/gpu-panel"
chmod 700 "$HOME/.config/gpu-panel"
test ! -e "$HOME/.local/bin/gp" &&
test ! -e "$HOME/.config/gpu-panel/access" &&
install -m 755 "$CONFIG_STAGE/.local/bin/gp" "$HOME/.local/bin/gp" &&
install -m 600 "$CONFIG_STAGE/.config/gpu-panel/access" \
  "$HOME/.config/gpu-panel/access"
export PATH="$HOME/.local/bin:$PATH"
```

不要`cat`凭据文件，不把token粘贴进聊天、shell参数或环境变量。恢复完成后，暂存目录含明文凭据，应在核对目标路径后仅清理这次暂存副本；加密备份保留到迁移验收结束。不要用宽泛的清理命令处理用户目录。

若Linux用户名不为`owen`，项目记忆路径键和原有本机绝对路径需要按新路径调整；不要修改历史实验记录中的远端路径或把历史SHA重算成新值。

## 5. 新Ubuntu：重建Python和Node环境

首次迁移沿用旧版本，不同时升级依赖。旧本机版本为Python 3.10.12、uv 0.11.16、Node 24.14.1、npm 11.11.0。

### 5.1 Python与uv

从uv官方安装源安装：

```bash
curl -LsSf https://astral.sh/uv/0.11.16/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
uv python install 3.10.12
cd "$HOME/Image3D-SceneGraph"
uv sync --frozen --python 3.10.12
uv run --frozen python --version
```

`--frozen`使用仓库已有锁文件，不重新求解升级。默认安装基础依赖和开发组；不加`--extra gpu`、`--extra cloud`或`--all-extras`。若旧shell导出了`UV_PROJECT_ENVIRONMENT`、`VIRTUAL_ENV`等，先清除不适用的旧路径，不能继续使用旧`.venv`。

### 5.2 Node与前端依赖

Node安装在WSL里，不使用Windows版`node.exe`。以下用nvm的固定安装版本；安装脚本来自nvm官方仓库：

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
export NVM_DIR="$HOME/.nvm"
. "$NVM_DIR/nvm.sh"
nvm install 24.14.1
nvm alias default 24.14.1
nvm use 24.14.1
node --version
npm --version
```

如需与旧机器完全一致，可执行`npm install -g npm@11.11.0`。然后：

```bash
cd "$HOME/Image3D-SceneGraph/frontend"
npm ci
```

不要搬旧`node_modules`，不使用`npm update`来解决迁移问题。安装失败时先看网络、代理和具体错误，不直接改锁文件。

### 5.3 Git与工具路径

```bash
cd "$HOME/Image3D-SceneGraph"
git config --local core.autocrlf false
command -v git
command -v uv
command -v node
```

这些开发工具应来自Linux路径，而不是`/mnt/c/.../*.exe`。在新WSL配置与原来一致的Git姓名/邮箱及认证；不要将含凭据的remote URL或配置打印到公共日志。可以用`git ls-remote origin HEAD`只读检查远端仓库访问，不会push或改工作区。

Windows和WSL不要交替使用两套Git/Python/Node管理同一工作区。正常开发全部从WSL进入。

## 6. Windows：配置编辑器

在Windows安装VS Code和Microsoft的WSL扩展。不必在Ubuntu里再安装一套Linux桌面版VS Code。

从WSL终端执行：

```bash
cd "$HOME/Image3D-SceneGraph"
code .
```

确认VS Code左下角显示`WSL: Ubuntu-22.04`，集成终端为Linux，Python解释器选择项目的`.venv/bin/python`。Python等工作区扩展安装到WSL端；VS Code会管理自己的WSL服务端文件。

Windows资源管理器可访问同一份项目：

```text
\\wsl.localhost\Ubuntu-22.04\home\owen\Image3D-SceneGraph
```

开发助手在WSL重新安装、登录，恢复上节的规则/技能/记忆。当前项目的`CLAUDE.md`和部分配置未受Git跟踪，因此不能省略工作区与用户配置恢复。

## 7. 新Ubuntu：最小迁移验收

### 7.1 CPU mock链路

默认本机不启用云渲染。新建独立输出目录，避免冒烟启动时读取旧Job状态：

```bash
cd "$HOME/Image3D-SceneGraph"
SMOKE_ROOT="$HOME/image3d-migration-smoke-20261102/jobs"
test ! -e "$SMOKE_ROOT" &&
IMAGE3D_CLOUD_ENABLED=0 IMAGE3D_OUTPUT_ROOT="$SMOKE_ROOT" \
  uv run --frozen python scripts/smoke_backend.py
```

该脚本使用mock后端，创建测试Job，不训练Gaussian，也不需要CUDA。成功时打印`ok job_id=...`。若失败，先修复这个具体问题，不直接启动全套GPU检查。

### 7.2 前端构建

```bash
cd "$HOME/Image3D-SceneGraph/frontend"
npm run build
```

### 7.3 启动开发页面

WSL终端一：

```bash
cd "$HOME/Image3D-SceneGraph"
IMAGE3D_CLOUD_ENABLED=0 IMAGE3D_OUTPUT_ROOT="$HOME/image3d-dev/jobs" \
  uv run --frozen uvicorn backend.main:app \
  --host 127.0.0.1 --port 8000 --reload
```

WSL终端二：

```bash
cd "$HOME/Image3D-SceneGraph/frontend"
npm run dev -- --host 127.0.0.1
```

Windows浏览器访问`http://localhost:8081`。接口可检查`http://localhost:8000/api/health`。Vite现有`/api`代理目标是`127.0.0.1:8000`，不需要为迁移改代理代码。

这里使用全新的本地输出目录，不把旧任务数据库直接交给自动worker。缺少COLMAP/VGGT/gsplat等真实后端显示不可用，是轻量开发环境的预期，不表示迁移失败。需要查看旧模型时只恢复必要资产；大型重建和数值验证仍在远端完成。

如果Windows打不开页面，先从WSL内用`curl http://127.0.0.1:8000/api/health`确认后端，再检查Windows到WSL的localhost转发、端口占用和代理，不为排障关闭全部防火墙。

## 8. 新Ubuntu：恢复远端只读访问

先连接公司内网/VPN，再在WSL中执行：

```bash
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}10.186.96.23"
export no_proxy="${no_proxy:+$no_proxy,}10.186.96.23"
gp health
gp ping
gp exec 'cd /usr/local/3dgs_new/Image3D-SceneGraph && hostname && pwd && git rev-parse HEAD'
```

预期主机为`i-94B8D131`，项目路径与上面相同。若`gp`不在PATH，先使用`$HOME/.local/bin/gp`，再检查shell初始化配置。

- Windows能访问VPN，不代表WSL的路由/DNS/代理一定正常；超时先排查此层。
- 不把面板token放进环境变量；旧`GP_BASE_URL`/`GP_TOKEN`环境变量会覆盖配置文件，要检查是否遗留，但不要输出取值。
- 不经公网隧道转发面板，不新增SSH或未认证端口，不改服务器防火墙，不自动`git pull`。
- 这些检查只读。迁移验收不启动训练、重跑前缀、下载权重或安装远端依赖。

## 9. 迁移后继续工作的边界

工作流保持：**WSL修改代码 → 必要本机轻量检查 → 按要求提交 → 手动Git同步服务器 → 在明确授权范围内运行远端实验。**

2026-11-02编写本手册时，应保留两个已有状态：

1. O单独续跑实现已提交到`8792816`，尚无本会话中的O远端运行成功证据。迁移不自动触发O，更不重跑V/M、A/B或公共前缀。
2. `image3d-turn.service`按用户要求可恢复地停止并mask，为同事让出8082。**不要因迁移本机、恢复配置或测试云连接而启动它。** 恢复必须等待用户通知，具体备份位置和现场状态见[云端渲染文档](cloud-gaussian-rendering.md)。本手册不再次查询或修改远端服务。

## 10. 完成条件与回退

满足以下条件即可认为开发环境迁移完成，无需追加训练验收：

- [ ] Ubuntu是WSL2，虚拟磁盘位于D盘；交换文件配置也指向D盘。
- [ ] 项目位于Linux的`/home/owen/`，不是Windows挂载目录。
- [ ] HEAD与备份一致；未提交修改、个人文档、项目规则、所需实验记录均保留。
- [ ] Python环境、Node依赖在WSL重建；锁文件未被顺手升级。
- [ ] CPU mock链路与前端构建通过，Windows浏览器能打开开发页面。
- [ ] WSL中`gp`能完成只读访问；凭据文件权限为0600、父目录0700。
- [ ] 远端源码、训练任务、8082停用状态未因迁移改变。
- [ ] 被排除的大文件已另行备份，确认唯一资产没有遗漏。

完成前保留旧Ubuntu和原始备份。新环境出问题时可回到旧Ubuntu继续开发，不需要回滚远端服务。不要在两台开发环境同时修改同一任务而不经过Git协调；停用旧环境、删除备份和压缩虚拟磁盘都不是本次迁移的自动步骤。

## 参考

- [微软：WSL基本命令，含安装位置](https://learn.microsoft.com/en-us/windows/wsl/basic-commands)
- [微软：WSL配置及交换文件](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)
- [微软：WSL文件系统使用建议](https://learn.microsoft.com/en-us/windows/wsl/filesystems)
- [uv安装说明](https://docs.astral.sh/uv/getting-started/installation/)
- [nvm安装说明](https://github.com/nvm-sh/nvm#installing-and-updating)
- [VS Code WSL开发](https://code.visualstudio.com/docs/remote/wsl)
