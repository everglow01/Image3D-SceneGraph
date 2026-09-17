# 新 num4 视频 4K 双臂 Train-only 实验

## 授权与目标

2026-09-15 用户同意先做 4K/内存适配，再启动 Project v7 与 MCMC 两个实验任务。仅在 i-94B8D131 的 `/usr/local/3dgs_new/Image3D-SceneGraph` 执行；新代码通过 Git 同步。原视频为 `/usr/local/3Ddataset/num_4_room_new/DSC_2385.MOV`，现场元数据为 601.3007 秒、3840×2160、30000/1001 fps、HEVC Main/BT.709，11,297,479,872 字节。正式准备绑定完整源哈希，不以文件名替代身份。

目标是比较新场景上两个方法配置在原生 UHD 目标分辨率、相同优化/相机采样预算下的最终效果，以及各自 Train-only 的增量。不是等 Gaussian 容量、等耗时、跨场景统计结论或默认推广实验。

## 共同准备

- 独占新根目录 `outputs/experiments/num4-new-4k-train-only-v1`，保留失败/started 文件，禁止覆盖和盲重试。
- 原视频同文件系统硬链接，不复制第二份大视频，也不改源视频。
- `standard_v2`，6 fps 候选、4 fps 基础、最多5 fps初选；601.3007 秒对应基础预算2,405、初选上限3,007。质量拒绝和注册恢复后的实际计数以合同为准。
- 普通 COLMAP：SIFT/Brute-force、sequential_loop、default_v1 verification、shared OPENCV、Incremental；保留现有1,000种子、注册扩展与缺口恢复，不降低12/70%/80%门槛，>2秒空洞仍为软警告。
- 目标最长边3840，去畸变实际尺寸逐项记录，禁止静默降至1280/3072；坐标仍为归一化任意单位。
- 只准备一次几何、相机、划分、初始化与replay。`--prepare-only`完成这些阶段后返回，不启动主训练。
- `protocol.json`冻结代码、源、dataset/replay/config文件哈希、划分计数、实际图像尺寸与预算；启动每臂必须传确切协议SHA，重新验证replay。

## 两个正式训练任务

Project先运行，完成后MCMC运行；每臂均使用两张L2，不是一张卡一个模型同时训练。

| 配置 | Project v7 | MCMC |
|---|---|---|
| 数据、几何、初始化、划分 | 同一replay | 同左 |
| 目标最长边 | 3840 | 3840 |
| 主训练 | 30,000 updates / 60,000 camera samples | 同左 |
| seed | 20260729 | 同左 |
| recovery-prune | on | off |
| Gaussian容量 | 原Project策略，无新增硬cap | 冻结全局3M cap |
| Selection | 原Validation选模 + SOR | 同左 |
| SOR | k=30、std=2、opacity band<0.05 | 同左 |
| 末期 | Train-only 2,000 updates / 4,000 samples | 同左 |

Train-only从各自Selection模型开始，而非必然从30k最后模型继续。沿用 `train_only_control_v1`：fresh Adam、固定拓扑、position最终LR、其余LR 0.1倍、SH3、源L1/SSIM/clamp；无strategy、noise、regularizer、prune/reset或Validation优化。固定2k后的模型是最终产物，不再按追加阶段Validation挑步数。

每臂顺序：train → SOR → Selection Validation → Train-only → lineage-bound export。SOR/评价/导出失败即该实验任务失败，不静默跳过。保留Selection与末期模型/评价、完整进度、模型/评价哈希、检查点及末期导出。中间Validation PNG禁用，但8个原定验证时点的全视角指标、选模和末次完整PNG保持；Selection与Train-only分别保存完整评价预览。

## 内存、磁盘与失败边界

- 512 MiB解码图像缓存/collection/rank；相机元数据常驻，图像按需加载。此预算不包括当前图像、CUDA张量、模型、Adam、临时导出数组和文件系统缓存。
- 未修改采样顺序算法、随机种子状态或checkpoint字段。缓存可丢弃且不占用随机数。
- 准备前至少40 GiB空闲磁盘，每臂启动前至少20 GiB，每个后续子阶段前至少8 GiB；主机可用内存至少12 GiB，必须恰有两张可见CUDA卡。阶段资源门槛不是运行峰值保证。
- 现场样本4K JPEG约1.54–2.06MB，PNG约5.72–7.46MB；这是五帧估算，最终以实际文件和资源日志为准。
- 不自动清理旧实验、不下调分辨率/帧密度/模型cap、不增加swap或改驱动。资源不满足、OOM、几何门槛失败应保留证据并停止，不能伪装成完整成功。
- 调度使用独立dashboard启动任务（flock/once）与只读监控；前一阶段失败不自动重试，不进入后一臂。

## 评价边界

全部报告使用本次新数据集的冻结Validation，不挑选有利视角；原生raw/display PSNR、SSIM及逐视角差值分开，记录退化数量、分位数、Gaussian数、耗时、主机/显存与产物大小。Validation未参与Train-only优化，但曾用于主训练选模，不是全新Test证据。

Test不优化、不解码评价、不创建消费标记；合同验证可能读取Test文件字节做哈希，这不等于Test tensor加载。旧场景已消费的Test不参与本轮。无导航、默认推广或线上服务重启。实验任务不是伪装成完整产品Job的manifest，不自动发布前端条目。

## 执行入口

```bash
.venv/bin/python scripts/run_video_4k_comparison.py prepare \
  --source /usr/local/3Ddataset/num_4_room_new/DSC_2385.MOV \
  --output-dir outputs/experiments/num4-new-4k-train-only-v1

.venv/bin/python scripts/run_video_4k_comparison.py run-arm \
  --output-dir outputs/experiments/num4-new-4k-train-only-v1 \
  --arm project --protocol-sha256 <冻结后核验的SHA>

.venv/bin/python scripts/run_video_4k_comparison.py run-arm \
  --output-dir outputs/experiments/num4-new-4k-train-only-v1 \
  --arm mcmc --protocol-sha256 <同一个SHA>
```

以上命令须置于授权远端dashboard任务，设置 `CUDA_VISIBLE_DEVICES=0,1`；不是本机执行指令，也不包含Test授权。

## 2026-09-17 重拍视频：原生1080p / 3,500种子

用户确认使用 `/usr/local/3Ddataset/num_4_room_new/VID20260917094108.mp4` 启动全新实验，并明确同意原生1080p和初始Mapper最多3,500种子；不实现通用按时长自动加种子的默认策略。现场视频流约848.447秒、1920×1080、HEVC，带90度显示旋转；沿用已有自动方向处理，目标最长边1920，不插值到4K。正式准备记录完整源SHA、旋转和实际去畸变尺寸，使用独立 `num4_retake_1080_train_only_v1` 身份，不冒充UHD实验。

- 独立根：`outputs/experiments/num4-retake-1080-seed3500-train-only-v1/`，调度位于同名 `-dispatch/`。全新取帧、特征、匹配、几何，不复用旧视频数据库/相机，不覆盖旧目录。
- 3,500是本次显式上限，约保持先前601秒/2,500种子的时间密度，不保证几何成功。实际数量受可用帧数限制；继续使用既有规则：基础池足够则从基础池均匀选，否则从全部已选帧均匀选。约848秒的基础预算约3,394张，不能误称为保留全部基础帧再补足3,500。全部已选帧参与原有匹配，非种子仍走后续注册扩展。
- 其余沿用上述合同：standard-v2、SIFT/Brute-force/sequential_loop、共享OPENCV、Incremental及既有精确崩溃恢复、原位姿与12/70%/80%门槛；>2秒缺口仅软警告。几何通过后冻结同一replay，Project先、MCMC后，各两卡30k+2k Train-only、相同SOR与Validation、无Test消费。
- 40/20/8 GiB磁盘、12 GiB主机内存资源门槛不变。现场释放后约64 GiB空闲不代表峰值保证；失败保留证据，不自动降级、清理或重试。
- 原4K fresh入口仍默认3840/1,000，`prepare-reuse`仍绑定原4K/2,500；产品/API默认不变。本研究任务不自动发布为前端产品Job，不重启既有服务。

```bash
.venv/bin/python scripts/run_video_4k_comparison.py prepare \
  --source /usr/local/3Ddataset/num_4_room_new/VID20260917094108.mp4 \
  --output-dir outputs/experiments/num4-retake-1080-seed3500-train-only-v1 \
  --longest-edge 1920 --mapper-seed-limit 3500
```

冻结后使用本文件原有 `run-arm` 命令，传本次新根和同一协议SHA；不得沿用旧模型协议或越过几何失败。
