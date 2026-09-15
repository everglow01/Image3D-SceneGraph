# 新视频 2,500 帧 Mapper 种子对照

## 2026-09-15 冻结范围

用户明确授权：按2,500张做对照，几何通过后继续Project v7和MCMC两个4K/Train-only实验。它是既有1,000种子失败后的独立研究臂，不覆盖原失败目录，不改变生产默认。

- 旧根：`outputs/experiments/num4-new-4k-train-only-v1/`。
- 新根：`outputs/experiments/num4-new-4k-seed2500-train-only-v1/`；调度位于同名`-dispatch/`。
- 原视频SHA256：`eaf1e6a53549ac4d9da9183a5def07a7439ebcc250629576bfc794cd2d4b538a`。
- 旧选择JSON SHA256：`bc0714258d74053bdd14d9edc24278af8c997cd9d4f274d11d52ac9459789041`。
- 旧匹配数据库SHA256：`019b95d09d90da837847037ee04e7b9bbfdcab260e67b3e6fa4afb3a78d78808`，实际运行时必须与旧恢复诊断和现场文件一致。
- 输入仍为全部2,787张已接受4K JPEG（2,405基础+382自适应）；不重新取帧，不重新执行这些帧的初始特征提取或匹配。

## 唯一初始求解变量

初始Mapper种子上限由1,000改为2,500。沿用现有确定性规则：基础帧数量不少于预算时，在基础帧池中均匀选择；否则在全部已接受帧池中均匀选择。索引为`round(i*(n-1)/(target-1))`，保留时间首尾、唯一图像名。

本次基础帧2,405少于2,500，因此在全部2,787帧中均匀选2,500帧，包含部分自适应帧。它不是“保留全部基础帧再补95张”，也不保证是旧1,000种子的严格超集。此处只参数化既有规则，不加入图连通性评分、匹配阈值调优或人为挑图。

SIFT/Brute-force、sequential_loop、default_v1 verification、shared OPENCV、Incremental、seed 0、8线程、v2 Mapper/BA参数和12/70%/80%门槛均保持。初始注册率按各自种子集合计算（旧1,000、新2,500），不能把分母差异隐藏。相机位姿健康、注册扩展、Global/core恢复、去畸变和最终BA仍走原合同；初始模型不满足门槛则停止。既有局部缺口恢复如在后续触发，仍可按原规则新增局部候选的特征/匹配，这不等于重跑既有2,787帧的前端。

## 复用与完整性

- 旧图像只读校验逐文件SHA后硬链接到新workspace；原视频也只硬链接。可能被后续恢复更新的`selection.json`必须复制，禁止与旧选择记录硬链接。
- 原SQLite要求非空文件、无非空WAL，必须通过冻结文件哈希；以`mode=ro`连接源，通过SQLite backup写入新数据库，禁止硬链接可写DB。
- 除`v2_mapper_seed_count`外，新旧前端合同字段必须相等；图像名集合必须完全一致。
- 复制前后逐表计算有序行内容哈希与行数，覆盖所有非SQLite内部表（包括camera/image/feature/match及rig/frame元数据）；复制后重新核验原数据库字节哈希。源连接和目的连接均显式关闭。
- 新前端合同标记`retained_feature_database_v1`，记录源DB/前端哈希、快照哈希和表摘要。SQLite快照文件字节哈希不必与源文件相同，内容一致以逐表摘要核验。
- 几何成功后，再次核验旧源记录、数据库及旧图像未改变，冻结新replay和协议。新代码身份单独记录，不声称与旧运行代码哈希相同。
- 旧取帧计时JSON原样保留，`video_preparation_reused=true`说明它是继承证据，不能把旧686秒取帧耗时当成本轮重新执行的耗时。

## 资源与后续训练

复用准备前要求至少30 GiB空闲磁盘，而非fresh准备的40 GiB，因为不会再存一份原视频和初选图像；这不是运行峰值保证。每臂启动前仍要求20 GiB、后续阶段8 GiB、主机可用内存12 GiB和两张无其他计算进程的CUDA卡。资源不足不清理旧数据、不降低分辨率或门槛。

仅几何通过并冻结协议后，继续[原双臂训练合同](video-4k-train-only-comparison.md)：Project先、MCMC后，各双卡30k主训练+2k Train-only，默认1280不变、实验目标3840，保留所有Validation指标、限制中间PNG。无Test消费、导航、默认推广或服务重启。两个任务有独立once/exit/lock和只读监控；依赖失败时后一臂退出而非重试前一臂。

入口：

```bash
.venv/bin/python scripts/run_video_4k_comparison.py prepare-reuse \
  --source-experiment outputs/experiments/num4-new-4k-train-only-v1 \
  --output-dir outputs/experiments/num4-new-4k-seed2500-train-only-v1
```

随后使用相同脚本的`run-arm --arm project|mcmc --protocol-sha256 <新协议SHA>`。这些命令仅在授权远端dashboard任务执行；不是本机GPU运行指令。

## 2026-09-15 空模型 BA 崩溃修复续跑

首个2,500种子任务在Mapper处理后续小子模型时，触发`ba_config.NumImages() >= 2 (0 vs. 2)`并以SIGABRT退出。最大已保存Incremental模型注册1,251/2,500张（50.04%），有146,321个三维点；它尚未通过完整健康验收，不能当作合格几何。该次没有执行Global recovery、冻结replay或启动Gaussian，也不构成完整的2,500种子质量终评。

用户随后授权修复并在几何通过后继续双臂。本次只修复项目runner的已知异常恢复边界，不修改或重编译外部COLMAP：仅视频Gaussian初始Incremental Mapper的负SIGABRT返回码和上述零图像BA错误同时匹配时，允许已有二进制候选进入原有完整解析、位姿健康及12/70%/80%验收链。没有可用的落盘候选仍失败。OOM、取消、其他断言、特征/匹配失败、显式Global以及后续扩展/BA失败均不由此豁免。候选头部计数和文件存在性只是进入验收的前置条件，不替代完整解析或质量门槛。

`diagnostics/sfm_mapper_failure.log`保留原命令、返回码和完整stdout/stderr；同名JSON绑定日志和已保存候选文件SHA。`sfm_pose_recovery.json`包含`mapper_failure`，若直接接受落盘Incremental候选则状态为`recovered_after_mapper_abort`、`recovery_applied=true`，不得标作正常完成Mapper。后续Global/core身份维持原合同。成功冻结协议时绑定异常日志、异常记录和恢复诊断哈希。

续跑根为`outputs/experiments/num4-new-4k-seed2500-abortfix-train-only-v1/`，调度在同名`-dispatch/`。仍从原1,000种子实验复用经过哈希校验的2,787帧和独立SQLite快照，重跑2,500种子Mapper；不直接复用上次未验收模型，不覆盖两次旧失败目录或删除once/exit标记。除新增精确异常分类外，求解参数、种子规则、几何门槛及双臂预算保持不变。它是新代码版本的恢复续跑，不冒充原对照已成功完成。
