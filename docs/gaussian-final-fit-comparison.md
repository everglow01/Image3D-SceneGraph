# MCMC final-fit 未拟合视角与等预算对照 v1

## 范围与预先固定的比较

本实验只研究重建质量，不调整生产默认、不更换浏览器、不重跑 SfM/SOR/30k 主训练。
复用当前冻结 dataset `9de071505b4f175b0f219c3ffbb4ecfd8e20a1832c68051ba5515b6183bb367a`：
1,376 Train、179 Validation、171 Test，三组不重新划分。

| 候选 | 来源 | 新增优化 |
|---|---|---|
| Selection | 冻结 MCMC SOR `054bbb6ca1563de47e8201b9b713bc112dc29a1a24d34b4cab5658a1d7d31586` | 无 |
| Train-only | 同一 Selection 快照 | 仅 Train，2,000 updates |
| Train+Validation | 已完成的 final-fit `a9650d439bed30fae4eec5e854dae0069c7f32f9b1ac575061fb088cb8184b9f` | Train∪Validation，2,000 updates |

新增 Train-only 必须使用源模型相同的 resolved config、seed、两张 GPU、fresh Adam、LR、最大 SH 和损失；固定拓扑，无 MCMC 噪声/正则或任何策略更新。两臂均为 4,000 次相机采样，而非相同 wall-clock 或相同 epoch 数。采样总体不同，具体相机序列不能要求相同。只有完成固定预算后的最后模型进入终评，不能根据控制臂 Validation 再选步数或参数。

三项预先固定的配对差值：

1. Train-only − Selection：同监督下额外短优化的收益。
2. Train+Validation − Selection：交付补拟合的总收益。
3. Train+Validation − Train-only：同预算下改变 RGB 监督集合的增量。

当前 Train+Validation 是历史冻结产物。复用时核验相同环境指纹，并在 protocol/comparison 中保留两臂各自的 code hash；不能声称两次执行的代码哈希相同、跨运行逐位复现，或单种子已经完全证明因果。

## 评价边界

- 179 个原 Validation 视角已经用于 Train+Validation 优化，不能用来比较泛化。Train-only 的末尾 Validation 报告用 `control_validation` / `held_out_after_train_only_control` / `selection_eligible=false`，不是新一轮选模。
- 使用冻结的全部 171 个 Test 相机，不挑选有利子集、不事后排除失败相机。先完成两臂、冻结三路模型及协议，再一次性终评。
- Test 只用于终评，不用于训练或调参；已有 `write_frozen_candidate` / Test-consumption schema 1 保护保持不变。执行必须获得明确的 Test 消费授权并传 `--authorize-test`。
- Test 的相机/特征曾参与共享 SfM。这里的 held-out 是 **RGB 优化监督隔离**，不是未知相机估计、新场景、完全隔离几何或米制精度证据。
- 现有训练 CLI/runtime 的 `validate_contract(..., dataset_root)` 会读取包括 Test 在内的图像文件字节来校验 SHA；`test_rgb=not_loaded` 仅表示没有把 Test 解码加载为训练/评价张量，不表示文件从未打开。新 `freeze` 阶段不传 dataset root，只读取合同和模型/记录，不读取任何 dataset RGB。
- 终评统一为原生 gsplat、相同配置和单 GPU 全模型渲染。同时报告 raw-float 与 display-clamped PSNR/SSIM；不冒充浏览器指标。
- 报告包含三路绝对分布、三项逐相机差值、均值/分位数、改善/退化数量及最差十视角 ID；对应预览由现有 evaluator 保存。任何相机失败都不能产生有效比较报告。
- 不预设自动推广 PASS：单场景、单种子、均值提高都不足以默认开启 final-fit。静态相机指标也不能证明自由移动时没有闪烁。

## 执行顺序

所有真实 CUDA 执行仅在授权远端进行，源码只能通过 Git 同步。新实验使用独立目录，不能覆盖既有 final-fit 或 viewing job。现有任务和服务不需要重启。

1. 预检目标实例/目录、当前 GPU 数量与空闲状态、内存/磁盘、源模型与配置；必须保持两 GPU 训练预算，不能发现只有一张空卡就悄悄改为单卡。
2. 使用既有 `scripts/run_gaussian_final_fit.py`，传入原 MCMC dataset/root/source/config/selection-evaluation，以及新增 `--train-only-control --distributed` 和独立 `--output-dir`。不加入新的更新次数或 LR 参数。任务沿用 dashboard 启动持锁、防重复提交与独立只读监控规则。
3. 控制臂成功后运行 CPU-only 冻结命令。下面的变量均指向已核验的现有文件或新的目标目录，不创建替代数据：

```bash
.venv/bin/python scripts/evaluate_gaussian_final_fit.py freeze \
  --dataset-contract "$DATASET" \
  --resolved-config-json "$CONFIG" \
  --source-model "$SOURCE_MODEL" \
  --selection-evaluation "$SELECTION_EVALUATION" \
  --train-only-dir "$CONTROL_DIR" \
  --train-validation-dir "$EXISTING_FINAL_FIT_DIR" \
  --output-dir "$COMPARISON_DIR"
```

冻结会核验 source/final/evaluation/profile/split hashes、每步采样 IDs、两臂预算、模型实际行数/SH 和评价身份。输出三份 frozen-candidate 与最后写入的 `protocol.json`；无任何 Test-consumption 文件。记录 `sha256sum "$COMPARISON_DIR/protocol.json"` 的实际结果，不能只信路径。

4. **取得单独的 Test 消费授权后**，将冻结协议的 SHA 作为显式参数运行：

```bash
.venv/bin/python scripts/evaluate_gaussian_final_fit.py evaluate \
  --output-dir "$COMPARISON_DIR" \
  --protocol-sha256 "$PROTOCOL_SHA256" \
  --dataset-root "$DATASET_ROOT" \
  --authorize-test
```

执行前再次核验三路所有源文件和 frozen-candidate；创建一次性 `test-started.json`，按 Selection / Train-only / Train+Validation 顺序调用既有 Test evaluator。每路各消费一次 Test，不重复训练或选模。失败保留证据，不能删消费标记、换目录 refreeze 或盲重试；需要先检查实际状态并另行决定处理方式。

5. 成功后自动生成 `comparison.json`。若仅报告阶段失败且三路评价已经完整，可单独运行 CPU-only `report --output-dir ... --protocol-sha256 ...`，它不读取 RGB、不消费 Test，也不覆盖已存在的报告。

产物始终保留在 git-ignored outputs 中。此代码实现本身不表示控制臂已执行、候选已冻结、Test 已消费或泛化改善已成立。
