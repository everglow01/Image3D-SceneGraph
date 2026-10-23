# 单步 B1→P2→P3 连通传播

**MCMC-B1-E006** · [所属支线 B1](../../../../README.md#b1) · [身份](experiment.json)

## 问题与边界

同一原生状态下，固定 B1 后的下游是否可重复？改变实测 B1 是否产生额外的 P2 梯度、P3 Adam 更新差异？本实验实际重算，不拼接历史 P2/P3，不以局部结果证明长期恢复不对称的唯一根因。

**实现与本机轻量检查完成，远端未运行。** 新代码先经 Git、用户手动同步；明确新协议 SHA 和执行授权后才能运行。不沿用 E004 授权，不自动恢复旧六分支。

## 冻结设计

源状态为原库 baseline 的 iteration 12000 checkpoint，仅计算12001；名义日程仍30000。B1来自 E004 `workspace-20261023-v1` 的两相机五次实测输出，按同序号配对，不选最大误差。具体 SHA、case 列表与计算预算以 `run.py` 的 CONTRACT及prepare生成的protocol为权威。

| 组 | 次数 | 行为 |
|---|---:|---|
| native | 3 | 原 backward 原样返回；带同一探针，不是无观测基线。 |
| same | 3 | 每次注入两相机 trial-0，独立恢复父状态。 |
| varied | 4 | 依次注入 trial-1/2/3/4；trial-0复用same-0。 |

顺序：native-0 → same-0/1/2 → native-1 → varied-1/2/3/4 → native-2。最多10次全局单步更新、20次相机forward、20次loss backward，其中6次真实rasterizer backward、14次替换返回；每rank六个Adam参数组各step一次，总计120次组step。最后一组Adam完成并记录P3后立即停止，strategy post、P4、噪声、relocation均为0。

## 准入与判据

- 每次恢复前验真并导入固定预编译扩展，不调用预热、不允许JIT fallback；随后仅执行原生checkpoint RNG恢复。历史P0/P1/B0若不匹配就停止，不对初始化差异作隐式豁免。
- 每次从同一只读父副本恢复。P0/P1/B0与历史源、先前native-0位级核对，包含RNG、相机、layout。B1接入前两rank共同确认；不恢复RNG掩盖副作用，不遇失败后剔除字段。
- 只替换rasterizer backward的12项返回值；检查rank、shape/dtype/device/stride、有限性、SHA和存储别名。不改变loss正则分支、反向通信、投影或原生Adam。
- same内三对P2/P3及P3−P0更新量必须在原容差内；否则保留报告并停止后续case，不自动重试、放宽容差或扩大样本。
- 主要结果为两owner全部六参数组的梯度、参数、Adam矩和P3−P0更新量；六个预选点仅作解释。透明度使用CPU float64 sigmoid记录距`.005`阈值的诊断性距离和跨越，不冒充实际执行的原生strategy死亡掩码，也不混同render cutoff。
- 原容差 `2e-6 + 2e-4*abs(reference)`，actual/reference方向固定。3次same不是全局误差界；varied与same的比较仅支持冻结条件下的单步证据，未重现也必须如实报告。
- 禁止Test/Validation图像解码、评估、导出和新训练checkpoint；允许原trainer创建不解码的Validation相机元数据。

## 执行与资源

入口是 `python -m experiment.mcmc_shard_imbalance.branches.b1_observation_reliability.experiments.e006_connected_propagation.run --help`，导入/help不加载Torch、NumPy或gsplat。所有产物限本实验 `runs/<new-run-id>/`，拒绝覆盖与软链逃逸。

prepare要求执行源码已受Git跟踪且与HEAD一致，记录实现commit和文件SHA，然后验真/冻结协议；run需执行与比较双授权，先复制一份父checkpoint，再逐case独立恢复，自动门禁与最终compare。不支持接续失败run；失败必须另行审阅。compare仅对已完成case作CPU分析；monitor只读、不占GPU锁。

主存无限额、每卡reserved≤18GiB、磁盘4GiB停止。磁盘准入按一份父checkpoint与十份六阶段快照实际字节估算，另留8GiB且不低于20GiB；不足时停止，不删除旧证据。执行含比较总时限6小时；唯一执行/只读监控、lease和持久once防重复。

面板命令须运行显式子进程（如 `bash -lc '…'`），不在面板顶层直接exit或exec；monitor自身作为Python子进程返回，让面板包装器记录结束状态。此改动不追认旧监控终态、不修改面板服务。

## 验证与后续

本机仅旁置标准库/NoTorch测试与catalog/history。远端测试使用新的 `IMAGE3D_AUTHORIZE_CONNECTED_PROPAGATION_TESTS=1`；旧变量不能开启。微型检查与真实预算分列：2个测试，合成图forward/backward各2次、有效注入返回4次、Adam参数组step共12次、双rank共同拒绝检查1次；真实rasterizer forward/backward与strategy post均为0。不解码任何图像，未执行不能记为通过。

本实验之后，独立无观测A/A与非干预验收仍需另立协议，使用独立基线和验收数据；旧Gate1不追认。通过可信观测准入后才返回Q1短窗口生命周期。
