# 公共前缀完成：资源收尾失败与Gate 0只读重建

2026-10-29 · MCMC-Q1-E004 · **修复后的合成前置通过，真实100步产物与完整checkpoint验真；原执行包装器仍为失败，不补造原始退出记录。**

## 执行边界

- 用户授权新版前置→100步公共前缀→Gate 0后停止，没有授权真实A/B。实现提交a269eb1df1153756f20ae531c60982f239053fb6，64份依赖与远端一致。
- 执行任务`20261029-102938-0448`与唯一监控`20261029-102938-feb9`均finished=true、running=false、exit_code=1。执行02:29:38 UTC开始、约02:35:01结束，约323秒；监控约02:35:09结束。事后CPU验真另计，不是追加训练。
- 合成前置`1 passed in 65.48s`，服务66.097秒；prepare成功44.059秒；prefix服务212.394秒。原零来源边界修复通过本次独立前置，不将v1失败改为通过。
- 两rank原始worker均记录100步、各100次forward/backward、600次Adam参数组step，日志均见Job Done；公共前缀合计100全局更新、200 forward/backward、1200 Adam组step。保存了原生12100完整checkpoint。
- 原父进程在收尾调用缺失的`Monitor.record()`时抛错，未写`case.exit.json`、`complete.json`或顶层`gate0.json`。子进程原始returncode/resource_failure没有可靠落盘，不能补填0/None或称原套件通过。
- 原始两份owner Gate 0与worker/native-tail已存在，可独立只读验真；正式A、B、arms协议及once均不存在，未新建或运行两臂。

## 只读验真

先按原64份代码、来源与前置绑定检查，再独立检查两rank100步计数、Train相机、SH与六组学习率。原校验函数错误读取每行顶层position_lr；实际只在诊断步mcmc_health_by_rank中记录。没有跳过学习率：按冻结Train相机中心计算scene_scale、原指数日程逐步重算全部六组lr，100步全部精确一致，12100诊断行两rank嵌套position_lr也一致。

所有worker绑定快照manifest及数组SHA验证，原生checkpoint元数据与全部组件SHA验证。随后单独设置CUDA_VISIBLE_DEVICES为空，仅在CPU解码模型/optimizer/strategy/camera/RNG张量重组完整状态；两rank摘要均与12100保存时的prefix-state完全一致。未实例化模型或optimizer、未恢复活跃RNG、未初始化CUDA，新增forward/backward/optimizer/训练均0。

Gate 0由同一保存状态重新计算。native Adam betas在typed快照中是tuple，在JSON中是list；对结果仅做精确JSON表示归一化后，与原两份owner报告逐值一致，未应用额外数值容差。该表示差异也会影响当前verified_gate0的直接字典比较，后续修复应一并覆盖。

## Gate 0实测结果

各owner的六组参数step均为12100，迁移前后不变；全部S/T六参数在relocation后、noise前位级一致；S六组两矩清零，T/U两矩位级保留。以下非零率和数值表专指opacity，不把它扩大为六组所有元素都非零。

| owner | 唯一来源S | 目标T | S清零前opacity两矩非零率 | S清零后两矩非零率 | T迁移后两矩非零率 |
|---|---:|---:|---:|---:|---:|
| 0 | 80,384 | 87,900 | 两者均100% | 两者均0% | 两者均100% |
| 1 | 142,316 | 168,827 | 两者均100% | 两者均0% | 两者均100% |

| owner | S的m绝对值中位数 | S的v中位数 | T的m绝对值中位数 | T的v中位数 | S归一化历史绝对值中位数 |
|---|---:|---:|---:|---:|---:|
| 0 | 1.91785e-09 | 3.28817e-16 | 8.43065e-12 | 5.49641e-20 | 0.115296 |
| 1 | 1.36168e-09 | 1.45718e-16 | 6.45588e-12 | 2.04697e-20 | 0.125649 |

m为exp_avg，v为exp_avg_sq；归一化历史指m/(sqrt(v)+eps)，不是实际下一步更新。两组opacity学习率均0.05，betas为(0.9,0.999)，eps为1e-15。目标值为目标点数口径，来源值为唯一来源口径，不当作家族等权因果比较。

| owner | S清零前m为正 | S清零前m为负 | T保留m为正 |
|---|---:|---:|---:|
| 0 | 52.854% | 47.146% | 94.716% |
| 1 | 52.655% | 47.345% | 95.982% |

这排除了“本次S清零前opacity历史两矩本来就全零”的情形。S的一阶矩符号约53%正、47%负，不支持统一的单向动量解释；原始m/v很小，也不能忽略归一化、step/bias correction、当前梯度与参数舍入。**非零不等于干预必然有效，更不等于保留历史一定有益。**

这里没有12101首步或20步真实干预数据，不能把新前缀slot与旧T0/T1未来命运配对，也不能证明来源死亡、owner长期失衡或质量改善的因果关系。

## 故障与后续边界

需修复的是实验编排/验真接口，而非修改原生训练机制：

1. Monitor补齐run_stage要求的record接口，并用实际轻量子进程验证成功/失败收尾，不能只mock整个run_stage。
2. 按原生日志真实层级检查学习率：逐步核冻结日程，诊断行核mcmc_health_by_rank，不假设每行顶层有position_lr。
3. Gate 0 JSON表示归一化后作精确比较，保留数值不变与完整字段验证。

**不应为这些收尾问题重跑已经验真的100步。** 下一步优先设计独立零训练恢复/重新绑定协议，绑定旧执行提交与已有完整checkpoint；原failed、缺失case.exit与complete继续保留，不伪造历史成功。新的A/B仍需在用户审阅Gate 0并明确授权后才能运行。本次未修执行代码、未新建恢复协议、未重训或启动A/B。

## 资源与证据

- 输出8,558,917,349字节（约7.97GiB），低于16GiB预算。两rank峰值reserved分别4,347,396,096/4,332,716,032字节，最终遥测finished=true；原生退出资源记录缺失，不能据此补造该记录。
- 主存max/high/swap均无限额。03:05:10 UTC两卡无计算进程、各3MiB，image3d-cloud active；远端代码未修改。
- 前置SHA：`6c649c14cfa03324149c53f96aaec7d0dc274764fe26f39c68451a59170ffb14`。
- 协议SHA：`0332814e20f2dd4fcbbe0234e201a9fc22633ac52d169472b4a74fd0a64e793c`。
- 原生checkpoint hash：`ed3257af8a07625b8abeacfe6df4016d24c9db9e8a4cca95336fa11888f326f9`。
- 取回130份必要文本、3,521,813字节，逐文件SHA验证；未下载模型/优化器二进制、NPY、图像或完整checkpoint历史。
- [机器可读只读重建](prefix-gate0-20261029-v2-readonly.json)、[独立运行记录](../records/adam-intervention-20261029-v2-prefix.json)。
- 本地核查原样保留：第一次复用原verify_worker因position_lr字段错误停止；第二次比较因betas tuple/list表示不同停止；第三次收集触及8MiB文本上限，最终排除重复完整checkpoint历史后完成。上述都是只读后验检查，没有重新运行训练或放宽实验门槛。
