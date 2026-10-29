# 来源Adam前置：零来源张量摘要失败

2026-10-29 · MCMC-Q1-E004 · **合成前置失败；真实公共前缀、真实Gate 0及A/B均未启动。**

## 执行结论

用户仅授权“前置通过后运行公共前缀、Gate 0后停下”。远端main为e98c8285db797ebcb148eeb27ce8f8b667916fd7，64份依赖与本地一致；新目录、once和GPU租约保护生效，未同步或修改远端代码。

执行任务`20261029-092127-67af`与唯一只读监控`20261029-092127-4541`均finished=true、running=false、exit_code=1。执行于01:21:27 UTC启动，约01:22:41退出；监控约01:22:57退出。pytest为`1 failed in 73.05s`，systemd前置服务运行73.673秒。

控制日志只有preflight阶段；新版preflight.json不存在，prepare未执行，正式输出目录`adam-intervention-20261029-v1`不存在。因此没有正式协议、100步公共前缀checkpoint、真实Gate 0或真实两臂结果。

## 已完成的合成部分与失败边界

- 两rank的8点合成公共前缀各完成4步，合计4全局更新、8次相机forward、8次loss backward、48次Adam参数组step。
- 合成prefix已有两rank完成记录及其微型原生checkpoint。合成rank0有2个来源/2个目标；rank1刻意设置零来源/零目标，两rank对应的合成Gate 0结构检查均已返回。
- 合成A在首次`_next_camera_batch`的初始化保护内失败，尚未进入第一个新batch/render/backward/Adam；合成B未启动。后续公式、完整两臂、noise等价性与负例检查未全部到达，不能将整个前置追认为通过。
- 以上仅是8点合成fixture，不是新的真实100步前缀，也不是可用于判断实际Adam动量大小的Gate 0。

## 失败定位

rank1拒绝记录与异常栈一致：

```text
RuntimeError: self.stride(-1) must be 1 to view Float as Byte
(different element sizes), but got 0
```

调用链：`initialize_arm → restore_source_moments → digest_state → reshape(-1).contiguous().view(torch.uint8)`。

`restore_source_moments`在零来源rank上仍构造并摘要空来源moment子张量。冻结的公共`digest_state`会把浮点张量重解释为字节；此次空张量的最后一维stride为0，`.contiguous()`没有使其满足该dtype-view要求，因而报错。报错行同时调用current与expected的摘要，现有栈没有单独标出具体是哪一次；不要声称已动态区分这两者。

rank0记录的是`peer rejected`，说明共同边界拒绝传播生效，而非另一个已定位的独立数值错误。没有数据支持把此错误解释为来源Adam干预无效、训练loss不稳定或显存OOM。

## 后续修复边界（尚未实施）

建议只在Q1-E004内为来源子张量摘要增加空张量处理：保留dtype与原shape，空数据段无需执行dtype字节视图；非空张量继续使用原摘要规则。保留零来源owner用例、原RNG守卫与T/U保护，不为过关删除这个测试，也不修改冻结公共helper或生产训练器。

应补本机无Torch的空张量适配回归，并保留远端真实空来源用例。修复后经Git提交、用户手动同步，再在新的尝试目录明确授权前置；本次失败目录和once不覆盖、不恢复。本次未改执行代码、未补跑，真实公共前缀继续未启动。

## 验真

只读取回32份必要文本，共717,819字节，逐文件SHA通过；未下载Torch checkpoint二进制、NPY或图像。核验程序未导入Torch/NumPy/gsplat。pytest生成的current目录链接只记录、不跟随。

01:25:54 UTC现场两卡均无计算进程、各3MiB，image3d-cloud active，远端tracked工作树无变化。原失败产物保留。

[独立运行记录](../records/adam-intervention-20261029-v1-preflight.json)绑定面板终态、控制日志、部分合成证据和只读检查。完整文本位于`runs/adam-intervention-20261029-v1-local/incoming/`。
