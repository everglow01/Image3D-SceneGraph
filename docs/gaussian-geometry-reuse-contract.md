# 标准 Gaussian Job 的阵列配对与几何复用

本合同约束 `rig_neighbors_vocab_v1` 与 `gaussian_geometry_source_job_id`。二者都是显式选项，不改变现有默认。本阶段仅支持 multi-image、Project Gaussian 的 ordinary COLMAP，以及 Project/MCMC 两种原生训练器；不支持视频、VGGT-BA、Graphdeco、跨项目引用或鱼眼。

## 显式采集清单与配对

`sfm_pairing=rig_neighbors_vocab_v1` 要求 `sfm_camera_calibration=folder_grouped_opencv_v1`，并通过 `sfm_capture_metadata` 表单字符串传入 JSON：

```json
{
  "schema_version": 1,
  "images": {
    "10/10_DSC0001.jpg": {"camera_id": "10", "capture_index": 0},
    "11/11_DSC0001.jpg": {"camera_id": "11", "capture_index": 0}
  }
}
```

- 键必须与本次上传的相对文件名完全相同，包含上传方保留的目录前缀；不得漏图、加图或使用重复上传路径。
- `camera_id` 必须等于相对父目录；同相机同采集序号唯一。仅接受相机身份和非负整数采集序号，不接受位姿/内参字段。
- 同一 `capture_index` 表示已知的同一次采集/站位；不能把无序图片随意编号后冒充阵列数据。仅对有明确采集依据的数据启用。
- 固定最多10,000张图、2–64个相机组、100万个邻接对；越界失败，不静默截断。
- 先执行描述子兼容的词袋检索，固定 `num_images=100`；然后通过 `matches_importer --match_type pairs` 补充邻接对。两阶段保持同一特征/局部匹配器/几何验证/GPU设置，不切换成另一算法。
- 邻接包含同相机下后续10个有效采集、同采集跨相机，以及相邻有效采集的跨相机组合；去重、排序后保存，不使用官方位姿推断空间邻近，不施加刚性rig外参约束。
- `diagnostics/rig_pairing.json` 绑定清单与邻接对文件SHA、固定参数、图像/候选数量；`sfm_frontend_contract.json`、timing及SfM诊断保留独立profile。候选数量不等于成功匹配数量。
- 缺清单、覆盖不一致、树缺失、命令失败均明确失败；不自动回退Exhaustive。既有pose/readiness门禁保留，此实现不证明与全量配对等质。

`prepare_eyeful_capture_metadata.py` 仅为Eyeful提供清单导出：读取官方Train列表和已有KRT记录中的cameraId/frameId，检查同名采集token跨相机一致，再为同次采集但缺位姿的Train图补齐采集身份。未观察过的采集token拒绝猜测；不使用K/T和畸变参数，不读取Test RGB，不修改原图，输出拒绝覆盖。输出名使用场景内相对路径（例如 `10/10_DSC0001.jpg`），服务器本地入队须使用同名；若浏览器增加外层目录前缀，清单也必须显式匹配。

## 两个正常且独立的Job

A 正常计算几何并训练；B 通过 `gaussian_geometry_source_job_id=A` 引用它。B 仍通过 `JobStore.enqueue_job`、串行worker和原来的attempt机制执行，而非研究结果导入器。

1. A 在几何健康检查和dataset划分后、模型训练前，生成 `diagnostics/geometry_bundle.json`（schema 1）。当前版本的multi-image ordinary-COLMAP Project/MCMC Job均可生成；旧Job不自动补造该记录。
2. 包记录源Job ID、输入路径及真实文件SHA、几何选项、Train/Validation/Test ID划分，以及COLMAP文件、去畸变图片、相机/点云和必需诊断的SHA。包不包含Gaussian模型、优化器、训练结果或前端展示资产。
3. B 的来源必须位于同一个JobStore、处于done、是普通Gaussian Job且具有该包。拒绝派生只读结果、路径穿越、符号链接、缺项及来源ID不一致。
4. 入队固定包SHA。未显式提供的几何选项继承来源；显式提供而不一致则拒绝。训练器和各自训练默认参数不继承，Project/MCMC仍独立选择；几何图像分辨率必须一致。
5. B 执行时再次确认来源状态/包SHA、源和目标原图字节，以及几何配置；以独立文件复制到B的attempt，逐块计算SHA并响应取消。拒绝覆盖，不用硬链接/软链接，不引用A的路径来提供B资产。
6. 校验通过后才写 `diagnostics/geometry_reuse.json`。Adapter仅跳过几何外部命令；仍检查几何证据、重新构造并核对相同的数据划分，执行本Job的初始化、训练、Validation、SOR、可选final-fit、导出、导航与原子发布。
7. B 的 `gaussian_geometry_origin=reused` 与来源ID/包SHA明确暴露；源timing是被复用几何的历史证据，不是B重新计算的耗时。继承原几何的实际solver/profile，不伪装成B新运行了一次SfM。
8. 任何失败遵循正常failed状态并保留partial attempt，不能自动重算；取消仍为cancelled。显式重试创建新attempt并重新校验来源，不自动retry。B完成后删除A不会影响B文件展示，但执行前来源已删除会使B失败。

## 资产与展示

A和B都有自己的Job ID、manifest、日志、模型、评估、PLY/浏览器资产与导出包。新增资产角色：

- `gaussian_geometry_bundle` → `diagnostics/geometry_bundle.json`
- `gaussian_geometry_reuse` → `diagnostics/geometry_reuse.json`（仅B）
- `sfm_rig_pairing` → `diagnostics/rig_pairing.json`（显式混合配对）

原稳定角色如 `scene_splat`、`gaussian_model`、`gaussian_evaluation`不改变。B的模型不是A的模型副本；前端照常展示B，并标注几何复用来源。

## 验证边界

CPU/模拟测试必须覆盖：配对覆盖与确定性、匹配执行顺序、元数据拒错、A/B独立生命周期、真实Adapter跳过SfM而进入各自训练器入口、路径/哈希/配置漂移、取消和重试、来源删除后B资产独立。测试不得把模拟训练视为真实训练成功。真实COLMAP召回率、注册率、几何质量与展示质量仍须另行授权验证；实现阶段不启动训练。
