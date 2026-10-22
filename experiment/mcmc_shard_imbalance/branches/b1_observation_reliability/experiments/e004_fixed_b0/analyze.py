"""固定 B0 v2 历史文本复算；不执行原脚本、模型、CUDA 或新下游传播。"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_SHA256 = "1f8490d61d1f156f3518a2fbab4446f09d532809a36017aee5e2f66f22304e58"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_identity(path):
    value = path.read_bytes()
    return {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}


def checked_file(root, name):
    relative = Path(name)
    require(
        not relative.is_absolute() and ".." not in relative.parts,
        "input path escapes root",
    )
    path = root
    for part in relative.parts:
        path = path / part
        require(not path.is_symlink(), "symlink input rejected")
    require(path.resolve().is_relative_to(root.resolve()), "input path escapes root")
    require(path.is_file(), "missing input: " + name)
    return path


def load_source():
    path = HERE / "analysis-source.json"
    require(not path.is_symlink(), "source manifest symlink rejected")
    require(
        file_identity(path)["sha256"] == SOURCE_SHA256,
        "source manifest identity differs",
    )
    source = read(path)
    require(
        source["source_run_id"] == "fixed-b0-results-20261022-v2",
        "only frozen v2 is supported",
    )
    return source


def verify_inputs(root, source):
    require(
        root.is_dir() and not root.is_symlink(), "input root must be a real directory"
    )
    identities = {}
    for name, expected in source["inputs"].items():
        path = checked_file(root, name)
        actual = file_identity(path)
        require(actual == expected, "input identity differs: " + name)
        identities[name] = actual
    incoming = list((root / "incoming").rglob("*"))
    require(not any(p.is_symlink() for p in incoming), "symlink input rejected")
    actual = {str(p.relative_to(root)) for p in incoming if p.is_file()}
    expected = {p for p in source["inputs"] if p.startswith("incoming/")}
    require(actual == expected, "incoming file set differs")
    require(
        identities["incoming/protocol.json"]["sha256"]
        == source["source_protocol_sha256"],
        "source protocol differs",
    )
    return identities


def check_output(root, output):
    runs = HERE / "runs"
    require(not runs.is_symlink(), "runs symlink rejected")
    require(not output.exists() and not output.is_symlink(), "output already exists")
    resolved = output.resolve()
    require(
        resolved != runs.resolve() and resolved.is_relative_to(runs.resolve()),
        "output must be a new directory under this experiment's runs",
    )
    require(
        not resolved.is_relative_to(root.resolve())
        and not root.resolve().is_relative_to(resolved),
        "input and output overlap",
    )
    for parent in (output, *output.parents):
        require(not parent.is_symlink(), "symlink output rejected")
        if parent == HERE:
            break


def compute_sensitivity(report):
    points = report["historical_point_analysis"]
    pairs = []
    maximum_reference_disagreement = 0.0
    for point in points:
        for group in point["historical_AA_by_parameter"].values():
            for b in group["branches"].values():
                config = b["update_param_group"]
                b1, b2 = config["betas"]
                step = b["P0_step"] + 1
                require(b["P3_step"] == step, "historical Adam step differs")
                for i, (p, g, m, v) in enumerate(
                    zip(
                        b["P0_parameter"],
                        b["P2_gradient"],
                        b["P0_exp_avg"],
                        b["P0_exp_avg_sq"],
                    )
                ):
                    denominator = (
                        math.sqrt(b2 * v + (1 - b2) * g * g) / math.sqrt(1 - b2**step)
                        + config["eps"]
                    )
                    parameter = (
                        p
                        - (config["lr"] / (1 - b1**step))
                        * (b1 * m + (1 - b1) * g)
                        / denominator
                    )
                    error = abs(
                        parameter - b["float64_scalar_reference"][i]["parameter"]
                    )
                    maximum_reference_disagreement = max(
                        maximum_reference_disagreement, error
                    )
                    require(error < 1e-12, "historical Adam formula differs")
    for p in points:
        if p["role"] != "prior_anomaly":
            continue
        group = p["parameter"]
        data = p["historical_AA_by_parameter"][group]
        d = data["a_minus_b"]["P3_parameter"]
        i = max(range(len(d)), key=lambda j: abs(d[j]))
        control = next(
            q
            for q in points
            if q.get("matched_to") == p["slot"] and q["owner"] == p["owner"]
        )
        records = []
        for q in (p, control):
            branches = q["historical_AA_by_parameter"][group]["branches"]
            a, b = branches["a"], branches["b"]
            for key in (
                "P0_parameter",
                "P0_exp_avg",
                "P0_exp_avg_sq",
                "P0_step",
                "update_param_group",
            ):
                require(a[key] == b[key], "historical initial state differs: " + key)
            branch_details = {}
            for name, v in branches.items():
                cfg = v["update_param_group"]
                beta1, beta2 = cfg["betas"]
                g = v["P2_gradient"][i]
                old_v = beta2 * v["P0_exp_avg_sq"][i]
                fresh_v = (1 - beta2) * g * g
                f = v["float64_scalar_reference"][i]
                branch_details[name] = {
                    "gradient": g,
                    "actual_parameter": v["P3_parameter"][i],
                    "actual_update": v["actual_update"][i],
                    "old_second_moment_contribution": old_v,
                    "fresh_second_moment_contribution": fresh_v,
                    "old_second_moment_fraction": old_v / (old_v + fresh_v)
                    if old_v + fresh_v
                    else None,
                    "corrected_sigma_over_eps": f["sigma_over_eps"],
                    "eps_fraction_of_denominator": 1 / (1 + f["sigma_over_eps"]),
                    "float64_formula_parameter_error": v["parameter_reference_error"][
                        i
                    ],
                }
            delta_g = a["P2_gradient"][i] - b["P2_gradient"][i]
            delta_p = a["P3_parameter"][i] - b["P3_parameter"][i]
            records.append(
                {
                    "owner": q["owner"],
                    "slot": q["slot"],
                    "role": q["role"],
                    "P0_exp_avg": a["P0_exp_avg"][i],
                    "P0_exp_avg_sq": a["P0_exp_avg_sq"][i],
                    "P0_step": a["P0_step"],
                    "lr": a["update_param_group"]["lr"],
                    "eps": a["update_param_group"]["eps"],
                    "gradient_a_minus_b": delta_g,
                    "parameter_a_minus_b": delta_p,
                    "observed_abs_parameter_delta_over_gradient_delta": abs(
                        delta_p / delta_g
                    )
                    if delta_g
                    else None,
                    "float64_formula_update_a_minus_b": a["float64_scalar_reference"][
                        i
                    ]["update"]
                    - b["float64_scalar_reference"][i]["update"],
                    "branches": branch_details,
                }
            )
        pairs.append(
            {
                "parameter": group,
                "flat_component": i,
                "selection": "largest_historical_target_group_P3_difference_descriptive_only",
                "source": "historical_backward_boundary_AA_not_new_replay_propagation",
                "points": records,
            }
        )
    result = {
        "pairs": pairs,
        "independent_float64_reference_max_disagreement": maximum_reference_disagreement,
        "no_new_P2_P3_or_optimizer_execution": True,
        "no_causal_matching_claim": True,
    }
    return result


def summarize(root, sensitivity):
    report = read(root / "incoming/comparison.json")
    verification = read(root / "remote-verification.json")
    independent = read(root / "independent-metrics.json")
    require(
        verification["status"] == independent["status"] == "verified",
        "independent verification incomplete",
    )
    require(
        independent["independent_tensor_pairs"] == 80, "independent pair budget differs"
    )
    require(
        report["protocol_sha256"] == verification["protocol_sha256"],
        "verification protocol differs",
    )
    require(
        hashlib.sha256((root / "incoming/protocol.json").read_bytes()).hexdigest()
        == report["protocol_sha256"],
        "source protocol differs",
    )
    checked = {}
    for path in (root / "incoming").rglob("*"):
        if not path.is_file():
            continue
        name = str(path.relative_to(root / "incoming"))
        if name.startswith("execution/"):
            remote = "outputs/analysis/fixed-b0-preflight-20261022-v2/" + name
        else:
            remote = "outputs/analysis/fixed-b0-20261022-v2/" + name
        expected = verification["files"][remote]
        actual = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        require(actual == expected, "download identity differs: " + name)
        checked[name] = actual
    statuses = {n: read(root / (n + "-status.json")) for n in ("execution", "monitor")}
    require(
        all(
            s["finished"] and not s["running"] and s["exit_code"] == 0
            for s in statuses.values()
        ),
        "source tasks did not complete successfully",
    )
    rows = []
    ulp = Counter()
    all_pairs = []
    for rank, outputs in report["ranks"].items():
        for name, data in outputs.items():
            pairs = list(data["pairs"].values())
            all_pairs.extend(pairs)
            for p in pairs:
                ulp.update({int(k): v for k, v in p["ulp_histogram"].items()})
            rows.append(
                {
                    "camera": int(rank),
                    "output": name,
                    "elements_per_pair": data["element_count"],
                    "strict_pair_count_min": min(
                        p["num_strict_unequal"] for p in pairs
                    ),
                    "strict_pair_count_max": max(
                        p["num_strict_unequal"] for p in pairs
                    ),
                    "above_tolerance_pair_total": sum(p["num_mismatch"] for p in pairs),
                    "unique_changed_elements": data["summary"][
                        "unique_changed_elements"
                    ],
                    "unique_above_tolerance_elements": data["summary"][
                        "unique_above_tolerance_elements"
                    ],
                    "max_abs_error": max(p["max_abs_error"] for p in pairs),
                    "max_ulp": max(p["max_ulp"] for p in pairs),
                    "max_five_repeat_range": data["observed_max_range"],
                    "native_a_reference_max_abs": max(
                        p["max_abs_error"] for p in data["native_a_reference_only"]
                    ),
                    "native_a_reference_above_tolerance_total": sum(
                        p["num_mismatch"] for p in data["native_a_reference_only"]
                    ),
                }
            )

    def flat(value):
        if isinstance(value, list):
            for item in value:
                yield from flat(item)
        else:
            yield float(value)

    def maxabs(value):
        return max((abs(x) for x in flat(value)), default=0.0)

    points = []
    for point in report["historical_point_analysis"]:
        groups = point["historical_AA_by_parameter"]
        differences = {
            name: maxabs(data["a_minus_b"]["P3_parameter"])
            for name, data in groups.items()
        }
        target = groups[point["parameter"]]
        formula = max(
            maxabs(b["parameter_reference_error"])
            for g in groups.values()
            for b in g["branches"].values()
        )
        replay = [
            v
            for v in report["repeat_point_outputs"]
            if v["owner"] == point["owner"] and v["slot"] == point["slot"]
        ]
        points.append(
            {
                "owner": point["owner"],
                "slot": point["slot"],
                "role": point["role"],
                "target_group": point["parameter"],
                "matched_to": point.get("matched_to"),
                "squared_distance": point.get("squared_distance"),
                "visibility": point["visibility"],
                "cameras": point["cameras"],
                "raw_quaternion_norm": point["raw_quaternion_norm"],
                "scale_axis_ratio": point["scale_axis_ratio"],
                "opacity": point["opacity"],
                "historical_target_gradient_max_abs_difference": maxabs(
                    target["a_minus_b"]["P2_gradient"]
                ),
                "historical_target_parameter_max_abs_difference": maxabs(
                    target["a_minus_b"]["P3_parameter"]
                ),
                "historical_parameter_difference_by_group": differences,
                "historical_float64_formula_max_abs_error": formula,
                "new_B1_max_range_by_camera_output": {
                    f"{v['camera_rank']}/{v['output']}": maxabs(v["range"])
                    for v in replay
                },
            }
        )
    changed = sum(n for d, n in ulp.items() if d > 0)
    assessment = {
        "status": report["observed_pairwise_status"],
        "execution_completed": True,
        "protocol_sha256": report["protocol_sha256"],
        "tasks": statuses,
        "preflight": "3_passed",
        "real_backward_calls": 10,
        "synthetic_backward_calls": 10,
        "extra_warmup_calls": 0,
        "pair_comparisons": 20,
        "tensor_pair_comparisons": len(all_pairs),
        "pair_element_comparisons": sum(p["elements"] for p in all_pairs),
        "strict_unequal_pair_elements": sum(p["num_strict_unequal"] for p in all_pairs),
        "bitwise_unequal_pair_elements": sum(
            p["num_bitwise_unequal"] for p in all_pairs
        ),
        "above_tolerance_pair_elements": sum(p["num_mismatch"] for p in all_pairs),
        "nonfinite_pair_elements": sum(p["num_nonfinite"] for p in all_pairs),
        "changed_pair_ULP_le_4_fraction": sum(n for d, n in ulp.items() if 0 < d <= 4)
        / changed
        if changed
        else None,
        "changed_pair_ULP_le_16_fraction": sum(n for d, n in ulp.items() if 0 < d <= 16)
        / changed
        if changed
        else None,
        "outputs": rows,
        "points": points,
        "initializations": report["initializations"],
        "downloaded_text_files": len(checked),
        "downloaded_bytes": sum(v["bytes"] for v in checked.values()),
        "independent_verified_tensor_pairs": 80,
        "independent_verified_frequency_range_arrays": 32,
        "global_error_bound_proven": False,
        "new_P2_P3_computed": False,
        "long_term_MCMC_causality_proven": False,
        "old_gate1": "not_reclassified",
        "new_training_updates": 0,
        "Test_Validation_images": "not_read",
    }
    complete = read(root / "incoming/replay/complete.json")
    assessment.update(
        unique_changed_camera_output_elements=sum(
            x["unique_changed_elements"] for x in rows
        ),
        native_a_reference_above_tolerance_total=sum(
            x["native_a_reference_above_tolerance_total"] for x in rows
        ),
        sensitivity_file="sensitivity.json",
        independent_adam_float64_reference_max_disagreement=sensitivity[
            "independent_float64_reference_max_disagreement"
        ],
        per_rank_peak_reserved_bytes=complete["resources"][
            "per_rank_peak_reserved_bytes"
        ],
        stage_unit_seconds={
            "preflight": 7.247,
            "prepare": 100.054,
            "run": 66.064,
            "compare": 126.513,
        },
    )
    lines = [
        "# 固定 B0 五次重复与少量 Gaussian 分析",
        "",
        "## 执行与数值结果",
        "",
        f"前置3项通过；每camera/GPU在各自worker进程内执行5次原backward，共10次，另有前置解析上下文10次；未调用额外forward/backward热身，无训练更新。程序完成，配对数值状态为 **{assessment['status']}**。",
        "",
        f"每相机取全部10个i<j配对（j为参考），合计20对、80项张量比较、{assessment['pair_element_comparisons']:,}个配对元素。严格不等{assessment['strict_unequal_pair_elements']:,}，位级不等{assessment['bitwise_unequal_pair_elements']:,}，超容差{assessment['above_tolerance_pair_elements']:,}，非有限{assessment['nonfinite_pair_elements']:,}。容差固定为2e-6 + 2e-4 × abs(reference)，没有事后放宽。",
        "",
        "以下计数是同一元素在多个配对中的重复观测，不是独立Gaussian数量。两个相机输入不同，不跨相机比较。",
        "",
        "| 相机 | 输出 | 每配对严格不等数范围 | 全配对超容差 | 最大绝对差 | 五次最大range |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['camera']} | {r['output']} | {r['strict_pair_count_min']:,}–{r['strict_pair_count_max']:,} | {r['above_tolerance_pair_total']:,} | {r['max_abs_error']:.9g} | {r['max_five_repeat_range']:.9g} |"
        )
    if changed:
        lines += [
            "",
            f"非零ULP差异的配对元素中，{assessment['changed_pair_ULP_le_4_fraction']:.3%}≤4 ULP，{assessment['changed_pair_ULP_le_16_fraction']:.3%}≤16 ULP。signed zero在ULP距离中合并，但bitwise统计保留；近零ULP长尾不能单独代表大绝对误差。",
        ]
    lines += [
        "",
        "逐元素严格/bitwise/超容差频次及range以32份NPY保留在远端，极值坐标、配对频次和完整ULP分布在comparison.json；极值并列取首个flat坐标。原生a/B1参考距离单列，不作为第六次重复，也不混入上述统计。",
        "",
        "## 初始化审计",
        "",
    ]
    for rank, init in report["initializations"].items():
        lines.append(
            f"- rank {rank}：初始化变化的RNG类别={init['changed_rng_components']}；扩展文件SHA={init['extension_sha256']}；显式热身forward/backward=0/0。初始化后RNG与全部5次trial绑定一致，未恢复RNG掩盖副作用。"
        )
    lines += [
        "",
        "首轮v1前置失败仍保留，不改写为通过。本轮使用v2独立目录与初始化前后审计。",
        "",
        "## 预选六点：新回放与历史传播分开",
        "",
        "三个异常点来自此前证据，并非本轮最大差异点的事后挑选。每个对照按同owner/相机可见模式、源一级A/A目标组P2/P3均在容差内筛选，以固定四个log特征平方距离匹配；不是无偏样本或因果匹配设计。",
        "",
        "| owner/slot | 角色 | 目标组 | 历史P2最大绝对差 | 历史P3最大绝对差 | 历史六组Adam公式最大误差 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for p in points:
        lines.append(
            f"| {p['owner']}/{p['slot']} | {p['role']} | {p['target_group']} | {p['historical_target_gradient_max_abs_difference']:.9g} | {p['historical_target_parameter_max_abs_difference']:.9g} | {p['historical_float64_formula_max_abs_error']:.9g} |"
        )
    lines += [
        "",
        "表中P2/P3及Adam公式全部来自历史一级A/A，而非将本轮B1回传得到。本轮没有重建projection VJP、通信或执行Adam；原始四元数参数差不是旋转角度。六点的新B1五次分量、几何/tile/轴比/norm及历史六组参数/矩/更新完整记录于comparison.json，assessment.json给出摘要。",
        "",
        "## 少量异常点的敏感性解释\n\n以下仍只解释历史一级A/A，具体分量按各异常点历史目标组P3最大差选择，属于描述性分析；不是新回放的误差传播干预。完整数值在`sensitivity.json`，独立float64公式写法与原参考最大相差8.67e-19，未执行真实Adam内核。\n\n1. **owner0/223955，quats分量1**：历史梯度差2.321e-6，对应参数差9.239e-5。旧二阶矩占更新后v的99.46%—99.80%，分母并非eps主导；对照0/621199同分量梯度差只有1.910e-11，公式预测更新差约2.502e-9，最终float32参数相同。此处不仅要看Adam状态，还要看到达P2的扰动幅度。\n2. **owner1/709801，quats分量1**：旧v约6.034e-20，历史梯度差4.594e-9，对应参数差9.197e-4。新v中旧状态占约45.70%／80.10%，校正sigma仍是eps的约36.3万／27.4万倍，所以这不是eps占主导；小二阶矩尺度及其与当前g²的混合使更新对梯度幅值敏感。对照1/413834旧m/v恰为0，但同号梯度、eps可忽略时，m与sqrt(v)的梯度幅值近乎抵消，两个实际更新均约−0.00316225。**零历史矩本身并不充分导致A/A更新分化**；两点历史梯度扰动也不同，不能把差异唯一归因于旧矩。\n3. **owner1/1488267，shN[10,2]**：旧m/v为0，历史该分量梯度差约4.018e-16，却对应参数差3.411e-6。校正sigma/eps仅0.207—0.220，eps占分母约82%，这里才是eps主导的高敏感区。对照1/403649的sigma/eps约5792，历史该分量梯度差1.669e-15反而更大，但参数差仅3.725e-9。不能由B1绝对误差小或处于固定容差内，推断下游更新一定可忽略。\n\n两个owner1对照的匹配距离为0，只表示四项匹配摘要相同；投影conic和优化器历史并非全相同，不能据此反推复制亲缘、relocation来源/目标角色或长期恢复机制。三个异常点及其对照本轮均有可见相机B1波动；不可见相机相应输出range为0。以shN异常点为例，本轮camera1 v_colors最大range为4.607e-15，而对照为5.662e-15，B1波动大小本身并不能排序历史Adam敏感性。所有这些跨运行并列观察均不组成新的一条已干预验证的因果轨迹。\n\n## 频次、极值和资源补充\n\n- 按camera×output×元素坐标去重，5次中有4,418,921个元素出现数值变化，超容差元素0；不是4,418,921个独立Gaussian。原生a/B1单列参考的40项张量比较也均无超容差，不计入本轮五次分布。\n- camera0 conic最大差0.0078125发生在约−23044的数值上，只是4 ULP；global slot2475769的分量0在10个配对中5次成为首个最大绝对误差坐标。其余输出极值并不固定在同一点。\n- 最大ULP为camera1 colors的5,439,488，落在预选owner1/1488267的近零分量：约1.343e-14与8.826e-15，绝对差4.607e-15。大ULP长尾与小绝对扰动可以同时成立，不能只看其中一项。\n- 前置unit7.247秒，prepare100.054秒，run66.064秒（其中replay子进程20.888秒），compare126.513秒，整轮约5分钟。峰值reserved分别585,105,408／572,522,496字节，约0.545／0.533GiB；不是吞吐量基准。主存无限额，资源保护无触发；终态GPU空闲、服务active。",
        "## 能与不能得出的结论",
        "",
        "本轮只测得固定输入、固定环境、五次同进程执行的观测范围；不能证明所有输入/调度下的全局误差上界。它与原生A/A的调度、内存布局和进程组生命周期不同。即使B1全部在容差内，也不能抹去既有P2/P3超容差，更不能据此认定Adam差异导致来源池恢复不对称或长期Gaussian退化。",
        "",
        "现有结果支持优先设计A/A基线约束下的非干预验收，而不是为追求所有SHA相同大规模重写rasterizer。未重写rasterizer、改变确定性/优化器/通信或启动原六分支；旧Gate1仍未通过。若讨论非干预验收，下一步应单独预注册多次A/A基线、严格状态/调用预算约束以及数值/分布等价标准，而不是以本轮五次最大值直接设阈值并自动晋级。",
        "",
        "## 核验与证据",
        "",
        f"协议 `{report['protocol_sha256']}`。执行与唯一监控均finished/exit0。独立CPU核对80项配对的严格/bitwise/isclose/maxabs/ULP及32份频次/range数组一致；另核验协议绑定输入、安装包、快照file/canonical SHA及调用预算。下载{len(checked)}份必要文本、共{assessment['downloaded_bytes']:,}字节，SHA/大小与远端一致，未下载大张量或图像。",
        "",
        "RAM无限额及原GPU/磁盘/超时/租约保护保留；无Test/Validation图像读取、渲染或评价，无新checkpoint或模型。完整证据为incoming/、remote-verification.json、independent-metrics.json、download-verification.json；本分析脚本仅标准库。",
        "",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return {
        "assessment.json": json.dumps(
            assessment, ensure_ascii=False, indent=2, allow_nan=False
        ),
        "sensitivity.json": json.dumps(
            sensitivity, ensure_ascii=False, indent=2, allow_nan=False
        ),
        "download-verification.json": json.dumps(
            checked, ensure_ascii=False, indent=2, allow_nan=False
        ),
        "repetition-metrics.csv": buffer.getvalue(),
        "report.md": "\n".join(lines),
    }


def analyze(input_root, output_dir):
    started = datetime.now(timezone.utc).isoformat()
    root = Path(input_root).absolute()
    output = Path(output_dir).absolute()
    check_output(root, output)
    source = load_source()
    inputs = verify_inputs(root, source)
    sensitivity = compute_sensitivity(read(root / "incoming/comparison.json"))
    outputs = summarize(root, sensitivity)
    require(verify_inputs(root, source) == inputs, "inputs changed during analysis")
    output.mkdir(parents=True, exist_ok=False)
    for name, content in outputs.items():
        with (output / name).open("xb") as handle:
            handle.write(content.encode("utf-8"))
    record = {
        "status": "complete",
        "kind": "local_analysis_only",
        "analysis_profile": source["analysis_profile"],
        "source_run_id": source["source_run_id"],
        "source_protocol_sha256": source["source_protocol_sha256"],
        "analysis_code": file_identity(Path(__file__)),
        "analysis_source_sha256": SOURCE_SHA256,
        "started_at_utc": started,
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(root),
        "inputs": inputs,
        "outputs": {name: file_identity(output / name) for name in outputs},
        "new_forward_backward_calls": 0,
        "new_P2_P3_computed": False,
        "new_optimizer_updates": 0,
        "old_gate1": "not_reclassified",
    }
    with (output / "analysis-run.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        record = analyze(args.input_root, args.output_dir)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "status": record["status"],
                "kind": record["kind"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
