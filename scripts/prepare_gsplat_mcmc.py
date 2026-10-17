#!/usr/bin/env python3
"""Build a hash-pinned, isolated MCMC relocation overlay; keep the rasterizer binary."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path


SOURCE_SHA256 = {
    "relocation.py": "6843cb21b97844df891edfcba833d7fe97beccd88f365e3a68786d71f54c82ea",
    "strategy/ops.py": "b30fa9e4d60a97ec619eb54d43d158d3e9dc8102c3d971acbb46da8385b53074",
    "cuda/csrc/RelocationCUDA.cu": "dca4b1cef039e94eb2c902f20e645e777cca3f9be352da8bab9cffa4ac794051",
}
MCMC_SHA256 = "0c2fc042833f39fae0b785de07d0472fe0d1f22437108b0d04a45c0e4fe67c16"
EXTENSION_NAME = "_project_mcmc_relocation_v1"
BINDING = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

namespace gsplat {
void launch_relocation_kernel(at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                              const int, float, at::Tensor, at::Tensor);
}

std::tuple<at::Tensor, at::Tensor> relocation(
    at::Tensor opacity, at::Tensor scales, at::Tensor ratios, at::Tensor binoms,
    int n_max, float min_opacity
) {
    TORCH_CHECK(opacity.is_cuda() && opacity.scalar_type() == at::kFloat,
                "relocation requires CUDA float32");
    TORCH_CHECK(scales.scalar_type() == at::kFloat && binoms.scalar_type() == at::kFloat
                && ratios.scalar_type() == at::kInt, "relocation dtype mismatch");
    for (const auto &tensor : {opacity, scales, ratios, binoms}) {
        TORCH_CHECK(tensor.device() == opacity.device() && tensor.is_contiguous(),
                    "relocation requires contiguous tensors on the same device");
    }
    TORCH_CHECK(opacity.dim() == 1 && scales.dim() == 2 && scales.size(1) == 3
                && scales.size(0) == opacity.numel() && ratios.dim() == 1
                && ratios.numel() == opacity.numel(), "relocation shape mismatch");
    TORCH_CHECK(n_max > 0 && binoms.dim() == 2 && binoms.size(0) == n_max
                && binoms.size(1) == n_max, "relocation binomial table mismatch");
    TORCH_CHECK(min_opacity > 0 && min_opacity < 1, "invalid minimum opacity");
    const c10::cuda::CUDAGuard guard(opacity.device());
    auto new_opacity = at::empty_like(opacity);
    auto new_scales = at::empty_like(scales);
    gsplat::launch_relocation_kernel(opacity, scales, ratios, binoms, n_max,
                                     min_opacity, new_opacity, new_scales);
    return std::make_tuple(new_opacity, new_scales);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("relocation", &relocation);
}
'''


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"MCMC patch anchor is not unique: {old!r}")
    return source.replace(old, new, 1)


def patch_sources(sources: dict[str, str]) -> dict[str, str]:
    for name, digest in SOURCE_SHA256.items():
        if hashlib.sha256(sources[name].encode()).hexdigest() != digest:
            raise ValueError(f"gsplat source does not match audited wheel: {name}")
    wrapper = sources["relocation.py"]
    wrapper = replace_once(wrapper, "from .cuda._wrapper import _make_lazy_cuda_func",
                           f"from .{EXTENSION_NAME} import relocation")
    wrapper = replace_once(wrapper, "    binoms: Tensor,  # [n_max, n_max]\n",
                           "    binoms: Tensor,  # [n_max, n_max]\n    min_opacity: float = 0.005,\n")
    wrapper = replace_once(wrapper, '_make_lazy_cuda_func("relocation")(\n        opacities, scales, ratios, binoms, n_max',
                           "relocation(\n        opacities, scales, ratios, binoms, n_max, min_opacity")
    wrapper += "\nPROJECT_MCMC_RELOCATION_VERSION = 1\n"
    ops = sources["strategy/ops.py"]
    anchor = "        binoms=binoms,\n    )\n    new_opacities = torch.clamp"
    if ops.count(anchor) != 2:
        raise ValueError("both relocate and sample_add must pass the opacity floor")
    ops = ops.replace(anchor, "        binoms=binoms,\n        min_opacity=min_opacity,\n    )\n    new_opacities = torch.clamp")
    kernel = sources["cuda/csrc/RelocationCUDA.cu"]
    kernel = replace_once(kernel, '#include "Common.h"\n#include "Relocation.h"', '#include <cfloat>')
    kernel = replace_once(kernel, "    int n_max,\n", "    int n_max,\n    float min_opacity,\n")
    kernel = replace_once(kernel, "    const int n_max,\n", "    const int n_max,\n    float min_opacity,\n")
    kernel = replace_once(kernel, "                    n_max,\n", "                    n_max,\n                    min_opacity,\n")
    anchor = "    new_opacities[idx] = 1.0f - powf(1.0f - opacities[idx], 1.0f / n_idx);"
    kernel = replace_once(kernel, anchor, anchor + "\n    // The floor intentionally changes peak transparency; scales use the actual opacity.\n    new_opacities[idx] = fminf(fmaxf(new_opacities[idx], min_opacity), 1.0f - FLT_EPSILON);")
    return {"relocation.py": wrapper, "strategy/ops.py": ops,
            "cuda/csrc/RelocationCUDA.cu": kernel}


def patch_destination_moments(source: str) -> str:
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256["strategy/ops.py"]:
        raise ValueError("gsplat ops.py does not match audited wheel")
    return replace_once(source, "        v[sampled_idxs] = 0\n        return v",
                        "        v[sampled_idxs] = 0\n        v[dead_indices] = 0\n        return v") + (
        '\nPROJECT_MCMC_OVERLAY_PROFILE = "mcmc_destination_moments_overlay_v1"\n'
    )



def patch_capacity(ops: str, strategy: str) -> dict[str, str]:
    if hashlib.sha256(ops.encode()).hexdigest() != SOURCE_SHA256["strategy/ops.py"]:
        raise ValueError("gsplat ops.py does not match audited wheel")
    if hashlib.sha256(strategy.encode()).hexdigest() != MCMC_SHA256:
        raise ValueError("gsplat mcmc.py does not match audited wheel")
    strategy = replace_once(strategy, "            relocate(\n", "            n_gs = relocate(\n")
    strategy = replace_once(strategy, "            sample_add(\n", "            n_gs = sample_add(\n")
    anchor = "            n_relocated_gs = self._relocate_gs(params, optimizers, binoms)"
    strategy = replace_once(strategy, anchor, anchor +
                            '\n            state["project_mcmc_relocated_count"] = n_relocated_gs')
    return {
        "strategy/ops.py": ops + '\nfrom ._project_mcmc_capacity import relocate, sample_add\n'
                          'PROJECT_MCMC_OVERLAY_PROFILE = "mcmc_capacity_overlay_v1"\n',
        "strategy/mcmc.py": strategy,
        "strategy/_project_mcmc_capacity.py": Path(__file__).with_name("mcmc_capacity_ops.py").read_text(),
    }


def prepare(output_dir: Path, variant: str = "relocation_floor") -> dict:
    spec = importlib.util.find_spec("gsplat")
    if spec is None or spec.origin is None:
        raise ValueError("the pinned gsplat wheel must already be installed")
    package = Path(spec.origin).resolve().parent
    if variant not in {"relocation_floor", "destination_moments", "capacity"}:
        raise ValueError("unknown MCMC overlay variant")
    sources = {name: (package / name).read_text() for name in SOURCE_SHA256}
    patched = patch_sources(sources)
    if variant == "destination_moments":
        patched = {"strategy/ops.py": patch_destination_moments(sources["strategy/ops.py"])}
    audited = dict(SOURCE_SHA256)
    if variant == "capacity":
        patched = patch_capacity(sources["strategy/ops.py"], (package / "strategy/mcmc.py").read_text())
        audited["strategy/mcmc.py"] = MCMC_SHA256
    if not list(package.glob("csrc*.so")):
        raise ValueError("the audited precompiled rasterizer is required; do not rebuild it")
    output_dir.mkdir(parents=True, exist_ok=False)
    destination = output_dir / "gsplat"
    shutil.copytree(package, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name, source in patched.items():
        (destination / name).write_text(source)
    if variant == "relocation_floor":
        binding_path = destination / f"{EXTENSION_NAME}.cpp"
        binding_path.write_text(BINDING)
        build_dir = output_dir / "build"
        build_dir.mkdir()
        os.environ["MAX_JOBS"] = "1"
        from torch.utils.cpp_extension import load

        extension = load(
            name=EXTENSION_NAME,
            sources=[str(binding_path), str(destination / "cuda/csrc/RelocationCUDA.cu")],
            build_directory=str(build_dir), extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=True,
        )
        shutil.copy2(extension.__file__, destination / Path(extension.__file__).name)
    files = {
        str(path.relative_to(output_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".so", ".cpp", ".cu", ".h"}
    }
    record = {
        "profile": f"mcmc_{variant}_overlay_v1", "source_package": str(package),
        "audited_source_sha256": audited, "files": files,
        "installed_package_modified": False, "rasterizer_binary_reused": True,
        "quality_status": "not_evaluated",
        "reference_kernel_sha256": "9f77224f21bf62f7440494248c4549a2af0556cb2540ae7964af7b8faff50eda",
    }
    (output_dir / "overlay.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("relocation_floor", "destination_moments", "capacity"), default="relocation_floor")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir.resolve(), args.variant), indent=2))


if __name__ == "__main__":
    main()
