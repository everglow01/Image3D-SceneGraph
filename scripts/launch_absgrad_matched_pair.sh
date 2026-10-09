#!/usr/bin/env bash
# Submit via the panel job channel; this file does not grant experiment authorization.
set -euo pipefail
repo=/usr/local/3dgs_new/Image3D-SceneGraph
unit=image3d-absgrad-matched-20261009-v1.service
memory_max=12G; swap_max=0; continuation=()
if [[ "${1:-}" == --continue-authorized || "${1:-}" == --watch-continued ]]; then
 unit=image3d-absgrad-continued-20261009-v1.service
 memory_max=infinity; swap_max=infinity
 continuation=(--continue-recovered-signed)
fi
if [[ "${1:-}" == --help ]]; then
 echo '用法：--watch/--watch-continued 输出目录；或 --execute-authorized/--continue-authorized 输出目录 合同 合同SHA 提交SHA 历史signed目录 overlay目录'
 exit 0
fi
test "$(hostname)" = i-94B8D131
cd "$repo"
mode=${1:?缺少操作}; out=${2:?缺少输出目录}
python3 -I - "$out" "$repo" <<'PY'
from pathlib import Path
import sys
out, repo = map(Path, sys.argv[1:])
assert out.is_absolute() and out == out.resolve() and out.is_relative_to(repo / 'outputs/experiments')
assert out != repo / 'outputs/experiments'
PY
if [[ "$mode" == --watch || "$mode" == --watch-continued ]]; then
 for step in $(seq 1 5760); do
  date -u
  if test -f "$out/exit-code"; then
   cat "$out/exit-code"
   test "$(cat "$out/exit-code")" = 0 && test -f "$out/experiment/complete.json"
   exit $?
  fi
  if test -f "$out/driver.log"; then tail -n 5 "$out/driver.log"; fi
  state=$(systemctl show "$unit" -p ActiveState --value)
  systemctl show "$unit" -p ActiveState -p Result -p ExecMainStatus
  if [[ "$step" -gt 2 && "$state" != active && "$state" != activating ]]; then
   echo '服务已不运行且缺少实验终态；不能判为成功'
   exit 3
  fi
  sleep 30
 done
 exit 124
fi
[[ "$mode" == --execute-authorized || "$mode" == --continue-authorized ]]
test "$#" = 7
gate=$3; gate_sha=$4; expected=$5; historical=$6; overlay=$7
test "$(git rev-parse HEAD)" = "$expected"
git diff --exit-code HEAD
test "$(sha256sum "$gate" | cut -d ' ' -f 1)" = "$gate_sha"
test -d "$overlay/gsplat"
exec 9>"$out.lock"
flock -n 9
test ! -e "$out.once"
test ! -e "$out"
test ! -L "$out"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test "$(awk '$1=="MemAvailable:" {print $2}' /proc/meminfo)" -ge 18874368
mkdir "$out"
set -C
finish() {
 rc=$?
 trap - EXIT
 if test "$rc" -eq 0 && ! test -f "$out/experiment/complete.json"; then rc=1; fi
 printf '%s\n' "$rc" > "$out/exit-code"
 date -u > "$out/end.utc"
 exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
date -u > "$out.once"
git rev-parse HEAD > "$out/code-sha.txt"
nvidia-smi -L > "$out/gpus.txt"
set +e
systemd-run --wait --pipe --unit="$unit" \
 --property="MemoryMax=$memory_max" --property=MemoryHigh=infinity --property="MemorySwapMax=$swap_max" \
 --property=OOMPolicy=kill --property=KillMode=control-group \
 --property=OOMScoreAdjust=500 --property=TimeoutStopSec=5 \
 --working-directory="$repo" \
 --setenv="PYTHONPATH=$overlay:$repo/src:$repo/scripts" \
 --setenv=CUDA_VISIBLE_DEVICES=0,1 --setenv=OMP_NUM_THREADS=1 \
 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
 --setenv=PYTHONUNBUFFERED=1 \
 "$repo/.venv/bin/python" scripts/run_absgrad_matched_pair.py "${continuation[@]}" \
 --output-dir "$out/experiment" --historical-signed "$historical" \
 --gate-contract "$gate" --gate-sha256 "$gate_sha" --expected-revision "$expected" \
 > "$out/driver.log" 2>&1
rc=$?
set -e
systemctl show "$unit" -p Result -p ExecMainCode -p ExecMainStatus -p ActiveState \
 > "$out/service-terminal.txt" || true
exit "$rc"
