#!/usr/bin/env bash
set -euo pipefail

if (( $# > 1 )); then
    printf '%s\n' '用法：bash scripts/open_rtx_browser.sh [http(s)://项目地址]' >&2
    exit 2
fi
url=${1:-http://localhost:8081/}
case "$url" in
    http://?*|https://?*) ;;
    *) printf '%s\n' '项目地址必须以 http:// 或 https:// 开头。' >&2; exit 2 ;;
esac

exec env __NV_PRIME_RENDER_OFFLOAD=1 \
    __GLX_VENDOR_LIBRARY_NAME=nvidia \
    __VK_LAYER_NV_optimus=NVIDIA_only \
    google-chrome-stable \
    --ozone-platform=x11 \
    --use-angle=vulkan \
    --user-data-dir="$HOME/.cache/image3d-rtx4060-chrome" \
    --new-window "$url"
