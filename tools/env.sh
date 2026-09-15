#!/usr/bin/env bash
# Source before `cargo build`: points torch-sys at the libtorch shipped inside the
# hippocampus-foundation venv's torch wheel (2.13.0+cu130), so tch-rs links the very
# library PyTorch uses. Nothing is downloaded. Usage: source tools/env.sh
HF_FOUNDATION=${HF_FOUNDATION:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../hippocampus-foundation" && pwd)"}
export HF_FOUNDATION
export LIBTORCH="$(cd "$HF_FOUNDATION" && uv run python -c 'import torch, os; print(os.path.dirname(torch.__file__))')"
export LIBTORCH_CXX11_ABI=1
echo "LIBTORCH=$LIBTORCH"
