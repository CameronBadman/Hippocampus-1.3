#!/usr/bin/env bash
# The engine's pre-commit checks: format, clippy with warnings denied, every test.
# LIBTORCH comes from .cargo/config.toml (the sibling foundation venv) or tools/env.sh.
set -eu; cd "$(dirname "$0")/.."
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo nextest run
