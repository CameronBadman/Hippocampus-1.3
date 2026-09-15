# hippo-13 — the real-walk engine in Rust

The stage-0 real-network traversal pipeline of
[`hippocampus-foundation`](../hippocampus-foundation) — episode sampler,
split I/O, embedding cache, the batched walk, the model and its training —
rewritten in Rust after rung 3 of the ladder was declared not admissible at
the Python substrate (2026-09-14). The foundation keeps governance, the
preregistration discipline, the readers and every byte of private data; this
workspace produces command-line binaries that read and write the same
artifacts those readers consume.

Plan: the foundation's plan record ("The real-walk redesign in Rust — plan,
2026-09-15"). Decisions fixed there: everything in Rust; redesign, not a port;
`tch-rs` on the libtorch shipped inside the foundation venv's torch wheel
(2.13.0+cu130); single-target first with k targets designed in;
implementation first, the training process as the next step.

## Layout

```
crates/hf-core      canonical JSON (RFC 8785), sha256:<hex>, CPython-compatible
                    random.Random, exclusive files with the foundation's modes,
                    Python-style pretty JSON, typed errors → exit 2
crates/hf-model     the model on libtorch via tch (step 0: the toolchain gate)
tools/env.sh        exports LIBTORCH from the foundation venv; source before cargo
tools/goldens/      CPython scripts that generate the cross-language goldens
```

## Build and check

```console
source tools/env.sh            # LIBTORCH=<foundation venv>/site-packages/torch
cargo build --release
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo nextest run
./target/release/hf-smoke      # cuda available: true, matmul + backward on the GPU
```

Nothing is downloaded: `torch-sys` links the wheel's `libtorch` and the
binaries carry an rpath to it (see `crates/hf-model/build.rs`, which also
force-links `libtorch_cuda` — the linker otherwise drops it as unreferenced
and CUDA comes up unavailable).

## Rules carried over

Real graphs only. Removals by a fixed rule, never model-dependent. No holdout
path is ever touched (`refuse_holdout`). Hidden streams 0600, visible 0644.
`training_authorized: false` in every artifact. No daemon is started by any
binary. Commits only after the checks above pass, with explicit `git add`.
