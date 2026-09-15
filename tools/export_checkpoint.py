"""Export a Python runner checkpoint (`checkpoint.pt`, torch's pickle format)
to the engine's layout: `checkpoint.safetensors` (the state_dict, float32) and
`checkpoint.json` (config, seed, train_draws, git head, pin) — so the Rust
runner can re-evaluate the Python checkpoints under `raw-v5` (the step-12
gate) without reading pickles.

Run with the hippocampus-foundation environment:
    cd ../hippocampus-foundation && uv run python ../hippo-13/tools/export_checkpoint.py <run dir> <out dir>
Nothing under private/ is copied anywhere else; the output stays beside the run.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import torch


def save_safetensors(state: dict, path: Path) -> None:
    header, blobs, offset = {}, [], 0
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        dtype = {torch.float32: "F32", torch.int64: "I64", torch.bool: "BOOL", torch.float64: "F64"}[t.dtype]
        data = t.numpy().tobytes()
        header[name] = {"dtype": dtype, "shape": list(t.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    h = json.dumps(header, separators=(",", ":")).encode()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


if __name__ == "__main__":
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    if "holdout" in str(run).lower() or "heldout" in str(run).lower():
        raise SystemExit("refusing a holdout path")
    saved = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    out.mkdir(parents=True, exist_ok=True)
    save_safetensors(saved["state_dict"], out / "checkpoint.safetensors")
    meta = {
        "record_kind": "hippo13_checkpoint_v1",
        "exported_from": str(run / "checkpoint.pt"),
        "config": saved.get("config"),
        "seed": saved.get("seed"),
        "train_draws": saved.get("train_draws"),
        "update": saved.get("update"),
        "preregistration_commit": saved.get("preregistration_commit"),
        "git_head": saved.get("git_head"),
        "training_authorized": False,
    }
    (out / "checkpoint.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print("exported", out, "tensors", len(saved["state_dict"]))
