"""Model goldens from the foundation's `model_v5` + `training_v5` for hf-model's
tests: a small fixture-configured model with the greedy prior, its weights
perturbed from a seeded normal so the residual path is exercised, exported
as safetensors (written by hand: no extra dependency); the walk it takes on
fixture episodes; `score_decisions` inputs and outputs (scores, residuals) on
those decisions; stop-head logits; `walk_losses` on a four-episode batch with
the residual penalty; the runner's `_state_digest`; and the trainable
parameter counts of the fixture config and the rung-3 prior config.

Run with the hippocampus-foundation environment:
    cd ../hippocampus-foundation && PYTHONPATH=src:scripts uv run python ../hippo-13/tools/goldens/gen_model_goldens.py
Writes crates/hf-model/tests/goldens/{model.json,fixture.safetensors}. Synthetic; never evidence.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import torch
from torch.nn import functional

from hippocampus_foundation.read_run.io_v5 import read_real_split_v5
from hippocampus_foundation.read_run.model_v5 import (
    RealEpisodeIndex,
    build_read_model_v5,
    trainable_parameter_count_v5,
)
from hippocampus_foundation.read_run.training_v5 import walk_losses
from real_walk_stage0 import _state_digest, fixture_world

HERE = Path(__file__).resolve()
SPLITS = HERE.parents[2] / "crates" / "hf-io" / "tests" / "goldens" / "fixture-split"
OUT = HERE.parents[2] / "crates" / "hf-model" / "tests" / "goldens"
FIXTURE_CONFIG = HERE.parents[3] / "hippocampus-foundation" / "experiments" / "real_walk_v1" / "training-config.stage0.fixture.json"
RUNG3_CONFIG = HERE.parents[3] / "hippocampus-foundation" / "experiments" / "real_walk_v1" / "training-config.stage0.wikidata5m-r3-g0.25-prior.json"


def save_safetensors(state: dict[str, torch.Tensor], path: Path) -> None:
    header = {}
    blobs = []
    offset = 0
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        if t.dtype != torch.float32:
            raise SystemExit(f"{name}: only float32 is exported ({t.dtype})")
        data = t.numpy().tobytes()
        header[name] = {"dtype": "F32", "shape": list(t.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    h = json.dumps(header, separators=(",", ":")).encode()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(5)
    model_cfg = json.loads(FIXTURE_CONFIG.read_text())["model"]
    model_cfg = {**model_cfg, "greedy_prior": True, "greedy_prior_scale": 10.0, "embedding_dimension": 8}
    model = build_read_model_v5({"model": model_cfg})
    gen = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name == "greedy_tau":
                continue
            p.copy_(torch.randn(p.shape, generator=gen) * 0.2)
    model.eval()
    state = model.state_dict()
    save_safetensors(state, OUT / "fixture.safetensors")

    _graph, embeddings = fixture_world(5)
    episodes = list(read_real_split_v5(SPLITS / "train"))[:6] + list(read_real_split_v5(SPLITS / "screen"))[:2]
    indexes = [RealEpisodeIndex(e, embeddings, embedding_dimension=8) for e in episodes]
    walks = []
    items = []
    for index in indexes:
        walk = model.walk(index, stop_rule="exhaust", record_candidates=True)
        walks.append(
            {
                "episode_id": index.episode.episode_id,
                "expanded": walk["expanded"],
                "registered_at": walk["registered_at"],
                "stop_reason": walk["stop_reason"],
                "examined": walk["examined"],
                "margins": walk["margins"],
                "cosine_margins": walk["cosine_margins"],
                "residuals": walk["residuals"],
                "candidates": walk["candidates"],
                "stop_features": [d.stop_features for d in walk["decisions"]],
            }
        )
        for d in walk["decisions"]:
            items.append((index, d, walk["expanded"][: d.expansions_before]))
    with torch.no_grad():
        scores, residuals = model.score_decisions(items, with_residual=True)
        stop = model.stop_logits([d.stop_features for _i, d, _c in items])
    decisions = []
    for k, (index, d, context) in enumerate(items):
        n = len(d.frontier)
        decisions.append(
            {
                "episode_id": index.episode.episode_id,
                "frontier": d.frontier,
                "parents": d.parents,
                "depths": d.depths,
                "context": context,
                "rows": [[float(x) for x in r] for r in d.rows],
                "scores": scores[k][:n].tolist(),
                "residuals": residuals[k][:n].tolist(),
                "stop_logit": float(stop[k]),
                "chosen": d.chosen,
            }
        )
    # the learned stop rule on the same episodes
    learned = []
    for index in indexes:
        w = model.walk(index, stop_rule="learned")
        learned.append({"episode_id": index.episode.episode_id, "expanded": w["expanded"], "stop_reason": w["stop_reason"]})
    # losses on a batch of four, with and without the residual penalty
    model.train()
    batch = indexes[:4]
    out = model(batch, stop_rule="exhaust")
    losses = {}
    for name, kw in (("plain", {}), ("penalty", {"residual_penalty": 0.5})):
        l = walk_losses(torch, functional, out, [i.episode.hidden for i in batch], **kw)
        losses[name] = {k: float(v) for k, v in l.items()}
    # gradient norm of the total under the penalty (pre-clip), for the optimiser check
    model.zero_grad()
    l = walk_losses(torch, functional, out, [i.episode.hidden for i in batch], residual_penalty=0.5)
    l["total"].backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    rung3_cfg = json.loads(RUNG3_CONFIG.read_text())["model"]
    rung3 = build_read_model_v5({"model": {**rung3_cfg, "embedding_dimension": 768}})
    golden = {
        "model_config": model_cfg,
        "state_digest": _state_digest(state),
        "state_digest_excluding_tau": _state_digest(state, exclude={"greedy_tau"}),
        "parameter_count": trainable_parameter_count_v5(model),
        "rung3_parameter_count": trainable_parameter_count_v5(rung3),
        "rung3_config": rung3_cfg,
        "walks": walks,
        "decisions": decisions,
        "learned": learned,
        "losses": losses,
        "grad_norm_penalty": grad_norm,
        "batch_episode_ids": [i.episode.episode_id for i in batch],
    }
    (OUT / "model.json").write_text(json.dumps(golden, indent=1) + "\n")
    print("wrote", OUT, "decisions", len(decisions), "params", golden["parameter_count"], "rung3", golden["rung3_parameter_count"])
