"""Baseline-policy traces from the foundation's `policies_v5` on the io
goldens' fixture splits, for hf-policies' tests: every policy's expansion
order, expansions, registration index, stop reason, parents and route, plus
the fixture embeddings the greedy policy used (dict-of-lists, float64 — the
exact path a Rust implementation can match bit for bit).

Run with the hippocampus-foundation environment:
    cd ../hippocampus-foundation && PYTHONPATH=src:scripts uv run python ../hippo-13/tools/goldens/gen_policies_goldens.py
Writes crates/hf-policies/tests/goldens/policies.json. Synthetic; never evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

from hippocampus_foundation.read_run.io_v5 import read_real_split_v5
from hippocampus_foundation.read_run.policies_v5 import POLICIES_V5, policy_row_v5
from real_walk_stage0 import fixture_world

HERE = Path(__file__).resolve()
SPLITS = HERE.parents[2] / "crates" / "hf-io" / "tests" / "goldens" / "fixture-split"
OUT = HERE.parents[2] / "crates" / "hf-policies" / "tests" / "goldens"


def route_or_none(trace, start, target):
    try:
        return trace.route(start, target)
    except KeyError:  # bidirectional sets no parents; its route is undefined
        return None


def trace_json(trace, episode):
    start = episode.visible["start_node"]
    target = episode.hidden["target_set"][0]
    return {
        "examined": list(trace.examined),
        "expansions": trace.expansions,
        "registered_at": trace.registered_at,
        "stop_reason": trace.stop_reason,
        "parents": dict(trace.parents),
        "route": route_or_none(trace, start, target),
        "row": policy_row_v5(episode, trace),
    }


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    _graph, embeddings = fixture_world(5)
    out = {"splits": {}, "embeddings": {}}
    nodes = set()
    for name in ("train", "screen", "train-greedy"):
        episodes = list(read_real_split_v5(SPLITS / name))
        records = []
        for e in episodes:
            nodes.update(n["node"] for n in e.visible["nodes"])
            traces = {policy: trace_json(fn(e, embeddings=embeddings), e) for policy, fn in POLICIES_V5.items()}
            records.append({"episode_id": e.episode_id, "traces": traces})
        out["splits"][name] = records
    out["embeddings"] = {n: embeddings[n] for n in sorted(nodes)}
    (OUT / "policies.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("wrote", OUT, len(nodes), "nodes")
