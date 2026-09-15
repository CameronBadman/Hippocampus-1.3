"""A Python-written fixture split for hf-io's tests, from the foundation's own
sampler and writer (`sample_episode`, `write_real_split_v5`) on the runner's
synthetic world — so the Rust reader is tested on bytes the Python writer
produced, including a greedy-path split carrying `removal_recipe`,
`greedy_overshoot` and `sampler.greedy_share`.

Run with the hippocampus-foundation environment:
    cd ../hippocampus-foundation && PYTHONPATH=src:scripts uv run python ../hippo-13/tools/goldens/gen_io_goldens.py
Writes crates/hf-io/tests/goldens/fixture-split/{train,screen,train-greedy}/ and
expected.json. Synthetic; never evidence.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

from hippocampus_foundation.read_run.episodes_v5 import SamplerConfig, sample_episode
from hippocampus_foundation.read_run.io_v5 import RealEpisode, write_real_split_v5
from real_walk_stage0 import fixture_world

HERE = Path(__file__).resolve()
OUT = HERE.parents[2] / "crates" / "hf-io" / "tests" / "goldens" / "fixture-split"

GRAPH_MANIFEST = {
    "record_kind": "real_graph_manifest_v5",
    "family": "fixture",
    "source": {"dataset": "synthetic fixture world", "seed": 5},
    "node_count": 400,
    "edge_count": 1600,
    "typed": False,
    "text_coverage": 1.0,
    "edges_sha256": "sha256:" + "0" * 64,
    "text_sha256": "sha256:" + "1" * 64,
    "training_authorized": False,
}


def sample(graph, sampler, split, count, embeddings=None):
    episodes, dropped, index = [], {}, 0
    while len(episodes) < count and index < count * 40:
        e = sample_episode(graph, sampler, index=index, split=split, embeddings=embeddings)
        index += 1
        if isinstance(e, RealEpisode):
            episodes.append(e)
        else:
            dropped[e["dropped"]] = dropped.get(e["dropped"], 0) + 1
    return episodes, dropped, index


def write(name, graph, sampler, split, count, embeddings=None):
    episodes, dropped, attempts = sample(graph, sampler, split, count, embeddings)
    nodes = set()
    for e in episodes:
        for record in e.visible["nodes"]:
            nodes.add(record["node"])
            record["text"] = ""  # as real_walk_write_splits does: text lives in texts.jsonl
    destination = OUT / name
    public, private = write_real_split_v5(
        family="fixture",
        stage="stage0_known_target",
        split=split,
        destination=destination,
        episodes=iter(episodes),
        graph_manifest=GRAPH_MANIFEST,
        sampler={**dataclasses.asdict(sampler), "text_mode": "texts.jsonl", "hub_percentile": None},
    )
    with (destination / "texts.jsonl").open("w") as f:
        for node in sorted(nodes):
            f.write(json.dumps({"node": node, "text": graph.text[node]}) + "\n")
    (destination / "sampling.json").write_text(
        json.dumps(
            {"attempts": attempts, "kept": len(episodes), "drops": dropped, "distinct_nodes": len(nodes), "texts_written": len(nodes), "training_authorized": False},
            indent=2,
            sort_keys=True,
        )
    )
    (destination / "nodes.txt").write_text("\n".join(sorted(nodes)) + "\n")
    return {
        "episode_ids": [e.episode_id for e in episodes],
        "public_manifest": public,
        "private_manifest": private,
        "first_visible": episodes[0].visible,
        "first_hidden": episodes[0].hidden,
        "attempts": attempts,
        "dropped": dropped,
    }


if __name__ == "__main__":
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    graph, embeddings = fixture_world(5)
    sampler = SamplerConfig(family="fixture", subgraph_size=64, target_distance=3, removal_level=2, cost_epsilon=0.5)
    greedy = dataclasses.replace(sampler, removal_rule="greedy-path", greedy_share=0.5)
    expected = {
        "train": write("train", graph, sampler, "train", 12),
        "screen": write("screen", graph, sampler, "screen", 6),
        "train-greedy": write("train-greedy", graph, greedy, "train", 12, embeddings),
        "graph_manifest": GRAPH_MANIFEST,
    }
    (OUT / "expected.json").write_text(json.dumps(expected, indent=1, sort_keys=True) + "\n")
    print("wrote", OUT)
