"""Golden outputs from CPython for hf-core's cross-language tests.

Run with the hippocampus-foundation environment (it has `rfc8785`):
    cd ../hippocampus-foundation && uv run python ../hippo-13/tools/goldens/gen_core_goldens.py
Writes crates/hf-core/tests/goldens/{canonical,pyrandom}.json. Nothing here is
evidence; the corpus is synthetic and committed so the Rust tests need no Python.
"""

from __future__ import annotations

import base64
import json
import random
from pathlib import Path

import rfc8785

HERE = Path(__file__).resolve()
OUT = HERE.parents[2] / "crates" / "hf-core" / "tests" / "goldens"

CANONICAL_CASES = [
    {"b": 1, "a": 2, "aa": 3, "A": 4, "é": 5, "z": {"y": [3, 2, 1], "x": {}}},
    {
        "floats": [
            1.0, 0.1, 0.5, 1e21, 1e20, 1e-7, 1e-6, 5e-324, 1.7976931348623157e308,
            -0.0, 0.0, 123456789012345680000.0, 3.141592653589793, 100.0, 1e300, 2.5e-5,
        ]
    },
    # rfc8785 refuses integers outside the IEEE-754 safe domain (|n| > 2**53 - 1); the Rust
    # side must refuse them too, so the corpus stops at the edge and a test covers the refusal.
    {"ints": [0, 1, -1, 9007199254740991, -9007199254740991, 2**32, -(2**32), 1234567890123]},
    {
        "strings": [
            "", "a", chr(0), "\n\t\r\b\f", '"quoted"', "back\\slash", "  ",
            "\U0001F600", "café", chr(31), chr(127), "",
        ]
    },
    {"keys": {"€": 1, "\U0001F600": 2, "～": 3, "a": 4, "é": 5, "Z": 6, "10": 7, "9": 8, "": 9}},
    {
        "episode_id": "wikidata5m-train-000001-2b061859",
        "visible": {
            "schema_version": "5.0.0",
            "nodes": [{"node": "Q1", "text": ""}],
            "edges": [{"edge_id": 0, "source": "Q1", "target": "Q2", "relation": None}],
            "removal_level": 2,
            "subgraph_size": 40,
        },
    },
    {"nested": [[], {}, [[]], [{}], {"a": []}, {"a": {}}, [None, True, False]]},
    {
        "sampler": {
            "cost_epsilon": 0.5, "family": "wikidata5m", "hub_degree_cap": 19, "max_paths": 512,
            "removal_level": 2, "screen_region": 0.5, "seed_label": "real-walk-v1",
            "subgraph_size": 40, "target_distance": 3,
        }
    },
]


def canonical_goldens():
    cases = []
    for value in CANONICAL_CASES:
        data = rfc8785.dumps(value)
        cases.append({"value": value, "canonical_b64": base64.b64encode(data).decode("ascii")})
    return {"cases": cases}


def bits(rng, k, n):
    return [rng.getrandbits(k) for _ in range(n)]


def train_sample_indices(n, k, base=5000, seed=20260912):
    k = min(k, n)
    if k <= base:
        return sorted(random.Random(seed).sample(range(n), k))
    head = random.Random(seed).sample(range(n), min(base, n))
    remaining = sorted(set(range(n)) - set(head))
    tail = random.Random(seed + 1).sample(remaining, k - len(head))
    return sorted(head + tail)


def pyrandom_goldens():
    out = {"seeds": {}}
    seeds = (0, 1, 1729, 2718, 3141, 20260912, 2**32 - 1, 2**32, 2**64 + 12345, 123456789012345678901234567890)
    for seed in seeds:
        rng = random.Random(seed)
        entry = {
            "genrand_uint32": bits(rng, 32, 16),
            "getrandbits_5": bits(rng, 5, 8),
            "getrandbits_64": [str(v) for v in bits(rng, 64, 4)],
            "getrandbits_100": [str(v) for v in bits(rng, 100, 2)],
            "randrange_40000": [rng.randrange(40000) for _ in range(16)],
            "randbelow_7": [rng._randbelow(7) for _ in range(16)],
            "sample_12_4": rng.sample(range(12), 4),
            "sample_30_8": rng.sample(range(30), 8),
            "sample_85_8": rng.sample(range(85), 8),
            "sample_86_8": rng.sample(range(86), 8),
            "sample_40000_8_first_20": [rng.sample(range(40000), 8) for _ in range(20)],
        }
        state = rng.getstate()
        entry["state_after"] = {"mt": list(state[1][:624]), "index": state[1][624]}
        entry["after_state_randrange_40000"] = [rng.randrange(40000) for _ in range(8)]
        out["seeds"][str(seed)] = entry
    replay = {}
    for seed in (1729, 2718, 3141):
        rng = random.Random(seed)
        counts = {}
        first = []
        batch = []
        for update in range(5000):
            batch = rng.sample(range(40000), 8)
            if update < 3:
                first.append(batch)
            for i in batch:
                counts[i] = counts.get(i, 0) + 1
        histogram = {}
        for c in counts.values():
            histogram[str(c)] = histogram.get(str(c), 0) + 1
        replay[str(seed)] = {
            "first_batches": first,
            "distinct_seen": len(counts),
            "views_histogram": dict(sorted(histogram.items(), key=lambda kv: int(kv[0]))),
            "last_batch": batch,
        }
    out["train_draw_replay"] = replay
    ts = train_sample_indices(40000, 10000)
    out["train_sample_40000_10000"] = {
        "first_50": ts[:50],
        "last_50": ts[-50:],
        "count": len(ts),
        "sum": sum(ts),
        "first_50_of_5000": train_sample_indices(40000, 5000)[:50],
    }
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "canonical.json").write_text(json.dumps(canonical_goldens(), indent=1, ensure_ascii=False) + "\n")
    (OUT / "pyrandom.json").write_text(json.dumps(pyrandom_goldens(), indent=1) + "\n")
    print("wrote", OUT)
