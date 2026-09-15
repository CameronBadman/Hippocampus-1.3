"""Graph goldens from the foundation's `graph_v5` for hf-graph's tests.

Run with the hippocampus-foundation environment:
    cd ../hippocampus-foundation && PYTHONPATH=src uv run python ../hippo-13/tools/goldens/gen_graph_goldens.py
Writes crates/hf-graph/tests/goldens/{fixture.edges.tsv,fixture.text.tsv,graph.json}.
The fixture is the runner's synthetic world (400 nodes, 1,600 edges, seed 5) with
a typed variant; nothing here is evidence.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from hippocampus_foundation.read_run.graph_v5 import RealGraph

HERE = Path(__file__).resolve()
OUT = HERE.parents[2] / "crates" / "hf-graph" / "tests" / "goldens"


def fixture_edges(seed: int, *, typed: bool) -> list[tuple[str, str | None, str]]:
    rng = random.Random(seed)
    edges: set[tuple[str, str | None, str]] = set()
    relations = ["P1", "P2", "P3", ""] if typed else [None]
    while len(edges) < 1600:
        head, tail = rng.randrange(400), rng.randrange(400)
        if head == tail:
            continue
        edges.add((f"n{head}", rng.choice(relations), f"n{tail}"))
    # a few parallel relations between the same pair, so both dedups are exercised
    if typed:
        edges.add(("n1", "P1", "n2"))
        edges.add(("n1", "P2", "n2"))
        edges.add(("n1", "", "n2"))
    return sorted(edges, key=str)


def write_tsv(path: Path, edges):
    with path.open("w") as f:
        for h, r, t in edges:
            f.write(f"{h}\t{r or ''}\t{t}\n")


def goldens(graph: RealGraph, name: str, *, text: dict[str, str]):
    out = {"name": name, "node_count": len(graph.nodes()), "edge_count": graph.edge_count(), "typed": graph.typed}
    out["percentiles"] = {str(p): graph.out_degree_percentile(p) for p in (0, 50, 90, 95, 99, 99.5, 100)}
    starts = sorted(graph.out)[:6]
    out["balls"] = []
    for start in starts:
        for size, cap in ((10, None), (40, None), (40, 3), (64, 2), (400, None)):
            ball = graph.ball(start, size, hub_cap=cap)
            out["balls"].append({"start": start, "size": size, "hub_cap": cap, "ball": ball})
        allow = lambda n: int(n[1:]) % 3 != 0  # noqa: E731
        out["balls"].append({"start": start, "size": 40, "hub_cap": 3, "allow_mod3": True, "ball": graph.ball(start, 40, hub_cap=3, allow=allow)})
    out["neighbours"] = {n: graph.out_neighbours(n) for n in starts}
    out["paths"] = []
    for start in starts[:3]:
        sub = graph.induced(graph.ball(start, 40, hub_cap=None))
        from_start = sub.distances_from(start)
        for d in (2, 3):
            targets = sorted(n for n, dist in from_start.items() if dist == d)[:2]
            for target in targets:
                for max_cost in (d, d + 1, d + 2):
                    paths = sub.simple_paths(start, target, max_cost)
                    out["paths"].append({"start": start, "target": target, "max_cost": max_cost, "ball": sub.nodes(), "paths": [list(p) for p in paths], "brute": [list(p) for p in sub.brute_force_paths(start, target, max_cost)]})
        out["paths"].append({"start": start, "distances_from": from_start, "distances_to": sub.distances_to(start)})
    out["text_attached"] = graph.load_text(OUT / f"{name}.text.tsv") if text else 0
    out["text_sample"] = {n: graph.text.get(n) for n in starts}
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, typed in (("fixture", False), ("fixture-typed", True)):
        edges = fixture_edges(5, typed=typed)
        write_tsv(OUT / f"{name}.edges.tsv", edges)
        text = {f"n{i}": f"node {i}" for i in range(0, 400, 2)}  # half the nodes have text
        with (OUT / f"{name}.text.tsv").open("w") as f:
            for node in sorted(text):
                f.write(f"{node}\t{text[node]}\n")
            f.write("n9999\tunknown node\n")
        graph = RealGraph.from_triples(OUT / f"{name}.edges.tsv", family=name)
        result[name] = goldens(graph, name, text=text)
    (OUT / "graph.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print("wrote", OUT)
