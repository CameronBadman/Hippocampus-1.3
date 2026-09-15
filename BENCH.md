# Bench record

Measured numbers behind the plan's targets, with the command that produced
each. Regenerable; the foundation's `audit/VERIFICATION.md` carries the ones
that gate a step. Machine: 32 cores, 30 GB, RTX 5070 Ti 16 GB.

| date | what | command | result | Python reference |
|---|---|---|---|---|
| 2026-09-15 | Wikidata5M structure load (4,594,149 nodes, 20,614,279 edges) + p99 out-degree + 2,000 rung-3 balls | `cargo run --release -p hf-graph --example load -- <foundation>/private/real-walk-v1/graphs/wikidata5m/edges.tsv 99` | load 19.1 s; p99 = 19 (the recorded rung-3 hub cap); 2,000 balls of 40 in 0.07 s (mean size 39.8); peak RSS 3,455 MB | `graph_v5.from_triples` structure-only: minutes, several GB (the 2026-09-08 note) |
