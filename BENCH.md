# Bench record

Measured numbers behind the plan's targets, with the command that produced
each. Regenerable; the foundation's `audit/VERIFICATION.md` carries the ones
that gate a step. Machine: 32 cores, 30 GB, RTX 5070 Ti 16 GB.

| date | what | command | result | Python reference |
|---|---|---|---|---|
| 2026-09-15 | Wikidata5M structure load (4,594,149 nodes, 20,614,279 edges) + p99 out-degree + 2,000 rung-3 balls | `cargo run --release -p hf-graph --example load_graph -- <foundation>/private/real-walk-v1/graphs/wikidata5m/edges.tsv 99` | load 19.1 s; p99 = 19 (the recorded rung-3 hub cap); 2,000 balls of 40 in 0.07 s (mean size 39.8); peak RSS 3,455 MB | `graph_v5.from_triples` structure-only: minutes, several GB (the 2026-09-08 note) |
| 2026-09-15 | rung-3 pool `wikidata5m-n40-d3-k2-r3-g0.25/train` (40,000 episodes, 1,401,529 node records) validated and read | `cargo run --release -p hf-io --example load_split -- <split>/train` | 2.70 s; peak RSS 841 MB | `read_real_split_v5_lazy` 17.2 s; the runner's `load_split_from_disk` with screens 61 s of an 80 s startup |
| 2026-09-15 | embedding cache `wikidata5m/nomic-embed-text` (313,310 × 768) | `cargo run --release -p hf-embed --example load_cache -- <cache>` | first load (parse + sidecar) 9.00 s; mapped sidecar 1.84 s; peak RSS 1,882 MB | `load_embedding_matrix` 21.1 s |
