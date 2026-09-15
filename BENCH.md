# Bench record

Measured numbers behind the plan's targets, with the command that produced
each. Regenerable; the foundation's `audit/VERIFICATION.md` carries the ones
that gate a step. Machine: 32 cores, 30 GB, RTX 5070 Ti 16 GB.

| date | what | command | result | Python reference |
|---|---|---|---|---|
| 2026-09-15 | Wikidata5M structure load (4,594,149 nodes, 20,614,279 edges) + p99 out-degree + 2,000 rung-3 balls | `cargo run --release -p hf-graph --example load_graph -- <foundation>/private/real-walk-v1/graphs/wikidata5m/edges.tsv 99` | load 19.1 s; p99 = 19 (the recorded rung-3 hub cap); 2,000 balls of 40 in 0.07 s (mean size 39.8); peak RSS 3,455 MB | `graph_v5.from_triples` structure-only: minutes, several GB (the 2026-09-08 note) |
| 2026-09-15 | rung-3 pool `wikidata5m-n40-d3-k2-r3-g0.25/train` (40,000 episodes, 1,401,529 node records) validated and read | `cargo run --release -p hf-io --example load_split -- <split>/train` | 2.70 s; peak RSS 841 MB | `read_real_split_v5_lazy` 17.2 s; the runner's `load_split_from_disk` with screens 61 s of an 80 s startup |
| 2026-09-15 | embedding cache `wikidata5m/nomic-embed-text` (313,310 × 768) | `cargo run --release -p hf-embed --example load_cache -- <cache>` | first load (parse + sidecar) 9.00 s; mapped sidecar 1.84 s; peak RSS 1,882 MB | `load_embedding_matrix` 21.1 s |
| 2026-09-15 | sampler identity and throughput on the rung-3 pool `wikidata5m-n40-d3-k2-r3-g0.25/train` (hub cap 19, region 0.5, greedy-path, share 0.25, with the real embedding cache) | `cargo run --release -p hf-episodes --example pool_check -- <graph>/edges.tsv <cache> <split>/train 2000` | **2,000 / 2,000 episodes identical** to the Python-written pool (ids, visible, hidden); 2,000 kept of 2,827 attempts in 0.1 s (36,148 attempts/s, 32 cores); drops subgraph_too_small 717, no_target 110; peak RSS 3,814 MB incl. the graph | the Python writer: 56,687 attempts for 40,000 kept in hours, single-threaded |
