//! The runner's synthetic fixture world (`real_walk_stage0.fixture_world`),
//! reproduced with the CPython-compatible twister so fixture splits sampled
//! here equal the Python-written ones. Never evidence.

use std::collections::{HashMap, HashSet};

use hf_core::PyRandom;
use hf_graph::RealGraph;

/// `fixture_world(seed, nodes=400, edges=1600, dim=8)`: a random digraph with
/// `node i` texts and uniform(-1, 1) embeddings.
pub fn fixture_world(
    seed: u128,
    nodes: u32,
    edges: usize,
    dim: usize,
) -> (RealGraph, HashMap<String, Vec<f64>>) {
    let mut rng = PyRandom::from_seed(seed);
    let mut edge_set: HashSet<(u32, u32)> = HashSet::new();
    while edge_set.len() < edges {
        let h = rng.randrange(nodes as u64) as u32;
        let t = rng.randrange(nodes as u64) as u32;
        if h != t {
            edge_set.insert((h, t));
        }
    }
    let names: Vec<(String, String)> = edge_set
        .iter()
        .map(|(h, t)| (format!("n{h}"), format!("n{t}")))
        .collect();
    let mut graph = RealGraph::from_edges(
        "fixture",
        names.iter().map(|(h, t)| (h.as_str(), None, t.as_str())),
    );
    let texts: Vec<(String, String)> = (0..nodes)
        .map(|i| (format!("n{i}"), format!("node {i}")))
        .collect();
    graph.set_texts(texts.iter().map(|(n, t)| (n.as_str(), t.as_str())));
    let embeddings = (0..nodes)
        .map(|i| {
            (
                format!("n{i}"),
                (0..dim).map(|_| rng.uniform(-1.0, 1.0)).collect(),
            )
        })
        .collect();
    (graph, embeddings)
}
