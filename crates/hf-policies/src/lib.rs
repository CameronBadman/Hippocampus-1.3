//! The baseline walks of `hippocampus_foundation.read_run.policies_v5`, with
//! the tie-breaks that every paired reading rests on:
//!
//! - the adjacency sorts **target ids as strings** (unlike the learned walk's
//!   `edge_id` order), one entry per visible edge, so a node with two relations
//!   to one tail is pushed twice in Python — and here;
//! - `_forward_walk` re-sorts the whole frontier by its priority key with a
//!   stable sort each step and pops the head; a skipped (non-expandable) node
//!   costs nothing;
//! - registration happens at examination: `registered_at` is the number of
//!   expansions completed when the target is first seen;
//! - similarity-greedy's key is `(-cos, counter)` with a missing vector at
//!   `+2.0`, the counter incrementing at every priority call (the start's
//!   included), the query being the target's own vector at stage 0.

use std::collections::{HashMap, HashSet, VecDeque};

pub const STOP_REGISTERED: &str = "target_registered";
pub const STOP_EXHAUSTED: &str = "exhausted";

/// A vector lookup by node id in float64 — the fixture path's `dict` of
/// Python floats is exact here; a float32 cache is widened (Python computes
/// its float32 cosines through numpy, which this matches to rounding).
pub trait Embeddings {
    fn vector(&self, node: &str) -> Option<Vec<f64>>;
}

impl Embeddings for hf_embed::EmbeddingMatrix {
    fn vector(&self, node: &str) -> Option<Vec<f64>> {
        self.get(node)
            .map(|v| v.iter().map(|x| *x as f64).collect())
    }
}

impl Embeddings for HashMap<String, Vec<f64>> {
    fn vector(&self, node: &str) -> Option<Vec<f64>> {
        self.get(node).cloned()
    }
}

/// What the baselines read of an episode: the visible adjacency (targets
/// sorted as strings) and the hidden truth they are allowed to use.
#[derive(Clone, Debug)]
pub struct EpisodeGraph {
    pub start: String,
    pub target: String,
    pub out: HashMap<String, Vec<String>>,
    /// Sources in first-appearance order — Python's dict insertion order, which
    /// fixes the backward side's queue order in the bidirectional walk.
    pub head_order: Vec<String>,
    pub surviving_paths: Vec<Vec<String>>,
    pub distance_to_target: HashMap<String, u32>,
    pub target_distance: u32,
    pub removal_level: u32,
    pub removed_count: u32,
}

impl EpisodeGraph {
    /// `_adjacency` over `(source, target)` pairs, in the order given; each list sorted.
    pub fn new(
        start: &str,
        target: &str,
        edges: impl IntoIterator<Item = (String, String)>,
        surviving_paths: Vec<Vec<String>>,
        distance_to_target: HashMap<String, u32>,
    ) -> Self {
        let mut out: HashMap<String, Vec<String>> = HashMap::new();
        let mut head_order = Vec::new();
        for (source, tail) in edges {
            if !out.contains_key(&source) {
                head_order.push(source.clone());
            }
            out.entry(source).or_default().push(tail);
        }
        for tails in out.values_mut() {
            tails.sort_unstable();
        }
        Self {
            start: start.to_string(),
            target: target.to_string(),
            out,
            head_order,
            surviving_paths,
            distance_to_target,
            target_distance: 0,
            removal_level: 0,
            removed_count: 0,
        }
    }

    pub fn from_episode(episode: &hf_io::RealEpisode) -> Self {
        let mut g = Self::new(
            &episode.visible.start_node,
            &episode.hidden.target_set[0],
            episode
                .visible
                .edges
                .iter()
                .map(|e| (e.source.clone(), e.target.clone())),
            episode.hidden.surviving_paths.clone(),
            episode
                .hidden
                .distance_to_target
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
        );
        g.target_distance = episode.hidden.target_distance;
        g.removal_level = episode.visible.removal_level;
        g.removed_count = episode.hidden.removed_count;
        g
    }

    fn tails(&self, node: &str) -> &[String] {
        self.out.get(node).map(Vec::as_slice).unwrap_or(&[])
    }
}

/// `WalkTraceV5`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WalkTrace {
    pub examined: Vec<String>,
    pub expansions: u32,
    pub registered_at: Option<u32>,
    pub stop_reason: String,
    pub parents: HashMap<String, String>,
}

impl WalkTrace {
    fn new() -> Self {
        Self {
            stop_reason: STOP_EXHAUSTED.into(),
            ..Default::default()
        }
    }

    fn register(&mut self) {
        self.registered_at = Some(self.expansions);
        self.stop_reason = STOP_REGISTERED.into();
    }

    pub fn registered(&self) -> bool {
        self.registered_at.is_some()
    }

    /// The route: the parent chain of the node whose expansion registered the target.
    pub fn route(&self, start: &str, target: &str) -> Vec<String> {
        if self.registered_at.is_none() || self.examined.is_empty() {
            return Vec::new();
        }
        let mut node = self.examined.last().unwrap().clone();
        let mut chain = vec![target.to_string(), node.clone()];
        while node != start {
            node = self.parents[&node].clone();
            chain.push(node.clone());
        }
        chain.reverse();
        chain
    }
}

/// A priority key: `(float, int)` compared lexicographically, as Python sorts
/// `(depth, counter)`, `(-cos, counter)` and a bare distance (`(d, 0)`).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Key(pub f64, pub u64);

fn forward_walk(
    g: &EpisodeGraph,
    mut priority: impl FnMut(&str, u32) -> Key,
    expandable: impl Fn(&str) -> bool,
) -> WalkTrace {
    let mut trace = WalkTrace::new();
    let mut frontier: Vec<(Key, String, u32)> = vec![(priority(&g.start, 0), g.start.clone(), 0)];
    let mut seen: HashSet<String> = HashSet::from([g.start.clone()]);
    if g.start == g.target {
        trace.registered_at = Some(0);
        trace.stop_reason = STOP_REGISTERED.into();
        return trace;
    }
    while !frontier.is_empty() {
        frontier.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite keys"));
        let (_, node, depth) = frontier.remove(0);
        if !expandable(&node) {
            continue;
        }
        trace.expansions += 1;
        trace.examined.push(node.clone());
        for tail in g.tails(&node) {
            if *tail == g.target {
                trace.register();
                return trace;
            }
            if !seen.contains(tail) {
                seen.insert(tail.clone());
                trace.parents.insert(tail.clone(), node.clone());
                frontier.push((priority(tail, depth + 1), tail.clone(), depth + 1));
            }
        }
    }
    trace
}

/// Breadth-first, depth then discovery order. The ceiling.
pub fn blind_exhaust_trace(g: &EpisodeGraph) -> WalkTrace {
    let mut counter = 0u64;
    forward_walk(
        g,
        |_, depth| {
            counter += 1;
            Key(depth as f64, counter)
        },
        |_| true,
    )
}

/// Expand only nodes on a surviving shortest path, nearest first. The floor.
pub fn oracle_trace(g: &EpisodeGraph) -> WalkTrace {
    let shortest = g.surviving_paths.iter().map(Vec::len).min().unwrap_or(0);
    let allowed: HashSet<&String> = g
        .surviving_paths
        .iter()
        .filter(|p| p.len() == shortest)
        .flat_map(|p| p.iter())
        .collect();
    forward_walk(
        g,
        |node, _| {
            Key(
                g.distance_to_target
                    .get(node)
                    .copied()
                    .map(f64::from)
                    .unwrap_or(1e9),
                0,
            )
        },
        |node| allowed.contains(&node.to_string()),
    )
}

/// The exact algorithm for a known target; both sides pay expansions.
pub fn bidirectional_bfs_trace(g: &EpisodeGraph) -> WalkTrace {
    let mut inc: HashMap<&str, Vec<&str>> = HashMap::new();
    for head in &g.head_order {
        for tail in &g.out[head] {
            inc.entry(tail.as_str()).or_default().push(head.as_str());
        }
    }
    let mut trace = WalkTrace::new();
    if g.start == g.target {
        trace.registered_at = Some(0);
        trace.stop_reason = STOP_REGISTERED.into();
        return trace;
    }
    let mut forward: VecDeque<&str> = VecDeque::from([g.start.as_str()]);
    let mut backward: VecDeque<&str> = VecDeque::from([g.target.as_str()]);
    let mut seen_f: HashSet<&str> = HashSet::from([g.start.as_str()]);
    let mut seen_b: HashSet<&str> = HashSet::from([g.target.as_str()]);
    while !forward.is_empty() || !backward.is_empty() {
        let side_forward =
            (forward.len() <= backward.len() && !forward.is_empty()) || backward.is_empty();
        if side_forward {
            for _ in 0..forward.len() {
                let node = forward.pop_front().unwrap();
                trace.expansions += 1;
                trace.examined.push(node.to_string());
                for tail in g.tails(node) {
                    if *tail == g.target || seen_b.contains(tail.as_str()) {
                        trace.register();
                        return trace;
                    }
                    if !seen_f.contains(tail.as_str()) {
                        seen_f.insert(tail.as_str());
                        forward.push_back(tail.as_str());
                    }
                }
            }
        } else {
            for _ in 0..backward.len() {
                let node = backward.pop_front().unwrap();
                trace.expansions += 1;
                trace.examined.push(node.to_string());
                for head in inc.get(node).map(Vec::as_slice).unwrap_or(&[]) {
                    if seen_f.contains(head) {
                        trace.register();
                        return trace;
                    }
                    if !seen_b.contains(head) {
                        seen_b.insert(head);
                        backward.push_back(head);
                    }
                }
            }
        }
    }
    trace
}

/// Cosine in float64, 0 when either norm is 0 (`policies_v5.cosine`).
pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Best-first by cosine to the query (the target's vector at stage 0); a node
/// with no vector sorts last; falls back to blind when the query has none.
pub fn similarity_greedy_trace(
    g: &EpisodeGraph,
    embeddings: &dyn Embeddings,
    query: Option<&[f64]>,
) -> WalkTrace {
    let q: Vec<f64> = match query
        .map(<[f64]>::to_vec)
        .or_else(|| embeddings.vector(&g.target))
    {
        Some(q) => q,
        None => return blind_exhaust_trace(g),
    };
    let mut counter = 0u64;
    forward_walk(
        g,
        |node, _| {
            counter += 1;
            let key = match embeddings.vector(node) {
                Some(v) => -cosine(&v, &q),
                None => 2.0,
            };
            Key(key, counter)
        },
        |_| true,
    )
}

/// `POLICIES_V5`, in report order.
pub const POLICY_NAMES: [&str; 4] = [
    "blind_exhaust",
    "bidirectional_bfs",
    "similarity_greedy",
    "oracle",
];

/// Every baseline on one episode, in `POLICY_NAMES` order.
pub fn all_traces(g: &EpisodeGraph, embeddings: &dyn Embeddings) -> Vec<(&'static str, WalkTrace)> {
    vec![
        ("blind_exhaust", blind_exhaust_trace(g)),
        ("bidirectional_bfs", bidirectional_bfs_trace(g)),
        (
            "similarity_greedy",
            similarity_greedy_trace(g, embeddings, None),
        ),
        ("oracle", oracle_trace(g)),
    ]
}

/// `policy_row_v5`.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct PolicyRow {
    pub registered: bool,
    pub expansions: u32,
    pub expansions_at_registration: Option<u32>,
    pub examined: usize,
    pub stop_reason: String,
    pub target_distance: u32,
    pub removal_level: u32,
    pub removed_count: u32,
}

pub fn policy_row(g: &EpisodeGraph, trace: &WalkTrace) -> PolicyRow {
    PolicyRow {
        registered: trace.registered_at.is_some(),
        expansions: trace.expansions,
        expansions_at_registration: trace.registered_at,
        examined: trace.examined.len(),
        stop_reason: trace.stop_reason.clone(),
        target_distance: g.target_distance,
        removal_level: g.removal_level,
        removed_count: g.removed_count,
    }
}
