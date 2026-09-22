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

pub mod ktargets;

use std::collections::{HashMap, HashSet, VecDeque};

pub use ktargets::{
    all_k_traces, bidirectional_sequential_trace, k_blind_exhaust_trace, k_greedy_frozen_trace,
    k_greedy_trace, k_oracle, k_oracle_trace, KOracle, K_POLICY_NAMES,
};

pub const STOP_REGISTERED: &str = "target_registered";
pub const STOP_EXHAUSTED: &str = "exhausted";
/// The k-oracle's search budget (`K_TARGETS_DESIGN.md` §3 item 3).
pub const ORACLE_STATE_BUDGET: u64 = 1_000_000;
pub const ORACLE_TIME_LIMIT: std::time::Duration = std::time::Duration::from_millis(200);

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
    /// The episode's targets, in the record's order (node-name order at k >= 2);
    /// one entry at k = 1, where `target()` is v1's `target`.
    pub targets: Vec<String>,
    pub out: HashMap<String, Vec<String>>,
    /// Sources in first-appearance order — Python's dict insertion order, which
    /// fixes the backward side's queue order in the bidirectional walk.
    pub head_order: Vec<String>,
    pub surviving_paths: Vec<Vec<String>>,
    pub distance_to_target: HashMap<String, u32>,
    pub target_distance: u32,
    pub removal_level: u32,
    pub removed_count: u32,
    /// The RUNG's ball size `n`, from which the default fixed budget
    /// `B_fix = n / 2` of `K_TARGETS_DESIGN.md` §4 is taken — the sampler
    /// block's `subgraph_size` on a committed episode (40 at rung 3, 80 at
    /// rung 4), NOT the realised ball, which is smaller wherever the region
    /// filter or the hub cap left fewer nodes. §4 fixes `B_fix` at 20 and 40,
    /// one number per rung, so the strata are not a function of ball size.
    pub subgraph_size: u32,
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
        Self::assemble(
            start,
            vec![target.to_string()],
            out,
            head_order,
            surviving_paths,
            distance_to_target,
        )
    }

    /// The k-target view: the same adjacency, every target in the record's
    /// order, and the union of the surviving paths.
    pub fn new_k(
        start: &str,
        targets: Vec<String>,
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
        Self::assemble(
            start,
            targets,
            out,
            head_order,
            surviving_paths,
            distance_to_target,
        )
    }

    fn assemble(
        start: &str,
        targets: Vec<String>,
        out: HashMap<String, Vec<String>>,
        head_order: Vec<String>,
        surviving_paths: Vec<Vec<String>>,
        distance_to_target: HashMap<String, u32>,
    ) -> Self {
        // the ball as the edge list shows it, the fallback when no record says
        let mut nodes: HashSet<&str> = HashSet::from([start]);
        for (head, tails) in &out {
            nodes.insert(head.as_str());
            nodes.extend(tails.iter().map(String::as_str));
        }
        nodes.extend(targets.iter().map(String::as_str));
        let subgraph_size = nodes.len() as u32;
        Self {
            start: start.to_string(),
            targets,
            out,
            head_order,
            surviving_paths,
            distance_to_target,
            target_distance: 0,
            removal_level: 0,
            removed_count: 0,
            subgraph_size,
        }
    }

    /// The first target — v1's `target` field, and the whole of a k = 1 episode.
    pub fn target(&self) -> &str {
        &self.targets[0]
    }

    /// How many targets the episode carries.
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    /// `K_TARGETS_DESIGN.md` §4's fixed budget `B_fix = n / 2`.
    pub fn b_fix(&self) -> u32 {
        self.subgraph_size / 2
    }

    /// The single-target view of a committed episode. **k = 1 only**: it takes
    /// `target_set[0]` and the whole of `surviving_paths`, which on a k >= 2
    /// record is the union over targets — `oracle_trace` on that union would
    /// expand whichever target's surviving paths are the shorter and call the
    /// result an oracle. The k-target baselines are `from_episode_k`'s
    /// (`k_greedy_trace`, `k_oracle_trace`, `bidirectional_sequential_trace`);
    /// this one is v1's reader and stays v1's.
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
        g.fill_from(episode);
        g
    }

    /// The k-target view of a committed episode: every target of `target_set`
    /// in the record's order, and the UNION of the surviving paths, which is
    /// what `K_TARGETS_DESIGN.md` §3 item 3 prunes the k-oracle's ball to.
    /// `distance_to_target` is the record's per-node minimum over targets; the
    /// k baselines take their own per-target distances from the graph, so no
    /// reader here mistakes a union for one target's.
    pub fn from_episode_k(episode: &hf_io::RealEpisode) -> Self {
        let mut g = Self::new_k(
            &episode.visible.start_node,
            episode.hidden.target_set.clone(),
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
        g.fill_from(episode);
        g
    }

    fn fill_from(&mut self, episode: &hf_io::RealEpisode) {
        self.target_distance = episode.hidden.target_distance;
        self.removal_level = episode.visible.removal_level;
        self.removed_count = episode.hidden.removed_count;
        self.subgraph_size = rung_ball_size(episode);
    }

    pub(crate) fn tails(&self, node: &str) -> &[String] {
        self.out.get(node).map(Vec::as_slice).unwrap_or(&[])
    }
}

/// `WalkTraceV5`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WalkTrace {
    pub examined: Vec<String>,
    pub expansions: u32,
    /// The expansions completed when the LAST target registered — v1's
    /// `registered_at` at k = 1, where the last is the only one.
    pub registered_at: Option<u32>,
    /// The expansions completed when each target registered, in `targets`
    /// order; one entry at k = 1.
    pub registered_at_by_target: Vec<Option<u32>>,
    pub stop_reason: String,
    pub parents: HashMap<String, String>,
}

impl WalkTrace {
    fn new() -> Self {
        Self::with_targets(1)
    }

    pub(crate) fn with_targets(k: usize) -> Self {
        Self {
            stop_reason: STOP_EXHAUSTED.into(),
            registered_at_by_target: vec![None; k],
            ..Default::default()
        }
    }

    fn register(&mut self) {
        self.registered_at = Some(self.expansions);
        self.registered_at_by_target = vec![Some(self.expansions)];
        self.stop_reason = STOP_REGISTERED.into();
    }

    pub fn registered(&self) -> bool {
        self.registered_at.is_some()
    }

    /// How many targets registered.
    pub fn registered_targets(&self) -> usize {
        self.registered_at_by_target
            .iter()
            .filter(|r| r.is_some())
            .count()
    }

    /// `K_TARGETS_DESIGN.md` §4's recall over k at a fixed budget: the share of
    /// targets registered within `budget` expansions.
    pub fn recall_at_budget(&self, budget: u32) -> Option<f64> {
        let k = self.registered_at_by_target.len();
        if k == 0 {
            return None;
        }
        let hit = self
            .registered_at_by_target
            .iter()
            .filter(|r| r.is_some_and(|at| at <= budget))
            .count();
        Some(hit as f64 / k as f64)
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
    if g.start == *g.target() {
        trace.registered_at = Some(0);
        trace.registered_at_by_target = vec![Some(0)];
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
            if *tail == *g.target() {
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
    if g.start == *g.target() {
        trace.registered_at = Some(0);
        trace.registered_at_by_target = vec![Some(0)];
        trace.stop_reason = STOP_REGISTERED.into();
        return trace;
    }
    let mut forward: VecDeque<&str> = VecDeque::from([g.start.as_str()]);
    let mut backward: VecDeque<&str> = VecDeque::from([g.target()]);
    let mut seen_f: HashSet<&str> = HashSet::from([g.start.as_str()]);
    let mut seen_b: HashSet<&str> = HashSet::from([g.target()]);
    while !forward.is_empty() || !backward.is_empty() {
        let side_forward =
            (forward.len() <= backward.len() && !forward.is_empty()) || backward.is_empty();
        if side_forward {
            for _ in 0..forward.len() {
                let node = forward.pop_front().unwrap();
                trace.expansions += 1;
                trace.examined.push(node.to_string());
                for tail in g.tails(node) {
                    if *tail == *g.target() || seen_b.contains(tail.as_str()) {
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
        .or_else(|| embeddings.vector(g.target()))
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

/// Every baseline on one episode, in `POLICY_NAMES` order, with the target's
/// own vector as similarity-greedy's query (stage 0).
pub fn all_traces(g: &EpisodeGraph, embeddings: &dyn Embeddings) -> Vec<(&'static str, WalkTrace)> {
    all_traces_with_query(g, embeddings, None)
}

/// The same list with similarity-greedy's query NAMED: `None` is the target's
/// own vector, which is stage 0 and what `all_traces` passes; `Some(q)` is the
/// episode's question vector, the opponent a stage-1 reading is against. Only
/// similarity-greedy reads a query at all — blind, bidirectional and the
/// oracle are unchanged by it.
pub fn all_traces_with_query(
    g: &EpisodeGraph,
    embeddings: &dyn Embeddings,
    query: Option<&[f64]>,
) -> Vec<(&'static str, WalkTrace)> {
    vec![
        ("blind_exhaust", blind_exhaust_trace(g)),
        ("bidirectional_bfs", bidirectional_bfs_trace(g)),
        (
            "similarity_greedy",
            similarity_greedy_trace(g, embeddings, query),
        ),
        ("oracle", oracle_trace(g)),
    ]
}

/// The rung's `n`: the sampler block's `subgraph_size`, falling back to the
/// realised ball of the visible payload when a record carries no sampler block.
pub fn rung_ball_size(episode: &hf_io::RealEpisode) -> u32 {
    episode
        .hidden
        .sampler
        .get("subgraph_size")
        .and_then(|v| v.as_u64())
        .map(|n| n as u32)
        .unwrap_or(episode.visible.subgraph_size)
}

/// `policy_row_v5`, with `K_TARGETS_DESIGN.md` §6 item 3's k fields.
///
/// `registered` is **all k registered**, which at k = 1 is v1's field. The
/// three k fields are written **only on a k >= 2 episode**: a k = 1 row is v5
/// key for key and value for value, which every committed reader and the
/// policy goldens depend on. §6 item 3 lists them flat; the k = 1 identity
/// makes them conditional, and this is where that is decided.
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
    /// How many targets registered.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub registered_targets: Option<usize>,
    /// The expansions completed when each target registered, in `targets` order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub registered_at: Option<Vec<Option<u32>>>,
    /// §4's recall over k at the fixed budget: targets registered / k.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recall_at_budget: Option<f64>,
}

/// The row at the default fixed budget `B_fix = n / 2` (§4).
pub fn policy_row(g: &EpisodeGraph, trace: &WalkTrace) -> PolicyRow {
    policy_row_at(g, trace, g.b_fix())
}

/// The row with the budget named: `B_fix` is a **coverage choice**, re-readable
/// at another value without re-drawing a pool.
pub fn policy_row_at(g: &EpisodeGraph, trace: &WalkTrace, budget: u32) -> PolicyRow {
    let k = g.target_count() > 1;
    PolicyRow {
        registered: trace.registered_at.is_some(),
        expansions: trace.expansions,
        expansions_at_registration: trace.registered_at,
        examined: trace.examined.len(),
        stop_reason: trace.stop_reason.clone(),
        target_distance: g.target_distance,
        removal_level: g.removal_level,
        removed_count: g.removed_count,
        registered_targets: k.then(|| trace.registered_targets()),
        registered_at: k.then(|| trace.registered_at_by_target.clone()),
        recall_at_budget: if k {
            trace.recall_at_budget(budget)
        } else {
            None
        },
    }
}
