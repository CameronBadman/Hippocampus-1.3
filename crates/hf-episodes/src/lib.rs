//! The stage-0 episode sampler of `hippocampus_foundation.read_run.episodes_v5`,
//! reproduced draw for draw: an episode is a pure function of
//! `(graph, config, split, index)`, so a Rust-sampled pool under the v1 seed
//! label carries the same episode ids and payloads as the Python-written one,
//! and a larger draw reproduces a smaller one as its prefix.
//!
//! What is reproduced exactly: the draw key (the config's key fields in
//! Python's `dict` repr), `_hash_int` (SHA-256 of the U+001F-joined parts,
//! first eight bytes big-endian), the twister seeded from it and its two
//! `randrange` draws (start, then target), the node partitions (`split_of`,
//! `in_screen_region`), the ball, the target candidates, the bounded path set,
//! cheapest-first and greedy-path removals with their hash tie-breaks, the
//! survivors-by-recomputation check, the node and edge order of the visible
//! payload, every hidden field, the six drop reasons, and the episode id.
//!
//! What is new (additive): sampling runs in parallel over attempt indices
//! with the kept ordinals assigned in index order, and `SamplerConfig.targets`
//! draws k = 2 by `K_TARGETS_DESIGN.md` §1 — a second target at the same
//! distance, rejected when either target lies inside a bounded path to the
//! other, and the joint-cut removal that cuts greedy's route to each target
//! with one protected survivor per target. `targets = 1` is v1 exactly: the
//! field enters neither the draw key, the sampler block, nor the episode id,
//! and every k = 1 payload is the byte the Python sampler wrote.

pub mod fixture;

use std::collections::{BTreeMap, HashSet};

use hf_core::files::python_float_repr;
use hf_core::{HfError, PyRandom};
use hf_graph::{NodeId, RealGraph, Subgraph};
use hf_policies::{oracle_trace, similarity_greedy_trace, Embeddings, EpisodeGraph};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

pub const SUBGRAPH_LADDER: [u32; 11] = [10, 20, 40, 64, 80, 128, 160, 256, 320, 512, 1024];
pub const REMOVAL_LADDER: [u32; 5] = [2, 4, 8, 16, 32];
pub const REMOVAL_RULES: [&str; 2] = ["cheapest-first", "greedy-path"];
pub const STAGE0: &str = "stage0_known_target";

/// `SamplerConfig`; field order is the draw key's order.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SamplerConfig {
    pub family: String,
    pub subgraph_size: u32,
    pub target_distance: u32,
    pub removal_level: u32,
    pub cost_epsilon: f64,
    pub max_paths: u32,
    pub seed_label: String,
    pub hub_degree_cap: Option<u32>,
    pub screen_region: f64,
    /// How many targets one episode draws (`K_TARGETS_DESIGN.md` §1). 1 is v1
    /// and enters neither the draw key nor the episode id, so every pool
    /// written before this field reproduces draw for draw.
    #[serde(default = "one_target")]
    pub targets: u32,
    pub removal_rule: String,
    pub greedy_share: f64,
}

/// The v1 target count, for a `sampler` block written before `targets` existed.
fn one_target() -> u32 {
    1
}

impl SamplerConfig {
    pub fn new(family: &str, subgraph_size: u32, target_distance: u32, removal_level: u32) -> Self {
        Self {
            family: family.to_string(),
            subgraph_size,
            target_distance,
            removal_level,
            cost_epsilon: 0.5,
            max_paths: 512,
            seed_label: "real-walk-v1".into(),
            hub_degree_cap: None,
            screen_region: 0.0,
            targets: 1,
            removal_rule: "cheapest-first".into(),
            greedy_share: 1.0,
        }
    }

    pub fn validate(&self) -> Result<(), HfError> {
        let bad = |m: String| Err(HfError::Invalid(m));
        if !SUBGRAPH_LADDER.contains(&self.subgraph_size) {
            return bad(format!(
                "subgraph size {} is not on the ladder",
                self.subgraph_size
            ));
        }
        if !REMOVAL_LADDER.contains(&self.removal_level) {
            return bad(format!(
                "removal level {} is not on the ladder",
                self.removal_level
            ));
        }
        if self.target_distance < 1 {
            return bad("target distance must be >= 1".into());
        }
        if !(0.0..=1.0).contains(&self.cost_epsilon) {
            return bad("cost epsilon must lie in [0, 1]".into());
        }
        if !(0.0..1.0).contains(&self.screen_region) {
            return bad("screen region must lie in [0, 1)".into());
        }
        if !REMOVAL_RULES.contains(&self.removal_rule.as_str()) {
            return bad(format!("unknown removal rule {:?}", self.removal_rule));
        }
        if matches!(self.hub_degree_cap, Some(0)) {
            return bad("hub degree cap must be >= 1".into());
        }
        if !(1..=2).contains(&self.targets) {
            return bad(format!(
                "targets {} is not 1 or 2; K_TARGETS_DESIGN.md draws k = 2 and leaves k > 2 to its own re-run",
                self.targets
            ));
        }
        if !(self.greedy_share > 0.0 && self.greedy_share <= 1.0) {
            return bad("greedy share must lie in (0, 1]".into());
        }
        Ok(())
    }

    /// `int(floor((1 + epsilon) * d))`.
    pub fn cost_bound(&self) -> u32 {
        ((1.0 + self.cost_epsilon) * self.target_distance as f64).floor() as u32
    }

    /// `key_fields()` rendered exactly as Python interpolates its `dict` repr.
    pub fn key_fields_repr(&self) -> String {
        let mut parts = vec![
            format!("'family': {}", python_str_repr(&self.family)),
            format!("'subgraph_size': {}", self.subgraph_size),
            format!("'target_distance': {}", self.target_distance),
            format!("'removal_level': {}", self.removal_level),
            format!("'cost_epsilon': {}", python_float_repr(self.cost_epsilon)),
            format!("'max_paths': {}", self.max_paths),
            format!("'seed_label': {}", python_str_repr(&self.seed_label)),
        ];
        if let Some(cap) = self.hub_degree_cap {
            parts.push(format!("'hub_degree_cap': {cap}"));
        }
        if self.screen_region != 0.0 {
            parts.push(format!(
                "'screen_region': {}",
                python_float_repr(self.screen_region)
            ));
        }
        if self.targets != 1 {
            parts.push(format!("'targets': {}", self.targets));
        }
        format!("{{{}}}", parts.join(", "))
    }

    /// The draw key of one attempt.
    pub fn draw_key(&self, split: &str, index: u64) -> String {
        format!("{}|{split}|{index}|{}", self.family, self.key_fields_repr())
    }

    /// `asdict(config)` as the hidden `sampler` block.
    pub fn as_value(&self) -> Value {
        let mut m = Map::new();
        m.insert("family".into(), self.family.clone().into());
        m.insert("subgraph_size".into(), self.subgraph_size.into());
        m.insert("target_distance".into(), self.target_distance.into());
        m.insert("removal_level".into(), self.removal_level.into());
        m.insert("cost_epsilon".into(), self.cost_epsilon.into());
        m.insert("max_paths".into(), self.max_paths.into());
        m.insert("seed_label".into(), self.seed_label.clone().into());
        m.insert(
            "hub_degree_cap".into(),
            self.hub_degree_cap.map(Value::from).unwrap_or(Value::Null),
        );
        m.insert("screen_region".into(), self.screen_region.into());
        if self.targets != 1 {
            m.insert("targets".into(), self.targets.into());
        }
        m.insert("removal_rule".into(), self.removal_rule.clone().into());
        m.insert("greedy_share".into(), self.greedy_share.into());
        Value::Object(m)
    }
}

/// Python's `repr(str)` for the identifiers that reach the draw key.
pub fn python_str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::new();
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32))
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python's `format(x, "g")` for the share suffix: up to six significant digits, no trailing zeros.
pub fn python_g(x: f64) -> String {
    if x == 0.0 {
        return "0".into();
    }
    let exp = x.abs().log10().floor() as i32;
    if !(-4..6).contains(&exp) {
        let s = format!("{:.5e}", x);
        let (m, e) = s.split_once('e').unwrap();
        let m = m.trim_end_matches('0').trim_end_matches('.');
        let e: i32 = e.parse().unwrap();
        return format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs());
    }
    let decimals = (5 - exp).max(0) as usize;
    let s = format!("{:.*}", decimals, x);
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

/// `_hash_int`: SHA-256 over the U+001F-joined parts, first eight bytes big-endian.
pub fn hash_int(parts: &[&str]) -> u64 {
    let mut h = Sha256::new();
    for (i, p) in parts.iter().enumerate() {
        if i > 0 {
            h.update(b"\x1f");
        }
        h.update(p.as_bytes());
    }
    let digest = h.finalize();
    u64::from_be_bytes(digest[..8].try_into().unwrap())
}

const TWO_POW_64: f64 = 18_446_744_073_709_551_616.0;

/// `split_of`: `"screen"` when the node's bucket is under the screen fraction (0.2).
pub fn split_of(node: &str, seed_label: &str, screen_fraction: f64) -> &'static str {
    let bucket = hash_int(&[seed_label, "split", node]) as f64 / TWO_POW_64;
    if bucket < screen_fraction {
        "screen"
    } else {
        "train"
    }
}

/// `in_screen_region`: the node-disjoint region, an independent hash domain.
pub fn in_screen_region(node: &str, seed_label: &str, fraction: f64) -> bool {
    if fraction <= 0.0 {
        return false;
    }
    hash_int(&[seed_label, "region", node]) as f64 / TWO_POW_64 < fraction
}

/// Why an attempt produced no episode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dropped {
    SubgraphTooSmall,
    NoTargetAtDistanceInSplit,
    NoPathWithinBound,
    PathSetOverCap,
    GreedyRouteMissing,
    SurvivorsMismatch,
    NoSecondTargetAtDistance,
    TargetsInterdependent,
    RemovalLeftNoPath,
    SurvivorNotRecovered,
}

impl Dropped {
    pub fn as_str(self) -> &'static str {
        match self {
            Dropped::SubgraphTooSmall => "subgraph_too_small",
            Dropped::NoTargetAtDistanceInSplit => "no_target_at_distance_in_split",
            Dropped::NoPathWithinBound => "no_path_within_bound",
            Dropped::PathSetOverCap => "path_set_over_cap",
            Dropped::GreedyRouteMissing => "greedy_route_missing",
            Dropped::SurvivorsMismatch => "survivors_mismatch",
            Dropped::NoSecondTargetAtDistance => "no_second_target_at_distance",
            Dropped::TargetsInterdependent => "targets_interdependent",
            Dropped::RemovalLeftNoPath => "removal_left_no_path",
            Dropped::SurvivorNotRecovered => "survivor_not_recovered",
        }
    }
}

/// A sampled episode: the id and both payloads as the split writer takes them.
#[derive(Clone, Debug)]
pub struct Sampled {
    pub episode_id: String,
    pub visible: Value,
    pub hidden: Value,
    /// The node names of the ball, for the sidecars.
    pub nodes: Vec<String>,
}

type Edge = (String, String);
/// `(removed edges, designated survivors, unremovable count)`.
pub type Removal = (HashSet<Edge>, Vec<Vec<String>>, u32);

fn edges_of(p: &[String]) -> HashSet<Edge> {
    p.windows(2).map(|w| (w[0].clone(), w[1].clone())).collect()
}

/// `choose_removals`: cheapest first, ties by seeded hash, at most `len - 1`
/// removed, each by one edge on no designated survivor.
pub fn choose_removals(
    paths: &[Vec<String>],
    level: u32,
    seed_label: &str,
    episode_key: &str,
) -> Result<Removal, HfError> {
    if paths.is_empty() {
        return Err(HfError::Invalid("no path to remove from".into()));
    }
    let mut ordered: Vec<(usize, u64, &Vec<String>)> = paths
        .iter()
        .map(|p| {
            (
                p.len(),
                hash_int(&[seed_label, episode_key, &p.join("|")]),
                p,
            )
        })
        .collect();
    ordered.sort_by_key(|a| (a.0, a.1));
    let n_remove = (level as usize).min(ordered.len() - 1);
    let (to_remove, keep) = ordered.split_at(n_remove);
    let mut survivors: Vec<Vec<String>> = keep.iter().map(|(_, _, p)| (*p).clone()).collect();
    let protected: HashSet<Edge> = survivors.iter().flat_map(|p| edges_of(p)).collect();
    let mut removed: HashSet<Edge> = HashSet::new();
    let mut unremovable = 0u32;
    for (_, _, path) in to_remove {
        let edges: Vec<Edge> = path
            .windows(2)
            .map(|w| (w[0].clone(), w[1].clone()))
            .collect();
        if edges.iter().any(|e| removed.contains(e)) {
            continue;
        }
        let mut candidates: Vec<(u64, Edge)> = edges
            .into_iter()
            .filter(|e| !protected.contains(e))
            .map(|e| (hash_int(&[seed_label, episode_key, &e.0, &e.1]), e))
            .collect();
        if candidates.is_empty() {
            unremovable += 1;
            survivors.push((*path).clone());
            continue;
        }
        candidates.sort_by_key(|a| a.0);
        removed.insert(candidates.swap_remove(0).1);
    }
    survivors.sort_unstable();
    survivors.dedup();
    Ok((removed, survivors, unremovable))
}

/// The designated survivor of amendment 5's recipe: the path sharing the fewest
/// edges with greedy's route, then the shortest, then the lexically first.
fn greedy_survivor<'p>(paths: &'p [Vec<String>], route: &[String]) -> &'p Vec<String> {
    let route_set: HashSet<Edge> = route
        .windows(2)
        .map(|w| (w[0].clone(), w[1].clone()))
        .collect();
    paths
        .iter()
        .min_by(|a, b| {
            let ka = (
                edges_of(a).intersection(&route_set).count(),
                a.len(),
                (*a).clone(),
            );
            let kb = (
                edges_of(b).intersection(&route_set).count(),
                b.len(),
                (*b).clone(),
            );
            ka.cmp(&kb)
        })
        .expect("non-empty")
}

/// `choose_greedy_removals`: cut greedy's route nearest the target first,
/// skipping the designated survivor's edges; unremovable = 1 when nothing was cut.
pub fn choose_greedy_removals(
    paths: &[Vec<String>],
    route: &[String],
    level: u32,
) -> Result<Removal, HfError> {
    if paths.is_empty() || route.len() < 2 {
        return Err(HfError::Invalid(
            "greedy removal needs paths and a registered route".into(),
        ));
    }
    let route_edges: Vec<Edge> = route
        .windows(2)
        .map(|w| (w[0].clone(), w[1].clone()))
        .collect();
    let survivor = greedy_survivor(paths, route);
    let protected = edges_of(survivor);
    let mut removed: HashSet<Edge> = HashSet::new();
    for edge in route_edges.iter().rev() {
        if removed.len() as u32 >= level {
            break;
        }
        if protected.contains(edge) {
            continue;
        }
        removed.insert(edge.clone());
    }
    let mut survivors: Vec<Vec<String>> = paths
        .iter()
        .filter(|p| edges_of(p).is_disjoint(&removed))
        .cloned()
        .collect();
    survivors.sort_unstable();
    if survivors.is_empty() {
        return Err(HfError::Invalid("greedy removal left no path".into()));
    }
    let unremovable = if removed.is_empty() { 1 } else { 0 };
    Ok((removed, survivors, unremovable))
}

/// `(removed edges, the designated survivors of each target, unremovable count)`.
/// The survivor lists are per target and in `target_set` order; a k = 1 episode
/// carries exactly one, which is v1's survivor list.
pub type JointRemoval = (HashSet<Edge>, Vec<Vec<Vec<String>>>, u32);

/// The joint cut of `K_TARGETS_DESIGN.md` §1, steps 1–3: designate one
/// protected survivor per target **before any edge is cut**, then cut each
/// target's single-target greedy route nearest that target first, up to `level`
/// **new** edges each, skipping every designated survivor's edges and every
/// edge already removed. Step 4 (one recomputation on the fully pruned graph)
/// and step 5 (containment) are `episode_from`'s.
///
/// `unremovable` counts the targets whose cut removed nothing, as the
/// single-target recipe counts its one.
pub fn choose_joint_greedy_removals(
    paths: &[Vec<Vec<String>>],
    routes: &[Vec<String>],
    level: u32,
) -> Result<JointRemoval, HfError> {
    if paths.is_empty() || paths.len() != routes.len() {
        return Err(HfError::Invalid(
            "the joint cut needs one path set and one route per target".into(),
        ));
    }
    let mut designated: Vec<Vec<Vec<String>>> = Vec::with_capacity(paths.len());
    for (ps, route) in paths.iter().zip(routes) {
        if ps.is_empty() || route.len() < 2 {
            return Err(HfError::Invalid(
                "greedy removal needs paths and a registered route".into(),
            ));
        }
        designated.push(vec![greedy_survivor(ps, route).clone()]);
    }
    let protected: HashSet<Edge> = designated
        .iter()
        .flatten()
        .flat_map(|p| edges_of(p))
        .collect();
    let mut removed: HashSet<Edge> = HashSet::new();
    let mut unremovable = 0u32;
    for route in routes {
        let route_edges: Vec<Edge> = route
            .windows(2)
            .map(|w| (w[0].clone(), w[1].clone()))
            .collect();
        let mut cut = 0u32;
        for edge in route_edges.iter().rev() {
            if cut >= level {
                break;
            }
            if protected.contains(edge) || removed.contains(edge) {
                continue;
            }
            removed.insert(edge.clone());
            cut += 1;
        }
        if cut == 0 {
            unremovable += 1;
        }
    }
    Ok((removed, designated, unremovable))
}

/// Amendment 1's cheapest-first removal over k targets — the rule for an
/// episode outside the greedy share. Which paths a cut aims at is fixed by the
/// `(length, seeded hash)` order alone, before anything is protected, so the
/// designation is not circular; every kept path of **every** target is then
/// protected from **every** cut, which is the joint cut's cross-protection
/// applied to the collateral the design note names (cutting one target's paths
/// can otherwise remove another target's). At k = 1 it is v1's rule.
pub fn choose_joint_removals(
    paths: &[Vec<Vec<String>>],
    level: u32,
    seed_label: &str,
    episode_key: &str,
) -> Result<JointRemoval, HfError> {
    if paths.is_empty() || paths.iter().any(Vec::is_empty) {
        return Err(HfError::Invalid("no path to remove from".into()));
    }
    let mut aimed: Vec<Vec<&Vec<String>>> = Vec::with_capacity(paths.len());
    let mut designated: Vec<Vec<Vec<String>>> = Vec::with_capacity(paths.len());
    for ps in paths {
        let mut ordered: Vec<(usize, u64, &Vec<String>)> = ps
            .iter()
            .map(|p| {
                (
                    p.len(),
                    hash_int(&[seed_label, episode_key, &p.join("|")]),
                    p,
                )
            })
            .collect();
        ordered.sort_by_key(|a| (a.0, a.1));
        let n_remove = (level as usize).min(ordered.len() - 1);
        let (to_remove, keep) = ordered.split_at(n_remove);
        aimed.push(to_remove.iter().map(|(_, _, p)| *p).collect());
        designated.push(keep.iter().map(|(_, _, p)| (*p).clone()).collect());
    }
    let protected: HashSet<Edge> = designated
        .iter()
        .flatten()
        .flat_map(|p| edges_of(p))
        .collect();
    let mut removed: HashSet<Edge> = HashSet::new();
    let mut unremovable = 0u32;
    for (i, to_remove) in aimed.iter().enumerate() {
        for path in to_remove {
            let edges: Vec<Edge> = path
                .windows(2)
                .map(|w| (w[0].clone(), w[1].clone()))
                .collect();
            if edges.iter().any(|e| removed.contains(e)) {
                continue;
            }
            let mut candidates: Vec<(u64, Edge)> = edges
                .into_iter()
                .filter(|e| !protected.contains(e))
                .map(|e| (hash_int(&[seed_label, episode_key, &e.0, &e.1]), e))
                .collect();
            if candidates.is_empty() {
                // every edge is protected, so the path outlives both cuts
                unremovable += 1;
                designated[i].push((*path).clone());
                continue;
            }
            candidates.sort_by_key(|a| a.0);
            removed.insert(candidates.swap_remove(0).1);
        }
        designated[i].sort_unstable();
        designated[i].dedup();
    }
    Ok((removed, designated, unremovable))
}

/// Every node `c` for which `target` is an interior node of some simple
/// `start → c` path of at most `bound` edges — §1's second rejection, the
/// bound-aware one: the walk must expand `target` to continue to `c`, so it
/// registered `target` on the way and k = 2 would collapse to k = 1 with a
/// bonus. Exact, not the `d(s,x) + d(x,y) <= bound` relaxation: the suffix is
/// enumerated only over nodes the prefix does not already use.
fn beyond_target(
    sub: &Subgraph,
    target: NodeId,
    paths_to_target: &[Vec<NodeId>],
    bound: u32,
) -> HashSet<NodeId> {
    fn extend(
        sub: &Subgraph,
        node: NodeId,
        left: u32,
        on_path: &mut HashSet<NodeId>,
        out: &mut HashSet<NodeId>,
    ) {
        if left == 0 {
            return;
        }
        for tail in sub.out_neighbours(node) {
            if on_path.contains(&tail) {
                continue;
            }
            out.insert(tail);
            on_path.insert(tail);
            extend(sub, tail, left - 1, on_path, out);
            on_path.remove(&tail);
        }
    }
    let mut out = HashSet::new();
    for p in paths_to_target {
        let used = (p.len() - 1) as u32;
        if used >= bound {
            continue;
        }
        let mut on_path: HashSet<NodeId> = p.iter().copied().collect();
        extend(sub, target, bound - used, &mut on_path, &mut out);
    }
    out
}

/// The start lists of a split, computed once per graph (`_split_starts` / `_member_starts`).
pub struct Sampler<'g> {
    pub graph: &'g RealGraph,
    pub config: SamplerConfig,
    starts: BTreeMap<String, Vec<NodeId>>,
}

impl<'g> Sampler<'g> {
    pub fn new(graph: &'g RealGraph, config: SamplerConfig) -> Result<Self, HfError> {
        config.validate()?;
        Ok(Self {
            graph,
            config,
            starts: BTreeMap::new(),
        })
    }

    fn member(&self, split: &str, node: NodeId) -> bool {
        let name = self.graph.name(node);
        if self.config.screen_region > 0.0 {
            let inside = in_screen_region(name, &self.config.seed_label, self.config.screen_region);
            if split == "screen" {
                inside
            } else {
                !inside
            }
        } else {
            split_of(name, &self.config.seed_label, 0.2) == split
        }
    }

    /// Compute (and cache) a split's start list; call before sampling in parallel.
    pub fn prepare(&mut self, split: &str) -> Result<&[NodeId], HfError> {
        if !self.starts.contains_key(split) {
            let starts: Vec<NodeId> = self
                .graph
                .nodes()
                .filter(|n| self.graph.has_out(*n) && self.member(split, *n))
                .collect();
            if starts.is_empty() {
                return Err(HfError::Invalid(format!(
                    "no start nodes fall in split {split}"
                )));
            }
            self.starts.insert(split.to_string(), starts);
        }
        Ok(&self.starts[split])
    }

    /// `sample_episode` for one attempt.
    pub fn sample(
        &self,
        split: &str,
        index: u64,
        embeddings: Option<&dyn Embeddings>,
    ) -> Result<Result<Sampled, Dropped>, HfError> {
        let config = &self.config;
        let greedy_rule = config.removal_rule == "greedy-path";
        if greedy_rule && embeddings.is_none() {
            return Err(HfError::Invalid(
                "the greedy-path removal rule needs embeddings".into(),
            ));
        }
        let starts = self.starts.get(split).ok_or_else(|| {
            HfError::Invalid(format!("prepare({split:?}) must run before sampling"))
        })?;
        let key = config.draw_key(split, index);
        let mut rng = PyRandom::from_seed(hash_int(&[&config.seed_label, &key]) as u128);
        let region = config.screen_region > 0.0;
        let start = starts[rng.randrange(starts.len() as u64) as usize];
        let member = |n: NodeId| self.member(split, n);
        let ball = self.graph.ball(
            start,
            config.subgraph_size as usize,
            config.hub_degree_cap,
            3,
            if region { Some(&member) } else { None },
        )?;
        if ball.len() < config.target_distance as usize + 1 {
            return Ok(Err(Dropped::SubgraphTooSmall));
        }
        let sub = self.graph.induced(&ball);
        let from_start = sub.distances_from(start);
        let mut candidates: Vec<NodeId> = from_start
            .iter()
            .filter(|(n, d)| **d == config.target_distance && self.member(split, **n))
            .map(|(n, _)| *n)
            .collect();
        candidates.sort_unstable();
        if candidates.is_empty() {
            return Ok(Err(Dropped::NoTargetAtDistanceInSplit));
        }
        let target = candidates[rng.randrange(candidates.len() as u64) as usize];
        let bound = config.cost_bound();
        let paths = sub.simple_paths(start, target, bound);
        if paths.is_empty() {
            return Ok(Err(Dropped::NoPathWithinBound));
        }
        if paths.len() > config.max_paths as usize {
            return Ok(Err(Dropped::PathSetOverCap));
        }
        let names = |p: &Vec<NodeId>| -> Vec<String> {
            p.iter().map(|n| self.graph.name(*n).to_string()).collect()
        };
        // the further targets of §1 steps 2-4: the same draw device on the same
        // candidate list, minus the targets already drawn, the interior nodes of
        // their bounded path sets, and everything reachable only through them
        let mut targets: Vec<NodeId> = vec![target];
        let mut raw_paths: Vec<Vec<Vec<NodeId>>> = vec![paths];
        if config.targets > 1 {
            if candidates.len() < config.targets as usize {
                return Ok(Err(Dropped::NoSecondTargetAtDistance));
            }
            let mut blocked: HashSet<NodeId> = HashSet::new();
            for drawn in 1..config.targets as usize {
                let last = targets[drawn - 1];
                let last_paths = &raw_paths[drawn - 1];
                blocked.insert(last);
                blocked.extend(
                    last_paths
                        .iter()
                        .flat_map(|p| p[1..p.len() - 1].iter().copied()),
                );
                blocked.extend(beyond_target(&sub, last, last_paths, bound));
                // a Vec filtered from the sorted candidates: randrange indexes it
                let filtered: Vec<NodeId> = candidates
                    .iter()
                    .copied()
                    .filter(|c| !blocked.contains(c))
                    .collect();
                if filtered.is_empty() {
                    return Ok(Err(Dropped::TargetsInterdependent));
                }
                let next = filtered[rng.randrange(filtered.len() as u64) as usize];
                let next_paths = sub.simple_paths(start, next, bound);
                if next_paths.is_empty() {
                    return Ok(Err(Dropped::NoPathWithinBound));
                }
                if next_paths.len() > config.max_paths as usize {
                    return Ok(Err(Dropped::PathSetOverCap));
                }
                targets.push(next);
                raw_paths.push(next_paths);
            }
            // T in node-name order, and everything downstream with it, so the
            // episode is a function of the set and not of the draw
            let mut order: Vec<usize> = (0..targets.len()).collect();
            order.sort_by(|a, b| {
                self.graph
                    .name(targets[*a])
                    .cmp(self.graph.name(targets[*b]))
            });
            targets = order.iter().map(|i| targets[*i]).collect();
            raw_paths = order.iter().map(|i| raw_paths[*i].clone()).collect();
        }
        let path_sets: Vec<Vec<Vec<String>>> = raw_paths
            .iter()
            .map(|ps| ps.iter().map(&names).collect())
            .collect();
        let greedy_member = greedy_rule
            && (config.greedy_share >= 1.0
                || (hash_int(&[&config.seed_label, &key, "greedy-share"]) as f64 / TWO_POW_64)
                    < config.greedy_share);
        let (removed, survivors, unremovable): JointRemoval = if greedy_member {
            // similarity-greedy's route to each target on the unpruned subgraph
            let mut routes: Vec<Vec<String>> = Vec::with_capacity(targets.len());
            for (i, t) in targets.iter().enumerate() {
                let draft =
                    self.episode_graph(&sub, &ball, start, *t, &HashSet::new(), &path_sets[i]);
                let trace = similarity_greedy_trace(&draft, embeddings.expect("checked"), None);
                let route = trace.route(&draft.start, draft.target());
                if route.is_empty() {
                    return Ok(Err(Dropped::GreedyRouteMissing));
                }
                routes.push(route);
            }
            if config.targets == 1 {
                let (removed, survivors, unremovable) =
                    choose_greedy_removals(&path_sets[0], &routes[0], config.removal_level)?;
                (removed, vec![survivors], unremovable)
            } else {
                choose_joint_greedy_removals(&path_sets, &routes, config.removal_level)?
            }
        } else if config.targets == 1 {
            let (removed, survivors, unremovable) = choose_removals(
                &path_sets[0],
                config.removal_level,
                &config.seed_label,
                &key,
            )?;
            (removed, vec![survivors], unremovable)
        } else {
            choose_joint_removals(&path_sets, config.removal_level, &config.seed_label, &key)?
        };
        let mut sampled = match self.episode_from(
            split,
            index,
            &key,
            start,
            &targets,
            &sub,
            &ball,
            &path_sets,
            bound,
            &removed,
            &survivors,
            unremovable,
        )? {
            Ok(sampled) => sampled,
            Err(reason) => return Ok(Err(reason)),
        };
        if greedy_rule {
            // one single-target overshoot per target, on that target's own
            // surviving paths and distances — never on the union, which would
            // oracle whichever target's paths are shorter
            let target_names: Vec<String> = targets
                .iter()
                .map(|t| self.graph.name(*t).to_string())
                .collect();
            let overshoots: Vec<i64> = target_names
                .iter()
                .map(|t| {
                    let g = self.episode_graph_of(&sampled, t);
                    let greedy =
                        similarity_greedy_trace(&g, embeddings.expect("checked"), None).expansions;
                    let oracle = oracle_trace(&g).expansions;
                    greedy as i64 - oracle as i64
                })
                .collect();
            let hidden = sampled.hidden.as_object_mut().unwrap();
            hidden.insert(
                "greedy_overshoot".into(),
                (*overshoots.iter().max().expect("a target")).into(),
            );
            if config.targets > 1 {
                hidden.insert("greedy_overshoots".into(), overshoots.into());
            }
            hidden.insert(
                "removal_recipe".into(),
                (if greedy_member {
                    "greedy-path"
                } else {
                    "cheapest-first"
                })
                .into(),
            );
        }
        Ok(Ok(sampled))
    }

    /// The pruned subgraph's edges in `node_order × (relation, tail)` order — the
    /// visible edge order — as `(source, relation, target)` names.
    fn pruned_edges(
        &self,
        sub: &Subgraph,
        ball: &[NodeId],
        removed: &HashSet<Edge>,
    ) -> Vec<(String, Option<String>, String)> {
        let mut out = Vec::new();
        for &h in ball {
            let hn = self.graph.name(h);
            for e in sub.out(h) {
                let tn = self.graph.name(e.tail);
                if removed.contains(&(hn.to_string(), tn.to_string())) {
                    continue;
                }
                let relation = if self.graph.typed() {
                    Some(self.graph.relation_name(e.relation).to_string())
                } else {
                    None
                };
                out.push((hn.to_string(), relation, tn.to_string()));
            }
        }
        out
    }

    fn episode_graph(
        &self,
        sub: &Subgraph,
        ball: &[NodeId],
        start: NodeId,
        target: NodeId,
        removed: &HashSet<Edge>,
        survivors: &[Vec<String>],
    ) -> EpisodeGraph {
        let edges = self.pruned_edges(sub, ball, removed);
        EpisodeGraph::new(
            self.graph.name(start),
            self.graph.name(target),
            edges.into_iter().map(|(s, _, t)| (s, t)),
            survivors.to_vec(),
            std::collections::HashMap::new(),
        )
    }

    /// The single-target view of one sampled episode, for one named target: its
    /// own surviving paths (the union split on each path's last node, which a
    /// path to another target can never be) and its own distances. A union is
    /// never passed whole — `oracle_trace` on one would oracle whichever
    /// target's surviving paths are the shorter.
    fn episode_graph_of(&self, sampled: &Sampled, target: &str) -> EpisodeGraph {
        let v = &sampled.visible;
        let h = &sampled.hidden;
        let edges = v["edges"].as_array().unwrap().iter().map(|e| {
            (
                e["source"].as_str().unwrap().to_string(),
                e["target"].as_str().unwrap().to_string(),
            )
        });
        let union: Vec<Vec<String>> = serde_json::from_value(h["surviving_paths"].clone()).unwrap();
        let survivors: Vec<Vec<String>> = union
            .into_iter()
            .filter(|p| p.last().map(String::as_str) == Some(target))
            .collect();
        let distances: std::collections::HashMap<String, u32> = match h.get("distance_to_targets") {
            Some(by_target) => serde_json::from_value(by_target[target].clone()).unwrap(),
            None => serde_json::from_value(h["distance_to_target"].clone()).unwrap(),
        };
        EpisodeGraph::new(
            v["start_node"].as_str().unwrap(),
            target,
            edges,
            survivors,
            distances,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn episode_from(
        &self,
        split: &str,
        index: u64,
        key: &str,
        start: NodeId,
        targets: &[NodeId],
        sub: &Subgraph,
        ball: &[NodeId],
        paths: &[Vec<Vec<String>>],
        bound: u32,
        removed: &HashSet<Edge>,
        designated: &[Vec<Vec<String>>],
        unremovable: u32,
    ) -> Result<Result<Sampled, Dropped>, HfError> {
        let config = &self.config;
        let k = targets.len();
        let target = targets[0];
        let target_names: Vec<String> = targets
            .iter()
            .map(|t| self.graph.name(*t).to_string())
            .collect();
        // prune by node pair and recompute the survivors on the FULLY pruned
        // subgraph, once, as the joint cut's step 4 requires
        let pruned = sub.without_pairs(|h, t| {
            removed.contains(&(
                self.graph.name(h).to_string(),
                self.graph.name(t).to_string(),
            ))
        });
        let recomputed: Vec<Vec<Vec<String>>> = targets
            .iter()
            .map(|t| {
                pruned
                    .simple_paths(start, *t, bound)
                    .iter()
                    .map(|p| p.iter().map(|n| self.graph.name(*n).to_string()).collect())
                    .collect()
            })
            .collect();
        let survivors: Vec<Vec<String>> = if k == 1 {
            // v1's test: the recomputed set must EQUAL the designated survivors
            if recomputed[0].is_empty() || recomputed[0] != designated[0] {
                return Ok(Err(Dropped::SurvivorsMismatch));
            }
            designated[0].clone()
        } else {
            // the k-recipe's step 5: each target keeps a path, and each
            // protected survivor is recovered — containment, not equality,
            // since cutting one target's route may remove another's path
            for (found, want) in recomputed.iter().zip(designated) {
                if found.is_empty() {
                    return Ok(Err(Dropped::RemovalLeftNoPath));
                }
                if want.iter().any(|p| !found.contains(p)) {
                    return Ok(Err(Dropped::SurvivorNotRecovered));
                }
            }
            let mut union: Vec<Vec<String>> = recomputed.iter().flatten().cloned().collect();
            union.sort_unstable();
            union.dedup();
            union
        };
        let edges = self.pruned_edges(sub, ball, removed);
        let edge_records: Vec<Value> = edges
            .iter()
            .enumerate()
            .map(|(i, (s, r, t))| {
                json!({"edge_id": i, "source": s, "target": t, "relation": r.clone().map(Value::from).unwrap_or(Value::Null)})
            })
            .collect();
        let node_names: Vec<String> = ball
            .iter()
            .map(|n| self.graph.name(*n).to_string())
            .collect();
        let nodes: Vec<Value> = ball
            .iter()
            .map(
                |n| json!({"node": self.graph.name(*n), "text": self.graph.text(*n).unwrap_or("")}),
            )
            .collect();
        // k >= 2 shows a second target, which v5's allow-list has no key for
        let schema_version = if k == 1 {
            hf_io::SCHEMA_VERSION_V5
        } else {
            hf_io::SCHEMA_VERSION_V6
        };
        let mut visible = json!({
            "schema_version": schema_version,
            "record_kind": hf_io::VISIBLE_KIND,
            "family": config.family,
            "stage": STAGE0,
            "start_node": self.graph.name(start),
            "subgraph_size": ball.len(),
            "removal_level": config.removal_level,
            "nodes": nodes,
            "edges": edge_records,
        });
        visible["target_node"] = self.graph.name(target).into();
        if k > 1 {
            visible["target_nodes"] = target_names.clone().into();
        }
        // one distance map per target; `distance_to_target` is the per-node
        // minimum over them, which keeps the committed type and is the one
        // target's own map at k = 1
        let distance_to_targets: Vec<BTreeMap<String, u32>> = targets
            .iter()
            .map(|t| {
                pruned
                    .distances_to(*t)
                    .into_iter()
                    .map(|(n, d)| (self.graph.name(n).to_string(), d))
                    .collect()
            })
            .collect();
        let mut distance_to_target: BTreeMap<String, u32> = BTreeMap::new();
        for map in &distance_to_targets {
            for (n, d) in map {
                let entry = distance_to_target.entry(n.clone()).or_insert(*d);
                *entry = (*entry).min(*d);
            }
        }
        let mut on_surviving: Vec<String> = survivors.iter().flatten().cloned().collect();
        on_surviving.sort_unstable();
        on_surviving.dedup();
        let mut removal_set: Vec<Vec<String>> = removed
            .iter()
            .map(|(h, t)| vec![h.clone(), t.clone()])
            .collect();
        removal_set.sort_unstable();
        let path_set: Vec<Vec<String>> = paths.iter().flatten().cloned().collect();
        let by_target: BTreeMap<&String, &BTreeMap<String, u32>> =
            target_names.iter().zip(&distance_to_targets).collect();
        let mut hidden = json!({
            "schema_version": schema_version,
            "record_kind": hf_io::HIDDEN_KIND,
            "family": config.family,
            "stage": STAGE0,
            "split": split,
            "index": index,
            "start_node": self.graph.name(start),
            "target_set": target_names.clone(),
            "target_distance": config.target_distance,
            "cost_bound": bound,
            "path_set": path_set,
            "surviving_paths": survivors,
            "removal_set": removal_set,
            "removed_count": removed.len(),
            "unremovable_count": unremovable,
            "nodes_on_surviving_path": on_surviving,
            "distance_to_target": distance_to_target,
            "sampler": config.as_value(),
        });
        if k > 1 {
            hidden["distance_to_targets"] = json!(by_target);
        }
        let mut episode_id = format!(
            "{}-{split}-{index:06}-{:08x}",
            config.family,
            hash_int(&[key]) % (1u64 << 32)
        );
        if config.targets != 1 {
            episode_id.push_str(&format!("-t{}", config.targets));
        }
        if config.removal_rule != "cheapest-first" {
            episode_id.push('-');
            episode_id.push_str(&config.removal_rule);
            if config.greedy_share < 1.0 {
                episode_id.push_str(&format!("-g{}", python_g(config.greedy_share)));
            }
        }
        hf_io::validate_visible(&visible)?;
        Ok(Ok(Sampled {
            episode_id,
            visible,
            hidden,
            nodes: node_names,
        }))
    }
}

/// The split writer's loop: attempts `0..` until `count` are kept or `40 × count`
/// attempts have been made, in parallel over chunks of indices with the kept
/// ordinals assigned in index order — so a larger draw reproduces a smaller
/// one as its prefix, and `attempts` and the drop tally are what the
/// sequential loop would have produced.
pub struct SampledSplit {
    pub episodes: Vec<Sampled>,
    pub attempts: u64,
    pub drops: BTreeMap<&'static str, u64>,
}

pub fn sample_split(
    sampler: &Sampler<'_>,
    split: &str,
    count: usize,
    embeddings: Option<&(dyn Embeddings + Sync)>,
    chunk: usize,
) -> Result<SampledSplit, HfError> {
    use rayon::prelude::*;
    let cap = count as u64 * 40;
    let mut episodes = Vec::with_capacity(count);
    let mut drops: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut next = 0u64;
    let mut attempts = 0u64;
    while episodes.len() < count && next < cap {
        let end = (next + chunk as u64).min(cap);
        let results: Vec<Result<Result<Sampled, Dropped>, HfError>> = (next..end)
            .into_par_iter()
            .map(|index| sampler.sample(split, index, embeddings.map(|e| e as &dyn Embeddings)))
            .collect();
        for (offset, result) in results.into_iter().enumerate() {
            attempts = next + offset as u64 + 1;
            match result? {
                Ok(episode) => episodes.push(episode),
                Err(reason) => *drops.entry(reason.as_str()).or_default() += 1,
            }
            if episodes.len() >= count {
                break;
            }
        }
        next = end;
    }
    Ok(SampledSplit {
        episodes,
        attempts,
        drops,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_draw_key_is_pythons_dict_repr() {
        let mut c = SamplerConfig::new("wikidata5m", 40, 3, 2);
        c.hub_degree_cap = Some(19);
        c.screen_region = 0.5;
        c.removal_rule = "greedy-path".into();
        c.greedy_share = 0.25;
        assert_eq!(
            c.draw_key("train", 1),
            "wikidata5m|train|1|{'family': 'wikidata5m', 'subgraph_size': 40, 'target_distance': 3, 'removal_level': 2, 'cost_epsilon': 0.5, 'max_paths': 512, 'seed_label': 'real-walk-v1', 'hub_degree_cap': 19, 'screen_region': 0.5}"
        );
        // verified against the on-disk pools: the base hash is the same under every rule and share
        assert_eq!(
            format!("{:08x}", hash_int(&[&c.draw_key("train", 1)]) % (1 << 32)),
            "2b061859"
        );
        let plain = SamplerConfig::new("wikidata5m", 40, 3, 2);
        assert!(plain
            .draw_key("train", 1)
            .ends_with("'seed_label': 'real-walk-v1'}"));
        assert_eq!(python_g(0.25), "0.25");
        assert_eq!(python_g(0.5), "0.5");
        assert_eq!(python_g(0.125), "0.125");
        assert_eq!(python_g(1.0), "1");
        assert_eq!(python_g(0.00001), "1e-05");
        assert_eq!(python_str_repr("it's"), "\"it's\"");
    }

    #[test]
    fn cost_bound_floors() {
        let mut c = SamplerConfig::new("f", 40, 3, 2);
        assert_eq!(c.cost_bound(), 4);
        c.target_distance = 2;
        assert_eq!(c.cost_bound(), 3);
        c.target_distance = 5;
        assert_eq!(c.cost_bound(), 7);
    }
}
