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
//! with the kept ordinals assigned in index order, and `target_set` is a list
//! ready for k > 1 (k = 1 is what this sampler draws).

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
    pub removal_rule: String,
    pub greedy_share: f64,
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
    let route_set: HashSet<Edge> = route_edges.iter().cloned().collect();
    let survivor = paths
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
        .expect("non-empty");
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
        let paths_named: Vec<Vec<String>> = paths.iter().map(names).collect();
        let greedy_member = greedy_rule
            && (config.greedy_share >= 1.0
                || (hash_int(&[&config.seed_label, &key, "greedy-share"]) as f64 / TWO_POW_64)
                    < config.greedy_share);
        let (removed, survivors, unremovable) = if greedy_member {
            // similarity-greedy's route on the unpruned subgraph
            let draft =
                self.episode_graph(&sub, &ball, start, target, &HashSet::new(), &paths_named);
            let trace = similarity_greedy_trace(&draft, embeddings.expect("checked"), None);
            let route = trace.route(&draft.start, &draft.target);
            if route.is_empty() {
                return Ok(Err(Dropped::GreedyRouteMissing));
            }
            choose_greedy_removals(&paths_named, &route, config.removal_level)?
        } else {
            choose_removals(&paths_named, config.removal_level, &config.seed_label, &key)?
        };
        let Some(mut sampled) = self.episode_from(
            split,
            index,
            &key,
            start,
            target,
            &sub,
            &ball,
            &paths_named,
            bound,
            &removed,
            &survivors,
            unremovable,
        )?
        else {
            return Ok(Err(Dropped::SurvivorsMismatch));
        };
        if greedy_rule {
            let g = self.episode_graph_of(&sampled);
            let greedy = similarity_greedy_trace(&g, embeddings.expect("checked"), None).expansions;
            let oracle = oracle_trace(&g).expansions;
            let hidden = sampled.hidden.as_object_mut().unwrap();
            hidden.insert(
                "greedy_overshoot".into(),
                (greedy as i64 - oracle as i64).into(),
            );
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

    fn episode_graph_of(&self, sampled: &Sampled) -> EpisodeGraph {
        let v = &sampled.visible;
        let h = &sampled.hidden;
        let edges = v["edges"].as_array().unwrap().iter().map(|e| {
            (
                e["source"].as_str().unwrap().to_string(),
                e["target"].as_str().unwrap().to_string(),
            )
        });
        let survivors: Vec<Vec<String>> =
            serde_json::from_value(h["surviving_paths"].clone()).unwrap();
        let distances: std::collections::HashMap<String, u32> =
            serde_json::from_value(h["distance_to_target"].clone()).unwrap();
        EpisodeGraph::new(
            v["start_node"].as_str().unwrap(),
            h["target_set"][0].as_str().unwrap(),
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
        target: NodeId,
        sub: &Subgraph,
        ball: &[NodeId],
        paths: &[Vec<String>],
        bound: u32,
        removed: &HashSet<Edge>,
        survivors: &[Vec<String>],
        unremovable: u32,
    ) -> Result<Option<Sampled>, HfError> {
        let config = &self.config;
        // prune by node pair and recompute the survivors on the pruned subgraph
        let pruned = sub.without_pairs(|h, t| {
            removed.contains(&(
                self.graph.name(h).to_string(),
                self.graph.name(t).to_string(),
            ))
        });
        let recomputed: Vec<Vec<String>> = pruned
            .simple_paths(start, target, bound)
            .iter()
            .map(|p| p.iter().map(|n| self.graph.name(*n).to_string()).collect())
            .collect();
        if recomputed.is_empty() || recomputed != survivors {
            return Ok(None);
        }
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
        let mut visible = json!({
            "schema_version": hf_io::SCHEMA_VERSION_V5,
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
        let mut distance_to_target: BTreeMap<String, u32> = BTreeMap::new();
        for (n, d) in pruned.distances_to(target) {
            distance_to_target.insert(self.graph.name(n).to_string(), d);
        }
        let mut on_surviving: Vec<String> = survivors.iter().flatten().cloned().collect();
        on_surviving.sort_unstable();
        on_surviving.dedup();
        let mut removal_set: Vec<Vec<String>> = removed
            .iter()
            .map(|(h, t)| vec![h.clone(), t.clone()])
            .collect();
        removal_set.sort_unstable();
        let hidden = json!({
            "schema_version": hf_io::SCHEMA_VERSION_V5,
            "record_kind": hf_io::HIDDEN_KIND,
            "family": config.family,
            "stage": STAGE0,
            "split": split,
            "index": index,
            "start_node": self.graph.name(start),
            "target_set": [self.graph.name(target)],
            "target_distance": config.target_distance,
            "cost_bound": bound,
            "path_set": paths,
            "surviving_paths": survivors,
            "removal_set": removal_set,
            "removed_count": removed.len(),
            "unremovable_count": unremovable,
            "nodes_on_surviving_path": on_surviving,
            "distance_to_target": distance_to_target,
            "sampler": config.as_value(),
        });
        let mut episode_id = format!(
            "{}-{split}-{index:06}-{:08x}",
            config.family,
            hash_int(&[key]) % (1u64 << 32)
        );
        if config.removal_rule != "cheapest-first" {
            episode_id.push('-');
            episode_id.push_str(&config.removal_rule);
            if config.greedy_share < 1.0 {
                episode_id.push_str(&format!("-g{}", python_g(config.greedy_share)));
            }
        }
        hf_io::validate_visible(&visible)?;
        Ok(Some(Sampled {
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
