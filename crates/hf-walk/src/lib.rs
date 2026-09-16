//! The learned walk of `model_v5.walk`, run in lockstep over a batch of
//! episodes so every step is one scoring call for all live episodes rather
//! than one call per decision per episode — the structural fix for the
//! per-decision GPU round trips the Python engine paid. Nothing about a
//! single episode's walk changes: children are pushed in `edge_id` order,
//! registration happens on sight, the first maximum wins, `frontier.pop(chosen)`
//! keeps insertion order, `examined` counts every child seen, the stop rules
//! are `exhaust` (stop on registration or an empty frontier) and `learned`
//! (the stop head, consulted after each decision and before the expansion).
//!
//! Three feature sets are shipped behind one trait: `RawV5`, the Python
//! candidate row (`[c, q, path_mean, parent]` + seven structure features, raw
//! context and query embeddings, the `[cos, is-parent]` pair channel);
//! `RelationalV6`, the redesign, in which no raw embedding coordinate enters
//! any channel; and `RelationalV6Prev`, that redesign with the
//! selective-previous-nodes channels on the context token and the pair
//! channel. A builder receives a `VisibleIndex` — the borrowed visible arrays
//! of one episode — so no feature can be a function of what the sampler kept
//! back. The scorer is a trait so the walk is tested with stubs and driven by
//! the libtorch model in `hf-model`.

pub mod features;
pub mod visible;

use std::collections::{HashMap, HashSet};

use hf_core::HfError;
use serde::Serialize;

pub use features::{FeatureSet, RawV5, RelationalV6, RelationalV6Prev, STOP_DIM, STRUCTURE_DIM};
pub use visible::VisibleIndex;

/// A node's index within one episode's subgraph.
pub type Local = u32;

/// One episode, indexed for the walk: adjacency in `edge_id` order, raw and
/// unit embeddings (zero when the cache lacks the node), the query (the shown
/// target's embedding at stage 0), and the hidden truth the losses need.
pub struct EpisodeIndex {
    pub episode_id: String,
    pub names: Vec<String>,
    pub start: Local,
    pub target_shown: Option<Local>,
    pub hidden_targets: Vec<Local>,
    pub out: Vec<Vec<Local>>,
    pub edim: usize,
    pub emb: Vec<f32>,
    pub unit: Vec<f32>,
    pub query: Vec<f32>,
    /// `nodes_on_surviving_path` membership, per local node.
    pub on_path: Vec<bool>,
    /// `distance_to_target` per local node, `None` when unreachable.
    pub distance: Vec<Option<u32>>,
    pub removed_count: u32,
    pub greedy_overshoot: Option<i64>,
}

impl EpisodeIndex {
    /// `RealEpisodeIndex(episode, embeddings, embedding_dimension)`.
    pub fn new(
        episode: &hf_io::RealEpisode,
        embeddings: &hf_embed::EmbeddingMatrix,
        edim: usize,
    ) -> Result<Self, HfError> {
        let names: Vec<String> = episode
            .visible
            .nodes
            .iter()
            .map(|n| n.node.clone())
            .collect();
        let local: HashMap<&str, Local> = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i as Local))
            .collect();
        let lookup = |name: &str| -> Result<Local, HfError> {
            local.get(name).copied().ok_or_else(|| {
                HfError::BandH(format!(
                    "{}: node {name} is not in the subgraph",
                    episode.episode_id
                ))
            })
        };
        let mut edges: Vec<(u32, Local, Local)> = episode
            .visible
            .edges
            .iter()
            .map(|e| Ok((e.edge_id, lookup(&e.source)?, lookup(&e.target)?)))
            .collect::<Result<_, HfError>>()?;
        edges.sort_unstable(); // Python sorts (edge_id, target, relation): edge_id order
        let mut out = vec![Vec::new(); names.len()];
        for (_, s, t) in edges {
            out[s as usize].push(t);
        }
        let n = names.len();
        let mut emb = vec![0f32; n * edim];
        let mut unit = vec![0f32; n * edim];
        for (i, name) in names.iter().enumerate() {
            if let Some(row) = embeddings.get(name) {
                if row.len() < edim {
                    return Err(HfError::BandH(format!(
                        "embedding of {name} has width {} < {edim}",
                        row.len()
                    )));
                }
                emb[i * edim..(i + 1) * edim].copy_from_slice(&row[..edim]);
                let norm = row[..edim]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-12);
                for j in 0..edim {
                    unit[i * edim + j] = row[j] / norm;
                }
            }
        }
        let start = lookup(&episode.visible.start_node)?;
        let target_shown = episode
            .visible
            .target_node
            .as_deref()
            .map(lookup)
            .transpose()?;
        let hidden_targets = episode
            .hidden
            .target_set
            .iter()
            .map(|t| lookup(t))
            .collect::<Result<Vec<_>, _>>()?;
        let query = match target_shown {
            Some(t) => emb[t as usize * edim..(t as usize + 1) * edim].to_vec(),
            None => {
                return Err(HfError::Invalid(
                    "stage 1 walks need a query; not implemented".into(),
                ))
            }
        };
        let on_set: HashSet<&str> = episode
            .hidden
            .nodes_on_surviving_path
            .iter()
            .map(String::as_str)
            .collect();
        let on_path = names.iter().map(|n| on_set.contains(n.as_str())).collect();
        let distance = names
            .iter()
            .map(|n| episode.hidden.distance_to_target.get(n).copied())
            .collect();
        Ok(Self {
            episode_id: episode.episode_id.clone(),
            names,
            start,
            target_shown,
            hidden_targets,
            out,
            edim,
            emb,
            unit,
            query,
            on_path,
            distance,
            removed_count: episode.hidden.removed_count,
            greedy_overshoot: episode.hidden.greedy_overshoot,
        })
    }

    pub fn degree(&self, node: Local) -> usize {
        self.out[node as usize].len()
    }

    pub fn emb(&self, node: Local) -> &[f32] {
        &self.emb[node as usize * self.edim..(node as usize + 1) * self.edim]
    }

    pub fn unit(&self, node: Local) -> &[f32] {
        &self.unit[node as usize * self.edim..(node as usize + 1) * self.edim]
    }
}

/// Cosine as `model_v5.cosine`: 0 when either norm is 0.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for (x, y) in a.iter().zip(b) {
        dot += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        (dot / (na.sqrt() * nb.sqrt())) as f32
    }
}

/// One frontier entry: `(node, parent, depth, path mean)`.
#[derive(Clone, Debug)]
pub struct Entry {
    pub node: Local,
    pub parent: Local,
    pub depth: u32,
    pub path_mean: Vec<f32>,
}

/// The tensors of one decision, flattened row-major; what a scorer consumes.
#[derive(Clone, Debug, Default)]
pub struct DecisionItem {
    pub frontier_len: usize,
    pub context_len: usize,
    pub cand: Vec<f32>,
    pub ctx: Vec<f32>,
    pub query: Vec<f32>,
    pub pair: Vec<f32>,
}

/// A batch of decisions across episodes.
#[derive(Debug, Default)]
pub struct DecisionBatch {
    pub items: Vec<DecisionItem>,
}

/// Scores per item (each `frontier_len` long) and, under a prior, the residuals.
#[derive(Debug, Default)]
pub struct Scored {
    pub scores: Vec<Vec<f32>>,
    pub residuals: Option<Vec<Vec<f32>>>,
}

/// What drives the walk: the model, or a stub in tests.
pub trait Scorer {
    fn score(&mut self, batch: &DecisionBatch) -> Result<Scored, HfError>;
    fn stop_logits(&mut self, rows: &[[f32; STOP_DIM]]) -> Result<Vec<f32>, HfError>;
}

/// One recorded decision, for the gradient pass and the readers.
#[derive(Clone, Debug)]
pub struct Decision {
    pub frontier: Vec<Local>,
    pub parents: Vec<Local>,
    pub depths: Vec<u32>,
    pub chosen: usize,
    pub stop_features: [f32; STOP_DIM],
    pub expansions_before: usize,
    pub registered_before: bool,
    /// The scoring input, kept when the caller asked for it.
    pub item: Option<DecisionItem>,
    /// `cosine(c, q)` per candidate — the greedy prior's column.
    pub cosines: Vec<f32>,
}

/// A dumped candidate record (`candidate_dump.jsonl.gz` line body).
#[derive(Clone, Debug, Serialize)]
pub struct CandidateRecord {
    pub frontier: Vec<String>,
    pub scores: Vec<f32>,
    pub cosines: Vec<f32>,
    pub chosen: usize,
    /// The name of each frontier candidate's discovery parent, in `frontier` order.
    pub parents: Vec<String>,
    /// Each frontier candidate's depth, in `frontier` order.
    pub depths: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopRule {
    Exhaust,
    Learned,
}

impl StopRule {
    pub fn parse(s: &str) -> Result<Self, HfError> {
        match s {
            "exhaust" => Ok(Self::Exhaust),
            "learned" => Ok(Self::Learned),
            other => Err(HfError::Invalid(format!("unknown stop rule: {other}"))),
        }
    }
}

/// The walk's result for one episode (`model.walk`'s dict).
#[derive(Clone, Debug)]
pub struct WalkResult {
    pub expanded: Vec<Local>,
    pub decisions: Vec<Decision>,
    pub registered_at: Option<usize>,
    pub stop_reason: &'static str,
    pub examined: usize,
    pub residuals: Option<Vec<Vec<f32>>>,
    pub margins: Vec<Option<f32>>,
    pub cosine_margins: Vec<Option<f32>>,
    pub candidates: Option<Vec<CandidateRecord>>,
}

impl WalkResult {
    pub fn registered(&self) -> bool {
        self.registered_at.is_some()
    }

    pub fn expansions(&self) -> usize {
        self.expanded.len()
    }
}

struct State<'a> {
    index: &'a EpisodeIndex,
    expanded: Vec<Local>,
    seen: HashSet<Local>,
    /// The discovery parent of every expanded node (the start has none).
    parent_of: HashMap<Local, Local>,
    frontier: Vec<Entry>,
    registered: Vec<bool>,
    registered_at: Option<usize>,
    stop_reason: &'static str,
    examined: usize,
    decisions: Vec<Decision>,
    residuals: Vec<Vec<f32>>,
    margins: Vec<Option<f32>>,
    cosine_margins: Vec<Option<f32>>,
    candidates: Option<Vec<CandidateRecord>>,
    live: bool,
}

impl<'a> State<'a> {
    fn new(index: &'a EpisodeIndex, record_candidates: bool) -> Self {
        let mut s = Self {
            index,
            expanded: Vec::new(),
            seen: HashSet::from([index.start]),
            parent_of: HashMap::new(),
            frontier: Vec::new(),
            registered: vec![false; index.hidden_targets.len()],
            registered_at: None,
            stop_reason: "exhausted",
            examined: 0,
            decisions: Vec::new(),
            residuals: Vec::new(),
            margins: Vec::new(),
            cosine_margins: Vec::new(),
            candidates: if record_candidates {
                Some(Vec::new())
            } else {
                None
            },
            live: true,
        };
        s.expanded.push(index.start);
        let start_mean = index.emb(index.start).to_vec();
        s.push_children(index.start, 0, &start_mean);
        s.check_registered();
        s
    }

    fn all_registered(&self) -> bool {
        self.registered.iter().all(|r| *r)
    }

    fn push_children(&mut self, node: Local, depth: u32, path_mean: &[f32]) {
        let index = self.index;
        for &child in &index.out[node as usize] {
            self.examined += 1;
            if let Some(t) = index.hidden_targets.iter().position(|t| *t == child) {
                if !self.registered[t] {
                    self.registered[t] = true;
                    if self.all_registered() && self.registered_at.is_none() {
                        self.registered_at = Some(self.expanded.len());
                    }
                }
            }
            if self.seen.contains(&child) {
                continue;
            }
            self.seen.insert(child);
            let k = depth + 1;
            let e = index.emb(child);
            let mean: Vec<f32> = path_mean
                .iter()
                .zip(e)
                .map(|(m, x)| (m * depth as f32 + x) / k as f32)
                .collect();
            self.frontier.push(Entry {
                node: child,
                parent: node,
                depth: k,
                path_mean: mean,
            });
        }
    }

    fn check_registered(&mut self) {
        if self.registered_at.is_some() && self.index.target_shown.is_some() {
            self.stop_reason = "target_registered";
            self.live = false;
        }
        if self.frontier.is_empty() {
            self.live = false;
        }
    }

    fn finish(self, with_prior: bool) -> WalkResult {
        WalkResult {
            expanded: self.expanded,
            decisions: self.decisions,
            registered_at: self.registered_at,
            stop_reason: self.stop_reason,
            examined: self.examined,
            residuals: if with_prior {
                Some(self.residuals)
            } else {
                None
            },
            margins: self.margins,
            cosine_margins: self.cosine_margins,
            candidates: self.candidates,
        }
    }
}

/// Top-2 gap of a row; `None` under two entries.
fn top2_gap(values: &[f32]) -> Option<f32> {
    if values.len() < 2 {
        return None;
    }
    let (mut best, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &v in values {
        if v > best {
            second = best;
            best = v;
        } else if v > second {
            second = v;
        }
    }
    Some(best - second)
}

/// First maximum, as `torch.argmax`.
fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best
}

/// Options for one batched walk.
#[derive(Clone, Copy, Debug)]
pub struct WalkOptions {
    pub stop_rule: StopRule,
    pub record_candidates: bool,
    /// Keep every decision's scoring input for a gradient pass.
    pub keep_items: bool,
    /// The scorer returns residuals (the greedy prior is on).
    pub with_prior: bool,
}

/// Walk every episode in lockstep: at each step, one scoring call covers
/// every live episode's frontier.
pub fn walk_batch(
    indexes: &[&EpisodeIndex],
    features: &dyn FeatureSet,
    scorer: &mut dyn Scorer,
    options: WalkOptions,
) -> Result<Vec<WalkResult>, HfError> {
    let mut states: Vec<State> = indexes
        .iter()
        .map(|i| State::new(i, options.record_candidates))
        .collect();
    loop {
        let live: Vec<usize> = states
            .iter()
            .enumerate()
            .filter(|(_, s)| s.live)
            .map(|(i, _)| i)
            .collect();
        if live.is_empty() {
            break;
        }
        // build every live episode's decision (parallel across episodes)
        let items: Vec<(DecisionItem, Vec<f32>)> = {
            use rayon::prelude::*;
            live.par_iter()
                .map(|&i| {
                    let s = &states[i];
                    let item = features.build(
                        VisibleIndex::new(s.index),
                        &s.frontier,
                        &s.expanded,
                        &s.parent_of,
                    );
                    let cosines: Vec<f32> = s
                        .frontier
                        .iter()
                        .map(|e| cosine(s.index.emb(e.node), &s.index.query))
                        .collect();
                    (item, cosines)
                })
                .collect()
        };
        let batch = DecisionBatch {
            items: items.iter().map(|(it, _)| it.clone()).collect(),
        };
        let scored = scorer.score(&batch)?;
        if scored.scores.len() != live.len() {
            return Err(HfError::BandH(format!(
                "the scorer returned {} rows for {} decisions",
                scored.scores.len(),
                live.len()
            )));
        }
        let mut stop_rows: Vec<(usize, [f32; STOP_DIM])> = Vec::new();
        for (k, &i) in live.iter().enumerate() {
            let (item, cosines) = &items[k];
            let s = &mut states[i];
            let n = s.frontier.len();
            let scores = &scored.scores[k];
            if scores.len() != n {
                return Err(HfError::BandH(format!(
                    "score row {} wide for a frontier of {n}",
                    scores.len()
                )));
            }
            let chosen = argmax(scores);
            let best = scores[chosen];
            let mean = scores.iter().sum::<f32>() / n as f32;
            s.margins.push(top2_gap(scores));
            s.cosine_margins.push(top2_gap(cosines));
            if let Some(r) = &scored.residuals {
                s.residuals.push(r[k].clone());
            }
            if let Some(c) = &mut s.candidates {
                c.push(CandidateRecord {
                    frontier: s
                        .frontier
                        .iter()
                        .map(|e| s.index.names[e.node as usize].clone())
                        .collect(),
                    scores: scores.clone(),
                    cosines: cosines.clone(),
                    chosen,
                    parents: s
                        .frontier
                        .iter()
                        .map(|e| s.index.names[e.parent as usize].clone())
                        .collect(),
                    depths: s.frontier.iter().map(|e| e.depth).collect(),
                });
            }
            let entry = &s.frontier[chosen];
            let stop_features = features::stop_row(
                s.expanded.len(),
                n,
                best,
                mean,
                entry.depth,
                cosines[chosen],
                s.examined,
            );
            let decision = Decision {
                frontier: s.frontier.iter().map(|e| e.node).collect(),
                parents: s.frontier.iter().map(|e| e.parent).collect(),
                depths: s.frontier.iter().map(|e| e.depth).collect(),
                chosen,
                stop_features,
                expansions_before: s.expanded.len(),
                registered_before: s.registered_at.is_some(),
                item: if options.keep_items {
                    Some(item.clone())
                } else {
                    None
                },
                cosines: cosines.clone(),
            };
            s.decisions.push(decision);
            if options.stop_rule == StopRule::Learned {
                stop_rows.push((i, stop_features));
            }
        }
        // the learned stop rule consults the stop head before expanding
        let mut stopped: HashSet<usize> = HashSet::new();
        if !stop_rows.is_empty() {
            let rows: Vec<[f32; STOP_DIM]> = stop_rows.iter().map(|(_, r)| *r).collect();
            let logits = scorer.stop_logits(&rows)?;
            for ((i, _), logit) in stop_rows.iter().zip(logits) {
                if logit > 0.0 {
                    stopped.insert(*i);
                }
            }
        }
        for &i in &live {
            let s = &mut states[i];
            if stopped.contains(&i) {
                s.stop_reason = "learned_stop";
                s.live = false;
                continue;
            }
            let chosen = s.decisions.last().unwrap().chosen;
            let Entry {
                node,
                parent,
                depth,
                path_mean,
            } = s.frontier.remove(chosen);
            s.parent_of.insert(node, parent);
            s.expanded.push(node);
            s.push_children(node, depth, &path_mean);
            s.check_registered();
        }
    }
    Ok(states
        .into_iter()
        .map(|s| s.finish(options.with_prior))
        .collect())
}

/// `candidate_labels`: per decision, on-path flags and `log1p(distance)` with `-1` for unreachable.
pub fn candidate_labels(
    index: &EpisodeIndex,
    decisions: &[Decision],
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let on: Vec<Vec<f32>> = decisions
        .iter()
        .map(|d| {
            d.frontier
                .iter()
                .map(|n| if index.on_path[*n as usize] { 1.0 } else { 0.0 })
                .collect()
        })
        .collect();
    let dist: Vec<Vec<f32>> = decisions
        .iter()
        .map(|d| {
            d.frontier
                .iter()
                .map(|n| {
                    index.distance[*n as usize]
                        .map(|v| (v as f64).ln_1p() as f32)
                        .unwrap_or(-1.0)
                })
                .collect()
        })
        .collect();
    (on, dist)
}

/// `stop_labels_v5`.
pub fn stop_labels(decisions: &[Decision]) -> Vec<f32> {
    decisions
        .iter()
        .map(|d| if d.registered_before { 1.0 } else { 0.0 })
        .collect()
}
