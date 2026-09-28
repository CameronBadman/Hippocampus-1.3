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
//! At k >= 2 the record shows every target, registration is per target and
//! the walk completes when **all** of them are registered, so at least one is
//! unregistered at every decision; `RelationalV6K` reduces every channel that
//! reads the query over that unregistered set (`K_TARGETS_DESIGN.md` §2).
//!
//! Four feature sets are shipped behind one trait: `RawV5`, the Python
//! candidate row (`[c, q, path_mean, parent]` + seven structure features, raw
//! context and query embeddings, the `[cos, is-parent]` pair channel);
//! `RelationalV6`, the redesign, in which no raw embedding coordinate enters
//! any channel; and `RelationalV6Prev`, that redesign with the
//! selective-previous-nodes channels on the context token and the pair
//! channel; and `RelationalV6K`, that redesign over k targets. A builder
//! receives a `VisibleIndex` — the borrowed visible arrays of one episode and
//! the walk's own registration mask — so no feature can be a function of what
//! the sampler kept back. The scorer is a trait so the walk is tested with
//! stubs and driven by the libtorch model in `hf-model`.

pub mod features;
pub mod visible;

use std::collections::{HashMap, HashSet};

use hf_core::HfError;
use serde::Serialize;

pub use features::{
    FeatureSet, RawV5, RelationalV6, RelationalV6K, RelationalV6Prev, STOP_DIM, STRUCTURE_DIM,
};
pub use visible::VisibleIndex;

/// A node's index within one episode's subgraph.
pub type Local = u32;

/// Where the walk's query vector comes from.
///
/// `TargetEmbedding` is stage 0 and the default: the query is the SHOWN
/// target's own embedding row, so the record names what is being looked for.
/// `EpisodeQuery` is stage 1: the visible payload shows no target at all and
/// the query is the episode's own question vector, read from a sidecar
/// embedding cache keyed by episode id. Under it the target reaches the engine
/// through the hidden payload only, for registration and the losses' labels —
/// `VisibleIndex` is handed an empty list of shown targets, so no feature
/// builder can be a function of the target's identity.
///
/// `DeletedPayload` is the R1 premise's probe (`R1_PREMISE_PLAN.md` §2 E2):
/// the query is a DELETED node's own embedding row — the stage-0 query type —
/// whose node is absent from the ball, so there is nothing to register. The
/// index has **no registration target at all** and the walk ends only on an
/// empty frontier or at the caller's expansion cap
/// ([`walk_batch_capped`]). It is built by [`EpisodeIndex::for_deletion`]
/// alone: `parse` does not accept it, so no training config can select it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QuerySource {
    #[default]
    TargetEmbedding,
    EpisodeQuery,
    DeletedPayload,
}

impl QuerySource {
    pub fn parse(s: &str) -> Result<Self, HfError> {
        match s {
            "target_embedding" => Ok(Self::TargetEmbedding),
            "episode_query" => Ok(Self::EpisodeQuery),
            other => Err(HfError::Invalid(format!(
                "unknown query source {other:?}; it is \"target_embedding\" or \"episode_query\""
            ))),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::TargetEmbedding => "target_embedding",
            Self::EpisodeQuery => "episode_query",
            Self::DeletedPayload => "deleted_payload",
        }
    }
}

/// The query's provenance for one run: the source and, under `episode_query`,
/// the sidecar cache the vectors are read from (an ordinary v5 embedding cache
/// whose "node" ids are EPISODE ids).
#[derive(Clone, Copy, Default)]
pub struct QueryVectors<'a> {
    pub source: QuerySource,
    pub cache: Option<&'a hf_embed::EmbeddingMatrix>,
}

impl<'a> QueryVectors<'a> {
    /// Stage 0: the query is the shown target's own row, no sidecar.
    pub fn target_embedding() -> Self {
        Self::default()
    }

    /// Stage 1: the query is this episode's own question vector.
    pub fn episode_query(cache: &'a hf_embed::EmbeddingMatrix) -> Self {
        Self {
            source: QuerySource::EpisodeQuery,
            cache: Some(cache),
        }
    }
}

/// One episode, indexed for the walk: adjacency in `edge_id` order, raw and
/// unit embeddings (zero when the cache lacks the node), the query (the shown
/// target's embedding at stage 0, the episode's own question vector under
/// `episode_query`), and the hidden truth the losses need.
pub struct EpisodeIndex {
    pub episode_id: String,
    pub names: Vec<String>,
    pub start: Local,
    /// Every shown target, in the record's order (`target_nodes`, which is
    /// node-name order, led by `target_node`); one entry at k = 1, and NONE
    /// under `episode_query`, where the visible payload shows no target.
    pub targets_shown: Vec<Local>,
    pub hidden_targets: Vec<Local>,
    /// Where `queries` came from; `registration_targets` follows it.
    pub query_source: QuerySource,
    pub out: Vec<Vec<Local>>,
    pub edim: usize,
    pub emb: Vec<f32>,
    pub unit: Vec<f32>,
    /// One query per shown target, flattened: the target's own embedding at
    /// stage 0. `queries[..edim]` is v1's single query. Under `episode_query`
    /// there is exactly one, the episode's own question vector, and it belongs
    /// to no target.
    pub queries: Vec<f32>,
    /// Each query unit-normalised, the context token's own arithmetic.
    pub unit_queries: Vec<f32>,
    /// `nodes_on_surviving_path` membership, per local node.
    pub on_path: Vec<bool>,
    /// `distance_to_target` per local node, `None` when unreachable.
    pub distance: Vec<Option<u32>>,
    pub removed_count: u32,
    pub greedy_overshoot: Option<i64>,
}

impl EpisodeIndex {
    /// `RealEpisodeIndex(episode, embeddings, embedding_dimension)` at stage 0:
    /// the query is the shown target's own embedding row.
    pub fn new(
        episode: &hf_io::RealEpisode,
        embeddings: &hf_embed::EmbeddingMatrix,
        edim: usize,
    ) -> Result<Self, HfError> {
        Self::new_with_query(episode, embeddings, edim, QueryVectors::target_embedding())
    }

    /// The same index with the query's provenance named. Under
    /// `QuerySource::EpisodeQuery` the record must be a stage-1 one — it may
    /// show no target, it must carry exactly one hidden target, and its query
    /// vector must be in the sidecar cache under its own episode id.
    pub fn new_with_query(
        episode: &hf_io::RealEpisode,
        embeddings: &hf_embed::EmbeddingMatrix,
        edim: usize,
        queries: QueryVectors<'_>,
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
        // The shown targets, from the VISIBLE side only: `target_nodes` on a
        // k >= 2 record (6.0.0), `target_node` otherwise. The walk's
        // registration mask is built from these, never from what the sampler
        // kept back (`K_TARGETS_DESIGN.md` §2, last paragraph).
        let shown: Vec<&str> = match &episode.visible.target_nodes {
            Some(t) => t.iter().map(String::as_str).collect(),
            None => episode.visible.target_node.as_deref().into_iter().collect(),
        };
        let mut targets_shown = shown
            .iter()
            .map(|t| lookup(t))
            .collect::<Result<Vec<_>, _>>()?;
        let hidden_targets = episode
            .hidden
            .target_set
            .iter()
            .map(|t| lookup(t))
            .collect::<Result<Vec<_>, _>>()?;
        let query_source = queries.source;
        let (queries, unit_queries) = match query_source {
            QuerySource::DeletedPayload => {
                return Err(HfError::Refused(format!(
                    "{}: a deleted_payload index is built by EpisodeIndex::for_deletion, \
                     never from a sampled record",
                    episode.episode_id
                )))
            }
            QuerySource::TargetEmbedding => {
                if queries.cache.is_some() {
                    return Err(HfError::Invalid(
                        "a query sidecar was given but the query source is target_embedding".into(),
                    ));
                }
                if targets_shown.is_empty() {
                    return Err(HfError::Invalid(format!(
                        "{}: the visible payload shows no target; a stage-1 walk needs \
                         data.query_source = \"episode_query\" and a query sidecar",
                        episode.episode_id
                    )));
                }
                let mut q = Vec::with_capacity(targets_shown.len() * edim);
                let mut u = Vec::with_capacity(targets_shown.len() * edim);
                for t in &targets_shown {
                    let row = &emb[*t as usize * edim..(*t as usize + 1) * edim];
                    q.extend_from_slice(row);
                    u.extend_from_slice(&visible::unit_query(row));
                }
                (q, u)
            }
            QuerySource::EpisodeQuery => {
                // stage 1 only: the record must show no target, and `hf-io`
                // already refuses a stage-1 payload carrying `target_node`.
                // The refusal is repeated here because an index is also built
                // from records held in memory, which no reader has validated.
                if episode.visible.stage != hf_io::STAGES[1] {
                    return Err(HfError::BandH(format!(
                        "{}: query_source episode_query needs a {} record, not {:?}",
                        episode.episode_id,
                        hf_io::STAGES[1],
                        episode.visible.stage
                    )));
                }
                if !targets_shown.is_empty() {
                    return Err(HfError::BandH(format!(
                        "{}: the visible payload names a target under query_source \
                         episode_query; the target is the hidden payload's alone",
                        episode.episode_id
                    )));
                }
                if hidden_targets.len() != 1 {
                    return Err(HfError::BandH(format!(
                        "{}: query_source episode_query is k = 1 only; this record carries \
                         {} targets",
                        episode.episode_id,
                        hidden_targets.len()
                    )));
                }
                let cache = queries.cache.ok_or_else(|| {
                    HfError::Invalid(
                        "query_source episode_query needs a query embedding cache".into(),
                    )
                })?;
                let row = cache.get(&episode.episode_id).ok_or_else(|| {
                    HfError::BandH(format!(
                        "the query cache holds no vector for episode {}",
                        episode.episode_id
                    ))
                })?;
                if row.len() < edim {
                    return Err(HfError::BandH(format!(
                        "the query of episode {} has width {} < {edim}",
                        episode.episode_id,
                        row.len()
                    )));
                }
                targets_shown = Vec::new();
                (row[..edim].to_vec(), visible::unit_query(&row[..edim]))
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
            targets_shown,
            hidden_targets,
            query_source,
            out,
            edim,
            emb,
            unit,
            queries,
            unit_queries,
            on_path,
            distance,
            removed_count: episode.hidden.removed_count,
            greedy_overshoot: episode.hidden.greedy_overshoot,
        })
    }

    /// The R1 premise's probe index (`QuerySource::DeletedPayload`): a ball
    /// rebuilt without the deleted node, walked on that node's own row.
    ///
    /// `nodes` and `edges` are the deletion record's VISIBLE payload (node
    /// names; `(edge_id, source, target)`); `query` is the sidecar row. There
    /// is no hidden input: no target is shown or registered, every on-path
    /// flag is false and every distance unknown — the probe computes no loss.
    /// The query's own node is not in the ball: `hf-splits deletions` refuses
    /// to write a record that names it. It is not re-checked here by vector
    /// equality, because distinct Wikidata nodes can share a text and so a
    /// row; the index is never told the deleted node's name.
    pub fn for_deletion(
        episode_id: &str,
        start_node: &str,
        nodes: &[String],
        edges: &[(u32, String, String)],
        embeddings: &hf_embed::EmbeddingMatrix,
        edim: usize,
        query: &[f32],
    ) -> Result<Self, HfError> {
        if query.len() < edim {
            return Err(HfError::BandH(format!(
                "{episode_id}: the query has width {} < {edim}",
                query.len()
            )));
        }
        let local: HashMap<&str, Local> = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i as Local))
            .collect();
        if local.len() != nodes.len() {
            return Err(HfError::BandH(format!(
                "{episode_id}: a node is listed twice"
            )));
        }
        let lookup = |name: &str| -> Result<Local, HfError> {
            local.get(name).copied().ok_or_else(|| {
                HfError::BandH(format!("{episode_id}: node {name} is not in the subgraph"))
            })
        };
        let mut sorted: Vec<(u32, Local, Local)> = edges
            .iter()
            .map(|(id, s, t)| Ok((*id, lookup(s)?, lookup(t)?)))
            .collect::<Result<_, HfError>>()?;
        sorted.sort_unstable();
        let n = nodes.len();
        let mut out = vec![Vec::new(); n];
        for (_, s, t) in sorted {
            out[s as usize].push(t);
        }
        let mut emb = vec![0f32; n * edim];
        let mut unit = vec![0f32; n * edim];
        for (i, name) in nodes.iter().enumerate() {
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
        let start = lookup(start_node)?;
        Ok(Self {
            episode_id: episode_id.to_string(),
            names: nodes.to_vec(),
            start,
            targets_shown: Vec::new(),
            hidden_targets: Vec::new(),
            query_source: QuerySource::DeletedPayload,
            out,
            edim,
            emb,
            unit,
            queries: query[..edim].to_vec(),
            unit_queries: visible::unit_query(&query[..edim]),
            on_path: vec![false; n],
            distance: vec![None; n],
            removed_count: 0,
            greedy_overshoot: None,
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

    /// The first shown target, v1's `target_shown`.
    pub fn target_shown(&self) -> Option<Local> {
        self.targets_shown.first().copied()
    }

    /// How many targets the record shows (`k`); zero under `episode_query`.
    pub fn target_count(&self) -> usize {
        self.targets_shown.len()
    }

    /// How many queries the row reads: `k` at stage 0, one under
    /// `episode_query`. It is the width of the walk's registration mask and
    /// the index set every query channel reduces over.
    pub fn query_count(&self) -> usize {
        self.queries.len().checked_div(self.edim).unwrap_or(0)
    }

    /// The targets the walk registers on: the SHOWN targets at stage 0, which
    /// is what `K_TARGETS_DESIGN.md` §2 requires of a record that names them;
    /// the episode's one hidden target under `episode_query`, where nothing
    /// visible names it. The mask built from these lives in the walk's own
    /// state and never enters `VisibleIndex`.
    pub fn registration_targets(&self) -> &[Local] {
        match self.query_source {
            QuerySource::TargetEmbedding => &self.targets_shown,
            QuerySource::EpisodeQuery => &self.hidden_targets,
            QuerySource::DeletedPayload => &[],
        }
    }

    /// The `t`-th shown target's query vector.
    pub fn query_of(&self, t: usize) -> &[f32] {
        &self.queries[t * self.edim..(t + 1) * self.edim]
    }

    /// The query: the first shown target's embedding at stage 0, the episode's
    /// own question vector under `episode_query`.
    pub fn query(&self) -> &[f32] {
        self.query_of(0)
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
    /// `score`, and from the SAME forward pass each item's final hidden
    /// states over its valid candidates, mean-pooled then max-pooled (`2H`
    /// values per item): the R1 stop head's pooled input
    /// (`R1_HEAD_DESIGN.md` §4). The scores must be bit-identical to `score`'s.
    fn score_pooled(&mut self, batch: &DecisionBatch) -> Result<(Scored, Vec<Vec<f32>>), HfError> {
        let _ = batch;
        Err(HfError::Invalid(
            "this scorer has no hidden states to pool (the R1 stop needs them)".into(),
        ))
    }
    /// The R1 stop head (`rstop_head`) on its 523-wide inputs → one logit each.
    fn rstop_logits(&mut self, inputs: &[RstopInput]) -> Result<Vec<f32>, HfError> {
        let _ = inputs;
        Err(HfError::Invalid("this scorer has no rstop_head".into()))
    }
}

/// The widths of the R1 stop's own row: `stop_row`'s eight values, then
/// `expansions / 80`, `|E| / 160` and `|frontier| / 160` (`R1_HEAD_DESIGN.md`
/// §4 item 2), which do not saturate before the cap.
pub const RSTOP_ROW_DIM: usize = STOP_DIM + 3;
/// The return item's extras per candidate: `is_expanded`, `ordinal / 80`,
/// `cos(v, q)`, `rank_E` (§3).
pub const RETURN_EXTRAS: usize = 4;
/// The normalisers of the R1 counts: the cap, and twice it.
pub const R1_EXPANSION_NORM: f32 = 80.0;
pub const R1_EXAMINED_NORM: f32 = 160.0;

/// `rstop_row`: `stop_row` unchanged, then the three unsaturated counts.
pub fn rstop_row(
    stop: &[f32; STOP_DIM],
    expansions: usize,
    examined_set: usize,
    frontier_size: usize,
) -> [f32; RSTOP_ROW_DIM] {
    let mut row = [0f32; RSTOP_ROW_DIM];
    row[..STOP_DIM].copy_from_slice(stop);
    row[STOP_DIM] = expansions as f32 / R1_EXPANSION_NORM;
    row[STOP_DIM + 1] = examined_set as f32 / R1_EXAMINED_NORM;
    row[STOP_DIM + 2] = frontier_size as f32 / R1_EXAMINED_NORM;
    row
}

/// The R1 stop head's input at one snapshot decision: the row and the
/// decision's pooled hidden states (`2H`), detached. `t` is the expansion
/// count the decision was made at (before expansion `t + 1`).
#[derive(Clone, Debug, PartialEq)]
pub struct RstopInput {
    pub t: usize,
    pub row: [f32; RSTOP_ROW_DIM],
    pub pooled: Vec<f32>,
}

impl RstopInput {
    /// The flat `RSTOP_ROW_DIM + 2H` input, as the head reads it.
    pub fn flat(&self) -> Vec<f32> {
        let mut v = self.row.to_vec();
        v.extend_from_slice(&self.pooled);
        v
    }
}

/// A return item (`R1_HEAD_DESIGN.md` §3) at one snapshot: the candidates
/// `E_t ∖ {s}` — the expanded nodes in expansion order, then the frontier in
/// insertion order — their decision item built by the walk's own feature set,
/// and the four extras per candidate. `t` is the snapshot asked for; `t_eff`
/// the expansion count it was taken at (smaller when the walk ended first).
#[derive(Clone, Debug)]
pub struct ReturnSnapshot {
    pub t: usize,
    pub t_eff: usize,
    pub candidates: Vec<Local>,
    pub item: DecisionItem,
    pub extras: Vec<f32>,
}

/// What an R1 walk records beside its `WalkResult`.
#[derive(Clone, Debug, Default)]
pub struct R1Trace {
    /// One per requested snapshot, in the order requested.
    pub snapshots: Vec<ReturnSnapshot>,
    /// The stop inputs at the requested decisions that were made (a walk
    /// that ended first has none at the later ones).
    pub rstop: Vec<RstopInput>,
    /// `(t, logit)` wherever `StopRule::LearnedR1` consulted `rstop_head`.
    pub rstop_logits: Vec<(usize, f32)>,
}

/// What an R1 walk records: return items at `return_at`, stop inputs at the
/// decisions made at `rstop_at` expansions.
#[derive(Clone, Debug, Default)]
pub struct R1Options {
    pub return_at: Vec<usize>,
    pub rstop_at: Vec<usize>,
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

/// `Exhaust` stops on registration or an empty frontier; `Learned` consults
/// the stage-0 `stop_head` after every decision. `LearnedR1` is the R1 stop
/// (`R1_HEAD_DESIGN.md` §4, ENG-3): it consults `rstop_head`, never
/// `stop_head`, and ONLY at the decisions made at the `snapshots` expansion
/// counts, stopping on `σ(logit) > theta`; otherwise it is `Exhaust`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StopRule {
    Exhaust,
    Learned,
    LearnedR1 { theta: f64, snapshots: [usize; 4] },
}

/// The R1 stop's snapshots: before expansions 17, 33, 49 and 65.
pub const R1_STOP_SNAPSHOTS: [usize; 4] = [16, 32, 48, 64];

impl StopRule {
    pub fn parse(s: &str) -> Result<Self, HfError> {
        match s {
            "exhaust" => Ok(Self::Exhaust),
            "learned" => Ok(Self::Learned),
            other => Err(HfError::Invalid(format!("unknown stop rule: {other}"))),
        }
    }

    /// `σ(logit) > theta`, in f64 over the f32 logit — the reader's rule.
    pub fn r1_fires(logit: f32, theta: f64) -> bool {
        1.0 / (1.0 + (-(logit as f64)).exp()) > theta
    }
}

/// The walk's result for one episode (`model.walk`'s dict).
#[derive(Clone, Debug)]
pub struct WalkResult {
    pub expanded: Vec<Local>,
    pub decisions: Vec<Decision>,
    /// The expansion count at which **every** shown target was registered.
    pub registered_at: Option<usize>,
    /// Per shown target, the expansion count at which it was registered.
    pub registered_at_by_target: Vec<Option<usize>>,
    /// Per registration target, its cosine rank at the instant it registered
    /// (`State::cosine_rank_of`); `None` where the target never registered.
    /// Nothing in the walk reads it — it is a diagnostic the rows carry.
    pub cosine_rank_at_registration: Vec<Option<usize>>,
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

    /// ENG-9's per-episode reading: the ONE target's cosine rank at the
    /// instant it registered. `None` when it never registered, and `None` at
    /// k >= 2, where the rank is per target and no single number is the
    /// episode's — the column is a k = 1 reading (and `episode_query`, the
    /// source that makes it mean anything, is k = 1 only).
    pub fn cosine_rank_at_registration_k1(&self) -> Option<usize> {
        match self.cosine_rank_at_registration.as_slice() {
            [one] => *one,
            _ => None,
        }
    }

    /// How many of the shown targets registered, at the walk's own stop.
    pub fn registered_targets(&self) -> usize {
        self.registered_at_by_target
            .iter()
            .filter(|r| r.is_some())
            .count()
    }

    /// `K_TARGETS_DESIGN.md` §4's recall over k at a fixed budget: the share
    /// of targets registered within `budget` expansions. `None` when the
    /// record shows no target.
    pub fn recall_at_budget(&self, budget: usize) -> Option<f64> {
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
}

struct State<'a> {
    index: &'a EpisodeIndex,
    expanded: Vec<Local>,
    seen: HashSet<Local>,
    /// The discovery parent of every expanded node (the start has none).
    parent_of: HashMap<Local, Local>,
    /// Each expanded node's frontier entry as it was pushed (parent, depth,
    /// path mean), in expansion order, the start excluded: the return item's
    /// rows for the expanded candidates.
    expanded_entries: Vec<Entry>,
    frontier: Vec<Entry>,
    /// One flag per **registration** target (`K_TARGETS_DESIGN.md` §2: at
    /// stage 0 the mask comes from the visible side and the walk's own
    /// examination history; under `episode_query` nothing visible names the
    /// target, so it comes from the hidden payload and stays HERE, outside
    /// `VisibleIndex`, where no feature builder can reach it).
    registered: Vec<bool>,
    /// The expansion count at which each shown target registered.
    registered_at_by_target: Vec<Option<usize>>,
    /// Each registration target's cosine rank at the instant it registered.
    cosine_rank_at_registration: Vec<Option<usize>>,
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
            expanded_entries: Vec::new(),
            frontier: Vec::new(),
            registered: vec![false; index.registration_targets().len()],
            registered_at_by_target: vec![None; index.registration_targets().len()],
            cosine_rank_at_registration: vec![None; index.registration_targets().len()],
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

    /// Completion is **all k registered** — and a record that shows no target
    /// is never complete, so an empty mask is false rather than vacuously true.
    fn all_registered(&self) -> bool {
        !self.registered.is_empty() && self.registered.iter().all(|r| *r)
    }

    /// The reduction's mask, one flag per QUERY: `true` where the query's
    /// target is still unregistered. At stage 0 there is one query per shown
    /// target and this is `!registered` entry for entry — the vector the walk
    /// has always handed the view. Under `episode_query` the one query has the
    /// one hidden target's flag; a hand-built index with a query and no
    /// registration target leaves it unregistered, which is what a walk that
    /// can never complete means.
    fn unregistered(&self) -> Vec<bool> {
        let mut mask = vec![true; self.index.query_count()];
        for (t, r) in self.registered.iter().enumerate() {
            if let Some(flag) = mask.get_mut(t) {
                *flag = !*r;
            }
        }
        mask
    }

    /// ENG-9: where the `t`-th registration target stands, by cosine to ITS
    /// query, among the nodes the walk has seen at this instant — the start it
    /// began from, every node it has examined as a child, and the target
    /// itself. `1` means nothing the walk had seen scored higher than the
    /// target, so cosine alone would have pointed straight at it.
    ///
    /// Ties go to the target: the count is of nodes scoring **strictly**
    /// higher, so the rank never depends on the `edge_id` order in which the
    /// children of one node happen to be examined.
    ///
    /// Under `QuerySource::TargetEmbedding` the query IS the target's own
    /// embedding row, so its cosine is the maximum by construction and the
    /// rank is 1 for every episode that registers. The number only means
    /// something under `episode_query`, where the query is the question.
    fn cosine_rank_of(&self, target: Local, t: usize) -> Option<usize> {
        let index = self.index;
        if t >= index.query_count() {
            return None;
        }
        let q = index.query_of(t);
        let theirs = cosine(index.emb(target), q);
        Some(
            1 + self
                .seen
                .iter()
                .filter(|n| **n != target && cosine(index.emb(**n), q) > theirs)
                .count(),
        )
    }

    fn push_children(&mut self, node: Local, depth: u32, path_mean: &[f32]) {
        let index = self.index;
        for &child in &index.out[node as usize] {
            self.examined += 1;
            if let Some(t) = index
                .registration_targets()
                .iter()
                .position(|t| *t == child)
            {
                if !self.registered[t] {
                    self.registered[t] = true;
                    self.registered_at_by_target[t] = Some(self.expanded.len());
                    // read BEFORE `child` joins `seen`, so the comparison set
                    // is exactly what the walk had seen when the target came
                    // into view, the target included
                    self.cosine_rank_at_registration[t] = self.cosine_rank_of(child, t);
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
        if self.registered_at.is_some() && !self.index.registration_targets().is_empty() {
            self.stop_reason = "target_registered";
            self.live = false;
        }
        if self.frontier.is_empty() {
            self.live = false;
        }
    }

    /// The return item of this instant (`R1_HEAD_DESIGN.md` §3).
    fn return_snapshot(&self, features: &dyn FeatureSet, t: usize) -> ReturnSnapshot {
        let mut entries: Vec<Entry> = self.expanded_entries.clone();
        entries.extend(self.frontier.iter().cloned());
        let mask = self.unregistered();
        let visible = VisibleIndex::new(self.index, &mask);
        let item = features.build(visible, &entries, &self.expanded, &self.parent_of);
        let n = entries.len();
        let cosines: Vec<f32> = entries
            .iter()
            .map(|e| visible.cos_query_max(self.index.emb(e.node)))
            .collect();
        // rank within E ∖ {s}, 0 = highest cosine, normalised: the builder's
        // own rank rule (a stable sort on the cosine, descending)
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|a, b| {
            cosines[*b]
                .partial_cmp(&cosines[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut rank = vec![0f32; n];
        for (r, i) in order.iter().enumerate() {
            rank[*i] = if n > 1 {
                r as f32 / (n - 1) as f32
            } else {
                0.0
            };
        }
        let expanded_count = self.expanded_entries.len();
        let mut extras = Vec::with_capacity(n * RETURN_EXTRAS);
        for (i, _) in entries.iter().enumerate() {
            let is_expanded = i < expanded_count;
            // the start is ordinal 0 and never a candidate, so an expanded
            // candidate's ordinal is its position in `expanded`
            let ordinal = if is_expanded { i + 1 } else { 0 };
            extras.extend_from_slice(&[
                if is_expanded { 1.0 } else { 0.0 },
                ordinal as f32 / R1_EXPANSION_NORM,
                cosines[i],
                rank[i],
            ]);
        }
        ReturnSnapshot {
            t,
            t_eff: self.expanded.len(),
            candidates: entries.iter().map(|e| e.node).collect(),
            item,
            extras,
        }
    }

    fn finish(self, with_prior: bool) -> WalkResult {
        WalkResult {
            expanded: self.expanded,
            decisions: self.decisions,
            registered_at: self.registered_at,
            registered_at_by_target: self.registered_at_by_target,
            cosine_rank_at_registration: self.cosine_rank_at_registration,
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
    walk_batch_capped(indexes, features, scorer, options, None)
}

/// [`walk_batch`] with an expansion cap: a walk whose expansion count (the
/// start counted, as `expanded` counts it) reaches `max_expansions` stops with
/// `stop_reason = "expansion_cap"` before its next decision. `None` is
/// `walk_batch` exactly. Nothing about a decision changes, so a capped walk's
/// `expanded` is a prefix of the uncapped walk's.
pub fn walk_batch_capped(
    indexes: &[&EpisodeIndex],
    features: &dyn FeatureSet,
    scorer: &mut dyn Scorer,
    options: WalkOptions,
    max_expansions: Option<usize>,
) -> Result<Vec<WalkResult>, HfError> {
    Ok(walk_batch_r1(
        indexes,
        features,
        scorer,
        options,
        max_expansions,
        &R1Options::default(),
    )?
    .into_iter()
    .map(|(w, _)| w)
    .collect())
}

/// [`walk_batch_capped`] that also records the R1 head's inputs
/// (`R1_HEAD_DESIGN.md` §8 ENG-3): a return item at each expansion count in
/// `r1.return_at` (and, for a count the walk never reaches, at its end), and
/// the stop head's input at the decisions made at `r1.rstop_at` expansions.
/// With an empty `r1` and any stop rule but `LearnedR1` it IS
/// `walk_batch_capped`: the scorer is called exactly as before.
pub fn walk_batch_r1(
    indexes: &[&EpisodeIndex],
    features: &dyn FeatureSet,
    scorer: &mut dyn Scorer,
    options: WalkOptions,
    max_expansions: Option<usize>,
    r1: &R1Options,
) -> Result<Vec<(WalkResult, R1Trace)>, HfError> {
    let r1_active = !r1.return_at.is_empty()
        || !r1.rstop_at.is_empty()
        || matches!(options.stop_rule, StopRule::LearnedR1 { .. });
    if r1_active && features.embeds_query_vector() {
        return Err(HfError::Refused(format!(
            "the R1 head reads a relational feature set; {:?} copies raw coordinates",
            features.name()
        )));
    }
    if let StopRule::LearnedR1 { theta, .. } = options.stop_rule {
        if !(0.0..1.0).contains(&theta) {
            return Err(HfError::Invalid(format!(
                "the R1 stop threshold is a probability in [0, 1), not {theta}"
            )));
        }
    }
    if max_expansions == Some(0) {
        return Err(HfError::Invalid(
            "an expansion cap counts the start, so it is at least 1".into(),
        ));
    }
    // A feature set that forms one query from one target cannot read a k >= 2
    // episode: the refusal of `K_TARGETS_DESIGN.md` §6 item 4, moved from the
    // index (which now builds every record) to the one place that pairs a
    // record with a feature set.
    if !features.k_aware() {
        if let Some(i) = indexes.iter().find(|i| i.target_count() > 1) {
            return Err(HfError::BandH(format!(
                "{}: the record shows {} targets; the single-target feature set {:?} \
                 cannot read a k >= 2 episode (K_TARGETS_DESIGN.md §6 item 4)",
                i.episode_id,
                i.target_count(),
                features.name()
            )));
        }
    }
    // A set that copies the query vector into the rows would put the raw
    // QUESTION into every candidate row and into the query token, which is not
    // the one variable stage 1 changes; refused here, where a record meets a
    // feature set, as the k refusal above is.
    if features.embeds_query_vector() {
        if let Some(i) = indexes
            .iter()
            .find(|i| i.query_source == QuerySource::EpisodeQuery)
        {
            return Err(HfError::Refused(format!(
                "{}: the feature set {:?} copies the raw query vector into every candidate \
                 row and into the query token; under query_source episode_query that vector \
                 is the question itself, so it is refused",
                i.episode_id,
                features.name()
            )));
        }
    }
    let mut states: Vec<State> = indexes
        .iter()
        .map(|i| State::new(i, options.record_candidates))
        .collect();
    let mut traces: Vec<R1Trace> = indexes.iter().map(|_| R1Trace::default()).collect();
    let stop_snapshots: &[usize] = match &options.stop_rule {
        StopRule::LearnedR1 { snapshots, .. } => snapshots,
        _ => &[],
    };
    // a snapshot is taken when the expansion count reaches a requested t
    let snapshot_now = |s: &State, trace: &mut R1Trace| {
        let t = s.expanded.len();
        if r1.return_at.contains(&t) && !trace.snapshots.iter().any(|x| x.t == t) {
            trace.snapshots.push(s.return_snapshot(features, t));
        }
    };
    for (s, trace) in states.iter().zip(traces.iter_mut()) {
        snapshot_now(s, trace);
    }
    let cap_reached = |s: &mut State| {
        if s.live && max_expansions.is_some_and(|cap| s.expanded.len() >= cap) {
            s.stop_reason = "expansion_cap";
            s.live = false;
        }
    };
    for s in states.iter_mut() {
        cap_reached(s);
    }
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
                    let mask = s.unregistered();
                    let visible = VisibleIndex::new(s.index, &mask);
                    let item = features.build(visible, &s.frontier, &s.expanded, &s.parent_of);
                    // the greedy prior's column, the stop row's best cosine and
                    // the dump's cosines all read the SAME reduction the row does
                    let cosines: Vec<f32> = s
                        .frontier
                        .iter()
                        .map(|e| visible.cos_query_max(s.index.emb(e.node)))
                        .collect();
                    (item, cosines)
                })
                .collect()
        };
        let batch = DecisionBatch {
            items: items.iter().map(|(it, _)| it.clone()).collect(),
        };
        // the pooled hidden states are asked for only at a decision the R1
        // stop reads; every other call is the plain `score` it always was
        let wants_pooled = |i: usize| {
            let t = states[i].expanded.len();
            r1.rstop_at.contains(&t) || stop_snapshots.contains(&t)
        };
        let (scored, pooled) = if live.iter().any(|&i| wants_pooled(i)) {
            let (scored, pooled) = scorer.score_pooled(&batch)?;
            if pooled.len() != live.len() {
                return Err(HfError::BandH(format!(
                    "the scorer pooled {} rows for {} decisions",
                    pooled.len(),
                    live.len()
                )));
            }
            (scored, Some(pooled))
        } else {
            (scorer.score(&batch)?, None)
        };
        if scored.scores.len() != live.len() {
            return Err(HfError::BandH(format!(
                "the scorer returned {} rows for {} decisions",
                scored.scores.len(),
                live.len()
            )));
        }
        let mut stop_rows: Vec<(usize, [f32; STOP_DIM])> = Vec::new();
        let mut r1_rows: Vec<(usize, RstopInput)> = Vec::new();
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
            if let Some(pooled) = &pooled {
                let t = s.expanded.len();
                if r1.rstop_at.contains(&t) || stop_snapshots.contains(&t) {
                    let input = RstopInput {
                        t,
                        row: rstop_row(&stop_features, t, s.seen.len(), n),
                        pooled: pooled[k].clone(),
                    };
                    if stop_snapshots.contains(&t) {
                        r1_rows.push((i, input.clone()));
                    }
                    if r1.rstop_at.contains(&t) {
                        traces[i].rstop.push(input);
                    }
                }
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
        // the R1 stop consults rstop_head, at its snapshots only
        let mut r1_stopped: HashSet<usize> = HashSet::new();
        if let StopRule::LearnedR1 { theta, .. } = options.stop_rule {
            if !r1_rows.is_empty() {
                let inputs: Vec<RstopInput> = r1_rows.iter().map(|(_, r)| r.clone()).collect();
                let logits = scorer.rstop_logits(&inputs)?;
                if logits.len() != inputs.len() {
                    return Err(HfError::BandH(format!(
                        "the scorer returned {} rstop logits for {} inputs",
                        logits.len(),
                        inputs.len()
                    )));
                }
                for ((i, input), logit) in r1_rows.iter().zip(logits) {
                    traces[*i].rstop_logits.push((input.t, logit));
                    if StopRule::r1_fires(logit, theta) {
                        r1_stopped.insert(*i);
                    }
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
            if r1_stopped.contains(&i) {
                s.stop_reason = "learned_r1_stop";
                s.live = false;
                continue;
            }
            let chosen = s.decisions.last().unwrap().chosen;
            let entry = s.frontier.remove(chosen);
            let (node, parent, depth) = (entry.node, entry.parent, entry.depth);
            s.parent_of.insert(node, parent);
            s.expanded.push(node);
            s.push_children(node, depth, &entry.path_mean);
            if r1_active {
                s.expanded_entries.push(entry);
            }
            s.check_registered();
            snapshot_now(s, &mut traces[i]);
            cap_reached(s);
        }
    }
    // a requested snapshot the walk never reached is taken at its end
    for (s, trace) in states.iter().zip(traces.iter_mut()) {
        for &t in &r1.return_at {
            if !trace.snapshots.iter().any(|x| x.t == t) {
                trace.snapshots.push(s.return_snapshot(features, t));
            }
        }
        let order = |t: usize| r1.return_at.iter().position(|x| *x == t);
        trace.snapshots.sort_by_key(|x| order(x.t));
    }
    Ok(states
        .into_iter()
        .zip(traces)
        .map(|(s, trace)| (s.finish(options.with_prior), trace))
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
