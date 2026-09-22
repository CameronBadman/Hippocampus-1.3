//! The visible view of one episode: what a feature builder is allowed to see.
//!
//! `VisibleIndex` borrows the visible arrays of an `EpisodeIndex` — the node
//! names, the adjacency in `edge_id` order, the raw and unit embeddings, the
//! shown targets with their queries, the start — and the walk's own
//! registration mask, which is derived from the shown targets and the walk's
//! examination history and from nothing else. It keeps slices rather than a
//! reference to the index, so a builder cannot reach the private payload of an
//! episode even from inside this crate, where a private field of a parent
//! module would otherwise be in scope. `FeatureSet::build` takes this view and
//! only this view.
//!
//! Every accessor here is a function of the visible split and the walk's own
//! state alone. Nothing that the sampler kept back is reachable through it, by
//! construction rather than by convention: the fields below are the whole of
//! its state.
//!
//! The query reduction of `K_TARGETS_DESIGN.md` §2(d) lives here, so that the
//! feature builders and the walk's own cosine column read one implementation:
//! every `cos(·, q)` is the **maximum over the currently unregistered targets**
//! of the per-target cosine, and the candidate row also carries the minimum and
//! the unregistered share. At k = 1 the maximum over the one target is that
//! target's cosine, computed by the same `cosine` call on the same vector, so
//! the reduction is the identity.
//!
//! The reduction runs over the QUERIES, one flag of the mask each. At stage 0
//! there is one query per shown target and the two counts are equal, so this
//! is the reduction the walk has always made. Under `episode_query` there is
//! one query — the episode's own question vector — and no shown target at all:
//! `targets_shown()` is empty, `query_count()` is one, and the mask the walk
//! hands in is built from what the record kept back, outside this view.

use crate::{cosine, EpisodeIndex, Local};

/// The visible half of one indexed episode, borrowed, with the walk's
/// registration mask over the shown targets.
#[derive(Clone, Copy, Debug)]
pub struct VisibleIndex<'a> {
    names: &'a [String],
    start: Local,
    targets_shown: &'a [Local],
    out: &'a [Vec<Local>],
    edim: usize,
    emb: &'a [f32],
    unit: &'a [f32],
    queries: &'a [f32],
    unit_queries: &'a [f32],
    unregistered: &'a [bool],
}

impl<'a> VisibleIndex<'a> {
    /// Borrow the visible arrays of an indexed episode together with the
    /// walk's unregistered mask (one flag per query, `true` while that query's
    /// target has not yet been examined).
    pub fn new(index: &'a EpisodeIndex, unregistered: &'a [bool]) -> Self {
        debug_assert_eq!(unregistered.len(), index.query_count());
        Self {
            names: &index.names,
            start: index.start,
            targets_shown: &index.targets_shown,
            out: &index.out,
            edim: index.edim,
            emb: &index.emb,
            unit: &index.unit,
            queries: &index.queries,
            unit_queries: &index.unit_queries,
            unregistered,
        }
    }

    /// The node names, in local order.
    pub fn names(&self) -> &'a [String] {
        self.names
    }

    /// One node's name.
    pub fn name(&self, node: Local) -> &'a str {
        &self.names[node as usize]
    }

    /// How many nodes the subgraph has.
    pub fn node_count(&self) -> usize {
        self.names.len()
    }

    /// The start node.
    pub fn start(&self) -> Local {
        self.start
    }

    /// The first shown target, when the episode has one.
    pub fn target_shown(&self) -> Option<Local> {
        self.targets_shown.first().copied()
    }

    /// Every shown target, in record order.
    pub fn targets_shown(&self) -> &'a [Local] {
        self.targets_shown
    }

    /// How many targets the record shows (`k`); none under `episode_query`.
    pub fn target_count(&self) -> usize {
        self.targets_shown.len()
    }

    /// How many queries the row reads: `k` at stage 0, one under
    /// `episode_query`. This is what every query channel reduces over and the
    /// width of the mask.
    pub fn query_count(&self) -> usize {
        self.queries.len().checked_div(self.edim).unwrap_or(0)
    }

    /// The unregistered mask over the queries, in the same order.
    pub fn unregistered(&self) -> &'a [bool] {
        self.unregistered
    }

    /// `|unregistered| / k` — the candidate row's third extra column. One
    /// when the record carries no query at all.
    pub fn unregistered_share(&self) -> f32 {
        let k = self.query_count();
        if k == 0 {
            return 1.0;
        }
        self.unregistered.iter().filter(|u| **u).count() as f32 / k as f32
    }

    /// One node's out-neighbours, in `edge_id` order.
    pub fn out(&self, node: Local) -> &'a [Local] {
        &self.out[node as usize]
    }

    /// Out-degree.
    pub fn degree(&self, node: Local) -> usize {
        self.out[node as usize].len()
    }

    /// The embedding width.
    pub fn edim(&self) -> usize {
        self.edim
    }

    /// One node's raw embedding (zeros when the cache lacked the node).
    pub fn emb(&self, node: Local) -> &'a [f32] {
        &self.emb[node as usize * self.edim..(node as usize + 1) * self.edim]
    }

    /// One node's unit-normalised embedding.
    pub fn unit(&self, node: Local) -> &'a [f32] {
        &self.unit[node as usize * self.edim..(node as usize + 1) * self.edim]
    }

    /// The query vector: the first shown target's embedding at stage 0.
    pub fn query(&self) -> &'a [f32] {
        &self.queries[..self.edim]
    }

    /// The `t`-th shown target's query vector.
    pub fn query_of(&self, t: usize) -> &'a [f32] {
        &self.queries[t * self.edim..(t + 1) * self.edim]
    }

    /// The `t`-th shown target's unit-normalised query vector.
    pub fn unit_query_of(&self, t: usize) -> &'a [f32] {
        &self.unit_queries[t * self.edim..(t + 1) * self.edim]
    }

    /// The reduction's index set: the unregistered targets, or — when every
    /// target is registered, which completion makes unreachable at a decision
    /// — all of them, so no channel is ever a maximum over nothing.
    fn reduced_over(&self) -> impl Iterator<Item = usize> + '_ {
        let any = self.unregistered.iter().any(|u| *u);
        (0..self.query_count()).filter(move |t| !any || self.unregistered[*t])
    }

    /// `(min, max)` over the unregistered targets of `cosine(v, q_t)`.
    pub fn cos_query_min_max(&self, v: &[f32]) -> (f32, f32) {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for t in self.reduced_over() {
            let c = cosine(v, self.query_of(t));
            lo = lo.min(c);
            hi = hi.max(c);
        }
        if hi == f32::NEG_INFINITY {
            (0.0, 0.0)
        } else {
            (lo, hi)
        }
    }

    /// `max` over the unregistered targets of `cosine(v, q_t)`.
    pub fn cos_query_max(&self, v: &[f32]) -> f32 {
        self.cos_query_min_max(v).1
    }

    /// `max` over the unregistered targets of the **unit** dot product
    /// `unit(v) · unit(q_t)` — the context token's own arithmetic, which
    /// differs from `cosine` in its accumulation and so is reduced separately.
    pub fn unit_dot_query_max(&self, unit_v: &[f32]) -> f32 {
        let mut hi = f32::NEG_INFINITY;
        for t in self.reduced_over() {
            let d: f32 = unit_v
                .iter()
                .zip(self.unit_query_of(t))
                .map(|(a, b)| a * b)
                .sum();
            hi = hi.max(d);
        }
        if hi == f32::NEG_INFINITY {
            0.0
        } else {
            hi
        }
    }
}

/// `q / ||q||` with the builders' floor, in f32 — the context token's
/// normalisation, kept in one place so every reduction over it agrees.
pub fn unit_query(q: &[f32]) -> Vec<f32> {
    let norm = q.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    q.iter().map(|x| x / norm).collect()
}
