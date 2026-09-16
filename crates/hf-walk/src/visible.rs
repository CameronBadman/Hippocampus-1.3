//! The visible view of one episode: what a feature builder is allowed to see.
//!
//! `VisibleIndex` borrows the visible arrays of an `EpisodeIndex` — the node
//! names, the adjacency in `edge_id` order, the raw and unit embeddings, the
//! query, the start and the shown target — and holds nothing else. It keeps
//! slices rather than a reference to the index, so a builder cannot reach the
//! private payload of an episode even from inside this crate, where a private
//! field of a parent module would otherwise be in scope. `FeatureSet::build`
//! takes this view and only this view.
//!
//! Every accessor here is a function of the visible split alone. Nothing that
//! the sampler kept back is reachable through it, by construction rather than
//! by convention: the fields below are the whole of its state.

use crate::{EpisodeIndex, Local};

/// The visible half of one indexed episode, borrowed.
#[derive(Clone, Copy, Debug)]
pub struct VisibleIndex<'a> {
    names: &'a [String],
    start: Local,
    target_shown: Option<Local>,
    out: &'a [Vec<Local>],
    edim: usize,
    emb: &'a [f32],
    unit: &'a [f32],
    query: &'a [f32],
}

impl<'a> VisibleIndex<'a> {
    /// Borrow the visible arrays of an indexed episode.
    pub fn new(index: &'a EpisodeIndex) -> Self {
        Self {
            names: &index.names,
            start: index.start,
            target_shown: index.target_shown,
            out: &index.out,
            edim: index.edim,
            emb: &index.emb,
            unit: &index.unit,
            query: &index.query,
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

    /// The shown target, when the episode has one.
    pub fn target_shown(&self) -> Option<Local> {
        self.target_shown
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

    /// The query vector (the shown target's embedding at stage 0).
    pub fn query(&self) -> &'a [f32] {
        self.query
    }
}
