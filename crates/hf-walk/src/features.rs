//! Feature sets: what a decision's candidates, context and query look like to
//! the scorer. `RawV5` is the Python candidate row; `RelationalV6` carries no
//! raw coordinate anywhere; `RelationalV6Prev` is `RelationalV6` with the
//! selective-previous-nodes channels added to the context token and the pair
//! channel. A builder sees one episode only through `VisibleIndex`, so the
//! payload the sampler kept back is not reachable here by type — the builders
//! take that view and the walk's own state.

use std::collections::HashMap;

use crate::{cosine, DecisionItem, Entry, Local, VisibleIndex};

pub const STRUCTURE_DIM: usize = 7;
pub const STOP_DIM: usize = 8;
const SCALE: f32 = 32.0;
/// The window of the previous-node recency feature, in expansions.
pub const RECENCY_WINDOW: usize = 16;

/// The seven structure features, in the Python order.
pub fn structure_features(
    index: VisibleIndex<'_>,
    e: &Entry,
    frontier_size: usize,
    cos_cq: f32,
    cos_pq: f32,
    cos_cp: f32,
) -> [f32; STRUCTURE_DIM] {
    [
        cos_cq,
        cos_pq,
        cos_cp,
        ((index.degree(e.parent) as f32).ln_1p()) / SCALE.ln(),
        (e.depth.min(16) as f32) / 16.0,
        (frontier_size.min(64) as f32) / 64.0,
        if index.degree(e.node) > 0 { 1.0 } else { 0.0 },
    ]
}

/// `stop_row`.
pub fn stop_row(
    expansions: usize,
    frontier_size: usize,
    best: f32,
    mean: f32,
    last_depth: u32,
    best_cosine: f32,
    examined: usize,
) -> [f32; STOP_DIM] {
    [
        (expansions.min(64) as f32) / 64.0,
        (frontier_size.min(64) as f32) / 64.0,
        1.0 / (1.0 + (-best).exp()),
        1.0 / (1.0 + (-mean).exp()),
        (last_depth.min(16) as f32) / 16.0,
        (best_cosine + 1.0) / 2.0,
        (examined.min(128) as f32) / 128.0,
        if frontier_size == 0 { 1.0 } else { 0.0 },
    ]
}

pub trait FeatureSet: Sync {
    fn name(&self) -> &'static str;
    fn candidate_dim(&self, edim: usize) -> usize;
    fn context_dim(&self, edim: usize) -> usize;
    /// Width of the query token, or `None` when the set has no query token
    /// (the model supplies a learned constant).
    fn query_dim(&self, edim: usize) -> Option<usize>;
    fn pair_dim(&self) -> usize;
    /// The column of the candidate row carrying `cos(c, q)` — the greedy prior's input.
    fn cosine_column(&self, edim: usize) -> usize;
    /// Build one decision's tensors from the episode's visible view, the
    /// frontier, the expansion order and the discovery parents.
    fn build(
        &self,
        index: VisibleIndex<'_>,
        frontier: &[Entry],
        expanded: &[Local],
        parent_of: &HashMap<Local, Local>,
    ) -> DecisionItem;
}

fn ancestors(entry: &Entry, parent_of: &HashMap<Local, Local>) -> Vec<Local> {
    // the candidate's discovery chain: parent, grandparent, … up to the start
    let mut chain = vec![entry.parent];
    let mut node = entry.parent;
    while let Some(p) = parent_of.get(&node) {
        chain.push(*p);
        node = *p;
        if chain.len() > 4096 {
            break;
        }
    }
    chain
}

/// The Python v5 layout: `[c | q | path_mean | parent | 7 structure]`, raw
/// context and query embeddings, pair `[cos(c, x), is-parent]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct RawV5;

impl FeatureSet for RawV5 {
    fn name(&self) -> &'static str {
        "raw-v5"
    }
    fn candidate_dim(&self, edim: usize) -> usize {
        4 * edim + STRUCTURE_DIM
    }
    fn context_dim(&self, edim: usize) -> usize {
        edim
    }
    fn query_dim(&self, edim: usize) -> Option<usize> {
        Some(edim)
    }
    fn pair_dim(&self) -> usize {
        2
    }
    fn cosine_column(&self, edim: usize) -> usize {
        4 * edim
    }
    fn build(
        &self,
        index: VisibleIndex<'_>,
        frontier: &[Entry],
        expanded: &[Local],
        _parent_of: &HashMap<Local, Local>,
    ) -> DecisionItem {
        let edim = index.edim();
        let query = index.query();
        let cdim = self.candidate_dim(edim);
        let mut cand = Vec::with_capacity(frontier.len() * cdim);
        for e in frontier {
            let c = index.emb(e.node);
            let p = index.emb(e.parent);
            cand.extend_from_slice(c);
            cand.extend_from_slice(query);
            cand.extend_from_slice(&e.path_mean);
            cand.extend_from_slice(p);
            cand.extend_from_slice(&structure_features(
                index,
                e,
                frontier.len(),
                cosine(c, query),
                cosine(p, query),
                cosine(c, p),
            ));
        }
        let mut ctx = Vec::with_capacity(expanded.len() * edim);
        for &x in expanded {
            ctx.extend_from_slice(index.emb(x));
        }
        let mut pair = Vec::with_capacity(frontier.len() * expanded.len() * 2);
        for e in frontier {
            let uc = index.unit(e.node);
            for &x in expanded {
                let dot: f32 = uc.iter().zip(index.unit(x)).map(|(a, b)| a * b).sum();
                pair.push(dot);
                pair.push(if e.parent == x { 1.0 } else { 0.0 });
            }
        }
        DecisionItem {
            frontier_len: frontier.len(),
            context_len: expanded.len(),
            cand,
            ctx,
            query: query.to_vec(),
            pair,
        }
    }
}

/// The redesign: every channel relational or structural.
///
/// Candidate row (16): `cos(c,q) cos(p,q) cos(c,p) cos(c,m) cos(m,q) cos(c,s)
/// rank z max_cos_expanded` plus the seven structure features (whose first
/// three repeat the cosines, kept so the structure block is the same seven in
/// both sets). Context token (4): `cos(x,q) cos(x,m_x) depth/16 log-degree`.
/// No query token. Pair (3): `cos(c,x) is-parent is-ancestor`, ancestry being
/// the candidate's own discovery chain through the expanded nodes.
#[derive(Clone, Copy, Debug, Default)]
pub struct RelationalV6;

pub const RELATIONAL_CANDIDATE_DIM: usize = 9 + STRUCTURE_DIM;
pub const RELATIONAL_CONTEXT_DIM: usize = 4;
pub const RELATIONAL_PAIR_DIM: usize = 3;

impl FeatureSet for RelationalV6 {
    fn name(&self) -> &'static str {
        "relational-v6"
    }
    fn candidate_dim(&self, _edim: usize) -> usize {
        RELATIONAL_CANDIDATE_DIM
    }
    fn context_dim(&self, _edim: usize) -> usize {
        RELATIONAL_CONTEXT_DIM
    }
    fn query_dim(&self, _edim: usize) -> Option<usize> {
        None
    }
    fn pair_dim(&self) -> usize {
        RELATIONAL_PAIR_DIM
    }
    fn cosine_column(&self, _edim: usize) -> usize {
        0
    }
    fn build(
        &self,
        index: VisibleIndex<'_>,
        frontier: &[Entry],
        expanded: &[Local],
        parent_of: &HashMap<Local, Local>,
    ) -> DecisionItem {
        relational_build(index, frontier, expanded, parent_of, None)
    }
}

/// `relational-v6` plus the selective-previous-nodes channels: the context
/// token and the pair channel each carry, for the expanded node `x`,
/// `was_abandoned(x)` and `recency(x)`. The candidate row is unchanged.
///
/// Context token (6): `cos(x,q) cos(x,m_x) ordinal/16 log-degree
/// was_abandoned(x) recency(x)`. Pair (5): `cos(c,x) is-parent is-ancestor
/// was_abandoned(x) recency(x)`.
///
/// `was_abandoned(x)` is 1 when `x` was expanded, none of `x`'s children has
/// been expanded since `x`, and at least one node that is not a child of `x`
/// was expanded after `x`. `recency(x)` is `min(k_now − k_x, 16) / 16`, where
/// `k_x` is `x`'s expansion ordinal and `k_now` is the number of expansions
/// made so far — so the newest expansion scores `1/16` and anything sixteen or
/// more expansions old scores 1. Both come from the walk's own state.
#[derive(Clone, Copy, Debug, Default)]
pub struct RelationalV6Prev;

pub const RELATIONAL_PREV_CONTEXT_DIM: usize = 6;
pub const RELATIONAL_PREV_PAIR_DIM: usize = 5;

impl FeatureSet for RelationalV6Prev {
    fn name(&self) -> &'static str {
        "relational-v6-prev"
    }
    fn candidate_dim(&self, _edim: usize) -> usize {
        RELATIONAL_CANDIDATE_DIM
    }
    fn context_dim(&self, _edim: usize) -> usize {
        RELATIONAL_PREV_CONTEXT_DIM
    }
    fn query_dim(&self, _edim: usize) -> Option<usize> {
        None
    }
    fn pair_dim(&self) -> usize {
        RELATIONAL_PREV_PAIR_DIM
    }
    fn cosine_column(&self, _edim: usize) -> usize {
        0
    }
    fn build(
        &self,
        index: VisibleIndex<'_>,
        frontier: &[Entry],
        expanded: &[Local],
        parent_of: &HashMap<Local, Local>,
    ) -> DecisionItem {
        let previous = PreviousNodes::of(expanded, parent_of);
        relational_build(index, frontier, expanded, parent_of, Some(&previous))
    }
}

/// `was_abandoned` and `recency` for every expanded node, in expansion order.
pub struct PreviousNodes {
    pub abandoned: Vec<f32>,
    pub recency: Vec<f32>,
}

impl PreviousNodes {
    /// From the expansion order and the discovery parents — nothing else.
    pub fn of(expanded: &[Local], parent_of: &HashMap<Local, Local>) -> Self {
        let k_now = expanded.len();
        let mut abandoned = Vec::with_capacity(k_now);
        let mut recency = Vec::with_capacity(k_now);
        for (k, x) in expanded.iter().enumerate() {
            let after = &expanded[k + 1..];
            // a child of x expanded since x; the discovery parent of an expanded
            // node is always expanded before it, so "since" is the whole suffix
            let child_since = after.iter().any(|y| parent_of.get(y) == Some(x));
            let other_after = after.iter().any(|y| parent_of.get(y) != Some(x));
            abandoned.push(if !child_since && other_after {
                1.0
            } else {
                0.0
            });
            recency.push(((k_now - k).min(RECENCY_WINDOW) as f32) / RECENCY_WINDOW as f32);
        }
        Self { abandoned, recency }
    }
}

/// The shared relational builder: `previous` is `None` for `relational-v6` and
/// carries the two previous-node columns for `relational-v6-prev`. The
/// candidate row is the same either way.
fn relational_build(
    index: VisibleIndex<'_>,
    frontier: &[Entry],
    expanded: &[Local],
    parent_of: &HashMap<Local, Local>,
    previous: Option<&PreviousNodes>,
) -> DecisionItem {
    let n = frontier.len();
    let edim = index.edim();
    let q = index.query();
    let pair_dim = match previous {
        Some(_) => RELATIONAL_PREV_PAIR_DIM,
        None => RELATIONAL_PAIR_DIM,
    };
    let ctx_dim = match previous {
        Some(_) => RELATIONAL_PREV_CONTEXT_DIM,
        None => RELATIONAL_CONTEXT_DIM,
    };
    let uq: Vec<f32> = {
        let norm = q.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        q.iter().map(|x| x / norm).collect()
    };
    let cos_cq: Vec<f32> = frontier
        .iter()
        .map(|e| cosine(index.emb(e.node), q))
        .collect();
    // rank (0 = highest cosine) normalised, and z-score within the frontier
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| {
        cos_cq[*b]
            .partial_cmp(&cos_cq[*a])
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
    let mean = cos_cq.iter().sum::<f32>() / n.max(1) as f32;
    let sd = (cos_cq.iter().map(|c| (c - mean) * (c - mean)).sum::<f32>() / n.max(1) as f32).sqrt();
    let mut cand = Vec::with_capacity(n * RELATIONAL_CANDIDATE_DIM);
    let mut pair = Vec::with_capacity(n * expanded.len() * pair_dim);
    for (i, e) in frontier.iter().enumerate() {
        let c = index.emb(e.node);
        let p = index.emb(e.parent);
        let uc = index.unit(e.node);
        // sibling mean: the other frontier nodes sharing this parent
        let mut sib = vec![0f32; edim];
        let mut count = 0;
        for f in frontier {
            if f.parent == e.parent && f.node != e.node {
                for (s, x) in sib.iter_mut().zip(index.emb(f.node)) {
                    *s += x;
                }
                count += 1;
            }
        }
        let cos_cs = if count > 0 { cosine(c, &sib) } else { 0.0 };
        let mut max_cos_expanded = -1.0f32;
        let ancestors = ancestors(e, parent_of);
        for (k, &x) in expanded.iter().enumerate() {
            let ux = index.unit(x);
            let dot: f32 = uc.iter().zip(ux).map(|(a, b)| a * b).sum();
            max_cos_expanded = max_cos_expanded.max(dot);
            pair.push(dot);
            pair.push(if e.parent == x { 1.0 } else { 0.0 });
            pair.push(if ancestors.contains(&x) { 1.0 } else { 0.0 });
            if let Some(prev) = previous {
                pair.push(prev.abandoned[k]);
                pair.push(prev.recency[k]);
            }
        }
        let cos_pq = cosine(p, q);
        let cos_cp = cosine(c, p);
        cand.extend_from_slice(&[
            cos_cq[i],
            cos_pq,
            cos_cp,
            cosine(c, &e.path_mean),
            cosine(&e.path_mean, q),
            cos_cs,
            rank[i],
            if sd > 0.0 {
                (cos_cq[i] - mean) / sd
            } else {
                0.0
            },
            if expanded.is_empty() {
                0.0
            } else {
                max_cos_expanded
            },
        ]);
        cand.extend_from_slice(&structure_features(index, e, n, cos_cq[i], cos_pq, cos_cp));
    }
    let mut ctx = Vec::with_capacity(expanded.len() * ctx_dim);
    for (k, &x) in expanded.iter().enumerate() {
        let ux = index.unit(x);
        let cos_xq: f32 = ux.iter().zip(&uq).map(|(a, b)| a * b).sum();
        // the running path mean is per frontier entry; for a context token use
        // the mean over the expansion order so far, a visible quantity
        let mut m = vec![0f32; edim];
        for &y in &expanded[..=k] {
            for (s, v) in m.iter_mut().zip(index.emb(y)) {
                *s += v;
            }
        }
        let cos_xm = cosine(index.emb(x), &m);
        ctx.extend_from_slice(&[
            cos_xq,
            cos_xm,
            (k.min(16) as f32) / 16.0,
            ((index.degree(x) as f32).ln_1p()) / SCALE.ln(),
        ]);
        if let Some(prev) = previous {
            ctx.extend_from_slice(&[prev.abandoned[k], prev.recency[k]]);
        }
    }
    DecisionItem {
        frontier_len: n,
        context_len: expanded.len(),
        cand,
        ctx,
        query: Vec::new(),
        pair,
    }
}
