//! The R1 premise's delete-a-node records (`experiments/real_walk_v2/
//! R1_PREMISE_PLAN.md` §2 E1, §3): for one start of a spent split, delete a
//! node X of its ball, rebuild the ball around the same start on the graph
//! WITHOUT X, and label X's former neighbourhood in what survives.
//!
//! - The stored ball is rebuilt first (`ball(G, s, n, cap, 3, region)`) and
//!   must equal the source record's node list, order for order — the ball is
//!   deterministic, so a mismatch is band H, never a skipped start.
//! - `ball′` is `hf_graph::RealGraph::ball_traced` with X **excluded ahead of
//!   the hub cap**, which is the ball of `G ∖ X`; filtering X through `allow`
//!   is not (`hf-graph`'s tests show the difference).
//! - `T = (in_G(X) ∪ out_G(X)) ∩ ball′ ∖ {s}`, with direction and relation;
//!   `L = N_G(X) ∩ (ball ∖ ball′) ∖ {s}` — the neighbours that left with X.
//! - Every node of `ball′` but the start keeps its BFS parent inside `ball′`
//!   (asserted per record, band H otherwise): nothing in the rebuilt ball is
//!   an orphan of the deletion.
//! - The draw: per start, one X uniform over the candidates (`U`) and one per
//!   degree stratum on `|T|` (`D1`–`D4`), each by
//!   `hash_int([label, episode_id, tag]) % len(sorted candidates)`. A
//!   candidate is a node of the stored ball other than the start, with a
//!   vector in the cache and a non-empty `T`; the ones with an empty `T` are
//!   counted, not drawn. The label is declared by the caller ([`DrawSpec`]);
//!   the premise's was [`DRAW_LABEL`].
//! - [`PerStart::Rotate`] (the R1 head's TRAIN, `R1_HEAD_DESIGN.md` §8
//!   ENG-1) keeps exactly one record per start: tag `DRAW_TAGS[ordinal mod
//!   5]`, falling back to `U` when that tag has no candidate.
//! - A source id outside the caller's declared raw-index ranges, or of an
//!   undeclared split, is refused before anything is computed. The premise's
//!   constants (`>= 73,360` in `train`, `>= 4,105` in `screen` refused,
//!   Cameron 2026-09-25) are [`DrawSpec::premise`].
//!
//! Nothing here reads a hidden payload: the inputs are the graph, the source
//! record's VISIBLE start and node list, the sampler block and the cache.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use hf_core::HfError;
use hf_graph::{NodeId, RealGraph};
use serde::Serialize;
use serde_json::{json, Value};

use crate::{hash_int, Sampler};

/// The premise's hash domain (`R1_PREMISE_PLAN.md`); a draw now declares its
/// own ([`DrawSpec`]).
pub const DRAW_LABEL: &str = "r1-premise-2026-09-25";
/// Seed selection's reserved raw draw indices (Cameron, 2026-09-25).
pub const RESERVED_TRAIN_FROM: u64 = 73_360;
pub const RESERVED_SCREEN_FROM: u64 = 4_105;
/// The sampler's hub fan-out (`Sampler::sample_one` passes 3).
pub const HUB_FANOUT: usize = 3;
/// The degree strata on `|T|`, and the draw tags in draw order.
pub const STRATA: [(&str, usize, usize); 4] = [
    ("D1", 1, 1),
    ("D2", 2, 3),
    ("D3", 4, 7),
    ("D4", 8, usize::MAX),
];
pub const DRAW_TAGS: [&str; 5] = ["U", "D1", "D2", "D3", "D4"];

/// The stratum of a neighbourhood size (or of a degree, with the same cuts).
pub fn stratum(n: usize) -> Option<&'static str> {
    STRATA
        .iter()
        .find(|(_, lo, hi)| n >= *lo && n <= *hi)
        .map(|(name, _, _)| *name)
}

/// The split and raw draw index of a sampler episode id
/// (`<family>-<split>-<index:06>-<hash>…`); the family may contain dashes.
pub fn raw_index(episode_id: &str) -> Option<(&'static str, u64)> {
    for split in ["train", "screen"] {
        let marker = format!("-{split}-");
        if let Some(at) = episode_id.rfind(&marker) {
            let rest = &episode_id[at + marker.len()..];
            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if digits.len() >= 6 {
                if let Ok(index) = digits.parse::<u64>() {
                    return Some((if split == "train" { "train" } else { "screen" }, index));
                }
            }
        }
    }
    None
}

/// Refuse an id inside seed selection's reserve, or one whose index cannot be
/// read (so the reserve cannot be checked).
pub fn refuse_reserved(episode_id: &str) -> Result<(), HfError> {
    match raw_index(episode_id) {
        None => Err(HfError::Refused(format!(
            "{episode_id}: no raw draw index can be read from this id, so seed selection's \
             reserve cannot be checked"
        ))),
        Some(("train", i)) if i >= RESERVED_TRAIN_FROM => Err(HfError::Refused(format!(
            "{episode_id}: train raw index {i} is reserved for seed selection (>= {RESERVED_TRAIN_FROM})"
        ))),
        Some(("screen", i)) if i >= RESERVED_SCREEN_FROM => Err(HfError::Refused(format!(
            "{episode_id}: screen raw index {i} is reserved for seed selection (>= {RESERVED_SCREEN_FROM})"
        ))),
        Some(_) => Ok(()),
    }
}

/// One declared raw-index range of one split, BOTH ends inclusive; `hi:
/// None` is open above. Written `split:lo..hi` or `split:lo..`, so the
/// premise's `train:0..73359` admits 73,359 and refuses 73,360.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllowedRange {
    pub split: &'static str,
    pub lo: u64,
    pub hi: Option<u64>,
}

impl AllowedRange {
    pub fn parse(text: &str) -> Result<Self, HfError> {
        let bad = || {
            HfError::Invalid(format!(
                "--allowed-range {text:?}: expected <train|screen>:<lo>..<hi> or <train|screen>:<lo>.. \
                 (both ends inclusive)"
            ))
        };
        let (split, range) = text.split_once(':').ok_or_else(bad)?;
        let split = match split {
            "train" => "train",
            "screen" => "screen",
            _ => return Err(bad()),
        };
        let (lo, hi) = range.split_once("..").ok_or_else(bad)?;
        let lo: u64 = lo.parse().map_err(|_| bad())?;
        let hi: Option<u64> = if hi.is_empty() {
            None
        } else {
            Some(hi.parse().map_err(|_| bad())?)
        };
        if hi.is_some_and(|h| h < lo) {
            return Err(bad());
        }
        Ok(Self { split, lo, hi })
    }

    pub fn contains(&self, split: &str, index: u64) -> bool {
        split == self.split && index >= self.lo && self.hi.is_none_or(|h| index <= h)
    }

    /// The range as it is written on the command line and in a manifest.
    pub fn label(&self) -> String {
        match self.hi {
            Some(h) => format!("{}:{}..{}", self.split, self.lo, h),
            None => format!("{}:{}..", self.split, self.lo),
        }
    }
}

/// What a deletion draw declares: its hash label and the raw-index ranges its
/// source ids must fall in (`R1_HEAD_DESIGN.md` §8 ENG-1).
#[derive(Clone, Debug)]
pub struct DrawSpec {
    pub label: String,
    pub ranges: Vec<AllowedRange>,
}

impl DrawSpec {
    /// The premise's draw: its label and the complement of seed selection's
    /// reserve, `train:0..73359` and `screen:0..4104`.
    pub fn premise() -> Self {
        Self {
            label: DRAW_LABEL.to_string(),
            ranges: vec![
                AllowedRange {
                    split: "train",
                    lo: 0,
                    hi: Some(RESERVED_TRAIN_FROM - 1),
                },
                AllowedRange {
                    split: "screen",
                    lo: 0,
                    hi: Some(RESERVED_SCREEN_FROM - 1),
                },
            ],
        }
    }

    /// Refuse an id whose raw index cannot be read, whose split no range
    /// declares, or whose index lies outside every range of its split.
    pub fn check(&self, episode_id: &str) -> Result<(), HfError> {
        let Some((split, index)) = raw_index(episode_id) else {
            return Err(HfError::Refused(format!(
                "{episode_id}: no raw draw index can be read from this id, so the declared \
                 ranges cannot be checked"
            )));
        };
        if !self.ranges.iter().any(|r| r.split == split) {
            return Err(HfError::Refused(format!(
                "{episode_id}: split {split} is not declared by any --allowed-range"
            )));
        }
        if !self.ranges.iter().any(|r| r.contains(split, index)) {
            let declared: Vec<String> = self.ranges.iter().map(AllowedRange::label).collect();
            return Err(HfError::Refused(format!(
                "{episode_id}: {split} raw index {index} is outside the declared ranges {declared:?}"
            )));
        }
        Ok(())
    }
}

/// How many records a start yields: `All` is one per distinct pick over the
/// five tags (the premise); `Rotate` is exactly one, tag
/// `DRAW_TAGS[ordinal mod 5]`, falling back to `U` when that tag is empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PerStart {
    All,
    Rotate,
}

impl PerStart {
    pub fn parse(s: &str) -> Result<Self, HfError> {
        match s {
            "all" => Ok(Self::All),
            "rotate" => Ok(Self::Rotate),
            other => Err(HfError::Invalid(format!(
                "--per-start is all or rotate, not {other:?}"
            ))),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Rotate => "rotate",
        }
    }
}

/// The cap on a record's written whole-graph neighbour list (a hub's can run
/// to hundreds of thousands); past it the list is truncated and flagged.
pub const NEIGHBOURS_WRITTEN_CAP: usize = 50_000;

/// The reverse adjacency of the whole graph, distinct heads per tail, as CSR.
pub struct InIndex {
    offsets: Vec<u64>,
    heads: Vec<NodeId>,
}

impl InIndex {
    pub fn new(graph: &RealGraph) -> Self {
        let n = graph.node_count();
        let mut degree = vec![0u64; n + 1];
        for head in graph.nodes() {
            for tail in graph.out_neighbours(head) {
                degree[tail as usize + 1] += 1;
            }
        }
        for i in 0..n {
            degree[i + 1] += degree[i];
        }
        let mut heads = vec![0 as NodeId; degree[n] as usize];
        let mut fill = degree.clone();
        for head in graph.nodes() {
            for tail in graph.out_neighbours(head) {
                heads[fill[tail as usize] as usize] = head;
                fill[tail as usize] += 1;
            }
        }
        Self {
            offsets: degree,
            heads,
        }
    }

    /// The distinct heads of edges into `node`, in node-id (= name) order.
    pub fn heads(&self, node: NodeId) -> &[NodeId] {
        &self.heads[self.offsets[node as usize] as usize..self.offsets[node as usize + 1] as usize]
    }
}

/// X's whole-graph neighbourhood `N_G(X)` (in ∪ out, distinct, sorted by
/// name, X excluded), capped at [`NEIGHBOURS_WRITTEN_CAP`]; the bool says
/// whether the cap cut it.
pub fn graph_neighbours(graph: &RealGraph, in_index: &InIndex, x: NodeId) -> (Vec<String>, bool) {
    let mut all: BTreeSet<NodeId> = graph.out_neighbours(x).into_iter().collect();
    all.extend(in_index.heads(x).iter().copied());
    all.remove(&x);
    let truncated = all.len() > NEIGHBOURS_WRITTEN_CAP;
    let mut names: Vec<String> = all.iter().map(|n| graph.name(*n).to_string()).collect();
    names.sort();
    names.truncate(NEIGHBOURS_WRITTEN_CAP);
    (names, truncated)
}

/// One former neighbour of X still present in `ball′`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Neighbour {
    pub node: String,
    /// `"in"` (node → X), `"out"` (X → node) or `"both"`.
    pub direction: &'static str,
    /// Every relation of every such edge, sorted and distinct (empty when the
    /// graph is untyped).
    pub relations: Vec<String>,
}

/// One deletion: X, the rebuilt ball and the labels.
#[derive(Clone, Debug)]
pub struct Deletion {
    pub deleted: NodeId,
    /// `ball′` in its breadth-first order, the start first.
    pub ball: Vec<NodeId>,
    /// `T`, sorted by node name.
    pub targets: Vec<Neighbour>,
    /// Whether the start itself was a neighbour of X (then `T⁺ = T ∪ {s}`).
    pub start_is_neighbour: bool,
    /// `L`, sorted by node name.
    pub left: Vec<String>,
    /// Nodes of `ball′` absent from the stored ball, and the reverse.
    pub entered: usize,
    pub departed: usize,
}

/// X's in- and out-neighbours inside a node set, excluding one node.
fn neighbours_in(
    graph: &RealGraph,
    x: NodeId,
    within: &[NodeId],
    except: NodeId,
) -> BTreeMap<String, (bool, bool, BTreeSet<String>)> {
    let members: BTreeSet<NodeId> = within.iter().copied().collect();
    let rel = |r: u32| -> Option<String> {
        if graph.typed() {
            Some(graph.relation_name(r).to_string())
        } else {
            None
        }
    };
    let mut out: BTreeMap<String, (bool, bool, BTreeSet<String>)> = BTreeMap::new();
    for e in graph.out(x) {
        if e.tail != except && e.tail != x && members.contains(&e.tail) {
            let entry = out.entry(graph.name(e.tail).to_string()).or_insert((
                false,
                false,
                BTreeSet::new(),
            ));
            entry.1 = true;
            entry.2.extend(rel(e.relation));
        }
    }
    for &n in within {
        if n == except || n == x {
            continue;
        }
        for e in graph.out(n) {
            if e.tail == x {
                let entry =
                    out.entry(graph.name(n).to_string())
                        .or_insert((false, false, BTreeSet::new()));
                entry.0 = true;
                entry.2.extend(rel(e.relation));
            }
        }
    }
    out
}

/// Delete `x` from around `start`: rebuild the ball on `G ∖ x` and label it.
/// `stored` is the stored ball (with x), used only for `L` and the counts.
pub fn delete_node(
    graph: &RealGraph,
    sampler: &Sampler<'_>,
    split: &str,
    start: NodeId,
    stored: &[NodeId],
    x: NodeId,
) -> Result<Deletion, HfError> {
    let member = |n: NodeId| sampler.member(split, n);
    let region = sampler.config.screen_region > 0.0;
    let traced = graph.ball_traced(
        start,
        sampler.config.subgraph_size as usize,
        sampler.config.hub_degree_cap,
        HUB_FANOUT,
        if region { Some(&member) } else { None },
        Some(x),
    )?;
    let ball = traced.order;
    if ball.contains(&x) {
        return Err(HfError::BandH(format!(
            "{} is in the ball rebuilt without it",
            graph.name(x)
        )));
    }
    // the in-edge invariant: every node but the start keeps its BFS parent
    // inside the rebuilt ball, on an edge of G
    let in_ball: BTreeSet<NodeId> = ball.iter().copied().collect();
    for n in &ball[1..] {
        let p = traced.parent.get(n).copied();
        let ok = p
            .is_some_and(|p| in_ball.contains(&p) && p != x && graph.out_neighbours(p).contains(n));
        if !ok {
            return Err(HfError::BandH(format!(
                "{} has no in-edge from inside the ball rebuilt without {}",
                graph.name(*n),
                graph.name(x)
            )));
        }
    }
    let found = neighbours_in(graph, x, &ball, start);
    let targets: Vec<Neighbour> = found
        .into_iter()
        .map(|(node, (inward, outward, relations))| Neighbour {
            node,
            direction: match (inward, outward) {
                (true, true) => "both",
                (true, false) => "in",
                _ => "out",
            },
            relations: relations.into_iter().collect(),
        })
        .collect();
    let start_is_neighbour =
        graph.out_neighbours(x).contains(&start) || graph.out_neighbours(start).contains(&x);
    let gone: Vec<NodeId> = stored
        .iter()
        .copied()
        .filter(|n| !in_ball.contains(n) && *n != x)
        .collect();
    let left: Vec<String> = neighbours_in(graph, x, &gone, start).into_keys().collect();
    let stored_set: BTreeSet<NodeId> = stored.iter().copied().collect();
    Ok(Deletion {
        deleted: x,
        entered: ball.iter().filter(|n| !stored_set.contains(n)).count(),
        departed: gone.len(),
        ball,
        targets,
        start_is_neighbour,
        left,
    })
}

/// One drawn deletion of one start, with every draw tag that picked it.
#[derive(Clone, Debug)]
pub struct Drawn {
    pub draws: Vec<&'static str>,
    pub deletion: Deletion,
    /// X's discovery parent in the STORED ball (the locality read, §4).
    pub bfs_parent: Option<String>,
    /// `d(s, X)` in the stored ball's induced graph.
    pub d_ball: Option<u32>,
    /// `|out_G(X)| + |in_G(X)|`, distinct neighbours per direction.
    pub degree_graph: usize,
}

/// What one start yields.
#[derive(Clone, Debug)]
pub struct StartDraws {
    pub drawn: Vec<Drawn>,
    pub candidates: usize,
    pub without_neighbour: usize,
    pub without_vector: usize,
    /// candidates whose deletion leaves the start with no out-edge, so no
    /// ball can be rebuilt around it (the start's only child)
    pub start_isolated: usize,
    /// candidates per stratum, `U` excluded
    pub stratum_sizes: BTreeMap<&'static str, usize>,
    /// Stored-ball nodes other than the start that the cache lacks: the X
    /// candidates whose row is missing (listed for coverage, whether or not
    /// they were counted out of the pool).
    pub uncached_candidates: Vec<String>,
    /// Under [`PerStart::Rotate`], the rotated tag that had no candidate and
    /// fell back to `U`.
    pub fallback: Option<&'static str>,
}

/// The per-start options of [`draw_for_start`].
#[derive(Clone, Copy, Debug)]
pub struct StartOptions<'a> {
    pub spec: &'a DrawSpec,
    pub per_start: PerStart,
    /// The start's position in the source stream (the rotation's ordinal).
    pub ordinal: usize,
    /// Treat every candidate as if the cache held its row (`--coverage-only`:
    /// the draw a fully covered cache would make, so one embedding pass
    /// reaches the fixed point).
    pub assume_covered: bool,
}

/// The draw for one start of a spent split (§3.3). `stored_names` is the
/// source record's visible node list; `has_vector` says whether the cache
/// holds a node's row.
#[allow(clippy::too_many_arguments)]
pub fn draw_for_start(
    graph: &RealGraph,
    sampler: &Sampler<'_>,
    split: &str,
    episode_id: &str,
    start_name: &str,
    stored_names: &[String],
    has_vector: &(dyn Fn(&str) -> bool + Sync),
    in_index: &InIndex,
    options: StartOptions<'_>,
) -> Result<StartDraws, HfError> {
    options.spec.check(episode_id)?;
    let start = graph
        .id(start_name)
        .ok_or_else(|| HfError::BandH(format!("{episode_id}: start {start_name} not in graph")))?;
    let member = |n: NodeId| sampler.member(split, n);
    let region = sampler.config.screen_region > 0.0;
    let stored = graph.ball_traced(
        start,
        sampler.config.subgraph_size as usize,
        sampler.config.hub_degree_cap,
        HUB_FANOUT,
        if region { Some(&member) } else { None },
        None,
    )?;
    let rebuilt: Vec<&str> = stored.order.iter().map(|n| graph.name(*n)).collect();
    let want: Vec<&str> = stored_names.iter().map(String::as_str).collect();
    if rebuilt != want {
        return Err(HfError::BandH(format!(
            "{episode_id}: the rebuilt ball ({} nodes) is not the stored ball ({} nodes)",
            rebuilt.len(),
            want.len()
        )));
    }
    let sub = graph.induced(&stored.order);
    let dist = sub.distances_from(start);
    let mut names: Vec<(String, NodeId)> = stored.order[1..]
        .iter()
        .map(|n| (graph.name(*n).to_string(), *n))
        .collect();
    names.sort();
    let mut without_vector = 0usize;
    let mut without_neighbour = 0usize;
    let mut start_isolated = 0usize;
    let mut uncached_candidates = Vec::new();
    let start_tails = graph.out_neighbours(start);
    let mut pool: Vec<Deletion> = Vec::new();
    for (name, x) in &names {
        if !has_vector(name) {
            uncached_candidates.push(name.clone());
            if !options.assume_covered {
                without_vector += 1;
                continue;
            }
        }
        if start_tails.iter().all(|t| t == x) {
            start_isolated += 1;
            continue;
        }
        let d = delete_node(graph, sampler, split, start, &stored.order, *x)?;
        if d.targets.is_empty() {
            without_neighbour += 1;
            continue;
        }
        pool.push(d);
    }
    let mut stratum_sizes: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut chosen: BTreeMap<NodeId, (Vec<&'static str>, usize)> = BTreeMap::new();
    let members_of = |tag: &str| -> Vec<usize> {
        (0..pool.len())
            .filter(|i| tag == "U" || stratum(pool[*i].targets.len()) == Some(tag))
            .collect()
    };
    let label = options.spec.label.as_str();
    for tag in DRAW_TAGS {
        let members = members_of(tag);
        if tag != "U" {
            stratum_sizes.insert(tag, members.len());
        }
    }
    let mut fallback = None;
    let tags: Vec<&'static str> = match options.per_start {
        PerStart::All => DRAW_TAGS.to_vec(),
        PerStart::Rotate => {
            let tag = DRAW_TAGS[options.ordinal % DRAW_TAGS.len()];
            if tag != "U" && members_of(tag).is_empty() {
                fallback = Some(tag);
                vec!["U"]
            } else {
                vec![tag]
            }
        }
    };
    for tag in tags {
        let members = members_of(tag);
        if members.is_empty() {
            continue;
        }
        let pick = members[(hash_int(&[label, episode_id, tag]) % members.len() as u64) as usize];
        chosen
            .entry(pool[pick].deleted)
            .or_insert_with(|| (Vec::new(), pick))
            .0
            .push(tag);
    }
    // in draw-tag order of each pick's first tag, so the output order is fixed
    let mut picks: Vec<(Vec<&'static str>, usize)> = chosen.into_values().collect();
    picks.sort_by_key(|(tags, _)| DRAW_TAGS.iter().position(|t| *t == tags[0]));
    let drawn = picks
        .into_iter()
        .map(|(draws, i)| {
            let deletion = pool[i].clone();
            let x = deletion.deleted;
            Drawn {
                draws,
                bfs_parent: stored.parent.get(&x).map(|p| graph.name(*p).to_string()),
                d_ball: dist.get(&x).copied(),
                degree_graph: graph.out_neighbours(x).len() + in_index.heads(x).len(),
                deletion,
            }
        })
        .collect();
    Ok(StartDraws {
        drawn,
        candidates: pool.len(),
        without_neighbour,
        without_vector,
        start_isolated,
        stratum_sizes,
        uncached_candidates,
        fallback,
    })
}

/// The visible payload of a rebuilt ball: the start, the node names and the
/// induced edges in ball order with `edge_id`s, exactly as the sampler builds
/// an unpruned ball's edge list; no text, no target, no X.
pub fn visible_payload(graph: &RealGraph, family: &str, start: NodeId, ball: &[NodeId]) -> Value {
    let sub = graph.induced(ball);
    let mut edges = Vec::new();
    for &h in ball {
        for e in sub.out(h) {
            let relation = if graph.typed() {
                Value::from(graph.relation_name(e.relation))
            } else {
                Value::Null
            };
            edges.push(json!({
                "edge_id": edges.len(),
                "source": graph.name(h),
                "target": graph.name(e.tail),
                "relation": relation,
            }));
        }
    }
    json!({
        "record_kind": "r1_deletion_visible_v1",
        "family": family,
        "start_node": graph.name(start),
        "nodes": ball.iter().map(|n| graph.name(*n)).collect::<Vec<_>>(),
        "edges": edges,
    })
}

/// The dense top-1 of a ball by cosine to a query (first maximum in ball
/// order; nodes without a vector are skipped): the start-at-h variant's `h`.
pub fn dense_top1(
    graph: &RealGraph,
    ball: &[NodeId],
    query: &[f64],
    vector: &dyn Fn(&str) -> Option<Vec<f64>>,
) -> Option<NodeId> {
    let mut best: Option<(f64, NodeId)> = None;
    for &n in ball {
        if let Some(v) = vector(graph.name(n)) {
            let c = hf_policies::cosine(&v, query);
            if best.is_none_or(|(b, _)| c > b) {
                best = Some((c, n));
            }
        }
    }
    best.map(|(_, n)| n)
}

/// Every node name the payload mentions (nodes and edge endpoints) and the
/// episode id — what test (c) and the writer's own guard check X against.
pub fn mentions(episode_id: &str, visible: &Value) -> BTreeSet<String> {
    let mut out: BTreeSet<String> = BTreeSet::new();
    out.insert(episode_id.to_string());
    out.insert(visible["start_node"].as_str().unwrap_or("").to_string());
    for n in visible["nodes"].as_array().into_iter().flatten() {
        out.insert(n.as_str().unwrap_or("").to_string());
    }
    for e in visible["edges"].as_array().into_iter().flatten() {
        for k in ["source", "target"] {
            out.insert(e[k].as_str().unwrap_or("").to_string());
        }
    }
    out
}

/// A name → id map of a ball, for callers that hold names.
pub fn ids_of(graph: &RealGraph, names: &[String]) -> Result<HashMap<String, NodeId>, HfError> {
    names
        .iter()
        .map(|n| {
            graph
                .id(n)
                .map(|i| (n.clone(), i))
                .ok_or_else(|| HfError::BandH(format!("{n} is not in the graph")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SamplerConfig;

    fn graph(pairs: &[(&str, &str)]) -> RealGraph {
        RealGraph::from_edges("t", pairs.iter().map(|(h, t)| (*h, None, *t)))
    }

    fn sampler_on(g: &RealGraph, size: u32, cap: Option<u32>) -> Sampler<'_> {
        let mut c = SamplerConfig::new("t", size, 1, 2);
        c.hub_degree_cap = cap;
        Sampler::new(g, c).unwrap()
    }

    #[test]
    fn raw_index_reads_the_sampler_id_and_the_reserve_is_refused() {
        assert_eq!(
            raw_index("wikidata5m-screen-004103-2b7c4d01-greedy-path"),
            Some(("screen", 4103))
        );
        assert_eq!(
            raw_index("vault-quartz-docs-train-072452-00ff00ff"),
            Some(("train", 72452))
        );
        assert!(refuse_reserved("wikidata5m-screen-004104-aa").is_ok());
        assert!(refuse_reserved("wikidata5m-screen-004105-aa").is_err());
        assert!(refuse_reserved("wikidata5m-train-073359-aa").is_ok());
        assert!(refuse_reserved("wikidata5m-train-073360-aa").is_err());
        assert!(refuse_reserved("no-index-here").is_err());
    }

    /// ENG-1 (b), the parser half: both ends inclusive, `lo..` open, a
    /// reversed or malformed range refused; the premise's ranges are exactly
    /// the complement of the old reserve.
    #[test]
    fn allowed_ranges_are_inclusive_and_the_premise_spec_is_the_old_reserve() {
        let r = AllowedRange::parse("train:0..72452").unwrap();
        assert!(r.contains("train", 0) && r.contains("train", 72452));
        assert!(!r.contains("train", 72453) && !r.contains("screen", 5));
        let open = AllowedRange::parse("screen:8214..").unwrap();
        assert!(!open.contains("screen", 8213) && open.contains("screen", 8214));
        assert!(open.contains("screen", 999_999));
        assert_eq!(open.label(), "screen:8214..");
        assert_eq!(r.label(), "train:0..72452");
        for bad in [
            "train:5..4",
            "vault:0..1",
            "train:0",
            "train:a..b",
            "0..1",
            "train:..5",
        ] {
            assert!(AllowedRange::parse(bad).is_err(), "{bad}");
        }
        let premise = DrawSpec::premise();
        for id in [
            "wikidata5m-screen-004104-aa",
            "wikidata5m-train-073359-aa",
            "wikidata5m-train-000000-aa",
        ] {
            assert!(premise.check(id).is_ok(), "{id}");
            assert!(refuse_reserved(id).is_ok(), "{id}");
        }
        for id in [
            "wikidata5m-screen-004105-aa",
            "wikidata5m-train-073360-aa",
            "no-index-here",
        ] {
            assert!(premise.check(id).is_err(), "{id}");
            assert!(refuse_reserved(id).is_err(), "{id}");
        }
        let train_only = DrawSpec {
            label: "x".into(),
            ranges: vec![AllowedRange::parse("train:76043..").unwrap()],
        };
        let e = train_only.check("wikidata5m-screen-009000-aa").unwrap_err();
        assert!(e.to_string().contains("not declared"), "{e}");
    }

    #[test]
    fn strata_cut_where_the_plan_says() {
        let got: Vec<Option<&str>> = [0, 1, 2, 3, 4, 7, 8, 30]
            .iter()
            .map(|n| stratum(*n))
            .collect();
        assert_eq!(
            got,
            vec![
                None,
                Some("D1"),
                Some("D2"),
                Some("D2"),
                Some("D3"),
                Some("D3"),
                Some("D4"),
                Some("D4")
            ]
        );
    }

    /// (a) a node reachable only via X leaves; (d) in- and out-neighbours both
    /// enter T with their directions; L holds the neighbour that left with X;
    /// (b) the in-edge invariant holds on the rebuilt ball.
    #[test]
    fn deletion_labels_in_and_out_neighbours_and_what_left() {
        let g = graph(&[
            ("s", "a"),
            ("s", "x"),
            ("a", "x"),
            ("x", "b"),
            ("s", "b"),
            ("x", "y"),
            ("b", "c"),
        ]);
        let sampler = sampler_on(&g, 20, None);
        let s = g.id("s").unwrap();
        let x = g.id("x").unwrap();
        let stored = g.ball(s, 20, None, 3, None).unwrap();
        let d = delete_node(&g, &sampler, "train", s, &stored, x).unwrap();
        let names: Vec<&str> = d.ball.iter().map(|n| g.name(*n)).collect();
        assert!(!names.contains(&"x"));
        assert!(!names.contains(&"y"), "y was reachable only through x");
        let t: Vec<(&str, &str)> = d
            .targets
            .iter()
            .map(|n| (n.node.as_str(), n.direction))
            .collect();
        assert_eq!(t, vec![("a", "in"), ("b", "out")]);
        assert!(d.start_is_neighbour, "s -> x");
        assert_eq!(d.left, vec!["y".to_string()]);
        assert_eq!(d.departed, 1);
        for n in &d.ball[1..] {
            assert!(d.ball.iter().any(|p| g.out_neighbours(*p).contains(n)));
        }
    }

    /// The reverse index is the out-adjacency transposed, distinct heads, and
    /// `N_G(X)` joins both directions without X itself.
    #[test]
    fn the_in_index_transposes_the_graph() {
        let g = graph(&[("a", "x"), ("b", "x"), ("x", "c"), ("x", "a"), ("a", "b")]);
        let idx = InIndex::new(&g);
        let x = g.id("x").unwrap();
        let heads: Vec<&str> = idx.heads(x).iter().map(|n| g.name(*n)).collect();
        assert_eq!(heads, vec!["a", "b"]);
        let (all, cut) = graph_neighbours(&g, &idx, x);
        assert_eq!(all, vec!["a", "b", "c"]);
        assert!(!cut);
        for n in g.nodes() {
            for h in idx.heads(n) {
                assert!(g.out_neighbours(*h).contains(&n));
            }
        }
    }

    /// (e) the region filter holds on every node of the rebuilt ball.
    #[test]
    fn the_region_filter_is_respected() {
        let (g, _) = crate::fixture::fixture_world(5, 400, 1600, 8);
        let mut c = SamplerConfig::new("fixture", 40, 3, 2);
        c.screen_region = 0.5;
        let mut sampler = Sampler::new(&g, c).unwrap();
        let starts = sampler.prepare("screen").unwrap().to_vec();
        let mut checked = 0;
        for &s in starts.iter().take(20) {
            let Ok(stored) = g.ball(s, 40, None, 3, Some(&|n| sampler.member("screen", n))) else {
                continue;
            };
            for &x in stored.iter().skip(1).take(5) {
                let d = delete_node(&g, &sampler, "screen", s, &stored, x).unwrap();
                for n in &d.ball[1..] {
                    assert!(sampler.member("screen", *n));
                }
                checked += 1;
            }
        }
        assert!(checked > 20);
    }
}
