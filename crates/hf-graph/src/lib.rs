//! A real network, as `hippocampus_foundation.read_run.graph_v5.RealGraph`
//! reads one from an adapter's `edges.tsv` / `text.tsv`, with the same
//! semantics where they are load-bearing for episode identity:
//!
//! - exact `(head, relation, tail)` triples are de-duplicated at build time and
//!   each adjacency list is ordered by `(relation or "", tail)`;
//! - `out_neighbours` is the sorted set of distinct tails (parallel relations
//!   between one pair collapse: paths are node sequences);
//! - `ball` is a breadth-first order with the hub fan-out keyed by SHA-256;
//! - `out_degree_percentile` counts edge multiplicity and rounds half to even,
//!   while `ball` caps on distinct neighbours — the Python asymmetry, kept;
//! - a three-column file with an empty relation column loads as `Some("")`,
//!   not `None`, so `typed` reads true for vault graphs exactly as in Python.
//!
//! Node ids are interned in string order, so comparing ids is comparing names:
//! sorted paths and sorted neighbour lists come out in Python's order.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use hf_core::HfError;
use sha2::{Digest, Sha256};

/// A node id: the node's rank among all node names in byte (= code point) order.
pub type NodeId = u32;

/// One out-edge: the relation's rank (0 is the empty relation or none) and the tail.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct OutEdge {
    pub relation: u32,
    pub tail: NodeId,
}

/// A ball's breadth-first order and the discovery parent of every node but
/// the start (the node whose capped tail list first reached it).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BallTrace {
    pub order: Vec<NodeId>,
    pub parent: HashMap<NodeId, NodeId>,
}

#[derive(Clone, Debug)]
pub struct RealGraph {
    pub family: String,
    names: Vec<String>,
    index: HashMap<String, NodeId>,
    relations: Vec<String>,
    /// CSR: edges of node `n` are `edges[offsets[n]..offsets[n + 1]]`, sorted by `(relation, tail)`.
    offsets: Vec<u32>,
    edges: Vec<OutEdge>,
    /// Whether any line carried a relation column (Python's `typed`: any relation `is not None`).
    relation_column: bool,
    text: Vec<Option<String>>,
}

fn open_maybe_gz(path: &Path) -> std::io::Result<Box<dyn Read>> {
    let file = std::fs::File::open(path)?;
    if path.to_string_lossy().ends_with(".gz") {
        Ok(Box::new(flate2::read::MultiGzDecoder::new(file)))
    } else {
        Ok(Box::new(file))
    }
}

/// Python's `round()` on a non-negative float: half to even.
pub fn round_half_even(x: f64) -> f64 {
    let floor = x.floor();
    let diff = x - floor;
    let round_up = diff > 0.5 || (diff == 0.5 && floor % 2.0 != 0.0);
    if round_up {
        floor + 1.0
    } else {
        floor
    }
}

impl RealGraph {
    /// `RealGraph.from_edges`: exact triples de-duplicated, adjacency sorted by `(relation or "", tail)`.
    pub fn from_edges<'a>(
        family: &str,
        triples: impl IntoIterator<Item = (&'a str, Option<&'a str>, &'a str)>,
    ) -> Self {
        let mut names: Vec<&str> = Vec::new();
        let mut relation_names: Vec<&str> = Vec::new();
        let mut raw: Vec<(&str, Option<&str>, &str)> = Vec::new();
        let mut relation_column = false;
        for (h, r, t) in triples {
            names.push(h);
            names.push(t);
            if let Some(r) = r {
                relation_column = true;
                if !r.is_empty() {
                    relation_names.push(r);
                }
            }
            raw.push((h, r, t));
        }
        names.sort_unstable();
        names.dedup();
        relation_names.sort_unstable();
        relation_names.dedup();
        let index: HashMap<String, NodeId> = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.to_string(), i as NodeId))
            .collect();
        let relation_index: HashMap<&str, u32> = relation_names
            .iter()
            .enumerate()
            .map(|(i, r)| (*r, i as u32 + 1))
            .collect();
        let mut triples: Vec<(NodeId, u32, NodeId)> = raw
            .iter()
            .map(|(h, r, t)| {
                let rel = match r {
                    Some(r) if !r.is_empty() => relation_index[r],
                    _ => 0,
                };
                (index[*h], rel, index[*t])
            })
            .collect();
        triples.sort_unstable();
        triples.dedup();
        let mut offsets = vec![0u32; names.len() + 1];
        for (h, _, _) in &triples {
            offsets[*h as usize + 1] += 1;
        }
        for i in 0..names.len() {
            offsets[i + 1] += offsets[i];
        }
        let edges = triples
            .iter()
            .map(|(_, r, t)| OutEdge {
                relation: *r,
                tail: *t,
            })
            .collect();
        let mut relations = vec![String::new()];
        relations.extend(relation_names.iter().map(|r| r.to_string()));
        Self {
            family: family.to_string(),
            text: vec![None; names.len()],
            names: names.into_iter().map(str::to_string).collect(),
            index,
            relations,
            offsets,
            edges,
            relation_column,
        }
    }

    /// `RealGraph.from_triples`: a TSV of `head<TAB>relation<TAB>tail` (three
    /// columns) or `head<TAB>tail` (two); any other arity is skipped; `.gz` is sniffed.
    pub fn from_triples(path: &Path, family: &str, limit: Option<usize>) -> Result<Self, HfError> {
        let mut reader = BufReader::with_capacity(
            1 << 20,
            open_maybe_gz(path)
                .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))?,
        );
        let mut content = String::new();
        reader
            .read_to_string(&mut content)
            .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))?;
        let mut triples = Vec::new();
        for (i, line) in content.split_inclusive('\n').enumerate() {
            if let Some(limit) = limit {
                if i >= limit {
                    break;
                }
            }
            let line = line.strip_suffix('\n').unwrap_or(line);
            let mut parts = line.split('\t');
            let (a, b, c, d) = (parts.next(), parts.next(), parts.next(), parts.next());
            match (a, b, c, d) {
                (Some(h), Some(r), Some(t), None) => triples.push((h, Some(r), t)),
                (Some(h), Some(t), None, None) => triples.push((h, None, t)),
                _ => continue,
            }
        }
        Ok(Self::from_edges(family, triples))
    }

    /// `RealGraph.load_text`: attach `node<TAB>text` lines for known nodes; returns the count attached.
    pub fn load_text(&mut self, path: &Path) -> Result<usize, HfError> {
        let reader = BufReader::with_capacity(
            1 << 20,
            open_maybe_gz(path)
                .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))?,
        );
        let mut attached = 0;
        for line in reader.split(b'\n') {
            let line = line.map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))?;
            let line = String::from_utf8_lossy(&line);
            let (node, body) = line.split_once('\t').unwrap_or((&line, ""));
            if let Some(id) = self.index.get(node) {
                self.text[*id as usize] = Some(body.to_string());
                attached += 1;
            }
        }
        Ok(attached)
    }

    /// Attach texts for known nodes from memory (the fixture world's `text=` map).
    pub fn set_texts<'a>(&mut self, texts: impl IntoIterator<Item = (&'a str, &'a str)>) -> usize {
        let mut attached = 0;
        for (node, body) in texts {
            if let Some(id) = self.index.get(node) {
                self.text[*id as usize] = Some(body.to_string());
                attached += 1;
            }
        }
        attached
    }

    pub fn node_count(&self) -> usize {
        self.names.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Python's `typed`: any relation is not `None` — an empty relation column counts.
    pub fn typed(&self) -> bool {
        self.relation_column
    }

    pub fn id(&self, name: &str) -> Option<NodeId> {
        self.index.get(name).copied()
    }

    pub fn name(&self, id: NodeId) -> &str {
        &self.names[id as usize]
    }

    pub fn relation_name(&self, relation: u32) -> &str {
        &self.relations[relation as usize]
    }

    pub fn text(&self, id: NodeId) -> Option<&str> {
        self.text[id as usize].as_deref()
    }

    /// All node ids, which is the sorted node-name order.
    pub fn nodes(&self) -> impl Iterator<Item = NodeId> {
        0..self.names.len() as NodeId
    }

    /// The out-edges of a node in `(relation, tail)` order, with multiplicity.
    pub fn out(&self, node: NodeId) -> &[OutEdge] {
        let (lo, hi) = (
            self.offsets[node as usize] as usize,
            self.offsets[node as usize + 1] as usize,
        );
        &self.edges[lo..hi]
    }

    /// Whether the node has any out-edge (`node in graph.out` in Python).
    pub fn has_out(&self, node: NodeId) -> bool {
        self.offsets[node as usize] != self.offsets[node as usize + 1]
    }

    /// `out_neighbours`: the sorted distinct tails.
    pub fn out_neighbours(&self, node: NodeId) -> Vec<NodeId> {
        let mut tails: Vec<NodeId> = self.out(node).iter().map(|e| e.tail).collect();
        tails.sort_unstable();
        tails.dedup();
        tails
    }

    /// `out_degree_percentile`: over nodes with out-edges, edge multiplicity
    /// counted, the sorted degree at `round(p / 100 * (n - 1))` with half-to-even.
    pub fn out_degree_percentile(&self, percentile: f64) -> Result<u32, HfError> {
        let mut degrees: Vec<u32> = (0..self.names.len())
            .map(|n| self.offsets[n + 1] - self.offsets[n])
            .filter(|d| *d > 0)
            .collect();
        if degrees.is_empty() {
            return Err(HfError::Invalid("graph has no out-edges".into()));
        }
        degrees.sort_unstable();
        let last = (degrees.len() - 1) as f64;
        let k = round_half_even(percentile / 100.0 * last).max(0.0) as usize;
        Ok(degrees[k.min(degrees.len() - 1)])
    }

    /// `ball`: breadth-first from `start` to `size` nodes; a node other than the
    /// start with more than `hub_cap` distinct neighbours contributes only its
    /// `hub_fanout` neighbours ranked by `sha256(start ␟ node ␟ tail)`; `allow`
    /// filters tails only.
    pub fn ball(
        &self,
        start: NodeId,
        size: usize,
        hub_cap: Option<u32>,
        hub_fanout: usize,
        allow: Option<&dyn Fn(NodeId) -> bool>,
    ) -> Result<Vec<NodeId>, HfError> {
        self.ball_traced(start, size, hub_cap, hub_fanout, allow, None)
            .map(|t| t.order)
    }

    /// `ball` with the BFS discovery parent of every node beside the order,
    /// on the graph with `exclude` **deleted**: the excluded node is removed
    /// from every out-list BEFORE the hub cap is checked and before a hub's
    /// fan-out is ranked, so a hub whose only surplus tail was the excluded
    /// node is no longer capped and a hub that had hashed it among its
    /// `hub_fanout` picks takes its next-ranked tail instead. This is the ball
    /// of `G ∖ {exclude}`, which filtering the node out through `allow` is
    /// not: `allow` runs after the cap and after the ranking. With
    /// `exclude = None` it is `ball` exactly. The start is never excluded.
    pub fn ball_traced(
        &self,
        start: NodeId,
        size: usize,
        hub_cap: Option<u32>,
        hub_fanout: usize,
        allow: Option<&dyn Fn(NodeId) -> bool>,
        exclude: Option<NodeId>,
    ) -> Result<BallTrace, HfError> {
        if exclude == Some(start) {
            return Err(HfError::Invalid(
                "the ball centre cannot be excluded".into(),
            ));
        }
        let mut start_tails = self.out_neighbours(start);
        start_tails.retain(|t| Some(*t) != exclude);
        if start_tails.is_empty() {
            return Err(HfError::Invalid("ball centre has no out-edges".into()));
        }
        let mut order = vec![start];
        let mut parent: HashMap<NodeId, NodeId> = HashMap::new();
        let mut seen: HashSet<NodeId> = HashSet::from([start]);
        let mut queue = VecDeque::from([start]);
        'outer: while let Some(node) = queue.pop_front() {
            if order.len() >= size {
                break;
            }
            let mut tails = self.out_neighbours(node);
            tails.retain(|t| Some(*t) != exclude);
            if let Some(cap) = hub_cap {
                if node != start && tails.len() as u32 > cap {
                    let start_name = self.name(start);
                    let node_name = self.name(node);
                    let mut keyed: Vec<([u8; 32], NodeId)> = tails
                        .iter()
                        .map(|t| {
                            let mut h = Sha256::new();
                            h.update(start_name.as_bytes());
                            h.update(b"\x1f");
                            h.update(node_name.as_bytes());
                            h.update(b"\x1f");
                            h.update(self.name(*t).as_bytes());
                            (h.finalize().into(), *t)
                        })
                        .collect();
                    keyed.sort_unstable();
                    tails = keyed.into_iter().take(hub_fanout).map(|(_, t)| t).collect();
                }
            }
            for tail in tails {
                if seen.contains(&tail) {
                    continue;
                }
                if let Some(allow) = allow {
                    if !allow(tail) {
                        continue;
                    }
                }
                seen.insert(tail);
                order.push(tail);
                parent.insert(tail, node);
                queue.push_back(tail);
                if order.len() >= size {
                    break 'outer;
                }
            }
        }
        Ok(BallTrace { order, parent })
    }

    /// `induced`: the subgraph on a node set, edges with both ends inside.
    pub fn induced(&self, nodes: &[NodeId]) -> Subgraph {
        let members: HashSet<NodeId> = nodes.iter().copied().collect();
        let mut out: HashMap<NodeId, Vec<OutEdge>> = HashMap::new();
        for &n in nodes {
            let kept: Vec<OutEdge> = self
                .out(n)
                .iter()
                .copied()
                .filter(|e| members.contains(&e.tail))
                .collect();
            if !kept.is_empty() {
                out.insert(n, kept);
            }
        }
        Subgraph {
            nodes: nodes.to_vec(),
            members,
            out,
        }
    }
}

/// An induced subgraph over the parent's node ids, in the ball's discovery order.
#[derive(Clone, Debug)]
pub struct Subgraph {
    pub nodes: Vec<NodeId>,
    members: HashSet<NodeId>,
    out: HashMap<NodeId, Vec<OutEdge>>,
}

impl Subgraph {
    pub fn contains(&self, node: NodeId) -> bool {
        self.members.contains(&node)
    }

    /// Out-edges in `(relation, tail)` order with multiplicity (empty if none).
    pub fn out(&self, node: NodeId) -> &[OutEdge] {
        self.out.get(&node).map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn has_out(&self, node: NodeId) -> bool {
        self.out.contains_key(&node)
    }

    pub fn edge_count(&self) -> usize {
        self.out.values().map(Vec::len).sum()
    }

    /// The subgraph with every edge whose `(head, tail)` pair the predicate names
    /// removed — all parallel relations between the pair go at once, as the
    /// sampler prunes by node pair. Node order and edge order are kept.
    pub fn without_pairs(&self, remove: impl Fn(NodeId, NodeId) -> bool) -> Subgraph {
        let mut out: HashMap<NodeId, Vec<OutEdge>> = HashMap::new();
        for (head, edges) in &self.out {
            let kept: Vec<OutEdge> = edges
                .iter()
                .copied()
                .filter(|e| !remove(*head, e.tail))
                .collect();
            if !kept.is_empty() {
                out.insert(*head, kept);
            }
        }
        Subgraph {
            nodes: self.nodes.clone(),
            members: self.members.clone(),
            out,
        }
    }

    pub fn out_neighbours(&self, node: NodeId) -> Vec<NodeId> {
        let mut tails: Vec<NodeId> = self.out(node).iter().map(|e| e.tail).collect();
        tails.sort_unstable();
        tails.dedup();
        tails
    }

    /// Breadth-first distances from a node (unreachable nodes absent).
    pub fn distances_from(&self, start: NodeId) -> HashMap<NodeId, u32> {
        let mut dist = HashMap::from([(start, 0u32)]);
        let mut queue = VecDeque::from([start]);
        while let Some(node) = queue.pop_front() {
            let d = dist[&node];
            for tail in self.out_neighbours(node) {
                if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(tail) {
                    e.insert(d + 1);
                    queue.push_back(tail);
                }
            }
        }
        dist
    }

    /// Breadth-first distances to a node over reversed edges.
    pub fn distances_to(&self, target: NodeId) -> HashMap<NodeId, u32> {
        let mut incoming: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for (head, edges) in &self.out {
            for e in edges {
                incoming.entry(e.tail).or_default().push(*head);
            }
        }
        let mut dist = HashMap::from([(target, 0u32)]);
        let mut queue = VecDeque::from([target]);
        while let Some(node) = queue.pop_front() {
            let d = dist[&node];
            if let Some(heads) = incoming.get(&node) {
                for head in heads {
                    if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(*head) {
                        e.insert(d + 1);
                        queue.push_back(*head);
                    }
                }
            }
        }
        dist
    }

    /// Enumerate simple paths, returning `None` as soon as more than `max_paths`
    /// have been found. A completed result has exactly the same ordering as
    /// [`Self::simple_paths`].
    pub fn simple_paths_capped(
        &self,
        start: NodeId,
        target: NodeId,
        max_cost: u32,
        max_paths: usize,
    ) -> Option<Vec<Vec<NodeId>>> {
        self.enumerate_simple_paths(start, target, max_cost, Some(max_paths))
    }

    /// `simple_paths`: every simple path from `start` to `target` with at most
    /// `max_cost` edges, pruned by the exact distance-to-target, sorted.
    pub fn simple_paths(&self, start: NodeId, target: NodeId, max_cost: u32) -> Vec<Vec<NodeId>> {
        self.enumerate_simple_paths(start, target, max_cost, None)
            .expect("an uncapped enumeration cannot exceed its cap")
    }

    fn enumerate_simple_paths(
        &self,
        start: NodeId,
        target: NodeId,
        max_cost: u32,
        max_paths: Option<usize>,
    ) -> Option<Vec<Vec<NodeId>>> {
        let need = self.distances_to(target);
        let mut found = Vec::new();
        if !need.contains_key(&start) {
            return Some(found);
        }
        let mut path = vec![start];
        let mut on_path: HashSet<NodeId> = HashSet::from([start]);
        // iterative DFS: a stack of (node, next-neighbour cursor, neighbours)
        let mut stack: Vec<(Vec<NodeId>, usize)> = vec![(self.out_neighbours(start), 0)];
        if start == target {
            found.push(path.clone());
            return (max_paths.is_none_or(|cap| found.len() <= cap)).then_some(found);
        }
        while let Some((neighbours, cursor)) = stack.last_mut() {
            if *cursor >= neighbours.len() {
                stack.pop();
                let left = path.pop().expect("path");
                on_path.remove(&left);
                continue;
            }
            let tail = neighbours[*cursor];
            *cursor += 1;
            if on_path.contains(&tail) {
                continue;
            }
            let Some(n) = need.get(&tail) else { continue };
            let used = (path.len() - 1) as u32;
            if used + 1 + n > max_cost {
                continue;
            }
            if tail == target {
                let mut p = path.clone();
                p.push(tail);
                found.push(p);
                if max_paths.is_some_and(|cap| found.len() > cap) {
                    return None;
                }
                continue;
            }
            on_path.insert(tail);
            path.push(tail);
            stack.push((self.out_neighbours(tail), 0));
        }
        found.sort_unstable();
        Some(found)
    }

    /// The unpruned reference enumeration, for tests.
    pub fn brute_force_paths(
        &self,
        start: NodeId,
        target: NodeId,
        max_cost: u32,
    ) -> Vec<Vec<NodeId>> {
        let mut found = Vec::new();
        let mut path = vec![start];
        let mut on_path: HashSet<NodeId> = HashSet::from([start]);
        fn dfs(
            g: &Subgraph,
            node: NodeId,
            target: NodeId,
            max_cost: u32,
            path: &mut Vec<NodeId>,
            on_path: &mut HashSet<NodeId>,
            found: &mut Vec<Vec<NodeId>>,
        ) {
            if node == target {
                found.push(path.clone());
                return;
            }
            if (path.len() - 1) as u32 >= max_cost {
                return;
            }
            for tail in g.out_neighbours(node) {
                if on_path.contains(&tail) {
                    continue;
                }
                on_path.insert(tail);
                path.push(tail);
                dfs(g, tail, target, max_cost, path, on_path, found);
                path.pop();
                on_path.remove(&tail);
            }
        }
        dfs(
            self,
            start,
            target,
            max_cost,
            &mut path,
            &mut on_path,
            &mut found,
        );
        found.sort_unstable();
        found
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_is_half_to_even() {
        assert_eq!(round_half_even(0.5), 0.0);
        assert_eq!(round_half_even(1.5), 2.0);
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(2.51), 3.0);
        assert_eq!(round_half_even(3.0), 3.0);
    }

    #[test]
    fn triples_dedup_and_sort_like_python() {
        let g = RealGraph::from_edges(
            "t",
            [
                ("b", Some("P2"), "a"),
                ("b", Some("P1"), "c"),
                ("b", Some("P1"), "c"),
                ("b", Some(""), "z"),
                ("a", None, "b"),
            ],
        );
        assert_eq!(g.edge_count(), 4);
        let b = g.id("b").unwrap();
        let names: Vec<(String, String)> = g
            .out(b)
            .iter()
            .map(|e| {
                (
                    g.relation_name(e.relation).to_string(),
                    g.name(e.tail).to_string(),
                )
            })
            .collect();
        assert_eq!(
            names,
            vec![
                ("".into(), "z".into()),
                ("P1".into(), "c".into()),
                ("P2".into(), "a".into())
            ]
        );
        assert_eq!(
            g.out_neighbours(b)
                .iter()
                .map(|t| g.name(*t))
                .collect::<Vec<_>>(),
            vec!["a", "c", "z"]
        );
        assert!(g.typed());
    }

    fn graph_of(triples: &[(String, String)]) -> RealGraph {
        RealGraph::from_edges(
            "t",
            triples.iter().map(|(h, t)| (h.as_str(), None, t.as_str())),
        )
    }

    fn names(g: &RealGraph, ids: &[NodeId]) -> Vec<String> {
        ids.iter().map(|n| g.name(*n).to_string()).collect()
    }

    /// `ball_traced` with nothing excluded is `ball`, order for order.
    #[test]
    fn traced_ball_without_exclusion_is_the_ball() {
        let mut triples = Vec::new();
        for i in 0..30u32 {
            for j in [1u32, 7, 11] {
                triples.push((format!("n{i}"), format!("n{}", (i * j + 3) % 30)));
            }
        }
        let g = graph_of(&triples);
        let s = g.id("n0").unwrap();
        for cap in [None, Some(2)] {
            let plain = g.ball(s, 12, cap, 3, None).unwrap();
            let traced = g.ball_traced(s, 12, cap, 3, None, None).unwrap();
            assert_eq!(plain, traced.order);
            for n in &traced.order[1..] {
                let p = traced.parent[n];
                assert!(g.out_neighbours(p).contains(n));
            }
        }
    }

    /// A node reachable only through the excluded node leaves the ball; the
    /// excluded node itself never enters it.
    #[test]
    fn excluding_a_node_drops_what_only_it_reached() {
        let pairs = [("s", "a"), ("s", "x"), ("x", "y"), ("a", "b")];
        let triples: Vec<(String, String)> = pairs
            .iter()
            .map(|(h, t)| (h.to_string(), t.to_string()))
            .collect();
        let g = graph_of(&triples);
        let s = g.id("s").unwrap();
        let x = g.id("x").unwrap();
        let with = names(&g, &g.ball(s, 10, None, 3, None).unwrap());
        assert!(with.contains(&"y".to_string()));
        let without = g.ball_traced(s, 10, None, 3, None, Some(x)).unwrap();
        let without = names(&g, &without.order);
        assert_eq!(without, vec!["s", "a", "b"]);
        assert!(g.ball_traced(s, 10, None, 3, None, Some(s)).is_err());
    }

    /// The ball of G minus x, for a hub test: the same triples with every edge
    /// touching x deleted, BFS'd on its own graph.
    fn ball_of_deleted(triples: &[(String, String)], x: &str, cap: u32) -> Vec<String> {
        let kept: Vec<(String, String)> = triples
            .iter()
            .filter(|(h, t)| h != x && t != x)
            .cloned()
            .collect();
        let g = graph_of(&kept);
        names(
            &g,
            &g.ball(g.id("s").unwrap(), 50, Some(cap), 3, None).unwrap(),
        )
    }

    /// Plan test (h): x is one of a capped hub's three hashed tails. Deleting
    /// it must hand the hub's slot to its NEXT-ranked tail, as the ball of the
    /// graph with x physically deleted does; filtering x through `allow` (which
    /// runs after the ranking) must be shown to differ.
    #[test]
    fn excluding_a_hashed_hub_tail_takes_the_next_ranked_tail() {
        let cap = 3u32;
        let mut found = false;
        for candidate in 0..200 {
            let x = format!("x{candidate:03}");
            let mut triples: Vec<(String, String)> = vec![("s".into(), "h".into())];
            for t in ["t0", "t1", "t2", "t3", "t4", "t5"] {
                triples.push(("h".into(), t.into()));
            }
            triples.push(("h".into(), x.clone()));
            let g = graph_of(&triples);
            let s = g.id("s").unwrap();
            let xid = g.id(&x).unwrap();
            let full = names(&g, &g.ball(s, 50, Some(cap), 3, None).unwrap());
            if !full.contains(&x) {
                continue; // x is not among the hub's hashed picks; try another name
            }
            found = true;
            let traced = g.ball_traced(s, 50, Some(cap), 3, None, Some(xid)).unwrap();
            let traced = names(&g, &traced.order);
            let deleted = ball_of_deleted(&triples, &x, cap);
            assert_eq!(traced, deleted, "exclusion must equal physical deletion");
            assert_eq!(traced.len(), full.len(), "the hub keeps three picks");
            let not_x = |n: NodeId| n != xid;
            let allow_only = names(&g, &g.ball(s, 50, Some(cap), 3, Some(&not_x)).unwrap());
            assert_ne!(allow_only, traced, "allow-only filtering is not deletion");
            assert_eq!(allow_only.len() + 1, traced.len());
            break;
        }
        assert!(found, "no candidate name was hashed among the hub's picks");
    }

    /// Plan test (k): a hub at exactly cap + 1 tails, x among them, is capped
    /// in G and UNCAPPED in G minus x, so every one of its remaining tails
    /// enters the ball.
    #[test]
    fn a_hub_at_cap_plus_one_is_uncapped_once_x_is_gone() {
        let cap = 3u32;
        let mut triples: Vec<(String, String)> = vec![("s".into(), "h".into())];
        for t in ["a", "b", "c", "x"] {
            triples.push(("h".into(), t.into()));
        }
        let g = graph_of(&triples);
        let s = g.id("s").unwrap();
        let x = g.id("x").unwrap();
        let full = g.ball(s, 50, Some(cap), 3, None).unwrap();
        assert_eq!(full.len(), 2 + 3, "capped in G: fan-out 3 of 4");
        let traced = g.ball_traced(s, 50, Some(cap), 3, None, Some(x)).unwrap();
        let got = names(&g, &traced.order);
        for t in ["a", "b", "c"] {
            assert!(got.contains(&t.to_string()), "{t} missing from {got:?}");
        }
        assert_eq!(got, ball_of_deleted(&triples, "x", cap));
    }

    #[test]
    fn capped_paths_match_complete_paths_until_the_cap() {
        let g = RealGraph::from_edges(
            "t",
            [
                ("s", None, "a"),
                ("s", None, "b"),
                ("a", None, "t"),
                ("b", None, "t"),
            ],
        );
        let nodes: Vec<NodeId> = g.nodes().collect();
        let sub = g.induced(&nodes);
        let start = g.id("s").unwrap();
        let target = g.id("t").unwrap();
        let complete = sub.simple_paths(start, target, 2);
        assert_eq!(sub.simple_paths_capped(start, target, 2, 2), Some(complete));
        assert_eq!(sub.simple_paths_capped(start, target, 2, 1), None);
    }

    #[test]
    fn capped_paths_abandon_a_combinatorial_graph() {
        let mut triples: Vec<(String, Option<String>, String)> = Vec::new();
        let mut previous = vec!["s".to_string()];
        for layer in 0..24 {
            let next = vec![format!("l{layer}a"), format!("l{layer}b")];
            for head in &previous {
                for tail in &next {
                    triples.push((head.clone(), None, tail.clone()));
                }
            }
            previous = next;
        }
        for head in &previous {
            triples.push((head.clone(), None, "t".to_string()));
        }
        let refs = triples
            .iter()
            .map(|(h, r, t)| (h.as_str(), r.as_deref(), t.as_str()));
        let g = RealGraph::from_edges("t", refs);
        let nodes: Vec<NodeId> = g.nodes().collect();
        let sub = g.induced(&nodes);
        assert_eq!(
            sub.simple_paths_capped(g.id("s").unwrap(), g.id("t").unwrap(), 25, 512),
            None
        );
    }
}
