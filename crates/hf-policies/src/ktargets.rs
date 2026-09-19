//! The k-target baselines of `K_TARGETS_DESIGN.md` §3 — k-greedy with its
//! recompute rule and the frozen variant beside it, the exact k-oracle by
//! branch-and-bound with a budget and an admissible sandwich, bidirectional
//! BFS run sequentially per target, and blind exhaust on all k.
//!
//! Every one of them reduces to its v1 baseline at k = 1, and the identity is
//! tested rather than asserted: the walk here is written out again rather than
//! delegated to `forward_walk`, so `tests/k_targets.rs` compares two
//! independent implementations on every fixture episode.
//!
//! Two facts fix the cost unit, from the note's own M1 clarification: an
//! expansion examines one node's out-edge group, the start is expanded first
//! and counts, and a target registers **on sight** as a child — so the fewest
//! expansions registering one target is `d(start, t)`, the shortest path
//! minus `t` itself.

use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use crate::{cosine, Embeddings, EpisodeGraph, Key, WalkTrace, STOP_EXHAUSTED, STOP_REGISTERED};

/// `K_POLICIES`, in report order.
pub const K_POLICY_NAMES: [&str; 5] = [
    "blind_exhaust",
    "bidirectional_sequential",
    "k_greedy",
    "k_greedy_frozen",
    "k_oracle",
];

/// The k-walk's registration bookkeeping: one flag per target, the expansion
/// count at which each first appeared as a child, and whether the unregistered
/// set changed at the step just taken (k-greedy's recompute trigger).
struct Registry {
    at: Vec<Option<u32>>,
    changed: bool,
}

impl Registry {
    fn new(k: usize) -> Self {
        Self {
            at: vec![None; k],
            changed: false,
        }
    }

    fn all(&self) -> bool {
        !self.at.is_empty() && self.at.iter().all(Option::is_some)
    }

    fn unregistered(&self) -> Vec<usize> {
        (0..self.at.len())
            .filter(|t| self.at[*t].is_none())
            .collect()
    }

    /// Register `t` on sight, at `expansions` expansions completed.
    fn see(&mut self, t: usize, expansions: u32) {
        if self.at[t].is_none() {
            self.at[t] = Some(expansions);
            self.changed = true;
        }
    }
}

fn finish(trace: &mut WalkTrace, reg: Registry) {
    let all = reg.all();
    trace.registered_at_by_target = reg.at;
    if all {
        trace.registered_at = trace
            .registered_at_by_target
            .iter()
            .flatten()
            .copied()
            .max();
        trace.stop_reason = STOP_REGISTERED.into();
    }
}

/// `_forward_walk` over k targets: the same stable re-sort of the whole
/// frontier each step and the same head pop, but registration is per target
/// and the walk ends only when **all** of them are registered. A target that
/// registers without completing the set is pushed onto the frontier like any
/// other child, exactly as the learned walk's `push_children` does.
fn k_forward_walk(
    g: &EpisodeGraph,
    mut priority: impl FnMut(&str, u32) -> Key,
    expandable: impl Fn(&str) -> bool,
) -> WalkTrace {
    let k = g.target_count();
    let mut trace = WalkTrace::with_targets(k);
    let mut reg = Registry::new(k);
    let mut frontier: Vec<(Key, String, u32)> = vec![(priority(&g.start, 0), g.start.clone(), 0)];
    let mut seen: HashSet<String> = HashSet::from([g.start.clone()]);
    for (t, target) in g.targets.iter().enumerate() {
        if g.start == *target {
            reg.see(t, 0);
        }
    }
    if reg.all() {
        finish(&mut trace, reg);
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
            if let Some(t) = g.targets.iter().position(|x| x == tail) {
                reg.see(t, trace.expansions);
                if reg.all() {
                    finish(&mut trace, reg);
                    return trace;
                }
            }
            if !seen.contains(tail) {
                seen.insert(tail.clone());
                trace.parents.insert(tail.clone(), node.clone());
                frontier.push((priority(tail, depth + 1), tail.clone(), depth + 1));
            }
        }
    }
    finish(&mut trace, reg);
    trace
}

/// Breadth-first, depth then discovery order, over all k. The ceiling.
pub fn k_blind_exhaust_trace(g: &EpisodeGraph) -> WalkTrace {
    let mut counter = 0u64;
    k_forward_walk(
        g,
        |_, depth| {
            counter += 1;
            Key(depth as f64, counter)
        },
        |_| true,
    )
}

/// §3 item 1: similarity-greedy toward the nearest **unregistered** target.
///
/// The key is `(-max over unregistered t of cos(node, t), counter)`, a
/// vectorless node at `+2.0`, the counter incrementing at every priority call.
/// The whole frontier is re-keyed, **in frontier order**, at the start of
/// every step at which the unregistered set changed, and at no other: a stale
/// key after a registration leaves the walk chasing a target it already has,
/// which would be a baseline weak by accident. `frozen` keeps v1's rule (each
/// node keyed once, at push) and is the variant reported beside.
pub fn k_greedy_walk(g: &EpisodeGraph, embeddings: &dyn Embeddings, frozen: bool) -> WalkTrace {
    let k = g.target_count();
    let queries: Vec<Option<Vec<f64>>> = g.targets.iter().map(|t| embeddings.vector(t)).collect();
    if queries.iter().all(Option::is_none) {
        return k_blind_exhaust_trace(g);
    }
    let mut counter = 0u64;
    let key_of = |node: &str, live: &[usize], counter: &mut u64| -> Key {
        *counter += 1;
        let value = match embeddings.vector(node) {
            Some(v) => {
                let best = live
                    .iter()
                    .filter_map(|t| queries[*t].as_ref())
                    .map(|q| cosine(&v, q))
                    .fold(f64::NEG_INFINITY, f64::max);
                if best == f64::NEG_INFINITY {
                    2.0
                } else {
                    -best
                }
            }
            None => 2.0,
        };
        Key(value, *counter)
    };

    let mut trace = WalkTrace::with_targets(k);
    let mut reg = Registry::new(k);
    let mut live = reg.unregistered();
    let mut frontier: Vec<(Key, String, u32)> =
        vec![(key_of(&g.start, &live, &mut counter), g.start.clone(), 0)];
    let mut seen: HashSet<String> = HashSet::from([g.start.clone()]);
    for (t, target) in g.targets.iter().enumerate() {
        if g.start == *target {
            reg.see(t, 0);
        }
    }
    if reg.all() {
        finish(&mut trace, reg);
        return trace;
    }
    while !frontier.is_empty() {
        if reg.changed {
            reg.changed = false;
            live = reg.unregistered();
            if !frozen {
                for entry in frontier.iter_mut() {
                    entry.0 = key_of(&entry.1, &live, &mut counter);
                }
            }
        }
        frontier.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite keys"));
        let (_, node, depth) = frontier.remove(0);
        trace.expansions += 1;
        trace.examined.push(node.clone());
        for tail in g.tails(&node) {
            if let Some(t) = g.targets.iter().position(|x| x == tail) {
                reg.see(t, trace.expansions);
                if reg.all() {
                    finish(&mut trace, reg);
                    return trace;
                }
            }
            if !seen.contains(tail) {
                seen.insert(tail.clone());
                trace.parents.insert(tail.clone(), node.clone());
                let key = key_of(tail, &reg.unregistered(), &mut counter);
                frontier.push((key, tail.clone(), depth + 1));
            }
        }
    }
    finish(&mut trace, reg);
    trace
}

/// k-greedy with the frontier re-keyed on every registration (§3 item 1).
pub fn k_greedy_trace(g: &EpisodeGraph, embeddings: &dyn Embeddings) -> WalkTrace {
    k_greedy_walk(g, embeddings, false)
}

/// `k-greedy-frozen`: the same key, computed once at push. Reported beside so
/// the recompute choice stays auditable.
pub fn k_greedy_frozen_trace(g: &EpisodeGraph, embeddings: &dyn Embeddings) -> WalkTrace {
    k_greedy_walk(g, embeddings, true)
}

/// §3 item 4: the exact single-target algorithm to each target in turn,
/// expansions summed. It double-counts the shared work, so it is an honest
/// **upper** bound and never a floor.
pub fn bidirectional_sequential_trace(g: &EpisodeGraph) -> WalkTrace {
    let mut trace = WalkTrace::with_targets(g.target_count());
    let mut total = 0u32;
    for (t, target) in g.targets.iter().enumerate() {
        let mut one = g.clone();
        one.targets = vec![target.clone()];
        let leg = crate::bidirectional_bfs_trace(&one);
        trace.examined.extend(leg.examined.iter().cloned());
        if let Some(at) = leg.registered_at {
            trace.registered_at_by_target[t] = Some(total + at);
        }
        total += leg.expansions;
    }
    trace.expansions = total;
    if trace.registered_at_by_target.iter().all(Option::is_some) {
        trace.registered_at = trace
            .registered_at_by_target
            .iter()
            .flatten()
            .copied()
            .max();
        trace.stop_reason = STOP_REGISTERED.into();
    } else {
        trace.stop_reason = STOP_EXHAUSTED.into();
    }
    trace
}

/// What the k-oracle returns: the minimum expansion sequence it found, whether
/// branch-and-bound proved it minimal inside the budget, and the admissible
/// sandwich — `lower_bound` the largest single-target optimum, `upper_bound`
/// the size of the union of one minimum expansion set per target, which is
/// itself feasible. `searched` counts the states expanded.
#[derive(Clone, Debug)]
pub struct KOracle {
    pub trace: WalkTrace,
    pub exact: bool,
    pub lower_bound: u32,
    pub upper_bound: u32,
    pub searched: u64,
    pub pruned_nodes: usize,
}

/// §3 item 3: the **minimum expansion set** examining all k, exact by
/// branch-and-bound over expansion orders on the ball pruned to the nodes on
/// any surviving path to any target.
///
/// The state is the expanded set, which must be reachable from `s`; the cost
/// of a state is its size (the start counts, as `expansions` does); the search
/// is best-first under the admissible bound `h(S) = max over unregistered t of
/// (dist(S, t) − 1)`, distances taken in the pruned graph. Over
/// `ORACLE_STATE_BUDGET` states or `ORACLE_TIME_LIMIT`, or on a pruned ball too
/// wide for the state word, the search stops, `exact` is false and the
/// **feasible** upper bound's own order is returned — never called an oracle.
pub fn k_oracle(g: &EpisodeGraph) -> KOracle {
    let k = g.target_count();
    // V': the nodes on any surviving path to any target, plus the start
    let mut allowed: HashSet<&str> = g
        .surviving_paths
        .iter()
        .flat_map(|p| p.iter().map(String::as_str))
        .collect();
    allowed.insert(g.start.as_str());
    let mut nodes: Vec<&str> = allowed.into_iter().collect();
    nodes.sort_unstable();
    let idx: HashMap<&str, usize> = nodes.iter().enumerate().map(|(i, n)| (*n, i)).collect();
    let n = nodes.len();
    let succ: Vec<Vec<usize>> = nodes
        .iter()
        .map(|x| {
            g.tails(x)
                .iter()
                .filter_map(|t| idx.get(t.as_str()).copied())
                .collect::<HashSet<usize>>()
                .into_iter()
                .collect::<Vec<usize>>()
        })
        .map(|mut v| {
            v.sort_unstable();
            v
        })
        .collect();
    // a target may sit outside V' only if it has no surviving path at all
    let targets: Vec<Option<usize>> = g
        .targets
        .iter()
        .map(|t| idx.get(t.as_str()).copied())
        .collect();
    let start = idx[g.start.as_str()];
    // distances in the pruned graph, from every node to every target
    let dist: Vec<Vec<Option<u32>>> = targets
        .iter()
        .map(|t| match t {
            Some(t) => distances_to(*t, &succ, n),
            None => vec![None; n],
        })
        .collect();

    // the sandwich, computed whatever the search does
    let per_target: Vec<Option<Vec<usize>>> = (0..k)
        .map(|t| single_target_set(start, &succ, &dist[t]))
        .collect();
    let lower_bound = per_target
        .iter()
        .map(|s| s.as_ref().map_or(0, |v| v.len() as u32))
        .max()
        .unwrap_or(0);
    let mut union: HashSet<usize> = HashSet::new();
    let mut feasible = true;
    for s in &per_target {
        match s {
            Some(v) => union.extend(v.iter().copied()),
            None => feasible = false,
        }
    }
    let upper_bound = if feasible { union.len() as u32 } else { 0 };

    let fallback = |exact: bool, searched: u64| -> KOracle {
        let order = if feasible {
            reachable_order(start, &union, &succ)
        } else {
            Vec::new()
        };
        let trace = replay(g, &nodes, &order);
        // the feasible order is itself an upper bound, and replaying it may
        // register everything before the union is exhausted
        let upper_bound = if feasible {
            upper_bound.min(trace.expansions)
        } else {
            upper_bound
        };
        KOracle {
            trace,
            exact,
            lower_bound,
            upper_bound,
            searched,
            pruned_nodes: n,
        }
    };
    if !feasible || n > 128 {
        return fallback(false, 0);
    }

    // best-first branch-and-bound over expanded sets
    let deadline = std::time::Instant::now() + crate::ORACLE_TIME_LIMIT;
    // the nodes with an edge to each target: a target registers exactly when
    // the expanded set meets its parent mask
    let parents_of: Vec<u128> = (0..k)
        .map(|t| match targets[t] {
            Some(t) => (0..n).fold(0u128, |m, x| {
                if succ[x].contains(&t) {
                    m | 1u128 << x
                } else {
                    m
                }
            }),
            None => 0,
        })
        .collect();
    let registered = |s: u128| -> Vec<bool> { (0..k).map(|t| s & parents_of[t] != 0).collect() };
    let h = |s: u128, reg: &[bool]| -> Option<u32> {
        let mut worst = 0;
        for t in 0..k {
            if reg[t] {
                continue;
            }
            let d = (0..n)
                .filter(|x| s >> x & 1 == 1)
                .filter_map(|x| dist[t][x])
                .min()?;
            worst = worst.max(d.saturating_sub(1));
        }
        Some(worst)
    };
    let s0 = 1u128 << start;
    let reg0 = registered(s0);
    let Some(h0) = h(s0, &reg0) else {
        return fallback(false, 0);
    };
    let mut heap: BinaryHeap<std::cmp::Reverse<(u32, u32, u128)>> = BinaryHeap::new();
    heap.push(std::cmp::Reverse((1 + h0, 1, s0)));
    let mut came: HashMap<u128, (u128, usize)> = HashMap::new();
    let mut seen: HashSet<u128> = HashSet::from([s0]);
    let mut searched = 0u64;
    while let Some(std::cmp::Reverse((_, g_cost, state))) = heap.pop() {
        searched += 1;
        if searched.is_multiple_of(256) && std::time::Instant::now() > deadline {
            return fallback(false, searched);
        }
        if searched > crate::ORACLE_STATE_BUDGET {
            return fallback(false, searched);
        }
        let reg = registered(state);
        if reg.iter().all(|r| *r) {
            let order = unwind(state, start, &came);
            return KOracle {
                trace: replay(g, &nodes, &order),
                exact: true,
                lower_bound,
                upper_bound,
                searched,
                pruned_nodes: n,
            };
        }
        // successors: any not-yet-expanded node reachable from the set
        let mut next: Vec<usize> = Vec::new();
        for (x, tails) in succ.iter().enumerate() {
            if state >> x & 1 == 0 {
                continue;
            }
            for y in tails {
                if state >> *y & 1 == 0 && !next.contains(y) {
                    next.push(*y);
                }
            }
        }
        next.sort_unstable();
        for y in next {
            let child = state | 1u128 << y;
            if !seen.insert(child) {
                continue;
            }
            came.insert(child, (state, y));
            let creg = registered(child);
            let Some(hc) = h(child, &creg) else { continue };
            heap.push(std::cmp::Reverse((g_cost + 1 + hc, g_cost + 1, child)));
        }
    }
    fallback(false, searched)
}

/// The k-oracle's trace alone (§3 item 3); `k_oracle` carries the bounds.
pub fn k_oracle_trace(g: &EpisodeGraph) -> WalkTrace {
    k_oracle(g).trace
}

/// BFS distances to `target` over the reversed pruned adjacency.
fn distances_to(target: usize, succ: &[Vec<usize>], n: usize) -> Vec<Option<u32>> {
    let mut inc: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (x, tails) in succ.iter().enumerate() {
        for y in tails {
            inc[*y].push(x);
        }
    }
    let mut d = vec![None; n];
    d[target] = Some(0);
    let mut queue = VecDeque::from([target]);
    while let Some(x) = queue.pop_front() {
        let dx = d[x].unwrap();
        for y in &inc[x] {
            if d[*y].is_none() {
                d[*y] = Some(dx + 1);
                queue.push_back(*y);
            }
        }
    }
    d
}

/// One minimum expansion set registering a single target: the nodes of a
/// shortest `start → parent(t)` walk, the tie broken by the lowest index at
/// every step, so the set is a function of the graph and of nothing else. It
/// has `d(start, t)` members, the M1 clarification's `o(t)`.
fn single_target_set(
    start: usize,
    succ: &[Vec<usize>],
    dist: &[Option<u32>],
) -> Option<Vec<usize>> {
    let mut d = dist[start]?;
    if d == 0 {
        return Some(Vec::new()); // the start IS the target: registered before any expansion
    }
    let mut set = vec![start];
    let mut node = start;
    while d > 1 {
        let next = *succ[node]
            .iter()
            .filter(|y| dist[**y] == Some(d - 1))
            .min()?;
        set.push(next);
        node = next;
        d -= 1;
    }
    Some(set)
}

/// An expansion order for a reachable set: repeatedly take the lowest-indexed
/// member reachable from what is already expanded.
fn reachable_order(start: usize, set: &HashSet<usize>, succ: &[Vec<usize>]) -> Vec<usize> {
    let mut order = vec![start];
    let mut have: HashSet<usize> = HashSet::from([start]);
    loop {
        let next = set
            .iter()
            .filter(|y| !have.contains(y))
            .filter(|y| order.iter().any(|x| succ[*x].contains(y)))
            .min()
            .copied();
        match next {
            Some(y) => {
                order.push(y);
                have.insert(y);
            }
            None => break,
        }
    }
    order
}

/// The expansion order that reached `state`, from the search's parent links.
fn unwind(state: u128, start: usize, came: &HashMap<u128, (u128, usize)>) -> Vec<usize> {
    let mut order = Vec::new();
    let mut s = state;
    while let Some((parent, added)) = came.get(&s) {
        order.push(*added);
        s = *parent;
    }
    order.push(start);
    order.reverse();
    order
}

/// Turn an expansion order into a `WalkTrace`, registering on sight.
fn replay(g: &EpisodeGraph, nodes: &[&str], order: &[usize]) -> WalkTrace {
    let mut trace = WalkTrace::with_targets(g.target_count());
    let mut reg = Registry::new(g.target_count());
    let mut seen: HashSet<&str> = HashSet::from([g.start.as_str()]);
    for (t, target) in g.targets.iter().enumerate() {
        if g.start == *target {
            reg.see(t, 0);
        }
    }
    for x in order {
        let node = nodes[*x];
        trace.expansions += 1;
        trace.examined.push(node.to_string());
        for tail in g.tails(node) {
            if let Some(t) = g.targets.iter().position(|x| x == tail) {
                reg.see(t, trace.expansions);
            }
            if seen.insert(tail.as_str()) {
                trace.parents.insert(tail.clone(), node.to_string());
            }
        }
        if reg.all() {
            break;
        }
    }
    finish(&mut trace, reg);
    trace
}

/// Every k-target baseline on one episode, in `K_POLICY_NAMES` order.
pub fn all_k_traces(
    g: &EpisodeGraph,
    embeddings: &dyn Embeddings,
) -> Vec<(&'static str, WalkTrace)> {
    vec![
        ("blind_exhaust", k_blind_exhaust_trace(g)),
        (
            "bidirectional_sequential",
            bidirectional_sequential_trace(g),
        ),
        ("k_greedy", k_greedy_trace(g, embeddings)),
        ("k_greedy_frozen", k_greedy_frozen_trace(g, embeddings)),
        ("k_oracle", k_oracle_trace(g)),
    ]
}
