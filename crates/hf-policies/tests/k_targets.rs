//! The k-target baselines of `K_TARGETS_DESIGN.md` §3: the k = 1 identities
//! against v1's own traces on every fixture episode, the cost unit the note's
//! M1 clarification fixes, the k-oracle against an independently written brute
//! force on small graphs, and the recompute rule on a hand-built case where
//! freezing the keys and recomputing them genuinely differ.
//!
//! Never real data: the fixture split and hand-built graphs only.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use hf_policies::{
    all_k_traces, all_traces, bidirectional_bfs_trace, bidirectional_sequential_trace,
    blind_exhaust_trace, k_blind_exhaust_trace, k_greedy_frozen_trace, k_greedy_trace, k_oracle,
    policy_row, similarity_greedy_trace, EpisodeGraph, WalkTrace, K_POLICY_NAMES,
};
use serde_json::Value;

fn fixture() -> (Vec<hf_io::RealEpisode>, HashMap<String, Vec<f64>>) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let golden: Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("tests/goldens/policies.json")).unwrap(),
    )
    .unwrap();
    let embeddings: HashMap<String, Vec<f64>> = golden["embeddings"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap())
                    .collect(),
            )
        })
        .collect();
    let splits = root.join("../hf-io/tests/goldens/fixture-split");
    let mut episodes = Vec::new();
    for name in ["train", "screen", "train-greedy"] {
        episodes.extend(hf_io::read_split(&splits.join(name)).unwrap().0);
    }
    (episodes, embeddings)
}

fn same(a: &WalkTrace, b: &WalkTrace, what: &str) {
    assert_eq!(a.examined, b.examined, "{what}: expansion order");
    assert_eq!(a.expansions, b.expansions, "{what}: expansions");
    assert_eq!(a.registered_at, b.registered_at, "{what}: registered_at");
    assert_eq!(a.stop_reason, b.stop_reason, "{what}: stop reason");
    assert_eq!(a.parents, b.parents, "{what}: parents");
    assert_eq!(
        a.registered_at_by_target, b.registered_at_by_target,
        "{what}: per-target registration"
    );
}

/// At k = 1 every k-baseline is its v1 baseline, trace field for trace field,
/// on every fixture episode — two independently written walks, not one walk
/// called twice: `k_forward_walk` and `k_greedy_walk` are written out in
/// `ktargets.rs` rather than delegating to `forward_walk`, so this test can
/// fail.
#[test]
fn at_one_target_every_k_baseline_is_its_v1_baseline() {
    let (episodes, embeddings) = fixture();
    let mut checked = 0;
    for episode in &episodes {
        let g = EpisodeGraph::from_episode_k(episode);
        assert_eq!(g.target_count(), 1, "the fixture split is k = 1");
        let v1 = EpisodeGraph::from_episode(episode);
        same(
            &k_blind_exhaust_trace(&g),
            &blind_exhaust_trace(&v1),
            "blind",
        );
        same(
            &k_greedy_trace(&g, &embeddings),
            &similarity_greedy_trace(&v1, &embeddings, None),
            "k-greedy",
        );
        same(
            &k_greedy_frozen_trace(&g, &embeddings),
            &similarity_greedy_trace(&v1, &embeddings, None),
            "k-greedy-frozen",
        );
        same(
            &bidirectional_sequential_trace(&g),
            &bidirectional_bfs_trace(&v1),
            "bidirectional sequential",
        );
        // and the row a k = 1 episode writes is v5's, key for key: the three k
        // fields are absent, which the policy goldens depend on
        for (_, trace) in all_traces(&v1, &embeddings) {
            let row = serde_json::to_value(policy_row(&v1, &trace)).unwrap();
            let keys: Vec<&str> = row
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect();
            assert_eq!(
                keys,
                [
                    "registered",
                    "expansions",
                    "expansions_at_registration",
                    "examined",
                    "stop_reason",
                    "target_distance",
                    "removal_level",
                    "removed_count",
                ],
                "a k = 1 row carries no k field"
            );
        }
        checked += 1;
    }
    assert_eq!(checked, 30, "12 + 6 + 12 fixture episodes");
}

/// The cost unit the M1 clarification fixes: the start is expanded first and
/// counts, a target registers on sight, so the fewest expansions registering a
/// target at distance 3 is 3 — `s → a → b`, with `b → t`.
#[test]
fn the_minimum_expansion_set_of_one_target_at_distance_three_has_three_members() {
    let edges = [("s", "a"), ("a", "b"), ("b", "t")];
    let g = EpisodeGraph::new_k(
        "s",
        vec!["t".into()],
        edges
            .iter()
            .map(|(a, b)| ((*a).to_string(), (*b).to_string())),
        vec![vec!["s".into(), "a".into(), "b".into(), "t".into()]],
        HashMap::new(),
    );
    let oracle = k_oracle(&g);
    assert!(oracle.exact);
    assert_eq!(oracle.trace.examined, ["s", "a", "b"]);
    assert_eq!(oracle.trace.expansions, 3);
    assert_eq!(oracle.trace.registered_at, Some(3));
    assert_eq!(oracle.lower_bound, 3);
    assert_eq!(oracle.upper_bound, 3);
}

/// Two targets sharing a parent: one expansion registers both, and the joint
/// optimum is strictly under the sequential sum — the decision the substrate
/// is about.
#[test]
fn the_k_oracle_shares_the_work_between_two_targets() {
    let edges = [
        ("s", "a"),
        ("a", "b"),
        ("b", "t1"),
        ("b", "t2"),
        ("s", "c"),
        ("c", "d"),
        ("d", "t2"),
    ];
    let g = EpisodeGraph::new_k(
        "s",
        vec!["t1".into(), "t2".into()],
        edges
            .iter()
            .map(|(a, b)| ((*a).to_string(), (*b).to_string())),
        vec![
            vec!["s".into(), "a".into(), "b".into(), "t1".into()],
            vec!["s".into(), "a".into(), "b".into(), "t2".into()],
            vec!["s".into(), "c".into(), "d".into(), "t2".into()],
        ],
        HashMap::new(),
    );
    let oracle = k_oracle(&g);
    assert!(oracle.exact);
    assert_eq!(oracle.trace.examined, ["s", "a", "b"]);
    assert_eq!(oracle.trace.registered_targets(), 2);
    assert_eq!(oracle.lower_bound, 3, "each target alone costs 3");
}

/// A deterministic small-graph generator: a tiny LCG, never a real graph.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self, n: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (self.0 >> 33) % n
    }
}

/// Brute force, written independently of the branch-and-bound: the smallest
/// set of pruned nodes containing the start, orderable so that every member
/// but the start is a child of an earlier member, and meeting every target's
/// parent set. Exhaustive over subsets — hence the 12-node ceiling.
fn brute_force(g: &EpisodeGraph) -> Option<u32> {
    let mut nodes: Vec<&str> = g
        .surviving_paths
        .iter()
        .flat_map(|p| p.iter().map(String::as_str))
        .collect::<HashSet<&str>>()
        .into_iter()
        .collect();
    if !nodes.contains(&g.start.as_str()) {
        nodes.push(g.start.as_str());
    }
    nodes.sort_unstable();
    let n = nodes.len();
    assert!(n <= 12, "brute force is exhaustive");
    let child_of = |x: &str, y: &str| g.out.get(x).is_some_and(|t| t.iter().any(|v| v == y));
    let mut best: Option<u32> = None;
    for mask in 0u32..(1 << n) {
        let set: Vec<&str> = (0..n)
            .filter(|i| mask >> i & 1 == 1)
            .map(|i| nodes[i])
            .collect();
        if !set.contains(&g.start.as_str()) {
            continue;
        }
        if best.is_some_and(|b| set.len() as u32 >= b) {
            continue;
        }
        // orderable from the start?
        let mut have: Vec<&str> = vec![g.start.as_str()];
        loop {
            let next = set
                .iter()
                .find(|y| !have.contains(y) && have.iter().any(|x| child_of(x, y)))
                .copied();
            match next {
                Some(y) => have.push(y),
                None => break,
            }
        }
        if have.len() != set.len() {
            continue;
        }
        // every target seen as a child of some member?
        if g.targets.iter().all(|t| set.iter().any(|x| child_of(x, t))) {
            best = Some(set.len() as u32);
        }
    }
    best
}

/// On random digraphs of at most twelve nodes the branch-and-bound's exact
/// answer is the brute force's, and its trace really registers every target in
/// that many expansions.
#[test]
fn the_k_oracle_equals_brute_force_on_small_graphs() {
    let mut rng = Lcg(20_260_919);
    let mut compared = 0;
    let mut with_two = 0;
    for _ in 0..600 {
        let n = 6 + rng.next(7) as usize; // 6..=12
        let names: Vec<String> = (0..n).map(|i| format!("n{i}")).collect();
        let mut edges: Vec<(String, String)> = Vec::new();
        for a in 0..n {
            for b in 0..n {
                if a != b && rng.next(100) < 22 {
                    edges.push((names[a].clone(), names[b].clone()));
                }
            }
        }
        let start = names[0].clone();
        // every simple path from the start, so the pruned ball is honest
        let mut out: HashMap<&str, Vec<&str>> = HashMap::new();
        for (a, b) in &edges {
            out.entry(a.as_str()).or_default().push(b.as_str());
        }
        let mut paths: Vec<Vec<String>> = Vec::new();
        let mut stack = vec![vec![start.as_str()]];
        while let Some(path) = stack.pop() {
            if path.len() > 5 {
                continue;
            }
            paths.push(path.iter().map(|s| (*s).to_string()).collect());
            for y in out
                .get(path[path.len() - 1])
                .map(Vec::as_slice)
                .unwrap_or(&[])
            {
                if !path.contains(y) {
                    let mut next = path.clone();
                    next.push(y);
                    stack.push(next);
                }
            }
        }
        let reachable: HashSet<&str> = paths
            .iter()
            .flat_map(|p| p.iter().map(String::as_str))
            .filter(|x| *x != start)
            .collect();
        if reachable.len() < 2 {
            continue;
        }
        let mut choices: Vec<&str> = reachable.into_iter().collect();
        choices.sort_unstable();
        let t1 = choices[rng.next(choices.len() as u64) as usize].to_string();
        let t2 = choices[rng.next(choices.len() as u64) as usize].to_string();
        if t1 == t2 {
            continue;
        }
        let mut targets = vec![t1, t2];
        targets.sort();
        // keep only the paths ending at a target — the surviving-path union
        let surviving: Vec<Vec<String>> = paths
            .iter()
            .filter(|p| targets.contains(p.last().unwrap()))
            .cloned()
            .collect();
        if surviving.is_empty() {
            continue;
        }
        let pruned = surviving
            .iter()
            .flatten()
            .collect::<HashSet<&String>>()
            .len();
        if pruned > 12 {
            continue;
        }
        let g = EpisodeGraph::new_k(
            &start,
            targets.clone(),
            edges.iter().cloned(),
            surviving,
            HashMap::new(),
        );
        let want = brute_force(&g);
        let oracle = k_oracle(&g);
        match want {
            Some(best) => {
                assert!(oracle.exact, "the budget cannot bind on {pruned} nodes");
                assert_eq!(
                    oracle.trace.expansions, best,
                    "the branch-and-bound found {} where brute force found {best}",
                    oracle.trace.expansions
                );
                assert_eq!(
                    oracle.trace.registered_targets(),
                    2,
                    "the minimum set must register both"
                );
                assert!(
                    oracle.lower_bound <= best && best <= oracle.upper_bound,
                    "the sandwich {}..{} must hold the optimum {best}",
                    oracle.lower_bound,
                    oracle.upper_bound
                );
                with_two += 1;
            }
            None => assert!(
                !oracle.exact || oracle.trace.registered_targets() < 2,
                "brute force found nothing feasible but the oracle claims a set"
            ),
        }
        compared += 1;
    }
    assert!(compared > 100, "only {compared} graphs compared");
    assert!(with_two > 50, "only {with_two} feasible graphs");
}

/// §3 item 1's recompute rule, on the case it exists for: freezing the keys
/// leaves the walk climbing the target it already has.
///
/// `s → {a, t1}`, `a → b`, `b → t1` and `s → c`, `c → t2`. `t1` registers on
/// the first expansion. `a` is nearer `t1` than `c` is to anything, so with
/// frozen keys the walk expands `a` and `b` before `c`; recomputed, the moment
/// `t1` registers every key is a cosine to `t2` alone and `c` comes first.
#[test]
fn recomputing_the_keys_on_registration_changes_the_walk() {
    let edges = [
        ("s", "a"),
        ("s", "t1"),
        ("s", "c"),
        ("a", "b"),
        ("b", "t1"),
        ("c", "t2"),
    ];
    let embeddings: HashMap<String, Vec<f64>> = [
        ("s", vec![0.0, 0.0, 1.0]),
        ("a", vec![0.95, 0.05, 0.0]),
        ("b", vec![0.9, 0.1, 0.0]),
        ("c", vec![0.3, 0.7, 0.0]),
        ("t1", vec![1.0, 0.0, 0.0]),
        ("t2", vec![0.0, 1.0, 0.0]),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    let g = EpisodeGraph::new_k(
        "s",
        vec!["t1".into(), "t2".into()],
        edges
            .iter()
            .map(|(a, b)| ((*a).to_string(), (*b).to_string())),
        vec![
            vec!["s".into(), "t1".into()],
            vec!["s".into(), "c".into(), "t2".into()],
        ],
        HashMap::new(),
    );
    let live = k_greedy_trace(&g, &embeddings);
    let frozen = k_greedy_frozen_trace(&g, &embeddings);
    assert_eq!(live.registered_at_by_target[0], Some(1), "t1 on sight");
    assert_eq!(frozen.registered_at_by_target[0], Some(1));
    assert_eq!(
        live.examined,
        ["s", "c"],
        "recomputed: the key turns to t2 the moment t1 registers"
    );
    assert_eq!(
        frozen.examined,
        ["s", "a", "c"],
        "frozen: `a`'s stale key still points at the target already registered"
    );
    assert!(
        live.expansions < frozen.expansions,
        "the two rules must differ here, or the case is not the case"
    );
    assert_eq!(live.registered_at, Some(2));
    assert_eq!(frozen.registered_at, Some(3));
}

/// The row at k >= 2 carries the three k fields, recall over k lands on the
/// k + 1 values a k-target episode admits, and the budget is a parameter.
#[test]
fn the_k_row_carries_recall_over_k_at_the_budget() {
    let edges = [("s", "a"), ("a", "t1"), ("a", "b"), ("b", "c"), ("c", "t2")];
    let g = EpisodeGraph::new_k(
        "s",
        vec!["t1".into(), "t2".into()],
        edges
            .iter()
            .map(|(a, b)| ((*a).to_string(), (*b).to_string())),
        vec![
            vec!["s".into(), "a".into(), "t1".into()],
            vec!["s".into(), "a".into(), "b".into(), "c".into(), "t2".into()],
        ],
        HashMap::new(),
    );
    // t1 registers when `a` is expanded; `t1` then sits on the frontier like
    // any other child and is expanded in its turn, exactly as the learned
    // walk's `push_children` pushes a registered target — so `t2` registers at
    // 5, not 4, and the k walk's cost unit is the walk's own
    let trace = k_blind_exhaust_trace(&g);
    assert_eq!(trace.examined, ["s", "a", "b", "t1", "c"]);
    assert_eq!(trace.registered_at_by_target, [Some(2), Some(5)]);
    for (budget, want) in [
        (0u32, 0.0),
        (1, 0.0),
        (2, 0.5),
        (3, 0.5),
        (4, 0.5),
        (5, 1.0),
    ] {
        assert_eq!(
            trace.recall_at_budget(budget),
            Some(want),
            "at B = {budget}"
        );
    }
    let row = serde_json::to_value(hf_policies::policy_row_at(&g, &trace, 2)).unwrap();
    assert_eq!(row["registered_targets"], 2);
    assert_eq!(row["recall_at_budget"], 0.5);
    assert_eq!(row["registered_at"], serde_json::json!([2, 5]));
    assert_eq!(
        row["registered"], true,
        "all k registered by the walk's stop"
    );
    // the default budget is B_fix = n / 2 over the ball the graph shows
    assert_eq!(g.b_fix(), g.subgraph_size / 2);
}

/// Every k baseline reports under its own name, and each registers both
/// targets on a graph where both are reachable.
#[test]
fn every_k_baseline_registers_both_targets_when_both_are_reachable() {
    let edges = [("s", "a"), ("a", "t1"), ("a", "b"), ("b", "t2")];
    let embeddings: HashMap<String, Vec<f64>> = [
        ("s", vec![0.0, 1.0]),
        ("a", vec![0.6, 0.8]),
        ("b", vec![0.8, 0.6]),
        ("t1", vec![1.0, 0.0]),
        ("t2", vec![0.9, 0.436]),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    let g = EpisodeGraph::new_k(
        "s",
        vec!["t1".into(), "t2".into()],
        edges
            .iter()
            .map(|(a, b)| ((*a).to_string(), (*b).to_string())),
        vec![
            vec!["s".into(), "a".into(), "t1".into()],
            vec!["s".into(), "a".into(), "b".into(), "t2".into()],
        ],
        HashMap::new(),
    );
    let traces = all_k_traces(&g, &embeddings);
    let names: Vec<&str> = traces.iter().map(|(n, _)| *n).collect();
    assert_eq!(names, K_POLICY_NAMES);
    for (name, trace) in &traces {
        assert_eq!(trace.registered_targets(), 2, "{name} registers both");
        assert_eq!(
            trace.recall_at_budget(trace.expansions),
            Some(1.0),
            "{name} recall at its own stop"
        );
        assert!(trace.registered_at.is_some(), "{name} completes");
    }
}

/// `B_fix = n / 2` takes `n` from the RUNG — the sampler block's
/// `subgraph_size`, 40 at rung 3 and 80 at rung 4, giving §4's stated 20 and
/// 40 — and not from the realised ball, which the region filter and the hub
/// cap leave smaller on a minority of episodes. Reading the visible
/// `subgraph_size` instead would make the budget, and so the strata, a
/// function of ball size.
#[test]
fn the_fixed_budget_is_the_rungs_n_and_not_the_realised_ball() {
    let (episodes, _) = fixture();
    let mut checked = 0;
    for episode in &episodes {
        let n = episode.hidden.sampler["subgraph_size"].as_u64().unwrap() as u32;
        let g = EpisodeGraph::from_episode(episode);
        assert_eq!(g.subgraph_size, n);
        assert_eq!(g.b_fix(), n / 2);
        // shrink the realised ball: the budget must not move
        let mut smaller = episode.clone();
        smaller.visible.subgraph_size = 6;
        assert_eq!(EpisodeGraph::from_episode(&smaller).b_fix(), n / 2);
        assert_eq!(EpisodeGraph::from_episode_k(&smaller).b_fix(), n / 2);
        checked += 1;
    }
    assert_eq!(checked, 30);
}
