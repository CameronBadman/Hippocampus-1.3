//! The k = 2 draw and the joint-cut removal (`K_TARGETS_DESIGN.md` §1), on the
//! fixture world and on hand-built graphs — never on real data.
//!
//! The v1 identity is `tests/goldens.rs`'s: these tests only assert what k = 2
//! adds, and one of them asserts that adding it changed neither the draw key
//! nor the sampler block at `targets = 1`.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use hf_episodes::fixture::fixture_world;
use hf_episodes::{
    choose_joint_greedy_removals, sample_split, Dropped, Sampled, Sampler, SamplerConfig,
};
use serde_json::Value;

const KNOWN_DROPS: [&str; 10] = [
    "subgraph_too_small",
    "no_target_at_distance_in_split",
    "no_path_within_bound",
    "path_set_over_cap",
    "greedy_route_missing",
    "survivors_mismatch",
    "no_second_target_at_distance",
    "targets_interdependent",
    "removal_left_no_path",
    "survivor_not_recovered",
];

fn k2_config(rule: &str) -> SamplerConfig {
    let mut c = SamplerConfig::new("fixture", 64, 3, 2);
    c.cost_epsilon = 0.5;
    c.targets = 2;
    c.removal_rule = rule.into();
    c
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .expect("an array")
        .iter()
        .map(|s| s.as_str().expect("a string").to_string())
        .collect()
}

fn paths(v: &Value) -> Vec<Vec<String>> {
    v.as_array()
        .expect("an array")
        .iter()
        .map(strings)
        .collect()
}

fn sample(rule: &str, split: &str, count: usize) -> Vec<Sampled> {
    let (graph, embeddings) = fixture_world(5, 400, 1600, 8);
    let greedy = rule == "greedy-path";
    let mut sampler = Sampler::new(&graph, k2_config(rule)).unwrap();
    sampler.prepare(split).unwrap();
    sample_split(
        &sampler,
        split,
        count,
        if greedy { Some(&embeddings) } else { None },
        16,
    )
    .unwrap()
    .episodes
}

/// Every kept episode shows two distinct targets, each at exactly `d` in the
/// unpruned ball, and keeps at least one surviving path to EACH of them after
/// the joint cut; the unions and the per-target maps agree.
#[test]
fn every_k2_episode_carries_two_targets_and_a_surviving_path_to_each() {
    for rule in ["cheapest-first", "greedy-path"] {
        let episodes = sample(rule, "train", 40);
        assert_eq!(episodes.len(), 40, "{rule}: the fixture fills the split");
        for e in &episodes {
            let (v, h) = (&e.visible, &e.hidden);
            let targets = strings(&h["target_set"]);
            assert_eq!(targets.len(), 2, "{rule}: two targets");
            assert!(targets[0] < targets[1], "{rule}: T is in node-name order");
            // the visible side shows both, and a v1 reader still sees one
            assert_eq!(v["schema_version"], "6.0.0");
            assert_eq!(h["schema_version"], "6.0.0");
            assert_eq!(v["target_node"].as_str().unwrap(), targets[0]);
            assert_eq!(strings(&v["target_nodes"]), targets);
            assert!(e.episode_id.contains("-t2"), "the id carries the k marker");

            let bound = h["cost_bound"].as_u64().unwrap() as usize;
            let d = h["target_distance"].as_u64().unwrap() as usize;
            let path_set = paths(&h["path_set"]);
            let surviving = paths(&h["surviving_paths"]);
            let visible_edges: HashSet<(String, String)> = v["edges"]
                .as_array()
                .unwrap()
                .iter()
                .map(|e| {
                    (
                        e["source"].as_str().unwrap().to_string(),
                        e["target"].as_str().unwrap().to_string(),
                    )
                })
                .collect();
            for target in &targets {
                // the draw put both targets at exactly d, so the unpruned path
                // set holds a d-edge path to each and none shorter
                let to_target: Vec<&Vec<String>> = path_set
                    .iter()
                    .filter(|p| p.last() == Some(target))
                    .collect();
                assert!(!to_target.is_empty(), "{rule}: a path set for {target}");
                let shortest = to_target.iter().map(|p| p.len() - 1).min().unwrap();
                assert_eq!(shortest, d, "{rule}: {target} sits at exactly d");
                assert!(to_target.iter().all(|p| p.len() - 1 <= bound));
                // and the joint cut left at least one path to it standing
                let kept: Vec<&Vec<String>> = surviving
                    .iter()
                    .filter(|p| p.last() == Some(target))
                    .collect();
                assert!(
                    !kept.is_empty(),
                    "{rule}: {} kept no path to {target}",
                    e.episode_id
                );
                for p in kept {
                    assert!(path_set.contains(p), "a survivor comes from the path set");
                    for w in p.windows(2) {
                        assert!(
                            visible_edges.contains(&(w[0].clone(), w[1].clone())),
                            "{rule}: a surviving path uses a pruned edge"
                        );
                    }
                }
            }
            // the unions are unions
            let mut want: Vec<String> = surviving.iter().flatten().cloned().collect();
            want.sort_unstable();
            want.dedup();
            assert_eq!(
                strings(&h["nodes_on_surviving_path"]),
                want,
                "{rule}: nodes_on_surviving_path is the deduped union"
            );
            let removal_set = paths(&h["removal_set"]);
            assert_eq!(
                removal_set.len(),
                h["removed_count"].as_u64().unwrap() as usize
            );
            for pair in &removal_set {
                assert!(
                    !visible_edges.contains(&(pair[0].clone(), pair[1].clone())),
                    "{rule}: a removed edge is still visible"
                );
            }
            // distance_to_target is the per-node minimum over the per-target maps
            let by_target: BTreeMap<String, BTreeMap<String, u32>> =
                serde_json::from_value(h["distance_to_targets"].clone()).unwrap();
            assert_eq!(
                by_target.keys().cloned().collect::<Vec<_>>(),
                targets,
                "one distance map per target"
            );
            let mut minimum: BTreeMap<String, u32> = BTreeMap::new();
            for map in by_target.values() {
                for (n, dist) in map {
                    let cell = minimum.entry(n.clone()).or_insert(*dist);
                    *cell = (*cell).min(*dist);
                }
            }
            let stored: BTreeMap<String, u32> =
                serde_json::from_value(h["distance_to_target"].clone()).unwrap();
            assert_eq!(stored, minimum, "{rule}: distance_to_target is the minimum");
            for target in &targets {
                assert_eq!(stored.get(target), Some(&0));
            }
        }
    }
}

/// The greedy arm's per-target overshoots, and the scalar as their maximum.
#[test]
fn greedy_overshoot_is_the_maximum_of_the_per_target_overshoots() {
    for e in sample("greedy-path", "train", 20) {
        let h = &e.hidden;
        let each: Vec<i64> = h["greedy_overshoots"]
            .as_array()
            .expect("a list per target")
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        assert_eq!(each.len(), 2);
        // through the typed reader too: a mistyped field would land in `extra`,
        // re-serialize byte for byte, and leave every §6 item 5 reader None
        let typed: hf_io::Hidden = serde_json::from_value(h.clone()).unwrap();
        assert_eq!(typed.greedy_overshoots.as_deref(), Some(&each[..]));
        assert_eq!(typed.greedy_overshoot, Some(*each.iter().max().unwrap()));
        assert!(typed.distance_to_targets.is_some());
        assert!(
            typed.extra.is_empty(),
            "a k = 2 field fell through to extra"
        );
        assert_eq!(
            h["greedy_overshoot"].as_i64().unwrap(),
            *each.iter().max().unwrap()
        );
        assert_eq!(h["removal_recipe"], "greedy-path");
    }
}

/// The second draw refuses a candidate that lies on a bounded path to the first
/// — §1's rejection 2, on the cost bound and not on shortest paths.
///
/// `s → a → b → t1`, `b → c` and `t1 → c`: both `t1` and `c` sit at distance 3,
/// and `s → a → b → t1 → c` is four edges, the bound, with `t1` interior. No
/// SHORTEST path to `c` passes through `t1` (`s → a → b → c` is three), so the
/// weaker shortest-path rule would keep this episode; §1's rule drops it.
#[test]
fn the_second_draw_refuses_a_target_on_a_bounded_path_to_the_first() {
    let graph = hf_graph::RealGraph::from_edges(
        "ktest",
        [
            ("s", None, "a"),
            ("a", None, "b"),
            ("b", None, "t1"),
            ("b", None, "c"),
            ("t1", None, "c"),
        ],
    );
    let mut config = SamplerConfig::new("ktest", 10, 3, 2);
    config.targets = 2;
    let mut sampler = Sampler::new(&graph, config).unwrap();
    sampler.prepare("train").unwrap();
    let mut drops: BTreeMap<&str, u32> = BTreeMap::new();
    for index in 0..40 {
        match sampler.sample("train", index, None).unwrap() {
            Ok(e) => panic!("kept {} on an interdependent pair", e.episode_id),
            Err(reason) => *drops.entry(reason.as_str()).or_default() += 1,
        }
    }
    assert!(
        drops.get("targets_interdependent").copied().unwrap_or(0) > 0,
        "the interdependence rejection never fired: {drops:?}"
    );
    // the same graph keeps episodes at k = 1, so the drop is the k = 2 rule's
    let mut single = SamplerConfig::new("ktest", 10, 3, 2);
    single.targets = 1;
    let mut sampler = Sampler::new(&graph, single).unwrap();
    sampler.prepare("train").unwrap();
    let kept = (0..40)
        .filter(|i| sampler.sample("train", *i, None).unwrap().is_ok())
        .count();
    assert!(kept > 0, "k = 1 keeps what k = 2 refuses");
}

/// Every drop is one of the ten reasons, the tally accounts for every attempt,
/// and the k = 2 draw's own reasons are counted under their own names.
#[test]
fn the_k2_drop_reasons_are_counted() {
    let (graph, _) = fixture_world(5, 400, 1600, 8);
    let mut sampler = Sampler::new(&graph, k2_config("cheapest-first")).unwrap();
    sampler.prepare("screen").unwrap();
    let sampled = sample_split(&sampler, "screen", 200, None, 32).unwrap();
    let total: u64 = sampled.drops.values().sum();
    assert_eq!(
        sampled.episodes.len() as u64 + total,
        sampled.attempts,
        "every attempt is kept or counted"
    );
    for reason in sampled.drops.keys() {
        assert!(KNOWN_DROPS.contains(reason), "unknown drop reason {reason}");
    }
    assert!(
        sampled
            .drops
            .get("no_second_target_at_distance")
            .copied()
            .unwrap_or(0)
            > 0,
        "the fixture screen split has balls with one candidate: {:?}",
        sampled.drops
    );
    // the k = 1 tally over the same split never names a k = 2 reason
    let mut single = k2_config("cheapest-first");
    single.targets = 1;
    let mut sampler = Sampler::new(&graph, single).unwrap();
    sampler.prepare("screen").unwrap();
    let v1 = sample_split(&sampler, "screen", 200, None, 32).unwrap();
    for reason in v1.drops.keys() {
        assert!(
            !KNOWN_DROPS[6..].contains(reason),
            "a k = 1 draw reported {reason}"
        );
    }
}

/// Same seed, same episodes — whatever the chunking.
#[test]
fn the_k2_draw_is_deterministic_per_seed() {
    for rule in ["cheapest-first", "greedy-path"] {
        let (graph, embeddings) = fixture_world(5, 400, 1600, 8);
        let greedy = rule == "greedy-path";
        let emb: Option<&(dyn hf_policies::Embeddings + Sync)> =
            if greedy { Some(&embeddings) } else { None };
        let mut sampler = Sampler::new(&graph, k2_config(rule)).unwrap();
        sampler.prepare("train").unwrap();
        let a = sample_split(&sampler, "train", 12, emb, 4).unwrap();
        let b = sample_split(&sampler, "train", 12, emb, 7).unwrap();
        assert_eq!(a.attempts, b.attempts);
        assert_eq!(a.drops, b.drops);
        for (x, y) in a.episodes.iter().zip(&b.episodes) {
            assert_eq!(x.episode_id, y.episode_id);
            assert_eq!(
                hf_core::canonical_bytes(&x.visible).unwrap(),
                hf_core::canonical_bytes(&y.visible).unwrap()
            );
            assert_eq!(
                hf_core::canonical_bytes(&x.hidden).unwrap(),
                hf_core::canonical_bytes(&y.hidden).unwrap()
            );
        }
        // and a larger draw reproduces the smaller as its prefix
        let large = sample_split(&sampler, "train", 24, emb, 5).unwrap();
        let small: Vec<&str> = a.episodes.iter().map(|e| e.episode_id.as_str()).collect();
        let big: Vec<&str> = large
            .episodes
            .iter()
            .map(|e| e.episode_id.as_str())
            .collect();
        assert_eq!(&big[..small.len()], &small[..]);
    }
}

/// Steps 1–3 of the joint cut, on a hand-built pair: the designated survivors
/// are protected from BOTH cuts, each cut removes at most `level` NEW edges,
/// and an edge another cut already removed is skipped, not re-counted.
#[test]
fn the_joint_cut_protects_every_designated_survivor() {
    let p = |ns: &[&str]| ns.iter().map(|s| s.to_string()).collect::<Vec<String>>();
    // two targets, t1 and t2, sharing the s → a leg
    let to_t1 = vec![p(&["s", "a", "t1"]), p(&["s", "b", "t1"])];
    let to_t2 = vec![p(&["s", "a", "t2"]), p(&["s", "c", "t2"])];
    let route1 = p(&["s", "a", "t1"]);
    let route2 = p(&["s", "a", "t2"]);
    let (removed, designated, unremovable) =
        choose_joint_greedy_removals(&[to_t1.clone(), to_t2.clone()], &[route1, route2], 2)
            .unwrap();
    assert_eq!(designated.len(), 2, "one protected survivor per target");
    // the survivor of each target is the path sharing fewest edges with its route
    assert_eq!(designated[0], vec![p(&["s", "b", "t1"])]);
    assert_eq!(designated[1], vec![p(&["s", "c", "t2"])]);
    let protected: BTreeSet<(String, String)> = designated
        .iter()
        .flatten()
        .flat_map(|path| {
            path.windows(2)
                .map(|w| (w[0].clone(), w[1].clone()))
                .collect::<Vec<_>>()
        })
        .collect();
    for edge in &removed {
        assert!(!protected.contains(edge), "{edge:?} is a survivor's edge");
    }
    // each route gave up its two edges; (s, a) is shared and cut once
    assert!(removed.contains(&("a".into(), "t1".into())));
    assert!(removed.contains(&("a".into(), "t2".into())));
    assert!(removed.contains(&("s".into(), "a".into())));
    assert_eq!(removed.len(), 3, "the shared leg is not cut twice");
    assert_eq!(unremovable, 0);
    // with the level at 1 only the edge nearest each target goes
    let (removed, _, _) = choose_joint_greedy_removals(
        &[to_t1, to_t2],
        &[p(&["s", "a", "t1"]), p(&["s", "a", "t2"])],
        1,
    )
    .unwrap();
    assert_eq!(removed.len(), 2);
    assert!(!removed.contains(&("s".into(), "a".into())));
}

/// `targets = 1` is v1: the field enters neither the draw key, the sampler
/// block, nor the episode id, and `Dropped` still carries the six v1 reasons.
#[test]
fn one_target_leaves_the_v1_draw_untouched() {
    let mut c = SamplerConfig::new("wikidata5m", 40, 3, 2);
    c.hub_degree_cap = Some(19);
    c.screen_region = 0.5;
    let v1_key = c.draw_key("train", 1);
    assert!(!v1_key.contains("targets"), "{v1_key}");
    assert!(c.as_value().get("targets").is_none());
    c.targets = 2;
    assert_eq!(
        c.draw_key("train", 1),
        format!("{}, 'targets': 2}}", v1_key.trim_end_matches('}'))
    );
    assert_eq!(c.as_value()["targets"], 2);
    // and a k value the design note does not draw is refused
    c.targets = 3;
    assert!(c.validate().is_err());
    c.targets = 0;
    assert!(c.validate().is_err());
    assert_eq!(
        Dropped::TargetsInterdependent.as_str(),
        "targets_interdependent"
    );
    assert_eq!(Dropped::RemovalLeftNoPath.as_str(), "removal_left_no_path");
    assert_eq!(
        Dropped::SurvivorNotRecovered.as_str(),
        "survivor_not_recovered"
    );
    assert_eq!(
        Dropped::NoSecondTargetAtDistance.as_str(),
        "no_second_target_at_distance"
    );
}
