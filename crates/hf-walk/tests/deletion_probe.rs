//! The R1 premise's probe index (`QuerySource::DeletedPayload`,
//! `R1_PREMISE_PLAN.md` §2 E2): a ball rebuilt without the deleted node X,
//! walked on X's own embedding row, with no registration target at all.
//!
//! - Registration never fires, and the walk runs to an empty frontier.
//! - `walk_batch_capped` honours its cap, and a capped walk's expansion order
//!   is a prefix of the uncapped one's (the plan's E_B prefix property).
//! - The index takes no hidden input: it has no shown or hidden target, every
//!   on-path flag is false, and the view a feature builder receives shows no
//!   target.
//! - Equivalence: the probe walk is the stage-0 walk of the same ball whose
//!   target was cut off from it — a stage-0 record naming X as its target with
//!   every edge of X removed walks the same decisions, row for row, and never
//!   registers. So the probe asks the trained walk exactly the stage-0
//!   question with the answer deleted.

use hf_embed::EmbeddingMatrix;
use hf_io::RealEpisode;
use hf_walk::{
    walk_batch, walk_batch_capped, DecisionBatch, EpisodeIndex, FeatureSet, QuerySource,
    RelationalV6, RelationalV6Prev, Scored, Scorer, StopRule, VisibleIndex, WalkOptions, STOP_DIM,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const EDIM: usize = 8;

const NODES: [&str; 7] = ["s", "a", "b", "c", "d", "e", "f"];
const EDGES: [(u32, &str, &str); 8] = [
    (0, "s", "a"),
    (1, "s", "c"),
    (2, "a", "b"),
    (3, "c", "d"),
    (4, "c", "e"),
    (5, "d", "e"),
    (6, "b", "f"),
    (7, "e", "f"),
];

fn vector(name: &str) -> Vec<f32> {
    let mut h = Sha256::new();
    h.update(name.as_bytes());
    let d = h.finalize();
    (0..EDIM).map(|i| (d[i] as f32 - 128.0) / 128.0).collect()
}

/// Every fixture node and the deleted node `x` (the cache holds X's row, as
/// the real cache does; the ball does not hold X).
fn cache() -> EmbeddingMatrix {
    let mut names: Vec<String> = NODES.iter().map(|n| n.to_string()).collect();
    names.push("x".into());
    let data: Vec<f32> = names.iter().flat_map(|n| vector(n)).collect();
    EmbeddingMatrix::from_rows(names, EDIM, data)
}

fn names() -> Vec<String> {
    NODES.iter().map(|n| n.to_string()).collect()
}

fn edges() -> Vec<(u32, String, String)> {
    EDGES
        .iter()
        .map(|(i, s, t)| (*i, s.to_string(), t.to_string()))
        .collect()
}

fn probe_index(cache: &EmbeddingMatrix) -> EpisodeIndex {
    EpisodeIndex::for_deletion(
        "fixture-screen-000001-00000000-del-U",
        "s",
        &names(),
        &edges(),
        cache,
        EDIM,
        &vector("x"),
    )
    .expect("the probe index builds")
}

struct HashScorer {
    cdim: usize,
}

impl Scorer for HashScorer {
    fn score(&mut self, batch: &DecisionBatch) -> Result<Scored, hf_core::HfError> {
        Ok(Scored {
            scores: batch
                .items
                .iter()
                .map(|it| {
                    (0..it.frontier_len)
                        .map(|i| {
                            let row = &it.cand[i * self.cdim..(i + 1) * self.cdim];
                            let mut h = Sha256::new();
                            for v in row {
                                h.update(v.to_le_bytes());
                            }
                            let d = h.finalize();
                            (u32::from_le_bytes([d[0], d[1], d[2], d[3]]) as f32) / u32::MAX as f32
                        })
                        .collect()
                })
                .collect(),
            residuals: None,
        })
    }
    fn stop_logits(&mut self, rows: &[[f32; STOP_DIM]]) -> Result<Vec<f32>, hf_core::HfError> {
        Ok(vec![-1.0; rows.len()])
    }
}

fn exhaust() -> WalkOptions {
    WalkOptions {
        stop_rule: StopRule::Exhaust,
        record_candidates: true,
        keep_items: true,
        with_prior: false,
    }
}

fn walk(
    index: &EpisodeIndex,
    features: &dyn FeatureSet,
    cap: Option<usize>,
) -> hf_walk::WalkResult {
    let mut scorer = HashScorer {
        cdim: features.candidate_dim(EDIM),
    };
    walk_batch_capped(&[index], features, &mut scorer, exhaust(), cap)
        .expect("the walk runs")
        .remove(0)
}

#[test]
fn the_probe_never_registers_and_walks_to_an_empty_frontier() {
    let cache = cache();
    let index = probe_index(&cache);
    assert_eq!(index.query_source, QuerySource::DeletedPayload);
    assert!(index.registration_targets().is_empty());
    assert_eq!(index.target_count(), 0);
    assert!(index.hidden_targets.is_empty());
    assert!(index.on_path.iter().all(|f| !f));
    assert!(index.distance.iter().all(Option::is_none));
    assert_eq!(index.query(), &vector("x")[..]);
    for features in [&RelationalV6 as &dyn FeatureSet, &RelationalV6Prev] {
        let r = walk(&index, features, None);
        assert_eq!(r.registered_at, None);
        assert!(r.registered_at_by_target.is_empty());
        assert_eq!(r.stop_reason, "exhausted");
        assert_eq!(
            r.expanded.len(),
            NODES.len(),
            "every node is reachable and expanded"
        );
    }
}

#[test]
fn the_cap_is_honoured_and_a_capped_walk_is_a_prefix() {
    let cache = cache();
    let index = probe_index(&cache);
    let full = walk(&index, &RelationalV6Prev, None);
    for cap in 1..=NODES.len() + 2 {
        let capped = walk(&index, &RelationalV6Prev, Some(cap));
        let want = cap.min(full.expanded.len());
        assert_eq!(capped.expanded.len(), want, "cap {cap}");
        assert_eq!(&capped.expanded[..], &full.expanded[..want], "cap {cap}");
        if cap < full.expanded.len() {
            assert_eq!(capped.stop_reason, "expansion_cap");
        } else {
            assert_eq!(capped.stop_reason, "exhausted");
        }
    }
    let mut scorer = HashScorer {
        cdim: RelationalV6Prev.candidate_dim(EDIM),
    };
    assert!(walk_batch_capped(
        &[&index],
        &RelationalV6Prev,
        &mut scorer,
        exhaust(),
        Some(0)
    )
    .is_err());
}

#[test]
fn the_view_shows_no_target_and_the_index_refuses_a_sampled_record() {
    let cache = cache();
    let index = probe_index(&cache);
    let mask = vec![true; index.query_count()];
    let view = VisibleIndex::new(&index, &mask);
    assert_eq!(view.target_count(), 0);
    assert_eq!(view.target_shown(), None);
    assert_eq!(view.query_count(), 1);
    // a node the payload does not list is band H, and a sampled record can
    // never be read under the probe's query source
    let bad = EpisodeIndex::for_deletion(
        "id",
        "s",
        &names(),
        &[(0, "s".into(), "zz".into())],
        &cache,
        EDIM,
        &vector("x"),
    );
    assert!(bad.is_err());
    let episode = stage0_episode();
    let refused = EpisodeIndex::new_with_query(
        &episode,
        &cache,
        EDIM,
        hf_walk::QueryVectors {
            source: QuerySource::DeletedPayload,
            cache: None,
        },
    );
    assert!(refused.is_err());
    assert!(
        QuerySource::parse("deleted_payload").is_err(),
        "no config can select it"
    );
}

/// A stage-0 record over the same ball PLUS the deleted node `x` as its shown
/// target, with every edge of `x` removed: the target is in the node list and
/// unreachable.
fn stage0_episode() -> RealEpisode {
    let mut nodes: Vec<Value> = NODES
        .iter()
        .map(|n| json!({"node": n, "text": ""}))
        .collect();
    nodes.push(json!({"node": "x", "text": ""}));
    let visible = json!({
        "schema_version": hf_io::SCHEMA_VERSION_V5,
        "record_kind": hf_io::VISIBLE_KIND,
        "family": "fixture",
        "stage": hf_io::STAGES[0],
        "start_node": "s",
        "target_node": "x",
        "subgraph_size": nodes.len(),
        "removal_level": 0,
        "nodes": nodes,
        "edges": EDGES.iter().map(|(id, s, t)| json!({"edge_id": id, "relation": Value::Null, "source": s, "target": t})).collect::<Vec<_>>(),
    });
    hf_io::validate_visible(&visible).expect("valid");
    let hidden = json!({
        "schema_version": hf_io::SCHEMA_VERSION_V5,
        "record_kind": hf_io::HIDDEN_KIND,
        "family": "fixture",
        "stage": hf_io::STAGES[0],
        "split": "screen",
        "index": 0,
        "start_node": "s",
        "target_set": ["x"],
        "target_distance": 0,
        "cost_bound": 0,
        "path_set": [],
        "surviving_paths": [],
        "removal_set": [],
        "removed_count": 0,
        "unremovable_count": 0,
        "nodes_on_surviving_path": [],
        "distance_to_target": {},
        "sampler": {"subgraph_size": 8},
    });
    RealEpisode {
        episode_id: "fixture-screen-000001-00000000".into(),
        visible: serde_json::from_value(visible).expect("visible"),
        hidden: serde_json::from_value(hidden).expect("hidden"),
    }
}

/// The equivalence the plan names: the probe walk on X's row over the ball
/// without X takes the same decisions, on the same feature rows and scores, as
/// the stage-0 walk whose shown target X has been cut off from the same ball.
#[test]
fn the_probe_is_the_stage0_walk_with_its_target_cut_off() {
    let cache = cache();
    let probe = probe_index(&cache);
    let episode = stage0_episode();
    let stage0 = EpisodeIndex::new(&episode, &cache, EDIM).expect("stage-0 index");
    assert_eq!(stage0.query(), probe.query());
    for features in [&RelationalV6 as &dyn FeatureSet, &RelationalV6Prev] {
        let mut s1 = HashScorer {
            cdim: features.candidate_dim(EDIM),
        };
        let mut s2 = HashScorer {
            cdim: features.candidate_dim(EDIM),
        };
        let a = walk_batch(&[&probe], features, &mut s1, exhaust())
            .unwrap()
            .remove(0);
        let b = walk_batch(&[&stage0], features, &mut s2, exhaust())
            .unwrap()
            .remove(0);
        assert_eq!(b.registered_at, None, "the cut-off target never registers");
        let names_a: Vec<&str> = a
            .expanded
            .iter()
            .map(|n| probe.names[*n as usize].as_str())
            .collect();
        let names_b: Vec<&str> = b
            .expanded
            .iter()
            .map(|n| stage0.names[*n as usize].as_str())
            .collect();
        assert_eq!(names_a, names_b, "{}", features.name());
        assert_eq!(a.decisions.len(), b.decisions.len());
        for (da, db) in a.decisions.iter().zip(&b.decisions) {
            let (ia, ib) = (da.item.as_ref().unwrap(), db.item.as_ref().unwrap());
            assert_eq!(ia.cand, ib.cand, "{}", features.name());
            assert_eq!(ia.ctx, ib.ctx);
            assert_eq!(ia.pair, ib.pair);
            assert_eq!(ia.query, ib.query);
            assert_eq!(da.cosines, db.cosines);
            assert_eq!(da.chosen, db.chosen);
        }
    }
}
