//! Stage 1 on the engine: the query is the episode's own question vector and
//! the target reaches the walk through the hidden payload alone.
//!
//! The test this file exists for is `the_hidden_target_moves_no_feature_no_score_and_no_dump_row`:
//! two episodes that differ ONLY in which node the hidden payload names as the
//! target, with the question held fixed, must produce the same feature rows,
//! the same scores, the same cosine column, the same stop rows and the same
//! dumped candidate records, byte for byte. It is a test that can fail:
//! `a_query_taken_from_the_hidden_target_fails_the_same_comparison` runs the
//! SAME comparison over a deliberately leaky index — one whose query is the
//! hidden target's own embedding, which is the bug `episode_query` exists to
//! prevent — and requires it to differ, and
//! `the_same_swap_at_stage_0_moves_the_features` shows the swap moving the
//! rows under the default query source, where the target IS shown.
//!
//! Beside them: the refusals (a stage-1 payload that names a target, a record
//! with more than one hidden target, `raw-v5`, an episode the query cache does
//! not cover) and the view a feature builder receives, which under
//! `episode_query` shows no target at all.

use hf_embed::EmbeddingMatrix;
use hf_io::RealEpisode;
use hf_walk::{
    walk_batch, DecisionBatch, EpisodeIndex, FeatureSet, QuerySource, QueryVectors, RawV5,
    RelationalV6, Scored, Scorer, StopRule, VisibleIndex, WalkOptions, STOP_DIM,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const EDIM: usize = 8;
const EPISODE_ID: &str = "stage1-fixture-0001";

/// The eight nodes of the fixture, with the two candidate targets `t1` and
/// `t2` hanging off the SAME parent `b`: whichever of them the hidden payload
/// names, it is enumerated at the same expansion, so the two walks stop after
/// the same number of decisions and a difference in the rows cannot be a
/// difference in how long the walk ran.
const NODES: [&str; 8] = ["s", "a", "b", "c", "d", "e", "t1", "t2"];
const EDGES: [(u32, &str, &str); 8] = [
    (1, "s", "a"),
    (2, "s", "c"),
    (3, "a", "b"),
    (4, "c", "d"),
    (5, "c", "e"),
    (6, "d", "e"),
    (7, "b", "t1"),
    (8, "b", "t2"),
];

/// A deterministic vector per node; `t1` and `t2` differ, so a walk that reads
/// the target's identity cannot help but show it.
fn vector(name: &str, salt: u8) -> Vec<f32> {
    let mut h = Sha256::new();
    h.update(name.as_bytes());
    h.update([salt]);
    let d = h.finalize();
    (0..EDIM)
        .map(|i| (d[i] as f32 - 128.0) / 128.0)
        .collect::<Vec<f32>>()
}

fn node_cache() -> EmbeddingMatrix {
    let names: Vec<String> = NODES.iter().map(|n| (*n).to_string()).collect();
    let data: Vec<f32> = names.iter().flat_map(|n| vector(n, 0)).collect();
    EmbeddingMatrix::from_rows(names, EDIM, data)
}

/// The question vectors, keyed by EPISODE id, as `real_walk_teacher.py` will
/// write them: an ordinary v5 cache whose "node" ids are episode ids.
fn query_cache() -> EmbeddingMatrix {
    EmbeddingMatrix::from_rows(
        vec![EPISODE_ID.to_string()],
        EDIM,
        vector("the question the teacher wrote", 7),
    )
}

fn visible(stage1: bool, target: &str) -> Value {
    let mut payload = json!({
        "schema_version": hf_io::SCHEMA_VERSION_V5,
        "record_kind": hf_io::VISIBLE_KIND,
        "family": "fixture",
        "stage": if stage1 { hf_io::STAGES[1] } else { hf_io::STAGES[0] },
        "start_node": "s",
        "subgraph_size": NODES.len(),
        "removal_level": 0,
        "nodes": NODES.iter().map(|n| json!({"node": n, "text": ""})).collect::<Vec<_>>(),
        "edges": EDGES.iter().map(|(id, s, t)| json!({"edge_id": id, "relation": Value::Null, "source": s, "target": t})).collect::<Vec<_>>(),
    });
    let object = payload.as_object_mut().expect("an object");
    if stage1 {
        object.insert("query".into(), json!("which node answers this?"));
    } else {
        object.insert("target_node".into(), json!(target));
    }
    payload
}

fn hidden(target: &str) -> Value {
    json!({
        "schema_version": hf_io::SCHEMA_VERSION_V5,
        "record_kind": hf_io::HIDDEN_KIND,
        "family": "fixture",
        "stage": hf_io::STAGES[1],
        "split": "screen",
        "index": 0,
        "start_node": "s",
        "target_set": [target],
        "target_distance": 3,
        "cost_bound": 3,
        "path_set": [["s", "a", "b", target]],
        "surviving_paths": [["s", "a", "b", target]],
        "removal_set": [],
        "removed_count": 0,
        "unremovable_count": 0,
        "nodes_on_surviving_path": ["s", "a", "b", target],
        "distance_to_target": {"s": 3, "a": 2, "b": 1, target: 0},
        "sampler": {"subgraph_size": 8},
    })
}

/// One episode of the fixture. `stage1` writes the stage-1 payload (a question,
/// no target shown); otherwise the stage-0 one, which names the target.
fn episode(stage1: bool, target: &str) -> RealEpisode {
    let raw = visible(stage1, target);
    hf_io::validate_visible(&raw).expect("the payload is a valid one");
    RealEpisode {
        episode_id: EPISODE_ID.to_string(),
        visible: serde_json::from_value(raw).expect("visible"),
        hidden: serde_json::from_value(hidden(target)).expect("hidden"),
    }
}

/// A pure function of the scoring input: the walk is then a function of the
/// feature rows and of nothing else, so two walks agree exactly when their
/// rows do.
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
                            h.update((it.context_len as u32).to_le_bytes());
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

/// Everything one walk would hand a model or write to `candidate_dump.jsonl.gz`,
/// as one string: per decision the candidate rows, the context tokens, the pair
/// channel, the query token, the cosine column, the stop row, and the dumped
/// record (names, scores, cosines, chosen, parents, depths). Nothing here is
/// hidden data — which is the point: swapping the hidden target may not move a
/// byte of it.
fn walk_fingerprint(index: &EpisodeIndex) -> (usize, bool, String) {
    let features = RelationalV6;
    let mut scorer = HashScorer {
        cdim: features.candidate_dim(EDIM),
    };
    let result = walk_batch(
        &[index],
        &features,
        &mut scorer,
        WalkOptions {
            stop_rule: StopRule::Exhaust,
            record_candidates: true,
            keep_items: true,
            with_prior: false,
        },
    )
    .expect("the walk runs")
    .remove(0);
    let records = result.candidates.as_ref().expect("records were asked for");
    let mut lines: Vec<Value> = Vec::new();
    for (k, d) in result.decisions.iter().enumerate() {
        let item = d.item.as_ref().expect("the item was kept");
        lines.push(json!({
            "decision": k,
            "cand": item.cand,
            "ctx": item.ctx,
            "pair": item.pair,
            "query": item.query,
            "frontier_len": item.frontier_len,
            "context_len": item.context_len,
            "cosines": d.cosines,
            "stop_features": d.stop_features,
            "chosen": d.chosen,
            "dump": serde_json::to_value(&records[k]).expect("a record"),
        }));
    }
    (
        result.decisions.len(),
        result.registered(),
        serde_json::to_string(&lines).expect("a fingerprint"),
    )
}

fn stage1_index(target: &str, nodes: &EmbeddingMatrix, queries: &EmbeddingMatrix) -> EpisodeIndex {
    EpisodeIndex::new_with_query(
        &episode(true, target),
        nodes,
        EDIM,
        QueryVectors::episode_query(queries),
    )
    .expect("a stage-1 record builds under episode_query")
}

/// **The leak test.** Two episodes that differ only in the hidden target, with
/// the question held fixed: every feature row, every score, the cosine column,
/// every stop row and every dumped candidate record are byte-identical.
#[test]
fn the_hidden_target_moves_no_feature_no_score_and_no_dump_row() {
    let nodes = node_cache();
    let queries = query_cache();
    let one = stage1_index("t1", &nodes, &queries);
    let other = stage1_index("t2", &nodes, &queries);
    // the swap is real: the walk registers on a different node
    assert_ne!(
        one.hidden_targets, other.hidden_targets,
        "the two episodes name different hidden targets"
    );
    assert_eq!(one.registration_targets(), one.hidden_targets);
    assert!(one.targets_shown.is_empty() && other.targets_shown.is_empty());
    let (decisions, registered, a) = walk_fingerprint(&one);
    let (other_decisions, other_registered, b) = walk_fingerprint(&other);
    // not vacuous: both walks ran, both completed, and there was something to
    // compare at every decision
    assert!(
        decisions >= 2,
        "only {decisions} decisions were compared — the fixture stops too early"
    );
    assert!(registered && other_registered, "both walks register");
    assert_eq!(
        decisions, other_decisions,
        "the walk itself diverged when the hidden target was swapped"
    );
    assert_eq!(
        a, b,
        "a feature, a score or a dumped row moved when the hidden target was swapped"
    );
}

/// The same comparison, on an index whose query was taken from the HIDDEN
/// target's own embedding — the stage-0 query kept although the sidecar was
/// read. Nothing in the engine builds an index this way; it is built here by
/// hand so that the comparison above is known to be able to fail.
#[test]
fn a_query_taken_from_the_hidden_target_fails_the_same_comparison() {
    let nodes = node_cache();
    let queries = query_cache();
    let leak = |target: &str| -> EpisodeIndex {
        let mut index = stage1_index(target, &nodes, &queries);
        let t = index.hidden_targets[0];
        let q = index.emb(t).to_vec();
        index.unit_queries = hf_walk::visible::unit_query(&q);
        index.queries = q;
        index
    };
    let (_, _, a) = walk_fingerprint(&leak("t1"));
    let (_, _, b) = walk_fingerprint(&leak("t2"));
    assert_ne!(
        a, b,
        "the comparison cannot fail: a query read off the hidden target left every row alike"
    );
}

/// And under the default query source, where the record SHOWS the target, the
/// same swap moves the rows — the sensitivity of the comparison again, this
/// time through the ordinary constructor.
#[test]
fn the_same_swap_at_stage_0_moves_the_features() {
    let nodes = node_cache();
    let index = |target: &str| -> EpisodeIndex {
        EpisodeIndex::new(&episode(false, target), &nodes, EDIM).expect("a stage-0 record builds")
    };
    let one = index("t1");
    assert_eq!(one.query_source, QuerySource::TargetEmbedding);
    assert_eq!(one.registration_targets(), one.targets_shown);
    let (_, _, a) = walk_fingerprint(&one);
    let (_, _, b) = walk_fingerprint(&index("t2"));
    assert_ne!(a, b, "at stage 0 the target IS the query");
}

/// What a feature builder is handed under `episode_query`: no target, one
/// query, and every query channel reducing over that one query — the
/// registration mask the walk keeps is not a mask over anything it can see.
#[test]
fn the_view_a_builder_receives_shows_no_target_under_episode_query() {
    let nodes = node_cache();
    let queries = query_cache();
    let index = stage1_index("t1", &nodes, &queries);
    let mask = [true];
    let view = VisibleIndex::new(&index, &mask);
    assert!(view.targets_shown().is_empty(), "no target is shown");
    assert_eq!(view.target_count(), 0);
    assert_eq!(view.target_shown(), None);
    assert_eq!(view.query_count(), 1, "the episode's own question vector");
    assert_eq!(view.query(), queries.get(EPISODE_ID).expect("the question"));
    // the reduction over one query is that query's own cosine
    let c = view.emb(index.start);
    assert_eq!(view.cos_query_max(c), hf_walk::cosine(c, view.query()));
    assert_eq!(view.cos_query_min_max(c).0, view.cos_query_max(c));
}

/// A visible payload that names a target under stage 1 is refused — by the
/// reader that validates a split, and again by the index, which also reads
/// records that no reader has seen.
#[test]
fn a_stage_1_payload_that_names_a_target_is_refused() {
    let mut raw = visible(true, "t1");
    raw.as_object_mut()
        .unwrap()
        .insert("target_node".into(), json!("t1"));
    let refused = hf_io::validate_visible(&raw).expect_err("a stage-1 payload carrying a target");
    assert!(
        format!("{refused}").contains("exactly one of target_node / query"),
        "{refused}"
    );
    // and the same payload with the query dropped, which is stage 0's shape
    // under stage 1's label
    let mut stage0_shaped = visible(true, "t1");
    let object = stage0_shaped.as_object_mut().unwrap();
    object.remove("query");
    object.insert("target_node".into(), json!("t1"));
    assert!(hf_io::validate_visible(&stage0_shaped).is_err());
    // the index refuses it too: the record is built in memory here, so no
    // reader has been past it
    let nodes = node_cache();
    let queries = query_cache();
    let mut e = episode(true, "t1");
    e.visible.target_node = Some("t1".into());
    let refused =
        EpisodeIndex::new_with_query(&e, &nodes, EDIM, QueryVectors::episode_query(&queries))
            .map(|_| ())
            .expect_err("a visible target under episode_query");
    assert!(format!("{refused}").contains("names a target"), "{refused}");
}

/// `episode_query` is k = 1 only (`hf-io` admits no other stage-1 record); the
/// engine says so itself rather than indexing the first of several targets.
#[test]
fn a_stage_1_record_with_two_hidden_targets_is_refused() {
    let nodes = node_cache();
    let queries = query_cache();
    let mut e = episode(true, "t1");
    e.hidden.target_set = vec!["t1".into(), "t2".into()];
    let refused =
        EpisodeIndex::new_with_query(&e, &nodes, EDIM, QueryVectors::episode_query(&queries))
            .map(|_| ())
            .expect_err("two hidden targets");
    assert!(format!("{refused}").contains("k = 1 only"), "{refused}");
}

/// A stage-0 record under `episode_query` is refused: its query would be the
/// question of an episode that has none.
#[test]
fn a_stage_0_record_is_refused_under_episode_query() {
    let nodes = node_cache();
    let queries = query_cache();
    let refused = EpisodeIndex::new_with_query(
        &episode(false, "t1"),
        &nodes,
        EDIM,
        QueryVectors::episode_query(&queries),
    )
    .map(|_| ())
    .expect_err("a stage-0 record");
    assert!(
        format!("{refused}").contains("stage1_described_target"),
        "{refused}"
    );
}

/// An episode the sidecar does not cover exits 2 rather than walking on a zero
/// vector — the silent-zero-vector class, closed where the query is read.
#[test]
fn an_episode_the_query_cache_does_not_cover_is_refused() {
    let nodes = node_cache();
    let empty = EmbeddingMatrix::from_rows(vec!["another-episode".into()], EDIM, vec![0.0; EDIM]);
    let refused = EpisodeIndex::new_with_query(
        &episode(true, "t1"),
        &nodes,
        EDIM,
        QueryVectors::episode_query(&empty),
    )
    .map(|_| ())
    .expect_err("an uncovered episode");
    assert!(
        format!("{refused}").contains("no vector for episode"),
        "{refused}"
    );
}

/// `raw-v5` copies the query vector into every candidate row and into the query
/// token; under `episode_query` that vector is the question itself, so the
/// pairing is refused where a record meets a feature set.
#[test]
fn raw_v5_is_refused_under_episode_query() {
    let nodes = node_cache();
    let queries = query_cache();
    let index = stage1_index("t1", &nodes, &queries);
    let mut scorer = HashScorer {
        cdim: RawV5.candidate_dim(EDIM),
    };
    let refused = walk_batch(
        &[&index],
        &RawV5,
        &mut scorer,
        WalkOptions {
            stop_rule: StopRule::Exhaust,
            record_candidates: false,
            keep_items: false,
            with_prior: false,
        },
    )
    .expect_err("raw-v5 under episode_query");
    assert!(format!("{refused}").contains("raw-v5"), "{refused}");
    // the relational set, which reads the query only through cosines, is not
    assert!(walk_batch(
        &[&index],
        &RelationalV6,
        &mut HashScorer {
            cdim: RelationalV6.candidate_dim(EDIM)
        },
        WalkOptions {
            stop_rule: StopRule::Exhaust,
            record_candidates: false,
            keep_items: false,
            with_prior: false,
        },
    )
    .is_ok());
}

/// The premise of the bit-identity claim: at stage 0 there is one query per
/// shown target, so the count the query channels reduce over — now the QUERY
/// count — is the count they always reduced over.
#[test]
fn stage_0_has_one_query_per_shown_target() {
    let goldens = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../hf-io/tests/goldens/fixture-split");
    let (_, embeddings) = hf_episodes::fixture::fixture_world(5, 400, 1600, 8);
    let mut names: Vec<String> = embeddings.keys().cloned().collect();
    names.sort();
    let data: Vec<f32> = names
        .iter()
        .flat_map(|n| embeddings[n].iter().map(|x| *x as f32))
        .collect();
    let cache = EmbeddingMatrix::from_rows(names, 8, data);
    let mut seen = 0usize;
    for split in ["train", "screen", "train-greedy"] {
        for e in hf_io::read_split(&goldens.join(split))
            .expect("a golden split")
            .0
        {
            let index = EpisodeIndex::new(&e, &cache, 8).expect("a stage-0 record");
            assert_eq!(index.query_count(), index.target_count());
            assert_eq!(index.registration_targets(), index.targets_shown);
            seen += 1;
        }
    }
    assert!(seen > 0, "the goldens were empty");
}

/// The children of one node, in the `edge_id` order the walk examines them.
fn children_of(node: &str) -> Vec<&'static str> {
    let mut out: Vec<(u32, &'static str, &'static str)> = EDGES
        .iter()
        .copied()
        .filter(|(_, source, _)| *source == node)
        .collect();
    out.sort_unstable();
    out.into_iter().map(|(_, _, tail)| tail).collect()
}

/// ENG-9's rank, re-derived from `EDGES`, the caches and the expansion order
/// the walk reports — not from the walk's own bookkeeping. The comparison set
/// is rebuilt the way the definition states it: the start node, then every
/// child examined in `edge_id` order until the target itself comes into view.
fn rank_from_the_definition(expanded: &[String], target: &str, query: &[f32]) -> usize {
    let nodes = node_cache();
    let mut seen: Vec<&str> = vec!["s"];
    'walk: for node in expanded {
        for child in children_of(node) {
            if child == target {
                break 'walk;
            }
            if !seen.contains(&child) {
                seen.push(child);
            }
        }
    }
    assert!(
        !seen.contains(&target),
        "the target is not in the set it is ranked against"
    );
    let theirs = hf_walk::cosine(nodes.get(target).expect("a vector"), query);
    1 + seen
        .iter()
        .filter(|n| hf_walk::cosine(nodes.get(n).expect("a vector"), query) > theirs)
        .count()
}

fn exhaust(index: &EpisodeIndex) -> hf_walk::WalkResult {
    let features = RelationalV6;
    let mut scorer = HashScorer {
        cdim: features.candidate_dim(EDIM),
    };
    walk_batch(
        &[index],
        &features,
        &mut scorer,
        WalkOptions {
            stop_rule: StopRule::Exhaust,
            record_candidates: false,
            keep_items: false,
            with_prior: false,
        },
    )
    .expect("the walk runs")
    .remove(0)
}

/// A stage-1 record whose hidden target `u` sits in the subgraph but has no
/// edge into it: the walk exhausts the frontier without ever examining it.
fn unreachable_target_episode() -> RealEpisode {
    let mut raw = visible(true, "t1");
    let object = raw.as_object_mut().expect("an object");
    object["subgraph_size"] = json!(NODES.len() + 1);
    object["nodes"]
        .as_array_mut()
        .expect("nodes")
        .push(json!({"node": "u", "text": ""}));
    object["edges"]
        .as_array_mut()
        .expect("edges")
        .push(json!({"edge_id": 9, "relation": Value::Null, "source": "u", "target": "s"}));
    hf_io::validate_visible(&raw).expect("the payload is a valid one");
    RealEpisode {
        episode_id: EPISODE_ID.to_string(),
        visible: serde_json::from_value(raw).expect("visible"),
        hidden: serde_json::from_value(hidden("u")).expect("hidden"),
    }
}

/// **ENG-9.** The rank the walk records is the target's place, by cosine to
/// the QUESTION, among the nodes it had seen when the target came into view —
/// checked against that definition re-derived from the fixture; `None` when
/// the target never registers; and 1, vacuously, under the default query
/// source, where the query IS the target's own embedding row.
#[test]
fn the_cosine_rank_at_registration_is_the_targets_place_among_what_the_walk_had_seen() {
    let nodes = node_cache();
    let queries = query_cache();
    let question: Vec<f32> = queries.get(EPISODE_ID).expect("a question").to_vec();
    let mut moved = 0;
    for target in ["t1", "t2"] {
        let index = stage1_index(target, &nodes, &queries);
        let result = exhaust(&index);
        assert!(result.registered(), "{target} registers");
        let names: Vec<String> = result
            .expanded
            .iter()
            .map(|n| index.names[*n as usize].clone())
            .collect();
        let want = rank_from_the_definition(&names, target, &question);
        assert_eq!(
            result.cosine_rank_at_registration_k1(),
            Some(want),
            "{target}: the recorded rank is not the one the definition gives"
        );
        assert_eq!(result.cosine_rank_at_registration, vec![Some(want)]);
        if want > 1 {
            moved += 1;
        }
    }
    assert!(
        moved > 0,
        "neither target was out-ranked by anything the walk had seen; \
         a rank that is always 1 makes the check vacuous"
    );

    // never registered: null, not 1
    let index = EpisodeIndex::new_with_query(
        &unreachable_target_episode(),
        &nodes,
        EDIM,
        QueryVectors::episode_query(&queries),
    )
    .expect("a stage-1 record");
    let result = exhaust(&index);
    assert!(!result.registered(), "the hidden target is unreachable");
    assert_eq!(result.cosine_rank_at_registration_k1(), None);

    // the default source ranks the target against its OWN embedding row, so
    // the rank is 1 and carries no information; it is recorded, not hidden
    for target in ["t1", "t2"] {
        let index =
            EpisodeIndex::new(&episode(false, target), &nodes, EDIM).expect("a stage-0 record");
        let result = exhaust(&index);
        assert!(result.registered());
        assert_eq!(
            result.cosine_rank_at_registration_k1(),
            Some(1),
            "at stage 0 the query is the target's own row"
        );
    }
}
