//! The k-target walk of `K_TARGETS_DESIGN.md` §2 on the fixture world, never
//! on real data: the query reduction's k = 1 identity with `relational-v6`,
//! the switch when a target registers, registration per target from the
//! VISIBLE side, batched == single at k = 2, and recall over k at `B_fix`.

use std::collections::HashMap;
use std::path::PathBuf;

use hf_episodes::fixture::fixture_world;
use hf_episodes::{sample_split, Sampler, SamplerConfig};
use hf_walk::{
    walk_batch, DecisionBatch, EpisodeIndex, FeatureSet, RelationalV6, RelationalV6K, Scored,
    Scorer, StopRule, VisibleIndex, WalkOptions, STOP_DIM,
};

fn cache(embeddings: &HashMap<String, Vec<f64>>, tag: &str) -> hf_embed::EmbeddingMatrix {
    let dir = std::env::temp_dir().join(format!("hf-k-walk-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut names: Vec<&String> = embeddings.keys().collect();
    names.sort();
    let mut f = std::fs::File::create(dir.join("vectors.jsonl")).unwrap();
    for n in &names {
        hf_embed::append_vector(&mut f, n, &embeddings[*n]).unwrap();
    }
    hf_embed::write_manifest(
        &dir,
        &hf_embed::Manifest {
            record_kind: hf_embed::MANIFEST_KIND.into(),
            model: "fixture".into(),
            model_digest: "fixture".into(),
            base_url: "none".into(),
            dimension: 8,
            count: names.len() as u64,
            text_char_limit: 6000,
            text_sha256: None,
            truncated: Default::default(),
            training_authorized: false,
            extra: Default::default(),
        },
    )
    .unwrap();
    hf_embed::EmbeddingMatrix::load(&dir).unwrap()
}

/// The committed k = 1 fixture split, as every other test reads it.
fn k1_episodes() -> (
    Vec<hf_io::RealEpisode>,
    hf_embed::EmbeddingMatrix,
    HashMap<String, Vec<f64>>,
) {
    let goldens =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../hf-io/tests/goldens/fixture-split");
    let (_, embeddings) = fixture_world(5, 400, 1600, 8);
    let matrix = cache(&embeddings, "k1");
    let mut all = Vec::new();
    for name in ["train", "screen"] {
        all.extend(hf_io::read_split(&goldens.join(name)).unwrap().0);
    }
    (all, matrix, embeddings)
}

/// `--targets 2` on the fixture world: the k = 2 records the walk must read.
fn k2_episodes(
    count: usize,
) -> (
    Vec<hf_io::RealEpisode>,
    hf_embed::EmbeddingMatrix,
    HashMap<String, Vec<f64>>,
) {
    let (graph, embeddings) = fixture_world(5, 400, 1600, 8);
    let mut config = SamplerConfig::new("fixture", 64, 3, 2);
    config.cost_epsilon = 0.5;
    config.targets = 2;
    config.removal_rule = "greedy-path".into();
    let mut sampler = Sampler::new(&graph, config).unwrap();
    sampler.prepare("train").unwrap();
    let sampled = sample_split(&sampler, "train", count, Some(&embeddings), 16)
        .unwrap()
        .episodes;
    assert_eq!(sampled.len(), count, "the fixture fills the split");
    let episodes: Vec<hf_io::RealEpisode> = sampled
        .iter()
        .map(|s| hf_io::RealEpisode {
            episode_id: s.episode_id.clone(),
            visible: serde_json::from_value(s.visible.clone()).unwrap(),
            hidden: serde_json::from_value(s.hidden.clone()).unwrap(),
        })
        .collect();
    let matrix = cache(&embeddings, "k2");
    (episodes, matrix, embeddings)
}

/// Scores each candidate by its cosine column: the walk becomes similarity-greedy.
struct CosineScorer {
    column: usize,
    cdim: usize,
}

impl Scorer for CosineScorer {
    fn score(&mut self, batch: &DecisionBatch) -> Result<Scored, hf_core::HfError> {
        Ok(Scored {
            scores: batch
                .items
                .iter()
                .map(|it| {
                    (0..it.frontier_len)
                        .map(|i| it.cand[i * self.cdim + self.column])
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

/// A pure function of the scoring input: the candidate row hashed to a score.
struct HashScorer {
    cdim: usize,
}

impl Scorer for HashScorer {
    fn score(&mut self, batch: &DecisionBatch) -> Result<Scored, hf_core::HfError> {
        use sha2::{Digest, Sha256};
        Ok(Scored {
            scores: batch
                .items
                .iter()
                .map(|it| {
                    (0..it.frontier_len)
                        .map(|i| {
                            let mut h = Sha256::new();
                            for v in &it.cand[i * self.cdim..(i + 1) * self.cdim] {
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

const EXTRA_AT: usize = hf_walk::features::RELATIONAL_K_EXTRA_AT;

/// At k = 1 `relational-v6-k`'s row is `relational-v6`'s with three columns
/// inserted at 9..12, **bit for bit**, and those three are `cos(c,q)`, a
/// spread of zero and an unregistered share of one. The reduction is the
/// identity there — not approximately, exactly: the maximum over one target is
/// that target's own cosine, computed by the same call on the same vector.
///
/// The two walks are run independently and their recorded scoring inputs
/// compared decision by decision; a divergence in the walk itself would show
/// as a frontier of a different length before any column is read.
#[test]
fn the_reduction_at_one_target_is_relational_v6_bit_for_bit() {
    let (episodes, matrix, _) = k1_episodes();
    let indexes: Vec<EpisodeIndex> = episodes
        .iter()
        .map(|e| EpisodeIndex::new(e, &matrix, 8).unwrap())
        .collect();
    let refs: Vec<&EpisodeIndex> = indexes.iter().collect();
    let v6 = RelationalV6;
    let v6k = RelationalV6K;
    let cdim = v6.candidate_dim(8);
    let kdim = v6k.candidate_dim(8);
    assert_eq!(kdim, cdim + 3);
    assert_eq!(kdim, 12 + hf_walk::STRUCTURE_DIM);
    assert_eq!(v6k.context_dim(8), v6.context_dim(8));
    assert_eq!(v6k.pair_dim(), v6.pair_dim());
    assert_eq!(v6k.cosine_column(8), v6.cosine_column(8));
    let options = WalkOptions {
        stop_rule: StopRule::Exhaust,
        record_candidates: false,
        keep_items: true,
        with_prior: false,
    };
    let a = walk_batch(&refs, &v6, &mut CosineScorer { column: 0, cdim }, options).unwrap();
    let b = walk_batch(
        &refs,
        &v6k,
        &mut CosineScorer {
            column: 0,
            cdim: kdim,
        },
        options,
    )
    .unwrap();
    let mut rows = 0;
    for ((x, y), index) in a.iter().zip(&b).zip(&indexes) {
        assert_eq!(x.expanded, y.expanded, "{}", index.episode_id);
        assert_eq!(x.decisions.len(), y.decisions.len());
        for (da, db) in x.decisions.iter().zip(&y.decisions) {
            assert_eq!(da.frontier, db.frontier);
            assert_eq!(da.chosen, db.chosen);
            assert_eq!(da.cosines, db.cosines, "the greedy prior's column");
            assert_eq!(da.stop_features, db.stop_features, "the stop row");
            let (ia, ib) = (da.item.as_ref().unwrap(), db.item.as_ref().unwrap());
            assert_eq!(ib.ctx, ia.ctx, "the context token is unchanged");
            assert_eq!(ib.pair, ia.pair, "the pair channel is unchanged");
            for i in 0..ia.frontier_len {
                let v6_row = &ia.cand[i * cdim..(i + 1) * cdim];
                let k_row = &ib.cand[i * kdim..(i + 1) * kdim];
                let stripped: Vec<f32> = k_row[..EXTRA_AT]
                    .iter()
                    .chain(&k_row[EXTRA_AT + 3..])
                    .copied()
                    .collect();
                assert_eq!(stripped, v6_row, "the v6 row survives the insertion");
                assert_eq!(k_row[EXTRA_AT], v6_row[0], "min == max at k = 1");
                assert_eq!(k_row[EXTRA_AT + 1], 0.0, "the spread is zero at k = 1");
                assert_eq!(k_row[EXTRA_AT + 2], 1.0, "one target, unregistered");
                rows += 1;
            }
        }
    }
    assert!(rows > 500, "only {rows} candidate rows compared");
}

/// The k = 1 walk is bit-identical under the two feature sets when the scorer
/// reads the cosine column: the reduction changes no decision.
#[test]
fn a_cosine_scored_walk_is_the_same_under_v6_and_v6_k_at_one_target() {
    let (episodes, matrix, _) = k1_episodes();
    let indexes: Vec<EpisodeIndex> = episodes
        .iter()
        .map(|e| EpisodeIndex::new(e, &matrix, 8).unwrap())
        .collect();
    let refs: Vec<&EpisodeIndex> = indexes.iter().collect();
    let options = WalkOptions {
        stop_rule: StopRule::Exhaust,
        record_candidates: false,
        keep_items: false,
        with_prior: false,
    };
    let v6 = RelationalV6;
    let v6k = RelationalV6K;
    let a = walk_batch(
        &refs,
        &v6,
        &mut CosineScorer {
            column: 0,
            cdim: v6.candidate_dim(8),
        },
        options,
    )
    .unwrap();
    let b = walk_batch(
        &refs,
        &v6k,
        &mut CosineScorer {
            column: 0,
            cdim: v6k.candidate_dim(8),
        },
        options,
    )
    .unwrap();
    for ((x, y), index) in a.iter().zip(&b).zip(&indexes) {
        assert_eq!(x.expanded, y.expanded, "{}", index.episode_id);
        assert_eq!(x.registered_at, y.registered_at);
        assert_eq!(x.registered_at_by_target, y.registered_at_by_target);
        assert_eq!(x.stop_reason, y.stop_reason);
        assert_eq!(x.examined, y.examined);
    }
    assert_eq!(a.len(), 18, "12 train + 6 screen fixture episodes");
}

/// A k >= 2 record is refused by a feature set that forms one query from one
/// target — the guard of §6 item 4, at the one place that pairs a record with
/// a feature set.
#[test]
fn a_two_target_record_is_refused_by_a_single_target_feature_set() {
    let (episodes, matrix, _) = k2_episodes(4);
    let index = EpisodeIndex::new(&episodes[0], &matrix, 8).unwrap();
    assert_eq!(index.target_count(), 2);
    let options = WalkOptions {
        stop_rule: StopRule::Exhaust,
        record_candidates: false,
        keep_items: false,
        with_prior: false,
    };
    let v6 = RelationalV6;
    let error = match walk_batch(
        &[&index],
        &v6,
        &mut CosineScorer {
            column: 0,
            cdim: v6.candidate_dim(8),
        },
        options,
    ) {
        Ok(_) => panic!("a k >= 2 record must be refused, not half-read"),
        Err(e) => e,
    };
    assert!(
        format!("{error}").contains("k >= 2"),
        "expected the k >= 2 refusal, got {error}"
    );
    // and relational-v6-k reads it
    let v6k = RelationalV6K;
    walk_batch(
        &[&index],
        &v6k,
        &mut CosineScorer {
            column: 0,
            cdim: v6k.candidate_dim(8),
        },
        options,
    )
    .expect("relational-v6-k reads a k = 2 record");
}

/// At k = 2 the walk registers per target, completes only when both are
/// registered, and carries the per-target registration and recall over k that
/// §4 reads. Recall at a budget takes the k + 1 values a k = 2 episode admits.
#[test]
fn the_k_two_walk_registers_per_target_and_completes_on_both() {
    let (episodes, matrix, _) = k2_episodes(24);
    let indexes: Vec<EpisodeIndex> = episodes
        .iter()
        .map(|e| EpisodeIndex::new(e, &matrix, 8).unwrap())
        .collect();
    let refs: Vec<&EpisodeIndex> = indexes.iter().collect();
    let v6k = RelationalV6K;
    let results = walk_batch(
        &refs,
        &v6k,
        &mut CosineScorer {
            column: 0,
            cdim: v6k.candidate_dim(8),
        },
        WalkOptions {
            stop_rule: StopRule::Exhaust,
            record_candidates: false,
            keep_items: false,
            with_prior: false,
        },
    )
    .unwrap();
    let mut both = 0;
    for (result, episode) in results.iter().zip(&episodes) {
        let shown = episode.visible.target_nodes.as_ref().unwrap();
        assert_eq!(shown.len(), 2);
        assert_eq!(result.registered_at_by_target.len(), 2);
        // completion is ALL k: `registered_at` is set only when both are
        assert_eq!(
            result.registered_at.is_some(),
            result.registered_at_by_target.iter().all(Option::is_some),
            "{}",
            episode.episode_id
        );
        if result.registered() {
            both += 1;
            assert_eq!(result.stop_reason, "target_registered");
            assert_eq!(
                result.registered_at,
                result
                    .registered_at_by_target
                    .iter()
                    .flatten()
                    .copied()
                    .max()
            );
        }
        let recall = result.recall_at_budget(20).unwrap();
        assert!(
            [0.0, 0.5, 1.0].contains(&recall),
            "recall over k = 2 is one of 0, 1/2, 1, not {recall}"
        );
    }
    assert!(both > 12, "only {both} of 24 episodes registered both");
}

/// The mask the reduction reads comes from the VISIBLE targets and the walk's
/// own examination history: registration lands exactly on the nodes
/// `target_nodes` names, in that order.
#[test]
fn registration_reads_the_visible_targets_in_their_own_order() {
    let (episodes, matrix, _) = k2_episodes(8);
    let v6k = RelationalV6K;
    for episode in &episodes {
        let index = EpisodeIndex::new(episode, &matrix, 8).unwrap();
        let shown = episode.visible.target_nodes.as_ref().unwrap();
        let named: Vec<&str> = index
            .targets_shown
            .iter()
            .map(|t| index.names[*t as usize].as_str())
            .collect();
        assert_eq!(named, shown.iter().map(String::as_str).collect::<Vec<_>>());
        let result = walk_batch(
            &[&index],
            &v6k,
            &mut CosineScorer {
                column: 0,
                cdim: v6k.candidate_dim(8),
            },
            WalkOptions {
                stop_rule: StopRule::Exhaust,
                record_candidates: false,
                keep_items: false,
                with_prior: false,
            },
        )
        .unwrap()
        .remove(0);
        // replay the expansion order and check each target's registration
        for (t, name) in shown.iter().enumerate() {
            let target = index.names.iter().position(|n| n == name).unwrap() as u32;
            let mut at = None;
            for (k, x) in result.expanded.iter().enumerate() {
                if index.out[*x as usize].contains(&target) {
                    at = Some(k + 1);
                    break;
                }
            }
            assert_eq!(
                result.registered_at_by_target[t], at,
                "{} target {name}",
                episode.episode_id
            );
        }
    }
}

/// The reduction switches: once one target registers, every query channel is
/// the cosine to the OTHER target alone. A max over ALL targets would not move.
#[test]
fn the_reduction_drops_a_registered_target_from_every_query_channel() {
    let (episodes, matrix, _) = k2_episodes(8);
    let v6k = RelationalV6K;
    let mut switched = 0;
    for episode in &episodes {
        let index = EpisodeIndex::new(episode, &matrix, 8).unwrap();
        let frontier = vec![hf_walk::Entry {
            node: index.start,
            parent: index.start,
            depth: 1,
            path_mean: index.emb(index.start).to_vec(),
        }];
        let expanded: Vec<u32> = vec![index.start];
        let parent_of = HashMap::new();
        let both = VisibleIndex::new(&index, &[true, true]);
        let only_second = VisibleIndex::new(&index, &[false, true]);
        let a = v6k.build(both, &frontier, &expanded, &parent_of);
        let b = v6k.build(only_second, &frontier, &expanded, &parent_of);
        // the share column moves from 1 to 1/2 whatever the cosines do
        assert_eq!(a.cand[EXTRA_AT + 2], 1.0);
        assert_eq!(b.cand[EXTRA_AT + 2], 0.5);
        // and with one target left the spread is zero and min == max
        assert_eq!(b.cand[EXTRA_AT + 1], 0.0);
        assert_eq!(b.cand[EXTRA_AT], b.cand[0]);
        let single = hf_walk::cosine(index.emb(index.start), index.query_of(1));
        assert_eq!(
            b.cand[0], single,
            "the maximum over one target is its cosine"
        );
        if a.cand[0] != b.cand[0] {
            switched += 1;
        }
    }
    assert!(switched > 0, "no episode's query channel moved at all");
}

/// Walking a k = 2 batch in lockstep equals walking each episode alone,
/// decision for decision, with a scorer that is a pure function of the row.
#[test]
fn batched_and_single_k_two_walks_agree_decision_for_decision() {
    let (episodes, matrix, _) = k2_episodes(16);
    let indexes: Vec<EpisodeIndex> = episodes
        .iter()
        .map(|e| EpisodeIndex::new(e, &matrix, 8).unwrap())
        .collect();
    let refs: Vec<&EpisodeIndex> = indexes.iter().collect();
    let v6k = RelationalV6K;
    let cdim = v6k.candidate_dim(8);
    let options = WalkOptions {
        stop_rule: StopRule::Exhaust,
        record_candidates: false,
        keep_items: false,
        with_prior: false,
    };
    let batched = walk_batch(&refs, &v6k, &mut HashScorer { cdim }, options).unwrap();
    let mut decisions = 0;
    for (i, index) in indexes.iter().enumerate() {
        let alone = walk_batch(&[index], &v6k, &mut HashScorer { cdim }, options)
            .unwrap()
            .remove(0);
        assert_eq!(alone.expanded, batched[i].expanded, "{}", index.episode_id);
        assert_eq!(alone.registered_at, batched[i].registered_at);
        assert_eq!(
            alone.registered_at_by_target,
            batched[i].registered_at_by_target
        );
        assert_eq!(alone.stop_reason, batched[i].stop_reason);
        assert_eq!(alone.decisions.len(), batched[i].decisions.len());
        for (a, b) in alone.decisions.iter().zip(&batched[i].decisions) {
            assert_eq!(a.frontier, b.frontier);
            assert_eq!(a.chosen, b.chosen);
            assert_eq!(a.cosines, b.cosines);
            decisions += 1;
        }
    }
    assert!(decisions > 100, "only {decisions} decisions compared");
}

/// k-greedy registers both targets on the fixture's k = 2 episodes, and its
/// recall at `B_fix` is the stratum field §4 names.
#[test]
fn k_greedy_registers_both_targets_on_the_fixture_episodes() {
    let (episodes, _, embeddings) = k2_episodes(24);
    let mut both = 0;
    let mut recalls: Vec<f64> = Vec::new();
    for episode in &episodes {
        let g = hf_policies::EpisodeGraph::from_episode_k(episode);
        assert_eq!(g.target_count(), 2);
        let trace = hf_policies::k_greedy_trace(&g, &embeddings);
        let recall = trace.recall_at_budget(g.b_fix()).unwrap();
        assert!([0.0, 0.5, 1.0].contains(&recall));
        recalls.push(recall);
        if trace.registered_targets() == 2 {
            both += 1;
        }
        // the frozen variant is reported beside and never below on registration
        let frozen = hf_policies::k_greedy_frozen_trace(&g, &embeddings);
        assert_eq!(frozen.registered_at_by_target.len(), 2);
    }
    assert_eq!(
        both, 24,
        "k-greedy registers both targets on every easy fixture episode"
    );
    // and §4's strata are both non-empty on this fixture: recall over k at
    // B_fix takes all three values a k = 2 episode admits (measured here as
    // 6 / 9 / 9 of 24 — the fixture world is a random digraph, not evidence
    // about any real pool)
    let zero = recalls.iter().filter(|r| **r == 0.0).count();
    let half = recalls.iter().filter(|r| **r == 0.5).count();
    let one = recalls.iter().filter(|r| **r == 1.0).count();
    assert_eq!(zero + half + one, 24, "recall over k = 2 is 0, 1/2 or 1");
    assert!(one > 0 && zero + half > 0, "{zero} / {half} / {one}");
}
