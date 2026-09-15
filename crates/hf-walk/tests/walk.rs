//! The batched walk against the baselines and against itself: with a scorer
//! that returns the cosine column, the walk is similarity-greedy node for
//! node on the fixture episodes (untyped graph, so the two adjacency orders
//! coincide); with a deterministic pseudo-random scorer, walking 64 episodes
//! in one batch equals walking each alone, decision for decision; the v6
//! feature set carries no raw coordinate; the candidate records satisfy the
//! ranking probe's gate 2 invariants.

use std::collections::HashMap;
use std::path::PathBuf;

use hf_walk::{
    walk_batch, DecisionBatch, EpisodeIndex, FeatureSet, RawV5, RelationalV6, Scored, Scorer,
    StopRule, WalkOptions, STOP_DIM,
};
use sha2::{Digest, Sha256};

fn fixture_cache(
    embeddings: &HashMap<String, Vec<f64>>,
    dir: &std::path::Path,
) -> hf_embed::EmbeddingMatrix {
    std::fs::create_dir_all(dir).unwrap();
    let mut names: Vec<&String> = embeddings.keys().collect();
    names.sort();
    let mut f = std::fs::File::create(dir.join("vectors.jsonl")).unwrap();
    for n in &names {
        hf_embed::append_vector(&mut f, n, &embeddings[*n]).unwrap();
    }
    let m = hf_embed::Manifest {
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
    };
    hf_embed::write_manifest(dir, &m).unwrap();
    hf_embed::EmbeddingMatrix::load(dir).unwrap()
}

fn episodes() -> (
    Vec<hf_io::RealEpisode>,
    hf_embed::EmbeddingMatrix,
    HashMap<String, Vec<f64>>,
) {
    let goldens =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../hf-io/tests/goldens/fixture-split");
    let (_, embeddings) = hf_episodes::fixture::fixture_world(5, 400, 1600, 8);
    let dir = std::env::temp_dir().join(format!("hf-walk-{}", std::process::id()));
    let cache = fixture_cache(&embeddings, &dir);
    let mut all = Vec::new();
    for name in ["train", "screen", "train-greedy"] {
        all.extend(hf_io::read_split(&goldens.join(name)).unwrap().0);
    }
    (all, cache, embeddings)
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

/// A pure function of the scoring input: sha256 of the candidate row bytes.
struct HashScorer {
    cdim: usize,
}

impl Scorer for HashScorer {
    fn score(&mut self, batch: &DecisionBatch) -> Result<Scored, hf_core::HfError> {
        let scores = batch
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
            .collect();
        Ok(Scored {
            scores,
            residuals: None,
        })
    }
    fn stop_logits(&mut self, rows: &[[f32; STOP_DIM]]) -> Result<Vec<f32>, hf_core::HfError> {
        Ok(rows
            .iter()
            .map(|r| if r[0] > 0.1 { 1.0 } else { -1.0 })
            .collect())
    }
}

#[test]
fn cosine_scored_walk_is_similarity_greedy_on_every_fixture_episode() {
    let (episodes, cache, embeddings) = episodes();
    let indexes: Vec<EpisodeIndex> = episodes
        .iter()
        .map(|e| EpisodeIndex::new(e, &cache, 8).unwrap())
        .collect();
    let refs: Vec<&EpisodeIndex> = indexes.iter().collect();
    let features = RawV5;
    let mut scorer = CosineScorer {
        column: features.cosine_column(8),
        cdim: features.candidate_dim(8),
    };
    let results = walk_batch(
        &refs,
        &features,
        &mut scorer,
        WalkOptions {
            stop_rule: StopRule::Exhaust,
            record_candidates: true,
            keep_items: false,
            with_prior: false,
        },
    )
    .unwrap();
    for ((episode, index), result) in episodes.iter().zip(&indexes).zip(&results) {
        let g = hf_policies::EpisodeGraph::from_episode(episode);
        let greedy = hf_policies::similarity_greedy_trace(&g, &embeddings, None);
        let expanded: Vec<&str> = result
            .expanded
            .iter()
            .map(|n| index.names[*n as usize].as_str())
            .collect();
        assert_eq!(expanded, greedy.examined, "{}", episode.episode_id);
        assert_eq!(
            result.registered_at.map(|r| r as u32),
            greedy.registered_at,
            "{}",
            episode.episode_id
        );
        assert_eq!(result.stop_reason, greedy.stop_reason);
        // gate 2: every dumped decision is an argmax and the chosen node is the next expansion
        let records = result.candidates.as_ref().unwrap();
        assert_eq!(records.len(), result.expanded.len() - 1);
        for (k, rec) in records.iter().enumerate() {
            let best = rec.scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(rec.scores[rec.chosen], best);
            assert_eq!(
                rec.frontier[rec.chosen],
                index.names[result.expanded[k + 1] as usize]
            );
            assert_eq!(rec.scores.len(), rec.cosines.len());
        }
        assert_eq!(result.margins.len(), result.decisions.len());
        assert_eq!(result.cosine_margins.len(), result.decisions.len());
    }
}

#[test]
fn batched_and_single_walks_agree_decision_for_decision() {
    let (episodes, cache, _) = episodes();
    let indexes: Vec<EpisodeIndex> = episodes
        .iter()
        .map(|e| EpisodeIndex::new(e, &cache, 8).unwrap())
        .collect();
    for (features, name) in [
        (&RawV5 as &dyn FeatureSet, "raw"),
        (&RelationalV6 as &dyn FeatureSet, "v6"),
    ] {
        for rule in [StopRule::Exhaust, StopRule::Learned] {
            let options = WalkOptions {
                stop_rule: rule,
                record_candidates: true,
                keep_items: true,
                with_prior: false,
            };
            let refs: Vec<&EpisodeIndex> = indexes.iter().collect();
            let mut scorer = HashScorer {
                cdim: features.candidate_dim(8),
            };
            let batched = walk_batch(&refs, features, &mut scorer, options).unwrap();
            for (i, index) in indexes.iter().enumerate() {
                let single = walk_batch(&[index], features, &mut scorer, options)
                    .unwrap()
                    .remove(0);
                assert_eq!(
                    single.expanded, batched[i].expanded,
                    "{name} {rule:?} {}",
                    index.episode_id
                );
                assert_eq!(single.stop_reason, batched[i].stop_reason);
                assert_eq!(single.examined, batched[i].examined);
                assert_eq!(single.decisions.len(), batched[i].decisions.len());
                for (a, b) in single.decisions.iter().zip(&batched[i].decisions) {
                    assert_eq!(a.chosen, b.chosen);
                    assert_eq!(a.frontier, b.frontier);
                    assert_eq!(a.stop_features, b.stop_features);
                    assert_eq!(a.item.as_ref().unwrap().cand, b.item.as_ref().unwrap().cand);
                    assert_eq!(a.item.as_ref().unwrap().pair, b.item.as_ref().unwrap().pair);
                }
            }
            if rule == StopRule::Learned {
                assert!(
                    batched.iter().any(|r| r.stop_reason == "learned_stop"),
                    "{name}: the stub stop head fires somewhere"
                );
            }
        }
    }
}

#[test]
fn relational_features_carry_no_raw_coordinate() {
    let (episodes, cache, _) = episodes();
    let index = EpisodeIndex::new(&episodes[0], &cache, 8).unwrap();
    let features = RelationalV6;
    let mut scorer = HashScorer {
        cdim: features.candidate_dim(8),
    };
    let result = walk_batch(
        &[&index],
        &features,
        &mut scorer,
        WalkOptions {
            stop_rule: StopRule::Exhaust,
            record_candidates: false,
            keep_items: true,
            with_prior: false,
        },
    )
    .unwrap()
    .remove(0);
    for d in &result.decisions {
        let item = d.item.as_ref().unwrap();
        assert_eq!(
            item.cand.len(),
            d.frontier.len() * features.candidate_dim(8)
        );
        assert_eq!(
            item.ctx.len(),
            d.expansions_before * features.context_dim(8)
        );
        assert_eq!(
            item.pair.len(),
            d.frontier.len() * d.expansions_before * features.pair_dim()
        );
        assert!(item.query.is_empty(), "no query token in v6");
        for v in item.cand.iter().chain(&item.ctx).chain(&item.pair) {
            assert!(v.is_finite());
        }
        // every relational value is a cosine, a rank, a z-score or a bounded structure feature
        for i in 0..d.frontier.len() {
            let row =
                &item.cand[i * features.candidate_dim(8)..(i + 1) * features.candidate_dim(8)];
            for (j, v) in row.iter().enumerate() {
                if j != 7 {
                    assert!(v.abs() <= 1.0 + 1e-5, "column {j} = {v}");
                }
            }
            assert_eq!(row[0], d.cosines[i], "column 0 is cos(c, q)");
        }
    }
}
