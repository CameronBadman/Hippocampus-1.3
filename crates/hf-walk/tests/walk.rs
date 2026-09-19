//! The batched walk against the baselines and against itself: with a scorer
//! that returns the cosine column, the walk is similarity-greedy node for
//! node on the fixture episodes (untyped graph, so the two adjacency orders
//! coincide); with a deterministic pseudo-random scorer, walking 64 episodes
//! in one batch equals walking each alone, decision for decision; the v6
//! feature set carries no raw coordinate; the candidate records satisfy the
//! ranking probe's gate 2 invariants. The view a builder receives carries no
//! accessor for anything the sampler kept back; `relational-v6-prev` leaves
//! the candidate row of `relational-v6` untouched and its two previous-node
//! columns agree between the context token and the pair channel.

use std::collections::HashMap;
use std::path::PathBuf;

use hf_walk::{
    walk_batch, DecisionBatch, EpisodeIndex, FeatureSet, RawV5, RelationalV6, RelationalV6Prev,
    Scored, Scorer, StopRule, WalkOptions, STOP_DIM,
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
            assert_eq!(rec.parents.len(), rec.frontier.len());
            assert_eq!(rec.depths.len(), rec.frontier.len());
            let d = &result.decisions[k];
            let want_parents: Vec<&str> = d
                .parents
                .iter()
                .map(|p| index.names[*p as usize].as_str())
                .collect();
            assert_eq!(rec.parents, want_parents);
            assert_eq!(rec.depths, d.depths);
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
        (&RelationalV6Prev as &dyn FeatureSet, "v6-prev"),
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
                    assert_eq!(a.item.as_ref().unwrap().ctx, b.item.as_ref().unwrap().ctx);
                    assert_eq!(a.item.as_ref().unwrap().pair, b.item.as_ref().unwrap().pair);
                    assert_eq!(a.parents, b.parents);
                    assert_eq!(a.depths, b.depths);
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

/// A hand-built episode with no target to register, so the walk runs to
/// exhaustion and every expansion order is observable: `s -> a -> b` and
/// `s -> c -> t`, embeddings chosen so the cosine-greedy walk expands
/// s, a, b, c, t in that order. At the decision whose expansions are
/// [s, a, b, c], b is abandoned (no child of b expanded since, and c, which is
/// not a child of b, was) while a is not (its child b was expanded since).
fn abandonment_episode() -> EpisodeIndex {
    let names: Vec<String> = ["s", "a", "b", "c", "t"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    let rows: [[f32; 2]; 5] = [
        [0.0, 1.0],
        [0.9, 0.435_889_9],
        [0.8, 0.6],
        [0.1, 0.994_987_4],
        [1.0, 0.0],
    ];
    let emb: Vec<f32> = rows.iter().flatten().copied().collect();
    let unit = emb.clone(); // every row above is already unit length
    EpisodeIndex {
        episode_id: "hand-built".into(),
        names,
        start: 0,
        target_shown: Some(4),
        hidden_targets: Vec::new(),
        out: vec![vec![1, 3], vec![2], vec![], vec![4], vec![]],
        edim: 2,
        emb,
        unit,
        query: vec![1.0, 0.0],
        on_path: vec![false; 5],
        distance: vec![None; 5],
        removed_count: 0,
        greedy_overshoot: None,
    }
}

#[test]
fn the_abandonment_and_recency_columns_read_the_expansion_order() {
    let index = abandonment_episode();
    let features = RelationalV6Prev;
    let mut scorer = CosineScorer {
        column: features.cosine_column(2),
        cdim: features.candidate_dim(2),
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
    let order: Vec<&str> = result
        .expanded
        .iter()
        .map(|n| index.names[*n as usize].as_str())
        .collect();
    assert_eq!(order, ["s", "a", "b", "c", "t"]);
    let ctx_dim = features.context_dim(2);
    let pair_dim = features.pair_dim();
    // the decision whose expansions are [s, a, b, c]
    let d = &result.decisions[3];
    assert_eq!(d.expansions_before, 4);
    let item = d.item.as_ref().unwrap();
    let abandoned: Vec<f32> = (0..4).map(|k| item.ctx[k * ctx_dim + 4]).collect();
    let recency: Vec<f32> = (0..4).map(|k| item.ctx[k * ctx_dim + 5]).collect();
    assert_eq!(abandoned, vec![0.0, 0.0, 1.0, 0.0], "s, a, b, c");
    assert_eq!(
        recency,
        vec![4.0 / 16.0, 3.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0]
    );
    // the pair channel carries the same two columns for every candidate
    for i in 0..item.frontier_len {
        for k in 0..4 {
            let base = (i * 4 + k) * pair_dim;
            assert_eq!(item.pair[base + 3], abandoned[k]);
            assert_eq!(item.pair[base + 4], recency[k]);
        }
    }
    // one expansion earlier, nothing has been abandoned yet
    let earlier = result.decisions[2].item.as_ref().unwrap();
    assert_eq!(result.decisions[2].expansions_before, 3);
    assert_eq!(
        (0..3)
            .map(|k| earlier.ctx[k * ctx_dim + 4])
            .collect::<Vec<f32>>(),
        vec![0.0, 0.0, 0.0],
        "s, a, b"
    );
    assert_eq!(
        (0..3)
            .map(|k| earlier.ctx[k * ctx_dim + 5])
            .collect::<Vec<f32>>(),
        vec![3.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0]
    );
}

#[test]
fn relational_v6_prev_keeps_the_v6_candidate_row_and_extends_the_other_channels() {
    let (episodes, cache, _) = episodes();
    let indexes: Vec<EpisodeIndex> = episodes
        .iter()
        .map(|e| EpisodeIndex::new(e, &cache, 8).unwrap())
        .collect();
    let refs: Vec<&EpisodeIndex> = indexes.iter().collect();
    let options = WalkOptions {
        stop_rule: StopRule::Exhaust,
        record_candidates: false,
        keep_items: true,
        with_prior: false,
    };
    // the hash scorer reads the candidate rows only, so identical rows give
    // identical walks: any divergence here is a divergence in the row
    let mut scorer = HashScorer {
        cdim: RelationalV6.candidate_dim(8),
    };
    let base = walk_batch(&refs, &RelationalV6, &mut scorer, options).unwrap();
    let prev = walk_batch(&refs, &RelationalV6Prev, &mut scorer, options).unwrap();
    assert_eq!(
        RelationalV6Prev.candidate_dim(8),
        RelationalV6.candidate_dim(8)
    );
    assert_eq!(RelationalV6Prev.context_dim(8), 6);
    assert_eq!(RelationalV6Prev.pair_dim(), 5);
    let mut decisions = 0usize;
    for (i, (b, p)) in base.iter().zip(&prev).enumerate() {
        assert_eq!(b.expanded, p.expanded, "{}", indexes[i].episode_id);
        assert_eq!(b.decisions.len(), p.decisions.len());
        for (db, dp) in b.decisions.iter().zip(&p.decisions) {
            decisions += 1;
            let (ib, ip) = (db.item.as_ref().unwrap(), dp.item.as_ref().unwrap());
            assert_eq!(ib.cand, ip.cand, "the candidate row is v6's, unchanged");
            for k in 0..db.expansions_before {
                assert_eq!(ip.ctx[k * 6..k * 6 + 4], ib.ctx[k * 4..k * 4 + 4]);
                for i in 0..db.frontier.len() {
                    let (bb, bp) = (
                        (i * db.expansions_before + k) * 3,
                        (i * db.expansions_before + k) * 5,
                    );
                    assert_eq!(ip.pair[bp..bp + 3], ib.pair[bb..bb + 3]);
                    assert_eq!(ip.pair[bp + 3], ip.ctx[k * 6 + 4]);
                    assert_eq!(ip.pair[bp + 4], ip.ctx[k * 6 + 5]);
                }
            }
        }
    }
    assert!(decisions > 100, "only {decisions} decisions compared");
}

#[test]
fn the_visible_view_exposes_nothing_the_sampler_kept_back() {
    let source = include_str!("../src/visible.rs");
    for name in [
        "hidden_targets",
        "on_path",
        "distance",
        "removed_count",
        "greedy_overshoot",
    ] {
        assert!(
            !source.contains(name),
            "{name} appears in the view a feature builder receives"
        );
    }
    // the whole state of the view, and the whole of its surface
    let fields: Vec<&str> = source
        .split("pub struct VisibleIndex<'a> {")
        .nth(1)
        .unwrap()
        .split("\n}")
        .next()
        .unwrap()
        .lines()
        .filter_map(|l| l.trim().split(':').next())
        .filter(|l| !l.is_empty())
        .collect();
    assert_eq!(
        fields,
        [
            "names",
            "start",
            "target_shown",
            "out",
            "edim",
            "emb",
            "unit",
            "query"
        ]
    );
    let mut methods: Vec<&str> = source
        .lines()
        .filter_map(|l| l.trim().strip_prefix("pub fn "))
        .filter_map(|l| l.split(['(', '<']).next())
        .collect();
    methods.sort_unstable();
    assert_eq!(
        methods,
        [
            "degree",
            "edim",
            "emb",
            "name",
            "names",
            "new",
            "node_count",
            "out",
            "query",
            "start",
            "target_shown",
            "unit"
        ]
    );
}

/// A k ≥ 2 record is refused, not half-read: the builder makes one query from
/// one shown target, and `K_TARGETS_DESIGN.md` §6 item 4's builder is not
/// written. The guard is on the visible side, where the shown targets are.
#[test]
fn a_two_target_record_is_refused_by_the_single_target_builder() {
    let (episodes, cache, _) = episodes();
    let mut episode = episodes[0].clone();
    EpisodeIndex::new(&episode, &cache, 8).expect("a k = 1 record still builds");
    let first = episode.visible.target_node.clone().unwrap();
    let second = episode.visible.nodes[1].node.clone();
    let mut shown = vec![first, second];
    shown.sort();
    episode.visible.schema_version = hf_io::SCHEMA_VERSION_V6.into();
    episode.visible.target_node = Some(shown[0].clone());
    episode.visible.target_nodes = Some(shown);
    let error = match EpisodeIndex::new(&episode, &cache, 8) {
        Ok(_) => panic!("a k >= 2 record must be refused, not half-read"),
        Err(e) => e,
    };
    assert!(
        format!("{error}").contains("k >= 2"),
        "expected the k >= 2 refusal, got {error}"
    );
}
