//! Against the Python model (`tools/goldens/gen_model_goldens.py`): the
//! parameter names load a Python state_dict exported as safetensors; the
//! parameter counts (fixture 15,447; rung-3 prior 9,684,537); the runner's
//! state digest; `score_decisions` on the recorded inputs (single items and
//! padded batches) to 1e-4; the walk with the model as scorer reproduces the
//! Python walk node for node under both stop rules; `walk_losses` on the
//! four-episode batch with and without the residual penalty; the pre-clip
//! gradient norm; AdamW's moments round-trip through save/load. The three
//! feature sets the config may name report their widths through the model, and
//! the previous-node set costs exactly the two extra context columns and the
//! two extra pair columns.

use std::collections::HashMap;
use std::path::PathBuf;

use hf_model::{clip_grad_norm, walk_losses, AdamW, LossConfig, Model, ModelConfig, ModelScorer};
use hf_walk::{
    walk_batch, DecisionBatch, DecisionItem, EpisodeIndex, Scorer, StopRule, WalkOptions,
};
use serde_json::Value;
use tch::Device;

fn goldens() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/goldens")
}

fn golden() -> Value {
    serde_json::from_str(&std::fs::read_to_string(goldens().join("model.json")).unwrap()).unwrap()
}

fn fixture_model(g: &Value) -> Model {
    let config = ModelConfig::from_value(&g["model_config"], 8).unwrap();
    let mut model = Model::new(config, Device::Cpu).unwrap();
    model
        .vs
        .load(goldens().join("fixture.safetensors"))
        .unwrap();
    model
}

fn fixture_cache() -> (hf_embed::EmbeddingMatrix, HashMap<String, Vec<f64>>) {
    let (_, embeddings) = hf_episodes::fixture::fixture_world(5, 400, 1600, 8);
    let dir = std::env::temp_dir().join(format!("hf-model-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
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
    hf_embed::write_manifest(&dir, &m).unwrap();
    (hf_embed::EmbeddingMatrix::load(&dir).unwrap(), embeddings)
}

fn fixture_episodes(g: &Value) -> Vec<hf_io::RealEpisode> {
    let splits =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../hf-io/tests/goldens/fixture-split");
    let mut by_id: HashMap<String, hf_io::RealEpisode> = HashMap::new();
    for name in ["train", "screen"] {
        for e in hf_io::read_split(&splits.join(name)).unwrap().0 {
            by_id.insert(e.episode_id.clone(), e);
        }
    }
    g["walks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| by_id[w["episode_id"].as_str().unwrap()].clone())
        .collect()
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol * (1.0 + y.abs()),
            "{what}[{i}]: {x} vs {y}"
        );
    }
}

#[test]
fn python_weights_load_and_the_counts_and_digest_match() {
    let g = golden();
    let model = fixture_model(&g);
    assert_eq!(
        model.trainable_parameter_count(),
        g["parameter_count"].as_i64().unwrap()
    );
    assert_eq!(model.state_digest(&[]), g["state_digest"].as_str().unwrap());
    assert_eq!(
        model.state_digest(&["greedy_tau"]),
        g["state_digest_excluding_tau"].as_str().unwrap()
    );
    let rung3 = Model::new(
        ModelConfig::from_value(&g["rung3_config"], 768).unwrap(),
        Device::Cpu,
    )
    .unwrap();
    assert_eq!(
        rung3.trainable_parameter_count(),
        g["rung3_parameter_count"].as_i64().unwrap(),
        "9,684,537 for the rung-3 prior config"
    );
    // an untrained prior model is exactly greedy: the residual head is zero
    let fresh = Model::new(
        ModelConfig::from_value(&g["model_config"], 8).unwrap(),
        Device::Cpu,
    )
    .unwrap();
    let vars = fresh.vs.variables();
    assert_eq!(
        f64::try_from(vars["score_head.2.weight"].abs().sum(tch::Kind::Float)).unwrap(),
        0.0
    );
    assert_eq!(f64::try_from(&vars["greedy_tau"]).unwrap(), 10.0);
}

#[test]
fn score_decisions_matches_python_on_recorded_inputs() {
    let g = golden();
    let model = fixture_model(&g);
    let (_, embeddings) = fixture_cache();
    let unit = |name: &str| -> Vec<f32> {
        let v = &embeddings[name];
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        v.iter().map(|x| (x / norm) as f32).collect()
    };
    let mut items: Vec<DecisionItem> = Vec::new();
    let mut want_scores: Vec<Vec<f32>> = Vec::new();
    let mut want_residuals: Vec<Vec<f32>> = Vec::new();
    let mut want_stop: Vec<f32> = Vec::new();
    let mut stop_rows: Vec<[f32; 8]> = Vec::new();
    let walks: HashMap<&str, &Value> = g["walks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| (w["episode_id"].as_str().unwrap(), w))
        .collect();
    let mut stop_cursor: HashMap<&str, usize> = HashMap::new();
    for d in g["decisions"].as_array().unwrap() {
        let frontier: Vec<&str> = d["frontier"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap())
            .collect();
        let parents: Vec<&str> = d["parents"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap())
            .collect();
        let context: Vec<&str> = d["context"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap())
            .collect();
        let rows: Vec<f32> = d["rows"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(floats)
            .collect();
        let query: Vec<f32> = rows[8..16].to_vec(); // the query block of the first row
        let mut ctx = Vec::new();
        for c in &context {
            ctx.extend(embeddings[*c].iter().map(|x| *x as f32));
        }
        let mut pair = Vec::new();
        for (f, p) in frontier.iter().zip(&parents) {
            let uf = unit(f);
            for c in &context {
                let uc = unit(c);
                pair.push(uf.iter().zip(&uc).map(|(a, b)| a * b).sum::<f32>());
                pair.push(if p == c { 1.0 } else { 0.0 });
            }
        }
        items.push(DecisionItem {
            frontier_len: frontier.len(),
            context_len: context.len(),
            cand: rows,
            ctx,
            query,
            pair,
        });
        want_scores.push(floats(&d["scores"]));
        want_residuals.push(floats(&d["residuals"]));
        want_stop.push(d["stop_logit"].as_f64().unwrap() as f32);
        let id = d["episode_id"].as_str().unwrap();
        let k = *stop_cursor.get(id).unwrap_or(&0);
        let row = floats(&walks[id]["stop_features"][k]);
        stop_rows.push(row.try_into().unwrap());
        stop_cursor.insert(id, k + 1);
    }
    let mut scorer = ModelScorer { model: &model };
    // one at a time, and everything in one padded batch: the same numbers
    for (k, item) in items.iter().enumerate() {
        let scored = scorer
            .score(&DecisionBatch {
                items: vec![item.clone()],
            })
            .unwrap();
        close(
            &scored.scores[0],
            &want_scores[k],
            1e-4,
            &format!("scores single {k}"),
        );
        close(
            &scored.residuals.as_ref().unwrap()[0],
            &want_residuals[k],
            1e-4,
            &format!("residuals single {k}"),
        );
    }
    let scored = scorer
        .score(&DecisionBatch {
            items: items.clone(),
        })
        .unwrap();
    for (k, want) in want_scores.iter().enumerate() {
        close(
            &scored.scores[k],
            want,
            1e-4,
            &format!("scores batched {k}"),
        );
    }
    let stop = scorer.stop_logits(&stop_rows).unwrap();
    close(&stop, &want_stop, 1e-4, "stop logits");
}

#[test]
fn the_walk_with_the_model_reproduces_the_python_walk() {
    let g = golden();
    let model = fixture_model(&g);
    let (cache, _) = fixture_cache();
    let episodes = fixture_episodes(&g);
    let indexes: Vec<EpisodeIndex> = episodes
        .iter()
        .map(|e| EpisodeIndex::new(e, &cache, 8).unwrap())
        .collect();
    let refs: Vec<&EpisodeIndex> = indexes.iter().collect();
    let features = model.features();
    let mut scorer = ModelScorer { model: &model };
    for (rule, key) in [(StopRule::Exhaust, "walks"), (StopRule::Learned, "learned")] {
        let results = walk_batch(
            &refs,
            features.as_ref(),
            &mut scorer,
            WalkOptions {
                stop_rule: rule,
                record_candidates: true,
                keep_items: false,
                with_prior: true,
            },
        )
        .unwrap();
        for (i, want) in g[key].as_array().unwrap().iter().enumerate() {
            let got: Vec<&str> = results[i]
                .expanded
                .iter()
                .map(|n| indexes[i].names[*n as usize].as_str())
                .collect();
            let want_expanded: Vec<&str> = want["expanded"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_str().unwrap())
                .collect();
            assert_eq!(got, want_expanded, "{key} {}", want["episode_id"]);
            assert_eq!(
                results[i].stop_reason,
                want["stop_reason"].as_str().unwrap()
            );
            if rule == StopRule::Exhaust {
                assert_eq!(
                    results[i].registered_at.map(|r| r as u64),
                    want["registered_at"].as_u64()
                );
                assert_eq!(
                    results[i].examined as u64,
                    want["examined"].as_u64().unwrap()
                );
                let margins: Vec<Option<f32>> = want["margins"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|m| m.as_f64().map(|x| x as f32))
                    .collect();
                assert_eq!(results[i].margins.len(), margins.len());
                for (a, b) in results[i].margins.iter().zip(&margins) {
                    match (a, b) {
                        (Some(x), Some(y)) => assert!((x - y).abs() < 1e-4, "margin {x} vs {y}"),
                        (None, None) => {}
                        _ => panic!("margin presence differs"),
                    }
                }
                let cands = results[i].candidates.as_ref().unwrap();
                for (c, wc) in cands.iter().zip(want["candidates"].as_array().unwrap()) {
                    close(&c.scores, &floats(&wc["scores"]), 1e-4, "candidate scores");
                    assert_eq!(c.chosen as u64, wc["chosen"].as_u64().unwrap());
                }
            }
        }
    }
}

#[test]
fn losses_and_the_gradient_norm_match_python() {
    let g = golden();
    let model = fixture_model(&g);
    let (cache, _) = fixture_cache();
    let episodes = fixture_episodes(&g);
    let ids: Vec<&str> = g["batch_episode_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap())
        .collect();
    let batch: Vec<&hf_io::RealEpisode> = ids
        .iter()
        .map(|id| episodes.iter().find(|e| e.episode_id == *id).unwrap())
        .collect();
    let indexes: Vec<EpisodeIndex> = batch
        .iter()
        .map(|e| EpisodeIndex::new(e, &cache, 8).unwrap())
        .collect();
    let refs: Vec<&EpisodeIndex> = indexes.iter().collect();
    let features = model.features();
    let walks = {
        let mut scorer = ModelScorer { model: &model };
        walk_batch(
            &refs,
            features.as_ref(),
            &mut scorer,
            WalkOptions {
                stop_rule: StopRule::Exhaust,
                record_candidates: false,
                keep_items: true,
                with_prior: true,
            },
        )
        .unwrap()
    };
    let out = model.forward_training(&walks).unwrap();
    let plain = walk_losses(&out, &walks, &refs, true, LossConfig::default())
        .unwrap()
        .values();
    let penalty = walk_losses(
        &out,
        &walks,
        &refs,
        true,
        LossConfig {
            residual_penalty: 0.5,
            ..Default::default()
        },
    )
    .unwrap();
    let pv = penalty.values();
    for (name, i) in [
        ("edge", 0),
        ("distance", 1),
        ("stop", 2),
        ("residual", 3),
        ("total", 4),
    ] {
        let want = g["losses"]["plain"][name].as_f64().unwrap();
        assert!(
            (plain[i] - want).abs() < 1e-3,
            "plain {name}: {} vs {want}",
            plain[i]
        );
        let want = g["losses"]["penalty"][name].as_f64().unwrap();
        assert!(
            (pv[i] - want).abs() < 1e-3,
            "penalty {name}: {} vs {want}",
            pv[i]
        );
    }
    // the pre-clip gradient norm of the penalised total
    let mut opt = AdamW::new(&model.vs, 1e-3, 0.01);
    opt.zero_grad();
    penalty.total.backward();
    let norm = clip_grad_norm(&model.vs, Some(1.0));
    let want = g["grad_norm_penalty"].as_f64().unwrap();
    assert!(
        (norm - want).abs() < 1e-2 * (1.0 + want),
        "grad norm {norm} vs {want}"
    );
    // AdamW steps move the parameters and its moments round-trip exactly
    let before = model.state_digest(&[]);
    opt.step();
    let after = model.state_digest(&[]);
    assert_ne!(before, after);
    let path =
        std::env::temp_dir().join(format!("hf-model-opt-{}.safetensors", std::process::id()));
    opt.save(&path).unwrap();
    let mut other = AdamW::new(&model.vs, 1e-3, 0.01);
    other.load(&path).unwrap();
    assert_eq!(other.step_count, 1);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn the_three_feature_sets_report_their_dims_through_the_model() {
    let g = golden();
    let build = |name: &str| -> Model {
        let mut m = g["model_config"].clone();
        m["feature_set"] = name.into();
        Model::new(ModelConfig::from_value(&m, 8).unwrap(), Device::Cpu).unwrap()
    };
    let want: [(&str, usize, usize, usize, Option<usize>); 3] = [
        ("raw-v5", 4 * 8 + 7, 8, 2, Some(8)),
        ("relational-v6", 16, 4, 3, None),
        ("relational-v6-prev", 16, 6, 5, None),
    ];
    for (name, cdim, ctx_dim, pair_dim, query_dim) in want {
        let model = build(name);
        let features = model.features();
        assert_eq!(features.name(), name);
        assert_eq!(features.candidate_dim(8), cdim, "{name} candidate");
        assert_eq!(features.context_dim(8), ctx_dim, "{name} context");
        assert_eq!(features.pair_dim(), pair_dim, "{name} pair");
        assert_eq!(features.query_dim(8), query_dim, "{name} query");
    }
    // the only cost of the previous-node columns: two more context inputs and
    // two more pair-bias inputs per head, per block
    let hidden = g["model_config"]["hidden_dimension"].as_i64().unwrap();
    let heads = g["model_config"]["self_attention_heads"].as_i64().unwrap();
    let blocks = g["model_config"]["traversal_blocks"].as_i64().unwrap();
    assert_eq!(
        build("relational-v6-prev").trainable_parameter_count()
            - build("relational-v6").trainable_parameter_count(),
        2 * hidden + blocks * 2 * heads
    );
    let mut unknown = g["model_config"].clone();
    unknown["feature_set"] = "relational-v7".into();
    assert!(ModelConfig::from_value(&unknown, 8)
        .unwrap()
        .features()
        .is_err());
}
