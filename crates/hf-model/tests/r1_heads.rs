//! The R1 heads (`R1_HEAD_DESIGN.md` §8 ENG-2) and the losses (ENG-4 (a)).

use hf_embed::EmbeddingMatrix;
use hf_model::{
    r1, AdamW, Model, ModelConfig, ModelScorer, R1Heads, R1_HEAD_PATTERNS, RETURN_HEAD_PATTERN,
    RSTOP_HEAD_PATTERN,
};
use hf_walk::{EpisodeIndex, StopRule, WalkOptions};
use sha2::{Digest, Sha256};
use tch::Device;

const EDIM: usize = 8;

fn config(feature_set: &str, return_head: bool) -> ModelConfig {
    let mut m = serde_json::json!({
        "hidden_dimension": 32, "self_attention_heads": 2, "feedforward_multiplier": 2,
        "score_hidden_dimension": 16, "coverage_hidden_dimension": 8,
        "traversal_blocks": 1, "dropout": 0.0, "feature_set": feature_set,
        "greedy_prior": true, "greedy_prior_scale": 10.0,
    });
    if return_head {
        m["return_head"] = true.into();
    }
    ModelConfig::from_value(&m, EDIM as i64).unwrap()
}

fn vector(name: &str) -> Vec<f32> {
    let mut h = Sha256::new();
    h.update(name.as_bytes());
    let d = h.finalize();
    (0..EDIM).map(|i| (d[i] as f32 - 128.0) / 128.0).collect()
}

/// A 40-node ring-with-chords deletion ball, and X's row as the query.
fn deletion_index(id: &str) -> (EpisodeIndex, EmbeddingMatrix) {
    let names: Vec<String> = (0..40).map(|i| format!("v{i}")).collect();
    let mut edges = Vec::new();
    for i in 0..40u32 {
        for d in [1u32, 3, 7] {
            edges.push((
                edges.len() as u32,
                format!("v{i}"),
                format!("v{}", (i + d) % 40),
            ));
        }
    }
    let mut all = names.clone();
    all.push("x".into());
    let data: Vec<f32> = all.iter().flat_map(|n| vector(n)).collect();
    let cache = EmbeddingMatrix::from_rows(all, EDIM, data);
    let index =
        EpisodeIndex::for_deletion(id, "v0", &names, &edges, &cache, EDIM, &vector("x")).unwrap();
    (index, cache)
}

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("hf-model-r1-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn head_names(model: &Model) -> Vec<String> {
    model.names_matching(&R1_HEAD_PATTERNS)
}

fn digest_without_heads(model: &Model) -> String {
    let heads = head_names(model);
    let refs: Vec<&str> = heads.iter().map(String::as_str).collect();
    model.state_digest(&refs)
}

/// A few return items from a capped walk of the fixture ball, for driving the
/// trunk with gradients on.
fn return_items(model: &Model) -> Vec<hf_walk::ReturnSnapshot> {
    let (index, _cache) = deletion_index("fixture-screen-000001-00000000-del-U");
    let features = model.features();
    let mut scorer = ModelScorer { model };
    let out = hf_walk::walk_batch_r1(
        &[&index],
        features.as_ref(),
        &mut scorer,
        WalkOptions {
            stop_rule: StopRule::Exhaust,
            record_candidates: false,
            keep_items: false,
            with_prior: true,
        },
        Some(30),
        &hf_walk::R1Options {
            return_at: vec![5, 16, 30],
            rstop_at: vec![],
        },
    )
    .unwrap();
    out.into_iter().next().unwrap().1.snapshots
}

/// ENG-2 (a): `return_head: false` builds today's model — the capacity and
/// the seed's `state_digest` captured from the premise engine (b8b074b)
/// before the heads existed. With the heads on, the trunk's digest is still
/// that one (the heads are created last), and the count grows by exactly the
/// two heads' parameters.
#[test]
fn without_the_heads_the_model_is_todays_and_with_them_the_trunk_is_unchanged() {
    tch::manual_seed(11);
    let off = Model::new(config("relational-v6-prev", false), Device::Cpu).unwrap();
    assert_eq!(off.trainable_parameter_count(), 14397);
    assert_eq!(
        off.state_digest(&[]),
        "sha256:b7ca1f069235b558f888ad4272727ab2882cad6aaef541229f235d6b63a2a134"
    );
    assert!(head_names(&off).is_empty());
    tch::manual_seed(11);
    let raw = Model::new(config("raw-v5", false), Device::Cpu).unwrap();
    assert_eq!(raw.trainable_parameter_count(), 15447);
    assert_eq!(
        raw.state_digest(&[]),
        "sha256:cd48183a4d3a5d108dc1476f619f499a09c36a0ec6c1fe553638ac5a7ac7ad94"
    );
    tch::manual_seed(11);
    let on = Model::new(config("relational-v6-prev", true), Device::Cpu).unwrap();
    assert_eq!(digest_without_heads(&on), off.state_digest(&[]));
    // H = 32: return (36*16+16) + (16+1) = 609; rstop (75*8+8) + (8+1) = 617
    assert_eq!(on.parameter_count_matching(&R1_HEAD_PATTERNS), 609 + 617);
    assert_eq!(on.trainable_parameter_count(), 14397 + 1226);
    let rstop_only = Model::new_with_heads(
        config("relational-v6-prev", false),
        Device::Cpu,
        R1Heads::rstop_only(),
    )
    .unwrap();
    assert_eq!(rstop_only.trainable_parameter_count(), 14397 + 617);
    assert_eq!(
        rstop_only.names_matching(&[RETURN_HEAD_PATTERN]),
        Vec::<String>::new()
    );
}

/// ENG-2's arithmetic on arm C's model block (hidden 256, score 283,
/// coverage 64, 8 blocks, edim 768): 8,508,921 without the heads, and the
/// design's 74,147 + 33,601 = 107,748 more with them — 8,616,669.
#[test]
fn arm_c_capacity_with_the_heads_is_the_designs_8616669() {
    let block = |return_head: bool| {
        let mut m = serde_json::json!({
            "hidden_dimension": 256, "self_attention_heads": 8, "feedforward_multiplier": 4,
            "score_hidden_dimension": 283, "coverage_hidden_dimension": 64,
            "traversal_blocks": 8, "dropout": 0.0, "greedy_prior": true,
            "greedy_prior_scale": 10.0, "feature_set": "relational-v6-prev",
        });
        if return_head {
            m["return_head"] = true.into();
        }
        ModelConfig::from_value(&m, 768).unwrap()
    };
    let off = Model::new(block(false), Device::Cpu).unwrap();
    assert_eq!(off.trainable_parameter_count(), 8_508_921);
    let on = Model::new(block(true), Device::Cpu).unwrap();
    assert_eq!(on.parameter_count_matching(&[RETURN_HEAD_PATTERN]), 74_147);
    assert_eq!(on.parameter_count_matching(&[RSTOP_HEAD_PATTERN]), 33_601);
    assert_eq!(on.trainable_parameter_count(), 8_616_669);
}

/// What tch's own `VarStore::load` does with a checkpoint carrying tensors
/// the model lacks: it loads it without complaint. This is why the stage-0
/// re-evaluation path now loads strictly (`Model::load_strict`) — without
/// it an R1 checkpoint would be accepted silently, not refused.
#[test]
fn tch_varstore_load_ignores_extra_tensors_which_is_why_loading_is_strict() {
    let dir = tmp("varstore");
    tch::manual_seed(3);
    let with = Model::new(config("relational-v6-prev", true), Device::Cpu).unwrap();
    let path = dir.join("r1.safetensors");
    with.vs.save(&path).unwrap();
    let mut without = Model::new(config("relational-v6-prev", false), Device::Cpu).unwrap();
    assert!(
        without.vs.load(&path).is_ok(),
        "tch's load refuses extra tensors after all: the strict loader is still right, \
         but the design's premise changed"
    );
    let e = without.load_strict(&path, &[], &[]).unwrap_err();
    assert!(e.to_string().contains("is not a parameter"), "{e}");
    assert!(without.load_strict(&path, &[], &R1_HEAD_PATTERNS).is_ok());
}

/// ENG-2 (b): `--init-from` a stage-0-shaped checkpoint loads into the R1
/// model, the heads alone missing; a name or a shape mismatch is band H, and
/// nothing is copied when a check fails.
#[test]
fn init_from_loads_a_stage0_checkpoint_and_refuses_a_name_or_shape_mismatch() {
    let dir = tmp("init");
    tch::manual_seed(4);
    let stage0 = Model::new(config("relational-v6-prev", false), Device::Cpu).unwrap();
    let path = dir.join("stage0.safetensors");
    stage0.vs.save(&path).unwrap();
    tch::manual_seed(99);
    let mut r1 = Model::new(config("relational-v6-prev", true), Device::Cpu).unwrap();
    let before_heads = r1.state_digest(&[]);
    let (ignored, missing) = r1.load_strict(&path, &R1_HEAD_PATTERNS, &[]).unwrap();
    assert!(ignored.is_empty());
    assert_eq!(missing, head_names(&r1));
    assert_eq!(digest_without_heads(&r1), stage0.state_digest(&[]));
    assert_ne!(r1.state_digest(&[]), before_heads);
    // a missing trunk tensor is band H
    let mut tensors: Vec<(String, tch::Tensor)> = stage0.vs.variables().into_iter().collect();
    tensors.retain(|(n, _)| n != "score_head.0.bias");
    let refs: Vec<(&str, tch::Tensor)> = tensors
        .iter()
        .map(|(n, t)| (n.as_str(), t.shallow_clone()))
        .collect();
    let short = dir.join("short.safetensors");
    tch::Tensor::write_safetensors(&refs, &short).unwrap();
    tch::manual_seed(99);
    let mut fresh = Model::new(config("relational-v6-prev", true), Device::Cpu).unwrap();
    let untouched = fresh.state_digest(&[]);
    let e = fresh
        .load_strict(&short, &R1_HEAD_PATTERNS, &[])
        .unwrap_err();
    assert!(matches!(e, hf_core::HfError::BandH(_)), "{e}");
    assert_eq!(
        fresh.state_digest(&[]),
        untouched,
        "a refused load copies nothing"
    );
    // a renamed tensor is band H
    let mut renamed: Vec<(String, tch::Tensor)> = stage0.vs.variables().into_iter().collect();
    for (n, _) in renamed.iter_mut() {
        if n == "score_head.0.bias" {
            *n = "score_head.0.bias_renamed".into();
        }
    }
    let refs: Vec<(&str, tch::Tensor)> = renamed
        .iter()
        .map(|(n, t)| (n.as_str(), t.shallow_clone()))
        .collect();
    let odd = dir.join("renamed.safetensors");
    tch::Tensor::write_safetensors(&refs, &odd).unwrap();
    assert!(matches!(
        fresh.load_strict(&odd, &R1_HEAD_PATTERNS, &[]).unwrap_err(),
        hf_core::HfError::BandH(_)
    ));
    // a shape mismatch is band H: a wider model's checkpoint
    let mut wide = config("relational-v6-prev", false);
    wide.score_hidden_dimension = 17;
    let other = Model::new(wide, Device::Cpu).unwrap();
    let wpath = dir.join("wide.safetensors");
    other.vs.save(&wpath).unwrap();
    let e = fresh
        .load_strict(&wpath, &R1_HEAD_PATTERNS, &[])
        .unwrap_err();
    assert!(e.to_string().contains("shape"), "{e}");
    assert_eq!(fresh.state_digest(&[]), untouched);
}

/// ENG-2 (c) and (d), driven THROUGH the trunk with gradients on (the cache
/// path never builds the trunk into the graph, so it could not tell): after
/// `freeze_except(heads)` and an optimiser built after it, every trunk
/// parameter's grad is undefined after `backward`, the trunk's digest after N
/// updates equals the init checkpoint's, and the heads' digests moved.
#[test]
fn only_the_heads_train_and_the_trunk_gets_no_gradient() {
    let dir = tmp("frozen");
    tch::manual_seed(5);
    let stage0 = Model::new(config("relational-v6-prev", false), Device::Cpu).unwrap();
    let path = dir.join("init.safetensors");
    stage0.vs.save(&path).unwrap();
    let init_digest = stage0.state_digest(&[]);
    tch::manual_seed(6);
    let mut model = Model::new(config("relational-v6-prev", true), Device::Cpu).unwrap();
    model.load_strict(&path, &R1_HEAD_PATTERNS, &[]).unwrap();
    let kept = model
        .freeze_except(&["return_head.*".to_string(), "rstop_head.*".to_string()])
        .unwrap();
    assert_eq!(kept, head_names(&model));
    let heads_before: Vec<(String, tch::Tensor)> = model
        .vs
        .variables()
        .into_iter()
        .filter(|(n, _)| kept.contains(n))
        .map(|(n, t)| (n, t.detach().copy()))
        .collect();
    let mut opt = AdamW::new(&model.vs, 1e-2, 0.01);
    let snaps = return_items(&model);
    let items: Vec<&hf_walk::DecisionItem> = snaps.iter().map(|s| &s.item).collect();
    let extras: Vec<&[f32]> = snaps.iter().map(|s| s.extras.as_slice()).collect();
    for _ in 0..3 {
        opt.zero_grad();
        let logits = model.return_logits(&items, &extras).unwrap();
        let loss = tch::Tensor::cat(&logits, 0)
            .sigmoid()
            .mean(tch::Kind::Float);
        loss.backward();
        for (name, var) in model.vs.variables() {
            if kept.contains(&name) {
                continue;
            }
            assert!(!var.grad().defined(), "{name} received a gradient");
        }
        assert!(model
            .vs
            .variables()
            .iter()
            .filter(|(n, _)| n.starts_with("return_head"))
            .all(|(_, v)| v.grad().defined()));
        opt.step();
    }
    assert_eq!(digest_without_heads(&model), init_digest);
    for (name, before) in &heads_before {
        if name.starts_with("return_head") {
            assert!(
                !model.vs.variables()[name].equal(before),
                "{name} did not move"
            );
        }
    }
    // a pattern matching nothing is refused
    assert!(model.freeze_except(&["retrun_head.*".to_string()]).is_err());
}

/// A hand-built record for the loss tests: two snapshots (t = 16 and the cap
/// 80), H-wide hidden rows and extras from a counter, one stop input at 16.
fn hand_record(hdim: usize, id: &str, cand16: &[&str], cand80: &[&str], seed: f32) -> r1::R1Record {
    let mut k = seed;
    let mut next = || {
        k = (k * 1.37 + 0.11) % 1.0;
        k - 0.5
    };
    let snap = |t: usize, cands: &[&str], next: &mut dyn FnMut() -> f32| r1::SnapshotFeatures {
        t,
        t_eff: t,
        candidates: cands.iter().map(|c| c.to_string()).collect(),
        hidden: (0..cands.len() * hdim).map(|_| next()).collect(),
        extras: (0..cands.len() * 4).map(|_| next()).collect(),
    };
    let s16 = snap(16, cand16, &mut next);
    let s80 = snap(80, cand80, &mut next);
    let rstop: Vec<f32> = (0..hf_walk::RSTOP_ROW_DIM + 2 * hdim)
        .map(|_| next())
        .collect();
    r1::R1Record {
        episode_id: id.into(),
        walk_expanded: vec![],
        t_end: 80,
        stop_reason: "expansion_cap".into(),
        snapshots: vec![s16, s80],
        rstop: vec![(16, Some(rstop))],
    }
}

fn spec_16_80() -> r1::R1Spec {
    r1::R1Spec {
        cap: 80,
        snapshots: vec![16, 80],
    }
}

/// ENG-2 (e): the heads share no parameter and the stop's input is
/// detached — a loss on `rstop` alone leaves every `return_head` grad
/// undefined, and a loss on the return head alone leaves `rstop_head`'s
/// undefined; and the pooled hidden states the walk hands the stop are
/// detached from the trunk.
#[test]
fn the_two_heads_are_detached_from_each_other_and_from_the_trunk() {
    tch::manual_seed(8);
    let model = Model::new(config("relational-v6-prev", true), Device::Cpu).unwrap();
    let r = hand_record(32, "e1", &["a", "b"], &["a", "b", "c"], 0.3);
    let t: std::collections::HashSet<String> = ["b".to_string()].into();
    let grads = |pick: &dyn Fn(&r1::R1Losses) -> tch::Tensor| {
        for (_, v) in model.vs.variables() {
            let mut v = v;
            v.zero_grad();
        }
        let l = r1::r1_losses(&model, &[&r], &[&t], &spec_16_80()).unwrap();
        pick(&l).backward();
        let defined = |prefix: &str| {
            model
                .vs
                .variables()
                .iter()
                .filter(|(n, _)| n.starts_with(prefix))
                .map(|(_, v)| {
                    v.grad().defined()
                        && f64::try_from(v.grad().abs().sum(tch::Kind::Float)).unwrap() > 0.0
                })
                .collect::<Vec<_>>()
        };
        (
            defined("return_head"),
            defined("rstop_head"),
            defined("blocks"),
        )
    };
    let (ret, stop, trunk) = grads(&|l| l.stop.shallow_clone());
    assert!(ret.iter().all(|d| !d), "a stop loss reached return_head");
    assert!(stop.iter().all(|d| *d));
    assert!(trunk.iter().all(|d| !d));
    let (ret, stop, trunk) = grads(&|l| l.ret.shallow_clone());
    assert!(ret.iter().all(|d| *d));
    assert!(stop.iter().all(|d| !d), "a return loss reached rstop_head");
    assert!(trunk.iter().all(|d| !d));
    // the pooled hidden states come off the trunk detached
    let snaps = return_items(&model);
    let items: Vec<&hf_walk::DecisionItem> = snaps.iter().map(|s| &s.item).collect();
    let (_, _, pooled) = model.score_decisions_with_hidden(&items);
    assert!(!pooled.requires_grad());
    assert_eq!(pooled.size(), vec![items.len() as i64, 64]);
}

/// ENG-4 (a): the loss arithmetic on hand-built walks. `y_t` switches
/// exactly at `t*`, the first snapshot whose examined set holds every target
/// the capped walk examines; and each BCE is averaged per item, then per
/// episode, then over the batch — checked against an f64 recomputation from
/// the heads' own logits.
#[test]
fn the_stop_label_switches_at_t_star_and_the_bce_nests_item_episode_batch() {
    let t: std::collections::HashSet<String> = ["b".to_string(), "d".to_string()].into();
    let cap: Vec<String> = ["a", "b", "c", "d", "e"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let at = |xs: &[&str]| xs.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    assert_eq!(
        r1::stop_label(&t, &at(&["a", "b"]), &cap),
        0.0,
        "d not yet examined"
    );
    assert_eq!(
        r1::stop_label(&t, &at(&["a", "b", "c", "d"]), &cap),
        1.0,
        "t*"
    );
    assert_eq!(r1::stop_label(&t, &cap, &cap), 1.0);
    // a target the capped walk never examines does not hold the label at 0
    let far: std::collections::HashSet<String> = ["b".to_string(), "zz".to_string()].into();
    assert_eq!(r1::stop_label(&far, &at(&["b"]), &cap), 1.0);
    // no target examined at all: every snapshot is a stop
    let none: std::collections::HashSet<String> = ["zz".to_string()].into();
    assert_eq!(r1::stop_label(&none, &at(&[]), &cap), 1.0);

    tch::manual_seed(9);
    let model = Model::new(config("relational-v6-prev", true), Device::Cpu).unwrap();
    let r1a = hand_record(32, "e1", &["a", "b"], &["a", "b", "c", "d"], 0.1);
    let r1b = hand_record(
        32,
        "e2",
        &["a", "b", "c", "d"],
        &["a", "b", "c", "d", "e"],
        0.7,
    );
    let ta: std::collections::HashSet<String> = ["b".to_string(), "d".to_string()].into();
    let tb = ta.clone();
    let l = r1::r1_losses(&model, &[&r1a, &r1b], &[&ta, &tb], &spec_16_80()).unwrap();
    // recompute in f64 from the head's logits, item by item
    let bce = |logit: f64, y: f64| logit.max(0.0) - logit * y + (1.0 + (-logit.abs()).exp()).ln();
    let head = |x: Vec<f32>, w: usize, f: &dyn Fn(&tch::Tensor) -> tch::Tensor| -> Vec<f64> {
        let n = x.len() / w;
        let t = tch::Tensor::from_slice(&x).view([n as i64, w as i64]);
        let out = f(&t);
        let mut v = vec![0f32; n];
        out.copy_data(&mut v, n);
        v.into_iter().map(|x| x as f64).collect()
    };
    let mut ret_eps = Vec::new();
    let mut stop_eps = Vec::new();
    for (r, t) in [(&r1a, &ta), (&r1b, &tb)] {
        let mut items = Vec::new();
        for s in &r.snapshots {
            let mut x = Vec::new();
            for k in 0..s.candidates.len() {
                x.extend_from_slice(&s.hidden[k * 32..(k + 1) * 32]);
                x.extend_from_slice(&s.extras[k * 4..(k + 1) * 4]);
            }
            let logits = head(x, 36, &|t| model.return_head_logits(t).unwrap());
            let per: Vec<f64> = s
                .candidates
                .iter()
                .zip(&logits)
                .map(|(c, l)| bce(*l, if t.contains(c) { 1.0 } else { 0.0 }))
                .collect();
            items.push(per.iter().sum::<f64>() / per.len() as f64);
        }
        ret_eps.push(items.iter().sum::<f64>() / items.len() as f64);
        let input = r.rstop[0].1.clone().unwrap();
        let logit = head(input, 75, &|t| model.rstop_head_logits(t).unwrap())[0];
        let y = r1::stop_label(t, &r.snapshots[0].candidates, &r.snapshots[1].candidates) as f64;
        stop_eps.push(bce(logit, y));
    }
    // e1: d is examined only at the cap -> y = 0; e2: both by 16 -> y = 1
    let want_ret = ret_eps.iter().sum::<f64>() / 2.0;
    let want_stop = stop_eps.iter().sum::<f64>() / 2.0;
    let got = |t: &tch::Tensor| f64::try_from(t).unwrap();
    assert!(
        (got(&l.ret) - want_ret).abs() < 1e-5,
        "{} vs {want_ret}",
        got(&l.ret)
    );
    assert!(
        (got(&l.stop) - want_stop).abs() < 1e-5,
        "{} vs {want_stop}",
        got(&l.stop)
    );
    assert!((got(&l.total) - want_ret - want_stop).abs() < 1e-5);
    // a flat mean over candidates would differ here (items of 2 and 4 and 5)
    assert_eq!(l.candidates, 2 + 4 + 4 + 5);
    assert_eq!(l.positives, 1 + 2 + 2 + 2);
}

/// A stage-0 record on a 120-node ring whose shown target is far from the
/// start, so the walk is still unregistered at 16 expansions.
fn far_stage0_episode() -> (hf_io::RealEpisode, EmbeddingMatrix) {
    let n = 120u32;
    let names: Vec<String> = (0..n).map(|i| format!("v{i}")).collect();
    let mut edges = Vec::new();
    for i in 0..n {
        for d in [1u32, 3] {
            edges.push(serde_json::json!({
                "edge_id": edges.len(), "relation": null,
                "source": format!("v{i}"), "target": format!("v{}", (i + d) % n),
            }));
        }
    }
    let visible = serde_json::json!({
        "schema_version": hf_io::SCHEMA_VERSION_V5,
        "record_kind": hf_io::VISIBLE_KIND,
        "family": "fixture",
        "stage": hf_io::STAGES[0],
        "start_node": "v0",
        "target_node": "v90",
        "subgraph_size": n,
        "removal_level": 0,
        "nodes": names.iter().map(|v| serde_json::json!({"node": v, "text": ""})).collect::<Vec<_>>(),
        "edges": edges,
    });
    hf_io::validate_visible(&visible).unwrap();
    let hidden = serde_json::json!({
        "schema_version": hf_io::SCHEMA_VERSION_V5,
        "record_kind": hf_io::HIDDEN_KIND,
        "family": "fixture",
        "stage": hf_io::STAGES[0],
        "split": "screen",
        "index": 0,
        "start_node": "v0",
        "target_set": ["v90"],
        "target_distance": 0,
        "cost_bound": 0,
        "path_set": [],
        "surviving_paths": [],
        "removal_set": [],
        "removed_count": 0,
        "unremovable_count": 0,
        "nodes_on_surviving_path": [],
        "distance_to_target": {},
        "sampler": {"subgraph_size": n},
    });
    let data: Vec<f32> = names.iter().flat_map(|v| vector(v)).collect();
    (
        hf_io::RealEpisode {
            episode_id: "fixture-screen-000007-00000000".into(),
            visible: serde_json::from_value(visible).unwrap(),
            hidden: serde_json::from_value(hidden).unwrap(),
        },
        EmbeddingMatrix::from_rows(names, EDIM, data),
    )
}

fn learned_r1_walk(
    model: &Model,
    index: &EpisodeIndex,
    rule: StopRule,
) -> (hf_walk::WalkResult, hf_walk::R1Trace) {
    let features = model.features();
    let mut scorer = ModelScorer { model };
    hf_walk::walk_batch_r1(
        &[index],
        features.as_ref(),
        &mut scorer,
        WalkOptions {
            stop_rule: rule,
            record_candidates: false,
            keep_items: false,
            with_prior: true,
        },
        None,
        &hf_walk::R1Options::default(),
    )
    .unwrap()
    .remove(0)
}

/// ENG-4 (b2), the model half of `--trunk-plus-rstop`: the stage-0 model
/// plus `rstop_head` alone, loaded from an R1 checkpoint (return_head
/// ignored), gives the rstop logits of the R1 model loaded in full, bit for
/// bit, on a stage-0 walk; `learned-r1` reads `rstop_head` and not
/// `stop_head` — overwriting `stop_head` leaves every stop unchanged while
/// the stage-0 `Learned` rule does change — and it consults only the
/// snapshots; a checkpoint without `rstop_head.*` is band H.
#[test]
fn trunk_plus_rstop_is_the_full_models_stop_and_never_reads_stop_head() {
    let dir = tmp("plus-rstop");
    tch::manual_seed(31);
    let full = Model::new(config("relational-v6-prev", true), Device::Cpu).unwrap();
    let ckpt = dir.join("r1.safetensors");
    full.vs.save(&ckpt).unwrap();
    let mut plus = Model::new_with_heads(
        config("relational-v6-prev", false),
        Device::Cpu,
        R1Heads::rstop_only(),
    )
    .unwrap();
    let (ignored, missing) = plus
        .load_strict(&ckpt, &[], &[RETURN_HEAD_PATTERN])
        .unwrap();
    assert_eq!(ignored, full.names_matching(&[RETURN_HEAD_PATTERN]));
    assert!(missing.is_empty());
    let (episode, emb) = far_stage0_episode();
    let index = EpisodeIndex::new(&episode, &emb, EDIM).unwrap();
    for theta in [0.0, 0.5, 0.999] {
        let rule = StopRule::LearnedR1 {
            theta,
            snapshots: hf_walk::R1_STOP_SNAPSHOTS,
        };
        let (wa, ta) = learned_r1_walk(&full, &index, rule);
        let (wb, tb) = learned_r1_walk(&plus, &index, rule);
        assert_eq!(wa.expanded, wb.expanded, "theta {theta}");
        assert_eq!(wa.stop_reason, wb.stop_reason);
        let bits = |t: &hf_walk::R1Trace| {
            t.rstop_logits
                .iter()
                .map(|(t, l)| (*t, l.to_bits()))
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&ta), bits(&tb));
        assert!(
            !ta.rstop_logits.is_empty(),
            "the walk must reach a snapshot unregistered"
        );
        assert!(ta
            .rstop_logits
            .iter()
            .all(|(t, _)| hf_walk::R1_STOP_SNAPSHOTS.contains(t)));
        if theta == 0.0 {
            assert_eq!(wa.stop_reason, "learned_r1_stop");
            assert_eq!(wa.expansions(), 16);
        }
    }
    // overwrite stop_head: learned-r1 is unchanged, Learned is not
    let rule = StopRule::LearnedR1 {
        theta: 0.5,
        snapshots: hf_walk::R1_STOP_SNAPSHOTS,
    };
    let set_stop_bias = |value: f64| {
        tch::no_grad(|| {
            for (name, var) in plus.vs.variables() {
                if name == "stop_head.2.bias" {
                    let mut v = var;
                    let _ = v.fill_(value);
                }
            }
        })
    };
    set_stop_bias(50.0); // the old head says stop at once
    let before_r1 = learned_r1_walk(&plus, &index, rule);
    let before_learned = learned_r1_walk(&plus, &index, StopRule::Learned);
    set_stop_bias(-50.0); // the old head says never stop
    let after_r1 = learned_r1_walk(&plus, &index, rule);
    let after_learned = learned_r1_walk(&plus, &index, StopRule::Learned);
    assert_eq!(before_learned.0.expansions(), 1);
    assert_eq!(before_r1.0.expanded, after_r1.0.expanded);
    assert_eq!(before_r1.1.rstop_logits, after_r1.1.rstop_logits);
    assert_ne!(
        before_learned.0.expanded, after_learned.0.expanded,
        "stop_head drives the stage-0 learned rule"
    );
    // a checkpoint without rstop_head is band H under --trunk-plus-rstop
    let stage0 = Model::new(config("relational-v6-prev", false), Device::Cpu).unwrap();
    let s0 = dir.join("stage0.safetensors");
    stage0.vs.save(&s0).unwrap();
    let mut again = Model::new_with_heads(
        config("relational-v6-prev", false),
        Device::Cpu,
        R1Heads::rstop_only(),
    )
    .unwrap();
    let e = again
        .load_strict(&s0, &[], &[RETURN_HEAD_PATTERN])
        .unwrap_err();
    assert!(e.to_string().contains("lacks rstop_head"), "{e}");
}
