//! The f32 walk cache (`R1_HEAD_DESIGN.md` §8 ENG-7) on the CPU: what the
//! cache holds is, bit for bit, what `compute_features` computes on the fly
//! with the same batching; heads trained from the cache and from the same
//! features on the fly end identical; a truncated or partial cache is
//! refused; and (ENG-3 (c)) the episode id reaches no feature and no logit.

use std::collections::HashSet;
use std::path::PathBuf;

use hf_embed::EmbeddingMatrix;
use hf_model::r1::{self, CacheWriter, R1Cache, R1Record, R1Spec};
use hf_model::{AdamW, Model, ModelConfig, R1_HEAD_PATTERNS};
use hf_walk::EpisodeIndex;
use sha2::{Digest, Sha256};
use tch::Device;

const EDIM: usize = 8;

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("hf-model-r1c-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn config() -> ModelConfig {
    let m = serde_json::json!({
        "hidden_dimension": 32, "self_attention_heads": 2, "feedforward_multiplier": 2,
        "score_hidden_dimension": 16, "coverage_hidden_dimension": 8,
        "traversal_blocks": 1, "dropout": 0.0, "feature_set": "relational-v6-prev",
        "greedy_prior": true, "greedy_prior_scale": 10.0, "return_head": true,
    });
    ModelConfig::from_value(&m, EDIM as i64).unwrap()
}

fn vector(name: &str) -> Vec<f32> {
    let mut h = Sha256::new();
    h.update(name.as_bytes());
    let d = h.finalize();
    (0..EDIM).map(|i| (d[i] as f32 - 128.0) / 128.0).collect()
}

/// A ring of `n` nodes with chords, walked on the row of `x`.
fn ball(n: u32, id: &str, x: &str) -> EpisodeIndex {
    let names: Vec<String> = (0..n).map(|i| format!("v{i}")).collect();
    let mut edges = Vec::new();
    for i in 0..n {
        for d in [1u32, 3, 7] {
            edges.push((
                edges.len() as u32,
                format!("v{i}"),
                format!("v{}", (i + d) % n),
            ));
        }
    }
    let mut all = names.clone();
    all.push(x.into());
    let data: Vec<f32> = all.iter().flat_map(|v| vector(v)).collect();
    let cache = EmbeddingMatrix::from_rows(all, EDIM, data);
    EpisodeIndex::for_deletion(id, "v0", &names, &edges, &cache, EDIM, &vector(x)).unwrap()
}

fn fixture_indexes() -> Vec<EpisodeIndex> {
    (0..6)
        .map(|k| {
            ball(
                if k % 3 == 2 { 40 } else { 110 + k },
                &format!("fixture-screen-00000{k}-00000000-del-U"),
                &format!("x{k}"),
            )
        })
        .collect()
}

/// On the fly, in batches of `batch`, in order.
fn on_the_fly(model: &Model, indexes: &[EpisodeIndex], batch: usize) -> Vec<R1Record> {
    let mut out = Vec::new();
    for chunk in indexes.chunks(batch) {
        let refs: Vec<&EpisodeIndex> = chunk.iter().collect();
        out.extend(r1::compute_features(model, &refs, &R1Spec::design()).unwrap());
    }
    out
}

fn write_cache(dir: &std::path::Path, records: &[R1Record]) {
    let mut w = CacheWriter::create(dir, 32).unwrap();
    for r in records {
        w.append(r).unwrap();
    }
    w.finish(serde_json::json!({"note": "test"})).unwrap();
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn assert_same_bits(a: &R1Record, b: &R1Record, check_id: bool) {
    if check_id {
        assert_eq!(a.episode_id, b.episode_id);
    }
    assert_eq!(a.walk_expanded, b.walk_expanded);
    assert_eq!(a.t_end, b.t_end);
    assert_eq!(a.stop_reason, b.stop_reason);
    assert_eq!(a.snapshots.len(), b.snapshots.len());
    for (x, y) in a.snapshots.iter().zip(&b.snapshots) {
        assert_eq!((x.t, x.t_eff), (y.t, y.t_eff));
        assert_eq!(x.candidates, y.candidates);
        assert_eq!(bits(&x.hidden), bits(&y.hidden), "h_v at t = {}", x.t);
        assert_eq!(bits(&x.extras), bits(&y.extras), "extras at t = {}", x.t);
    }
    assert_eq!(a.rstop.len(), b.rstop.len());
    for ((t1, x), (t2, y)) in a.rstop.iter().zip(&b.rstop) {
        assert_eq!(t1, t2);
        assert_eq!(
            x.as_deref().map(bits),
            y.as_deref().map(bits),
            "rstop at t = {t1}"
        );
    }
}

/// ENG-7 (a): every cached value — `h_v`, the extras, the rstop input,
/// `walk_expanded` — read back through the memory map is bit-identical to
/// the on-the-fly computation under the same batch composition and order.
/// The fixture holds walks that reach the cap and walks that end first.
#[test]
fn the_cache_reproduces_the_on_the_fly_features_bit_for_bit() {
    tch::manual_seed(21);
    let model = Model::new(config(), Device::Cpu).unwrap();
    let indexes = fixture_indexes();
    let fly = on_the_fly(&model, &indexes, 4);
    assert!(fly.iter().any(|r| r.t_end == 80) && fly.iter().any(|r| r.t_end < 80));
    let dir = tmp("bits");
    // the cache is written from a SECOND on-the-fly pass, as the CLI does
    write_cache(&dir, &on_the_fly(&model, &indexes, 4));
    let cache = R1Cache::open(&dir).unwrap();
    assert_eq!(cache.len(), fly.len());
    for (i, want) in fly.iter().enumerate() {
        assert_same_bits(&cache.get(i).unwrap(), want, true);
    }
    // a record's rstop input is the 8 + 3 row then 2H pooled values
    let r = cache.get(0).unwrap();
    let first = r.rstop[0].1.as_ref().unwrap();
    assert_eq!(first.len(), 11 + 64);
    assert_eq!(first[8], 16.0 / 80.0, "expansions / 80 at t = 16");
}

/// ENG-7 (b): heads trained from the cache and heads trained from the same
/// features computed on the fly give identical checkpoints (every tensor,
/// and so every digest) after the same draws.
#[test]
fn heads_trained_from_the_cache_and_on_the_fly_are_identical() {
    tch::manual_seed(22);
    let model = Model::new(config(), Device::Cpu).unwrap();
    let init = tmp("init.safetensors");
    let _ = std::fs::remove_file(&init);
    model.vs.save(&init).unwrap();
    let indexes = fixture_indexes();
    let fly = on_the_fly(&model, &indexes, 3);
    let dir = tmp("train");
    write_cache(&dir, &fly);
    let cache = R1Cache::open(&dir).unwrap();
    let t: HashSet<String> = ["v1".into(), "v3".into(), "v7".into(), "v20".into()].into();
    let train = |source: &dyn Fn(usize) -> R1Record| -> Model {
        tch::manual_seed(22);
        let mut m = Model::new(config(), Device::Cpu).unwrap();
        m.load_strict(&init, &[], &[]).unwrap();
        m.freeze_except(&["return_head.*".into(), "rstop_head.*".into()])
            .unwrap();
        let mut opt = AdamW::new(&m.vs, 1e-2, 0.01);
        let mut rng = hf_core::PyRandom::from_seed(7);
        for _ in 0..6 {
            let idx = rng.sample(indexes.len(), 3);
            let recs: Vec<R1Record> = idx.iter().map(|i| source(*i)).collect();
            let refs: Vec<&R1Record> = recs.iter().collect();
            let tg: Vec<&HashSet<String>> = idx.iter().map(|_| &t).collect();
            let out = r1::train_step(
                &m,
                &mut opt,
                &refs,
                &tg,
                &R1Spec::design(),
                Some(1.0),
                false,
            )
            .unwrap();
            assert!(out.finite);
        }
        m
    };
    let a = train(&|i| cache.get(i).unwrap());
    let b = train(&|i| fly[i].clone());
    assert_eq!(a.state_digest(&[]), b.state_digest(&[]));
    let (pa, pb) = (tmp("a.safetensors"), tmp("b.safetensors"));
    a.vs.save(&pa).unwrap();
    b.vs.save(&pb).unwrap();
    assert_eq!(std::fs::read(&pa).unwrap(), std::fs::read(&pb).unwrap());
    // and they did train: the heads moved, the trunk did not
    let heads = model.names_matching(&R1_HEAD_PATTERNS);
    let refs: Vec<&str> = heads.iter().map(String::as_str).collect();
    assert_eq!(a.state_digest(&refs), model.state_digest(&refs));
    assert_ne!(a.state_digest(&[]), model.state_digest(&[]));
}

/// ENG-7 (d): a truncated or partial cache is refused — a short f32 file, a
/// dropped record line, a flipped byte of the same length, a missing file.
#[test]
fn a_truncated_or_partial_cache_is_refused() {
    tch::manual_seed(23);
    let model = Model::new(config(), Device::Cpu).unwrap();
    let records = on_the_fly(&model, &fixture_indexes()[..3], 3);
    let fresh = |name: &str| {
        let dir = tmp(name);
        write_cache(&dir, &records);
        assert!(R1Cache::open(&dir).is_ok());
        dir
    };
    // a short hidden.f32
    let dir = fresh("short");
    let bytes = std::fs::read(dir.join("hidden.f32")).unwrap();
    std::fs::write(dir.join("hidden.f32"), &bytes[..bytes.len() - 4]).unwrap();
    let e = R1Cache::open(&dir).err().unwrap();
    assert!(e.to_string().contains("truncated or partial"), "{e}");
    // a record dropped from the index, the manifest's file entry restamped:
    // the record count still refuses it
    let dir = fresh("count");
    let text = std::fs::read_to_string(dir.join("records.jsonl")).unwrap();
    let kept: String = text.lines().skip(1).map(|l| format!("{l}\n")).collect();
    std::fs::write(dir.join("records.jsonl"), &kept).unwrap();
    let mpath = dir.join(r1::CACHE_MANIFEST);
    let mut m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&mpath).unwrap()).unwrap();
    let (b, s) = hf_core::sha256_file(&dir.join("records.jsonl")).unwrap();
    m["files"]["records.jsonl"] = serde_json::json!({"bytes": b, "sha256": s});
    std::fs::write(&mpath, serde_json::to_string(&m).unwrap()).unwrap();
    let e = R1Cache::open(&dir).err().unwrap();
    assert!(e.to_string().contains("records"), "{e}");
    // one byte changed, the length kept: the sha refuses it
    let dir = fresh("flip");
    let mut bytes = std::fs::read(dir.join("rstop.f32")).unwrap();
    bytes[10] ^= 0x01;
    std::fs::write(dir.join("rstop.f32"), &bytes).unwrap();
    assert!(R1Cache::open(&dir).is_err());
    // a missing file, and a missing manifest
    let dir = fresh("missing");
    std::fs::remove_file(dir.join("extras.f32")).unwrap();
    assert!(R1Cache::open(&dir).is_err());
    let dir = fresh("nomanifest");
    std::fs::remove_file(dir.join(r1::CACHE_MANIFEST)).unwrap();
    assert!(R1Cache::open(&dir).is_err());
    // a cache is written once
    assert!(CacheWriter::create(&fresh("again"), 32).is_err());
}

/// ENG-3 (c) with the model: the draw tag and the rest of the episode id
/// reach nothing — the features and the eval logits of the same ball under
/// `-del-U`, `-del-D3` and an id of garbage are bit-identical.
#[test]
fn the_episode_id_and_its_draw_tag_reach_no_feature_and_no_logit() {
    tch::manual_seed(24);
    let model = Model::new(config(), Device::Cpu).unwrap();
    let ids = [
        "fixture-screen-000001-00000000-del-U",
        "fixture-screen-000001-00000000-del-D3",
        "zzz-garbage",
    ];
    let recs: Vec<R1Record> = ids
        .iter()
        .map(|id| {
            let index = ball(110, id, "x9");
            r1::compute_features(&model, &[&index], &R1Spec::design())
                .unwrap()
                .remove(0)
        })
        .collect();
    for r in &recs[1..] {
        assert_same_bits(&recs[0], r, false);
    }
    let rows: Vec<serde_json::Value> = recs
        .iter()
        .map(|r| {
            let mut v = r1::eval_logits(&model, r).unwrap();
            v["episode_id"] = serde_json::Value::Null;
            v
        })
        .collect();
    assert_eq!(rows[0], rows[1]);
    assert_eq!(rows[0], rows[2]);
}
