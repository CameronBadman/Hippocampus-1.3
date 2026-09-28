//! The R1 head's runner paths (`R1_HEAD_DESIGN.md` §8 ENG-4 and ENG-7), on
//! the fixture world, on the CPU.

use std::path::{Path, PathBuf};

fn foundation() -> PathBuf {
    std::env::var("HF_FOUNDATION")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../hippocampus-foundation")
        })
}

fn raw_config() -> PathBuf {
    foundation().join("experiments/real_walk_v1/training-config.stage0.fixture.json")
}

const CPU_ONLY: &[(&str, &str)] = &[("CUDA_VISIBLE_DEVICES", "")];

fn run(args: &[&str]) -> std::process::Output {
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_hf-stage0"));
    command.args(args);
    for (k, v) in CPU_ONLY {
        command.env(k, v);
    }
    command.output().unwrap()
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("hf-stage0-r1-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn sha(path: &Path) -> String {
    hf_core::sha256_file(path).unwrap().1
}

/// The stage-0 model block of the R1 fixture: relational-v6-prev with the
/// greedy prior, at the fixture world's sizes (arm C's shape, scaled down).
fn stage0_model() -> serde_json::Value {
    serde_json::json!({
        "hidden_dimension": 32, "self_attention_heads": 2, "feedforward_multiplier": 2,
        "score_hidden_dimension": 16, "coverage_hidden_dimension": 8,
        "traversal_blocks": 1, "dropout": 0.0, "feature_set": "relational-v6-prev",
        "greedy_prior": true, "greedy_prior_scale": 10.0
    })
}

fn write_config(
    dir: &Path,
    name: &str,
    model: serde_json::Value,
    extra: serde_json::Value,
) -> PathBuf {
    let mut config = serde_json::json!({
        "record_kind": "real_walk_stage0_config",
        "note": "fixture configuration written by the test; never evidence",
        "training_authorized": false,
        "model": model,
        "sampler": {"subgraph_size": 64, "target_distance": 3, "removal_level": 2, "cost_epsilon": 0.5},
        "training": {"learning_rate": 0.001, "weight_decay": 0.0, "update_count": 2, "microbatch_size": 4}
    });
    if let Some(obj) = extra.as_object() {
        for (k, v) in obj {
            config[k] = v.clone();
        }
    }
    let path = dir.join(name);
    std::fs::write(&path, serde_json::to_string_pretty(&config).unwrap()).unwrap();
    path
}

/// A stage-0 fixture training run of `updates` updates, checkpoint saved.
fn train_stage0(cfg: &Path, out: &Path) {
    let o = run(&[
        "--config",
        cfg.to_str().unwrap(),
        "--output",
        out.to_str().unwrap(),
        "--model-seed",
        "5",
        "--fixture",
        "--train-episodes",
        "8",
        "--screen-episodes",
        "6",
        "--updates",
        "2",
        "--eval-every",
        "2",
        "--save-checkpoint",
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
}

fn reevaluate(cfg: &Path, ck: &Path, out: &Path, extra: &[&str]) -> std::process::Output {
    let mut args = vec![
        "--config",
        cfg.to_str().unwrap(),
        "--output",
        out.to_str().unwrap(),
        "--model-seed",
        "5",
        "--fixture",
        "--train-episodes",
        "8",
        "--screen-episodes",
        "6",
        "--reevaluate-checkpoint",
        ck.to_str().unwrap(),
    ];
    args.extend_from_slice(extra);
    run(&args)
}

// --------------------------------------------------------------------------
// fixtures

/// A deletion split over fixture-world balls of up to 64 nodes, in the
/// layout `hf-splits deletions` writes, labels included (T = X's in- and
/// out-neighbours in the rebuilt ball, the start excluded).
fn write_deletion_split(root: &Path, n: usize) -> (PathBuf, PathBuf) {
    use std::io::Write;
    let (graph, embeddings) = hf_episodes::fixture::fixture_world(5, 400, 1600, 8);
    let cache = root.join("cache");
    std::fs::create_dir_all(&cache).unwrap();
    let mut f = std::fs::File::create(cache.join("vectors.jsonl")).unwrap();
    let mut names: Vec<&String> = embeddings.keys().collect();
    names.sort();
    for name in &names {
        hf_embed::append_vector(&mut f, name, &embeddings[*name]).unwrap();
    }
    let manifest = |count: u64| hf_embed::Manifest {
        record_kind: hf_embed::MANIFEST_KIND.into(),
        model: "fixture".into(),
        model_digest: "fixture".into(),
        base_url: "none".into(),
        dimension: 8,
        count,
        text_char_limit: 6000,
        text_sha256: None,
        truncated: Default::default(),
        training_authorized: false,
        extra: Default::default(),
    };
    hf_embed::write_manifest(&cache, &manifest(names.len() as u64)).unwrap();
    let split = root.join("del");
    std::fs::create_dir_all(split.join("queries")).unwrap();
    let gz = |name: &str| {
        flate2::write::GzEncoder::new(
            std::fs::File::create(split.join(name)).unwrap(),
            flate2::Compression::default(),
        )
    };
    let (mut vis, mut lab) = (gz("visible.jsonl.gz"), gz("labels.jsonl.gz"));
    let mut q = std::fs::File::create(split.join("queries/vectors.jsonl")).unwrap();
    let mut written = 0;
    let mut start = 0u32;
    while written < n {
        start += 7;
        let s = graph.id(&format!("n{start}")).unwrap();
        let Ok(stored) = graph.ball(s, 64, None, 3, None) else {
            continue;
        };
        let mut picked = None;
        for &x in stored.iter().skip(1) {
            let Ok(traced) = graph.ball_traced(s, 64, None, 3, None, Some(x)) else {
                continue;
            };
            let ball = traced.order;
            let t: Vec<&str> = ball[1..]
                .iter()
                .filter(|v| {
                    graph.out_neighbours(**v).contains(&x) || graph.out_neighbours(x).contains(v)
                })
                .map(|v| graph.name(*v))
                .collect();
            if !t.is_empty() && ball.len() >= 40 {
                picked = Some((
                    x,
                    ball.clone(),
                    t.iter().map(|v| v.to_string()).collect::<Vec<_>>(),
                ));
                break;
            }
        }
        let Some((x, ball, t)) = picked else { continue };
        let sub = graph.induced(&ball);
        let mut edges = Vec::new();
        for &h in &ball {
            for e in sub.out(h) {
                edges.push(serde_json::json!({"edge_id": edges.len(), "source": graph.name(h), "target": graph.name(e.tail), "relation": null}));
            }
        }
        let id = format!("fixture-screen-{written:06}-00000000-del-U");
        let record = serde_json::json!({"episode_id": id, "visible": {
            "record_kind": "r1_deletion_visible_v1", "family": "fixture",
            "start_node": graph.name(s),
            "nodes": ball.iter().map(|v| graph.name(*v)).collect::<Vec<_>>(),
            "edges": edges,
        }});
        vis.write_all(serde_json::to_string(&record).unwrap().as_bytes())
            .unwrap();
        vis.write_all(b"\n").unwrap();
        let label = serde_json::json!({
            "episode_id": id, "record_kind": "r1_deletion_labels_v1",
            "deleted_node": graph.name(x), "draws": ["U"],
            "targets": t.iter().map(|v| serde_json::json!({"node": v, "direction": "both", "relations": []})).collect::<Vec<_>>(),
            "target_count": t.len(), "bfs_parent": graph.name(s),
        });
        lab.write_all(serde_json::to_string(&label).unwrap().as_bytes())
            .unwrap();
        lab.write_all(b"\n").unwrap();
        hf_embed::append_vector(&mut q, &id, &embeddings[graph.name(x)]).unwrap();
        written += 1;
    }
    vis.finish().unwrap();
    lab.finish().unwrap();
    hf_embed::write_manifest(&split.join("queries"), &manifest(n as u64)).unwrap();
    std::fs::write(
        split.join("deletions.manifest.json"),
        serde_json::json!({
            "record_kind": "r1_deletion_split_manifest_v1", "records_written": n,
            "draw_label": "r1-head-fixture", "allowed_ranges": ["screen:0.."],
            "per_start": "all", "training_authorized": false,
        })
        .to_string(),
    )
    .unwrap();
    (split, cache)
}

/// The R1 config: the stage-0 model plus `return_head`, the design's training
/// block at fixture sizes, and the r1 block.
fn r1_config(dir: &Path, init_sha: Option<&str>, edit: impl Fn(&mut serde_json::Value)) -> PathBuf {
    let mut model = stage0_model();
    model["return_head"] = true.into();
    let mut r1 = serde_json::json!({
        "max_expansions": 80, "snapshots": [16, 32, 48, 64, 80],
        "draw_label": "r1-head-fixture", "allowed_ranges": ["screen:0.."],
        "cache_batch": 4,
    });
    if let Some(sha) = init_sha {
        r1["init_from"] = serde_json::json!({"sha256": sha});
    }
    let mut config = serde_json::json!({
        "record_kind": "real_walk_r1_head_config",
        "note": "fixture configuration written by the test; never evidence",
        "training_authorized": false,
        "model": model,
        "training": {
            "learning_rate": 0.001, "weight_decay": 0.01, "clip_max_norm": 1.0,
            "update_count": 20, "microbatch_size": 4,
            "trainable": ["return_head.*", "rstop_head.*"],
            "decay_exempt": ["*bias", "*norm*.weight"],
        },
        "r1": r1,
    });
    edit(&mut config);
    let path = dir.join(format!(
        "r1-config-{}.json",
        hf_core::sha256_bytes(config.to_string().as_bytes())
            .split(':')
            .nth(1)
            .unwrap()
    ));
    std::fs::write(&path, serde_json::to_string_pretty(&config).unwrap()).unwrap();
    path
}

struct World {
    root: PathBuf,
    stage0_cfg: PathBuf,
    init: PathBuf,
    split: PathBuf,
    emb: PathBuf,
    cfg: PathBuf,
}

/// A stage-0 fixture checkpoint (the "2236-shaped" init), a deletion split
/// and the R1 config pinned to that checkpoint.
fn world(name: &str, records: usize) -> Option<World> {
    if !raw_config().exists() {
        eprintln!("skipped: the foundation checkout is not beside this one");
        return None;
    }
    let root = tmp(name);
    let stage0_cfg = write_config(&root, "stage0.json", stage0_model(), serde_json::json!({}));
    let train = root.join("stage0");
    train_stage0(&stage0_cfg, &train);
    let init = train.join("checkpoint.safetensors");
    let (split, emb) = write_deletion_split(&root, records);
    let cfg = r1_config(&root, Some(&sha(&init)), |_| {});
    Some(World {
        root,
        stage0_cfg,
        init,
        split,
        emb,
        cfg,
    })
}

fn r1_run(args: &[&str]) -> std::process::Output {
    let mut all = vec!["--model-seed", "3", "--fixture"];
    all.extend_from_slice(args);
    run(&all)
}

fn build_cache(w: &World, cfg: &Path, out: &Path) -> std::process::Output {
    r1_run(&[
        "--config",
        cfg.to_str().unwrap(),
        "--r1-cache",
        w.split.to_str().unwrap(),
        "--init-from",
        w.init.to_str().unwrap(),
        "--embeddings-dir",
        w.emb.to_str().unwrap(),
        "--output",
        out.to_str().unwrap(),
    ])
}

fn train_heads(
    w: &World,
    cfg: &Path,
    cache: &Path,
    out: &Path,
    extra: &[&str],
) -> std::process::Output {
    let mut args = vec![
        "--config",
        cfg.to_str().unwrap(),
        "--r1-train",
        cache.to_str().unwrap(),
        "--init-from",
        w.init.to_str().unwrap(),
        "--output",
        out.to_str().unwrap(),
    ];
    args.extend_from_slice(extra);
    r1_run(&args)
}

fn eval_heads(cfg: &Path, cache: &Path, ckpt: &Path, out: &Path) -> std::process::Output {
    r1_run(&[
        "--config",
        cfg.to_str().unwrap(),
        "--r1-eval",
        cache.to_str().unwrap(),
        "--reevaluate-checkpoint",
        ckpt.to_str().unwrap(),
        "--output",
        out.to_str().unwrap(),
    ])
}

fn ok(o: &std::process::Output) {
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
}

fn refused(o: &std::process::Output, needle: &str) {
    assert_eq!(
        o.status.code(),
        Some(2),
        "expected exit 2: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.contains(needle), "expected {needle:?} in: {err}");
}

fn json_file(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn lines(path: &Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

// --------------------------------------------------------------------------
// tests

/// ENG-3 (d) and ENG-4 (b2)'s last clause: with no R1 flag, the stage-0
/// fixture runs are byte for byte what the premise engine (b8b074b) wrote —
/// the training rows, the re-evaluation rows (the `learned` rule on the old
/// stop_head included) and the checkpoint — for raw-v5 and for
/// relational-v6-prev with the prior. The goldens were captured from that
/// build before any R1 code existed.
#[test]
fn stage0_runs_are_todays_byte_for_byte() {
    if !raw_config().exists() {
        eprintln!("skipped: the foundation checkout is not beside this one");
        return;
    }
    let root = tmp("goldens");
    let goldens = [
        (
            "raw",
            raw_config(),
            "sha256:a877f5724382ee4a367d5507d6e11bb2d26ad283f6ca5f01b072ac793c968a8f",
            "sha256:7849ce36c6cf3ac8803f9193969d2478e2b90aba8097f5e7bab9d840094b85b7",
            "sha256:4507e8a7acba96c96f65ea5e2b007b3e2614326a9ae3cd474f27e09081d5d595",
        ),
        (
            "prev",
            write_config(&root, "prev.json", stage0_model(), serde_json::json!({})),
            "sha256:8d5c61c6a708e03dbc49607bb438541484748e801fafb8925da8e547f6933ba6",
            "sha256:369d669147e10597a3fc68e19ff541b26aa99052ca6e04010062682cbcc3f052",
            "sha256:99c082f52c5a3273afe1dd580b0d47c169c75d0a5f7f32e11cb10653f605f424",
        ),
    ];
    for (name, cfg, train_rows, reeval_rows, ckpt) in goldens {
        let train = root.join(format!("train-{name}"));
        train_stage0(&cfg, &train);
        let re = root.join(format!("re-{name}"));
        ok(&reevaluate(&cfg, &train.join("checkpoint.json"), &re, &[]));
        assert_eq!(
            sha(&train.join("evaluation_rows.jsonl")),
            train_rows,
            "{name} training rows"
        );
        assert_eq!(
            sha(&re.join("evaluation_rows.jsonl")),
            reeval_rows,
            "{name} re-evaluation rows"
        );
        assert_eq!(
            sha(&train.join("checkpoint.safetensors")),
            ckpt,
            "{name} checkpoint"
        );
        let reeval = json_file(&re.join("reeval.json"));
        assert_eq!(reeval["trunk_only"], false);
        assert_eq!(reeval["rstop_loaded"], false);
    }
}

/// ENG-7 (a) and (c) through the CLI: the cache `--r1-cache` writes holds,
/// bit for bit, what the init checkpoint's trunk computes on the fly in this
/// process with the cache's own batching; and the writer never opens the
/// labels — built with them deleted, or replaced by garbage, the four cache
/// files are byte-identical.
#[test]
fn the_cli_cache_is_the_on_the_fly_computation_and_never_opens_labels() {
    let Some(w) = world("cache", 6) else { return };
    let cache = w.root.join("cache-a");
    ok(&build_cache(&w, &w.cfg, &cache));
    let m = json_file(&cache.join("cache.manifest.json"));
    assert_eq!(m["labels_opened"], false);
    assert_eq!(m["evidence"], false);
    assert_eq!(m["records"], 6);
    assert_eq!(m["batch_size"], 4);
    assert_eq!(m["init_sha256"], sha(&w.init));
    // in-process, on the fly, from the same checkpoint
    let config: serde_json::Value = json_file(&w.cfg);
    let mc = hf_model::ModelConfig::from_value(&config["model"], 8).unwrap();
    let mut model = hf_model::Model::new(mc, tch::Device::Cpu).unwrap();
    model
        .load_strict(&w.init, &hf_model::R1_HEAD_PATTERNS, &[])
        .unwrap();
    let emb = hf_embed::EmbeddingMatrix::load(&w.emb).unwrap();
    let queries = hf_embed::EmbeddingMatrix::load(&w.split.join("queries")).unwrap();
    let visible =
        hf_model::r1::read_deletion_visible(&w.split.join("visible.jsonl.gz"), 0).unwrap();
    let opened = hf_model::r1::R1Cache::open(&cache).unwrap();
    let mut k = 0;
    for chunk in visible.chunks(4) {
        let indexes: Vec<hf_walk::EpisodeIndex> = chunk
            .iter()
            .map(|r| {
                hf_walk::EpisodeIndex::for_deletion(
                    &r.episode_id,
                    &r.start,
                    &r.nodes,
                    &r.edges,
                    &emb,
                    8,
                    queries.get(&r.episode_id).unwrap(),
                )
                .unwrap()
            })
            .collect();
        let refs: Vec<&hf_walk::EpisodeIndex> = indexes.iter().collect();
        for want in
            hf_model::r1::compute_features(&model, &refs, &hf_model::r1::R1Spec::design()).unwrap()
        {
            let got = opened.get(k).unwrap();
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(got.episode_id, want.episode_id);
            assert_eq!(got.walk_expanded, want.walk_expanded);
            assert_eq!(got.t_end, want.t_end);
            for (a, b) in got.snapshots.iter().zip(&want.snapshots) {
                assert_eq!(a.candidates, b.candidates);
                assert_eq!(
                    bits(&a.hidden),
                    bits(&b.hidden),
                    "{} h_v t={}",
                    got.episode_id,
                    a.t
                );
                assert_eq!(bits(&a.extras), bits(&b.extras));
            }
            for ((_, a), (_, b)) in got.rstop.iter().zip(&want.rstop) {
                assert_eq!(a.as_deref().map(bits), b.as_deref().map(bits));
            }
            k += 1;
        }
    }
    assert_eq!(k, 6);
    // the labels: deleted, then garbage
    let saved = std::fs::read(w.split.join("labels.jsonl.gz")).unwrap();
    std::fs::remove_file(w.split.join("labels.jsonl.gz")).unwrap();
    let without = w.root.join("cache-b");
    ok(&build_cache(&w, &w.cfg, &without));
    std::fs::write(w.split.join("labels.jsonl.gz"), b"garbage, not even gzip").unwrap();
    let garbage = w.root.join("cache-c");
    ok(&build_cache(&w, &w.cfg, &garbage));
    std::fs::write(w.split.join("labels.jsonl.gz"), saved).unwrap();
    for f in ["records.jsonl", "hidden.f32", "extras.f32", "rstop.f32"] {
        assert_eq!(
            sha(&cache.join(f)),
            sha(&without.join(f)),
            "{f}: labels deleted"
        );
        assert_eq!(
            sha(&cache.join(f)),
            sha(&garbage.join(f)),
            "{f}: labels garbage"
        );
    }
}

/// ENG-4 (g), (c), (d) and the post-run check: a `--fixture` run of 20
/// updates from the cache and an eval are labelled `evidence: false`; every
/// update row carries its keys; the trunk did not move (`frozen_ok`, and
/// `--frozen-check` says ok); eval rows are logits only, deterministic,
/// independent of the thread count, and byte-identical with the labels file
/// deleted; a moved trunk tensor fails `--frozen-check`.
#[test]
fn a_fixture_smoke_run_trains_from_the_cache_and_evaluates_without_labels() {
    let Some(w) = world("smoke", 8) else { return };
    let cache = w.root.join("r1cache");
    ok(&build_cache(&w, &w.cfg, &cache));
    let run_dir = w.root.join("heads");
    ok(&train_heads(&w, &w.cfg, &cache, &run_dir, &[]));
    let rows = lines(&run_dir.join("updates.jsonl"));
    assert_eq!(rows.len(), 20);
    for r in &rows {
        for key in [
            "update",
            "ret",
            "stop",
            "total",
            "grad_norm",
            "finite",
            "expansions",
            "examined",
            "pos_share",
            "seconds",
        ] {
            assert!(r.get(key).is_some(), "{key}");
        }
        assert_eq!(r["finite"], true);
    }
    let summary = json_file(&run_dir.join("r1_train.json"));
    assert_eq!(summary["record_kind"], "r1_head_train_FIXTURE");
    assert_eq!(summary["evidence"], false);
    assert_eq!(summary["frozen_ok"], true);
    assert_eq!(summary["training_authorized"], false);
    assert_eq!(summary["trainable_parameters"], 1226);
    let ckpt = run_dir.join("checkpoint.safetensors");
    // the frozen check
    let fc = w.root.join("fc");
    ok(&r1_run(&[
        "--config",
        w.cfg.to_str().unwrap(),
        "--frozen-check",
        ckpt.to_str().unwrap(),
        "--init-from",
        w.init.to_str().unwrap(),
        "--embeddings-dir",
        w.emb.to_str().unwrap(),
        "--output",
        fc.to_str().unwrap(),
    ]));
    assert_eq!(json_file(&fc.join("frozen_check.json"))["ok"], true);
    // eval, twice, with other thread counts, and without the labels
    let e1 = w.root.join("eval1");
    ok(&eval_heads(&w.cfg, &cache, &ckpt, &e1));
    let ev = json_file(&e1.join("r1_eval.json"));
    assert_eq!(ev["evidence"], false);
    assert_eq!(ev["labels_opened"], false);
    let rows = lines(&e1.join("r1_rows.jsonl"));
    assert_eq!(rows.len(), 8);
    for r in &rows {
        assert!(
            r.get("returned").is_none() && r.get("stop").is_none(),
            "logits only"
        );
        let snaps = r["snapshots"].as_array().unwrap();
        assert_eq!(snaps.len(), 5);
        for s in snaps {
            assert_eq!(
                s["candidates"].as_array().unwrap().len(),
                s["return_logits"].as_array().unwrap().len()
            );
        }
        assert_eq!(r["rstop"].as_array().unwrap().len(), 4);
    }
    let e2 = w.root.join("eval2");
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_hf-stage0"));
    command.args([
        "--model-seed",
        "3",
        "--fixture",
        "--config",
        w.cfg.to_str().unwrap(),
        "--r1-eval",
        cache.to_str().unwrap(),
        "--reevaluate-checkpoint",
        ckpt.to_str().unwrap(),
        "--output",
        e2.to_str().unwrap(),
    ]);
    command
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("OMP_NUM_THREADS", "7")
        .env("RAYON_NUM_THREADS", "3")
        .env("MKL_NUM_THREADS", "7");
    ok(&command.output().unwrap());
    assert_eq!(
        sha(&e1.join("r1_rows.jsonl")),
        sha(&e2.join("r1_rows.jsonl")),
        "threads"
    );
    let saved = std::fs::read(w.split.join("labels.jsonl.gz")).unwrap();
    std::fs::remove_file(w.split.join("labels.jsonl.gz")).unwrap();
    let e3 = w.root.join("eval3");
    let o = eval_heads(&w.cfg, &cache, &ckpt, &e3);
    std::fs::write(w.split.join("labels.jsonl.gz"), saved).unwrap();
    ok(&o);
    assert_eq!(
        sha(&e1.join("r1_rows.jsonl")),
        sha(&e3.join("r1_rows.jsonl")),
        "labels deleted"
    );
    // a trunk tensor moved: --frozen-check fails
    let mut tensors: Vec<(String, tch::Tensor)> = tch::Tensor::read_safetensors(&ckpt).unwrap();
    for (n, t) in tensors.iter_mut() {
        if n == "blocks.0.feedforward.0.bias" {
            *t = &*t + 1.0;
        }
    }
    let refs: Vec<(&str, tch::Tensor)> = tensors
        .iter()
        .map(|(n, t)| (n.as_str(), t.shallow_clone()))
        .collect();
    let moved = w.root.join("moved.safetensors");
    tch::Tensor::write_safetensors(&refs, &moved).unwrap();
    let fc2 = w.root.join("fc2");
    let o = r1_run(&[
        "--config",
        w.cfg.to_str().unwrap(),
        "--frozen-check",
        moved.to_str().unwrap(),
        "--init-from",
        w.init.to_str().unwrap(),
        "--embeddings-dir",
        w.emb.to_str().unwrap(),
        "--output",
        fc2.to_str().unwrap(),
    ]);
    refused(&o, "frozen");
    let fcj = json_file(&fc2.join("frozen_check.json"));
    assert_eq!(fcj["ok"], false);
    assert_eq!(
        fcj["moved_parameters"],
        serde_json::json!(["blocks.0.feedforward.0.bias"])
    );
}

/// ENG-4 (b) and (f): training and evaluation refuse a cache whose trunk
/// digest, engine head or deletion visible sha disagrees, and a config whose
/// r1 block disagrees with the cache (cap, snapshots, draw label, ranges) or
/// whose init sha is another; a record id outside the declared ranges; an R1
/// path with a stage-0 split; a holdout path; a config without the r1 block
/// or without the heads.
#[test]
fn the_r1_paths_refuse_every_mismatch() {
    let Some(w) = world("refuse", 4) else { return };
    let cache = w.root.join("r1cache");
    ok(&build_cache(&w, &w.cfg, &cache));
    let heads = w.root.join("heads");
    ok(&train_heads(
        &w,
        &w.cfg,
        &cache,
        &heads,
        &["--updates", "2"],
    ));
    let ckpt = heads.join("checkpoint.safetensors");
    let doctored = |name: &str, edit: &dyn Fn(&mut serde_json::Value)| -> PathBuf {
        let dir = w.root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        for f in ["records.jsonl", "hidden.f32", "extras.f32", "rstop.f32"] {
            std::fs::copy(cache.join(f), dir.join(f)).unwrap();
        }
        let mut m = json_file(&cache.join("cache.manifest.json"));
        edit(&mut m);
        std::fs::write(
            dir.join("cache.manifest.json"),
            serde_json::to_string_pretty(&m).unwrap(),
        )
        .unwrap();
        dir
    };
    let both = |dir: &Path, needle: &str| {
        refused(
            &train_heads(
                &w,
                &w.cfg,
                dir,
                &w.root.join(format!("t-{needle}")),
                &["--updates", "1"],
            ),
            needle,
        );
        refused(
            &eval_heads(&w.cfg, dir, &ckpt, &w.root.join(format!("e-{needle}"))),
            needle,
        );
    };
    // (b) the three provenance mismatches
    both(
        &doctored("c-trunk", &|m| m["trunk_state_digest"] = "sha256:00".into()),
        "trunk_state_digest",
    );
    both(
        &doctored("c-engine", &|m| {
            m["engine_build"]["engine_build_head"] = "0000000".into()
        }),
        "engine_build_head",
    );
    let vpath = w.split.join("visible.jsonl.gz");
    let saved = std::fs::read(&vpath).unwrap();
    let mut changed = saved.clone();
    changed.extend_from_slice(b"\n");
    std::fs::write(&vpath, &changed).unwrap();
    both(&cache, "deletion_visible_sha256");
    std::fs::write(&vpath, &saved).unwrap();
    // (f) the r1 block against the cache
    for (needle, edit) in [
        (
            "max_expansions",
            Box::new(|c: &mut serde_json::Value| {
                c["r1"]["max_expansions"] = 64.into();
                c["r1"]["snapshots"] = serde_json::json!([16, 32, 48, 64]);
            }) as Box<dyn Fn(&mut serde_json::Value)>,
        ),
        (
            "snapshots",
            Box::new(|c: &mut serde_json::Value| {
                c["r1"]["snapshots"] = serde_json::json!([20, 40, 60, 80])
            }),
        ),
        (
            "deletion_draw_label",
            Box::new(|c: &mut serde_json::Value| c["r1"]["draw_label"] = "r1-head-other".into()),
        ),
        (
            "deletion_allowed_ranges",
            Box::new(|c: &mut serde_json::Value| {
                c["r1"]["allowed_ranges"] = serde_json::json!(["screen:0..3"])
            }),
        ),
        (
            "init_from.sha256",
            Box::new(|c: &mut serde_json::Value| {
                c["r1"]["init_from"]["sha256"] = "sha256:00".into()
            }),
        ),
    ] {
        let cfg = r1_config(&w.root, Some(&sha(&w.init)), |c| edit(c));
        let o = train_heads(
            &w,
            &cfg,
            &cache,
            &w.root.join(format!("f-{needle}")),
            &["--updates", "1"],
        );
        if needle == "init_from.sha256" {
            refused(&o, "init_sha256");
        } else {
            refused(&o, needle);
            refused(
                &eval_heads(&cfg, &cache, &ckpt, &w.root.join(format!("fe-{needle}"))),
                needle,
            );
        }
    }
    // an id outside the declared ranges: the cache build refuses it
    let narrow = r1_config(&w.root, Some(&sha(&w.init)), |c| {
        c["r1"]["allowed_ranges"] = serde_json::json!(["screen:2.."])
    });
    let mut m = json_file(&w.split.join("deletions.manifest.json"));
    let keep = m.clone();
    m["allowed_ranges"] = serde_json::json!(["screen:2.."]);
    std::fs::write(w.split.join("deletions.manifest.json"), m.to_string()).unwrap();
    let o = build_cache(&w, &narrow, &w.root.join("cache-narrow"));
    std::fs::write(w.split.join("deletions.manifest.json"), keep.to_string()).unwrap();
    refused(&o, "outside the declared ranges");
    // ... and training refuses one read from a cache (manifest and config agreeing)
    let c_narrow = doctored("c-narrow", &|m| {
        m["deletion_allowed_ranges"] = serde_json::json!(["screen:2.."])
    });
    refused(
        &train_heads(
            &w,
            &narrow,
            &c_narrow,
            &w.root.join("t-narrow"),
            &["--updates", "1"],
        ),
        "outside the declared ranges",
    );
    // a stage-0 split with an R1 path; a holdout output; the flags' pairing
    refused(
        &train_heads(
            &w,
            &w.cfg,
            &cache,
            &w.root.join("t-split"),
            &["--splits-dir", w.root.to_str().unwrap()],
        ),
        "deleted_payload",
    );
    refused(
        &train_heads(&w, &w.cfg, &cache, &w.root.join("holdout-out"), &[]),
        "holdout",
    );
    refused(&build_cache(&w, &w.cfg, &cache), "exists");
    // a config without the r1 block, and one without the heads
    let no_block = r1_config(&w.root, None, |c| {
        c.as_object_mut().unwrap().remove("r1");
    });
    refused(
        &train_heads(&w, &no_block, &cache, &w.root.join("t-noblock"), &[]),
        "r1 block",
    );
    let no_heads = r1_config(&w.root, None, |c| {
        c["model"].as_object_mut().unwrap().remove("return_head");
    });
    refused(
        &train_heads(&w, &no_heads, &cache, &w.root.join("t-noheads"), &[]),
        "return_head",
    );
    // a trainable set reaching into the trunk
    let wide = r1_config(&w.root, Some(&sha(&w.init)), |c| {
        c["training"]["trainable"] =
            serde_json::json!(["return_head.*", "rstop_head.*", "score_head.*"])
    });
    refused(
        &train_heads(
            &w,
            &wide,
            &cache,
            &w.root.join("t-wide"),
            &["--updates", "1"],
        ),
        "exactly the R1 heads",
    );
    // the stage-0 path refuses a config with the heads
    let with_sampler = r1_config(&w.root, None, |c| {
        c["sampler"] = serde_json::json!({"subgraph_size": 64, "target_distance": 3, "removal_level": 2, "cost_epsilon": 0.5});
    });
    refused(
        &reevaluate(&with_sampler, &w.init, &w.root.join("re-heads"), &[]),
        "head-less",
    );
}

/// ENG-4 (e) and (b2) through the CLI. `--trunk-only`: an R1 fixture
/// checkpoint re-evaluated on the stage-0 fixture walks every episode
/// exactly as its init checkpoint does — the whole `evaluation_rows.jsonl`,
/// `learned` rule included, is byte-identical; without the flag the R1
/// checkpoint is refused; a checkpoint with one extra non-head tensor is band
/// H; `learned-r1` is refused beside it. `--trunk-plus-rstop`: the rows are
/// again the init's byte for byte and the extra `r1_stop_rows.jsonl`
/// consults only snapshots; a checkpoint without `rstop_head.*` is band H;
/// the flag without `--stop-rule learned-r1` is refused.
#[test]
fn trunk_only_and_trunk_plus_rstop_reevaluate_stage0_as_the_init_did() {
    let Some(w) = world("trunk", 4) else { return };
    let cache = w.root.join("r1cache");
    ok(&build_cache(&w, &w.cfg, &cache));
    let heads = w.root.join("heads");
    ok(&train_heads(
        &w,
        &w.cfg,
        &cache,
        &heads,
        &["--updates", "3"],
    ));
    let r1_json = heads.join("checkpoint.json");
    let base = w.root.join("re-init");
    ok(&reevaluate(
        &w.stage0_cfg,
        &w.init.with_extension("json"),
        &base,
        &[],
    ));
    // --trunk-only
    let to = w.root.join("re-trunk-only");
    ok(&reevaluate(&w.stage0_cfg, &r1_json, &to, &["--trunk-only"]));
    assert_eq!(
        sha(&base.join("evaluation_rows.jsonl")),
        sha(&to.join("evaluation_rows.jsonl"))
    );
    let rj = json_file(&to.join("reeval.json"));
    assert_eq!(rj["trunk_only"], true);
    assert_eq!(rj["rstop_loaded"], false);
    assert_eq!(
        rj["checkpoint_sha256"],
        sha(&heads.join("checkpoint.safetensors"))
    );
    assert_eq!(rj["ignored_tensors"].as_array().unwrap().len(), 8);
    refused(
        &reevaluate(&w.stage0_cfg, &r1_json, &w.root.join("re-noflag"), &[]),
        "is not a parameter",
    );
    refused(
        &reevaluate(
            &w.stage0_cfg,
            &r1_json,
            &w.root.join("re-to-r1"),
            &[
                "--trunk-only",
                "--stop-rule",
                "learned-r1",
                "--stop-theta",
                "0.5",
            ],
        ),
        "only --trunk-plus-rstop loads",
    );
    // one extra non-head tensor
    let odd_dir = w.root.join("odd");
    std::fs::create_dir_all(&odd_dir).unwrap();
    let mut tensors: Vec<(String, tch::Tensor)> =
        tch::Tensor::read_safetensors(heads.join("checkpoint.safetensors")).unwrap();
    tensors.push((
        "extra_head.weight".into(),
        tch::Tensor::zeros([2], (tch::Kind::Float, tch::Device::Cpu)),
    ));
    let refs: Vec<(&str, tch::Tensor)> = tensors
        .iter()
        .map(|(n, t)| (n.as_str(), t.shallow_clone()))
        .collect();
    tch::Tensor::write_safetensors(&refs, odd_dir.join("checkpoint.safetensors")).unwrap();
    std::fs::copy(&r1_json, odd_dir.join("checkpoint.json")).unwrap();
    refused(
        &reevaluate(
            &w.stage0_cfg,
            &odd_dir.join("checkpoint.json"),
            &w.root.join("re-odd"),
            &["--trunk-only"],
        ),
        "extra_head.weight",
    );
    refused(
        &reevaluate(
            &w.stage0_cfg,
            &odd_dir.join("checkpoint.json"),
            &w.root.join("re-odd2"),
            &[
                "--trunk-plus-rstop",
                "--stop-rule",
                "learned-r1",
                "--stop-theta",
                "0.5",
            ],
        ),
        "extra_head.weight",
    );
    // --trunk-plus-rstop
    let tp = w.root.join("re-plus");
    ok(&reevaluate(
        &w.stage0_cfg,
        &r1_json,
        &tp,
        &[
            "--trunk-plus-rstop",
            "--stop-rule",
            "learned-r1",
            "--stop-theta",
            "0.0",
        ],
    ));
    assert_eq!(
        sha(&base.join("evaluation_rows.jsonl")),
        sha(&tp.join("evaluation_rows.jsonl"))
    );
    let rj = json_file(&tp.join("reeval.json"));
    assert_eq!(rj["rstop_loaded"], true);
    assert_eq!(rj["trunk_only"], false);
    assert_eq!(
        rj["checkpoint_sha256"],
        sha(&heads.join("checkpoint.safetensors"))
    );
    assert_eq!(rj["r1_stop"]["stop_rule"], "learned-r1");
    assert_eq!(rj["r1_stop"]["rstop_head_parameters"], 617);
    let stop_rows = lines(&tp.join("r1_stop_rows.jsonl"));
    assert_eq!(stop_rows.len(), 6, "one per screen episode");
    for r in &stop_rows {
        for c in r["rstop_logits"].as_array().unwrap() {
            assert!([16, 32, 48, 64].contains(&c["t"].as_u64().unwrap()), "{c}");
        }
        let stopped = r["stopped_before_registration"].as_bool().unwrap();
        assert_eq!(stopped, r["stop_reason"] == "learned_r1_stop");
        if stopped {
            assert!([16, 32, 48, 64].contains(&r["expansions"].as_u64().unwrap()));
            assert_eq!(r["unregistered_at_16"], true);
        }
    }
    let splits = &rj["r1_stop"]["splits"]["screen"];
    assert_eq!(splits["episodes"], 6);
    // without rstop_head (the init checkpoint): band H
    refused(
        &reevaluate(
            &w.stage0_cfg,
            &w.init.with_extension("json"),
            &w.root.join("re-plus-init"),
            &[
                "--trunk-plus-rstop",
                "--stop-rule",
                "learned-r1",
                "--stop-theta",
                "0.5",
            ],
        ),
        "lacks rstop_head",
    );
    // the flag without the rule, the rule without the flag, the rule without theta
    refused(
        &reevaluate(
            &w.stage0_cfg,
            &r1_json,
            &w.root.join("re-inert"),
            &["--trunk-plus-rstop"],
        ),
        "inert",
    );
    refused(
        &reevaluate(
            &w.stage0_cfg,
            &r1_json,
            &w.root.join("re-rule"),
            &["--stop-rule", "learned-r1", "--stop-theta", "0.5"],
        ),
        "only --trunk-plus-rstop loads",
    );
    assert_eq!(
        reevaluate(
            &w.stage0_cfg,
            &r1_json,
            &w.root.join("re-theta"),
            &["--trunk-plus-rstop", "--stop-rule", "learned-r1"]
        )
        .status
        .code(),
        Some(2)
    );
}

fn verify_cache(w: &World, cfg: &Path, cache: &Path, out: &Path) -> std::process::Output {
    r1_run(&[
        "--config",
        cfg.to_str().unwrap(),
        "--r1-cache-verify",
        cache.to_str().unwrap(),
        "--init-from",
        w.init.to_str().unwrap(),
        "--embeddings-dir",
        w.emb.to_str().unwrap(),
        "--output",
        out.to_str().unwrap(),
    ])
}

/// SMOKE-R's two operator lines: `--r1-cache-verify` recomputes the cache on
/// this device with its own batching and finds it bit-identical — and a
/// cache with one hidden value changed (its manifest restamped, so it opens)
/// is band H; and a deletion split written by the premise's engine, whose
/// manifest records `reserved` rather than `allowed_ranges`, is read under
/// the premise's own ranges, while any other shape is refused.
#[test]
fn the_cache_verifies_on_its_device_and_a_premise_split_reads_under_its_reserve() {
    let Some(w) = world("verify", 5) else { return };
    let cache = w.root.join("r1cache");
    ok(&build_cache(&w, &w.cfg, &cache));
    let v1 = w.root.join("verify1");
    ok(&verify_cache(&w, &w.cfg, &cache, &v1));
    let report = json_file(&v1.join("cache_verify.json"));
    assert_eq!(report["ok"], true);
    assert_eq!(report["records"], 5);
    assert_eq!(report["batch_size"], 4);
    // one hidden value changed, the manifest restamped
    let bad = w.root.join("r1cache-bad");
    std::fs::create_dir_all(&bad).unwrap();
    for f in [
        "records.jsonl",
        "hidden.f32",
        "extras.f32",
        "rstop.f32",
        "cache.manifest.json",
    ] {
        std::fs::copy(cache.join(f), bad.join(f)).unwrap();
    }
    let mut bytes = std::fs::read(bad.join("hidden.f32")).unwrap();
    let at = bytes.len() / 2 / 4 * 4;
    let v = f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]) + 1e-3;
    bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
    std::fs::write(bad.join("hidden.f32"), &bytes).unwrap();
    let mut m = json_file(&bad.join("cache.manifest.json"));
    m["files"]["hidden.f32"]["sha256"] = sha(&bad.join("hidden.f32")).into();
    std::fs::write(
        bad.join("cache.manifest.json"),
        serde_json::to_string_pretty(&m).unwrap(),
    )
    .unwrap();
    let v2 = w.root.join("verify2");
    refused(
        &verify_cache(&w, &w.cfg, &bad, &v2),
        "differs from the on-the-fly",
    );
    let report = json_file(&v2.join("cache_verify.json"));
    assert_eq!(report["ok"], false);
    assert_eq!(report["mismatched"], 1);
    // a premise-era manifest: `reserved`, no `allowed_ranges`
    let mpath = w.split.join("deletions.manifest.json");
    let keep = std::fs::read(&mpath).unwrap();
    let premise_manifest = |reserved: serde_json::Value, label: &str| {
        serde_json::json!({
            "record_kind": "r1_deletion_split_manifest_v1", "records_written": 5,
            "draw_label": label, "reserved": reserved, "training_authorized": false,
        })
        .to_string()
    };
    let premise_cfg = r1_config(&w.root, Some(&sha(&w.init)), |c| {
        c["r1"]["draw_label"] = "r1-premise-2026-09-25".into();
        c["r1"]["allowed_ranges"] = serde_json::json!(["train:0..73359", "screen:0..4104"]);
    });
    std::fs::write(
        &mpath,
        premise_manifest(
            serde_json::json!({"train_from": 73360, "screen_from": 4105}),
            "r1-premise-2026-09-25",
        ),
    )
    .unwrap();
    let pc = w.root.join("r1cache-premise");
    let o = build_cache(&w, &premise_cfg, &pc);
    let pm = json_file(&pc.join("cache.manifest.json"));
    // any other shape is refused
    std::fs::write(
        &mpath,
        premise_manifest(
            serde_json::json!({"train_from": 73361, "screen_from": 4105}),
            "r1-premise-2026-09-25",
        ),
    )
    .unwrap();
    let o2 = build_cache(&w, &premise_cfg, &w.root.join("r1cache-odd"));
    std::fs::write(
        &mpath,
        premise_manifest(
            serde_json::json!({"train_from": 73360, "screen_from": 4105}),
            "another-label",
        ),
    )
    .unwrap();
    let o3 = build_cache(&w, &premise_cfg, &w.root.join("r1cache-odd2"));
    std::fs::write(&mpath, keep).unwrap();
    ok(&o);
    assert_eq!(pm["deletion_ranges_source"], "premise_reserve");
    assert_eq!(
        pm["deletion_allowed_ranges"],
        serde_json::json!(["train:0..73359", "screen:0..4104"])
    );
    refused(&o2, "neither allowed_ranges nor");
    refused(&o3, "neither allowed_ranges nor");
}
