//! The runner on the fixture world: a training run writes the artifacts with
//! their keys; a re-evaluation with a training sample and a candidate dump
//! satisfies the readers' invariants; the refusals (holdout path, missing
//! pin, doctored probe, flags without a re-evaluation) exit 2;
//! `--print-capacity` builds the config's model on the CPU and prints what it
//! would train, reading no data; `training.decay_exempt` reaches the probe and
//! the checkpoints as the patterns and the names they resolved to,
//! `training.clip_max_norm` as the clip in effect (absent = 1.0, null = none),
//! every `updates.jsonl` row carries the greedy prior's `greedy_tau` after the
//! step (`null` without the prior) and a `finite` flag, with a non-finite loss
//! or gradient norm halting the run (skipped step, `instability.json`, exit 2,
//! no `probe.json`),
//! and the engine's build-time provenance — what the binary was built from,
//! whether that tree was dirty, the binary's own digest — reaches
//! `--engine-info` and every artifact, with a real run refused when the binary
//! is not the checkout's.

use std::path::PathBuf;

fn foundation() -> PathBuf {
    std::env::var("HF_FOUNDATION")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../hippocampus-foundation")
        })
}

fn config() -> PathBuf {
    foundation().join("experiments/real_walk_v1/training-config.stage0.fixture.json")
}

fn run(args: &[&str]) -> std::process::Output {
    run_env(args, &[])
}

/// A real run picks CUDA when it is there; a CLI test must not take the GPU
/// from whatever is training on it, so every non-fixture run here is pinned to
/// the CPU (`tch::Cuda::is_available()` is then false).
const CPU_ONLY: &[(&str, &str)] = &[("CUDA_VISIBLE_DEVICES", "")];

fn run_env(args: &[&str], env: &[(&str, &str)]) -> std::process::Output {
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_hf-stage0"));
    command.args(args);
    for (k, v) in env {
        command.env(k, v);
    }
    command.output().unwrap()
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("hf-stage0-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

#[test]
fn fixture_training_and_reevaluation_write_the_readers_artifacts() {
    if !config().exists() {
        eprintln!("skipped: the foundation checkout is not beside this one");
        return;
    }
    let cfg = config();
    let root = tmp("fixture");
    let out = root.join("run");
    let base = |o: &PathBuf| -> Vec<String> {
        vec![
            "--config".into(),
            cfg.to_string_lossy().into(),
            "--output".into(),
            o.to_string_lossy().into(),
            "--model-seed".into(),
            "5".into(),
            "--fixture".into(),
            "--train-episodes".into(),
            "12".into(),
            "--screen-episodes".into(),
            "6".into(),
        ]
    };
    let mut args = base(&out);
    args.extend(
        [
            "--updates",
            "4",
            "--eval-every",
            "2",
            "--save-checkpoint",
            "--checkpoint-every",
            "2",
        ]
        .map(String::from),
    );
    let o = run(&args.iter().map(String::as_str).collect::<Vec<_>>());
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let probe: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("probe.json")).unwrap()).unwrap();
    assert_eq!(probe["record_kind"], "real_walk_stage0_probe_FIXTURE");
    assert_eq!(probe["evidence"], false);
    assert_eq!(probe["training_authorized"], false);
    assert_eq!(probe["train_draws"]["draws"], 16);
    assert_eq!(probe["evaluations"].as_array().unwrap().len(), 2);
    for key in [
        "learned",
        "exhaust",
        "baselines",
        "exhaust_vs_blind_exhaust",
        "exhaust_vs_similarity_greedy",
        "learned_vs_oracle",
        "realised_removal_mean",
    ] {
        assert!(probe["evaluations"][0].get(key).is_some(), "{key}");
    }
    let updates = std::fs::read_to_string(out.join("updates.jsonl")).unwrap();
    assert_eq!(updates.lines().count(), 4);
    assert!(updates.lines().all(|l| l.contains("\"grad_norm\"")));
    let rows: Vec<serde_json::Value> = std::fs::read_to_string(out.join("evaluation_rows.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(rows.len(), 12, "6 screen episodes x 2 evaluations");
    for r in &rows {
        assert_eq!(r["split"], "screen");
        assert!(r["update"].is_u64());
        assert_eq!(
            r["walk_expanded"].as_array().unwrap().len() as u64,
            r["walk_exhaust"].as_u64().unwrap()
        );
        assert_eq!(
            r["similarity_greedy_examined"].as_array().unwrap().len() as u64,
            r["similarity_greedy"].as_u64().unwrap()
        );
        for hidden in [
            "path_set",
            "surviving_paths",
            "removal_set",
            "nodes_on_surviving_path",
            "distance_to_target",
            "target_set",
        ] {
            assert!(r.get(hidden).is_none(), "{hidden} leaked into a row");
        }
    }
    assert!(
        out.join("checkpoint.safetensors").exists()
            && out.join("checkpoint-latest.optim.safetensors").exists()
    );
    // a re-evaluation with the training sample and the candidate dump
    let re = root.join("reeval");
    let mut args = base(&re);
    args.extend(
        [
            "--reevaluate-checkpoint",
            out.join("checkpoint.json").to_str().unwrap(),
            "--train-sample",
            "4",
            "--dump-candidates",
            "--train-draws-probe",
            out.join("probe.json").to_str().unwrap(),
        ]
        .map(String::from),
    );
    let o = run(&args.iter().map(String::as_str).collect::<Vec<_>>());
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let reeval: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(re.join("reeval.json")).unwrap()).unwrap();
    assert_eq!(reeval["record_kind"], "real_walk_stage0_reeval_FIXTURE");
    assert_eq!(reeval["checkpoint_seed"], 5);
    assert_eq!(reeval["train_sample"]["count"], 4);
    assert_eq!(reeval["train_sample"]["draw_counts_source"], "checkpoint");
    assert!(reeval["checkpoint"]
        .as_str()
        .unwrap()
        .ends_with("checkpoint.json"));
    let dump = std::fs::read(re.join("candidate_dump.jsonl.gz")).unwrap();
    let mut text = String::new();
    std::io::Read::read_to_string(&mut flate2::read::MultiGzDecoder::new(&dump[..]), &mut text)
        .unwrap();
    let lines: Vec<serde_json::Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert!(!lines.is_empty());
    for l in &lines {
        assert_eq!(l["update"], "reeval");
        let scores: Vec<f64> = l["scores"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let best = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(scores[l["chosen"].as_u64().unwrap() as usize], best);
        assert_eq!(l["frontier"].as_array().unwrap().len(), scores.len());
        assert_eq!(l["parents"].as_array().unwrap().len(), scores.len());
        assert_eq!(l["depths"].as_array().unwrap().len(), scores.len());
        for d in l["depths"].as_array().unwrap() {
            assert!(d.as_u64().unwrap() >= 1);
        }
    }
    // a second re-evaluation into the same directory is refused: the streams append
    let o = run(&args.iter().map(String::as_str).collect::<Vec<_>>());
    assert_eq!(o.status.code(), Some(2));
    // the flags need a re-evaluation
    let mut args = base(&root.join("bad"));
    args.push("--dump-candidates".into());
    assert_eq!(
        run(&args.iter().map(String::as_str).collect::<Vec<_>>())
            .status
            .code(),
        Some(2)
    );
    // a holdout path is refused, a real run needs a pin
    let mut args = base(&root.join("heldout-x"));
    assert_eq!(
        run(&args.iter().map(String::as_str).collect::<Vec<_>>())
            .status
            .code(),
        Some(2)
    );
    args = base(&root.join("real"));
    args.retain(|a| a != "--fixture");
    // --allow-stale-engine passes the engine gate, which stands before the pin
    // and refuses the binary these tests are built from (a dirty tree)
    args.push("--allow-stale-engine".into());
    let o = run(&args.iter().map(String::as_str).collect::<Vec<_>>());
    assert_eq!(o.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&o.stderr).contains("preregistration-commit"));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn print_capacity_builds_the_config_model_and_prints_the_count() {
    if !config().exists() {
        eprintln!("skipped: the foundation checkout is not beside this one");
        return;
    }
    let cfg = config();
    let o = run(&["--print-capacity", "8", "--config", cfg.to_str().unwrap()]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let line: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&o.stdout)).unwrap();
    assert_eq!(
        line.as_object()
            .unwrap()
            .keys()
            .map(|k| k.as_str())
            .collect::<Vec<_>>(),
        ["feature_set", "dim", "trainable_parameters"]
    );
    assert_eq!(line["feature_set"], "raw-v5");
    assert_eq!(line["dim"], 8);
    // the fixture model at width 8; the golden's count plus the greedy prior's
    // single scalar, which this config does not ask for
    assert_eq!(line["trainable_parameters"], 15446);
    // no output and no seed are needed, and nothing is written
    assert!(!PathBuf::from("probe.json").exists());
    // the other feature sets are reachable through the same flag
    let alt = std::env::temp_dir().join(format!("hf-stage0-cap-{}.json", std::process::id()));
    let mut value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&cfg).unwrap()).unwrap();
    value["model"]["feature_set"] = "relational-v6-prev".into();
    std::fs::write(&alt, serde_json::to_string(&value).unwrap()).unwrap();
    let o = run(&["--print-capacity", "8", "--config", alt.to_str().unwrap()]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let prev: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&o.stdout)).unwrap();
    assert_eq!(prev["feature_set"], "relational-v6-prev");
    assert!(prev["trainable_parameters"].as_i64().unwrap() > 0);
    // an unknown feature set is refused, not silently defaulted
    value["model"]["feature_set"] = "relational-v7".into();
    std::fs::write(&alt, serde_json::to_string(&value).unwrap()).unwrap();
    assert_eq!(
        run(&["--print-capacity", "8", "--config", alt.to_str().unwrap()])
            .status
            .code(),
        Some(2)
    );
    let _ = std::fs::remove_file(&alt);
}

/// The rule a run is governed by comes from its config, not from a constant:
/// the v2 configs name `experiments/real_walk_v2/RULE.md`, and the runner used
/// to stamp every probe with the v1 path. A config that names no rule keeps
/// that v1 default, which is what the v1 configs and the Python runner write.
#[test]
fn the_configs_governing_rule_round_trips_into_probe_and_reeval() {
    if !config().exists() {
        eprintln!("skipped: the foundation checkout is not beside this one");
        return;
    }
    let root = tmp("governed");
    std::fs::create_dir_all(&root).unwrap();
    let rule = "experiments/real_walk_v2/RULE.md";
    let mut cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(config()).unwrap()).unwrap();
    cfg["governed_by"] = rule.into();
    let named = root.join("training-config.governed.json");
    std::fs::write(&named, serde_json::to_string(&cfg).unwrap()).unwrap();
    let base = |cfg: &PathBuf, o: &PathBuf| -> Vec<String> {
        vec![
            "--config".into(),
            cfg.to_string_lossy().into(),
            "--output".into(),
            o.to_string_lossy().into(),
            "--model-seed".into(),
            "5".into(),
            "--fixture".into(),
            "--train-episodes".into(),
            "6".into(),
            "--screen-episodes".into(),
            "3".into(),
            "--updates".into(),
            "1".into(),
        ]
    };
    let read = |p: PathBuf| -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
    };
    // the config's own rule reaches probe.json and reeval.json
    let out = root.join("run");
    let mut args = base(&named, &out);
    args.push("--save-checkpoint".into());
    let o = run(&args.iter().map(String::as_str).collect::<Vec<_>>());
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(read(out.join("probe.json"))["governed_by"], rule);
    let re = root.join("reeval");
    let mut args = base(&named, &re);
    args.extend(
        [
            "--reevaluate-checkpoint",
            out.join("checkpoint.json").to_str().unwrap(),
        ]
        .map(String::from),
    );
    let o = run(&args.iter().map(String::as_str).collect::<Vec<_>>());
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(read(re.join("reeval.json"))["governed_by"], rule);
    // a config naming no rule keeps the v1 default
    let plain = root.join("plain");
    let args = base(&config(), &plain);
    let o = run(&args.iter().map(String::as_str).collect::<Vec<_>>());
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(
        read(plain.join("probe.json"))["governed_by"],
        "experiments/real_walk_v1/RULE.md"
    );
}

/// `training.decay_exempt` — the patterns whose matching parameters AdamW does
/// not decay — reaches `probe.json` and both checkpoint metadata blocks as the
/// patterns, the parameter names they resolved to and the patterns that matched
/// nothing, so a reader can tell which run had the exemption and over what. A
/// config that names none records the empty set; a malformed value is refused;
/// `--print-capacity` is untouched by the switch (it trains nothing).
#[test]
fn the_decay_exempt_set_round_trips_into_probe_and_the_checkpoints() {
    if !config().exists() {
        eprintln!("skipped: the foundation checkout is not beside this one");
        return;
    }
    let root = tmp("decay-exempt");
    std::fs::create_dir_all(&root).unwrap();
    let patterns = ["greedy_tau", "*norm*.weight", "*bias"];
    let mut cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(config()).unwrap()).unwrap();
    // the fixture config decays nothing; the exemption is only visible at a
    // non-zero decay, and the fixture model has no `greedy_tau` (no prior), so
    // that pattern also exercises the "matched nothing" record
    cfg["training"]["weight_decay"] = 0.01.into();
    cfg["training"]["decay_exempt"] = serde_json::json!(patterns);
    let named = root.join("training-config.decay-exempt.json");
    std::fs::write(&named, serde_json::to_string(&cfg).unwrap()).unwrap();
    let base = |cfg: &PathBuf, o: &PathBuf| -> Vec<String> {
        vec![
            "--config".into(),
            cfg.to_string_lossy().into(),
            "--output".into(),
            o.to_string_lossy().into(),
            "--model-seed".into(),
            "5".into(),
            "--fixture".into(),
            "--train-episodes".into(),
            "6".into(),
            "--screen-episodes".into(),
            "3".into(),
            "--updates".into(),
            "2".into(),
            "--save-checkpoint".into(),
            "--checkpoint-every".into(),
            "1".into(),
        ]
    };
    let read = |p: PathBuf| -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
    };
    let out = root.join("run");
    let args = base(&named, &out);
    let o = run(&args.iter().map(String::as_str).collect::<Vec<_>>());
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let probe = read(out.join("probe.json"));
    let exempt = &probe["decay_exempt"];
    assert_eq!(exempt["patterns"], serde_json::json!(patterns));
    assert_eq!(
        exempt["unmatched_patterns"],
        serde_json::json!(["greedy_tau"])
    );
    let names: Vec<&str> = exempt["parameters"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    for want in [
        "blocks.0.context_attention.in_proj_bias",
        "blocks.0.norm_ff.weight",
        "candidate_encoder.bias",
        "candidate_norm.weight",
        "context_norm.weight",
        "score_head.2.bias",
    ] {
        assert!(names.contains(&want), "{want} is not in {names:?}");
    }
    for never in [
        "candidate_encoder.weight",
        "blocks.0.context_bias.weight",
        "score_head.0.weight",
        "greedy_tau",
    ] {
        assert!(!names.contains(&never), "{never} must still be decayed");
    }
    // 24 on the same model with the greedy prior (`hf-model`'s `adamw.rs`),
    // less `greedy_tau`, which this config does not build
    assert_eq!(names.len(), 23);
    // both checkpoints carry the same record: the final one holds no config, so
    // the patterns would otherwise be unrecoverable from it
    for stem in ["checkpoint", "checkpoint-latest"] {
        assert_eq!(
            read(out.join(format!("{stem}.json")))["decay_exempt"],
            *exempt,
            "{stem}.json"
        );
    }
    // a config naming none records the empty set rather than nothing at all
    let plain = root.join("plain");
    let args = base(&config(), &plain);
    let o = run(&args.iter().map(String::as_str).collect::<Vec<_>>());
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(
        read(plain.join("probe.json"))["decay_exempt"],
        serde_json::json!({"patterns": [], "parameters": [], "unmatched_patterns": []})
    );
    // a malformed value is a broken config, not a silent empty list
    for bad in [serde_json::json!("greedy_tau"), serde_json::json!([7])] {
        cfg["training"]["decay_exempt"] = bad;
        std::fs::write(&named, serde_json::to_string(&cfg).unwrap()).unwrap();
        let args = base(&named, &root.join("bad"));
        let o = run(&args.iter().map(String::as_str).collect::<Vec<_>>());
        assert_eq!(o.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&o.stderr).contains("decay_exempt"));
    }
    // --print-capacity reads no training block of this kind: same bytes either way
    cfg["training"]["decay_exempt"] = serde_json::json!(patterns);
    std::fs::write(&named, serde_json::to_string(&cfg).unwrap()).unwrap();
    let with = run(&["--print-capacity", "8", "--config", named.to_str().unwrap()]);
    let without = run(&[
        "--print-capacity",
        "8",
        "--config",
        config().to_str().unwrap(),
    ]);
    assert!(with.status.success() && without.status.success());
    assert_eq!(with.stdout, without.stdout);
    // a resume carries the exemption in the config, never in the optimiser
    // state: the same config resumes, a changed `decay_exempt` is band H
    let latest = out.join("checkpoint-latest.json");
    let resume = |cfg_path: &PathBuf| -> std::process::Output {
        let mut args = base(cfg_path, &out);
        let updates = args.iter().position(|a| a == "--updates").unwrap();
        args[updates + 1] = "3".into();
        args.extend(["--resume", latest.to_str().unwrap()].map(String::from));
        run(&args.iter().map(String::as_str).collect::<Vec<_>>())
    };
    let mut changed = cfg.clone();
    changed["training"]["decay_exempt"] = serde_json::json!(["greedy_tau", "*bias"]);
    let changed_path = root.join("training-config.decay-exempt-changed.json");
    std::fs::write(&changed_path, serde_json::to_string(&changed).unwrap()).unwrap();
    let o = resume(&changed_path);
    assert_eq!(
        o.status.code(),
        Some(2),
        "a changed exemption must not resume"
    );
    assert!(String::from_utf8_lossy(&o.stderr).contains("config differs"));
    let o = resume(&named);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(read(out.join("probe.json"))["decay_exempt"], *exempt);
    let _ = std::fs::remove_dir_all(&root);
}

/// `training.clip_max_norm` — the gradient-norm clip every update applies —
/// reaches `probe.json` and both checkpoint metadata blocks as the value in
/// effect, and is the value the training step actually uses. ABSENT is 1.0, the
/// constant the runner passed before the key existed, and is RECORDED as 1.0 so
/// no reader has to infer it; `null` is no clipping at all, and the pre-clip
/// norm is still computed and still logged in `updates.jsonl`; a malformed value
/// is refused; a value changed across a resume is band H.
#[test]
fn the_clip_max_norm_round_trips_into_probe_and_the_checkpoints() {
    if !config().exists() {
        eprintln!("skipped: the foundation checkout is not beside this one");
        return;
    }
    let root = tmp("clip-max-norm");
    std::fs::create_dir_all(&root).unwrap();
    let stock: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(config()).unwrap()).unwrap();
    let with = |clip: serde_json::Value, name: &str| -> PathBuf {
        let mut cfg = stock.clone();
        cfg["training"]["clip_max_norm"] = clip;
        let p = root.join(format!("training-config.{name}.json"));
        std::fs::write(&p, serde_json::to_string(&cfg).unwrap()).unwrap();
        p
    };
    let base = |cfg: &PathBuf, o: &PathBuf| -> Vec<String> {
        vec![
            "--config".into(),
            cfg.to_string_lossy().into(),
            "--output".into(),
            o.to_string_lossy().into(),
            "--model-seed".into(),
            "5".into(),
            "--fixture".into(),
            "--train-episodes".into(),
            "6".into(),
            "--screen-episodes".into(),
            "3".into(),
            "--updates".into(),
            "2".into(),
            "--save-checkpoint".into(),
            "--checkpoint-every".into(),
            "1".into(),
        ]
    };
    let read = |p: PathBuf| -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
    };
    let first_row = |o: &PathBuf| -> serde_json::Value {
        let log = std::fs::read_to_string(o.join("updates.jsonl")).unwrap();
        serde_json::from_str(log.lines().next().unwrap()).unwrap()
    };
    let train = |cfg: &PathBuf, o: &PathBuf| {
        let args = base(cfg, o);
        let out = run(&args.iter().map(String::as_str).collect::<Vec<_>>());
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    };

    // the three configurations: the key absent, the key null, the key 10
    let null_cfg = with(serde_json::Value::Null, "null");
    let ten_cfg = with(serde_json::json!(10.0), "ten");
    let absent = root.join("absent");
    let none = root.join("none");
    let ten = root.join("ten");
    train(&config(), &absent);
    train(&null_cfg, &none);
    train(&ten_cfg, &ten);

    // (1) the value in effect is recorded in all three places, the absent key
    //     as the 1.0 it means rather than as nothing at all
    for (out, want) in [
        (&absent, serde_json::json!(1.0)),
        (&none, serde_json::Value::Null),
        (&ten, serde_json::json!(10.0)),
    ] {
        for file in ["probe.json", "checkpoint.json", "checkpoint-latest.json"] {
            let v = read(out.join(file));
            assert_eq!(
                v.get("clip_max_norm"),
                Some(&want),
                "{}/{file}",
                out.display()
            );
        }
    }

    // (2) the pre-clip norm is the same number in all three runs' first update —
    //     the clip never changes what is logged, `null` included — and it sits
    //     between the two clips, so 1.0 fires and 10 does not
    let norm = first_row(&absent)["grad_norm"].as_f64().unwrap();
    assert!(
        norm > 1.0 && norm < 10.0,
        "the premise of this test: the fixture's first pre-clip norm {norm} \
         must be above 1.0 and below 10"
    );
    for out in [&none, &ten] {
        assert_eq!(
            first_row(out)["grad_norm"].as_f64(),
            Some(norm),
            "{}: the pre-clip norm must not depend on the clip",
            out.display()
        );
        assert_eq!(first_row(out)["total"], first_row(&absent)["total"]);
    }

    // (3) and the value is USED, not merely recorded: with the norm above 1.0
    //     the clipped run reaches different weights, and with it below 10 the
    //     clip of 10 cannot fire, so that run's weights are the unclipped ones
    let weights = |o: &PathBuf| std::fs::read(o.join("checkpoint.safetensors")).unwrap();
    assert_ne!(
        weights(&absent),
        weights(&none),
        "a clip of 1.0 below the norm must move the run off the unclipped path"
    );
    assert_eq!(
        weights(&ten),
        weights(&none),
        "a clip above the norm is no clip at all"
    );

    // (4) a malformed value is a broken config, not a silent default
    for bad in [
        serde_json::json!(-1.0),
        serde_json::json!(0),
        serde_json::json!("1.0"),
        serde_json::json!([1.0]),
        serde_json::json!(true),
    ] {
        let cfg = with(bad.clone(), "bad");
        let args = base(&cfg, &root.join("bad"));
        let o = run(&args.iter().map(String::as_str).collect::<Vec<_>>());
        assert_eq!(o.status.code(), Some(2), "{bad} must be refused");
        assert!(String::from_utf8_lossy(&o.stderr).contains("clip_max_norm"));
    }

    // and `--print-capacity` is untouched by the key: it trains nothing, so it
    // reads no training block of this kind — the same bytes either way
    let with_key = run(&[
        "--print-capacity",
        "8",
        "--config",
        ten_cfg.to_str().unwrap(),
    ]);
    let without = run(&[
        "--print-capacity",
        "8",
        "--config",
        config().to_str().unwrap(),
    ]);
    assert!(with_key.status.success() && without.status.success());
    assert_eq!(with_key.stdout, without.stdout);

    // (5) the clip lives in the config: a resume under the same config carries
    //     it, and a resume that changes it — an absent key against an explicit
    //     null included, which are different configs — is band H
    let resume = |cfg_path: &PathBuf, o: &PathBuf| -> std::process::Output {
        let mut args = base(cfg_path, o);
        let updates = args.iter().position(|a| a == "--updates").unwrap();
        args[updates + 1] = "3".into();
        let latest = o.join("checkpoint-latest.json");
        args.extend(["--resume", latest.to_str().unwrap()].map(String::from));
        run(&args.iter().map(String::as_str).collect::<Vec<_>>())
    };
    for (cfg, out) in [(&config(), &none), (&null_cfg, &absent), (&ten_cfg, &none)] {
        let o = resume(cfg, out);
        assert_eq!(
            o.status.code(),
            Some(2),
            "a changed clip must not resume: {} into {}",
            cfg.display(),
            out.display()
        );
        assert!(String::from_utf8_lossy(&o.stderr).contains("config differs"));
    }
    let o = resume(&null_cfg, &none);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(
        read(none.join("probe.json"))["clip_max_norm"],
        serde_json::Value::Null
    );
    assert_eq!(read(none.join("probe.json"))["updates"], 3);
    let o = resume(&config(), &absent);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(read(absent.join("probe.json"))["clip_max_norm"], 1.0);
    let _ = std::fs::remove_dir_all(&root);
}

/// `greedy_tau` — the greedy prior's temperature as the optimiser step left it
/// — is logged on every `updates.jsonl` row, so a run's tau trajectory can be
/// read from the log rather than reconstructed from a checkpoint's tensors: a
/// finite number when the config asks for the prior, `null` when it does not.
/// The row RECORDS the step, it does not enter it: the keys that were there
/// keep their order with the new one last, and the training step's own golden —
/// the digest hf-model's `adamw` test measures after one and two steps — is
/// untouched, since reading a tensor after `step()` cannot move it.
#[test]
fn greedy_tau_is_logged_on_every_update_row() {
    if !config().exists() {
        eprintln!("skipped: the foundation checkout is not beside this one");
        return;
    }
    // the scale is named in the config rather than left to hf-model's default,
    // so the bound below tests this run and not that default
    const SCALE: f64 = 10.0;
    let root = tmp("greedy-tau");
    std::fs::create_dir_all(&root).unwrap();
    let mut cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(config()).unwrap()).unwrap();
    cfg["model"]["greedy_prior"] = serde_json::json!(true);
    cfg["model"]["greedy_prior_scale"] = serde_json::json!(SCALE);
    let prior_cfg = root.join("training-config.prior.json");
    std::fs::write(&prior_cfg, serde_json::to_string(&cfg).unwrap()).unwrap();

    let train = |cfg: &PathBuf, o: &PathBuf, updates: &str| {
        let out = run(&[
            "--config",
            cfg.to_str().unwrap(),
            "--output",
            o.to_str().unwrap(),
            "--model-seed",
            "5",
            "--fixture",
            "--train-episodes",
            "6",
            "--screen-episodes",
            "3",
            "--updates",
            updates,
        ]);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    };
    let rows = |o: &PathBuf| -> Vec<serde_json::Value> {
        std::fs::read_to_string(o.join("updates.jsonl"))
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    };
    let with = root.join("prior");
    let without = root.join("no-prior");
    train(&prior_cfg, &with, "4");
    train(&config(), &without, "2");
    let prior_rows = rows(&with);
    let plain_rows = rows(&without);

    // (1) both ways: every row carries the key, LAST, with the keys that were
    //     there before it unchanged and in their order
    assert_eq!(prior_rows.len(), 4);
    assert_eq!(plain_rows.len(), 2);
    for r in prior_rows.iter().chain(plain_rows.iter()) {
        let keys: Vec<&str> = r.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "update",
                "edge",
                "distance",
                "stop",
                "residual",
                "total",
                "grad_norm",
                "registered",
                "expansions",
                "draws",
                "distinct_seen",
                "seconds",
                "greedy_tau",
                "finite",
            ]
        );
    }

    // (2) with the prior: a finite tau on every row, at or below the scale it
    //     was initialised to by the time the first update logs it, and moving
    //     every update — a constant column would mean a value read once, or the
    //     initialisation read instead of the parameter
    let taus: Vec<f64> = prior_rows
        .iter()
        .map(|r| r["greedy_tau"].as_f64().expect("a number with the prior"))
        .collect();
    assert!(taus.iter().all(|t| t.is_finite()), "{taus:?}");
    assert!(
        taus[0] <= SCALE + 1e-6,
        "row 1 tau {} is above the initial {SCALE}",
        taus[0]
    );
    for w in taus.windows(2) {
        assert!((w[1] - w[0]).abs() > 0.0, "tau did not move: {taus:?}");
    }

    // (3) without the prior there is no tensor to read: the key is present and
    //     null, never absent and never a stand-in number
    for r in &plain_rows {
        assert_eq!(r["greedy_tau"], serde_json::Value::Null);
    }
    let _ = std::fs::remove_dir_all(&root);
}

/// The engine's provenance is what the BINARY was built from, not what `git`
/// says in the checkout beside it at run time: a release binary built at one
/// commit used to stamp every artifact with whatever HEAD had become. Both are
/// reported now, with the binary's own digest to tie an artifact to the bytes
/// that wrote it.
#[test]
fn engine_info_reports_the_build_the_binary_came_from() {
    let o = run(&["--engine-info"]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let info: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&o.stdout)).unwrap();
    for key in [
        "engine_build_head",
        "engine_build_dirty",
        "engine_build_time",
        "engine_binary_sha256",
    ] {
        assert!(info.get(key).is_some(), "{key} missing from --engine-info");
    }
    // the build script's values, as this test crate was given them too
    assert_eq!(info["engine_build_head"], env!("ENGINE_BUILD_HEAD"));
    assert_eq!(
        info["engine_build_dirty"],
        env!("ENGINE_BUILD_DIRTY") == "true"
    );
    assert_eq!(info["engine_build_time"], env!("ENGINE_BUILD_TIME"));
    let head = info["engine_build_head"].as_str().unwrap();
    assert!(
        head == "unknown" || (head.len() == 40 && head.chars().all(|c| c.is_ascii_hexdigit())),
        "{head} is neither a commit nor unknown"
    );
    // the digest is of this very binary
    let (_, digest) =
        hf_core::sha256_file(std::path::Path::new(env!("CARGO_BIN_EXE_hf-stage0"))).unwrap();
    assert_eq!(info["engine_binary_sha256"], digest);
    // engine_head keeps its meaning: the checkout's HEAD at run time
    assert!(info["engine_head"].is_string());
}

/// Every artifact a run writes carries the same provenance, so a reader holding
/// only a probe or a checkpoint can tell which binary produced it.
#[test]
fn a_run_stamps_every_artifact_with_the_binarys_provenance() {
    if !config().exists() {
        eprintln!("skipped: the foundation checkout is not beside this one");
        return;
    }
    let root = tmp("provenance");
    let out = root.join("run");
    let args: Vec<String> = vec![
        "--config".into(),
        config().to_string_lossy().into(),
        "--output".into(),
        out.to_string_lossy().into(),
        "--model-seed".into(),
        "5".into(),
        "--fixture".into(),
        "--train-episodes".into(),
        "8".into(),
        "--screen-episodes".into(),
        "4".into(),
        "--updates".into(),
        "2".into(),
        "--eval-every".into(),
        "2".into(),
        "--save-checkpoint".into(),
        "--checkpoint-every".into(),
        "2".into(),
    ];
    let o = run(&args.iter().map(String::as_str).collect::<Vec<_>>());
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let (_, digest) =
        hf_core::sha256_file(std::path::Path::new(env!("CARGO_BIN_EXE_hf-stage0"))).unwrap();
    let read = |p: PathBuf| -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{p:?}: {e}")))
            .unwrap()
    };
    // probe, both checkpoint metas, and a re-evaluation's reeval.json
    let re = root.join("reeval");
    let mut re_args: Vec<String> = args
        .iter()
        .take_while(|a| *a != "--updates")
        .cloned()
        .collect();
    re_args[3] = re.to_string_lossy().into();
    re_args.extend(
        [
            "--reevaluate-checkpoint",
            out.join("checkpoint.json").to_str().unwrap(),
        ]
        .map(String::from),
    );
    let o = run(&re_args.iter().map(String::as_str).collect::<Vec<_>>());
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    for artifact in [
        out.join("probe.json"),
        out.join("checkpoint.json"),
        out.join("checkpoint-latest.json"),
        re.join("reeval.json"),
    ] {
        let v = read(artifact.clone());
        let what = artifact.display();
        assert_eq!(v["engine_build_head"], env!("ENGINE_BUILD_HEAD"), "{what}");
        assert_eq!(
            v["engine_build_dirty"],
            env!("ENGINE_BUILD_DIRTY") == "true",
            "{what}"
        );
        assert_eq!(v["engine_binary_sha256"], digest, "{what}");
        assert_eq!(v["engine_stale_allowed"], false, "{what}");
        // the old key is kept, and still means the checkout at run time
        assert!(v["engine_head"].is_string(), "{what}");
    }
    let _ = std::fs::remove_dir_all(&root);
}

/// A binary that is not the checkout's is refused before anything is read: the
/// artifacts it would write would name a commit whose code never ran.
/// `HF_TEST_ENGINE_HEAD_OVERRIDE` moves the run-time head out of step with the
/// build's (a debug build only; the release binary ignores it).
#[test]
fn a_stale_binary_is_refused_unless_the_operator_allows_it() {
    if !config().exists() {
        eprintln!("skipped: the foundation checkout is not beside this one");
        return;
    }
    let root = tmp("stale");
    let elsewhere = "0000000000000000000000000000000000000000";
    let base = |o: &PathBuf| -> Vec<String> {
        vec![
            "--config".into(),
            config().to_string_lossy().into(),
            "--output".into(),
            o.to_string_lossy().into(),
            "--model-seed".into(),
            "5".into(),
            "--train-episodes".into(),
            "8".into(),
            "--screen-episodes".into(),
            "4".into(),
            "--updates".into(),
            "2".into(),
        ]
    };
    let env = [("HF_TEST_ENGINE_HEAD_OVERRIDE", elsewhere)];
    // a real run: band H, and nothing further is attempted
    let args = base(&root.join("real"));
    let o = run_env(&args.iter().map(String::as_str).collect::<Vec<_>>(), &env);
    assert_eq!(o.status.code(), Some(2));
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.contains("band H"), "{err}");
    assert!(err.contains("stale engine binary"), "{err}");
    assert!(err.contains("--allow-stale-engine"), "{err}");
    assert!(!root.join("real").exists(), "nothing was written");
    // with the flag the gate is passed, and the next gate — the pin — speaks
    let mut args = base(&root.join("real"));
    args.push("--allow-stale-engine".into());
    let o = run_env(&args.iter().map(String::as_str).collect::<Vec<_>>(), &env);
    assert_eq!(o.status.code(), Some(2));
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(!err.contains("stale engine binary"), "{err}");
    assert!(err.contains("preregistration-commit"), "{err}");
    // the fixture world is exempt, as it is from every governance gate: it is
    // never evidence, and it says which binary ran all the same
    let out = root.join("fixture");
    let mut args = base(&out);
    args.push("--fixture".into());
    let o = run_env(&args.iter().map(String::as_str).collect::<Vec<_>>(), &env);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let probe: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("probe.json")).unwrap()).unwrap();
    assert_eq!(probe["engine_head"], elsewhere);
    assert_eq!(probe["engine_build_head"], env!("ENGINE_BUILD_HEAD"));
    assert_eq!(probe["engine_stale_allowed"], false);
    // and the allowance is recorded where a reader will see it
    let allowed = root.join("allowed");
    let mut args = base(&allowed);
    args.extend(["--fixture", "--allow-stale-engine"].map(String::from));
    let o = run_env(&args.iter().map(String::as_str).collect::<Vec<_>>(), &env);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let probe: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(allowed.join("probe.json")).unwrap())
            .unwrap();
    assert_eq!(probe["engine_stale_allowed"], true);
    let _ = std::fs::remove_dir_all(&root);
}

/// `serde_json` writes a non-finite `f64` as `null`, so a NaN loss or gradient
/// norm used to reach `updates.jsonl` as an absent-looking key while the
/// optimiser carried the NaN into every weight and the run went on producing
/// rows for ever. Part B's `clip_max_norm: null` arm has nothing else to bound
/// it, so the engine now says on every row whether the update was finite and
/// stops at the first that is not, before the step.
#[test]
fn a_non_finite_update_halts_the_run_before_the_step() {
    if !config().exists() {
        eprintln!("skipped: the foundation checkout is not beside this one");
        return;
    }
    let root = tmp("non-finite");
    let out = root.join("run");
    let args: Vec<String> = [
        "--config",
        config().to_str().unwrap(),
        "--output",
        out.to_str().unwrap(),
        "--model-seed",
        "5",
        "--fixture",
        "--train-episodes",
        "6",
        "--screen-episodes",
        "3",
        "--updates",
        "4",
        "--eval-every",
        "2",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let argv: Vec<&str> = args.iter().map(String::as_str).collect();
    let o = run_env(&argv, &[("HF_TEST_INJECT_NAN_AT_UPDATE", "3")]);

    // (1) the run stops at update 3, exit 2, saying which update it was
    assert_eq!(o.status.code(), Some(2));
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        err.contains("band H: non-finite update 3"),
        "stderr was {err}"
    );

    // (2) the rows: three of them, the first two finite, the third not — and
    //     update 4 never happened
    let rows: Vec<serde_json::Value> = std::fs::read_to_string(out.join("updates.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(rows.len(), 3, "the run stops at the non-finite update");
    for r in &rows[..2] {
        assert_eq!(r["finite"], true, "{r}");
        assert!(r["total"].as_f64().unwrap().is_finite());
    }
    assert_eq!(rows[2]["update"], 3);
    assert_eq!(rows[2]["finite"], false);
    // the very silence the flag exists for: the NaN total is `null` in the row
    assert!(rows[2]["total"].is_null());

    // (3) instability.json names the update, the fields and the run
    let bad: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("instability.json")).unwrap())
            .unwrap();
    assert_eq!(bad["update"], 3);
    assert_eq!(bad["last_finite_update"], 2);
    assert_eq!(bad["model_seed"], 5);
    assert_eq!(bad["evidence"], false);
    assert_eq!(bad["injected"], true);
    // the fixture config names no clip, which is 1.0
    assert_eq!(bad["clip_max_norm"], 1.0);
    let fields: Vec<&str> = bad["non_finite"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(fields.contains(&"total"), "{fields:?}");
    assert!(fields.contains(&"grad_norm"), "{fields:?}");

    // (4) no probe.json — nothing downstream may read this run as finished —
    //     but the last finite weights are checkpointed, beside the marker a
    //     launcher can test
    assert!(!out.join("probe.json").exists());
    assert!(out.join("halted").exists());
    let ck: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("checkpoint-latest.json")).unwrap())
            .unwrap();
    assert_eq!(ck["update"], 3);

    // (5) the halted directory is not a base to build on: a second run into it
    //     refuses rather than truncating the log and finishing over the halt
    let again = run_env(&argv, &[]);
    assert_eq!(again.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&again.stderr).contains("halted"),
        "{}",
        String::from_utf8_lossy(&again.stderr)
    );
    assert!(!out.join("probe.json").exists());
    let _ = std::fs::remove_dir_all(&root);
}

/// The control for the test above: a run that does not diverge says so on every
/// row and writes none of the halt's artifacts.
#[test]
fn an_ordinary_run_says_every_update_was_finite() {
    if !config().exists() {
        eprintln!("skipped: the foundation checkout is not beside this one");
        return;
    }
    let root = tmp("all-finite");
    let out = root.join("run");
    let o = run(&[
        "--config",
        config().to_str().unwrap(),
        "--output",
        out.to_str().unwrap(),
        "--model-seed",
        "5",
        "--fixture",
        "--train-episodes",
        "6",
        "--screen-episodes",
        "3",
        "--updates",
        "4",
        "--eval-every",
        "4",
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let rows: Vec<serde_json::Value> = std::fs::read_to_string(out.join("updates.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(rows.len(), 4);
    for r in &rows {
        assert_eq!(r["finite"], true, "{r}");
        for k in ["edge", "distance", "stop", "residual", "total", "grad_norm"] {
            assert!(r[k].as_f64().expect(k).is_finite(), "{k} in {r}");
        }
    }
    assert!(out.join("probe.json").exists());
    assert!(!out.join("instability.json").exists());
    assert!(!out.join("halted").exists());
    let _ = std::fs::remove_dir_all(&root);
}

/// A fixture-mode config written here, so these cases need no foundation
/// checkout: the smoke configuration of `real_walk_v1`, with the `data` block
/// under test.
fn config_with_data(root: &PathBuf, data: Option<serde_json::Value>) -> PathBuf {
    std::fs::create_dir_all(root).unwrap();
    let mut config = serde_json::json!({
        "record_kind": "real_walk_stage0_config",
        "note": "fixture-mode smoke configuration written by the test; never evidence",
        "training_authorized": false,
        "model": {
            "hidden_dimension": 32, "self_attention_heads": 2, "feedforward_multiplier": 2,
            "score_hidden_dimension": 16, "coverage_hidden_dimension": 8,
            "traversal_blocks": 1, "dropout": 0.0
        },
        "sampler": {"subgraph_size": 64, "target_distance": 3, "removal_level": 2, "cost_epsilon": 0.5},
        "training": {"learning_rate": 0.001, "weight_decay": 0.0, "update_count": 1, "microbatch_size": 2}
    });
    if let Some(block) = data {
        config.as_object_mut().unwrap().insert("data".into(), block);
    }
    let path = root.join("config.json");
    std::fs::write(&path, serde_json::to_string_pretty(&config).unwrap()).unwrap();
    path
}

/// `data.query_source` and its flag come as a pair, and neither is silently
/// ignored: each alone exits 2, an unknown value exits 2, and the stage-1
/// source is refused on the fixture world, which samples stage-0 episodes and
/// has no questions to read.
#[test]
fn the_query_source_and_its_sidecar_refuse_every_half_configuration() {
    let root = tmp("query-source");
    let out_dir = root.join("run");
    let queries = root.join("queries");
    std::fs::create_dir_all(&queries).unwrap();
    let base = |cfg: &PathBuf| -> Vec<String> {
        vec![
            "--config".into(),
            cfg.to_string_lossy().into(),
            "--output".into(),
            out_dir.to_string_lossy().into(),
            "--model-seed".into(),
            "5".into(),
            "--fixture".into(),
            "--train-episodes".into(),
            "4".into(),
            "--screen-episodes".into(),
            "2".into(),
            "--updates".into(),
            "1".into(),
        ]
    };
    let go = |args: Vec<String>| run(&args.iter().map(String::as_str).collect::<Vec<_>>());

    // the sidecar without the config key
    let plain = config_with_data(&root.join("plain"), None);
    let mut args = base(&plain);
    args.extend([
        "--query-embeddings-dir".into(),
        queries.to_string_lossy().into(),
    ]);
    let out = go(args);
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(stderr.contains("data.query_source"), "{stderr}");

    // the config key without the sidecar
    let stage1 = config_with_data(
        &root.join("stage1"),
        Some(serde_json::json!({"query_source": "episode_query"})),
    );
    let out = go(base(&stage1));
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(stderr.contains("--query-embeddings-dir"), "{stderr}");

    // both, on the fixture world, which is stage 0
    let mut args = base(&stage1);
    args.extend([
        "--query-embeddings-dir".into(),
        queries.to_string_lossy().into(),
    ]);
    let out = go(args);
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(stderr.contains("fixture world"), "{stderr}");

    // an unknown source is a broken config, not the default
    let odd = config_with_data(
        &root.join("odd"),
        Some(serde_json::json!({"query_source": "the target, obviously"})),
    );
    let out = go(base(&odd));
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(stderr.contains("unknown query source"), "{stderr}");

    // and the default is the stage-0 source: an absent block runs
    let out = go(base(&plain));
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let probe: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out_dir.join("probe.json")).unwrap())
            .unwrap();
    assert_eq!(probe["query_source"], "target_embedding");
    assert_eq!(probe["query_embeddings"], serde_json::Value::Null);
    assert_eq!(probe["embedding_coverage"]["share"], 1.0);
    let _ = std::fs::remove_dir_all(&root);
}

/// `--expect-embedding-coverage`: a floor the cache cannot meet exits 2 before
/// anything is trained; one it meets is recorded in `probe.json`.
#[test]
fn the_embedding_coverage_floor_refuses_and_is_recorded() {
    let root = tmp("coverage");
    let cfg = config_with_data(&root, None);
    let impossible = root.join("never");
    let out = run(&[
        "--config",
        cfg.to_str().unwrap(),
        "--output",
        impossible.to_str().unwrap(),
        "--model-seed",
        "5",
        "--fixture",
        "--train-episodes",
        "4",
        "--screen-episodes",
        "2",
        "--updates",
        "1",
        "--expect-embedding-coverage",
        "1.1",
    ]);
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(stderr.contains("--expect-embedding-coverage"), "{stderr}");
    assert!(
        !impossible.join("probe.json").exists(),
        "nothing downstream reads a refused run as finished"
    );
    let met = root.join("met");
    let out = run(&[
        "--config",
        cfg.to_str().unwrap(),
        "--output",
        met.to_str().unwrap(),
        "--model-seed",
        "5",
        "--fixture",
        "--train-episodes",
        "4",
        "--screen-episodes",
        "2",
        "--updates",
        "1",
        "--expect-embedding-coverage",
        "1.0",
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let probe: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(met.join("probe.json")).unwrap()).unwrap();
    assert_eq!(probe["embedding_coverage"]["share"], 1.0);
    assert_eq!(probe["embedding_coverage"]["expected"], 1.0);
    assert!(probe["embedding_coverage"]["distinct"].as_u64().unwrap() > 0);
    let _ = std::fs::remove_dir_all(&root);
}

/// The fixture world's node vectors as a v5 cache on disk.
fn write_node_cache(dir: &std::path::Path) {
    let (_, embeddings) = hf_episodes::fixture::fixture_world(5, 400, 1600, 8);
    std::fs::create_dir_all(dir).unwrap();
    let mut f = std::fs::File::create(dir.join("vectors.jsonl")).unwrap();
    let mut names: Vec<&String> = embeddings.keys().collect();
    names.sort();
    for n in &names {
        hf_embed::append_vector(&mut f, n, &embeddings[*n]).unwrap();
    }
    hf_embed::write_manifest(dir, &fixture_manifest(names.len() as u64, 8)).unwrap();
}

fn fixture_manifest(count: u64, dimension: u32) -> hf_embed::Manifest {
    hf_embed::Manifest {
        record_kind: hf_embed::MANIFEST_KIND.into(),
        model: "fixture".into(),
        model_digest: "fixture".into(),
        base_url: "none".into(),
        dimension,
        count,
        text_char_limit: 6000,
        text_sha256: None,
        truncated: Default::default(),
        training_authorized: false,
        extra: Default::default(),
    }
}

/// A question-vector sidecar: an ordinary v5 cache whose ids are EPISODE ids.
fn write_query_cache(dir: &std::path::Path, ids: &[String]) {
    std::fs::create_dir_all(dir).unwrap();
    let mut f = std::fs::File::create(dir.join("vectors.jsonl")).unwrap();
    for (i, id) in ids.iter().enumerate() {
        let v: Vec<f64> = (0..8)
            .map(|j| ((i as f64 + 1.0) * 0.37 + j as f64 * 0.11).sin())
            .collect();
        hf_embed::append_vector(&mut f, id, &v).unwrap();
    }
    hf_embed::write_manifest(dir, &fixture_manifest(ids.len() as u64, 8)).unwrap();
}

/// The golden fixture split rewritten as the stage-1 one a teacher would
/// write: no `target_node`, a `query`, the hidden payload carried across.
/// Returns every episode id the run will read, in file order.
fn write_stage1_splits(source: &std::path::Path, destination: &std::path::Path) -> Vec<String> {
    let mut ids = Vec::new();
    for split in ["train", "screen"] {
        let (episodes, artifacts) = hf_io::read_split(&source.join(split)).unwrap();
        let out: Vec<Result<hf_io::EpisodeOut, hf_core::HfError>> = episodes
            .iter()
            .map(|e| {
                ids.push(e.episode_id.clone());
                let mut visible = serde_json::to_value(&e.visible).unwrap();
                let object = visible.as_object_mut().unwrap();
                object.remove("target_node");
                object.insert("stage".into(), hf_io::STAGES[1].into());
                object.insert(
                    "query".into(),
                    serde_json::json!(format!("what does {} lead to?", e.visible.start_node)),
                );
                Ok(hf_io::EpisodeOut {
                    episode_id: e.episode_id.clone(),
                    visible,
                    hidden: serde_json::to_value(&e.hidden).unwrap(),
                })
            })
            .collect();
        hf_io::write_split(
            "fixture",
            hf_io::STAGES[1],
            split,
            &destination.join(split),
            out,
            &artifacts.graph,
            &artifacts.public["sampler"],
        )
        .unwrap();
    }
    ids
}

/// A config whose model is `relational-v6` at the fixture world's sizes —
/// `raw-v5`, the default, is refused under `episode_query`.
fn relational_config(root: &PathBuf, data: Option<serde_json::Value>) -> PathBuf {
    std::fs::create_dir_all(root).unwrap();
    let mut config = serde_json::json!({
        "record_kind": "real_walk_stage0_config",
        "note": "fixture-mode smoke configuration written by the test; never evidence",
        "training_authorized": false,
        "model": {
            "hidden_dimension": 32, "self_attention_heads": 2, "feedforward_multiplier": 2,
            "score_hidden_dimension": 16, "coverage_hidden_dimension": 8,
            "traversal_blocks": 1, "dropout": 0.0, "feature_set": "relational-v6"
        },
        "sampler": {"subgraph_size": 64, "target_distance": 3, "removal_level": 2, "cost_epsilon": 0.5},
        "training": {"learning_rate": 0.001, "weight_decay": 0.0, "update_count": 1, "microbatch_size": 2}
    });
    if let Some(block) = data {
        config.as_object_mut().unwrap().insert("data".into(), block);
    }
    let path = root.join("config.json");
    std::fs::write(&path, serde_json::to_string_pretty(&config).unwrap()).unwrap();
    path
}

/// The foundation's own HEAD, which is an ancestor of itself and so a pin a
/// real run accepts.
fn foundation_head() -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(foundation())
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

/// **The zero-shot read.** `--query-embeddings-dir` and
/// `--expect-embedding-coverage` are honoured on `--reevaluate-checkpoint`,
/// not only on a training run: the sidecar drives the walk (the rows carry
/// `question_greedy_overshoot` and never the stage-0 key), both are recorded
/// in `reeval.json`, an unreachable coverage floor exits 2 before a row is
/// written, and an episode the sidecar does not cover exits 2 rather than
/// walking on a silent zero vector.
#[test]
fn a_reevaluation_reads_the_query_sidecar_and_the_coverage_floor() {
    if !config().exists() {
        eprintln!("skipped: the foundation checkout is not beside this one");
        return;
    }
    let Some(pin) = foundation_head() else {
        eprintln!("skipped: the foundation checkout has no git head");
        return;
    };
    let goldens =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../hf-io/tests/goldens/fixture-split");
    let root = tmp("reeval-stage1");
    std::fs::create_dir_all(&root).unwrap();
    let cache = root.join("cache");
    write_node_cache(&cache);
    let splits = root.join("splits");
    let ids = write_stage1_splits(&goldens, &splits);
    assert_eq!(ids.len(), 18, "12 train + 6 screen golden episodes");
    let queries = root.join("queries");
    write_query_cache(&queries, &ids);

    // a checkpoint to re-evaluate, trained on the fixture world at the SAME
    // model configuration (the fixture world is 8-dimensional, as this cache is)
    let stage0 = relational_config(&root.join("stage0"), None);
    let trained = root.join("trained");
    let o = run(&[
        "--config",
        stage0.to_str().unwrap(),
        "--output",
        trained.to_str().unwrap(),
        "--model-seed",
        "5",
        "--fixture",
        "--train-episodes",
        "4",
        "--screen-episodes",
        "2",
        "--updates",
        "1",
        "--save-checkpoint",
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));

    let stage1 = relational_config(
        &root.join("stage1"),
        Some(serde_json::json!({"query_source": "episode_query"})),
    );
    // a real run takes the GPU when one is present; this is a CLI check, not a
    // training job, so it is pinned to the CPU
    let reeval =
        |out: &PathBuf, query_dir: &std::path::Path, floor: &str| -> std::process::Output {
            run_env(
                &[
                    "--config",
                    stage1.to_str().unwrap(),
                    "--output",
                    out.to_str().unwrap(),
                    "--model-seed",
                    "5",
                    "--family",
                    "fixture",
                    "--splits-dir",
                    splits.to_str().unwrap(),
                    "--embeddings-dir",
                    cache.to_str().unwrap(),
                    "--query-embeddings-dir",
                    query_dir.to_str().unwrap(),
                    "--expect-embedding-coverage",
                    floor,
                    "--screen-episodes",
                    "6",
                    "--reevaluate-checkpoint",
                    trained.join("checkpoint.json").to_str().unwrap(),
                    "--preregistration-commit",
                    &pin,
                    "--foundation-root",
                    foundation().to_str().unwrap(),
                    "--allow-stale-engine",
                ],
                CPU_ONLY,
            )
        };

    let out = root.join("reeval");
    let o = reeval(&out, &queries, "1.0");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let record: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("reeval.json")).unwrap()).unwrap();
    assert_eq!(record["query_source"], "episode_query");
    assert_eq!(
        record["query_embeddings"],
        serde_json::Value::from(queries.to_string_lossy().to_string()),
        "the sidecar the run read is named in the artifact"
    );
    assert_eq!(record["query_embedding_manifest"]["dimension"], 8);
    assert_eq!(record["query_embedding_manifest"]["count"], ids.len());
    assert_eq!(record["embedding_coverage"]["expected"], 1.0);
    assert_eq!(record["embedding_coverage"]["share"], 1.0);
    assert!(record["embedding_coverage"]["distinct"].as_u64().unwrap() > 0);
    let rows: Vec<serde_json::Value> = std::fs::read_to_string(out.join("evaluation_rows.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(rows.len(), 6, "the screen split was re-evaluated");
    for r in &rows {
        assert_eq!(r["update"], "reeval");
        // the sidecar really drove the walk: the stage-1 key, never the stage-0 one
        assert!(r.get("greedy_overshoot").is_none());
        assert!(r["question_greedy_overshoot"].is_i64());
        // and ENG-9's column is on the re-evaluation's rows too
        let rank = &r["cosine_rank_at_registration"];
        assert!(
            rank.is_null() || rank.as_u64().is_some_and(|k| k >= 1),
            "{rank}"
        );
    }

    // the floor is a gate on this path, not decoration
    let refused = root.join("refused");
    let o = reeval(&refused, &queries, "1.1");
    assert_eq!(o.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&o.stderr).to_string();
    assert!(stderr.contains("--expect-embedding-coverage"), "{stderr}");
    assert!(
        !refused.join("reeval.json").exists() && !refused.join("evaluation_rows.jsonl").exists(),
        "a refused re-evaluation writes nothing a reader would take as finished"
    );

    // an episode the sidecar does not cover exits 2 rather than walking on a
    // silent zero vector
    let partial = root.join("partial-queries");
    write_query_cache(&partial, &ids[..ids.len() - 1]);
    let short = root.join("short");
    let o = reeval(&short, &partial, "1.0");
    assert_eq!(o.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&o.stderr).to_string();
    assert!(stderr.contains("no query vector"), "{stderr}");
    assert!(!short.join("reeval.json").exists());
    let _ = std::fs::remove_dir_all(&root);
}

/// **The ALL-SEEN layout wrapper.** The teacher writes the episodes its admit
/// gate DISCARDED as a sibling `screen/discarded` of the admitted screen, and
/// ALL-SEEN needs rows for them, which only a re-evaluation of the arm's final
/// checkpoint over that sibling produces. `--splits-dir <root>` loads
/// `<root>/train` and `<root>/screen` and reads the graph manifest from
/// `<root>/train`, so the sibling has to be presented AS a `<root>/screen`.
///
/// **The wrapper is two symlinks and no new engine flag** (`hf-stage0` gains
/// nothing; § 6.5 (i) assigns the wrapper to OPS and this test to ENG):
///
/// ```text
/// W=$T/allseen-wrapper                       # anywhere writable, NOT under the teacher dir
/// mkdir -p "$W"
/// ln -s "$(realpath private/real-walk-v1/teacher/stage1-q-r3/train)"           "$W/train"
/// ln -s "$(realpath private/real-walk-v1/teacher/stage1-q-r3/screen/discarded)" "$W/screen"
/// $HF_STAGE0_BIN … --splits-dir "$W" --reevaluate-checkpoint …
/// ```
///
/// `W/train` is the admitted TRAIN split, unchanged: it supplies the graph
/// manifest and the sampler agreement, and a re-evaluation walks no episode of
/// it (see the screen-only sidecar test). This test is the "ENG confirms the
/// exact invocation and that the wrapper changes no episode" half: the run
/// through the wrapper reads exactly the discarded episodes, the manifest it
/// records is the sibling's own, and the wrapper's own `visible_sha256` is the
/// sibling's.
#[test]
fn the_allseen_wrapper_presents_the_discarded_sibling_as_a_screen_unchanged() {
    if !config().exists() {
        eprintln!("skipped: the foundation checkout is not beside this one");
        return;
    }
    let Some(pin) = foundation_head() else {
        eprintln!("skipped: the foundation checkout has no git head");
        return;
    };
    let goldens =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../hf-io/tests/goldens/fixture-split");
    let root = tmp("allseen-wrapper");
    std::fs::create_dir_all(&root).unwrap();
    let cache = root.join("cache");
    write_node_cache(&cache);
    let splits = root.join("splits");
    let ids = write_stage1_splits(&goldens, &splits);
    let screen_ids: Vec<String> = ids[12..].to_vec();

    // the teacher's discarded sibling: the LAST two screen episodes, written
    // beside the admitted screen as `screen/discarded`
    let discarded_ids: Vec<String> = screen_ids[screen_ids.len() - 2..].to_vec();
    {
        let (episodes, artifacts) = hf_io::read_split(&splits.join("screen")).unwrap();
        let out: Vec<Result<hf_io::EpisodeOut, hf_core::HfError>> = episodes
            .iter()
            .filter(|e| discarded_ids.contains(&e.episode_id))
            .map(|e| {
                Ok(hf_io::EpisodeOut {
                    episode_id: e.episode_id.clone(),
                    visible: serde_json::to_value(&e.visible).unwrap(),
                    hidden: serde_json::to_value(&e.hidden).unwrap(),
                })
            })
            .collect();
        assert_eq!(out.len(), 2);
        hf_io::write_split(
            "fixture",
            hf_io::STAGES[1],
            "screen",
            &splits.join("screen").join("discarded"),
            out,
            &artifacts.graph,
            &artifacts.public["sampler"],
        )
        .unwrap();
    }
    let sibling = splits.join("screen").join("discarded");
    let sibling_manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(sibling.join("manifest.public.json")).unwrap(),
    )
    .unwrap();

    // THE WRAPPER: two symlinks, nothing copied and nothing rewritten
    let wrapper = root.join("wrapper");
    std::fs::create_dir_all(&wrapper).unwrap();
    std::os::unix::fs::symlink(splits.join("train"), wrapper.join("train")).unwrap();
    std::os::unix::fs::symlink(&sibling, wrapper.join("screen")).unwrap();

    let queries = root.join("queries");
    write_query_cache(&queries, &discarded_ids);

    let stage0 = relational_config(&root.join("stage0"), None);
    let trained = root.join("trained");
    let o = run(&[
        "--config",
        stage0.to_str().unwrap(),
        "--output",
        trained.to_str().unwrap(),
        "--model-seed",
        "5",
        "--fixture",
        "--train-episodes",
        "4",
        "--screen-episodes",
        "2",
        "--updates",
        "1",
        "--save-checkpoint",
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let stage1 = relational_config(
        &root.join("stage1"),
        Some(serde_json::json!({"query_source": "episode_query"})),
    );
    let out = root.join("allseen");
    let o = run_env(
        &[
            "--config",
            stage1.to_str().unwrap(),
            "--output",
            out.to_str().unwrap(),
            "--model-seed",
            "5",
            "--family",
            "fixture",
            "--splits-dir",
            wrapper.to_str().unwrap(),
            "--embeddings-dir",
            cache.to_str().unwrap(),
            "--query-embeddings-dir",
            queries.to_str().unwrap(),
            "--expect-embedding-coverage",
            "1.0",
            "--screen-episodes",
            "6",
            "--reevaluate-checkpoint",
            trained.join("checkpoint.json").to_str().unwrap(),
            "--preregistration-commit",
            &pin,
            "--foundation-root",
            foundation().to_str().unwrap(),
            "--allow-stale-engine",
        ],
        CPU_ONLY,
    );
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let rows: Vec<serde_json::Value> = std::fs::read_to_string(out.join("evaluation_rows.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let read: Vec<String> = rows
        .iter()
        .map(|r| r["episode_id"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(
        read, discarded_ids,
        "the wrapper reads the sibling's episodes, in its order"
    );
    for r in &rows {
        assert_eq!(r["split"], "screen");
        assert!(r["question_greedy_overshoot"].is_i64());
    }
    let record: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("reeval.json")).unwrap()).unwrap();
    // the wrapper changed no episode: the manifest the run recorded for its
    // `screen` IS the sibling's own, byte for byte on the payload digests
    assert_eq!(
        record["split_manifests"]["screen"]["visible_sha256"],
        sibling_manifest["visible_sha256"]
    );
    assert_eq!(
        record["split_manifests"]["screen"]["episode_count"],
        sibling_manifest["episode_count"]
    );
    // and it is NOT the admitted screen it sits inside
    let admitted: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(splits.join("screen").join("manifest.public.json")).unwrap(),
    )
    .unwrap();
    assert_ne!(
        sibling_manifest["visible_sha256"], admitted["visible_sha256"],
        "the fixture would not test anything if the two splits were the same"
    );
    assert_ne!(
        record["split_manifests"]["screen"]["visible_sha256"],
        admitted["visible_sha256"]
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// **A SCREEN-ONLY sidecar is enough for a re-evaluation, and for nothing
/// else.** The teacher writes one `queries/` cache per destination, so the
/// zero-shot reads of a stage-0 checkpoint on a question screen have vectors
/// for the screen episodes and none for the training pool. A
/// `--reevaluate-checkpoint` read with no `--train-sample` walks no training
/// episode, so the pre-check is narrowed to the episodes it walks and records
/// the narrowing. Every other path keeps the old demand: a training run, and a
/// re-evaluation that samples the pool, are still refused, and a screen
/// episode missing from the cache is still refused on every path.
#[test]
fn a_screen_only_query_sidecar_is_enough_to_reevaluate_and_nothing_more() {
    if !config().exists() {
        eprintln!("skipped: the foundation checkout is not beside this one");
        return;
    }
    let Some(pin) = foundation_head() else {
        eprintln!("skipped: the foundation checkout has no git head");
        return;
    };
    let goldens =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../hf-io/tests/goldens/fixture-split");
    let root = tmp("reeval-screen-only");
    std::fs::create_dir_all(&root).unwrap();
    let cache = root.join("cache");
    write_node_cache(&cache);
    let splits = root.join("splits");
    let ids = write_stage1_splits(&goldens, &splits);
    assert_eq!(ids.len(), 18, "12 train + 6 screen golden episodes");
    // `write_stage1_splits` writes train first, then screen
    let screen_ids = &ids[12..];
    assert_eq!(screen_ids.len(), 6);
    let screen_only = root.join("queries-screen-only");
    write_query_cache(&screen_only, screen_ids);

    let stage0 = relational_config(&root.join("stage0"), None);
    let trained = root.join("trained");
    let o = run(&[
        "--config",
        stage0.to_str().unwrap(),
        "--output",
        trained.to_str().unwrap(),
        "--model-seed",
        "5",
        "--fixture",
        "--train-episodes",
        "4",
        "--screen-episodes",
        "2",
        "--updates",
        "1",
        "--save-checkpoint",
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));

    let stage1 = relational_config(
        &root.join("stage1"),
        Some(serde_json::json!({"query_source": "episode_query"})),
    );
    let common = |out: &PathBuf, query_dir: &std::path::Path| -> Vec<String> {
        vec![
            "--config".into(),
            stage1.to_str().unwrap().into(),
            "--output".into(),
            out.to_str().unwrap().into(),
            "--model-seed".into(),
            "5".into(),
            "--family".into(),
            "fixture".into(),
            "--splits-dir".into(),
            splits.to_str().unwrap().into(),
            "--embeddings-dir".into(),
            cache.to_str().unwrap().into(),
            "--query-embeddings-dir".into(),
            query_dir.to_str().unwrap().into(),
            "--expect-embedding-coverage".into(),
            "1.0".into(),
            "--screen-episodes".into(),
            "6".into(),
            "--preregistration-commit".into(),
            pin.clone(),
            "--foundation-root".into(),
            foundation().to_str().unwrap().into(),
            "--allow-stale-engine".into(),
        ]
    };
    let go = |argv: Vec<String>| -> std::process::Output {
        let borrowed: Vec<&str> = argv.iter().map(String::as_str).collect();
        run_env(&borrowed, CPU_ONLY)
    };

    // the re-evaluation walks the screen only, so the screen's own cache is enough
    let out = root.join("reeval");
    let mut argv = common(&out, &screen_only);
    argv.push("--reevaluate-checkpoint".into());
    argv.push(trained.join("checkpoint.json").to_str().unwrap().into());
    let o = go(argv.clone());
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let record: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("reeval.json")).unwrap()).unwrap();
    let precheck = &record["query_precheck"];
    assert_eq!(precheck["walks_training_pool"], false);
    assert_eq!(
        precheck["splits"],
        serde_json::json!(["screen"]),
        "the pool is out of the pre-check and nothing else is in it"
    );
    assert_eq!(precheck["episodes"], 6);
    assert_eq!(
        precheck["train_episodes_excluded"], 12,
        "the narrowing is named with the count it skipped, not left invisible"
    );
    assert_eq!(record["query_embedding_manifest"]["count"], 6);
    let rows = std::fs::read_to_string(out.join("evaluation_rows.jsonl")).unwrap();
    assert_eq!(rows.lines().count(), 6);

    // --train-sample walks the pool, so the same cache is refused there
    let sampled = root.join("sampled");
    let mut argv = common(&sampled, &screen_only);
    argv.push("--reevaluate-checkpoint".into());
    argv.push(trained.join("checkpoint.json").to_str().unwrap().into());
    argv.push("--train-sample".into());
    argv.push("2".into());
    let o = go(argv);
    assert_eq!(o.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&o.stderr).to_string();
    assert!(stderr.contains("no query vector"), "{stderr}");
    assert!(!sampled.join("reeval.json").exists());

    // and a TRAINING run is refused exactly as it was before
    let training = root.join("training");
    let mut argv = common(&training, &screen_only);
    argv.push("--updates".into());
    argv.push("1".into());
    let o = go(argv);
    assert_eq!(o.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&o.stderr).to_string();
    assert!(stderr.contains("no query vector"), "{stderr}");
    assert!(!training.join("probe.json").exists());

    // a SCREEN episode missing from the cache is still refused on the narrowed
    // path: the narrowing drops the pool, never a walked episode
    let gappy = root.join("queries-gappy");
    write_query_cache(&gappy, &screen_ids[..screen_ids.len() - 1]);
    let short = root.join("short");
    let mut argv = common(&short, &gappy);
    argv.push("--reevaluate-checkpoint".into());
    argv.push(trained.join("checkpoint.json").to_str().unwrap().into());
    let o = go(argv);
    assert_eq!(o.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&o.stderr).to_string();
    assert!(stderr.contains("no query vector"), "{stderr}");
    assert!(!short.join("reeval.json").exists());
    let _ = std::fs::remove_dir_all(&root);
}

/// **ENG-3: the VAULT is accepted under `episode_query`, and pre-checked.**
/// `--heldout-splits-dir` names the held-out FAMILY -- the vault, whose
/// questions the second source writes -- and is read as a registration veto at
/// stage 1 exactly as at stage 0. It is not a holdout path, and the
/// string-based holdout refusal is a separate guard that still stands.
///
/// The vault is evaluated with the run's OWN query cache, so a vault episode
/// the cache does not cover used to fail at INDEX BUILD inside `hf-walk`, an
/// hour into a run. It is now part of the loader's pre-check: covered, the run
/// evaluates the vault and says so in `query_precheck`; uncovered, it exits 2
/// before a row is written, with the loader's message and not the walker's.
#[test]
fn the_vault_is_accepted_under_episode_query_and_its_ids_are_pre_checked() {
    if !config().exists() {
        eprintln!("skipped: the foundation checkout is not beside this one");
        return;
    }
    let Some(pin) = foundation_head() else {
        eprintln!("skipped: the foundation checkout has no git head");
        return;
    };
    let goldens =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../hf-io/tests/goldens/fixture-split");
    let root = tmp("vault-precheck");
    std::fs::create_dir_all(&root).unwrap();
    let cache = root.join("cache");
    write_node_cache(&cache);
    let splits = root.join("splits");
    let ids = write_stage1_splits(&goldens, &splits);
    let screen_ids: Vec<String> = ids[12..].to_vec();

    // the vault: another family's screen, stage 1 (the second source wrote its
    // questions), its episode ids its own
    let vault = root.join("vault");
    let mut vault_ids = Vec::new();
    {
        let (episodes, artifacts) = hf_io::read_split(&splits.join("screen")).unwrap();
        let out: Vec<Result<hf_io::EpisodeOut, hf_core::HfError>> = episodes
            .iter()
            .map(|e| {
                let id = format!("vault-{}", e.episode_id);
                vault_ids.push(id.clone());
                Ok(hf_io::EpisodeOut {
                    episode_id: id,
                    visible: serde_json::to_value(&e.visible).unwrap(),
                    hidden: serde_json::to_value(&e.hidden).unwrap(),
                })
            })
            .collect();
        hf_io::write_split(
            "vault-fixture",
            hf_io::STAGES[1],
            "screen",
            &vault.join("screen"),
            out,
            &artifacts.graph,
            &artifacts.public["sampler"],
        )
        .unwrap();
    }
    let vault_cache = root.join("vault-cache");
    write_node_cache(&vault_cache);

    let stage0 = relational_config(&root.join("stage0"), None);
    let trained = root.join("trained");
    let o = run(&[
        "--config",
        stage0.to_str().unwrap(),
        "--output",
        trained.to_str().unwrap(),
        "--model-seed",
        "5",
        "--fixture",
        "--train-episodes",
        "4",
        "--screen-episodes",
        "2",
        "--updates",
        "1",
        "--save-checkpoint",
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let stage1 = relational_config(
        &root.join("stage1"),
        Some(serde_json::json!({"query_source": "episode_query"})),
    );
    let go = |out: &PathBuf, query_dir: &std::path::Path| -> std::process::Output {
        run_env(
            &[
                "--config",
                stage1.to_str().unwrap(),
                "--output",
                out.to_str().unwrap(),
                "--model-seed",
                "5",
                "--family",
                "fixture",
                "--splits-dir",
                splits.to_str().unwrap(),
                "--embeddings-dir",
                cache.to_str().unwrap(),
                "--heldout-family",
                "vault-fixture",
                "--heldout-splits-dir",
                vault.to_str().unwrap(),
                "--heldout-embeddings-dir",
                vault_cache.to_str().unwrap(),
                "--query-embeddings-dir",
                query_dir.to_str().unwrap(),
                "--screen-episodes",
                "6",
                "--reevaluate-checkpoint",
                trained.join("checkpoint.json").to_str().unwrap(),
                "--preregistration-commit",
                &pin,
                "--foundation-root",
                foundation().to_str().unwrap(),
                "--allow-stale-engine",
            ],
            CPU_ONLY,
        )
    };

    // the screens are covered and the vault is not: refused by the LOADER,
    // which is the point -- the vault is accepted, its ids are checked
    let screens_only = root.join("queries-screens");
    write_query_cache(&screens_only, &screen_ids);
    let refused = root.join("refused");
    let o = go(&refused, &screens_only);
    assert_eq!(o.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&o.stderr).to_string();
    assert!(
        stderr.contains("of the run's episodes have no query vector"),
        "the loader's pre-check, not hf-walk's index build: {stderr}"
    );
    assert!(
        !stderr.contains("the query cache holds no vector for episode"),
        "the walker got to build an index: {stderr}"
    );
    assert!(
        !stderr.contains("held-out"),
        "the vault flag itself is accepted under episode_query: {stderr}"
    );
    assert!(!refused.join("evaluation_rows.jsonl").exists());
    assert!(!refused.join("reeval.json").exists());

    // with the vault covered it runs, the vault is evaluated, and the
    // pre-check names what it covered
    let mut all: Vec<String> = screen_ids.clone();
    all.extend(vault_ids.iter().cloned());
    let whole = root.join("queries-all");
    write_query_cache(&whole, &all);
    let out = root.join("reeval");
    let o = go(&out, &whole);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let record: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("reeval.json")).unwrap()).unwrap();
    assert_eq!(
        record["query_precheck"]["splits"],
        serde_json::json!(["screen", "heldout"])
    );
    assert_eq!(
        record["query_precheck"]["episodes"],
        (screen_ids.len() + vault_ids.len()) as u64
    );
    let splits_read: Vec<String> = std::fs::read_to_string(out.join("evaluation_rows.jsonl"))
        .unwrap()
        .lines()
        .map(|l| {
            serde_json::from_str::<serde_json::Value>(l).unwrap()["split"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
    assert_eq!(
        splits_read.iter().filter(|s| *s == "heldout").count(),
        vault_ids.len(),
        "the vault was walked, which is what makes its coverage load-bearing"
    );

    // and the string-based holdout refusal is untouched on this same path
    let named = root.join("a-heldout-vault");
    std::fs::create_dir_all(named.join("screen")).unwrap();
    let o = run_env(
        &[
            "--config",
            stage1.to_str().unwrap(),
            "--output",
            root.join("never").to_str().unwrap(),
            "--model-seed",
            "5",
            "--family",
            "fixture",
            "--splits-dir",
            splits.to_str().unwrap(),
            "--embeddings-dir",
            cache.to_str().unwrap(),
            "--heldout-family",
            "vault-fixture",
            "--heldout-splits-dir",
            named.to_str().unwrap(),
            "--heldout-embeddings-dir",
            vault_cache.to_str().unwrap(),
            "--query-embeddings-dir",
            whole.to_str().unwrap(),
            "--screen-episodes",
            "6",
            "--reevaluate-checkpoint",
            trained.join("checkpoint.json").to_str().unwrap(),
            "--preregistration-commit",
            &pin,
            "--foundation-root",
            foundation().to_str().unwrap(),
            "--allow-stale-engine",
        ],
        CPU_ONLY,
    );
    assert_eq!(o.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&o.stderr).to_string();
    assert!(
        stderr.contains("holdout material is never touched"),
        "{stderr}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// One split directory — `train` from `source`, `screen` from the golden
/// screen — with the sampler block the manifest will record. `rung` rewrites
/// `subgraph_size` and `target_distance`, which is how the test makes a pool
/// that a later rung's config disagrees with.
fn write_pool(
    goldens: &std::path::Path,
    destination: &std::path::Path,
    source: &str,
    rung: Option<(u64, u64)>,
) {
    for (split, from) in [("train", source), ("screen", "screen")] {
        let (episodes, artifacts) = hf_io::read_split(&goldens.join(from)).unwrap();
        let mut sampler = artifacts.public["sampler"].clone();
        if let Some((size, distance)) = rung {
            let object = sampler.as_object_mut().unwrap();
            object.insert("subgraph_size".into(), size.into());
            object.insert("target_distance".into(), distance.into());
        }
        let out: Vec<Result<hf_io::EpisodeOut, hf_core::HfError>> = episodes
            .iter()
            .map(|e| {
                Ok(hf_io::EpisodeOut {
                    episode_id: e.episode_id.clone(),
                    visible: serde_json::to_value(&e.visible).unwrap(),
                    hidden: serde_json::to_value(&e.hidden).unwrap(),
                })
            })
            .collect();
        hf_io::write_split(
            "fixture",
            hf_io::STAGES[0],
            split,
            &destination.join(split),
            out,
            &artifacts.graph,
            &sampler,
        )
        .unwrap();
    }
}

/// **P1, the cumulative pool.** `--splits-dir` repeated concatenates the
/// pools in the order given; the draw stays ONE draw over the concatenation,
/// so each pool's share is its size and `train_draws` replays exactly as it
/// does for one pool — checked here through the engine's own replay gate,
/// which goes band H when the replayed draws do not reproduce the probe's
/// histogram. A single `--splits-dir` is unchanged, sampler agreement and
/// all; only a cumulative pool stops being checked on `subgraph_size` and
/// `target_distance`, which `probe.json` then records per pool.
#[test]
fn a_cumulative_pool_concatenates_the_splits_dirs_and_keeps_the_draws_replayable() {
    if !config().exists() {
        eprintln!("skipped: the foundation checkout is not beside this one");
        return;
    }
    let Some(pin) = foundation_head() else {
        eprintln!("skipped: the foundation checkout has no git head");
        return;
    };
    let goldens =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../hf-io/tests/goldens/fixture-split");
    let root = tmp("cumulative");
    std::fs::create_dir_all(&root).unwrap();
    let cache = root.join("cache");
    write_node_cache(&cache);
    // the run's own rung, agreeing with the config, and an earlier one that
    // does not: a smaller ball at a shorter distance
    let rung3 = root.join("rung3");
    write_pool(&goldens, &rung3, "train", None);
    let rung2 = root.join("rung2");
    write_pool(&goldens, &rung2, "train-greedy", Some((20, 2)));
    let cfg = relational_config(&root.join("config"), None);

    let train = |out: &PathBuf, pools: &[&PathBuf], extra: &[&str]| -> std::process::Output {
        let mut args: Vec<String> = vec![
            "--config".into(),
            cfg.to_string_lossy().into(),
            "--output".into(),
            out.to_string_lossy().into(),
            "--model-seed".into(),
            "5".into(),
            "--family".into(),
            "fixture".into(),
            "--embeddings-dir".into(),
            cache.to_string_lossy().into(),
            "--screen-episodes".into(),
            "6".into(),
            "--updates".into(),
            "3".into(),
            "--eval-every".into(),
            "3".into(),
            "--preregistration-commit".into(),
            pin.clone(),
            "--foundation-root".into(),
            foundation().to_string_lossy().into(),
            "--allow-stale-engine".into(),
        ];
        for p in pools {
            args.push("--splits-dir".into());
            args.push(p.to_string_lossy().into());
        }
        args.extend(extra.iter().map(|s| (*s).to_string()));
        run_env(
            &args.iter().map(String::as_str).collect::<Vec<_>>(),
            CPU_ONLY,
        )
    };
    let probe_of = |out: &PathBuf| -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(out.join("probe.json")).unwrap()).unwrap()
    };

    // one pool: the old read, and a one-element `train_pools`
    let single = root.join("single");
    let o = train(&single, &[&rung3], &[]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let probe = probe_of(&single);
    assert_eq!(probe["train_episodes"], 12);
    let pools = probe["train_pools"].as_array().unwrap();
    assert_eq!(pools.len(), 1);
    assert_eq!(pools[0]["episodes"], 12);
    assert_eq!(pools[0]["episodes_used"], 12);
    assert_eq!(pools[0]["share"], 1.0);
    assert_eq!(pools[0]["subgraph_size"], 64);
    assert!(probe["split_manifests"].get("train2").is_none());

    // two pools: the concatenation, in the order given
    let both = root.join("both");
    let o = train(&both, &[&rung3, &rung2], &["--save-checkpoint"]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let probe = probe_of(&both);
    assert_eq!(probe["train_episodes"], 24, "12 + 12, concatenated");
    let pools = probe["train_pools"].as_array().unwrap();
    assert_eq!(pools.len(), 2);
    assert_eq!(pools[0]["splits_dir"], rung3.to_string_lossy().to_string());
    assert_eq!(pools[1]["splits_dir"], rung2.to_string_lossy().to_string());
    // the two keys the cumulative pool is no longer checked on are recorded
    assert_eq!(pools[0]["subgraph_size"], 64);
    assert_eq!(pools[0]["target_distance"], 3);
    assert_eq!(pools[1]["subgraph_size"], 20);
    assert_eq!(pools[1]["target_distance"], 2);
    assert_eq!(pools[0]["share"], 0.5);
    assert_eq!(pools[1]["share"], 0.5);
    assert!(probe["split_manifests"]["train2"]["sampler"]["subgraph_size"] == 20);
    // the screen is the FIRST pool's, and only the first pool's
    assert_eq!(probe["screen_episodes"], 6);
    assert_eq!(probe["train_draws"]["draws"], 6, "3 updates x microbatch 2");

    // the draw is one draw over the concatenation, so the engine's own replay
    // reproduces it: blank the checkpoint's recorded draws (as
    // `tools/export_checkpoint.py` leaves a Python checkpoint) and make the
    // re-evaluation replay them against this probe. A two-level draw could not
    // pass this gate.
    let meta = both.join("checkpoint.json");
    let mut saved: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&meta).unwrap()).unwrap();
    assert!(
        !saved["train_draws"].as_object().unwrap().is_empty(),
        "the checkpoint recorded the draws before the test blanked them"
    );
    saved["train_draws"] = serde_json::json!({});
    std::fs::write(&meta, serde_json::to_string_pretty(&saved).unwrap()).unwrap();
    let replayed = root.join("replayed");
    let o = train(
        &replayed,
        &[&rung3, &rung2],
        &[
            "--reevaluate-checkpoint",
            meta.to_str().unwrap(),
            "--train-sample",
            "8",
            "--train-draws-probe",
            both.join("probe.json").to_str().unwrap(),
        ],
    );
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let record: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(replayed.join("reeval.json")).unwrap())
            .unwrap();
    assert_eq!(record["train_sample"]["draw_counts_source"], "replay");
    assert_eq!(
        record["train_sample"]["draw_counts_replay"]["histogram_matches_probe"],
        true
    );
    assert_eq!(record["train_sample"]["pool_size"], 24);

    // ONE --splits-dir is checked exactly as it always was
    let refused = root.join("refused");
    let o = train(&refused, &[&rung2], &[]);
    assert_eq!(o.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&o.stderr).to_string();
    assert!(stderr.contains("band H"), "{stderr}");
    assert!(stderr.contains("subgraph_size"), "{stderr}");

    // and overlapping pools are refused rather than folded together: an
    // episode drawn twice would be one `train_draws` key with both counts
    let twice = root.join("twice");
    let o = train(&twice, &[&rung3, &rung3], &[]);
    assert_eq!(o.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&o.stderr).to_string();
    assert!(stderr.contains("the pools overlap"), "{stderr}");
    let _ = std::fs::remove_dir_all(&root);
}
