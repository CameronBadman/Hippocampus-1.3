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
