//! The runner on the fixture world: a training run writes the artifacts with
//! their keys; a re-evaluation with a training sample and a candidate dump
//! satisfies the readers' invariants; the refusals (holdout path, missing
//! pin, doctored probe, flags without a re-evaluation) exit 2;
//! `--print-capacity` builds the config's model on the CPU and prints what it
//! would train, reading no data; `training.decay_exempt` reaches the probe and
//! the checkpoints as the patterns and the names they resolved to.

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
    std::process::Command::new(env!("CARGO_BIN_EXE_hf-stage0"))
        .args(args)
        .output()
        .unwrap()
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
