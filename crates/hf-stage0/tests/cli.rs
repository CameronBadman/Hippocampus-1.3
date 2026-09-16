//! The runner on the fixture world: a training run writes the artifacts with
//! their keys; a re-evaluation with a training sample and a candidate dump
//! satisfies the readers' invariants; the refusals (holdout path, missing
//! pin, doctored probe, flags without a re-evaluation) exit 2;
//! `--print-capacity` builds the config's model on the CPU and prints what it
//! would train, reading no data.

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
