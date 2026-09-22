//! The writer on the fixture world written to disk as an adapter would:
//! the splits it writes are the Python-written goldens (same ids, same
//! records), the sidecars are shaped like `real_walk_write_splits.py`'s, a
//! larger screen passes the prefix check against the smaller one, and a
//! changed distance fails it.

use std::path::PathBuf;

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("hf-splits-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The fixture world as an adapter directory (edges.tsv, text.tsv, graph.manifest.json)
/// plus its embeddings as a v5 cache.
fn fixture_dirs(root: &std::path::Path) -> (PathBuf, PathBuf) {
    let (graph, embeddings) = hf_episodes::fixture::fixture_world(5, 400, 1600, 8);
    let gdir = root.join("graph");
    std::fs::create_dir_all(&gdir).unwrap();
    let mut edges = String::new();
    for h in graph.nodes() {
        for e in graph.out(h) {
            // two columns: the fixture world builds its edges with relation None, and a
            // three-column file with an empty relation would load as Some("") (the vault quirk)
            edges.push_str(&format!("{}\t{}\n", graph.name(h), graph.name(e.tail)));
        }
    }
    std::fs::write(gdir.join("edges.tsv"), edges).unwrap();
    let mut text = String::new();
    for n in graph.nodes() {
        text.push_str(&format!(
            "{}\t{}\n",
            graph.name(n),
            graph.text(n).unwrap_or("")
        ));
    }
    std::fs::write(gdir.join("text.tsv"), text).unwrap();
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../hf-io/tests/goldens/fixture-split/expected.json"),
        )
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        gdir.join("graph.manifest.json"),
        hf_core::dump_pretty(&manifest["graph_manifest"]),
    )
    .unwrap();
    let cache = root.join("cache");
    std::fs::create_dir_all(&cache).unwrap();
    let mut f = std::fs::File::create(cache.join("vectors.jsonl")).unwrap();
    let mut names: Vec<&String> = embeddings.keys().collect();
    names.sort();
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
    hf_embed::write_manifest(&cache, &m).unwrap();
    (gdir, cache)
}

fn run(args: &[&str]) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_hf-splits"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn writes_the_python_goldens_and_checks_prefixes() {
    let root = tmp("write");
    let (gdir, _cache) = fixture_dirs(&root);
    let dest = root.join("pool");
    let out = run(&[
        "write",
        "--family",
        "fixture",
        "--graph-dir",
        gdir.to_str().unwrap(),
        "--subgraph-size",
        "64",
        "--target-distance",
        "3",
        "--removal-level",
        "2",
        "--train",
        "12",
        "--screen",
        "6",
        "--chunk",
        "5",
        "--destination",
        dest.to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let goldens =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../hf-io/tests/goldens/fixture-split");
    for (split, golden) in [("train", "train"), ("screen", "screen")] {
        let (ours, _) = hf_io::read_split(&dest.join(split)).unwrap();
        let (theirs, _) = hf_io::read_split(&goldens.join(golden)).unwrap();
        assert_eq!(ours.len(), theirs.len());
        for (a, b) in ours.iter().zip(&theirs) {
            assert_eq!(a.episode_id, b.episode_id);
            assert_eq!(a.visible, b.visible, "{split} visible");
            assert_eq!(a.hidden, b.hidden, "{split} hidden");
        }
        // the containers are byte-identical to Python's
        assert_eq!(
            std::fs::read(dest.join(split).join("visible.jsonl.gz")).unwrap(),
            std::fs::read(goldens.join(golden).join("visible.jsonl.gz")).unwrap()
        );
        assert_eq!(
            std::fs::read(dest.join(split).join("hidden.jsonl.gz")).unwrap(),
            std::fs::read(goldens.join(golden).join("hidden.jsonl.gz")).unwrap()
        );
        assert_eq!(
            std::fs::read_to_string(dest.join(split).join("nodes.txt")).unwrap(),
            std::fs::read_to_string(goldens.join(golden).join("nodes.txt")).unwrap()
        );
        let sampling: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dest.join(split).join("sampling.json")).unwrap(),
        )
        .unwrap();
        let want: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(goldens.join(golden).join("sampling.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(sampling, want, "{split} sampling.json");
        let texts = hf_io::read_texts(&dest.join(split)).unwrap();
        let want = hf_io::read_texts(&goldens.join(golden)).unwrap();
        assert_eq!(texts, want, "{split} texts.jsonl");
    }
    // a larger screen reproduces the smaller as its prefix
    let bigger = root.join("pool-bigger");
    let out = run(&[
        "write",
        "--family",
        "fixture",
        "--graph-dir",
        gdir.to_str().unwrap(),
        "--subgraph-size",
        "64",
        "--target-distance",
        "3",
        "--removal-level",
        "2",
        "--train",
        "0",
        "--screen",
        "14",
        "--destination",
        bigger.to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !bigger.join("train").exists(),
        "--train 0 writes no train split"
    );
    let out = run(&[
        "prefix-check",
        dest.join("screen").to_str().unwrap(),
        bigger.join("screen").to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("OK: the first 6 records"));
    // a changed distance fails the check
    let other = root.join("pool-other");
    let out = run(&[
        "write",
        "--family",
        "fixture",
        "--graph-dir",
        gdir.to_str().unwrap(),
        "--subgraph-size",
        "64",
        "--target-distance",
        "2",
        "--removal-level",
        "2",
        "--train",
        "0",
        "--screen",
        "14",
        "--destination",
        other.to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = run(&[
        "prefix-check",
        dest.join("screen").to_str().unwrap(),
        other.join("screen").to_str().unwrap(),
    ]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stdout).contains("MISMATCH"));
    // an existing destination is refused, exit 2
    let out = run(&[
        "write",
        "--family",
        "fixture",
        "--graph-dir",
        gdir.to_str().unwrap(),
        "--subgraph-size",
        "64",
        "--target-distance",
        "3",
        "--removal-level",
        "2",
        "--train",
        "0",
        "--screen",
        "6",
        "--destination",
        dest.to_str().unwrap(),
    ]);
    assert_eq!(out.status.code(), Some(2));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn greedy_path_needs_a_cache_and_writes_the_greedy_golden() {
    let root = tmp("greedy");
    let (gdir, cache) = fixture_dirs(&root);
    let out = run(&[
        "write",
        "--family",
        "fixture",
        "--graph-dir",
        gdir.to_str().unwrap(),
        "--subgraph-size",
        "64",
        "--target-distance",
        "3",
        "--removal-level",
        "2",
        "--removal-rule",
        "greedy-path",
        "--train",
        "12",
        "--screen",
        "0",
        "--destination",
        root.join("no-cache").to_str().unwrap(),
    ]);
    assert_eq!(
        out.status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let dest = root.join("pool");
    let out = run(&[
        "write",
        "--family",
        "fixture",
        "--graph-dir",
        gdir.to_str().unwrap(),
        "--subgraph-size",
        "64",
        "--target-distance",
        "3",
        "--removal-level",
        "2",
        "--removal-rule",
        "greedy-path",
        "--embeddings",
        cache.to_str().unwrap(),
        "--greedy-share",
        "0.5",
        "--train",
        "12",
        "--screen",
        "6",
        "--destination",
        dest.to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let goldens =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../hf-io/tests/goldens/fixture-split");
    let (ours, _) = hf_io::read_split(&dest.join("train")).unwrap();
    let (theirs, _) = hf_io::read_split(&goldens.join("train-greedy")).unwrap();
    assert_eq!(ours.len(), theirs.len());
    for (a, b) in ours.iter().zip(&theirs) {
        assert_eq!(a.episode_id, b.episode_id);
        assert_eq!(a.hidden, b.hidden, "greedy-path hidden");
    }
    // the screen keeps share 1.0 under the greedy-path rule
    let (screen, art) = hf_io::read_split(&dest.join("screen")).unwrap();
    assert_eq!(art.public["sampler"]["greedy_share"], 1.0);
    assert!(screen
        .iter()
        .all(|e| e.hidden.removal_recipe() == "greedy-path"));
    let _ = std::fs::remove_dir_all(&root);
}

/// `--targets 2`: the writer's k = 2 split — both targets shown, the 6.0.0
/// schema on the records and the manifests, the k = 2 drop reasons in
/// `sampling.json`, a larger screen reproducing the smaller as its prefix, and
/// (when the foundation is beside this checkout) the disclosure verified:
/// Python's v5 artifact validator accepts the directory and its v5 record
/// reader refuses the records, so a k = 2 split needs a v6 reader.
#[test]
fn targets_two_writes_a_v6_split_and_keeps_the_prefix_property() {
    let root = tmp("targets2");
    let (gdir, _cache) = fixture_dirs(&root);
    let write = |screen: usize, dest: &std::path::Path| {
        run(&[
            "write",
            "--family",
            "fixture",
            "--graph-dir",
            gdir.to_str().unwrap(),
            "--subgraph-size",
            "64",
            "--target-distance",
            "3",
            "--removal-level",
            "2",
            "--targets",
            "2",
            "--train",
            "0",
            "--screen",
            &screen.to_string(),
            "--chunk",
            "3",
            "--destination",
            dest.to_str().unwrap(),
        ])
    };
    let small = root.join("pool-4");
    let out = write(4, &small);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let (episodes, artifacts) = hf_io::read_split(&small.join("screen")).unwrap();
    assert_eq!(episodes.len(), 4);
    assert_eq!(artifacts.public["schema_version"], "6.0.0");
    assert_eq!(artifacts.public["sampler"]["targets"], 2);
    for e in &episodes {
        assert!(e.episode_id.contains("-t2"));
        assert_eq!(e.visible.schema_version, "6.0.0");
        let shown = e.visible.target_nodes.as_ref().expect("both targets shown");
        assert_eq!(shown, &e.hidden.target_set);
        assert_eq!(e.visible.target_node.as_ref(), Some(&shown[0]));
        assert_eq!(shown.len(), 2);
        assert!(e.hidden.distance_to_targets.is_some());
    }
    let sampling: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(small.join("screen").join("sampling.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(sampling["kept"], 4);
    let drops = sampling["drops"].as_object().unwrap();
    assert!(
        drops.keys().all(|k| [
            "subgraph_too_small",
            "no_target_at_distance_in_split",
            "no_path_within_bound",
            "path_set_over_cap",
            "greedy_route_missing",
            "survivors_mismatch",
            "no_second_target_at_distance",
            "targets_interdependent",
            "removal_left_no_path",
            "survivor_not_recovered",
        ]
        .contains(&k.as_str())),
        "sampling.json carries an unknown drop reason: {drops:?}"
    );
    // a larger screen reproduces the smaller as its prefix
    let bigger = root.join("pool-8");
    let out = write(8, &bigger);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = run(&[
        "prefix-check",
        small.join("screen").to_str().unwrap(),
        bigger.join("screen").to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("OK: the first 4 records"));
    // a k = 1 pool is not a prefix of a k = 2 pool: the sampler blocks differ
    let single = root.join("pool-k1");
    let out = run(&[
        "write",
        "--family",
        "fixture",
        "--graph-dir",
        gdir.to_str().unwrap(),
        "--subgraph-size",
        "64",
        "--target-distance",
        "3",
        "--removal-level",
        "2",
        "--train",
        "0",
        "--screen",
        "4",
        "--destination",
        single.to_str().unwrap(),
    ]);
    assert!(out.status.success());
    let out = run(&[
        "prefix-check",
        single.join("screen").to_str().unwrap(),
        bigger.join("screen").to_str().unwrap(),
    ]);
    assert_eq!(out.status.code(), Some(2));
    // the disclosure, verified rather than asserted
    let foundation = std::env::var("HF_FOUNDATION")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../hippocampus-foundation")
        });
    if foundation.join("pyproject.toml").exists() {
        let script = format!(
            "from pathlib import Path\nfrom hippocampus_foundation.read_run.io_v5 import validate_real_split_artifacts_v5, read_real_split_v5\nfrom hippocampus_foundation.read_run.errors import IntegrityGateError\nroot = Path({:?})\nvalidate_real_split_artifacts_v5(root)\nprint('artifacts ok')\ntry:\n    for _ in read_real_split_v5(root):\n        pass\nexcept IntegrityGateError as e:\n    print('records refused:', e)\nelse:\n    print('records accepted')\n",
            small.join("screen").to_string_lossy()
        );
        let output = std::process::Command::new("uv")
            .args(["run", "python", "-c", &script])
            .env("PYTHONPATH", "src")
            .current_dir(&foundation)
            .output()
            .expect("uv run python");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "python failed on the k = 2 split: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(stdout.contains("artifacts ok"), "{stdout}");
        assert!(
            stdout.contains("records refused"),
            "the v5 record reader was expected to refuse a 6.0.0 record: {stdout}"
        );
    } else {
        eprintln!("skipped: foundation checkout not present for the Python disclosure check");
    }
    let _ = std::fs::remove_dir_all(&root);
}

// --------------------------------------------------------------------------
// `hf-splits baselines`: the model-free k baselines off one split directory

fn write_fixture_split(
    root: &std::path::Path,
    gdir: &std::path::Path,
    dest: &std::path::Path,
    targets: &str,
    screen: usize,
) {
    let _ = root;
    let out = run(&[
        "write",
        "--family",
        "fixture",
        "--graph-dir",
        gdir.to_str().unwrap(),
        "--subgraph-size",
        "64",
        "--target-distance",
        "3",
        "--removal-level",
        "2",
        "--targets",
        targets,
        "--train",
        "0",
        "--screen",
        &screen.to_string(),
        "--chunk",
        "3",
        "--destination",
        dest.to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn read_rows(path: &std::path::Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

/// One JSON line per episode, with the fields `K_TARGETS_DESIGN.md` §9 item 5
/// is read from: the examination order, the per-target registrations, the
/// recall at `B_fix`, the frozen variant beside it, and `o_k` inside its own
/// sandwich. No model is loaded and none is named: the manifest records
/// `model: null` and `training_authorized: false`.
#[test]
fn baselines_read_the_k_traces_off_a_k2_split() {
    let root = tmp("baselines-k2");
    let (gdir, cache) = fixture_dirs(&root);
    let dest = root.join("pool");
    write_fixture_split(&root, &gdir, &dest, "2", 6);
    let split = dest.join("screen");
    let rows_path = root.join("rows").join("k_rows.jsonl");
    let out = run(&[
        "baselines",
        "--split-dir",
        split.to_str().unwrap(),
        "--embeddings",
        cache.to_str().unwrap(),
        "--output",
        rows_path.to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let (episodes, _) = hf_io::read_split(&split).unwrap();
    let rows = read_rows(&rows_path);
    assert_eq!(rows.len(), episodes.len(), "one row per episode");
    let embeddings = hf_embed::EmbeddingMatrix::load(&cache).unwrap();
    for (row, e) in rows.iter().zip(&episodes) {
        assert_eq!(row["record_kind"], "k_baseline_row");
        assert_eq!(row["episode_id"], e.episode_id);
        assert_eq!(row["start_node"], e.visible.start_node);
        let targets = row["targets"].as_array().unwrap();
        assert_eq!(targets.len(), 2, "a k = 2 row names both targets");
        // B_fix is the RUNG's n / 2, not the realised ball's
        assert_eq!(row["subgraph_size"], 64);
        assert_eq!(row["b_fix"], 32);
        let registered = row["k_greedy_registered_at"].as_array().unwrap();
        assert_eq!(registered.len(), 2, "one registration slot per target");
        let examined = row["k_greedy_examined"].as_array().unwrap();
        assert_eq!(
            examined[0], e.visible.start_node,
            "the start is expanded first"
        );
        assert_eq!(
            examined.len() as u64,
            row["k_greedy_expansions"].as_u64().unwrap()
        );
        // the recall the strata are formed on, recomputed from the row itself
        let b_fix = row["b_fix"].as_u64().unwrap();
        let hit = registered
            .iter()
            .filter(|r| r.as_u64().is_some_and(|at| at <= b_fix))
            .count();
        assert_eq!(
            row["k_greedy_recall_at_budget"].as_f64().unwrap(),
            hit as f64 / 2.0,
            "recall over k is the share of targets registered by B_fix"
        );
        // the k-oracle inside its own admissible sandwich
        let o_k = row["k_oracle_expansions"].as_u64().unwrap();
        assert!(o_k >= row["k_oracle_lower_bound"].as_u64().unwrap());
        assert!(o_k <= row["k_oracle_upper_bound"].as_u64().unwrap());
        assert!(
            row["k_oracle_pruned_nodes"].as_u64().unwrap() >= 1,
            "|V'| counts the start"
        );
        assert!(row["k_oracle_exact"].is_boolean());
        // the frozen variant is a walk of its own, keyed once at push
        let g = hf_policies::EpisodeGraph::from_episode_k(e);
        let frozen = hf_policies::k_greedy_frozen_trace(&g, &embeddings);
        assert_eq!(
            row["k_greedy_frozen_expansions"].as_u64().unwrap(),
            frozen.expansions as u64
        );
    }
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("rows").join("k_rows.manifest.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(manifest["record_kind"], "k_baseline_manifest");
    assert_eq!(manifest["training_authorized"], false);
    assert_eq!(manifest["model"], serde_json::Value::Null);
    assert_eq!(manifest["checkpoint"], serde_json::Value::Null);
    assert_eq!(manifest["b_fix"], 32);
    assert_eq!(manifest["episodes_read"], rows.len());
    assert_eq!(manifest["targets_per_episode"], serde_json::json!([2]));
    let _ = std::fs::remove_dir_all(&root);
}

/// The golden the reader leans on: at k = 1 the rows the CLI writes are v1's
/// `similarity_greedy` walk — the SAME examination order, node for node, and
/// the same expansion count. `crates/hf-policies/tests/k_targets.rs` proves the
/// two traces equal field for field on the fixture episodes; this one proves
/// the CLI writes that trace and not another.
#[test]
fn at_one_target_the_rows_are_v1_similarity_greedys_own_walk() {
    let root = tmp("baselines-k1");
    let (gdir, cache) = fixture_dirs(&root);
    let dest = root.join("pool");
    write_fixture_split(&root, &gdir, &dest, "1", 6);
    let split = dest.join("screen");
    let rows_path = root.join("k1.jsonl");
    let out = run(&[
        "baselines",
        "--split-dir",
        split.to_str().unwrap(),
        "--embeddings",
        cache.to_str().unwrap(),
        "--output",
        rows_path.to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let (episodes, _) = hf_io::read_split(&split).unwrap();
    let embeddings = hf_embed::EmbeddingMatrix::load(&cache).unwrap();
    let rows = read_rows(&rows_path);
    assert_eq!(rows.len(), episodes.len());
    let mut checked = 0;
    for (row, e) in rows.iter().zip(&episodes) {
        let v1 = hf_policies::EpisodeGraph::from_episode(e);
        let trace = hf_policies::similarity_greedy_trace(&v1, &embeddings, None);
        let examined: Vec<String> = row["k_greedy_examined"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(examined, trace.examined, "{}", e.episode_id);
        assert_eq!(
            row["k_greedy_expansions"].as_u64().unwrap(),
            trace.expansions as u64
        );
        assert_eq!(row["targets"].as_array().unwrap().len(), 1);
        // and the frozen variant is the same walk again at k = 1: there is no
        // registration before the end, so there is nothing to re-key
        assert_eq!(row["k_greedy_frozen_examined"], row["k_greedy_examined"]);
        checked += 1;
    }
    assert!(checked > 0, "the fixture split was empty");
    let _ = std::fs::remove_dir_all(&root);
}

/// The standing refusal, on every path the subcommand takes.
#[test]
fn baselines_refuses_a_holdout_path() {
    let root = tmp("baselines-holdout");
    let out = run(&[
        "baselines",
        "--split-dir",
        root.join("holdout").join("train").to_str().unwrap(),
        "--embeddings",
        root.join("cache").to_str().unwrap(),
        "--output",
        root.join("rows.jsonl").to_str().unwrap(),
    ]);
    assert_eq!(out.status.code(), Some(2), "a holdout path exits 2");
    let out = run(&[
        "baselines",
        "--split-dir",
        root.join("pool").join("train").to_str().unwrap(),
        "--embeddings",
        root.join("cache").to_str().unwrap(),
        "--output",
        root.join("heldout-rows.jsonl").to_str().unwrap(),
    ]);
    assert_eq!(out.status.code(), Some(2), "a heldout output exits 2");
    let _ = std::fs::remove_dir_all(&root);
}

/// Rewrite a stage-0 split directory as the stage-1 one a teacher would write:
/// the visible payload drops `target_node` and gains `query`, and the hidden
/// payload — the targets, the paths, the labels — is carried across unchanged.
/// Returns the new split directory and the episode ids in file order.
fn as_stage1_split(source: &std::path::Path, destination: &std::path::Path) -> Vec<String> {
    let (episodes, artifacts) = hf_io::read_split(source).unwrap();
    let mut ids = Vec::new();
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
        "screen",
        destination,
        out,
        &artifacts.graph,
        &artifacts.public["sampler"],
    )
    .unwrap();
    ids
}

/// A question-vector sidecar: an ordinary v5 cache whose ids are EPISODE ids,
/// written by the node cache's encoder unless `digest` says otherwise.
fn write_query_cache(dir: &std::path::Path, ids: &[String], digest: &str, dimension: u32) {
    std::fs::create_dir_all(dir).unwrap();
    let mut f = std::fs::File::create(dir.join("vectors.jsonl")).unwrap();
    for (i, id) in ids.iter().enumerate() {
        let v: Vec<f64> = (0..dimension)
            .map(|j| ((i as f64 + 1.0) * 0.37 + j as f64 * 0.11).sin())
            .collect();
        hf_embed::append_vector(&mut f, id, &v).unwrap();
    }
    let m = hf_embed::Manifest {
        record_kind: hf_embed::MANIFEST_KIND.into(),
        model: "fixture".into(),
        model_digest: digest.into(),
        base_url: "none".into(),
        dimension,
        count: ids.len() as u64,
        text_char_limit: 6000,
        text_sha256: None,
        truncated: Default::default(),
        training_authorized: false,
        extra: Default::default(),
    };
    hf_embed::write_manifest(dir, &m).unwrap();
}

/// `--query-embeddings-dir`: greedy walks on the QUESTION and every row gains
/// `question_greedy_overshoot` — greedy's expansions minus the single-target
/// oracle's, the sampler's own definition of `greedy_overshoot` with the
/// question in the target's place. The row's value is recomputed here from the
/// policies directly, so the CLI is checked against the definition and not
/// against itself.
#[test]
fn baselines_on_the_question_write_the_overshoot_the_strata_are_re_derived_from() {
    let root = tmp("baselines-question");
    let (gdir, cache) = fixture_dirs(&root);
    let dest = root.join("pool");
    write_fixture_split(&root, &gdir, &dest, "1", 6);
    let stage1 = root.join("stage1");
    let ids = as_stage1_split(&dest.join("screen"), &stage1);
    let qdir = root.join("queries");
    write_query_cache(&qdir, &ids, "fixture", 8);
    let rows_path = root.join("question.jsonl");
    let out = run(&[
        "baselines",
        "--split-dir",
        stage1.to_str().unwrap(),
        "--embeddings",
        cache.to_str().unwrap(),
        "--query-embeddings-dir",
        qdir.to_str().unwrap(),
        "--expect-embedding-coverage",
        "1.0",
        "--output",
        rows_path.to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let (episodes, _) = hf_io::read_split(&stage1).unwrap();
    let embeddings = hf_embed::EmbeddingMatrix::load(&cache).unwrap();
    let queries = hf_embed::EmbeddingMatrix::load(&qdir).unwrap();
    let rows = read_rows(&rows_path);
    assert_eq!(rows.len(), episodes.len());
    let mut moved = 0;
    for (row, e) in rows.iter().zip(&episodes) {
        let g = hf_policies::EpisodeGraph::from_episode(e);
        let q: Vec<f64> = queries
            .get(&e.episode_id)
            .unwrap()
            .iter()
            .map(|x| *x as f64)
            .collect();
        let greedy = hf_policies::similarity_greedy_trace(&g, &embeddings, Some(&q));
        let oracle = hf_policies::oracle_trace(&g);
        assert_eq!(
            row["question_greedy_expansions"].as_u64().unwrap(),
            greedy.expansions as u64,
            "{}",
            e.episode_id
        );
        assert_eq!(
            row["oracle_expansions"].as_u64().unwrap(),
            oracle.expansions as u64
        );
        assert_eq!(
            row["question_greedy_overshoot"].as_i64().unwrap(),
            greedy.expansions as i64 - oracle.expansions as i64,
            "greedy on the question minus the oracle"
        );
        // and it is NOT the walk on the target's own vector, which is what a
        // carried stage-0 label would have been
        let on_target = hf_policies::similarity_greedy_trace(&g, &embeddings, None);
        if on_target.expansions != greedy.expansions {
            moved += 1;
        }
    }
    assert!(
        moved > 0,
        "the question changed no walk; the comparison would be vacuous"
    );
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("question.manifest.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(manifest["query_source"], "episode_query");
    assert_eq!(manifest["embedding_coverage"], 1.0);
    assert_eq!(manifest["expect_embedding_coverage"], 1.0);
    assert_eq!(manifest["query_embedding_manifest"]["dimension"], 8);
    let _ = std::fs::remove_dir_all(&root);
}

/// Without the flag the row is the one the k baselines have always written:
/// none of the question keys appears, and the manifest says which source it
/// read. Beside it, the DEFINITION the question overshoot re-uses, checked
/// against the sampler's own stored `greedy_overshoot` on a greedy-path split.
#[test]
fn without_the_flag_the_rows_are_unchanged_and_the_overshoot_definition_holds() {
    let root = tmp("baselines-definition");
    let (gdir, cache) = fixture_dirs(&root);
    let dest = root.join("pool");
    let out = run(&[
        "write",
        "--family",
        "fixture",
        "--graph-dir",
        gdir.to_str().unwrap(),
        "--subgraph-size",
        "64",
        "--target-distance",
        "3",
        "--removal-level",
        "2",
        "--removal-rule",
        "greedy-path",
        "--embeddings",
        cache.to_str().unwrap(),
        "--train",
        "0",
        "--screen",
        "6",
        "--chunk",
        "3",
        "--destination",
        dest.to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let split = dest.join("screen");
    let rows_path = root.join("plain.jsonl");
    let out = run(&[
        "baselines",
        "--split-dir",
        split.to_str().unwrap(),
        "--embeddings",
        cache.to_str().unwrap(),
        "--output",
        rows_path.to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let rows = read_rows(&rows_path);
    for row in &rows {
        for key in [
            "question_greedy_expansions",
            "question_greedy_registered_at",
            "question_greedy_stop_reason",
            "oracle_expansions",
            "question_greedy_overshoot",
        ] {
            assert!(
                row.get(key).is_none(),
                "{key} is written only with the flag"
            );
        }
    }
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("plain.manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["query_source"], "target_embedding");
    // the definition: the sampler's stored label is greedy on the TARGET's own
    // vector minus the single-target oracle, which is what the question row
    // recomputes with the question in the target's place
    let (episodes, _) = hf_io::read_split(&split).unwrap();
    let embeddings = hf_embed::EmbeddingMatrix::load(&cache).unwrap();
    let mut checked = 0;
    for e in &episodes {
        let stored = e.hidden.greedy_overshoot.expect("a greedy-path split");
        let g = hf_policies::EpisodeGraph::from_episode(e);
        let greedy = hf_policies::similarity_greedy_trace(&g, &embeddings, None);
        let oracle = hf_policies::oracle_trace(&g);
        assert_eq!(
            stored,
            greedy.expansions as i64 - oracle.expansions as i64,
            "{}",
            e.episode_id
        );
        checked += 1;
    }
    assert!(checked > 0);
    let _ = std::fs::remove_dir_all(&root);
}

/// The three refusals the sidecar brings, each exit 2: a question cache from
/// another encoder, an episode it does not cover, and a node-coverage floor
/// the node cache cannot meet.
#[test]
fn the_question_sidecar_refuses_another_encoder_a_missing_episode_and_thin_coverage() {
    let root = tmp("baselines-refusals");
    let (gdir, cache) = fixture_dirs(&root);
    let dest = root.join("pool");
    write_fixture_split(&root, &gdir, &dest, "1", 6);
    let stage1 = root.join("stage1");
    let ids = as_stage1_split(&dest.join("screen"), &stage1);
    let base = |qdir: &std::path::Path, extra: &[&str]| -> std::process::Output {
        let rows = root.join("rows.jsonl");
        let mut args: Vec<String> = vec![
            "baselines".into(),
            "--split-dir".into(),
            stage1.to_string_lossy().into(),
            "--embeddings".into(),
            cache.to_string_lossy().into(),
            "--query-embeddings-dir".into(),
            qdir.to_string_lossy().into(),
            "--output".into(),
            rows.to_string_lossy().into(),
        ];
        args.extend(extra.iter().map(|s| (*s).to_string()));
        run(&args.iter().map(String::as_str).collect::<Vec<_>>())
    };
    let wrong = root.join("queries-wrong-encoder");
    write_query_cache(&wrong, &ids, "another-encoder", 8);
    let out = base(&wrong, &[]);
    assert_eq!(out.status.code(), Some(2), "another encoder exits 2");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("another-encoder"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let narrow = root.join("queries-narrow");
    write_query_cache(&narrow, &ids, "fixture", 4);
    let out = base(&narrow, &[]);
    assert_eq!(out.status.code(), Some(2), "a narrower question exits 2");
    let partial = root.join("queries-partial");
    write_query_cache(&partial, &ids[..1], "fixture", 8);
    let out = base(&partial, &[]);
    assert_eq!(out.status.code(), Some(2), "a missing episode exits 2");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("no vector for episode"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    // a floor the node cache cannot meet: a cache holding one node only
    let thin = root.join("thin");
    std::fs::create_dir_all(&thin).unwrap();
    let (episodes, _) = hf_io::read_split(&stage1).unwrap();
    let one = episodes[0].visible.nodes[0].node.clone();
    let mut f = std::fs::File::create(thin.join("vectors.jsonl")).unwrap();
    hf_embed::append_vector(&mut f, &one, &[0.5; 8]).unwrap();
    drop(f);
    hf_embed::write_manifest(
        &thin,
        &hf_embed::Manifest {
            record_kind: hf_embed::MANIFEST_KIND.into(),
            model: "fixture".into(),
            model_digest: "fixture".into(),
            base_url: "none".into(),
            dimension: 8,
            count: 1,
            text_char_limit: 6000,
            text_sha256: None,
            truncated: Default::default(),
            training_authorized: false,
            extra: Default::default(),
        },
    )
    .unwrap();
    let out = run(&[
        "baselines",
        "--split-dir",
        stage1.to_str().unwrap(),
        "--embeddings",
        thin.to_str().unwrap(),
        "--output",
        root.join("thin.jsonl").to_str().unwrap(),
        "--expect-embedding-coverage",
        "1.0",
    ]);
    assert_eq!(out.status.code(), Some(2), "thin coverage exits 2");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("--expect-embedding-coverage"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let _ = std::fs::remove_dir_all(&root);
}
