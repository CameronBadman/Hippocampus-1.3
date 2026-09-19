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
