//! `hf-embed merge` (ENG-2): several caches into one, and every refusal.

use std::path::{Path, PathBuf};

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("hf-embed-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn vector(seed: f64, dimension: u32) -> Vec<f64> {
    (0..dimension)
        .map(|j| (seed * 0.37 + j as f64 * 0.11).sin())
        .collect()
}

/// A query cache as the teacher writes it: ids are episode ids, `keyed_by`.
fn cache(dir: &Path, rows: &[(&str, f64)], digest: &str, dimension: u32, keyed: bool) {
    std::fs::create_dir_all(dir).unwrap();
    let mut f = std::fs::File::create(dir.join("vectors.jsonl")).unwrap();
    for (id, seed) in rows {
        hf_embed::append_vector(&mut f, id, &vector(*seed, dimension)).unwrap();
    }
    let mut extra = serde_json::Map::new();
    if keyed {
        extra.insert("keyed_by".into(), "episode_id".into());
        extra.insert("source_stage0_split".into(), dir.to_string_lossy().into());
    }
    let m = hf_embed::Manifest {
        record_kind: hf_embed::MANIFEST_KIND.into(),
        model: "nomic-embed-text".into(),
        model_digest: digest.into(),
        base_url: "http://127.0.0.1:11434".into(),
        dimension,
        count: rows.len() as u64,
        text_char_limit: 6000,
        text_sha256: None,
        truncated: Default::default(),
        training_authorized: false,
        extra,
    };
    hf_embed::write_manifest(dir, &m).unwrap();
}

fn merge(into: &Path, sources: &[&Path]) -> std::process::Output {
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_hf-embed"));
    cmd.arg("merge").arg("--into").arg(into);
    for s in sources {
        cmd.arg(s);
    }
    cmd.output().unwrap()
}

#[test]
fn merges_several_query_caches_into_one_the_engine_loads() {
    let root = tmp("merge-ok");
    let (train, screen, vault) = (root.join("train"), root.join("screen"), root.join("vault"));
    cache(&train, &[("t1", 1.0), ("t2", 2.0)], "d", 8, true);
    // the screen repeats t2 with the SAME vector (kept once) and adds its own
    cache(&screen, &[("t2", 2.0), ("s1", 3.0)], "d", 8, true);
    cache(&vault, &[("v1", 4.0)], "d", 8, true);
    let into = root.join("merged");
    let out = merge(&into, &[&train, &screen, &vault]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let m = hf_embed::read_manifest(&into).unwrap();
    assert_eq!(m.count, 4);
    assert_eq!(m.model_digest, "d");
    assert_eq!(m.extra["keyed_by"], "episode_id");
    assert_eq!(m.extra["duplicates_identical"], 1);
    let from = m.extra["merged_from"].as_array().unwrap();
    assert_eq!(from.len(), 3);
    assert_eq!(from[1]["written"], 1);
    assert_eq!(from[1]["duplicates_identical"], 1);
    assert_eq!(
        from[0]["vectors_sha256"],
        hf_core::sha256_file(&train.join("vectors.jsonl"))
            .unwrap()
            .1
    );
    assert_eq!(
        from[2]["manifest"]["source_stage0_split"],
        vault.to_string_lossy().as_ref()
    );
    // the engine's own loader reads it, rows in source order
    let matrix = hf_embed::EmbeddingMatrix::load(&into).unwrap();
    assert_eq!(matrix.nodes, vec!["t1", "t2", "s1", "v1"]);
    let want: Vec<f32> = vector(3.0, 8).iter().map(|x| *x as f32).collect();
    assert_eq!(matrix.get("s1").unwrap(), want.as_slice());
    hf_embed::same_encoder(&hf_embed::read_manifest(&train).unwrap(), &m).unwrap();
    // an existing destination is refused
    let again = merge(&into, &[&train]);
    assert!(!again.status.success());
    assert!(String::from_utf8_lossy(&again.stderr).contains("already exists"));
}

#[test]
fn refuses_two_vectors_for_one_id_another_encoder_width_or_key_and_a_short_count() {
    let root = tmp("merge-refusals");
    let a = root.join("a");
    cache(&a, &[("e1", 1.0), ("e2", 2.0)], "d", 8, true);
    let case = |name: &str, other: &Path, needle: &str| {
        let into = root.join(format!("into-{name}"));
        let out = merge(&into, &[&a, other]);
        assert!(!out.status.success(), "{name} was accepted");
        let err = String::from_utf8_lossy(&out.stderr).to_string();
        assert!(err.contains(needle), "{name}: {err}");
        assert!(!into.exists(), "{name} left a destination");
    };
    // the test-writer's and the teacher's question for one episode differ
    let clash = root.join("clash");
    cache(&clash, &[("e2", 9.0)], "d", 8, true);
    case("clash", &clash, "DIFFERENT vectors");
    let digest = root.join("digest");
    cache(&digest, &[("e3", 3.0)], "other", 8, true);
    case("digest", &digest, "cosine between two encoders");
    let wide = root.join("wide");
    cache(&wide, &[("e3", 3.0)], "d", 16, true);
    case("wide", &wide, "wide");
    // a node cache (no keyed_by) cannot be folded into a query cache
    let nodes = root.join("nodes");
    cache(&nodes, &[("n1", 3.0)], "d", 8, false);
    case("keyed", &nodes, "keyed_by");
    // a manifest whose count disagrees with its lines
    let short = root.join("short");
    cache(&short, &[("e4", 4.0), ("e5", 5.0)], "d", 8, true);
    let mut m = hf_embed::read_manifest(&short).unwrap();
    m.count = 3;
    hf_embed::write_manifest(&short, &m).unwrap();
    case("count", &short, "manifest says 3");
    // the same source twice, and a holdout destination
    let twice = merge(&root.join("into-twice"), &[&a, &a]);
    assert!(!twice.status.success());
    assert!(String::from_utf8_lossy(&twice.stderr).contains("named twice"));
    let holdout = merge(&root.join("heldout-merged"), &[&a]);
    assert!(!holdout.status.success());
    assert!(!root.join("heldout-merged").exists());
    assert!(std::fs::read_dir(&root).unwrap().all(|e| !e
        .unwrap()
        .file_name()
        .to_string_lossy()
        .starts_with(".tmp-")));
}

/// The embedding CLI itself still parses as before: `merge` is dispatched on
/// the first argument only.
#[test]
fn the_embed_cli_is_unchanged() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_hf-embed"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(out.status.success());
    let help = String::from_utf8_lossy(&out.stdout);
    assert!(help.contains("--text") && help.contains("--expect-digest"));
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_hf-embed"))
        .args(["merge", "--help"])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("--into"));
}
