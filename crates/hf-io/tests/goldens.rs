//! Against splits the foundation's own sampler and writer produced
//! (`tools/goldens/gen_io_goldens.py`): every episode id, both manifests,
//! the first records, the greedy-path fields; then a Rust round trip whose
//! gzip framing matches Python's byte for byte in the header, and — when the
//! foundation checkout is beside this one — whose directory the Python
//! `validate_real_split_artifacts_v5` and `read_real_split_v5` accept.

use std::path::PathBuf;

use hf_io::{for_each_episode, read_split, validate_split_artifacts, write_split, EpisodeOut};
use serde_json::Value;

fn goldens() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/fixture-split")
}

fn canon0(v: &Value) -> String {
    String::from_utf8(hf_core::canonical_bytes(v).unwrap()).unwrap()
}

fn expected() -> Value {
    serde_json::from_str(&std::fs::read_to_string(goldens().join("expected.json")).unwrap())
        .unwrap()
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("hf-io-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

#[test]
fn python_written_splits_read_back_exactly() {
    let want = expected();
    for name in ["train", "screen", "train-greedy"] {
        let (episodes, artifacts) = read_split(&goldens().join(name)).unwrap();
        let ids: Vec<&str> = episodes.iter().map(|e| e.episode_id.as_str()).collect();
        let want_ids: Vec<&str> = want[name]["episode_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(ids, want_ids, "{name} ids");
        assert_eq!(
            canon0(&artifacts.public),
            canon0(&want[name]["public_manifest"]),
            "{name} public manifest"
        );
        assert_eq!(
            canon0(&artifacts.private),
            canon0(&want[name]["private_manifest"]),
            "{name} private manifest"
        );
        assert_eq!(canon0(&artifacts.graph), canon0(&want["graph_manifest"]));
        let first = &episodes[0];
        assert_eq!(
            canon0(&serde_json::to_value(&first.visible).unwrap()),
            canon0(&want[name]["first_visible"]),
            "{name} first visible"
        );
        assert_eq!(
            canon0(&serde_json::to_value(&first.hidden).unwrap()),
            canon0(&want[name]["first_hidden"]),
            "{name} first hidden"
        );
        assert!(
            first.visible.nodes.iter().all(|n| n.text.is_empty()),
            "visible text is empty on disk; texts.jsonl carries it"
        );
        assert_eq!(first.hidden.target_set.len(), 1);
        assert!(first
            .hidden
            .surviving_paths
            .iter()
            .all(|p| p.first() == Some(&first.visible.start_node)));
        if name == "train-greedy" {
            assert_eq!(first.hidden.removal_recipe(), "greedy-path");
            assert!(first.hidden.greedy_overshoot.is_some());
            assert_eq!(first.hidden.sampler["greedy_share"], 0.5);
            assert!(
                episodes
                    .iter()
                    .any(|e| e.hidden.removal_recipe() == "cheapest-first"),
                "share 0.5 leaves non-members"
            );
        } else {
            assert_eq!(first.hidden.removal_recipe(), "cheapest-first");
            assert!(first.hidden.greedy_overshoot.is_none());
        }
        let texts = hf_io::read_texts(&goldens().join(name)).unwrap();
        assert!(texts.values().all(|t| t.starts_with("node ")));
        assert!(first
            .visible
            .nodes
            .iter()
            .all(|n| texts.contains_key(&n.node)));
    }
}

#[test]
fn a_tampered_stream_is_band_h() {
    let dir = tmp("tamper");
    copy_dir(&goldens().join("screen"), &dir);
    let mut bytes = std::fs::read(dir.join("visible.jsonl.gz")).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    std::fs::write(dir.join("visible.jsonl.gz"), bytes).unwrap();
    let err = validate_split_artifacts(&dir).unwrap_err().to_string();
    assert!(err.starts_with("band H"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

fn copy_dir(from: &std::path::Path, to: &std::path::Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
    }
}

#[test]
fn rust_round_trip_matches_python_framing_and_validates_in_python() {
    let source = goldens().join("train-greedy");
    let (episodes, artifacts) = read_split(&source).unwrap();
    let dir = tmp("roundtrip");
    let out = episodes.iter().map(|e| {
        Ok(EpisodeOut {
            episode_id: e.episode_id.clone(),
            visible: serde_json::to_value(&e.visible).unwrap(),
            hidden: serde_json::to_value(&e.hidden).unwrap(),
        })
    });
    let (public, private) = write_split(
        "fixture",
        "stage0_known_target",
        "train",
        &dir,
        out,
        &artifacts.graph,
        &artifacts.public["sampler"],
    )
    .unwrap();
    // the two streams are byte-identical to Python's (same records, same canonical bytes, same gzip settings)
    for file in ["visible.jsonl.gz", "hidden.jsonl.gz"] {
        let ours = std::fs::read(dir.join(file)).unwrap();
        let theirs = std::fs::read(source.join(file)).unwrap();
        assert_eq!(
            &ours[..10],
            &theirs[..10],
            "{file}: gzip header (id, method, flags, mtime 0, xfl, os)"
        );
        assert_eq!(ours, theirs, "{file}: the whole container");
    }
    for key in [
        "episode_count",
        "removal_levels",
        "visible_bytes",
        "visible_sha256",
        "graph_manifest_sha256",
        "sampler",
        "family",
        "stage",
        "split",
    ] {
        assert_eq!(public[key], artifacts.public[key], "public {key}");
    }
    assert_eq!(private["hidden_sha256"], artifacts.private["hidden_sha256"]);
    use std::os::unix::fs::PermissionsExt;
    let mode = |name: &str| {
        std::fs::metadata(dir.join(name))
            .unwrap()
            .permissions()
            .mode()
            & 0o777
    };
    assert_eq!(mode("visible.jsonl.gz"), 0o644);
    assert_eq!(mode("hidden.jsonl.gz"), 0o600);
    assert_eq!(mode("manifest.private.json"), 0o600);
    assert_eq!(
        std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777,
        0o700
    );
    // re-read
    let mut n = 0;
    for_each_episode(&dir, |e, _| {
        assert_eq!(e.episode_id, episodes[n].episode_id);
        n += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(n, episodes.len());
    // an existing destination is refused
    assert!(write_split(
        "fixture",
        "stage0_known_target",
        "train",
        &dir,
        std::iter::empty(),
        &artifacts.graph,
        &Value::Null
    )
    .is_err());
    // Python validates and reads the Rust-written directory, when the foundation is present
    let foundation = std::env::var("HF_FOUNDATION")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../hippocampus-foundation")
        });
    if foundation.join("pyproject.toml").exists() {
        let script = format!(
            "from pathlib import Path\nfrom hippocampus_foundation.read_run.io_v5 import validate_real_split_artifacts_v5, read_real_split_v5\nroot = Path({:?})\nvalidate_real_split_artifacts_v5(root)\nprint(sum(1 for _ in read_real_split_v5(root)))\n",
            dir.to_string_lossy()
        );
        let output = std::process::Command::new("uv")
            .args(["run", "python", "-c", &script])
            .env("PYTHONPATH", "src")
            .current_dir(&foundation)
            .output()
            .expect("uv run python");
        assert!(
            output.status.success(),
            "python refused the Rust-written split: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&output.stdout).trim(),
            episodes.len().to_string()
        );
    } else {
        eprintln!("skipped: foundation checkout not present for the Python validation");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
