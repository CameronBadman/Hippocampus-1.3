//! The cache reader against a Python-shaped `vectors.jsonl` + manifest: the
//! sidecar is built on first load, memory-mapped on the second, refused when
//! the jsonl moves; and the ollama client against a stub server speaking the
//! `/api/tags` and `/api/embed` shapes, including the context-length overflow.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;

use hf_embed::{cosine, read_manifest, EmbeddingMatrix, OllamaEmbedClient};

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("hf-embed-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_cache(dir: &std::path::Path, rows: &[(&str, &[f64])]) {
    let manifest = format!(
        r#"{{
  "base_url": "http://127.0.0.1:11434",
  "count": {},
  "dimension": {},
  "model": "nomic-embed-text",
  "model_digest": "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f",
  "record_kind": "real_walk_embedding_manifest_v5",
  "text_char_limit": 6000,
  "text_sha256": "sha256:{}",
  "training_authorized": false,
  "truncated": {{}}
}}"#,
        rows.len(),
        rows[0].1.len(),
        "b".repeat(64)
    );
    std::fs::write(dir.join("manifest.json"), manifest).unwrap();
    let mut f = std::fs::File::create(dir.join("vectors.jsonl")).unwrap();
    for (node, v) in rows {
        hf_embed::append_vector(&mut f, node, v).unwrap();
    }
}

#[test]
fn cache_loads_builds_the_sidecar_and_refuses_a_moved_source() {
    let dir = tmp("cache");
    write_cache(
        &dir,
        &[
            ("Q1", &[1.0, 0.0, 0.0]),
            ("Q2", &[0.0, 3.0, 4.0]),
            ("Q3", &[0.0, 0.0, 0.0]),
        ],
    );
    let line = std::fs::read_to_string(dir.join("vectors.jsonl")).unwrap();
    assert!(
        line.starts_with("{\"node\": \"Q1\", \"vector\": [1.0, 0.0, 0.0]}\n"),
        "{line}"
    );
    let m = EmbeddingMatrix::load(&dir).unwrap();
    assert_eq!(m.len(), 3);
    assert_eq!(m.dimension, 3);
    assert_eq!(m.get("Q2").unwrap(), &[0.0, 3.0, 4.0]);
    assert_eq!(m.unit("Q2").unwrap(), &[0.0, 0.6, 0.8]);
    assert_eq!(
        m.unit("Q3").unwrap(),
        &[0.0, 0.0, 0.0],
        "a zero vector stays zero"
    );
    assert!(m.get("Q9").is_none());
    assert!(dir.join("vectors.f32").exists() && dir.join("vectors.index.json").exists());
    assert_eq!(cosine(m.get("Q1").unwrap(), m.get("Q2").unwrap()), 0.0);
    // second load is the mapped sidecar with the same answers
    let m2 = EmbeddingMatrix::load(&dir).unwrap();
    assert_eq!(m2.get("Q2"), m.get("Q2"));
    assert_eq!(m2.nodes, m.nodes);
    // a moved jsonl is refused until the sidecar is rebuilt
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .open(dir.join("vectors.jsonl"))
        .unwrap();
    hf_embed::append_vector(&mut f, "Q4", &[1.0, 1.0, 1.0]).unwrap();
    drop(f);
    let err = match EmbeddingMatrix::load(&dir) {
        Ok(_) => panic!("a moved vectors.jsonl must be refused"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("does not match vectors.jsonl"), "{err}");
    let manifest = read_manifest(&dir).unwrap();
    assert_eq!(manifest.count, 3);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_manifest_that_authorises_training_is_band_h() {
    let dir = tmp("authorised");
    write_cache(&dir, &[("Q1", &[1.0, 2.0])]);
    let text = std::fs::read_to_string(dir.join("manifest.json"))
        .unwrap()
        .replace(
            "\"training_authorized\": false",
            "\"training_authorized\": true",
        );
    std::fs::write(dir.join("manifest.json"), text).unwrap();
    assert!(read_manifest(&dir)
        .unwrap_err()
        .to_string()
        .starts_with("band H"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// A minimal ollama look-alike: `/api/tags` lists the model, `/api/embed`
/// returns one 4-vector per input and a 400 "context length" when any input
/// is longer than 100 characters.
fn stub_server() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let mut stream = stream.unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request = String::new();
            reader.read_line(&mut request).unwrap();
            let mut length = 0usize;
            loop {
                let mut header = String::new();
                reader.read_line(&mut header).unwrap();
                if header.trim().is_empty() {
                    break;
                }
                if let Some(v) = header.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = v.trim().parse().unwrap();
                }
            }
            let mut body = vec![0u8; length];
            reader.read_exact(&mut body).unwrap();
            let (status, response) = if request.starts_with("GET /api/tags") {
                ("200 OK", r#"{"models":[{"name":"other:latest","model":"other:latest","digest":"ffff"},{"name":"nomic-embed-text:latest","model":"nomic-embed-text:latest","digest":"0a109f42"}]}"#.to_string())
            } else {
                let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
                let inputs = v["input"].as_array().unwrap();
                if inputs
                    .iter()
                    .any(|t| t.as_str().unwrap().chars().count() > 100)
                {
                    (
                        "400 Bad Request",
                        r#"{"error":"input length exceeds the context length"}"#.to_string(),
                    )
                } else {
                    let vectors: Vec<String> = inputs
                        .iter()
                        .map(|t| {
                            format!(
                                "[{}.0, 1.5, -2.0, 0.25]",
                                t.as_str().unwrap().chars().count()
                            )
                        })
                        .collect();
                    (
                        "200 OK",
                        format!(
                            r#"{{"model":"nomic-embed-text","embeddings":[{}]}}"#,
                            vectors.join(",")
                        ),
                    )
                }
            };
            write!(stream, "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}", response.len()).unwrap();
        }
    });
    format!("http://{addr}")
}

#[test]
fn client_finds_the_served_digest_embeds_and_reports_overflow() {
    let base = stub_server();
    let client = OllamaEmbedClient::new(&base, "nomic-embed-text", 5.0);
    assert_eq!(
        client.served_digest().unwrap(),
        "0a109f42",
        "the :latest rule for an untagged name"
    );
    let missing = OllamaEmbedClient::new(&base, "absent-model", 5.0);
    assert!(missing
        .served_digest()
        .unwrap_err()
        .to_string()
        .contains("not served"));
    let vectors = client
        .embed(&["abc".into(), "hello".into()], 1)
        .unwrap()
        .unwrap();
    assert_eq!(
        vectors,
        vec![vec![3.0, 1.5, -2.0, 0.25], vec![5.0, 1.5, -2.0, 0.25]]
    );
    let long = "x".repeat(150);
    assert!(
        client.embed(&[long], 1).unwrap().is_err(),
        "a context-length 400 is an overflow, not a failure"
    );
}

#[test]
fn the_cli_writes_a_python_shaped_cache_and_resumes_only_a_matching_one() {
    let base = stub_server();
    let dir = tmp("cli");
    let text = dir.join("text.tsv");
    std::fs::write(
        &text,
        format!("Q1\tone\nQ2\t{}\nQ3\t\nQ4\tfour\n", "y".repeat(300)),
    )
    .unwrap();
    let bin = env!("CARGO_BIN_EXE_hf-embed");
    let dest = dir.join("cache");
    let run = |extra: &[&str]| {
        std::process::Command::new(bin)
            .args([
                "--text",
                text.to_str().unwrap(),
                "--destination",
                dest.to_str().unwrap(),
                "--base-url",
                &base,
                "--max-chars",
                "200",
                "--batch",
                "2",
            ])
            .args(extra)
            .output()
            .unwrap()
    };
    let out = run(&["--limit", "3"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let manifest = read_manifest(&dest).unwrap();
    assert_eq!(
        (manifest.count, manifest.dimension, manifest.text_char_limit),
        (3, 4, 200)
    );
    assert_eq!(manifest.model_digest, "0a109f42");
    assert_eq!(
        manifest.truncated.get("Q2"),
        Some(&100),
        "halved from 200 until accepted"
    );
    let m = EmbeddingMatrix::load(&dest).unwrap();
    assert_eq!(m.nodes, vec!["Q1", "Q2", "Q3"]);
    assert_eq!(
        m.get("Q3").unwrap()[0],
        2.0,
        "an empty text row embeds the node id"
    );
    // resume: the fourth node is appended, nothing re-embedded
    let out = run(&[]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(read_manifest(&dest).unwrap().count, 4);
    // a different character limit refuses to mix
    let out = std::process::Command::new(bin)
        .args([
            "--text",
            text.to_str().unwrap(),
            "--destination",
            dest.to_str().unwrap(),
            "--base-url",
            &base,
            "--max-chars",
            "50",
        ])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("refusing to mix"));
    // an expected digest that does not match is refused before anything is written
    let out = run(&["--expect-digest", "deadbeef"]);
    assert_eq!(out.status.code(), Some(2));
    let _ = std::fs::remove_dir_all(&dir);
}
