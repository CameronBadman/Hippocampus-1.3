//! `hf-splits deletions`, generalised for the R1 head (`R1_HEAD_DESIGN.md`
//! §8 ENG-1): a declared draw label and declared raw-index ranges replace the
//! premise's constants, `--per-start rotate` keeps one record per start,
//! `--coverage-only` writes no stream, `--require-coverage` refuses an
//! uncached node, and the manifest records the cache state the draw depends
//! on.

use std::path::{Path, PathBuf};

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("hf-splits-r1-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn run(args: &[&str]) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_hf-splits"))
        .args(args)
        .output()
        .unwrap()
}

fn fixture_manifest(count: u64) -> hf_embed::Manifest {
    hf_embed::Manifest {
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
    }
}

/// The fixture world as an adapter directory plus its embeddings as a v5
/// cache; `skip` names nodes the cache is written WITHOUT.
fn fixture_dirs(root: &Path, skip: &[&str]) -> (PathBuf, PathBuf) {
    let (graph, embeddings) = hf_episodes::fixture::fixture_world(5, 400, 1600, 8);
    let gdir = root.join("graph");
    std::fs::create_dir_all(&gdir).unwrap();
    let mut edges = String::new();
    for h in graph.nodes() {
        for e in graph.out(h) {
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
    let mut count = 0;
    for n in &names {
        if skip.contains(&n.as_str()) {
            continue;
        }
        hf_embed::append_vector(&mut f, n, &embeddings[*n]).unwrap();
        count += 1;
    }
    hf_embed::write_manifest(&cache, &fixture_manifest(count)).unwrap();
    (gdir, cache)
}

/// Append the named nodes' rows to the cache (as an `hf-embed` append would).
fn append_rows(cache: &Path, names: &[String]) {
    let (_, embeddings) = hf_episodes::fixture::fixture_world(5, 400, 1600, 8);
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .open(cache.join("vectors.jsonl"))
        .unwrap();
    for n in names {
        hf_embed::append_vector(&mut f, n, &embeddings[n]).unwrap();
    }
    let have = std::fs::read_to_string(cache.join("vectors.jsonl"))
        .unwrap()
        .lines()
        .count();
    hf_embed::write_manifest(cache, &fixture_manifest(have as u64)).unwrap();
}

/// A sampler-written fixture split (`screen` only) with `screen` episodes.
fn write_fixture_split(gdir: &Path, dest: &Path, screen: usize) {
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
        "1",
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

fn gz_records(path: &Path) -> Vec<serde_json::Value> {
    let raw = hf_io::read_maybe_gz(path).unwrap();
    raw.split(|b| *b == b'\n')
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_slice(l).unwrap())
        .collect()
}

fn sha(path: &Path) -> String {
    hf_core::sha256_file(path).unwrap().1
}

fn deletions(
    split: &Path,
    gdir: &Path,
    cache: &Path,
    dest: &Path,
    extra: &[&str],
) -> std::process::Output {
    let mut args: Vec<&str> = vec![
        "deletions",
        "--split-dir",
        split.to_str().unwrap(),
        "--graph-dir",
        gdir.to_str().unwrap(),
        "--embeddings",
        cache.to_str().unwrap(),
        "--destination",
        dest.to_str().unwrap(),
    ];
    args.extend_from_slice(extra);
    run(&args)
}

const PREMISE: [&str; 6] = [
    "--draw-label",
    "r1-premise-2026-09-25",
    "--allowed-range",
    "train:0..73359",
    "--allowed-range",
    "screen:0..4104",
];

/// Rewrite a split's visible stream through `edit` and restamp the public
/// manifest's digest, so a doctored source reaches the check under test.
fn doctor_visible(split: &Path, edit: impl Fn(usize, &mut serde_json::Value)) {
    use std::io::Write;
    let mut records = gz_records(&split.join("visible.jsonl.gz"));
    for (i, r) in records.iter_mut().enumerate() {
        edit(i, r);
    }
    let path = split.join("visible.jsonl.gz");
    let file = std::fs::File::create(&path).unwrap();
    let mut gz = flate2::write::GzEncoder::new(file, flate2::Compression::default());
    for r in &records {
        gz.write_all(&hf_core::canonical_bytes(r).unwrap()).unwrap();
        gz.write_all(b"\n").unwrap();
    }
    gz.finish().unwrap();
    let (bytes, sha) = hf_core::sha256_file(&path).unwrap();
    let mpath = split.join("manifest.public.json");
    let mut m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&mpath).unwrap()).unwrap();
    m["visible_bytes"] = bytes.into();
    m["visible_sha256"] = sha.into();
    std::fs::write(&mpath, hf_core::dump_pretty(&m)).unwrap();
}

fn manifest(dest: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(dest.join("deletions.manifest.json")).unwrap())
        .unwrap()
}

/// ENG-1 (a): with the premise's label and its old ranges, the draw is byte
/// for byte what the premise's engine (b8b074b) wrote on this fixture; the
/// goldens were captured from that build before any of this code existed.
#[test]
fn the_premises_label_and_ranges_reproduce_todays_draw_byte_for_byte() {
    let root = tmp("regress");
    let (gdir, cache) = fixture_dirs(&root, &[]);
    let pool = root.join("pool");
    write_fixture_split(&gdir, &pool, 12);
    let split = pool.join("screen");
    let dest = root.join("del");
    let mut extra = vec!["--threads", "2"];
    extra.extend_from_slice(&PREMISE);
    let out = deletions(&split, &gdir, &cache, &dest, &extra);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    for (f, want) in [
        (
            "visible.jsonl.gz",
            "sha256:d0fe51ce85f4467c0e065f36d6e7dbf7a5ea5eeec58681fba62200e2cf81103d",
        ),
        (
            "visible_h.jsonl.gz",
            "sha256:28bfb932c7de660c16957fd91d3d95bee1217548c7a6303db78cac5db5267ad1",
        ),
        (
            "labels.jsonl.gz",
            "sha256:7c05c1f127cc8de4afaf675b0cae31cb628815fd6d27666eca38b315e9c7f9bd",
        ),
        (
            "queries/vectors.jsonl",
            "sha256:107c7ccd1154df521c6e1c7882f907fa242442c124351f3cb1585539bc8ce376",
        ),
        (
            "uncached_nodes.txt",
            "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
    ] {
        assert_eq!(
            sha(&dest.join(f)),
            want,
            "{f} differs from the premise engine's"
        );
    }
    let m = manifest(&dest);
    assert_eq!(m["draw_label"], "r1-premise-2026-09-25");
    assert_eq!(
        m["allowed_ranges"],
        serde_json::json!(["train:0..73359", "screen:0..4104"])
    );
    assert_eq!(m["per_start"], "all");
    assert_eq!(m["records_written"], 48);
}

/// ENG-1 (b): a range refusal on each side of each bound, for both splits,
/// and for an undeclared split. A refused id exits 2 before anything is
/// written; an id on a bound is accepted (the ranges are inclusive).
#[test]
fn a_declared_range_refuses_on_each_side_of_each_bound_and_an_undeclared_split() {
    let root = tmp("ranges");
    let (gdir, cache) = fixture_dirs(&root, &[]);
    let ranges = [
        "--draw-label",
        "r1-head-test",
        "--allowed-range",
        "train:10..20",
        "--allowed-range",
        "screen:30..40",
        "--limit",
        "1",
        "--threads",
        "1",
    ];
    let cases: [(&str, u64, bool); 8] = [
        ("train", 9, false),
        ("train", 10, true),
        ("train", 20, true),
        ("train", 21, false),
        ("screen", 29, false),
        ("screen", 30, true),
        ("screen", 40, true),
        ("screen", 41, false),
    ];
    for (k, (split_name, index, ok)) in cases.iter().enumerate() {
        let pool = root.join(format!("pool{k}"));
        write_fixture_split(&gdir, &pool, 2);
        let split = pool.join("screen");
        doctor_visible(&split, |_, r| {
            let id = r["episode_id"].as_str().unwrap().to_string();
            let tail = id.splitn(4, '-').nth(3).unwrap().to_string();
            r["episode_id"] = format!("fixture-{split_name}-{index:06}-{tail}").into();
        });
        let dest = root.join(format!("del{k}"));
        let out = deletions(&split, &gdir, &cache, &dest, &ranges);
        if *ok {
            assert!(
                out.status.success(),
                "{split_name} {index}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        } else {
            assert_eq!(
                out.status.code(),
                Some(2),
                "{split_name} {index} must be refused"
            );
            assert!(String::from_utf8_lossy(&out.stderr).contains("outside the declared ranges"));
            assert!(!dest.exists(), "nothing is written for a refused id");
        }
    }
    // a split no range declares
    let pool = root.join("pool-undeclared");
    write_fixture_split(&gdir, &pool, 2);
    let dest = root.join("del-undeclared");
    let out = deletions(
        &pool.join("screen"),
        &gdir,
        &cache,
        &dest,
        &[
            "--draw-label",
            "r1-head-test",
            "--allowed-range",
            "train:0..",
            "--threads",
            "1",
        ],
    );
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("not declared"));
    assert!(!dest.exists());
    // the flags are required
    let out = deletions(
        &pool.join("screen"),
        &gdir,
        &cache,
        &root.join("no-flags"),
        &[],
    );
    assert_eq!(out.status.code(), Some(2));
}

/// ENG-1 (c): `rotate` keeps exactly one record per start, tagged
/// `DRAW_TAGS[ordinal mod 5]` or, when that tag has no candidate, `U` (each
/// fallback counted); it picks the same X that tag picks under `all`; its
/// output is deterministic and does not depend on the thread count.
#[test]
fn rotate_takes_one_record_per_start_falls_back_to_u_and_ignores_threads() {
    let root = tmp("rotate");
    let (gdir, cache) = fixture_dirs(&root, &[]);
    let pool = root.join("pool");
    write_fixture_split(&gdir, &pool, 15);
    let split = pool.join("screen");
    let label = [
        "--draw-label",
        "r1-head-test",
        "--allowed-range",
        "screen:0..",
    ];
    let go = |dest: &Path, per_start: &str, threads: &str| {
        let mut extra = label.to_vec();
        extra.extend_from_slice(&["--per-start", per_start, "--threads", threads]);
        let out = deletions(&split, &gdir, &cache, dest, &extra);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    };
    let one = root.join("rot1");
    let four = root.join("rot4");
    let again = root.join("rot1-again");
    let all = root.join("all");
    go(&one, "rotate", "1");
    go(&four, "rotate", "4");
    go(&again, "rotate", "1");
    go(&all, "all", "2");
    for f in [
        "visible.jsonl.gz",
        "visible_h.jsonl.gz",
        "labels.jsonl.gz",
        "queries/vectors.jsonl",
    ] {
        assert_eq!(sha(&one.join(f)), sha(&four.join(f)), "{f}: threads");
        assert_eq!(sha(&one.join(f)), sha(&again.join(f)), "{f}: rerun");
    }
    let source: Vec<String> = gz_records(&split.join("visible.jsonl.gz"))
        .iter()
        .map(|r| r["episode_id"].as_str().unwrap().to_string())
        .collect();
    let labels = gz_records(&one.join("labels.jsonl.gz"));
    let all_labels = gz_records(&all.join("labels.jsonl.gz"));
    let tags = ["U", "D1", "D2", "D3", "D4"];
    let mut seen = std::collections::BTreeSet::new();
    let mut fallbacks: std::collections::BTreeMap<String, u64> = Default::default();
    for l in &labels {
        let src = l["source_episode_id"].as_str().unwrap();
        assert!(
            seen.insert(src.to_string()),
            "{src}: two records for one start"
        );
        let ordinal = source.iter().position(|s| s == src).unwrap();
        let want = tags[ordinal % 5];
        let draws = l["draws"].as_array().unwrap();
        assert_eq!(draws.len(), 1);
        let got = draws[0].as_str().unwrap();
        assert!(l["episode_id"]
            .as_str()
            .unwrap()
            .ends_with(&format!("-del-{got}")));
        if got != want {
            assert_eq!(got, "U", "{src}: only a fall-back to U may change the tag");
            *fallbacks.entry(want.to_string()).or_default() += 1;
        }
        // the same X the tag picks under `all`
        let same = all_labels
            .iter()
            .find(|a| {
                a["source_episode_id"] == l["source_episode_id"]
                    && a["draws"].as_array().unwrap().iter().any(|t| t == got)
            })
            .expect("the tag's pick under all");
        assert_eq!(same["deleted_node"], l["deleted_node"]);
    }
    assert!(
        !fallbacks.is_empty(),
        "the fixture must exercise a fall-back"
    );
    let m = manifest(&one);
    assert_eq!(m["per_start"], "rotate");
    assert_eq!(
        m["rotate_fallbacks_to_u"],
        serde_json::to_value(&fallbacks).unwrap()
    );
    // every start with a candidate yields one record
    assert_eq!(
        labels.len() as u64 + m["drops"]["no_candidate"].as_u64().unwrap_or(0),
        source.len() as u64
    );
}

/// The ball′ node of some drawn record that is in no stored ball of the
/// source split: an entrant only a deletion brings in.
fn an_entrant(split: &Path, del: &Path) -> String {
    let stored: std::collections::BTreeSet<String> = gz_records(&split.join("visible.jsonl.gz"))
        .iter()
        .flat_map(|r| {
            r["visible"]["nodes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n["node"].as_str().unwrap().to_string())
                .collect::<Vec<_>>()
        })
        .collect();
    for r in gz_records(&del.join("visible.jsonl.gz")) {
        for n in r["visible"]["nodes"].as_array().unwrap() {
            let n = n.as_str().unwrap();
            if !stored.contains(n) {
                return n.to_string();
            }
        }
    }
    panic!("the fixture has no entrant");
}

/// ENG-1 (d), (e) and (f), and the fixed point the prereg's order relies on:
/// `--coverage-only` writes no stream and lists the uncached entrant;
/// `--require-coverage` exits 2 on it and writes nothing; once the listed
/// rows are appended, `--require-coverage` succeeds in ONE pass; and the
/// recorded cache prefix changes when the cache is appended to.
#[test]
fn coverage_only_writes_no_stream_and_require_coverage_refuses_until_covered() {
    let root = tmp("coverage");
    let pool = root.join("pool");
    // a first, fully covered draw finds an entrant
    let full = root.join("full");
    std::fs::create_dir_all(&full).unwrap();
    let (gdir, full_cache) = fixture_dirs(&full, &[]);
    write_fixture_split(&gdir, &pool, 12);
    let split = pool.join("screen");
    let probe = root.join("probe");
    let mut extra = vec!["--threads", "2"];
    extra.extend_from_slice(&PREMISE);
    assert!(deletions(&split, &gdir, &full_cache, &probe, &extra)
        .status
        .success());
    let entrant = an_entrant(&split, &probe);
    // the same world with that one row missing
    let lacking = root.join("lacking");
    std::fs::create_dir_all(&lacking).unwrap();
    let (gdir2, cache) = fixture_dirs(&lacking, &[entrant.as_str()]);
    // (e) require-coverage refuses and writes nothing
    let refused = root.join("refused");
    let mut req = extra.clone();
    req.push("--require-coverage");
    let out = deletions(&split, &gdir2, &cache, &refused, &req);
    assert_eq!(
        out.status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stderr).contains("--require-coverage"));
    assert!(!refused.exists(), "nothing is written");
    // (d) coverage-only writes the list and the manifest, and no stream
    let cov = root.join("cov");
    let mut covx = extra.clone();
    covx.push("--coverage-only");
    let out = deletions(&split, &gdir2, &cache, &cov, &covx);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    for absent in [
        "labels.jsonl.gz",
        "visible.jsonl.gz",
        "visible_h.jsonl.gz",
        "queries",
    ] {
        assert!(
            !cov.join(absent).exists(),
            "{absent} was written under --coverage-only"
        );
    }
    let listed: Vec<String> = std::fs::read_to_string(cov.join("uncached_nodes.txt"))
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect();
    assert!(
        listed.contains(&entrant),
        "{entrant} not listed: {listed:?}"
    );
    let m0 = manifest(&cov);
    assert_eq!(m0["mode"], "coverage_only");
    assert_eq!(m0["streams_written"], serde_json::json!([]));
    // the fixed point: append exactly the listed rows, then require coverage
    append_rows(&cache, &listed);
    let governed = root.join("governed");
    let out = deletions(&split, &gdir2, &cache, &governed, &req);
    assert!(
        out.status.success(),
        "one embedding pass must reach the fixed point: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let m1 = manifest(&governed);
    // (f) the cache prefix recorded moved with the append
    assert_ne!(
        m0["cache_state"]["vectors_prefix_sha256"],
        m1["cache_state"]["vectors_prefix_sha256"]
    );
    assert_eq!(
        m1["cache_state"]["vectors_lines"].as_u64().unwrap(),
        m0["cache_state"]["vectors_lines"].as_u64().unwrap() + listed.len() as u64
    );
    assert_eq!(m1["mode"], "require_coverage");
    // and the governed draw is the fully covered draw
    for f in ["visible.jsonl.gz", "labels.jsonl.gz"] {
        assert_eq!(
            gz_records(&governed.join(f)),
            gz_records(&probe.join(f)),
            "{f}"
        );
    }
}

/// `--source-ordinals` (prereg §2.5): only the source records at the declared
/// positions are drawn, each exactly as the whole-stream draw drew it (the
/// rotation keeps the absolute position); an open range runs to the end; a
/// range past the stream's end, a reversed or malformed range, and a range
/// with `--limit` exit 2 and write nothing; the manifest records the block.
#[test]
fn source_ordinals_draw_only_the_declared_block_exactly_as_the_whole_stream() {
    let root = tmp("ordinals");
    let (gdir, cache) = fixture_dirs(&root, &[]);
    let pool = root.join("pool");
    write_fixture_split(&gdir, &pool, 15);
    let split = pool.join("screen");
    let source: Vec<String> = gz_records(&split.join("visible.jsonl.gz"))
        .iter()
        .map(|r| r["episode_id"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(source.len(), 15);
    let go = |dest: &Path, extra: &[&str]| {
        let mut args = vec![
            "--draw-label",
            "r1-head-test",
            "--allowed-range",
            "screen:0..",
            "--per-start",
            "rotate",
            "--threads",
            "2",
        ];
        args.extend_from_slice(extra);
        deletions(&split, &gdir, &cache, dest, &args)
    };
    let whole = root.join("whole");
    assert!(go(&whole, &[]).status.success());
    let whole_labels = gz_records(&whole.join("labels.jsonl.gz"));
    let whole_visible = gz_records(&whole.join("visible.jsonl.gz"));
    for (spec, lo, hi) in [("6..9", 6usize, 9usize), ("11..", 11, 14)] {
        let dest = root.join(format!("block-{lo}"));
        let out = go(&dest, &["--source-ordinals", spec]);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let want: Vec<&String> = source[lo..=hi].iter().collect();
        let labels = gz_records(&dest.join("labels.jsonl.gz"));
        let visible = gz_records(&dest.join("visible.jsonl.gz"));
        assert!(!labels.is_empty());
        for l in &labels {
            assert!(want.contains(&&l["source_episode_id"].as_str().unwrap().to_string()));
        }
        // record for record, the whole draw's records of those starts
        let expect_l: Vec<&serde_json::Value> = whole_labels
            .iter()
            .filter(|l| want.contains(&&l["source_episode_id"].as_str().unwrap().to_string()))
            .collect();
        assert_eq!(labels.iter().collect::<Vec<_>>(), expect_l, "{spec}");
        let ids: Vec<&serde_json::Value> = expect_l.iter().map(|l| &l["episode_id"]).collect();
        let expect_v: Vec<&serde_json::Value> = whole_visible
            .iter()
            .filter(|v| ids.contains(&&v["episode_id"]))
            .collect();
        assert_eq!(visible.iter().collect::<Vec<_>>(), expect_v, "{spec}");
        let m = manifest(&dest);
        assert_eq!(m["source_ordinals"]["first"], lo);
        assert_eq!(m["source_ordinals"]["last"], hi);
        assert_eq!(m["source_ordinals"]["count"], hi - lo + 1);
        assert_eq!(m["source_episodes_read"], hi - lo + 1);
    }
    assert!(manifest(&whole)["source_ordinals"].is_null());
    for (k, bad) in [
        vec!["--source-ordinals", "10..15"],
        vec!["--source-ordinals", "15.."],
        vec!["--source-ordinals", "9..6"],
        vec!["--source-ordinals", "a..3"],
        vec!["--source-ordinals", "3"],
        vec!["--source-ordinals", "0..3", "--limit", "2"],
    ]
    .iter()
    .enumerate()
    {
        let dest = root.join(format!("bad{k}"));
        let out = go(&dest, bad);
        assert_eq!(out.status.code(), Some(2), "{bad:?}");
        assert!(!dest.exists(), "{bad:?} wrote something");
    }
}

// --------------------------------------------------------------------------
// `merge-deletions`: a draw built in --source-ordinals chunks

fn merge(parts: &[&Path], dest: &Path) -> std::process::Output {
    let mut args: Vec<String> = vec!["merge-deletions".into()];
    for p in parts {
        args.push("--part".into());
        args.push(p.to_str().unwrap().into());
    }
    args.push("--destination".into());
    args.push(dest.to_str().unwrap().into());
    run(&args.iter().map(String::as_str).collect::<Vec<_>>())
}

const MERGED_FILES: [&str; 6] = [
    "visible.jsonl.gz",
    "visible_h.jsonl.gz",
    "labels.jsonl.gz",
    "queries/vectors.jsonl",
    "queries/manifest.json",
    "uncached_nodes.txt",
];

/// The merged manifest with its one provenance key removed, re-serialised as
/// the writer serialises: must be the one-go manifest's exact bytes.
fn manifest_without_chunks(dir: &Path) -> String {
    let mut m = manifest(dir);
    assert!(
        m.get("chunks").is_some(),
        "the merged manifest records its parts"
    );
    m.as_object_mut().unwrap().remove("chunks");
    hf_core::files::python_json_pretty(&m)
}

/// Chunked = one-go, byte for byte: 4 blocks of 5 starts, given out of
/// order, merged, against one draw over positions 0..19 — every stream, the
/// query sidecar and its manifest, the uncached list, and the manifest bar
/// its `chunks` key. Three draws: the TRAIN shape (`rotate`,
/// `--require-coverage`), `all` with a node missing from the cache (the drop
/// path: drops summed, the uncached set united), and `--coverage-only`.
#[test]
fn a_chunked_draw_merges_into_the_one_go_draw_byte_for_byte() {
    let root = tmp("merge-eq");
    let full = root.join("full");
    std::fs::create_dir_all(&full).unwrap();
    let (gdir0, full_cache) = fixture_dirs(&full, &[]);
    let pool = root.join("pool");
    write_fixture_split(&gdir0, &pool, 20);
    let split = pool.join("screen");
    // an entrant of some draw, to leave out of a second cache
    let probe = root.join("probe");
    let mut pe = vec!["--threads", "2"];
    pe.extend_from_slice(&PREMISE);
    assert!(deletions(&split, &gdir0, &full_cache, &probe, &pe)
        .status
        .success());
    let entrant = an_entrant(&split, &probe);
    // and a node in stored balls of starts in the first AND the last block,
    // so the uncached set of two parts overlaps (a union, not a sum)
    let balls: Vec<std::collections::BTreeSet<String>> =
        gz_records(&split.join("visible.jsonl.gz"))
            .iter()
            .map(|r| {
                r["visible"]["nodes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .skip(1)
                    .map(|n| n["node"].as_str().unwrap().to_string())
                    .collect()
            })
            .collect();
    let early: std::collections::BTreeSet<&String> = balls[..5].iter().flatten().collect();
    let shared = balls[15..]
        .iter()
        .flatten()
        .find(|n| early.contains(n) && **n != entrant)
        .expect("a node shared by the first and last blocks")
        .clone();
    let lacking = root.join("lacking");
    std::fs::create_dir_all(&lacking).unwrap();
    let (gdir, thin_cache) = fixture_dirs(&lacking, &[entrant.as_str(), shared.as_str()]);
    let blocks = ["10..14", "0..4", "15..19", "5..9"];
    for (name, cache, extra) in [
        (
            "rotate",
            &full_cache,
            vec!["--per-start", "rotate", "--require-coverage"],
        ),
        ("all-drop", &thin_cache, vec!["--per-start", "all"]),
        (
            "coverage",
            &thin_cache,
            vec!["--per-start", "rotate", "--coverage-only"],
        ),
    ] {
        let base = [
            "--draw-label",
            "r1-head-test",
            "--allowed-range",
            "screen:0..",
            "--threads",
            "3",
        ];
        let go = |dest: &Path, ordinals: &str| {
            let mut a: Vec<&str> = base.to_vec();
            a.extend_from_slice(&extra);
            a.extend_from_slice(&["--source-ordinals", ordinals]);
            let out = deletions(&split, &gdir, cache, dest, &a);
            assert!(
                out.status.success(),
                "{name} {ordinals}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        };
        let one = root.join(format!("{name}-one"));
        go(&one, "0..19");
        let mut parts = Vec::new();
        for b in blocks {
            let d = root.join(format!("{name}-part-{b}"));
            go(&d, b);
            parts.push(d);
        }
        let merged = root.join(format!("{name}-merged"));
        let refs: Vec<&Path> = parts.iter().map(PathBuf::as_path).collect();
        let out = merge(&refs, &merged);
        assert!(
            out.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let files: &[&str] = if name == "coverage" {
            &["uncached_nodes.txt"]
        } else {
            &MERGED_FILES
        };
        for f in files {
            assert_eq!(sha(&merged.join(f)), sha(&one.join(f)), "{name}: {f}");
        }
        if name == "coverage" {
            assert!(!merged.join("labels.jsonl.gz").exists());
        }
        assert_eq!(
            manifest_without_chunks(&merged),
            std::fs::read_to_string(one.join("deletions.manifest.json")).unwrap(),
            "{name}: the manifest"
        );
        let m = manifest(&merged);
        assert_eq!(m["chunks"]["parts"].as_array().unwrap().len(), 4);
        assert_eq!(m["chunks"]["parts"][0]["first"], 0, "sorted by block");
        if name != "rotate" {
            // the uncached set is a union: some node is listed by two parts
            let mut listed = 0u64;
            for p in &parts {
                listed += manifest(p)["uncached_nodes"].as_u64().unwrap();
            }
            assert!(
                listed > m["uncached_nodes"].as_u64().unwrap(),
                "{name}: no shared uncached node"
            );
        }
        if name == "all-drop" {
            assert!(
                m["drops"]["uncached_in_ball"].as_u64().unwrap() > 0,
                "the drop path ran"
            );
        }
        if name == "rotate" {
            assert!(m["records_written"].as_u64().unwrap() > 0);
        }
    }
}

/// The refusals, each exit 2 with nothing written: another draw label,
/// another cache state, another per-start rule, another source; overlapping
/// or gapped blocks; a part drawn without --source-ordinals; a part whose
/// streams disagree with its count; a lone part; an existing destination.
#[test]
fn merge_deletions_refuses_mismatched_parts_and_broken_blocks() {
    let root = tmp("merge-refuse");
    let (gdir, cache) = fixture_dirs(&root, &[]);
    let pool = root.join("pool");
    write_fixture_split(&gdir, &pool, 12);
    let split = pool.join("screen");
    let pool2 = root.join("pool2");
    write_fixture_split_seeded(&gdir, &pool2, 12);
    let draw = |name: &str,
                src: &Path,
                cache: &Path,
                label: &str,
                per: &str,
                ord: Option<&str>|
     -> PathBuf {
        let dest = root.join(name);
        let mut a = vec![
            "--draw-label",
            label,
            "--allowed-range",
            "screen:0..",
            "--per-start",
            per,
            "--threads",
            "2",
        ];
        if let Some(o) = ord {
            a.extend_from_slice(&["--source-ordinals", o]);
        }
        let out = deletions(src, &gdir, cache, &dest, &a);
        assert!(
            out.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        dest
    };
    let a = draw("a", &split, &cache, "L", "rotate", Some("0..5"));
    let b = draw("b", &split, &cache, "L", "rotate", Some("6..11"));
    let ok_dest = root.join("ok");
    assert!(merge(&[&a, &b], &ok_dest).status.success());
    let bad_label = draw("b-label", &split, &cache, "M", "rotate", Some("6..11"));
    let bad_rule = draw("b-rule", &split, &cache, "L", "all", Some("6..11"));
    let bad_source = draw(
        "b-source",
        &pool2.join("screen"),
        &cache,
        "L",
        "rotate",
        Some("6..11"),
    );
    let overlap = draw("b-overlap", &split, &cache, "L", "rotate", Some("5..11"));
    let gap = draw("b-gap", &split, &cache, "L", "rotate", Some("7..11"));
    let no_ord = draw("b-noord", &split, &cache, "L", "rotate", None);
    // another cache state: the same rows plus one appended
    let cache2 = root.join("cache2");
    std::fs::create_dir_all(&cache2).unwrap();
    for f in ["vectors.jsonl", "manifest.json"] {
        std::fs::copy(cache.join(f), cache2.join(f)).unwrap();
    }
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .open(cache2.join("vectors.jsonl"))
        .unwrap();
    hf_embed::append_vector(&mut f, "not-a-node", &[0.5f64; 8]).unwrap();
    drop(f);
    let mut m2 = hf_embed::read_manifest(&cache2).unwrap();
    m2.count += 1;
    hf_embed::write_manifest(&cache2, &m2).unwrap();
    let bad_cache = draw("b-cache", &split, &cache2, "L", "rotate", Some("6..11"));
    // a part with one record cut from its labels stream
    let cut = root.join("b-cut");
    std::fs::create_dir_all(cut.join("queries")).unwrap();
    for f in [
        "visible.jsonl.gz",
        "visible_h.jsonl.gz",
        "queries/vectors.jsonl",
        "queries/manifest.json",
        "uncached_nodes.txt",
        "deletions.manifest.json",
    ] {
        std::fs::copy(b.join(f), cut.join(f)).unwrap();
    }
    {
        use std::io::Write;
        let recs = gz_records(&b.join("labels.jsonl.gz"));
        let mut gz = flate2::write::GzEncoder::new(
            std::fs::File::create(cut.join("labels.jsonl.gz")).unwrap(),
            flate2::Compression::default(),
        );
        for r in &recs[1..] {
            gz.write_all(&hf_core::canonical_bytes(r).unwrap()).unwrap();
            gz.write_all(b"\n").unwrap();
        }
        gz.finish().unwrap();
    }
    for (k, (other, needle)) in [
        (&bad_label, "draw_label"),
        (&bad_rule, "per_start"),
        (&bad_source, "source_visible_sha256"),
        (&bad_cache, "cache_state"),
        (&overlap, "overlap"),
        (&gap, "leave a gap"),
        (&no_ord, "without --source-ordinals"),
        (&cut, "labels.jsonl.gz holds"),
    ]
    .iter()
    .enumerate()
    {
        let dest = root.join(format!("refused{k}"));
        let out = merge(&[&a, other], &dest);
        assert_eq!(
            out.status.code(),
            Some(2),
            "{needle}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            String::from_utf8_lossy(&out.stderr).contains(needle),
            "{needle}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(!dest.exists(), "{needle}: something was written");
    }
    assert_eq!(merge(&[&a], &root.join("lone")).status.code(), Some(2));
    assert_eq!(
        merge(&[&a, &b], &ok_dest).status.code(),
        Some(2),
        "an existing destination"
    );
}

/// Another source split over the same graph (a larger ball).
fn write_fixture_split_seeded(gdir: &Path, dest: &Path, screen: usize) {
    let out = run(&[
        "write",
        "--family",
        "fixture",
        "--graph-dir",
        gdir.to_str().unwrap(),
        "--subgraph-size",
        "80",
        "--target-distance",
        "3",
        "--removal-level",
        "2",
        "--targets",
        "1",
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

/// The source stream is read selectively (what keeps a chunk's memory to its
/// own block — the whole TRAIN stream parsed was ~24.7 GB): a record outside
/// `--source-ordinals` is never parsed, so a malformed one there does not
/// stop the draw, while the same record inside the block does.
#[test]
fn records_outside_the_source_ordinals_are_not_parsed() {
    let root = tmp("selective");
    let (gdir, cache) = fixture_dirs(&root, &[]);
    let pool = root.join("pool");
    write_fixture_split(&gdir, &pool, 10);
    let split = pool.join("screen");
    doctor_visible(&split, |i, r| {
        if i == 0 {
            r["visible"]["nodes"] = "not a list".into();
        }
    });
    let base = [
        "--draw-label",
        "L",
        "--allowed-range",
        "screen:0..",
        "--threads",
        "1",
    ];
    let with = |ord: &str, dest: &Path| {
        let mut a = base.to_vec();
        a.extend_from_slice(&["--source-ordinals", ord]);
        deletions(&split, &gdir, &cache, dest, &a)
    };
    let out = with("5..9", &root.join("outside"));
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = with("0..4", &root.join("inside"));
    assert_eq!(out.status.code(), Some(2));
}
