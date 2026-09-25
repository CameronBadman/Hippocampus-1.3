//! `hf-splits deletions`: the R1 premise's delete-a-node split
//! (`experiments/real_walk_v2/R1_PREMISE_PLAN.md` §2 E1, §3), drawn from the
//! starts of ONE spent split directory.
//!
//! Reads the source's `manifest.public.json` and `visible.jsonl.gz` only —
//! the hidden stream and the private manifest are never opened — plus the
//! adapter graph and the node embedding cache. Writes, under a new
//! destination:
//!
//! - `visible.jsonl.gz`: per deletion, the start, the rebuilt ball's nodes and
//!   its induced edges (`r1_deletion_visible_v1`); no X, no labels.
//! - `visible_h.jsonl.gz`: the same ball with the start moved to `h`, the
//!   dense top-1 of the ball by cosine to X's row (the start-at-h variant).
//! - `labels.jsonl.gz`: T, L, the strata, the locality fields. A walk-side
//!   reader never opens it.
//! - `queries/`: an ordinary v5 embedding cache keyed by deletion-episode id,
//!   each row X's own row of the node cache — the query sidecar.
//! - `uncached_nodes.txt`: ball′ nodes the cache lacks (to embed, then re-run).
//! - `deletions.manifest.json`: provenance and every count.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};

use clap::Parser;
use hf_core::{refuse_holdout, HfError};
use hf_episodes::deletion::{self, Drawn, StartDraws};
use hf_episodes::{Sampler, SamplerConfig};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Parser, Debug)]
pub struct DeletionsArgs {
    /// ONE spent source split directory (its `screen` or `train` directory)
    #[arg(long)]
    split_dir: PathBuf,
    /// the adapter output the source was drawn from (edges.tsv, graph.manifest.json)
    #[arg(long)]
    graph_dir: PathBuf,
    /// the node embedding cache (X's row is the query; ball nodes are covered)
    #[arg(long)]
    embeddings: PathBuf,
    /// the new deletion split directory; refused when it exists
    #[arg(long)]
    destination: PathBuf,
    /// read only the first N source episodes (0 = every one)
    #[arg(long, default_value_t = 0)]
    limit: usize,
    /// worker threads (0 = rayon's default); the output does not depend on it
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// keep a deletion whose rebuilt ball holds a node the cache lacks
    /// (default: drop it, counted, and list the node in uncached_nodes.txt)
    #[arg(long)]
    keep_uncached: bool,
}

#[derive(Deserialize)]
struct VisibleLine {
    episode_id: String,
    visible: Value,
}

fn read_json(path: &Path) -> Result<Value, HfError> {
    serde_json::from_str(
        &std::fs::read_to_string(path)
            .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))?,
    )
    .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))
}

fn invalid(e: impl std::fmt::Display) -> HfError {
    HfError::Invalid(e.to_string())
}

struct Gz {
    inner: flate2::write::GzEncoder<std::fs::File>,
}

impl Gz {
    fn create(path: &Path) -> Result<Self, HfError> {
        let file = std::fs::File::create(path).map_err(invalid)?;
        Ok(Self {
            inner: flate2::write::GzEncoder::new(file, flate2::Compression::default()),
        })
    }

    fn record(&mut self, value: &Value) -> Result<(), HfError> {
        let bytes = hf_core::canonical_bytes(value)?;
        self.inner.write_all(&bytes).map_err(invalid)?;
        self.inner.write_all(b"\n").map_err(invalid)
    }

    fn finish(self) -> Result<(), HfError> {
        self.inner
            .finish()
            .map_err(invalid)?
            .sync_all()
            .map_err(invalid)
    }
}

/// The source's visible stream, checked against its public manifest's digest.
fn read_source(split_dir: &Path) -> Result<(Value, Vec<VisibleLine>), HfError> {
    let public = read_json(&split_dir.join("manifest.public.json"))?;
    if public.get("training_authorized") != Some(&Value::Bool(false)) {
        return Err(HfError::BandH(
            "the source manifest does not set training_authorized false".into(),
        ));
    }
    let path = split_dir.join("visible.jsonl.gz");
    let (bytes, sha) = hf_core::sha256_file(&path).map_err(invalid)?;
    if public.get("visible_bytes").and_then(Value::as_u64) != Some(bytes)
        || public.get("visible_sha256").and_then(Value::as_str) != Some(sha.as_str())
    {
        return Err(HfError::BandH(
            "visible.jsonl.gz does not match its public manifest digest".into(),
        ));
    }
    let raw = hf_io::read_maybe_gz(&path)?;
    let mut lines = Vec::new();
    for line in raw.split(|b| *b == b'\n') {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        lines.push(serde_json::from_slice::<VisibleLine>(line).map_err(invalid)?);
    }
    Ok((public, lines))
}

/// The source's sampler block as a config; an absent `greedy_share` (a split
/// written before amendment 6) reads as 1.0, as `prefix-check` reads it.
fn sampler_config(public: &Value) -> Result<SamplerConfig, HfError> {
    let mut block = public
        .get("sampler")
        .cloned()
        .ok_or_else(|| HfError::BandH("the source manifest has no sampler block".into()))?;
    if let Some(obj) = block.as_object_mut() {
        obj.entry("greedy_share").or_insert(1.0.into());
    }
    let config: SamplerConfig = serde_json::from_value(block).map_err(invalid)?;
    config.validate()?;
    Ok(config)
}

fn neighbours_value(drawn: &Drawn) -> Value {
    serde_json::to_value(&drawn.deletion.targets).unwrap_or(Value::Null)
}

pub fn deletions(args: DeletionsArgs) -> Result<(), HfError> {
    for path in [
        &args.split_dir,
        &args.graph_dir,
        &args.embeddings,
        &args.destination,
    ] {
        refuse_holdout(path)?;
    }
    if args.destination.exists() {
        return Err(HfError::Refused(format!(
            "{} exists; a deletion split is written once",
            args.destination.display()
        )));
    }
    let (public, source) = read_source(&args.split_dir)?;
    let take = if args.limit == 0 {
        source.len()
    } else {
        args.limit.min(source.len())
    };
    let source = &source[..take];
    // the reserve is refused before any graph is loaded or any file written
    for line in source {
        deletion::refuse_reserved(&line.episode_id)?;
    }
    let split = public
        .get("split")
        .and_then(Value::as_str)
        .ok_or_else(|| HfError::BandH("the source manifest names no split".into()))?
        .to_string();
    let family = public
        .get("family")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let config = sampler_config(&public)?;
    let graph_manifest = read_json(&args.graph_dir.join("graph.manifest.json"))?;
    if public.get("graph_manifest_sha256").and_then(Value::as_str)
        != Some(hf_core::canonical_sha256(&graph_manifest)?.as_str())
    {
        return Err(HfError::BandH(
            "the graph directory is not the one the source split was drawn from".into(),
        ));
    }
    let graph =
        hf_graph::RealGraph::from_triples(&args.graph_dir.join("edges.tsv"), &family, None)?;
    let embeddings = hf_embed::EmbeddingMatrix::load(&args.embeddings)?;
    let embedding_manifest = hf_embed::read_manifest(&args.embeddings)?;
    let in_index = deletion::InIndex::new(&graph);
    let sampler = Sampler::new(&graph, config.clone())?;
    let has_vector = |n: &str| embeddings.contains(n);
    let work = || -> Vec<Result<StartDraws, HfError>> {
        use rayon::prelude::*;
        source
            .par_iter()
            .map(|line| {
                let start = line.visible["start_node"]
                    .as_str()
                    .ok_or_else(|| HfError::BandH(format!("{}: no start", line.episode_id)))?;
                let stored: Vec<String> = line.visible["nodes"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|n| n["node"].as_str().unwrap_or("").to_string())
                    .collect();
                deletion::draw_for_start(
                    &graph,
                    &sampler,
                    &split,
                    &line.episode_id,
                    start,
                    &stored,
                    &has_vector,
                    &in_index,
                )
            })
            .collect()
    };
    let results = if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build()
            .map_err(invalid)?
            .install(work)
    } else {
        work()
    };
    std::fs::create_dir_all(&args.destination).map_err(invalid)?;
    let mut visible = Gz::create(&args.destination.join("visible.jsonl.gz"))?;
    let mut visible_h = Gz::create(&args.destination.join("visible_h.jsonl.gz"))?;
    let mut labels = Gz::create(&args.destination.join("labels.jsonl.gz"))?;
    let queries_dir = args.destination.join("queries");
    std::fs::create_dir_all(&queries_dir).map_err(invalid)?;
    let mut queries = std::fs::File::create(queries_dir.join("vectors.jsonl")).map_err(invalid)?;
    let mut uncached: BTreeSet<String> = BTreeSet::new();
    let mut per_tag: BTreeMap<&str, u64> = BTreeMap::new();
    let mut drops: BTreeMap<&str, u64> = BTreeMap::new();
    let (mut candidates, mut without_neighbour, mut without_vector) = (0u64, 0u64, 0u64);
    let mut start_isolated = 0u64;
    let mut written = 0u64;
    for (line, result) in source.iter().zip(results) {
        let draws = result?;
        candidates += draws.candidates as u64;
        without_neighbour += draws.without_neighbour as u64;
        without_vector += draws.without_vector as u64;
        start_isolated += draws.start_isolated as u64;
        if draws.drawn.is_empty() {
            *drops.entry("no_candidate").or_default() += 1;
        }
        for drawn in &draws.drawn {
            let d = &drawn.deletion;
            let x_name = graph.name(d.deleted).to_string();
            let missing: Vec<String> = d
                .ball
                .iter()
                .map(|n| graph.name(*n).to_string())
                .filter(|n| !embeddings.contains(n))
                .collect();
            if !missing.is_empty() && !args.keep_uncached {
                uncached.extend(missing);
                *drops.entry("uncached_in_ball").or_default() += 1;
                continue;
            }
            let episode_id = format!("{}-del-{}", line.episode_id, drawn.draws[0]);
            let (neighbours_graph, truncated) =
                deletion::graph_neighbours(&graph, &in_index, d.deleted);
            let start = d.ball[0];
            let payload = deletion::visible_payload(&graph, &family, start, &d.ball);
            let q: Vec<f64> = embeddings
                .get(&x_name)
                .ok_or_else(|| HfError::BandH(format!("{x_name}: no vector")))?
                .iter()
                .map(|v| *v as f64)
                .collect();
            let vector = |n: &str| {
                embeddings
                    .get(n)
                    .map(|v| v.iter().map(|x| *x as f64).collect())
            };
            let h = deletion::dense_top1(&graph, &d.ball, &q, &vector)
                .ok_or_else(|| HfError::BandH(format!("{episode_id}: no node has a vector")))?;
            let payload_h = deletion::visible_payload(&graph, &family, h, &d.ball);
            // X never reaches a visible stream, the id included
            for p in [&payload, &payload_h] {
                if deletion::mentions(&episode_id, p).contains(&x_name) {
                    return Err(HfError::BandH(format!(
                        "{episode_id}: the deleted node reached the visible payload"
                    )));
                }
            }
            visible.record(&json!({"episode_id": episode_id, "visible": payload}))?;
            visible_h.record(&json!({"episode_id": episode_id, "visible": payload_h}))?;
            labels.record(&json!({
                "episode_id": episode_id,
                "record_kind": "r1_deletion_labels_v1",
                "source_episode_id": line.episode_id,
                "deleted_node": x_name,
                "draws": drawn.draws,
                "targets": neighbours_value(drawn),
                "target_count": d.targets.len(),
                "stratum": deletion::stratum(d.targets.len()),
                "start_is_neighbour": d.start_is_neighbour,
                "left_with_x": d.left,
                "degree_graph": drawn.degree_graph,
                "neighbours_graph": neighbours_graph,
                "neighbours_graph_truncated": truncated,
                "stratum_degree": deletion::stratum(drawn.degree_graph),
                "bfs_parent": drawn.bfs_parent,
                "d_ball_start_x": drawn.d_ball,
                "ball_size": d.ball.len(),
                "entered": d.entered,
                "departed": d.departed,
                "h_node": graph.name(h),
                "uncached_in_ball": d.ball.iter().filter(|n| !embeddings.contains(graph.name(**n))).count(),
            }))?;
            hf_embed::append_vector(&mut queries, &episode_id, &q).map_err(invalid)?;
            for tag in &drawn.draws {
                *per_tag.entry(tag).or_default() += 1;
            }
            written += 1;
        }
    }
    visible.finish()?;
    visible_h.finish()?;
    labels.finish()?;
    queries.sync_all().map_err(invalid)?;
    drop(queries);
    let mut query_manifest = embedding_manifest.clone();
    query_manifest.count = written;
    query_manifest.truncated = Default::default();
    query_manifest.extra.insert(
        "query_kind".into(),
        "r1_deleted_node_row: each id is a deletion episode, each vector X's own row of the node cache".into(),
    );
    hf_embed::write_manifest(&queries_dir, &query_manifest)?;
    let mut list = String::new();
    for n in &uncached {
        list.push_str(n);
        list.push('\n');
    }
    std::fs::write(args.destination.join("uncached_nodes.txt"), list).map_err(invalid)?;
    let manifest = json!({
        "record_kind": "r1_deletion_split_manifest_v1",
        "engine": "hippo-13 hf-splits deletions",
        "governed_by": "experiments/real_walk_v2/R1_PREMISE_PLAN.md",
        "source_split_dir": args.split_dir.to_string_lossy(),
        "source_split": split,
        "source_visible_sha256": public.get("visible_sha256").cloned().unwrap_or(Value::Null),
        "source_episode_count": public.get("episode_count").cloned().unwrap_or(Value::Null),
        "source_episodes_read": source.len(),
        "streams_opened": ["manifest.public.json", "visible.jsonl.gz"],
        "sampler": serde_json::to_value(&config).map_err(invalid)?,
        "hub_fanout": deletion::HUB_FANOUT,
        "graph_manifest_sha256": public.get("graph_manifest_sha256").cloned().unwrap_or(Value::Null),
        "embeddings": args.embeddings.to_string_lossy(),
        "embedding_model": embedding_manifest.model,
        "embedding_model_digest": embedding_manifest.model_digest,
        "draw_label": deletion::DRAW_LABEL,
        "draw_tags": deletion::DRAW_TAGS,
        "strata_on": "target_count (|T|); stratum_degree beside, same cuts",
        "reserved": {"train_from": deletion::RESERVED_TRAIN_FROM, "screen_from": deletion::RESERVED_SCREEN_FROM},
        "records_written": written,
        "records_per_draw_tag": per_tag,
        "drops": drops,
        "candidates_with_neighbour": candidates,
        "candidates_without_neighbour": without_neighbour,
        "candidates_without_vector": without_vector,
        "candidates_isolating_the_start": start_isolated,
        "uncached_nodes": uncached.len(),
        "keep_uncached": args.keep_uncached,
        "training_authorized": false,
    });
    std::fs::write(
        args.destination.join("deletions.manifest.json"),
        hf_core::files::python_json_pretty(&manifest),
    )
    .map_err(invalid)?;
    println!(
        "deletions: {written} records from {} starts, drops {drops:?}, {} uncached nodes -> {}",
        source.len(),
        uncached.len(),
        args.destination.display()
    );
    Ok(())
}
