//! `hf-stage0 --deletion-probe DIR`: the R1 premise's walk probe
//! (`experiments/real_walk_v2/R1_PREMISE_PLAN.md` §2 E2, §5 item 5).
//!
//! A trained checkpoint walks every record of a deletion split written by
//! `hf-splits deletions`: the ball rebuilt without the deleted node X, with
//! X's own embedding row (the `queries/` sidecar) as the query, under
//! `QuerySource::DeletedPayload` — no target is shown or registered, so the
//! walk runs until its frontier empties or it reaches `--max-expansions`
//! (the start counted). One row per record carries the expansion order; the
//! reader derives every examined set `E_B` from it and the visible edges.
//!
//! What it opens: the config, the checkpoint, the node cache, the split's
//! `deletions.manifest.json`, ONE visible stream (`visible.jsonl.gz`, or
//! `visible_h.jsonl.gz` for the start-at-h variant) and `queries/`. It never
//! opens `labels.jsonl.gz`; the CLI test runs it with that file deleted.
//! It always runs on the CPU, whatever GPU is present (the plan's cost is in
//! CPU hours, and a probe must not take a GPU from a training run).

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use hf_core::HfError;
use hf_model::{Model, ModelConfig, ModelScorer};
use hf_walk::{walk_batch_capped, EpisodeIndex, StopRule, WalkOptions};
use serde_json::{json, Value};
use tch::Device;

use crate::{data, eval::EVAL_BATCH};

pub const STREAMS: [&str; 2] = ["visible.jsonl.gz", "visible_h.jsonl.gz"];

/// One deletion record's visible payload.
struct Record {
    episode_id: String,
    start: String,
    nodes: Vec<String>,
    edges: Vec<(u32, String, String)>,
}

fn invalid(e: impl std::fmt::Display) -> HfError {
    HfError::Invalid(e.to_string())
}

fn read_records(path: &Path, limit: usize) -> Result<Vec<Record>, HfError> {
    let raw = hf_io::read_maybe_gz(path)?;
    let mut out = Vec::new();
    for line in raw.split(|b| *b == b'\n') {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        if limit > 0 && out.len() >= limit {
            break;
        }
        let v: Value = serde_json::from_slice(line).map_err(invalid)?;
        let visible = &v["visible"];
        if visible["record_kind"] != "r1_deletion_visible_v1" {
            return Err(HfError::BandH(format!(
                "{}: not a deletion record ({})",
                path.display(),
                visible["record_kind"]
            )));
        }
        let text = |x: &Value| x.as_str().unwrap_or("").to_string();
        out.push(Record {
            episode_id: text(&v["episode_id"]),
            start: text(&visible["start_node"]),
            nodes: visible["nodes"]
                .as_array()
                .into_iter()
                .flatten()
                .map(text)
                .collect(),
            edges: visible["edges"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|e| {
                    (
                        e["edge_id"].as_u64().unwrap_or(0) as u32,
                        text(&e["source"]),
                        text(&e["target"]),
                    )
                })
                .collect(),
        });
    }
    Ok(out)
}

pub struct ProbeArgs<'a> {
    pub split_dir: &'a Path,
    pub stream: &'a str,
    pub config_path: &'a Path,
    /// the checkpoint's `.safetensors` weights, resolved by the caller
    pub checkpoint: &'a Path,
    pub embeddings_dir: &'a Path,
    pub output: &'a Path,
    pub max_expansions: usize,
    pub limit: usize,
    pub model_seed: u64,
    pub provenance: Value,
    pub evidence: bool,
}

pub fn run(a: ProbeArgs<'_>) -> Result<(), HfError> {
    for p in [
        a.split_dir,
        a.config_path,
        a.checkpoint,
        a.embeddings_dir,
        a.output,
    ] {
        hf_core::refuse_holdout(p)?;
    }
    if !STREAMS.contains(&a.stream) {
        return Err(HfError::Invalid(format!(
            "--probe-stream is one of {STREAMS:?}, not {:?}",
            a.stream
        )));
    }
    if a.max_expansions == 0 {
        return Err(HfError::Invalid(
            "--max-expansions counts the start; at least 1".into(),
        ));
    }
    let rows_path = a.output.join("probe_rows.jsonl");
    if rows_path.exists() {
        return Err(HfError::Refused(format!(
            "{} exists; a probe writes its rows once",
            rows_path.display()
        )));
    }
    let split_manifest = data::read_json(&a.split_dir.join("deletions.manifest.json"))?;
    if split_manifest["record_kind"] != "r1_deletion_split_manifest_v1"
        || split_manifest["training_authorized"] != Value::Bool(false)
    {
        return Err(HfError::BandH(
            "the probe reads a deletion split written by hf-splits deletions".into(),
        ));
    }
    let config = data::read_json(a.config_path)?;
    if config.get("training_authorized") != Some(&Value::Bool(false)) {
        return Err(HfError::Refused(
            "a stage 0 config must state training_authorized: false".into(),
        ));
    }
    let embeddings = hf_embed::EmbeddingMatrix::load(a.embeddings_dir)?;
    let node_manifest = hf_embed::read_manifest(a.embeddings_dir)?;
    let queries_dir = a.split_dir.join("queries");
    let query_manifest = hf_embed::read_manifest(&queries_dir)?;
    hf_embed::same_encoder(&node_manifest, &query_manifest)?;
    let queries = hf_embed::EmbeddingMatrix::load(&queries_dir)?;
    let dim = embeddings.dimension;
    tch::manual_seed(a.model_seed as i64);
    let model_config = ModelConfig::from_value(&config["model"], dim as i64)?;
    let mut model = Model::new(model_config, Device::Cpu)?;
    let capacity = model.trainable_parameter_count();
    if let Some(stated) = config["model"]
        .get("stated_capacity")
        .and_then(|s| s.get(dim.to_string()))
    {
        if stated.as_i64() != Some(capacity) {
            return Err(HfError::BandH(format!(
                "capacity {capacity} != stated {stated} at dimension {dim}"
            )));
        }
    }
    let weights = a.checkpoint.to_path_buf();
    model
        .vs
        .load(&weights)
        .map_err(|e| HfError::Invalid(format!("{}: {e}", weights.display())))?;
    let (_, weights_sha256) = hf_core::sha256_file(&weights).map_err(invalid)?;
    let features = model.features();
    let with_prior = model.config.greedy_prior;
    let records = read_records(&a.split_dir.join(a.stream), a.limit)?;
    if records.is_empty() {
        return Err(HfError::Invalid(
            "the deletion stream holds no record".into(),
        ));
    }
    std::fs::create_dir_all(a.output).map_err(invalid)?;
    let mut file = std::fs::File::create(&rows_path).map_err(invalid)?;
    let started = Instant::now();
    let mut stop_reasons: std::collections::BTreeMap<String, u64> = Default::default();
    for chunk in records.chunks(EVAL_BATCH) {
        let indexes: Vec<EpisodeIndex> = chunk
            .iter()
            .map(|r| {
                let q = queries.get(&r.episode_id).ok_or_else(|| {
                    HfError::BandH(format!(
                        "the query sidecar holds no row for {}",
                        r.episode_id
                    ))
                })?;
                EpisodeIndex::for_deletion(
                    &r.episode_id,
                    &r.start,
                    &r.nodes,
                    &r.edges,
                    &embeddings,
                    dim,
                    q,
                )
            })
            .collect::<Result<_, _>>()?;
        let refs: Vec<&EpisodeIndex> = indexes.iter().collect();
        let mut scorer = ModelScorer { model: &model };
        let walked = walk_batch_capped(
            &refs,
            features.as_ref(),
            &mut scorer,
            WalkOptions {
                stop_rule: StopRule::Exhaust,
                record_candidates: false,
                keep_items: false,
                with_prior,
            },
            Some(a.max_expansions),
        )?;
        for ((r, index), w) in chunk.iter().zip(&indexes).zip(&walked) {
            if w.registered_at.is_some() {
                return Err(HfError::BandH(format!(
                    "{}: a deletion probe registered a target",
                    r.episode_id
                )));
            }
            *stop_reasons.entry(w.stop_reason.to_string()).or_default() += 1;
            let row = json!({
                "episode_id": r.episode_id,
                "stream": a.stream,
                "start_node": r.start,
                "walk_expanded": w.expanded.iter().map(|n| index.names[*n as usize].as_str()).collect::<Vec<_>>(),
                "expansions": w.expansions(),
                "examined_count": w.examined,
                "stop_reason": w.stop_reason,
                "max_expansions": a.max_expansions,
            });
            writeln!(file, "{}", serde_json::to_string(&row).map_err(invalid)?).map_err(invalid)?;
        }
    }
    file.sync_all().map_err(invalid)?;
    let seconds = started.elapsed().as_secs_f64();
    let manifest = json!({
        "record_kind": "r1_deletion_probe_manifest_v1",
        "governed_by": "experiments/real_walk_v2/R1_PREMISE_PLAN.md",
        "engine": "hippo-13 hf-stage0 --deletion-probe",
        "evidence": a.evidence,
        "split_dir": a.split_dir.to_string_lossy(),
        "stream": a.stream,
        "streams_opened": [a.stream, "queries/", "deletions.manifest.json"],
        "labels_opened": false,
        "split_manifest_records": split_manifest["records_written"],
        "records_walked": records.len(),
        "query_source": "deleted_payload",
        "stop_rule": "exhaust",
        "max_expansions": a.max_expansions,
        "stop_reasons": stop_reasons,
        "device": "cpu",
        "seconds": seconds,
        "seconds_per_record": seconds / records.len() as f64,
        "checkpoint": weights.to_string_lossy(),
        "checkpoint_sha256": weights_sha256,
        "config": a.config_path.to_string_lossy(),
        "feature_set": features.name(),
        "greedy_prior": with_prior,
        "capacity": capacity,
        "model_seed": a.model_seed,
        "embeddings": a.embeddings_dir.to_string_lossy(),
        "embedding_model_digest": node_manifest.model_digest,
        "provenance": a.provenance,
        "training_authorized": false,
    });
    std::fs::write(
        a.output.join("probe.manifest.json"),
        hf_core::files::python_json_pretty(&manifest),
    )
    .map_err(invalid)?;
    println!(
        "deletion probe: {} records in {seconds:.1}s on the CPU, stops {stop_reasons:?} -> {}",
        records.len(),
        rows_path.display()
    );
    Ok(())
}
