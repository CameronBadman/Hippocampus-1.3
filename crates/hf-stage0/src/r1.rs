//! The R1 return/stop head's runner paths (`experiments/real_walk_v2/
//! R1_HEAD_DESIGN.md` §8 ENG-4 and ENG-7):
//!
//! - `--r1-cache DELDIR --init-from CKPT`: the frozen trunk walks every
//!   deletion record to the cap once, under `no_grad`, and the f32 cache is
//!   written (`hf_model::r1::CacheWriter`). It opens the split's visible
//!   stream, `queries/` and its manifest — never `labels.jsonl.gz`.
//! - `--r1-train CACHE --init-from CKPT`: the two heads train from the cache
//!   ONLY (it never walks); `T` comes from the deletion split's labels and is
//!   held by the loss alone.
//! - `--r1-eval CACHE --reevaluate-checkpoint R1CKPT`: per-candidate return
//!   logits and the rstop logits, from the cache; never opens labels, and
//!   writes logits, not a returned set or a stop.
//! - `--frozen-check R1CKPT --init-from CKPT`: every frozen parameter's digest
//!   against the init checkpoint's (ENG-2 (c) on a real checkpoint).
//!
//! Every path refuses a cache whose manifest disagrees with the config's `r1`
//! block, the init checkpoint's trunk digest, this binary's engine head or
//! the deletion split's visible sha.

use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use hf_core::{HfError, PyRandom};
use hf_model::r1::{self, R1Cache, R1Spec};
use hf_model::{AdamW, Model, ModelConfig, R1_HEAD_PATTERNS};
use hf_walk::EpisodeIndex;
use serde_json::{json, Value};
use tch::Device;

use crate::data;

fn invalid(e: impl std::fmt::Display) -> HfError {
    HfError::Invalid(e.to_string())
}

/// The config's `r1` block.
pub struct R1Config {
    pub spec: R1Spec,
    pub draw_label: String,
    pub allowed_ranges: Vec<String>,
    pub init_sha256: Option<String>,
    pub cache_batch: usize,
}

pub fn r1_config(config: &Value) -> Result<R1Config, HfError> {
    let block = config
        .get("r1")
        .ok_or_else(|| HfError::Refused("an R1 path needs the config's r1 block".into()))?;
    let cap = block["max_expansions"]
        .as_u64()
        .ok_or_else(|| HfError::Invalid("r1.max_expansions".into()))? as usize;
    let snapshots: Vec<usize> = block["snapshots"]
        .as_array()
        .ok_or_else(|| HfError::Invalid("r1.snapshots".into()))?
        .iter()
        .map(|v| v.as_u64().map(|x| x as usize))
        .collect::<Option<_>>()
        .ok_or_else(|| HfError::Invalid("r1.snapshots must be integers".into()))?;
    let spec = R1Spec { cap, snapshots };
    spec.validate()?;
    let draw_label = block["draw_label"]
        .as_str()
        .ok_or_else(|| HfError::Invalid("r1.draw_label".into()))?
        .to_string();
    let allowed_ranges: Vec<String> = block["allowed_ranges"]
        .as_array()
        .ok_or_else(|| HfError::Invalid("r1.allowed_ranges".into()))?
        .iter()
        .map(|v| v.as_str().map(str::to_string))
        .collect::<Option<_>>()
        .ok_or_else(|| HfError::Invalid("r1.allowed_ranges must be strings".into()))?;
    for r in &allowed_ranges {
        hf_episodes::deletion::AllowedRange::parse(r)?;
    }
    Ok(R1Config {
        spec,
        draw_label,
        allowed_ranges,
        init_sha256: block["init_from"]["sha256"].as_str().map(str::to_string),
        cache_batch: block["cache_batch"].as_u64().unwrap_or(64) as usize,
    })
}

/// The R1 model: the config's model block with `return_head: true`.
pub fn r1_model(config: &Value, dim: i64, device: Device) -> Result<(Model, ModelConfig), HfError> {
    let mc = ModelConfig::from_value(&config["model"], dim)?;
    if !mc.return_head {
        return Err(HfError::Refused(
            "an R1 path needs model.return_head: true in the config".into(),
        ));
    }
    if mc.feature_set == "raw-v5" {
        return Err(HfError::Refused(
            "the R1 head reads a relational feature set, not raw-v5".into(),
        ));
    }
    let model = Model::new(mc.clone(), device)?;
    check_capacity(config, &model, dim)?;
    Ok((model, mc))
}

/// `model.stated_capacity` at `dim`, when the config states it, against the
/// model's whole parameter count (heads included).
fn check_capacity(config: &Value, model: &Model, dim: i64) -> Result<(), HfError> {
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
    Ok(())
}

/// The trunk digest: every parameter but the R1 heads'.
pub fn trunk_digest(model: &Model) -> String {
    let heads = model.names_matching(&R1_HEAD_PATTERNS);
    let refs: Vec<&str> = heads.iter().map(String::as_str).collect();
    model.state_digest(&refs)
}

/// Load the init checkpoint (a stage-0, head-less one) into the R1 model:
/// every checkpoint tensor must be a model parameter of the same shape, and
/// the only parameters it may lack are the heads' (ENG-2 `--init-from`).
pub fn load_init(model: &mut Model, init: &Path) -> Result<String, HfError> {
    hf_core::refuse_holdout(init)?;
    let weights = weights_of(init)?;
    model.load_strict(&weights, &R1_HEAD_PATTERNS, &[])?;
    let (_, sha) = hf_core::sha256_file(&weights).map_err(invalid)?;
    Ok(sha)
}

/// A checkpoint named by its `.safetensors`, `.json` or directory.
pub fn weights_of(path: &Path) -> Result<PathBuf, HfError> {
    let p = if path.is_dir() {
        path.join("checkpoint.safetensors")
    } else if path.extension().and_then(|e| e.to_str()) == Some("json") {
        path.with_extension("safetensors")
    } else {
        path.to_path_buf()
    };
    if !p.exists() {
        return Err(HfError::Invalid(format!(
            "{}: no such checkpoint",
            p.display()
        )));
    }
    Ok(p)
}

pub struct Provenance {
    pub value: Value,
    pub evidence: bool,
}

/// What every R1 artifact records about the engine.
fn engine_fields() -> Value {
    json!({
        "engine_head": data::engine_head(),
        "engine_build_head": data::ENGINE_BUILD_HEAD,
        "engine_build_dirty": data::engine_build_dirty(),
        "engine_binary_sha256": data::engine_binary_sha256(),
    })
}

fn refuse_existing(path: &Path) -> Result<(), HfError> {
    if path.exists() {
        return Err(HfError::Refused(format!(
            "{} exists; refusing to write over it",
            path.display()
        )));
    }
    Ok(())
}

// --------------------------------------------------------------------------
// --r1-cache

pub struct CacheArgs<'a> {
    pub deletion_dir: &'a Path,
    pub config: &'a Value,
    pub config_path: &'a Path,
    pub init: &'a Path,
    pub embeddings_dir: &'a Path,
    pub output: &'a Path,
    pub model_seed: u64,
    pub device: Device,
    pub deterministic: bool,
    pub provenance: Provenance,
}

pub fn build_cache(a: CacheArgs<'_>) -> Result<(), HfError> {
    for p in [
        a.deletion_dir,
        a.init,
        a.embeddings_dir,
        a.output,
        a.config_path,
    ] {
        hf_core::refuse_holdout(p)?;
    }
    refuse_existing(a.output)?;
    let r1c = r1_config(a.config)?;
    let split_manifest = data::read_json(&a.deletion_dir.join("deletions.manifest.json"))?;
    if split_manifest["record_kind"] != "r1_deletion_split_manifest_v1"
        || split_manifest["training_authorized"] != Value::Bool(false)
    {
        return Err(HfError::BandH(
            "--r1-cache reads a deletion split written by hf-splits deletions".into(),
        ));
    }
    let (split_ranges, ranges_source) = deletion_ranges(&split_manifest)?;
    if split_manifest["draw_label"].as_str() != Some(r1c.draw_label.as_str())
        || split_ranges != r1c.allowed_ranges
    {
        return Err(HfError::BandH(format!(
            "the deletion split was drawn with label {} and ranges {split_ranges:?}, the \
             config's r1 block declares {:?} and {:?}",
            split_manifest["draw_label"], r1c.draw_label, r1c.allowed_ranges
        )));
    }
    let visible = a.deletion_dir.join("visible.jsonl.gz");
    let (_, visible_sha) = hf_core::sha256_file(&visible).map_err(invalid)?;
    let embeddings = hf_embed::EmbeddingMatrix::load(a.embeddings_dir)?;
    let node_manifest = hf_embed::read_manifest(a.embeddings_dir)?;
    let queries_dir = a.deletion_dir.join("queries");
    let query_manifest = hf_embed::read_manifest(&queries_dir)?;
    hf_embed::same_encoder(&node_manifest, &query_manifest)?;
    let queries = hf_embed::EmbeddingMatrix::load(&queries_dir)?;
    let dim = embeddings.dimension;
    tch::manual_seed(a.model_seed as i64);
    let (mut model, mc) = r1_model(a.config, dim as i64, a.device)?;
    let init_sha = load_init(&mut model, a.init)?;
    if let Some(want) = &r1c.init_sha256 {
        if *want != init_sha {
            return Err(HfError::BandH(format!(
                "--init-from is {init_sha}, the config's r1.init_from.sha256 is {want}"
            )));
        }
    }
    let trunk = trunk_digest(&model);
    let records = r1::read_deletion_visible(&visible, 0)?;
    if records.is_empty() {
        return Err(HfError::Invalid(
            "the deletion stream holds no record".into(),
        ));
    }
    let spec = DrawCheck::new(&r1c)?;
    for r in &records {
        spec.check(&r.episode_id)?;
    }
    let mut writer = r1::CacheWriter::create(a.output, model.hidden_dimension() as usize)?;
    let started = Instant::now();
    let mut stop_reasons: BTreeMap<String, u64> = BTreeMap::new();
    for chunk in records.chunks(r1c.cache_batch.max(1)) {
        let indexes = deletion_indexes(chunk, &queries, &embeddings, dim)?;
        let refs: Vec<&EpisodeIndex> = indexes.iter().collect();
        for rec in r1::compute_features(&model, &refs, &r1c.spec)? {
            *stop_reasons.entry(rec.stop_reason.clone()).or_default() += 1;
            writer.append(&rec)?;
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    let manifest = writer.finish(json!({
        "engine": "hippo-13 hf-stage0 --r1-cache",
        "evidence": a.provenance.evidence,
        "provenance": a.provenance.value,
        "engine_build": engine_fields(),
        "deletion_split": a.deletion_dir.to_string_lossy(),
        "deletion_visible_sha256": visible_sha,
        "deletion_draw_label": split_manifest["draw_label"],
        "deletion_allowed_ranges": split_ranges,
        "deletion_ranges_source": ranges_source,
        "deletion_per_start": split_manifest["per_start"],
        "deletion_records_written": split_manifest["records_written"],
        "streams_opened": ["visible.jsonl.gz", "queries/", "deletions.manifest.json"],
        "init_from": a.init.to_string_lossy(),
        "init_sha256": init_sha,
        "trunk_state_digest": trunk,
        "config": a.config_path.to_string_lossy(),
        "feature_set": mc.feature_set,
        "greedy_prior": mc.greedy_prior,
        "model_seed": a.model_seed,
        "max_expansions": r1c.spec.cap,
        "snapshots": r1c.spec.snapshots,
        "stop_snapshots": r1c.spec.stop_at(),
        "batch_size": r1c.cache_batch,
        "device": if a.device == Device::Cpu { "cpu" } else { "cuda" },
        "deterministic": a.deterministic,
        "stop_reasons": stop_reasons,
        "embeddings": a.embeddings_dir.to_string_lossy(),
        "embedding_model_digest": node_manifest.model_digest,
        "embedding_dimension": dim,
        "seconds": seconds,
        "seconds_per_record": seconds / records.len() as f64,
    }))?;
    println!(
        "r1 cache: {} records in {seconds:.1}s -> {}",
        manifest["records"],
        a.output.display()
    );
    Ok(())
}

/// The raw-index ranges a deletion split was drawn under, and where they
/// were read: `allowed_ranges` when its manifest records them (ENG-1); for a
/// manifest the premise's engine wrote (b8b074b), which records `reserved`
/// instead, the premise's own ranges `train:0..73359, screen:0..4104` —
/// derived ONLY when its label and its reserve are exactly the premise's
/// (SMOKE-R reads the spent P-dev split). Any other shape is band H.
pub fn deletion_ranges(m: &Value) -> Result<(Vec<String>, &'static str), HfError> {
    use hf_episodes::deletion::{DrawSpec, RESERVED_SCREEN_FROM, RESERVED_TRAIN_FROM};
    if let Some(list) = m.get("allowed_ranges") {
        let ranges: Vec<String> = list
            .as_array()
            .ok_or_else(|| HfError::BandH("allowed_ranges is not a list".into()))?
            .iter()
            .map(|v| v.as_str().map(str::to_string))
            .collect::<Option<_>>()
            .ok_or_else(|| HfError::BandH("allowed_ranges holds a non-string".into()))?;
        return Ok((ranges, "allowed_ranges"));
    }
    let premise = DrawSpec::premise();
    let reserve = json!({"train_from": RESERVED_TRAIN_FROM, "screen_from": RESERVED_SCREEN_FROM});
    if m["draw_label"].as_str() == Some(premise.label.as_str()) && m["reserved"] == reserve {
        let ranges = premise
            .ranges
            .iter()
            .map(hf_episodes::deletion::AllowedRange::label)
            .collect();
        return Ok((ranges, "premise_reserve"));
    }
    Err(HfError::BandH(format!(
        "the deletion manifest records neither allowed_ranges nor the premise's own label and \
         reserve (label {}, reserved {})",
        m["draw_label"], m["reserved"]
    )))
}

/// The deletion-probe index of each record, on X's own row (the sidecar).
pub fn deletion_indexes(
    chunk: &[r1::DeletionVisible],
    queries: &hf_embed::EmbeddingMatrix,
    embeddings: &hf_embed::EmbeddingMatrix,
    dim: usize,
) -> Result<Vec<EpisodeIndex>, HfError> {
    chunk
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
                embeddings,
                dim,
                q,
            )
        })
        .collect()
}

/// The config's declared ranges, as the deletion drawer checks them.
struct DrawCheck(hf_episodes::deletion::DrawSpec);

impl DrawCheck {
    fn new(r1c: &R1Config) -> Result<Self, HfError> {
        Ok(Self(hf_episodes::deletion::DrawSpec {
            label: r1c.draw_label.clone(),
            ranges: r1c
                .allowed_ranges
                .iter()
                .map(|r| hf_episodes::deletion::AllowedRange::parse(r))
                .collect::<Result<_, _>>()?,
        }))
    }

    /// The source id is the deletion id before its `-del-<tag>` suffix.
    fn check(&self, episode_id: &str) -> Result<(), HfError> {
        self.0.check(episode_id)
    }
}

/// A cache opened for training or evaluation, checked against the config,
/// the trunk, this engine and the deletion split.
fn open_checked(cache_dir: &Path, r1c: &R1Config, trunk: &str) -> Result<R1Cache, HfError> {
    let cache = R1Cache::open(cache_dir)?;
    let m = &cache.manifest;
    let mismatch = |what: &str, got: &Value, want: Value| -> Result<(), HfError> {
        if *got != want {
            return Err(HfError::BandH(format!(
                "the cache's {what} is {got}, expected {want}"
            )));
        }
        Ok(())
    };
    mismatch("max_expansions", &m["max_expansions"], json!(r1c.spec.cap))?;
    mismatch("snapshots", &m["snapshots"], json!(r1c.spec.snapshots))?;
    mismatch(
        "deletion_draw_label",
        &m["deletion_draw_label"],
        json!(r1c.draw_label),
    )?;
    mismatch(
        "deletion_allowed_ranges",
        &m["deletion_allowed_ranges"],
        json!(r1c.allowed_ranges),
    )?;
    mismatch("trunk_state_digest", &m["trunk_state_digest"], json!(trunk))?;
    mismatch(
        "engine_build_head",
        &m["engine_build"]["engine_build_head"],
        json!(data::ENGINE_BUILD_HEAD),
    )?;
    let split = PathBuf::from(m["deletion_split"].as_str().unwrap_or(""));
    hf_core::refuse_holdout(&split)?;
    let (_, visible_sha) = hf_core::sha256_file(&split.join("visible.jsonl.gz"))
        .map_err(|e| HfError::BandH(format!("the cache's deletion split: {e}")))?;
    mismatch(
        "deletion_visible_sha256",
        &m["deletion_visible_sha256"],
        json!(visible_sha),
    )?;
    if let Some(want) = &r1c.init_sha256 {
        mismatch("init_sha256", &m["init_sha256"], json!(want))?;
    }
    let check = DrawCheck::new(r1c)?;
    for i in 0..cache.len() {
        check.check(cache.episode_id(i))?;
    }
    Ok(cache)
}

// --------------------------------------------------------------------------
// --r1-cache-verify

pub struct VerifyArgs<'a> {
    pub cache_dir: &'a Path,
    pub config: &'a Value,
    pub init: &'a Path,
    pub embeddings_dir: &'a Path,
    pub output: &'a Path,
    pub device: Device,
    pub provenance: Provenance,
}

/// ENG-7 (a) as an operator line (SMOKE-R): recompute every record of the
/// cache on the fly, on THIS device, with the cache's own batching (its
/// manifest's `batch_size`, in its record order), and compare every value
/// bit for bit. Writes `cache_verify.json`; a mismatch is band H.
pub fn verify_cache(a: VerifyArgs<'_>) -> Result<bool, HfError> {
    for p in [a.cache_dir, a.init, a.embeddings_dir, a.output] {
        hf_core::refuse_holdout(p)?;
    }
    let out = a.output.join("cache_verify.json");
    refuse_existing(&out)?;
    let r1c = r1_config(a.config)?;
    let dim = embedding_dimension_of(a.cache_dir)?;
    let (mut model, _) = r1_model(a.config, dim, a.device)?;
    let init_sha = load_init(&mut model, a.init)?;
    let cache = open_checked(a.cache_dir, &r1c, &trunk_digest(&model))?;
    let m = cache.manifest.clone();
    if m["init_sha256"].as_str() != Some(init_sha.as_str()) {
        return Err(HfError::BandH(format!(
            "--init-from is {init_sha}, the cache was built from {}",
            m["init_sha256"]
        )));
    }
    let node_manifest = hf_embed::read_manifest(a.embeddings_dir)?;
    if json!(node_manifest.model_digest) != m["embedding_model_digest"] {
        return Err(HfError::BandH(
            "the node cache's encoder is not the one the cache was built with".into(),
        ));
    }
    let split = PathBuf::from(m["deletion_split"].as_str().unwrap_or(""));
    let embeddings = hf_embed::EmbeddingMatrix::load(a.embeddings_dir)?;
    let queries = hf_embed::EmbeddingMatrix::load(&split.join("queries"))?;
    let records = r1::read_deletion_visible(&split.join("visible.jsonl.gz"), 0)?;
    if records.len() != cache.len() {
        return Err(HfError::BandH(format!(
            "the split holds {} records, the cache {}",
            records.len(),
            cache.len()
        )));
    }
    let batch = m["batch_size"]
        .as_u64()
        .ok_or_else(|| HfError::BandH("the cache manifest has no batch_size".into()))?
        as usize;
    let mut k = 0usize;
    let mut mismatched: Vec<String> = Vec::new();
    for chunk in records.chunks(batch.max(1)) {
        let indexes = deletion_indexes(chunk, &queries, &embeddings, dim as usize)?;
        let refs: Vec<&EpisodeIndex> = indexes.iter().collect();
        for fresh in r1::compute_features(&model, &refs, &r1c.spec)? {
            if !r1::bit_equal(&fresh, &cache.get(k)?) {
                mismatched.push(fresh.episode_id.clone());
            }
            k += 1;
        }
    }
    let ok = mismatched.is_empty();
    std::fs::create_dir_all(a.output).map_err(invalid)?;
    std::fs::write(
        &out,
        hf_core::files::python_json_pretty(&json!({
            "record_kind": "r1_cache_verify_v1",
            "evidence": a.provenance.evidence,
            "ok": ok,
            "records": k,
            "mismatched": mismatched.len(),
            "mismatched_ids": mismatched.iter().take(20).collect::<Vec<_>>(),
            "device": if a.device == Device::Cpu { "cpu" } else { "cuda" },
            "cache_device": m["device"],
            "batch_size": batch,
            "cache_manifest_sha256": hf_core::sha256_file(&a.cache_dir.join(r1::CACHE_MANIFEST)).map_err(invalid)?.1,
            "provenance": a.provenance.value,
            "training_authorized": false,
        })),
    )
    .map_err(invalid)?;
    Ok(ok)
}

// --------------------------------------------------------------------------
// --r1-train

pub struct TrainArgs<'a> {
    pub cache_dir: &'a Path,
    pub config: &'a Value,
    pub init: &'a Path,
    pub output: &'a Path,
    pub model_seed: u64,
    pub updates: Option<u64>,
    pub device: Device,
    pub fixture: bool,
    pub provenance: Provenance,
}

fn string_list(v: &Value, what: &str) -> Result<Vec<String>, HfError> {
    v.as_array()
        .ok_or_else(|| HfError::Invalid(format!("{what} must be a list of patterns")))?
        .iter()
        .map(|p| {
            p.as_str()
                .map(str::to_string)
                .ok_or_else(|| HfError::Invalid(format!("{what}: {p} is not a string")))
        })
        .collect()
}

pub fn train(a: TrainArgs<'_>) -> Result<(), HfError> {
    for p in [a.cache_dir, a.init, a.output] {
        hf_core::refuse_holdout(p)?;
    }
    let r1c = r1_config(a.config)?;
    let training = &a.config["training"];
    let trainable = string_list(&training["trainable"], "training.trainable")?;
    let lr = training["learning_rate"]
        .as_f64()
        .ok_or_else(|| HfError::Invalid("training.learning_rate".into()))?;
    let weight_decay = training["weight_decay"].as_f64().unwrap_or(0.0);
    let updates = a
        .updates
        .unwrap_or(training["update_count"].as_u64().unwrap_or(0));
    let microbatch = training["microbatch_size"].as_u64().unwrap_or(8) as usize;
    let clip = match training.get("clip_max_norm") {
        None => Some(1.0),
        Some(Value::Null) => None,
        Some(v) => Some(
            v.as_f64()
                .filter(|m| m.is_finite() && *m > 0.0)
                .ok_or_else(|| HfError::Invalid("training.clip_max_norm".into()))?,
        ),
    };
    let decay_exempt = match training.get("decay_exempt") {
        None | Some(Value::Null) => Vec::new(),
        Some(v) => string_list(v, "training.decay_exempt")?,
    };
    let rows_path = a.output.join("updates.jsonl");
    refuse_existing(&rows_path)?;
    let dim = embedding_dimension_of(a.cache_dir)?;
    tch::manual_seed(a.model_seed as i64);
    let (mut model, _) = r1_model(a.config, dim, a.device)?;
    let init_sha = load_init(&mut model, a.init)?;
    let init_trunk = trunk_digest(&model);
    let cache = open_checked(a.cache_dir, &r1c, &init_trunk)?;
    if cache.manifest["init_sha256"].as_str() != Some(init_sha.as_str()) {
        return Err(HfError::BandH(format!(
            "--init-from is {init_sha}, the cache was built from {}",
            cache.manifest["init_sha256"]
        )));
    }
    // the trainable set: the heads, and nothing of the trunk
    let kept = model.freeze_except(&trainable)?;
    let heads = model.names_matching(&R1_HEAD_PATTERNS);
    if kept != heads {
        return Err(HfError::Refused(format!(
            "training.trainable must select exactly the R1 heads' parameters; it selects {kept:?}"
        )));
    }
    let mut optimiser = AdamW::new(&model.vs, lr, weight_decay);
    let (exempt, _) = optimiser.set_decay_exempt(&decay_exempt);
    // T per episode, from the deletion split's labels: the loss's alone
    let split = PathBuf::from(cache.manifest["deletion_split"].as_str().unwrap_or(""));
    let labels = r1::read_deletion_labels(&split.join("labels.jsonl.gz"))?;
    let mut targets: Vec<&HashSet<String>> = Vec::with_capacity(cache.len());
    for i in 0..cache.len() {
        targets.push(
            labels
                .get(cache.episode_id(i))
                .ok_or_else(|| HfError::BandH(format!("{}: no label row", cache.episode_id(i))))?,
        );
    }
    std::fs::create_dir_all(a.output).map_err(invalid)?;
    if a.output.join("halted").exists() {
        return Err(HfError::BandH(format!("{} halted", a.output.display())));
    }
    let inject_nan_at: Option<u64> = if a.fixture {
        std::env::var("HF_TEST_INJECT_NAN_AT_UPDATE")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
    } else {
        None
    };
    let mut log = std::fs::File::create(&rows_path).map_err(invalid)?;
    let mut rng = PyRandom::from_seed(a.model_seed as u128);
    let mut draws: BTreeMap<String, u64> = BTreeMap::new();
    let t0 = Instant::now();
    for update in 1..=updates {
        let idx = rng.sample(cache.len(), microbatch.min(cache.len()));
        let records: Vec<r1::R1Record> = idx
            .iter()
            .map(|i| cache.get(*i))
            .collect::<Result<_, _>>()?;
        for r in &records {
            *draws.entry(r.episode_id.clone()).or_default() += 1;
        }
        let refs: Vec<&r1::R1Record> = records.iter().collect();
        let tg: Vec<&HashSet<String>> = idx.iter().map(|i| targets[*i]).collect();
        let out = r1::train_step(
            &model,
            &mut optimiser,
            &refs,
            &tg,
            &r1c.spec,
            clip,
            inject_nan_at == Some(update),
        )?;
        let row = json!({
            "update": update,
            "ret": out.ret,
            "stop": out.stop,
            "total": out.total,
            "grad_norm": out.grad_norm,
            "finite": out.finite,
            "expansions": out.mean_expansions,
            "examined": out.mean_examined,
            "pos_share": if out.candidates == 0 { 0.0 } else { out.positives as f64 / out.candidates as f64 },
            "positives": out.positives,
            "candidates": out.candidates,
            "seconds": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0,
        });
        writeln!(log, "{row}").map_err(invalid)?;
        if !out.finite {
            std::fs::write(
                a.output.join("instability.json"),
                hf_core::files::python_json_pretty(&json!({
                    "record_kind": format!("r1_head_instability_v1{}", if a.fixture { "_FIXTURE" } else { "" }),
                    "evidence": a.provenance.evidence,
                    "update": update,
                    "injected": inject_nan_at == Some(update),
                })),
            )
            .map_err(invalid)?;
            std::fs::write(
                a.output.join("halted"),
                format!("non-finite update {update}\n"),
            )
            .map_err(invalid)?;
            return Err(HfError::BandH(format!("non-finite update {update}")));
        }
    }
    log.sync_all().map_err(invalid)?;
    let final_trunk = trunk_digest(&model);
    let frozen_ok = final_trunk == init_trunk;
    let mut model_block = a.config["model"].as_object().cloned().unwrap_or_default();
    model_block.insert("embedding_dimension".into(), dim.into());
    let meta = json!({
        "record_kind": "hippo13_r1_head_checkpoint_v1",
        "config": {"model": model_block},
        "seed": a.model_seed,
        "update": updates,
        "init_from": a.init.to_string_lossy(),
        "init_sha256": init_sha,
        "trunk_state_digest": final_trunk,
        "init_trunk_state_digest": init_trunk,
        "frozen_ok": frozen_ok,
        "cache": a.cache_dir.to_string_lossy(),
        "cache_manifest_sha256": hf_core::sha256_file(&a.cache_dir.join(r1::CACHE_MANIFEST)).map_err(invalid)?.1,
        "trainable": kept,
        "decay_exempt": exempt,
        "provenance": a.provenance.value,
        "training_authorized": false,
    });
    let weights = a.output.join("checkpoint.safetensors");
    model
        .vs
        .save(&weights)
        .map_err(|e| HfError::Invalid(format!("{}: {e}", weights.display())))?;
    std::fs::write(
        a.output.join("checkpoint.json"),
        serde_json::to_string_pretty(&meta).map_err(invalid)?,
    )
    .map_err(invalid)?;
    let summary = json!({
        "record_kind": format!("r1_head_train{}", if a.fixture { "_FIXTURE" } else { "" }),
        "evidence": a.provenance.evidence,
        "governed_by": "experiments/real_walk_v2/R1_HEAD_PREREGISTRATION.md",
        "updates": updates,
        "microbatch": microbatch,
        "draws": draws.values().sum::<u64>(),
        "distinct_seen": draws.len(),
        "pool": cache.len(),
        "capacity": model.trainable_parameter_count(),
        "trainable_parameters": model.parameter_count_matching(&R1_HEAD_PATTERNS),
        "frozen_ok": frozen_ok,
        "trunk_state_digest": final_trunk,
        "init_sha256": init_sha,
        "labels_opened_by": "the loss only",
        "provenance": a.provenance.value,
        "engine_build": engine_fields(),
        "training_authorized": false,
    });
    std::fs::write(
        a.output.join("r1_train.json"),
        hf_core::files::python_json_pretty(&summary),
    )
    .map_err(invalid)?;
    if !frozen_ok {
        return Err(HfError::BandH(
            "a frozen parameter moved during training".into(),
        ));
    }
    Ok(())
}

/// The node embedding width the cache was built at, from its manifest (the
/// model is built at it; the relational sets read no raw coordinate). The
/// manifest alone is read here; `R1Cache::open` checks it against the files.
fn embedding_dimension_of(cache_dir: &Path) -> Result<i64, HfError> {
    let m = data::read_json(&cache_dir.join(r1::CACHE_MANIFEST))?;
    m["embedding_dimension"]
        .as_i64()
        .ok_or_else(|| HfError::BandH("the cache manifest has no embedding_dimension".into()))
}

// --------------------------------------------------------------------------
// --r1-eval

pub struct EvalArgs<'a> {
    pub cache_dir: &'a Path,
    pub config: &'a Value,
    pub checkpoint: &'a Path,
    pub output: &'a Path,
    pub device: Device,
    pub fixture: bool,
    pub provenance: Provenance,
}

pub fn eval(a: EvalArgs<'_>) -> Result<(), HfError> {
    for p in [a.cache_dir, a.checkpoint, a.output] {
        hf_core::refuse_holdout(p)?;
    }
    let rows_path = a.output.join("r1_rows.jsonl");
    refuse_existing(&rows_path)?;
    let r1c = r1_config(a.config)?;
    let dim = embedding_dimension_of(a.cache_dir)?;
    let (mut model, _) = r1_model(a.config, dim, a.device)?;
    let weights = weights_of(a.checkpoint)?;
    model.load_strict(&weights, &[], &[])?;
    let (_, ck_sha) = hf_core::sha256_file(&weights).map_err(invalid)?;
    let cache = open_checked(a.cache_dir, &r1c, &trunk_digest(&model))?;
    // the head forward's reductions are pinned to one thread, so the rows do
    // not depend on the machine's thread count (ENG-4 (d))
    tch::set_num_threads(1);
    std::fs::create_dir_all(a.output).map_err(invalid)?;
    let mut file = std::fs::File::create(&rows_path).map_err(invalid)?;
    for i in 0..cache.len() {
        let row = r1::eval_logits(&model, &cache.get(i)?)?;
        writeln!(file, "{}", serde_json::to_string(&row).map_err(invalid)?).map_err(invalid)?;
    }
    file.sync_all().map_err(invalid)?;
    let summary = json!({
        "record_kind": format!("r1_head_eval{}", if a.fixture { "_FIXTURE" } else { "" }),
        "evidence": a.provenance.evidence,
        "cache": a.cache_dir.to_string_lossy(),
        "cache_manifest_sha256": hf_core::sha256_file(&a.cache_dir.join(r1::CACHE_MANIFEST)).map_err(invalid)?.1,
        "checkpoint": weights.to_string_lossy(),
        "checkpoint_sha256": ck_sha,
        "records": cache.len(),
        "labels_opened": false,
        "writes": "logits only: the reader applies tau and theta",
        "provenance": a.provenance.value,
        "engine_build": engine_fields(),
        "training_authorized": false,
    });
    std::fs::write(
        a.output.join("r1_eval.json"),
        hf_core::files::python_json_pretty(&summary),
    )
    .map_err(invalid)?;
    Ok(())
}

// --------------------------------------------------------------------------
// --frozen-check

pub fn frozen_check(
    config: &Value,
    checkpoint: &Path,
    init: &Path,
    output: &Path,
    dim: i64,
) -> Result<bool, HfError> {
    for p in [checkpoint, init, output] {
        hf_core::refuse_holdout(p)?;
    }
    let out = output.join("frozen_check.json");
    refuse_existing(&out)?;
    let (mut trained, _) = r1_model(config, dim, Device::Cpu)?;
    let weights = weights_of(checkpoint)?;
    trained.load_strict(&weights, &[], &[])?;
    let (mut initial, _) = r1_model(config, dim, Device::Cpu)?;
    let init_sha = load_init(&mut initial, init)?;
    let (got, want) = (trunk_digest(&trained), trunk_digest(&initial));
    // per parameter, so a mismatch names what moved
    let a = trained.vs.variables();
    let b = initial.vs.variables();
    let heads: HashSet<String> = trained
        .names_matching(&R1_HEAD_PATTERNS)
        .into_iter()
        .collect();
    let mut moved: Vec<String> = a
        .iter()
        .filter(|(n, _)| !heads.contains(*n))
        .filter(|(n, t)| b.get(*n).is_none_or(|u| !t.equal(u)))
        .map(|(n, _)| n.clone())
        .collect();
    moved.sort();
    let ok = got == want && moved.is_empty();
    std::fs::create_dir_all(output).map_err(invalid)?;
    std::fs::write(
        &out,
        hf_core::files::python_json_pretty(&json!({
            "record_kind": "r1_head_frozen_check_v1",
            "ok": ok,
            "checkpoint": weights.to_string_lossy(),
            "checkpoint_sha256": hf_core::sha256_file(&weights).map_err(invalid)?.1,
            "init": init.to_string_lossy(),
            "init_sha256": init_sha,
            "trunk_state_digest_checkpoint": got,
            "trunk_state_digest_init": want,
            "moved_parameters": moved,
            "training_authorized": false,
        })),
    )
    .map_err(invalid)?;
    Ok(ok)
}
