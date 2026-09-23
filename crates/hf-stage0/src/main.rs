//! `hf-stage0`: the real-walk stage-0 runner — `scripts/real_walk_stage0.py`
//! on the Rust engine, flag for flag, writing `updates.jsonl`,
//! `evaluation_rows.jsonl`, `probe.json` / `reeval.json`,
//! `candidate_dump.jsonl.gz` and checkpoints the foundation's readers consume.
//!
//! Governance: a real run needs `--preregistration-commit`, which must be an
//! ancestor of the **foundation** repository's HEAD; the config must state
//! `training_authorized: false`; an output path naming a holdout is refused;
//! `--fixture` trains on the synthetic world, on the CPU, and labels its
//! output `_FIXTURE` with `evidence: false`.
//!
//! Stability: every `updates.jsonl` row carries `finite`, and the first update
//! whose losses or pre-clip gradient norm are not all finite skips its optimiser
//! step, writes `instability.json` and a `halted` marker beside
//! `checkpoint-latest`, and exits 2 — no `probe.json` is written, so nothing
//! downstream reads the run as finished.
//!
//! Both repositories' heads are
//! recorded (`git_head` the foundation's, `engine_head` this one's, read from
//! the checkouts at RUN time), and beside them what the running binary was
//! BUILT from: `engine_build_head`, `engine_build_dirty` and the binary's own
//! `engine_binary_sha256`. A real run whose binary is not the tree's refuses
//! unless `--allow-stale-engine` says the operator knows.

// the probe's `json!` literal is one expansion deep per key, and the engine's
// provenance keys take it past serde_json's default 128
#![recursion_limit = "256"]

mod data;
mod eval;

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use hf_core::{HfError, PyRandom, PyRandomState};
use hf_model::{clip_grad_norm, walk_losses, AdamW, LossConfig, Model, ModelConfig, ModelScorer};
use hf_walk::{walk_batch, EpisodeIndex, StopRule, WalkOptions};
use serde_json::{json, Map, Value};
use tch::Device;

pub const TRAIN_SAMPLE_SEED: u128 = 20260912;
pub const TRAIN_SAMPLE_BASE: usize = 5000;

#[derive(Parser, Debug)]
#[command(
    name = "hf-stage0",
    about = "the real-walk stage-0 runner on the Rust engine"
)]
struct Args {
    #[arg(long, required_unless_present = "engine_info")]
    config: Option<PathBuf>,
    #[arg(long, required_unless_present_any = ["engine_info", "print_capacity"])]
    output: Option<PathBuf>,
    #[arg(long, required_unless_present_any = ["engine_info", "print_capacity"])]
    model_seed: Option<u64>,
    /// synthetic fixture; never evidence
    #[arg(long)]
    fixture: bool,
    #[arg(long)]
    family: Option<String>,
    #[arg(long)]
    graph_dir: Option<PathBuf>,
    #[arg(long)]
    embeddings_dir: Option<PathBuf>,
    /// an ordinary v5 embedding cache whose ids are EPISODE ids: the question
    /// vectors of a stage-1 split. It needs `data.query_source:
    /// "episode_query"` in the config, and every episode the run reads must be
    /// in it.
    #[arg(long)]
    query_embeddings_dir: Option<PathBuf>,
    /// refuse (exit 2) when the share of the run's DISTINCT visible nodes
    /// present in --embeddings-dir is below this; a node the cache lacks is a
    /// silent zero vector otherwise. It covers the node cache of the training
    /// family over train, screen and screen2; the question vectors are not
    /// covered by it — a missing episode id exits 2 whatever this says.
    #[arg(long)]
    expect_embedding_coverage: Option<f64>,
    /// the split directory to train and screen on. Repeat it (P1) to train on
    /// a CUMULATIVE pool: the training pool is then the concatenation of every
    /// directory's `train` split, in the order given. The FIRST is the run's
    /// own split — its `screen` is the evaluation set and its
    /// `graph.manifest.json` the run's; the others contribute their `train`
    /// and nothing else. A microbatch is one draw over the concatenation, so
    /// each pool's share is its size, not an equal share, and `train_draws`
    /// replays exactly as it does for one pool. With more than one, the
    /// manifest agreement stops checking `subgraph_size` and
    /// `target_distance` — a pool from an earlier rung has neither — and
    /// probe.json records them per pool in `train_pools` instead.
    #[arg(long)]
    splits_dir: Vec<PathBuf>,
    #[arg(long)]
    heldout_family: Option<String>,
    #[arg(long)]
    heldout_splits_dir: Option<PathBuf>,
    #[arg(long)]
    heldout_embeddings_dir: Option<PathBuf>,
    #[arg(long)]
    preregistration_commit: Option<String>,
    /// the foundation checkout the pin is checked against (env HF_FOUNDATION; default the sibling)
    #[arg(long)]
    foundation_root: Option<PathBuf>,
    #[arg(long)]
    updates: Option<u64>,
    #[arg(long, default_value_t = 250)]
    eval_every: u64,
    /// cap on training episodes; default the whole pool
    #[arg(long)]
    train_episodes: Option<usize>,
    #[arg(long, default_value_t = 400)]
    screen_episodes: usize,
    #[arg(long)]
    save_checkpoint: bool,
    #[arg(long, default_value_t = 0)]
    checkpoint_every: u64,
    #[arg(long, default_value_t = 0.0)]
    checkpoint_minutes: f64,
    /// continue from checkpoint-latest.json
    #[arg(long)]
    resume: Option<PathBuf>,
    /// evaluate a checkpoint (its .json or directory) instead of training
    #[arg(long)]
    reevaluate_checkpoint: Option<PathBuf>,
    #[arg(long)]
    screen2_splits_dir: Option<PathBuf>,
    /// Skip loaded screen2 rows. This preserves the stage-0 800 -> 400 split;
    /// filtered stage-1 splits must select source ordinals before admission.
    #[arg(long, default_value_t = 400)]
    screen2_skip: usize,
    #[arg(long)]
    override_greedy_tau: Option<f64>,
    #[arg(long)]
    deterministic: bool,
    #[arg(long, default_value_t = 0)]
    train_sample: usize,
    #[arg(long)]
    dump_candidates: bool,
    /// comma-separated feature groups to zero at scoring (raw-v5: candidate,query,path_mean,parent,pair or all)
    #[arg(long)]
    ablate_embedding_blocks: Option<String>,
    #[arg(long)]
    train_draws_probe: Option<PathBuf>,
    /// print the engine's provenance and exit
    #[arg(long)]
    engine_info: bool,
    /// run although this binary was not built from the engine checkout's HEAD,
    /// or was built from a dirty tree; recorded as `engine_stale_allowed`
    #[arg(long)]
    allow_stale_engine: bool,
    /// build the config's model at this embedding dimension on the CPU, print
    /// its trainable parameter count and exit; no data is read
    #[arg(long, value_name = "DIM")]
    print_capacity: Option<i64>,
}

fn foundation_root(args: &Args) -> PathBuf {
    args.foundation_root
        .clone()
        .or_else(|| std::env::var("HF_FOUNDATION").ok().map(PathBuf::from))
        .unwrap_or_else(|| data::engine_root().join("../hippocampus-foundation"))
}

/// The stale-binary gate. `engine_head` is the engine checkout's HEAD as `git`
/// reads it now; the binary running was built from `ENGINE_BUILD_HEAD`, and
/// once the tree has moved on the artifact would name a commit whose code
/// never ran. Band H when they disagree, or when the build's tree was dirty
/// (its sources are in no commit at all), unless `--allow-stale-engine`.
fn engine_preflight(args: &Args, engine_head: &str) -> Result<(), HfError> {
    let stale = if data::engine_build_dirty() {
        Some(format!(
            "built from a dirty tree at {}",
            data::ENGINE_BUILD_HEAD
        ))
    } else if data::ENGINE_BUILD_HEAD != engine_head {
        Some(format!(
            "built at {}, but the engine checkout is at {engine_head}",
            data::ENGINE_BUILD_HEAD
        ))
    } else {
        None
    };
    match stale {
        Some(why) if !args.allow_stale_engine => Err(HfError::BandH(format!(
            "stale engine binary: {why} (built {}); rebuild it (cargo build --release) \
             or pass --allow-stale-engine",
            data::ENGINE_BUILD_TIME
        ))),
        _ => Ok(()),
    }
}

fn preflight(args: &Args, foundation: &Path, engine_head: &str) -> Result<(), HfError> {
    if args.fixture {
        return Ok(());
    }
    engine_preflight(args, engine_head)?;
    let Some(pin) = &args.preregistration_commit else {
        return Err(HfError::Refused(
            "a real run needs --preregistration-commit".into(),
        ));
    };
    let ok = std::process::Command::new("git")
        .args(["merge-base", "--is-ancestor", pin, "HEAD"])
        .current_dir(foundation)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        return Err(HfError::Refused(format!(
            "HEAD of {} does not descend from the preregistration commit {pin}; refusing",
            foundation.display()
        )));
    }
    Ok(())
}

/// `--print-capacity <DIM>`: the config's model at that embedding dimension,
/// instantiated on the CPU, and the count it would train.
fn print_capacity(config_path: &Path, dim: i64) -> Result<(), HfError> {
    let config = data::read_json(config_path)?;
    let model_config = ModelConfig::from_value(&config["model"], dim)?;
    let model = Model::new(model_config.clone(), Device::Cpu)?;
    println!(
        "{}",
        json!({
            "feature_set": model_config.feature_set,
            "dim": dim,
            "trainable_parameters": model.trainable_parameter_count(),
        })
    );
    Ok(())
}

/// `training.decay_exempt`: the patterns whose matching parameters AdamW does
/// not decay (`hf_model::glob_match` on the parameter's full name). Absent —
/// every config written before the switch existed — is the empty list, which
/// decays everything exactly as the Python runner's single AdamW group does. A
/// value that is not a list of strings is a broken config, refused here.
fn decay_exempt_patterns(training: &Value) -> Result<Vec<String>, HfError> {
    match training.get("decay_exempt") {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| {
                v.as_str().map(str::to_string).ok_or_else(|| {
                    HfError::Invalid(format!(
                        "training.decay_exempt: {v} is not a pattern string"
                    ))
                })
            })
            .collect(),
        Some(other) => Err(HfError::Invalid(format!(
            "training.decay_exempt must be a list of patterns, not {other}"
        ))),
    }
}

/// `training.clip_max_norm`: the gradient-norm clip applied to every update
/// (`hf_model::clip_grad_norm`, `torch.nn.utils.clip_grad_norm_`'s rule).
/// ABSENT — every config written before the key existed — is 1.0, the constant
/// the runner passed unconditionally, so every existing config trains exactly
/// as it did. An explicit `null` is NO clipping: the pre-clip norm is still
/// computed and still logged in `updates.jsonl`, nothing is scaled. Anything
/// else must be a finite, strictly positive number; a negative, a zero, a
/// string, a list is a broken config, refused here rather than rounded into
/// some default.
fn clip_max_norm(training: &Value) -> Result<Option<f64>, HfError> {
    match training.get("clip_max_norm") {
        None => Ok(Some(1.0)),
        Some(Value::Null) => Ok(None),
        Some(other) => match other.as_f64() {
            Some(m) if m.is_finite() && m > 0.0 => Ok(Some(m)),
            _ => Err(HfError::Invalid(format!(
                "training.clip_max_norm must be a positive number or null, not {other}"
            ))),
        },
    }
}

/// `data.query_source`: where the walk's query comes from.
///
/// ABSENT — every config written before the key existed — is
/// `"target_embedding"`, the shown target's own embedding row, so every
/// existing config runs exactly as it did. `"episode_query"` is stage 1: the
/// query is the episode's own question vector, read from the sidecar named by
/// `--query-embeddings-dir`. Anything else is a broken config, refused here
/// rather than rounded into the default.
fn query_source(config: &Value) -> Result<hf_walk::QuerySource, HfError> {
    match config.get("data").and_then(|d| d.get("query_source")) {
        None | Some(Value::Null) => Ok(hf_walk::QuerySource::TargetEmbedding),
        Some(Value::String(s)) => hf_walk::QuerySource::parse(s),
        Some(other) => Err(HfError::Invalid(format!(
            "data.query_source must be a string, not {other}"
        ))),
    }
}

/// The rule a run is governed by: the config names it. The v1 rule is the
/// default the runner has always written, kept for a config that names none.
fn governed_by(config: &Value) -> Value {
    config
        .get("governed_by")
        .cloned()
        .unwrap_or_else(|| Value::from("experiments/real_walk_v1/RULE.md"))
}

fn engine_info() -> Value {
    json!({
        "engine": "hippo-13 hf-stage0",
        "engine_head": data::engine_head(),
        "engine_build_head": data::ENGINE_BUILD_HEAD,
        "engine_build_dirty": data::engine_build_dirty(),
        "engine_build_time": data::ENGINE_BUILD_TIME,
        "engine_binary_sha256": data::engine_binary_sha256(),
        "rust_toolchain": option_env!("RUSTUP_TOOLCHAIN").unwrap_or("rustup 1.95.0"),
        "cuda_available": tch::Cuda::is_available(),
        "cudnn_available": tch::Cuda::cudnn_is_available(),
    })
}

/// `train_sample_indices`: the prefix-nested draw.
pub fn train_sample_indices(n: usize, k: usize) -> Vec<usize> {
    let k = k.min(n);
    if k <= TRAIN_SAMPLE_BASE {
        let mut v = PyRandom::from_seed(TRAIN_SAMPLE_SEED).sample(n, k);
        v.sort_unstable();
        return v;
    }
    let head = PyRandom::from_seed(TRAIN_SAMPLE_SEED).sample(n, TRAIN_SAMPLE_BASE.min(n));
    let taken: std::collections::HashSet<usize> = head.iter().copied().collect();
    let remaining: Vec<usize> = (0..n).filter(|i| !taken.contains(i)).collect();
    let tail = PyRandom::from_seed(TRAIN_SAMPLE_SEED + 1).sample(remaining.len(), k - head.len());
    let mut v: Vec<usize> = head
        .into_iter()
        .chain(tail.into_iter().map(|p| remaining[p]))
        .collect();
    v.sort_unstable();
    v
}

/// `replay_train_draws` by index.
pub fn replay_train_draws(
    seed: u64,
    pool_size: usize,
    updates: u64,
    microbatch: usize,
) -> BTreeMap<usize, u64> {
    let mut rng = PyRandom::from_seed(seed as u128);
    let mut draws: BTreeMap<usize, u64> = BTreeMap::new();
    for _ in 0..updates {
        for i in rng.sample(pool_size, microbatch.min(pool_size)) {
            *draws.entry(i).or_default() += 1;
        }
    }
    draws
}

fn views_histogram(counts: impl Iterator<Item = u64>) -> Map<String, Value> {
    let mut h: BTreeMap<u64, u64> = BTreeMap::new();
    for c in counts {
        *h.entry(c).or_default() += 1;
    }
    h.into_iter()
        .map(|(k, v)| (k.to_string(), Value::from(v)))
        .collect()
}

struct Checkpoint {
    dir: PathBuf,
    stem: String,
}

impl Checkpoint {
    fn paths(&self) -> (PathBuf, PathBuf, PathBuf) {
        (
            self.dir.join(format!("{}.safetensors", self.stem)),
            self.dir.join(format!("{}.optim.safetensors", self.stem)),
            self.dir.join(format!("{}.json", self.stem)),
        )
    }

    /// Atomic: every file is written under a temporary name and renamed.
    fn write(&self, model: &Model, optimiser: Option<&AdamW>, meta: &Value) -> Result<(), HfError> {
        let (w, o, j) = self.paths();
        // VarStore::save picks its format from the extension, so the temporary
        // name keeps ".safetensors" as its suffix
        let tmp = |p: &Path| {
            let name = p
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            p.with_file_name(format!(
                ".{}.tmp.{}",
                name,
                if name.ends_with(".safetensors") {
                    "safetensors"
                } else {
                    "json"
                }
            ))
        };
        model
            .vs
            .save(tmp(&w))
            .map_err(|e| HfError::Invalid(format!("{}: {e}", w.display())))?;
        if let Some(opt) = optimiser {
            opt.save(&tmp(&o))?;
        }
        std::fs::write(tmp(&j), serde_json::to_string_pretty(meta).unwrap())
            .map_err(|e| HfError::Invalid(e.to_string()))?;
        std::fs::rename(tmp(&w), &w).map_err(|e| HfError::Invalid(e.to_string()))?;
        if optimiser.is_some() {
            std::fs::rename(tmp(&o), &o).map_err(|e| HfError::Invalid(e.to_string()))?;
        }
        std::fs::rename(tmp(&j), &j).map_err(|e| HfError::Invalid(e.to_string()))?;
        Ok(())
    }
}

/// A checkpoint named by its `.json`, its directory, or its `.safetensors`.
fn locate_checkpoint(path: &Path) -> Result<Checkpoint, HfError> {
    if path.is_dir() {
        return Ok(Checkpoint {
            dir: path.to_path_buf(),
            stem: "checkpoint".into(),
        });
    }
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("checkpoint")
        .trim_end_matches(".optim")
        .to_string();
    let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    Ok(Checkpoint { dir, stem })
}

fn main() {
    let args = Args::parse();
    if args.engine_info {
        println!("{}", serde_json::to_string_pretty(&engine_info()).unwrap());
        return;
    }
    if let Some(dim) = args.print_capacity {
        let config = args.config.clone().expect("required");
        if let Err(e) = print_capacity(&config, dim) {
            hf_core::exit_with("hf-stage0", &e);
        }
        return;
    }
    if let Err(e) = run(args) {
        hf_core::exit_with("hf-stage0", &e);
    }
}

fn run(args: Args) -> Result<(), HfError> {
    let output = args.output.clone().expect("required");
    let config_path = args.config.clone().expect("required");
    let model_seed = args.model_seed.expect("required");
    let lower = output.to_string_lossy().to_lowercase();
    if lower.contains("holdout") || lower.contains("heldout") {
        return Err(HfError::Refused("refusing a holdout path".into()));
    }
    let foundation = foundation_root(&args);
    let engine_head = data::engine_head();
    preflight(&args, &foundation, &engine_head)?;
    let config = data::read_json(&config_path)?;
    if config.get("training_authorized") != Some(&Value::Bool(false)) {
        return Err(HfError::Refused(
            "a stage 0 config must state training_authorized: false".into(),
        ));
    }
    let query_source = query_source(&config)?;
    // the stage-1 combinations that cannot mean anything, refused before a
    // path is opened or a pool is read
    match (query_source, args.query_embeddings_dir.is_some()) {
        (hf_walk::QuerySource::EpisodeQuery, false) => {
            return Err(HfError::Invalid(
                "data.query_source \"episode_query\" needs --query-embeddings-dir".into(),
            ))
        }
        (hf_walk::QuerySource::TargetEmbedding, true) => {
            return Err(HfError::Invalid(
                "--query-embeddings-dir needs data.query_source \"episode_query\" in the config"
                    .into(),
            ))
        }
        _ => {}
    }
    if query_source == hf_walk::QuerySource::EpisodeQuery {
        if let Some(d) = &args.query_embeddings_dir {
            hf_core::refuse_holdout(d)?;
        }
        if args.fixture {
            return Err(HfError::Refused(
                "the fixture world samples stage-0 episodes; it has no questions to read".into(),
            ));
        }
        // `--heldout-splits-dir` is NOT refused here. It names the held-out
        // FAMILY -- the vault, whose questions the second source writes, read
        // as a registration veto at stage 1 exactly as at stage 0 -- and not a
        // holdout path. Its episodes are part of the query sidecar's
        // pre-check exactly as the screens are (`data.rs`), so a vault id the
        // sidecar does not cover exits 2 at load rather than walking on a
        // silent zero vector. The string-based holdout refusal is untouched
        // and stands on this path as on every other: `hf_io::read_split`
        // calls `hf_core::refuse_holdout` on the directory it opens.
        if config["model"].get("feature_set").and_then(Value::as_str) == Some("raw-v5") {
            return Err(HfError::Refused(
                "the feature set raw-v5 copies the raw query vector into every candidate row \
                 and into the query token; under query_source episode_query that vector is \
                 the question itself"
                    .into(),
            ));
        }
    }
    let deterministic = if args.deterministic {
        if std::env::var_os("CUBLAS_WORKSPACE_CONFIG").is_none() {
            std::env::set_var("CUBLAS_WORKSPACE_CONFIG", ":4096:8");
        }
        Some(json!({
            "requested": true,
            "cublas_workspace_config": std::env::var("CUBLAS_WORKSPACE_CONFIG").unwrap_or_default(),
            "warn_only": true,
            "nondeterministic_warnings": [],
            "engine_note": "tch 0.26 binds no use_deterministic_algorithms; the seed and the cuBLAS workspace are set, the flag is not",
        }))
    } else {
        None
    };
    tch::manual_seed(model_seed as i64);
    let device = if tch::Cuda::is_available() && !args.fixture {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    let inputs = data::Inputs {
        family: args.family.as_deref(),
        splits_dirs: &args.splits_dir,
        graph_dir: args.graph_dir.as_deref(),
        embeddings_dir: args.embeddings_dir.as_deref(),
        heldout_family: args.heldout_family.as_deref(),
        heldout_splits_dir: args.heldout_splits_dir.as_deref(),
        heldout_embeddings_dir: args.heldout_embeddings_dir.as_deref(),
        screen2_splits_dir: args.screen2_splits_dir.as_deref(),
        screen2_skip: args.screen2_skip,
        train_episodes: args.train_episodes,
        screen_episodes: args.screen_episodes,
        fixture: args.fixture,
        model_seed,
        query_source,
        query_embeddings_dir: args.query_embeddings_dir.as_deref(),
        expect_embedding_coverage: args.expect_embedding_coverage,
        // a re-evaluation walks the screens and the vault; it touches a
        // training episode only under `--train-sample`
        walks_training_pool: args.reevaluate_checkpoint.is_none() || args.train_sample > 0,
    };
    let d = data::load(&inputs, &config)?;
    let model_config = ModelConfig::from_value(&config["model"], d.dim as i64)?;
    let model_config_value = {
        let mut m = config["model"].as_object().cloned().unwrap_or_default();
        m.insert("embedding_dimension".into(), d.dim.into());
        json!({"model": m})
    };
    let mut model = Model::new(model_config.clone(), device)?;
    let capacity = model.trainable_parameter_count();
    if let Some(stated) = config["model"]
        .get("stated_capacity")
        .and_then(|s| s.get(d.dim.to_string()))
    {
        if stated.as_i64() != Some(capacity) {
            return Err(HfError::BandH(format!(
                "capacity {capacity} != stated {stated} at dimension {}",
                d.dim
            )));
        }
    }
    for (flag, set) in [
        ("--override-greedy-tau", args.override_greedy_tau.is_some()),
        ("--train-sample", args.train_sample > 0),
        ("--dump-candidates", args.dump_candidates),
        (
            "--ablate-embedding-blocks",
            args.ablate_embedding_blocks.is_some(),
        ),
    ] {
        if set && args.reevaluate_checkpoint.is_none() {
            return Err(HfError::Invalid(format!(
                "{flag} needs --reevaluate-checkpoint"
            )));
        }
    }
    if args.train_draws_probe.is_some() && args.train_sample == 0 {
        return Err(HfError::Invalid(
            "--train-draws-probe needs --train-sample".into(),
        ));
    }
    std::fs::create_dir_all(&output).map_err(|e| HfError::Invalid(e.to_string()))?;
    let provenance = json!({
        "git_head": data::git_head(&foundation),
        "engine_head": engine_head,
        "engine_build_head": data::ENGINE_BUILD_HEAD,
        "engine_build_dirty": data::engine_build_dirty(),
        "engine_binary_sha256": data::engine_binary_sha256(),
        "engine_stale_allowed": args.allow_stale_engine,
        "engine": "hippo-13 hf-stage0",
    });
    if let Some(ck) = &args.reevaluate_checkpoint {
        return reevaluate(
            &args,
            &config,
            &d,
            &mut model,
            capacity,
            deterministic,
            &provenance,
            ck,
        );
    }
    train(
        &args,
        &config,
        &model_config_value,
        &d,
        &mut model,
        capacity,
        deterministic,
        &provenance,
        &foundation,
    )
}

#[allow(clippy::too_many_arguments)]
fn reevaluate(
    args: &Args,
    config: &Value,
    d: &data::Data,
    model: &mut Model,
    capacity: i64,
    deterministic: Option<Value>,
    provenance: &Value,
    ck_path: &Path,
) -> Result<(), HfError> {
    let ck = locate_checkpoint(ck_path)?;
    let (weights, _, meta_path) = ck.paths();
    let saved = data::read_json(&meta_path)?;
    model
        .vs
        .load(&weights)
        .map_err(|e| HfError::Invalid(format!("{}: {e}", weights.display())))?;
    let mut ablation = Value::Null;
    if let Some(spec) = &args.ablate_embedding_blocks {
        let mut names: Vec<String> = spec
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if names.iter().any(|n| n == "all") {
            let pair = names.iter().any(|n| n == "pair");
            names = ["candidate", "query", "path_mean", "parent"]
                .iter()
                .map(|s| s.to_string())
                .collect();
            if pair {
                names.push("pair".into());
            }
        }
        for n in &names {
            if !["candidate", "query", "path_mean", "parent", "pair"].contains(&n.as_str()) {
                return Err(HfError::BandH(format!("unknown embedding block {n:?}")));
            }
        }
        let mut seen = Vec::new();
        for n in names {
            if !seen.contains(&n) {
                seen.push(n);
            }
        }
        let before = model.state_digest(&[]);
        model.zero_blocks = seen.clone();
        if model.state_digest(&[]) != before {
            return Err(HfError::BandH("the ablation changed a parameter".into()));
        }
        ablation = json!({"zero_blocks": seen, "state_digest": before});
    }
    let mut greedy_tau = Value::Null;
    if model.config.greedy_prior {
        let trained = model.greedy_tau().unwrap_or(f64::NAN);
        let before = model.state_digest(&["greedy_tau"]);
        if let Some(t) = args.override_greedy_tau {
            let vars = model.vs.variables();
            let mut tau = vars["greedy_tau"].shallow_clone();
            tch::no_grad(|| {
                let _ = tau.fill_(t);
            });
        }
        let after = model.state_digest(&["greedy_tau"]);
        if before != after {
            return Err(HfError::BandH(
                "the tau override changed another parameter".into(),
            ));
        }
        greedy_tau = json!({
            "greedy_tau_trained": trained,
            "greedy_tau_override": args.override_greedy_tau,
            "greedy_tau_in_effect": model.greedy_tau(),
            "state_digest_excluding_greedy_tau": after,
        });
    } else if args.override_greedy_tau.is_some() {
        return Err(HfError::BandH(
            "--override-greedy-tau needs a greedy-prior model".into(),
        ));
    }
    if args
        .output
        .as_ref()
        .expect("required")
        .join("evaluation_rows.jsonl")
        .exists()
        || args
            .output
            .as_ref()
            .expect("required")
            .join("candidate_dump.jsonl.gz")
            .exists()
    {
        return Err(HfError::Refused(format!(
            "{} already holds a re-evaluation's rows (they append)",
            args.output.as_ref().expect("required").display()
        )));
    }
    let update = Value::from("reeval");
    let evaluate_split = |split: &str,
                          episodes: &[hf_io::RealEpisode],
                          emb: &hf_embed::EmbeddingMatrix|
     -> Result<Value, HfError> {
        let ev = eval::evaluate(
            model,
            episodes,
            emb,
            d.dim,
            args.dump_candidates,
            d.query_vectors(),
        )?;
        eval::write_evaluation_rows(
            args.output.as_ref().expect("required"),
            split,
            &update,
            &ev.rows,
        )?;
        if args.dump_candidates {
            eval::write_candidate_dump(
                args.output.as_ref().expect("required"),
                split,
                &update,
                &ev.candidates,
            )?;
        }
        Ok(ev.report)
    };
    let report = evaluate_split("screen", &d.screen, &d.embeddings)?;
    let report_heldout = match &d.heldout {
        Some(h) => evaluate_split("heldout", &h.episodes, &h.embeddings)?,
        None => Value::Null,
    };
    let report_screen2 = if d.screen2.is_empty() {
        Value::Null
    } else {
        evaluate_split("screen2", &d.screen2, &d.embeddings)?
    };
    let mut train_sample = Value::Null;
    let mut report_train_sample = Value::Null;
    if args.train_sample > 0 {
        let picked = train_sample_indices(d.train.len(), args.train_sample);
        let base = train_sample_indices(d.train.len(), TRAIN_SAMPLE_BASE.min(args.train_sample));
        let episodes: Vec<hf_io::RealEpisode> =
            picked.iter().map(|i| d.train[*i].clone()).collect();
        report_train_sample = evaluate_split("train_sample", &episodes, &d.embeddings)?;
        let pool_ids: Vec<&str> = d.train.iter().map(|e| e.episode_id.as_str()).collect();
        let (counts, source, replay_meta): (BTreeMap<String, u64>, &str, Value) = match saved
            .get("train_draws")
            .and_then(Value::as_object)
        {
            Some(td) if !td.is_empty() => (
                td.iter()
                    .map(|(k, v)| (k.clone(), v.as_u64().unwrap_or(0)))
                    .collect(),
                "checkpoint",
                Value::Null,
            ),
            _ => {
                let probe = match &args.train_draws_probe {
                    Some(p) => Some(data::read_json(p)?),
                    None => None,
                };
                let updates = probe
                    .as_ref()
                    .and_then(|p| p["updates"].as_u64())
                    .unwrap_or(config["training"]["update_count"].as_u64().unwrap_or(0));
                let microbatch =
                    config["training"]["microbatch_size"].as_u64().unwrap_or(8) as usize;
                let seed = saved["seed"]
                    .as_u64()
                    .ok_or_else(|| HfError::BandH("checkpoint carries no seed".into()))?;
                let by_index = replay_train_draws(seed, pool_ids.len(), updates, microbatch);
                let mut meta = json!({
                    "seed": seed, "updates": updates, "microbatch": microbatch, "pool_size": pool_ids.len(),
                    "probe": args.train_draws_probe.as_ref().map(|p| p.to_string_lossy().to_string()),
                    "histogram_matches_probe": Value::Null,
                });
                if let Some(p) = &probe {
                    let histogram = Value::Object(views_histogram(by_index.values().copied()));
                    let want = p["train_draws"]["views_histogram"].clone();
                    let matches = p["train_episodes"].as_u64() == Some(pool_ids.len() as u64)
                        && !want.is_null()
                        && want == histogram;
                    meta["histogram_matches_probe"] = Value::Bool(matches);
                    if !matches {
                        return Err(HfError::BandH("the replayed training draws do not reproduce the probe's views_histogram".into()));
                    }
                }
                (
                    by_index
                        .into_iter()
                        .map(|(i, c)| (pool_ids[i].to_string(), c))
                        .collect(),
                    "replay",
                    meta,
                )
            }
        };
        let draw_counts: Map<String, Value> = picked
            .iter()
            .map(|i| {
                (
                    pool_ids[*i].to_string(),
                    Value::from(*counts.get(pool_ids[*i]).unwrap_or(&0)),
                )
            })
            .collect();
        train_sample = json!({
            "seed": 20260912,
            "pool_size": d.train.len(),
            "count": picked.len(),
            "indices": picked,
            "episode_ids": picked.iter().map(|i| pool_ids[*i]).collect::<Vec<_>>(),
            "base": TRAIN_SAMPLE_BASE,
            "base_episode_ids": base.iter().map(|i| pool_ids[*i]).collect::<Vec<_>>(),
            "draw_counts": draw_counts,
            "draw_counts_source": source,
            "draw_counts_replay": replay_meta,
        });
    }
    let reeval = json!({
        "record_kind": format!("real_walk_stage0_reeval{}", if args.fixture { "_FIXTURE" } else { "" }),
        "evidence": !args.fixture,
        "governed_by": governed_by(config),
        "checkpoint": ck_path.to_string_lossy(),
        "checkpoint_seed": saved.get("seed"),
        "git_head": provenance["git_head"],
        "engine_head": provenance["engine_head"],
        "engine_build_head": provenance["engine_build_head"],
        "engine_build_dirty": provenance["engine_build_dirty"],
        "engine_binary_sha256": provenance["engine_binary_sha256"],
        "engine_stale_allowed": provenance["engine_stale_allowed"],
        "engine": provenance["engine"],
        "preregistration_commit": args.preregistration_commit,
        "family": d.family,
        "capacity": capacity,
        "query_source": d.query_source.as_str(),
        "query_embeddings": args.query_embeddings_dir.as_ref().map(|p| p.to_string_lossy().to_string()),
        "query_embedding_manifest": d.query_embedding_manifest,
        "query_precheck": d.query_precheck,
        "embedding_coverage": d.embedding_coverage,
        "screen_episodes": d.screen.len(),
        "screen_dropped": d.screen_dropped,
        "split_manifests": d.split_manifests,
        "evaluation": report,
        "heldout": d.heldout.as_ref().map(|h| h.meta.clone()),
        "evaluation_heldout": report_heldout,
        "greedy_tau": greedy_tau,
        "ablation": ablation,
        "deterministic": deterministic,
        "screen2_episodes": d.screen2.len(),
        "screen2_skip": if d.screen2.is_empty() { Value::Null } else { Value::from(args.screen2_skip) },
        "screen2_dropped": d.screen2_dropped,
        "evaluation_screen2": report_screen2,
        "train_episodes": d.train.len(),
        "train_sample": train_sample,
        "evaluation_train_sample": report_train_sample,
        "training_authorized": false,
    });
    std::fs::write(
        args.output.as_ref().expect("required").join("reeval.json"),
        hf_core::files::python_json_pretty(&reeval),
    )
    .map_err(|e| HfError::Invalid(e.to_string()))?;
    println!(
        "[{}] re-evaluated {}: exhaust reg {:.3} exp {:.2} | vs blind LB {}",
        d.family,
        ck_path.display(),
        reeval["evaluation"]["exhaust"]["registered"]
            .as_f64()
            .unwrap_or(f64::NAN),
        reeval["evaluation"]["exhaust"]["expansions_mean"]
            .as_f64()
            .unwrap_or(f64::NAN),
        reeval["evaluation"]["exhaust_vs_blind_exhaust"]["win_share_wilson_lower_bound"]
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn train(
    args: &Args,
    config: &Value,
    model_config_value: &Value,
    d: &data::Data,
    model: &mut Model,
    capacity: i64,
    deterministic: Option<Value>,
    provenance: &Value,
    foundation: &Path,
) -> Result<(), HfError> {
    let training = &config["training"];
    let lr = training["learning_rate"]
        .as_f64()
        .ok_or_else(|| HfError::Invalid("training.learning_rate".into()))?;
    let weight_decay = training["weight_decay"].as_f64().unwrap_or(0.0);
    let updates = args
        .updates
        .unwrap_or(training["update_count"].as_u64().unwrap_or(0));
    let microbatch = training["microbatch_size"].as_u64().unwrap_or(8) as usize;
    let loss_cfg = LossConfig {
        residual_penalty: training["residual_penalty"].as_f64().unwrap_or(0.0),
        residual_penalty_margin: training["residual_penalty_margin"].as_f64(),
        ..Default::default()
    };
    let mut optimiser = AdamW::new(&model.vs, lr, weight_decay);
    let exempt_patterns = decay_exempt_patterns(training)?;
    let (exempt_names, exempt_unmatched) = optimiser.set_decay_exempt(&exempt_patterns);
    if !exempt_patterns.is_empty() {
        println!(
            "[{}] weight decay {weight_decay} exempts {} of {} parameters: {}",
            d.family,
            exempt_names.len(),
            model.vs.trainable_variables().len(),
            exempt_names.join(" "),
        );
    }
    if !exempt_unmatched.is_empty() {
        println!(
            "[{}] training.decay_exempt patterns matching no parameter: {}",
            d.family,
            exempt_unmatched.join(" "),
        );
    }
    // the set the run actually used, beside the patterns that produced it: the
    // final checkpoint's metadata carries no config, so the patterns would be
    // unrecoverable from it otherwise
    let decay_exempt = json!({
        "patterns": exempt_patterns,
        "parameters": exempt_names,
        "unmatched_patterns": exempt_unmatched,
    });
    let clip = clip_max_norm(training)?;
    match clip {
        None => println!(
            "[{}] training.clip_max_norm: null — gradients are not clipped; \
             the pre-clip norm is still computed and logged",
            d.family
        ),
        Some(m) if m != 1.0 => println!("[{}] gradient clip {m}", d.family),
        Some(_) => {}
    }
    // null or the number in effect, recorded so a reader never has to infer an
    // absent key's meaning (and so the final checkpoint, which carries no
    // config, still says which clip trained it)
    let clip_max_norm_value = clip.map(Value::from).unwrap_or(Value::Null);
    let output = args.output.as_ref().expect("required");
    // A halted output is not a base to build on: its updates.jsonl carries the
    // non-finite row and its checkpoint-latest is the state that produced it.
    // Resuming (or re-running) into it would truncate the log past the halt and
    // finish by writing probe.json beside instability.json — a directory that
    // says both "finished" and "halted". The operator moves it aside first.
    if output.join("halted").exists() {
        return Err(HfError::BandH(format!(
            "{} halted on a non-finite update (see instability.json); \
             move it aside before running again",
            output.display()
        )));
    }
    // Test-only: `HF_TEST_INJECT_NAN_AT_UPDATE=n` multiplies update n's total
    // loss by NaN, so the guard below is exercised without waiting for a real
    // divergence. Honoured only under `--fixture`, whose artifacts are never
    // evidence; a real run ignores the variable entirely.
    let inject_nan_at: Option<u64> = if args.fixture {
        std::env::var("HF_TEST_INJECT_NAN_AT_UPDATE")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
    } else {
        None
    };
    if let Some(at) = inject_nan_at {
        eprintln!("hf-stage0: HF_TEST_INJECT_NAN_AT_UPDATE={at} (fixture only; test only)");
    }
    let mut started = hf_core::utc_now_iso();
    let head_at_start = provenance["git_head"]
        .as_str()
        .unwrap_or("unknown")
        .to_string();
    let mut evaluations: Vec<Value> = Vec::new();
    let mut evaluations_heldout: Vec<Value> = Vec::new();
    let mut evaluations_screen2: Vec<Value> = Vec::new();
    let mut train_draws: BTreeMap<String, u64> = BTreeMap::new();
    let mut rng = PyRandom::from_seed(args.model_seed.expect("required") as u128);
    let mut first_update = 1u64;
    let mut elapsed_before = 0.0f64;
    let mut resumed_from: Vec<Value> = Vec::new();
    if let Some(resume) = &args.resume {
        let ck = locate_checkpoint(resume)?;
        let (w, o, j) = ck.paths();
        let saved = data::read_json(&j)?;
        if saved["seed"].as_u64() != Some(args.model_seed.expect("required")) {
            return Err(HfError::BandH(
                "checkpoint seed differs from this run's".into(),
            ));
        }
        if saved["preregistration_commit"].as_str() != args.preregistration_commit.as_deref() {
            return Err(HfError::BandH(
                "checkpoint preregistration_commit differs from this run's".into(),
            ));
        }
        if saved["config"] != *config {
            return Err(HfError::BandH(
                "checkpoint config differs from this run's".into(),
            ));
        }
        model
            .vs
            .load(&w)
            .map_err(|e| HfError::Invalid(format!("{}: {e}", w.display())))?;
        optimiser.load(&o)?;
        let state: PyRandomState = serde_json::from_value(saved["sampler_rng"].clone())
            .map_err(|e| HfError::BandH(format!("sampler rng state: {e}")))?;
        rng = PyRandom::from_state(&state)?;
        evaluations = saved["evaluations"].as_array().cloned().unwrap_or_default();
        evaluations_heldout = saved["evaluations_heldout"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        evaluations_screen2 = saved["evaluations_screen2"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        train_draws = saved["train_draws"]
            .as_object()
            .map(|m| {
                m.iter()
                    .map(|(k, v)| (k.clone(), v.as_u64().unwrap_or(0)))
                    .collect()
            })
            .unwrap_or_default();
        first_update = saved["update"].as_u64().unwrap_or(0) + 1;
        elapsed_before = saved["elapsed_seconds"].as_f64().unwrap_or(0.0);
        started = saved["started_at"].as_str().unwrap_or(&started).to_string();
        resumed_from = saved["resumed_from"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        resumed_from.push(json!({
            "update": saved["update"],
            "checkpoint_git_head": saved["git_head"],
            "resumed_at": hf_core::utc_now_iso(),
            "resumed_git_head": head_at_start,
        }));
        // the update log is truncated to the rows before the resume point
        let log_path = output.join("updates.jsonl");
        if log_path.exists() {
            let kept: Vec<String> = std::fs::read_to_string(&log_path)
                .map_err(|e| HfError::Invalid(e.to_string()))?
                .lines()
                .filter(|l| {
                    serde_json::from_str::<Value>(l)
                        .ok()
                        .and_then(|v| v["update"].as_u64())
                        .map(|u| u < first_update)
                        .unwrap_or(false)
                })
                .map(str::to_string)
                .collect();
            std::fs::write(
                &log_path,
                kept.join("\n") + if kept.is_empty() { "" } else { "\n" },
            )
            .map_err(|e| HfError::Invalid(e.to_string()))?;
        }
        println!(
            "[{}] resumed at update {first_update} from {}",
            d.family,
            j.display()
        );
    }
    let mut log = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(output.join("updates.jsonl"))
        .map_err(|e| HfError::Invalid(e.to_string()))?;
    let latest = Checkpoint {
        dir: output.clone(),
        stem: "checkpoint-latest".into(),
    };
    let features = model.features();
    let with_prior = model.config.greedy_prior;
    let t0 = Instant::now();
    let mut last_checkpoint = Instant::now();
    let checkpoint_meta = |update: u64,
                           rng: &PyRandom,
                           evals: (&[Value], &[Value], &[Value]),
                           draws: &BTreeMap<String, u64>,
                           elapsed: f64,
                           started: &str,
                           resumed: &[Value]|
     -> Value {
        json!({
            "record_kind": "hippo13_checkpoint_v1",
            "update": update,
            "seed": args.model_seed.expect("required"),
            "preregistration_commit": args.preregistration_commit,
            "config": config,
            "model_config": model_config_value,
            "decay_exempt": decay_exempt,
            "clip_max_norm": clip_max_norm_value,
            "sampler_rng": rng.state(),
            "evaluations": evals.0,
            "evaluations_heldout": evals.1,
            "evaluations_screen2": evals.2,
            "train_draws": draws,
            "elapsed_seconds": elapsed,
            "started_at": started,
            "resumed_from": resumed,
            "git_head": provenance["git_head"],
            "engine_head": provenance["engine_head"],
            "engine_build_head": provenance["engine_build_head"],
            "engine_build_dirty": provenance["engine_build_dirty"],
            "engine_binary_sha256": provenance["engine_binary_sha256"],
            "engine_stale_allowed": provenance["engine_stale_allowed"],
            "training_authorized": false,
        })
    };
    let mut last_finite: Option<u64> = None;
    for update in first_update..=updates {
        let batch_idx = rng.sample(d.train.len(), microbatch.min(d.train.len()));
        let batch: Vec<&hf_io::RealEpisode> = batch_idx.iter().map(|i| &d.train[*i]).collect();
        for e in &batch {
            *train_draws.entry(e.episode_id.clone()).or_default() += 1;
        }
        let indexes: Vec<EpisodeIndex> = batch
            .iter()
            .map(|e| EpisodeIndex::new_with_query(e, &d.embeddings, d.dim, d.query_vectors()))
            .collect::<Result<_, _>>()?;
        let refs: Vec<&EpisodeIndex> = indexes.iter().collect();
        optimiser.zero_grad();
        let walks = {
            let mut scorer = ModelScorer { model };
            walk_batch(
                &refs,
                features.as_ref(),
                &mut scorer,
                WalkOptions {
                    stop_rule: StopRule::Exhaust,
                    record_candidates: false,
                    keep_items: true,
                    with_prior,
                },
            )?
        };
        let out = model.forward_training(&walks)?;
        let mut losses = walk_losses(&out, &walks, &refs, with_prior, loss_cfg)?;
        if inject_nan_at == Some(update) {
            // a multiply, not an add: the local gradient of `x + NaN` is 1, so
            // only a product carries the NaN back into the parameters' grads
            losses.total = f64::NAN * &losses.total;
        }
        losses.total.backward();
        let grad_norm = clip_grad_norm(&model.vs, clip);
        let v = losses.values();
        // serde_json writes a non-finite f64 as `null`, so a NaN or infinite
        // loss term — or a pre-clip gradient norm that has already blown up —
        // would be indistinguishable in updates.jsonl from a key that was never
        // written. Every row says outright whether this update was finite, and
        // the first that is not stops the run before the optimiser can carry the
        // NaN into the weights (a `clip_max_norm: null` arm has nothing else to
        // bound it).
        let mut non_finite: Vec<&str> = Vec::new();
        for (name, x) in ["edge", "distance", "stop", "residual", "total"]
            .iter()
            .zip(v.iter())
        {
            if !x.is_finite() {
                non_finite.push(name);
            }
        }
        if !grad_norm.is_finite() {
            non_finite.push("grad_norm");
        }
        let finite = non_finite.is_empty();
        if finite {
            optimiser.step();
        }
        let row = json!({
            "update": update,
            "edge": v[0], "distance": v[1], "stop": v[2], "residual": v[3], "total": v[4],
            "grad_norm": grad_norm,
            "registered": walks.iter().filter(|w| w.registered()).count() as f64 / walks.len() as f64,
            "expansions": walks.iter().map(|w| w.expansions() as f64).sum::<f64>() / walks.len() as f64,
            "draws": train_draws.values().sum::<u64>(),
            "distinct_seen": train_draws.len(),
            "seconds": ((elapsed_before + t0.elapsed().as_secs_f64()) * 10.0).round() / 10.0,
            // after the step: the prior's temperature as this update left it
            // (null when the model has no greedy prior)
            "greedy_tau": model.greedy_tau(),
            // every loss term and the pre-clip norm above are finite numbers,
            // and the optimiser stepped; false means the step was SKIPPED and
            // this is the run's last row
            "finite": finite,
        });
        log.write_all(row.to_string().as_bytes())
            .and_then(|_| log.write_all(b"\n"))
            .and_then(|_| log.flush())
            .map_err(|e| HfError::Invalid(e.to_string()))?;
        if !finite {
            // The weights are the last finite ones — this update's step was
            // skipped — but the sampler and train_draws have both consumed this
            // update's batch, so the checkpoint says `update` and a resume would
            // go on at `update + 1`. That is why `halted` is written beside it:
            // the state is sound, the run is not to be continued blind.
            let meta = checkpoint_meta(
                update,
                &rng,
                (&evaluations, &evaluations_heldout, &evaluations_screen2),
                &train_draws,
                elapsed_before + t0.elapsed().as_secs_f64(),
                &started,
                &resumed_from,
            );
            latest.write(model, Some(&optimiser), &meta)?;
            let instability = json!({
                "record_kind": format!("hippo13_instability_v1{}", if args.fixture { "_FIXTURE" } else { "" }),
                "evidence": !args.fixture,
                "update": update,
                "non_finite": non_finite,
                // the last finite update SEEN BY THIS PROCESS; null when the
                // first update it ran was the one that blew up — `first_update`
                // tells a resumed run's null from a fresh run's
                "last_finite_update": last_finite,
                "first_update": first_update,
                "model_seed": args.model_seed.expect("required"),
                "clip_max_norm": clip_max_norm_value,
                "family": d.family,
                "halted_at": hf_core::utc_now_iso(),
                "checkpoint_latest_update": update,
                "injected": inject_nan_at == Some(update),
            });
            std::fs::write(
                output.join("instability.json"),
                hf_core::files::python_json_pretty(&instability),
            )
            .map_err(|e| HfError::Invalid(e.to_string()))?;
            // the marker a launcher can test: no probe.json was written, so a
            // watcher that only asks "is probe.json there?" would restart this
            // seed for ever
            std::fs::write(
                output.join("halted"),
                format!("non-finite update {update}: {}\n", non_finite.join(" ")),
            )
            .map_err(|e| HfError::Invalid(e.to_string()))?;
            return Err(HfError::BandH(format!("non-finite update {update}")));
        }
        last_finite = Some(update);
        if update % args.eval_every == 0 || update == updates {
            let ev = eval::evaluate(
                model,
                &d.screen,
                &d.embeddings,
                d.dim,
                false,
                d.query_vectors(),
            )?;
            eval::write_evaluation_rows(output, "screen", &Value::from(update), &ev.rows)?;
            let mut report = ev.report;
            report["update"] = update.into();
            evaluations.push(report.clone());
            if !d.screen2.is_empty() {
                let ev2 = eval::evaluate(
                    model,
                    &d.screen2,
                    &d.embeddings,
                    d.dim,
                    false,
                    d.query_vectors(),
                )?;
                eval::write_evaluation_rows(output, "screen2", &Value::from(update), &ev2.rows)?;
                let mut r2 = ev2.report;
                r2["update"] = update.into();
                evaluations_screen2.push(r2);
            }
            if let Some(h) = &d.heldout {
                let evh = eval::evaluate(
                    model,
                    &h.episodes,
                    &h.embeddings,
                    d.dim,
                    false,
                    d.query_vectors(),
                )?;
                eval::write_evaluation_rows(output, "heldout", &Value::from(update), &evh.rows)?;
                let mut rh = evh.report;
                rh["update"] = update.into();
                println!(
                    "[{} s{}] update {update}/{updates} held-out {}: exhaust reg {:.3} exp {:.2} | blind {:.2} simgreedy {:.2} bidir {:.2} oracle {:.2}",
                    d.family, args.model_seed.expect("required"), h.family,
                    rh["exhaust"]["registered"].as_f64().unwrap_or(f64::NAN),
                    rh["exhaust"]["expansions_mean"].as_f64().unwrap_or(f64::NAN),
                    rh["baselines"]["blind_exhaust"]["expansions_mean"].as_f64().unwrap_or(f64::NAN),
                    rh["baselines"]["similarity_greedy"]["expansions_mean"].as_f64().unwrap_or(f64::NAN),
                    rh["baselines"]["bidirectional_bfs"]["expansions_mean"].as_f64().unwrap_or(f64::NAN),
                    rh["baselines"]["oracle"]["expansions_mean"].as_f64().unwrap_or(f64::NAN),
                );
                evaluations_heldout.push(rh);
            }
            println!(
                "[{} s{}] update {update}/{updates} total {:.4} | learned reg {:.3} exp {:.2} | bidir exp {:.2} simgreedy exp {:.2} oracle exp {:.2}",
                d.family, args.model_seed.expect("required"), v[4],
                report["learned"]["registered"].as_f64().unwrap_or(f64::NAN),
                report["learned"]["expansions_mean"].as_f64().unwrap_or(f64::NAN),
                report["baselines"]["bidirectional_bfs"]["expansions_mean"].as_f64().unwrap_or(f64::NAN),
                report["baselines"]["similarity_greedy"]["expansions_mean"].as_f64().unwrap_or(f64::NAN),
                report["baselines"]["oracle"]["expansions_mean"].as_f64().unwrap_or(f64::NAN),
            );
        }
        let due_by_count = args.checkpoint_every > 0 && update % args.checkpoint_every == 0;
        let due_by_time = args.checkpoint_minutes > 0.0
            && last_checkpoint.elapsed().as_secs_f64() >= args.checkpoint_minutes * 60.0;
        if due_by_count || due_by_time {
            let meta = checkpoint_meta(
                update,
                &rng,
                (&evaluations, &evaluations_heldout, &evaluations_screen2),
                &train_draws,
                elapsed_before + t0.elapsed().as_secs_f64(),
                &started,
                &resumed_from,
            );
            latest.write(model, Some(&optimiser), &meta)?;
            last_checkpoint = Instant::now();
        }
    }
    let _ = foundation;
    let histogram = views_histogram(train_draws.values().copied());
    let probe = json!({
        "record_kind": format!("real_walk_stage0_probe{}", if args.fixture { "_FIXTURE" } else { "" }),
        "evidence": !args.fixture,
        "governed_by": governed_by(config),
        "git_head": head_at_start,
        "git_head_at_finish": data::git_head(foundation),
        "engine_head": provenance["engine_head"],
        "engine_build_head": provenance["engine_build_head"],
        "engine_build_dirty": provenance["engine_build_dirty"],
        "engine_binary_sha256": provenance["engine_binary_sha256"],
        "engine_stale_allowed": provenance["engine_stale_allowed"],
        "engine": provenance["engine"],
        "preregistration_commit": args.preregistration_commit,
        "started_at": started,
        "finished_at": hf_core::utc_now_iso(),
        "family": d.family,
        "model_seed": args.model_seed.expect("required"),
        "device": if model.device() == Device::Cpu { "cpu" } else { "cuda" },
        "sampler": d.sampler.as_value(),
        "config": config,
        "decay_exempt": decay_exempt,
        "clip_max_norm": clip_max_norm_value,
        "graph_manifest": d.graph_manifest,
        "embedding_manifest": d.embedding_manifest,
        "query_source": d.query_source.as_str(),
        "query_embeddings": args.query_embeddings_dir.as_ref().map(|p| p.to_string_lossy().to_string()),
        "query_embedding_manifest": d.query_embedding_manifest,
        "query_precheck": d.query_precheck,
        "embedding_coverage": d.embedding_coverage,
        "capacity": capacity,
        "updates": updates,
        "train_episodes": d.train.len(),
        "train_episodes_cap": args.train_episodes,
        // P1: one record per --splits-dir, in the order given, carrying the
        // `subgraph_size` and `target_distance` a cumulative pool is no longer
        // checked on, and the share of the pool each one realised
        "train_pools": d.train_pools,
        "train_draws": {
            "draws": train_draws.values().sum::<u64>(),
            "distinct_seen": train_draws.len(),
            "views_histogram": histogram,
        },
        "train_dropped": d.train_dropped,
        "screen_episodes": d.screen.len(),
        "screen_dropped": d.screen_dropped,
        "evaluations": evaluations,
        "split_manifests": d.split_manifests,
        "heldout": d.heldout.as_ref().map(|h| h.meta.clone()),
        "evaluations_heldout": evaluations_heldout,
        "screen2_episodes": d.screen2.len(),
        "screen2_skip": if d.screen2.is_empty() { Value::Null } else { Value::from(args.screen2_skip) },
        "screen2_dropped": d.screen2_dropped,
        "evaluations_screen2": evaluations_screen2,
        "resumed_from": resumed_from,
        "deterministic": deterministic,
        "training_authorized": false,
    });
    std::fs::write(
        output.join("probe.json"),
        hf_core::files::python_json_pretty(&probe),
    )
    .map_err(|e| HfError::Invalid(e.to_string()))?;
    if args.save_checkpoint {
        let final_ck = Checkpoint {
            dir: output.clone(),
            stem: "checkpoint".into(),
        };
        let meta = json!({
            "record_kind": "hippo13_checkpoint_v1",
            "config": model_config_value,
            "decay_exempt": decay_exempt,
            "clip_max_norm": clip_max_norm_value,
            "seed": args.model_seed.expect("required"),
            "train_draws": train_draws,
            "update": updates,
            "preregistration_commit": args.preregistration_commit,
            "git_head": provenance["git_head"],
            "engine_head": provenance["engine_head"],
            "engine_build_head": provenance["engine_build_head"],
            "engine_build_dirty": provenance["engine_build_dirty"],
            "engine_binary_sha256": provenance["engine_binary_sha256"],
            "engine_stale_allowed": provenance["engine_stale_allowed"],
            "training_authorized": false,
        });
        final_ck.write(model, None, &meta)?;
    }
    Ok(())
}
