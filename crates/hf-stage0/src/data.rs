//! What a run reads before it trains: the config, the splits (from disk, or
//! sampled in-process for the fixture world), the embedding caches, the
//! held-out family — with the runner's refusals.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use hf_core::HfError;
use hf_embed::EmbeddingMatrix;
use hf_episodes::{sample_split, Sampler, SamplerConfig};
use hf_io::RealEpisode;
use serde_json::{json, Map, Value};

pub struct Heldout {
    pub family: String,
    pub episodes: Vec<RealEpisode>,
    pub embeddings: EmbeddingMatrix,
    pub meta: Value,
}

pub struct Data {
    pub family: String,
    pub dim: usize,
    pub train: Vec<RealEpisode>,
    pub screen: Vec<RealEpisode>,
    pub screen2: Vec<RealEpisode>,
    pub embeddings: EmbeddingMatrix,
    pub graph_manifest: Value,
    pub embedding_manifest: Value,
    pub split_manifests: Map<String, Value>,
    pub train_dropped: Value,
    pub screen_dropped: Value,
    pub screen2_dropped: Value,
    pub heldout: Option<Heldout>,
    pub sampler: SamplerConfig,
}

pub fn read_json(path: &Path) -> Result<Value, HfError> {
    serde_json::from_str(
        &std::fs::read_to_string(path)
            .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))?,
    )
    .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))
}

/// `SamplerConfig` from a training config's `sampler` block.
pub fn sampler_from_config(family: &str, block: &Value) -> Result<SamplerConfig, HfError> {
    let get = |k: &str| block.get(k);
    let mut c = SamplerConfig::new(
        family,
        get("subgraph_size")
            .and_then(Value::as_u64)
            .ok_or_else(|| HfError::Invalid("sampler.subgraph_size".into()))? as u32,
        get("target_distance")
            .and_then(Value::as_u64)
            .ok_or_else(|| HfError::Invalid("sampler.target_distance".into()))? as u32,
        get("removal_level")
            .and_then(Value::as_u64)
            .ok_or_else(|| HfError::Invalid("sampler.removal_level".into()))? as u32,
    );
    c.cost_epsilon = get("cost_epsilon").and_then(Value::as_f64).unwrap_or(0.5);
    c.max_paths = get("max_paths").and_then(Value::as_u64).unwrap_or(512) as u32;
    c.seed_label = get("seed_label")
        .and_then(Value::as_str)
        .unwrap_or("real-walk-v1")
        .to_string();
    Ok(c)
}

/// `load_split_from_disk`: the manifest's sampler must agree with the config
/// on size, distance, level and epsilon (band H otherwise); drops come from
/// `sampling.json` when present, else the manifest.
pub fn load_split_from_disk(
    root: &Path,
    split: &str,
    block: &Value,
) -> Result<(Vec<RealEpisode>, Value, Value), HfError> {
    let dir = root.join(split);
    let (episodes, artifacts) = hf_io::read_split(&dir)?;
    let manifest = artifacts.public;
    let recorded = &manifest["sampler"];
    for key in [
        "subgraph_size",
        "target_distance",
        "removal_level",
        "cost_epsilon",
    ] {
        let want = block.get(key).cloned().unwrap_or(if key == "cost_epsilon" {
            json!(0.5)
        } else {
            Value::Null
        });
        let got = recorded.get(key).cloned().unwrap_or(Value::Null);
        let same = match (want.as_f64(), got.as_f64()) {
            (Some(a), Some(b)) => (a - b).abs() < 1e-12,
            _ => want == got,
        };
        if !same {
            return Err(HfError::BandH(format!(
                "{}: sampler {key} {got} differs from the config's {want}",
                dir.display()
            )));
        }
    }
    let drops = match read_json(&dir.join("sampling.json")) {
        Ok(s) => s.get("drops").cloned().unwrap_or(json!({})),
        Err(_) => recorded.get("drops").cloned().unwrap_or(json!({})),
    };
    Ok((episodes, drops, manifest))
}

fn sampled_to_episode(s: hf_episodes::Sampled) -> Result<RealEpisode, HfError> {
    let visible: hf_io::Visible =
        serde_json::from_value(s.visible).map_err(|e| HfError::Invalid(e.to_string()))?;
    let hidden: hf_io::Hidden =
        serde_json::from_value(s.hidden).map_err(|e| HfError::Invalid(e.to_string()))?;
    Ok(RealEpisode {
        episode_id: s.episode_id,
        visible,
        hidden,
    })
}

fn drops_value(drops: &BTreeMap<&'static str, u64>) -> Value {
    Value::Object(
        drops
            .iter()
            .map(|(k, v)| ((*k).to_string(), Value::from(*v)))
            .collect(),
    )
}

pub struct Inputs<'a> {
    pub family: Option<&'a str>,
    pub splits_dir: Option<&'a Path>,
    pub graph_dir: Option<&'a Path>,
    pub embeddings_dir: Option<&'a Path>,
    pub heldout_family: Option<&'a str>,
    pub heldout_splits_dir: Option<&'a Path>,
    pub heldout_embeddings_dir: Option<&'a Path>,
    pub screen2_splits_dir: Option<&'a Path>,
    pub screen2_skip: usize,
    pub train_episodes: Option<usize>,
    pub screen_episodes: usize,
    pub fixture: bool,
    pub model_seed: u64,
}

/// The fixture world: sampled in-process, CPU, never evidence.
fn fixture_data(inputs: &Inputs, block: &Value) -> Result<Data, HfError> {
    let (graph, embeddings) =
        hf_episodes::fixture::fixture_world(inputs.model_seed as u128, 400, 1600, 8);
    let sampler = sampler_from_config("fixture", block)?;
    let mut s = Sampler::new(&graph, sampler.clone())?;
    s.prepare("train")?;
    s.prepare("screen")?;
    let train = sample_split(&s, "train", inputs.train_episodes.unwrap_or(2000), None, 64)?;
    let screen = sample_split(&s, "screen", inputs.screen_episodes, None, 64)?;
    let mut names: Vec<String> = embeddings.keys().cloned().collect();
    names.sort();
    let mut data = Vec::with_capacity(names.len() * 8);
    for n in &names {
        data.extend(embeddings[n].iter().map(|x| *x as f32));
    }
    let matrix = EmbeddingMatrix::from_rows(names, 8, data);
    let graph_manifest = json!({
        "record_kind": "real_graph_manifest_v5",
        "family": "fixture",
        "node_count": graph.node_count(),
        "edge_count": graph.edge_count(),
        "typed": false,
        "text_coverage": 1.0,
        "training_authorized": false,
    });
    Ok(Data {
        family: "fixture".into(),
        dim: 8,
        train: train
            .episodes
            .into_iter()
            .map(sampled_to_episode)
            .collect::<Result<_, _>>()?,
        screen: screen
            .episodes
            .into_iter()
            .map(sampled_to_episode)
            .collect::<Result<_, _>>()?,
        screen2: Vec::new(),
        embeddings: matrix,
        graph_manifest,
        embedding_manifest: json!({"model": "fixture-random", "model_digest": "fixture", "dimension": 8}),
        split_manifests: Map::new(),
        train_dropped: drops_value(&train.drops),
        screen_dropped: drops_value(&screen.drops),
        screen2_dropped: json!({}),
        heldout: None,
        sampler,
    })
}

pub fn load(inputs: &Inputs, config: &Value) -> Result<Data, HfError> {
    let block = &config["sampler"];
    if inputs.fixture {
        if inputs.screen2_splits_dir.is_some() {
            return Err(HfError::Invalid(
                "--screen2-splits-dir needs --splits-dir".into(),
            ));
        }
        return fixture_data(inputs, block);
    }
    let (Some(family), Some(edir)) = (inputs.family, inputs.embeddings_dir) else {
        return Err(HfError::Invalid(
            "a real run needs --family, --embeddings-dir and --splits-dir".into(),
        ));
    };
    let Some(splits_dir) = inputs.splits_dir else {
        return Err(HfError::Invalid(
            "a real run needs --splits-dir (sampling from --graph-dir in-process is not offered by this engine; write the splits with hf-splits)".into(),
        ));
    };
    let _ = inputs.graph_dir;
    let embedding_manifest = read_json(&edir.join("manifest.json"))?;
    let dim = embedding_manifest["dimension"]
        .as_u64()
        .ok_or_else(|| HfError::Invalid("embedding manifest lacks dimension".into()))?
        as usize;
    let embeddings = EmbeddingMatrix::load(edir)?;
    let graph_manifest = read_json(&splits_dir.join("train").join("graph.manifest.json"))?;
    let sampler = sampler_from_config(family, block)?;
    let mut split_manifests = Map::new();
    let (mut train, train_dropped, m) = load_split_from_disk(splits_dir, "train", block)?;
    split_manifests.insert("train".into(), m);
    let (mut screen, screen_dropped, m) = load_split_from_disk(splits_dir, "screen", block)?;
    split_manifests.insert("screen".into(), m);
    if let Some(cap) = inputs.train_episodes {
        if cap < train.len() {
            println!(
                "[{family}] training pool capped at {cap} of {} episodes",
                train.len()
            );
            train.truncate(cap);
        }
    }
    if inputs.screen_episodes < screen.len() {
        screen.truncate(inputs.screen_episodes);
    }
    let mut screen2 = Vec::new();
    let mut screen2_dropped = json!({});
    if let Some(dir) = inputs.screen2_splits_dir {
        let (all, dropped, m) = load_split_from_disk(dir, "screen", block)?;
        split_manifests.insert("screen2".into(), m);
        screen2_dropped = dropped;
        screen2 = all.into_iter().skip(inputs.screen2_skip).collect();
        if screen2.is_empty() {
            return Err(HfError::BandH(
                "screen2 has no episodes beyond --screen2-skip".into(),
            ));
        }
    }
    if train.is_empty() || screen.is_empty() {
        return Err(HfError::Invalid(format!(
            "no episodes: train dropped {train_dropped}, screen dropped {screen_dropped}"
        )));
    }
    let heldout = match inputs.heldout_splits_dir {
        None => None,
        Some(h_root) => {
            let (Some(h_family), Some(h_edir)) =
                (inputs.heldout_family, inputs.heldout_embeddings_dir)
            else {
                return Err(HfError::Invalid(
                    "--heldout-splits-dir needs --heldout-family and --heldout-embeddings-dir"
                        .into(),
                ));
            };
            if h_family == family {
                return Err(HfError::BandH(
                    "the held-out family is the training family".into(),
                ));
            }
            let (h_screen, artifacts) = hf_io::read_split(&h_root.join("screen"))?;
            let h_manifest = artifacts.public;
            let h_emb_manifest = read_json(&h_edir.join("manifest.json"))?;
            if h_emb_manifest["dimension"].as_u64() != Some(dim as u64) {
                return Err(HfError::BandH(
                    "held-out embeddings have a different dimension".into(),
                ));
            }
            let train_family = split_manifests["train"]["family"].as_str().unwrap_or("");
            if h_family == train_family {
                return Err(HfError::BandH(
                    "the held-out family is present in training".into(),
                ));
            }
            let meta = json!({
                "family": h_family,
                "screen_episodes": h_screen.len(),
                "screen_visible_sha256": h_manifest.get("visible_sha256"),
                "screen_sampler": h_manifest.get("sampler"),
                "embedding_manifest": h_emb_manifest,
            });
            Some(Heldout {
                family: h_family.to_string(),
                episodes: h_screen,
                embeddings: EmbeddingMatrix::load(h_edir)?,
                meta,
            })
        }
    };
    Ok(Data {
        family: family.to_string(),
        dim,
        train,
        screen,
        screen2,
        embeddings,
        graph_manifest,
        embedding_manifest,
        split_manifests,
        train_dropped,
        screen_dropped,
        screen2_dropped,
        heldout,
        sampler,
    })
}

/// `git rev-parse HEAD` of a repository, or `"unknown"`.
pub fn git_head(root: &Path) -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

/// The engine's own repository (this workspace), for provenance.
pub fn engine_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The engine's git HEAD when this binary was BUILT, embedded by `build.rs`.
/// `engine_head` — the key every artifact has always carried — is instead the
/// checkout's HEAD at RUN time, and the two differ for every run made with a
/// binary older than the tree beside it, which is exactly the confusion that
/// put one commit in the artifacts and another in the binary.
pub const ENGINE_BUILD_HEAD: &str = env!("ENGINE_BUILD_HEAD");
/// When the build ran (ISO-8601 UTC), embedded by `build.rs`.
pub const ENGINE_BUILD_TIME: &str = env!("ENGINE_BUILD_TIME");

/// Whether `git status --porcelain` was non-empty when this binary was built —
/// the sources it was compiled from are then in no commit at all.
pub fn engine_build_dirty() -> bool {
    env!("ENGINE_BUILD_DIRTY") == "true"
}

/// The engine checkout's HEAD at RUN time: what `engine_head` has always meant.
///
/// `HF_TEST_ENGINE_HEAD_OVERRIDE` replaces it, so a test can put the run-time
/// head out of step with the build's without committing anything. It is read
/// only when `debug_assertions` are on — a release binary, the one that writes
/// evidence, ignores it entirely — and a run that honours it says so on stderr.
pub fn engine_head() -> String {
    if cfg!(debug_assertions) {
        if let Ok(v) = std::env::var("HF_TEST_ENGINE_HEAD_OVERRIDE") {
            if !v.is_empty() {
                eprintln!("hf-stage0: HF_TEST_ENGINE_HEAD_OVERRIDE={v} (debug build; test only)");
                return v;
            }
        }
    }
    git_head(&engine_root())
}

/// `sha256:<hex>` of the running binary itself, read once from
/// `current_exe()`; `"unknown"` if it cannot be read. Two artifacts naming the
/// same digest were written by the same bytes, whatever their heads say.
pub fn engine_binary_sha256() -> &'static str {
    static SHA: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SHA.get_or_init(|| {
        std::env::current_exe()
            .ok()
            .and_then(|p| hf_core::sha256_file(&p).ok())
            .map(|(_, digest)| digest)
            .unwrap_or_else(|| "unknown".into())
    })
}
