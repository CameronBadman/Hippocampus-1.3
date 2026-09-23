//! What a run reads before it trains: the config, the splits (from disk, or
//! sampled in-process for the fixture world), the embedding caches, the
//! held-out family — with the runner's refusals.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use hf_core::HfError;
use hf_embed::EmbeddingMatrix;
use hf_episodes::{sample_split, Sampler, SamplerConfig};
use hf_io::RealEpisode;
use hf_walk::{QuerySource, QueryVectors};
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
    /// Where this run's query comes from, and — under `episode_query` — the
    /// sidecar of question vectors keyed by episode id.
    pub query_source: QuerySource,
    pub queries: Option<EmbeddingMatrix>,
    /// What the run read of both caches, for the artifacts.
    pub query_embedding_manifest: Value,
    /// Which splits the question-vector pre-check demanded a vector for, and
    /// why — a re-evaluation that walks no training episode is checked on the
    /// screens and the vault only, and the artifact says so rather than
    /// leaving a narrowed check invisible.
    pub query_precheck: Value,
    pub embedding_coverage: Value,
    pub graph_manifest: Value,
    pub embedding_manifest: Value,
    pub split_manifests: Map<String, Value>,
    /// P1: one record per `--splits-dir`, in the order given — its path, the
    /// episodes it drew, the `subgraph_size` and `target_distance` it was
    /// drawn at (which a cumulative pool is no longer checked on), how many of
    /// its episodes the cap left in the pool and the share of the pool they
    /// are. One `--splits-dir` gives a one-element array.
    pub train_pools: Value,
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

/// The sampler keys a pool's manifest must agree with the config on.
pub const SAMPLER_AGREEMENT: [&str; 4] = [
    "subgraph_size",
    "target_distance",
    "removal_level",
    "cost_epsilon",
];

/// The two of them a CUMULATIVE training pool is exempted from: a pool drawn
/// at an earlier rung is a smaller ball at a shorter distance by definition,
/// which is the whole point of stacking it under a later rung's config. They
/// are then recorded per pool in `probe.json` (`train_pools`) instead of
/// checked, so nothing is lost — the numbers move from a refusal to the
/// artifact. `removal_level` and `cost_epsilon` are NOT exempt: those describe
/// how hard each episode is cut, and a pool cut by another rule is not the
/// same experiment.
pub const CUMULATIVE_EXEMPT: [&str; 2] = ["subgraph_size", "target_distance"];

/// `load_split_from_disk`: the manifest's sampler must agree with the config
/// on size, distance, level and epsilon (band H otherwise); drops come from
/// `sampling.json` when present, else the manifest.
pub fn load_split_from_disk(
    root: &Path,
    split: &str,
    block: &Value,
) -> Result<(Vec<RealEpisode>, Value, Value), HfError> {
    load_split_from_disk_exempting(root, split, block, &[])
}

/// The same read with `exempt` sampler keys left unchecked — the cumulative
/// pool's relaxation, spelled at the call site so a single `--splits-dir`
/// passes `&[]` and is checked exactly as it always was.
pub fn load_split_from_disk_exempting(
    root: &Path,
    split: &str,
    block: &Value,
    exempt: &[&str],
) -> Result<(Vec<RealEpisode>, Value, Value), HfError> {
    let dir = root.join(split);
    let (episodes, artifacts) = hf_io::read_split(&dir)?;
    let manifest = artifacts.public;
    let recorded = &manifest["sampler"];
    for key in SAMPLER_AGREEMENT {
        if exempt.contains(&key) {
            continue;
        }
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
    /// Every `--splits-dir`, in the order the command line gave them. The
    /// FIRST is the run's own split: its `screen` is the evaluation set, its
    /// `graph.manifest.json` the run's graph manifest and its `train` the head
    /// of the training pool. Each further one contributes its `train` split to
    /// the pool, appended in order and nothing else.
    pub splits_dirs: &'a [PathBuf],
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
    /// `data.query_source`, already parsed from the config.
    pub query_source: QuerySource,
    /// The sidecar of question vectors, keyed by EPISODE id.
    pub query_embeddings_dir: Option<&'a Path>,
    /// The node-coverage floor `--expect-embedding-coverage` demands.
    pub expect_embedding_coverage: Option<f64>,
    /// Does this run walk its training pool? True for every training run and
    /// for a `--reevaluate-checkpoint` read with `--train-sample`; false for a
    /// re-evaluation that only reads the screens and the vault. It narrows the
    /// question-vector pre-check to the episodes actually walked and nothing
    /// else — no other check moves with it.
    pub walks_training_pool: bool,
}

fn select_screen2<T>(all: Vec<T>, stage: Option<&str>, skip: usize) -> Result<Vec<T>, HfError> {
    if stage == Some("stage1_described_target") && skip != 0 {
        return Err(HfError::BandH(
            "--screen2-skip counts rows after question admission and cannot select source \
             ordinals from a filtered stage-1 split; select the source slice before running the \
             teacher and pass --screen2-skip 0"
                .into(),
        ));
    }
    let selected: Vec<T> = all.into_iter().skip(skip).collect();
    if selected.is_empty() {
        return Err(HfError::BandH(
            "screen2 has no episodes beyond --screen2-skip".into(),
        ));
    }
    Ok(selected)
}

impl Data {
    /// The query provenance to hand every index this run builds.
    pub fn query_vectors(&self) -> QueryVectors<'_> {
        QueryVectors {
            source: self.query_source,
            cache: self.queries.as_ref(),
        }
    }
}

/// The share of the given episodes' DISTINCT visible nodes that the cache
/// holds, refused against the floor when one was named. A node the cache lacks
/// is a silent zero vector in every feature it touches.
fn check_coverage(
    label: &str,
    episodes: &[&[RealEpisode]],
    embeddings: &EmbeddingMatrix,
    floor: Option<f64>,
) -> Result<Value, HfError> {
    let (present, distinct) = hf_embed::coverage(
        episodes
            .iter()
            .flat_map(|split| split.iter())
            .flat_map(|e| e.visible.nodes.iter().map(|n| n.node.as_str())),
        embeddings,
    );
    let share = if distinct == 0 {
        1.0
    } else {
        present as f64 / distinct as f64
    };
    if let Some(floor) = floor {
        println!("[{label}] embedding coverage {share:.6} ({present} of {distinct} nodes)");
        if share < floor {
            return Err(HfError::BandH(format!(
                "[{label}] embedding coverage {share:.6} ({present} of {distinct} distinct \
                 nodes) is below the --expect-embedding-coverage floor {floor}"
            )));
        }
    }
    Ok(json!({"present": present, "distinct": distinct, "share": share, "expected": floor}))
}

/// The question vectors of every episode this run will index: the cache is the
/// node cache's encoder, and every episode id is in it (a missing one exits 2
/// here rather than at the first update).
fn load_queries(
    dir: &Path,
    embeddings_dir: &Path,
    episodes: &[&[RealEpisode]],
) -> Result<(EmbeddingMatrix, Value), HfError> {
    let nodes = hf_embed::read_manifest(embeddings_dir)?;
    let manifest = hf_embed::read_manifest(dir)?;
    hf_embed::same_encoder(&nodes, &manifest)?;
    let matrix = EmbeddingMatrix::load(dir)?;
    let missing: Vec<&str> = episodes
        .iter()
        .flat_map(|split| split.iter())
        .map(|e| e.episode_id.as_str())
        .filter(|id| !matrix.contains(id))
        .collect();
    if !missing.is_empty() {
        return Err(HfError::BandH(format!(
            "{}: {} of the run's episodes have no query vector (first: {})",
            dir.display(),
            missing.len(),
            missing[0]
        )));
    }
    let value = serde_json::to_value(&manifest).map_err(|e| HfError::Invalid(e.to_string()))?;
    Ok((matrix, value))
}

/// The fixture world: sampled in-process, CPU, never evidence.
fn fixture_data(inputs: &Inputs, block: &Value) -> Result<Data, HfError> {
    let (graph, embeddings) =
        hf_episodes::fixture::fixture_world(inputs.model_seed as u128, 400, 1600, 8);
    let sampler = sampler_from_config("fixture", block)?;
    let mut s = Sampler::new(&graph, sampler.clone())?;
    s.prepare("train")?;
    s.prepare("screen")?;
    let sampled_train = sample_split(&s, "train", inputs.train_episodes.unwrap_or(2000), None, 64)?;
    let sampled_screen = sample_split(&s, "screen", inputs.screen_episodes, None, 64)?;
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
    let train: Vec<RealEpisode> = sampled_train
        .episodes
        .into_iter()
        .map(sampled_to_episode)
        .collect::<Result<_, _>>()?;
    let screen: Vec<RealEpisode> = sampled_screen
        .episodes
        .into_iter()
        .map(sampled_to_episode)
        .collect::<Result<_, _>>()?;
    // the fixture world embeds every node it builds, so the share is 1.0 —
    // the flag is honoured here too rather than silently ignored on this path
    let embedding_coverage = check_coverage(
        "fixture",
        &[&train, &screen],
        &matrix,
        inputs.expect_embedding_coverage,
    )?;
    Ok(Data {
        family: "fixture".into(),
        dim: 8,
        train,
        screen,
        screen2: Vec::new(),
        embeddings: matrix,
        // the fixture world samples stage-0 episodes, so it is the default
        // source; `--query-embeddings-dir` is refused with `--fixture`
        query_source: QuerySource::TargetEmbedding,
        queries: None,
        query_embedding_manifest: Value::Null,
        query_precheck: Value::Null,
        embedding_coverage,
        graph_manifest,
        embedding_manifest: json!({"model": "fixture-random", "model_digest": "fixture", "dimension": 8}),
        split_manifests: Map::new(),
        // the fixture world samples its pool in process: one pool, no split
        // directory, and nothing for a reader to attribute a share to
        train_pools: Value::Array(Vec::new()),
        train_dropped: drops_value(&sampled_train.drops),
        screen_dropped: drops_value(&sampled_screen.drops),
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
    let Some(splits_dir) = inputs.splits_dirs.first() else {
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
    // P1: the training pool is the concatenation of every --splits-dir's
    // `train`, in the order given. One pool is the old read exactly — the same
    // strict sampler agreement, the same `train` manifest, the same drops.
    let cumulative = inputs.splits_dirs.len() > 1;
    let exempt: &[&str] = if cumulative { &CUMULATIVE_EXEMPT } else { &[] };
    let mut train: Vec<RealEpisode> = Vec::new();
    let mut train_dropped = json!({});
    let mut pools: Vec<Value> = Vec::new();
    for (i, dir) in inputs.splits_dirs.iter().enumerate() {
        let (mut pool, dropped, m) = load_split_from_disk_exempting(dir, "train", block, exempt)?;
        let recorded = m.get("sampler").cloned().unwrap_or(Value::Null);
        pools.push(json!({
            "index": i,
            "splits_dir": dir.to_string_lossy(),
            "episodes": pool.len(),
            "subgraph_size": recorded.get("subgraph_size"),
            "target_distance": recorded.get("target_distance"),
            "sampler": recorded,
            "visible_sha256": m.get("visible_sha256"),
            "dropped": dropped.clone(),
        }));
        if i == 0 {
            train_dropped = dropped;
            split_manifests.insert("train".into(), m);
        } else {
            split_manifests.insert(format!("train{}", i + 1), m);
        }
        train.append(&mut pool);
    }
    if cumulative {
        // an episode drawn into two pools would be one row of the pool drawn
        // twice, and `train_draws` — keyed by episode id — would fold the two
        // counts into one: the replay the readers check against would then
        // disagree with the run. Refused rather than silently deduplicated.
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        if let Some(e) = train.iter().find(|e| !seen.insert(e.episode_id.as_str())) {
            return Err(HfError::BandH(format!(
                "the cumulative training pool holds {} twice; the pools overlap",
                e.episode_id
            )));
        }
        println!(
            "[{family}] cumulative training pool: {} episodes over {} splits ({})",
            train.len(),
            inputs.splits_dirs.len(),
            pools
                .iter()
                .map(|p| format!(
                    "n={} d={} x{}",
                    p["subgraph_size"], p["target_distance"], p["episodes"]
                ))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
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
    // the realised share of the pool each split contributed, AFTER the cap —
    // the cap cuts the concatenation's tail, so a cap below the first pool's
    // size leaves the later ones at zero, and the artifact says so
    let mut left = train.len();
    for p in pools.iter_mut() {
        let drawn = p["episodes"].as_u64().unwrap_or(0) as usize;
        let used = drawn.min(left);
        left -= used;
        p["episodes_used"] = Value::from(used);
        p["share"] = if train.is_empty() {
            Value::Null
        } else {
            Value::from(used as f64 / train.len() as f64)
        };
    }
    let train_pools = Value::Array(pools);
    if inputs.screen_episodes < screen.len() {
        screen.truncate(inputs.screen_episodes);
    }
    let mut screen2 = Vec::new();
    let mut screen2_dropped = json!({});
    if let Some(dir) = inputs.screen2_splits_dir {
        let (all, dropped, m) = load_split_from_disk(dir, "screen", block)?;
        let stage = m.get("stage").and_then(Value::as_str).map(str::to_string);
        split_manifests.insert("screen2".into(), m);
        screen2_dropped = dropped;
        screen2 = select_screen2(all, stage.as_deref(), inputs.screen2_skip)?;
    }
    if train.is_empty() || screen.is_empty() {
        return Err(HfError::Invalid(format!(
            "no episodes: train dropped {train_dropped}, screen dropped {screen_dropped}"
        )));
    }
    let read: Vec<&[RealEpisode]> = vec![&train, &screen, &screen2];
    let embedding_coverage =
        check_coverage(family, &read, &embeddings, inputs.expect_embedding_coverage)?;
    // the held-out family is loaded BEFORE the query pre-check so its episodes
    // can be part of it (ENG-3): the vault is evaluated with this same query
    // cache, and a missing id used to fail at index build, an hour in.
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
    // Which episodes must have a question vector: the ones this run WALKS.
    // A training run walks the pool, both screens and the vault. A
    // `--reevaluate-checkpoint` read with no `--train-sample` walks no
    // training episode at all, so demanding a vector for the whole pool would
    // make the per-destination `queries/` cache the teacher writes unusable
    // for the zero-shot screen reads that need exactly it. The narrowing never
    // relaxes a refusal for an episode that IS walked, and it is recorded.
    let empty: &[RealEpisode] = &[];
    let heldout_episodes: &[RealEpisode] = match &heldout {
        Some(h) => &h.episodes,
        None => empty,
    };
    // The VAULT is in the pre-check too (ENG-3). It is evaluated with this
    // same query cache, so a vault episode the cache does not cover used to
    // fail at INDEX BUILD inside `hf-walk`, an hour into a run; it now exits 2
    // here, before the first update, exactly as a screen episode does.
    let mut names: Vec<&str> = Vec::new();
    let mut query_read: Vec<&[RealEpisode]> = Vec::new();
    if inputs.walks_training_pool {
        names.push("train");
        query_read.push(&train);
    }
    names.push("screen");
    query_read.push(&screen);
    if !screen2.is_empty() {
        names.push("screen2");
        query_read.push(&screen2);
    }
    if !heldout_episodes.is_empty() {
        names.push("heldout");
        query_read.push(heldout_episodes);
    }
    // null when there is no sidecar to check against, so the artifact never
    // describes a pre-check that did not run
    let query_precheck = match inputs.query_embeddings_dir {
        None => Value::Null,
        Some(_) => json!({
            "walks_training_pool": inputs.walks_training_pool,
            "splits": names,
            "episodes": query_read.iter().map(|s| s.len()).sum::<usize>(),
            "train_episodes_excluded": if inputs.walks_training_pool { 0 } else { train.len() },
            "why": if inputs.walks_training_pool {
                "the run walks its training pool"
            } else {
                "--reevaluate-checkpoint without --train-sample walks no training episode"
            },
        }),
    };
    let (queries, query_embedding_manifest) = match inputs.query_embeddings_dir {
        None => (None, Value::Null),
        Some(dir) => {
            let (matrix, manifest) = load_queries(dir, edir, &query_read)?;
            (Some(matrix), manifest)
        }
    };
    Ok(Data {
        family: family.to_string(),
        dim,
        train,
        screen,
        screen2,
        embeddings,
        query_source: inputs.query_source,
        queries,
        query_embedding_manifest,
        query_precheck,
        embedding_coverage,
        graph_manifest,
        embedding_manifest,
        split_manifests,
        train_pools,
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

#[cfg(test)]
mod tests {
    use super::select_screen2;

    #[test]
    fn stage0_screen2_keeps_the_existing_row_skip() {
        assert_eq!(
            select_screen2(vec![0, 1, 2, 3], Some("stage0_known_target"), 2).unwrap(),
            vec![2, 3]
        );
    }

    #[test]
    fn filtered_stage1_screen2_must_be_sliced_before_admission() {
        let error = select_screen2(
            vec!["admitted-source-401"],
            Some("stage1_described_target"),
            400,
        )
        .unwrap_err();
        assert!(error.to_string().contains("after question admission"));
        assert_eq!(
            select_screen2(
                vec!["admitted-source-401"],
                Some("stage1_described_target"),
                0,
            )
            .unwrap(),
            vec!["admitted-source-401"]
        );
    }
}
