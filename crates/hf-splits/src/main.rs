//! `hf-splits write`: `real_walk_write_splits.py` in Rust — a train split and a
//! screen split under one destination from an adapter's graph directory,
//! structure-only, with the same flags; the greedy-path rule needs an
//! embedding cache; `--greedy-share` applies to the train split only;
//! `--train 0` writes no train split (the screen-extension mode); `--targets 2`
//! draws the k = 2 episode of `K_TARGETS_DESIGN.md` §1 and writes a 6.0.0
//! split, `--targets 1` (the default) draws v1 exactly.
//!
//! `hf-splits baselines`: the model-free k baselines of
//! `K_TARGETS_DESIGN.md` §3 read off ONE split directory — k-greedy with its
//! recompute rule, `k-greedy-frozen` beside it and the branch-and-bound
//! k-oracle — one JSON line per episode, plus a sidecar manifest. It trains
//! nothing, loads no model and links no libtorch: it is `hf-stage0`'s
//! `evaluate` baseline half without the learned walk, which is the only part
//! that needs a checkpoint.
//!
//! `hf-splits prefix-check OLD NEW`: `real_walk_split_prefix_check.py` — the
//! first `episode_count(OLD)` records of both streams of NEW equal OLD's,
//! record for record (digests differ by construction and are not the check);
//! an absent `greedy_share` reads as 1.0, announced once.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use hf_core::{refuse_holdout, HfError};
use hf_episodes::{sample_split, Sampler, SamplerConfig};
use hf_io::EpisodeOut;
use serde_json::{json, Value};

#[derive(Parser, Debug)]
#[command(
    name = "hf-splits",
    about = "write and check real-walk split directories"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// sample and write train/ and screen/ under a destination
    Write(WriteArgs),
    /// read the k baselines off one split directory, one JSON line per episode
    Baselines(BaselinesArgs),
    /// check that NEW's first records reproduce OLD's, stream by stream
    PrefixCheck {
        old: PathBuf,
        new: PathBuf,
        /// print the embedding cache's model and digest beside the result
        #[arg(long)]
        embeddings: Option<PathBuf>,
    },
}

#[derive(Parser, Debug)]
struct WriteArgs {
    #[arg(long)]
    family: String,
    /// the adapter output: edges.tsv, text.tsv, graph.manifest.json
    #[arg(long)]
    graph_dir: PathBuf,
    #[arg(long)]
    subgraph_size: u32,
    #[arg(long)]
    target_distance: u32,
    #[arg(long)]
    removal_level: u32,
    #[arg(long, default_value_t = 0.5)]
    cost_epsilon: f64,
    /// cap ball expansion at this out-degree percentile (e.g. 99); the cap is recorded
    #[arg(long)]
    hub_percentile: Option<f64>,
    /// fraction of nodes held out as the node-disjoint screen region
    #[arg(long, default_value_t = 0.0)]
    screen_region: f64,
    #[arg(long, default_value = "cheapest-first")]
    removal_rule: String,
    /// targets per episode (K_TARGETS_DESIGN.md §1); 1 is v1 and draws as v1 does
    #[arg(long, default_value_t = 1)]
    targets: u32,
    /// the v5 embedding cache; required by greedy-path
    #[arg(long)]
    embeddings: Option<PathBuf>,
    /// share of TRAIN episodes whose removal set is greedy's route; the screen stays at 1.0
    #[arg(long, default_value_t = 1.0)]
    greedy_share: f64,
    /// 0 writes no train split
    #[arg(long, default_value_t = 2000)]
    train: usize,
    #[arg(long, default_value_t = 400)]
    screen: usize,
    #[arg(long)]
    destination: PathBuf,
    /// attempts sampled per parallel chunk
    #[arg(long, default_value_t = 256)]
    chunk: usize,
    /// keep the RNG-independent hub cap here instead of computing it (tests)
    #[arg(long)]
    hub_cap: Option<u32>,
}

#[derive(Parser, Debug)]
struct BaselinesArgs {
    /// ONE split directory — the `train` directory itself, not its parent
    #[arg(long)]
    split_dir: PathBuf,
    /// the v5 embedding cache the similarity key reads
    #[arg(long)]
    embeddings: PathBuf,
    /// an ordinary v5 embedding cache whose ids are EPISODE ids: the question
    /// vectors of a stage-1 split. With it every row gains the greedy walk on
    /// the question and `question_greedy_overshoot`; the manifest records
    /// `query_source: episode_query`.
    #[arg(long)]
    query_embeddings_dir: Option<PathBuf>,
    /// refuse (exit 2) when the share of the split's DISTINCT visible nodes
    /// present in --embeddings is below this; a missing node is a silent zero
    /// vector otherwise. The question vectors are not covered by it: a missing
    /// episode id exits 2 whatever this says.
    #[arg(long)]
    expect_embedding_coverage: Option<f64>,
    /// the JSONL of rows; `<stem>.manifest.json` is written beside it
    #[arg(long)]
    output: PathBuf,
    /// `K_TARGETS_DESIGN.md` §4's fixed budget; the default is the rung's
    /// `n / 2` read off each episode's own sampler block, and a run where the
    /// episodes disagree is refused rather than averaged
    #[arg(long)]
    b_fix: Option<u32>,
    /// read only the first N episodes (0 = every one)
    #[arg(long, default_value_t = 0)]
    limit: usize,
}

fn read_json(path: &Path) -> Result<Value, HfError> {
    serde_json::from_str(
        &std::fs::read_to_string(path)
            .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))?,
    )
    .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))
}

#[allow(clippy::too_many_arguments)]
fn write_one(
    sampler: &Sampler<'_>,
    split: &str,
    count: usize,
    embeddings: Option<&(dyn hf_policies::Embeddings + Sync)>,
    destination: &Path,
    graph_manifest: &Value,
    sampler_block: &Value,
    text_path: &Path,
    chunk: usize,
) -> Result<(), HfError> {
    let sampled = sample_split(sampler, split, count, embeddings, chunk)?;
    let mut nodes: BTreeSet<String> = BTreeSet::new();
    for e in &sampled.episodes {
        nodes.extend(e.nodes.iter().cloned());
    }
    let episodes = sampled.episodes.iter().map(|e| {
        let mut visible = e.visible.clone();
        for node in visible["nodes"].as_array_mut().unwrap() {
            node["text"] = Value::from(""); // text lives in texts.jsonl
        }
        Ok(EpisodeOut {
            episode_id: e.episode_id.clone(),
            visible,
            hidden: e.hidden.clone(),
        })
    });
    hf_io::write_split(
        &sampler.config.family,
        hf_episodes::STAGE0,
        split,
        destination,
        episodes,
        graph_manifest,
        sampler_block,
    )?;
    // one pass over text.tsv for the episode nodes, in file order
    let texts = stream_texts(text_path, &nodes)?;
    let written = hf_io::write_sidecars(
        destination,
        texts,
        &json!({
            "attempts": sampled.attempts,
            "kept": sampled.episodes.len(),
            "drops": sampled.drops,
            "distinct_nodes": nodes.len(),
            "texts_written": 0,
            "training_authorized": false,
        }),
        &nodes.iter().cloned().collect::<Vec<_>>(),
    )?;
    // sampling.json carries the count written; rewrite it with the real number
    let sampling = json!({
        "attempts": sampled.attempts,
        "kept": sampled.episodes.len(),
        "drops": sampled.drops,
        "distinct_nodes": nodes.len(),
        "texts_written": written,
        "training_authorized": false,
    });
    std::fs::write(
        destination.join("sampling.json"),
        hf_core::files::python_json_pretty(&sampling),
    )
    .map_err(|e| HfError::Invalid(e.to_string()))?;
    println!(
        "{split}: kept {} of {} attempts, drops {:?}, {} distinct nodes, {written} texts -> {}",
        sampled.episodes.len(),
        sampled.attempts,
        sampled.drops,
        nodes.len(),
        destination.display()
    );
    Ok(())
}

fn stream_texts(
    text_path: &Path,
    nodes: &BTreeSet<String>,
) -> Result<Vec<(String, String)>, HfError> {
    use std::io::BufRead;
    let file = std::fs::File::open(text_path)
        .map_err(|e| HfError::Invalid(format!("{}: {e}", text_path.display())))?;
    let mut out = Vec::new();
    for line in std::io::BufReader::with_capacity(1 << 20, file).lines() {
        let line = line.map_err(|e| HfError::Invalid(e.to_string()))?;
        let (node, body) = line.split_once('\t').unwrap_or((line.as_str(), ""));
        if nodes.contains(node) {
            out.push((node.to_string(), body.to_string()));
        }
    }
    Ok(out)
}

fn write(args: WriteArgs) -> Result<(), HfError> {
    refuse_holdout(&args.destination)?;
    let graph_manifest = read_json(&args.graph_dir.join("graph.manifest.json"))?;
    let graph =
        hf_graph::RealGraph::from_triples(&args.graph_dir.join("edges.tsv"), &args.family, None)?;
    let hub_cap = match (args.hub_cap, args.hub_percentile) {
        (Some(cap), _) => Some(cap),
        (None, Some(p)) => Some(graph.out_degree_percentile(p)?),
        (None, None) => None,
    };
    let mut config = SamplerConfig::new(
        &args.family,
        args.subgraph_size,
        args.target_distance,
        args.removal_level,
    );
    config.cost_epsilon = args.cost_epsilon;
    config.hub_degree_cap = hub_cap;
    config.screen_region = args.screen_region;
    config.targets = args.targets;
    config.removal_rule = args.removal_rule.clone();
    let embeddings = match (&args.embeddings, args.removal_rule.as_str()) {
        (Some(dir), _) => Some(hf_embed::EmbeddingMatrix::load(dir)?),
        (None, "greedy-path") => {
            return Err(HfError::Invalid(
                "--removal-rule greedy-path needs --embeddings".into(),
            ))
        }
        _ => None,
    };
    let text_path = args.graph_dir.join("text.tsv");
    let sampler_block = |c: &SamplerConfig| {
        let mut v = c.as_value();
        v["text_mode"] = "texts.jsonl".into();
        v["hub_percentile"] = args.hub_percentile.map(Value::from).unwrap_or(Value::Null);
        v
    };
    let emb: Option<&(dyn hf_policies::Embeddings + Sync)> = embeddings
        .as_ref()
        .map(|m| m as &(dyn hf_policies::Embeddings + Sync));
    if args.train > 0 {
        let mut train_config = config.clone();
        train_config.greedy_share = args.greedy_share;
        let mut sampler = Sampler::new(&graph, train_config)?;
        sampler.prepare("train")?;
        let block = sampler_block(&sampler.config);
        write_one(
            &sampler,
            "train",
            args.train,
            emb,
            &args.destination.join("train"),
            &graph_manifest,
            &block,
            &text_path,
            args.chunk,
        )?;
    }
    let mut sampler = Sampler::new(&graph, config)?;
    sampler.prepare("screen")?;
    let block = sampler_block(&sampler.config);
    write_one(
        &sampler,
        "screen",
        args.screen,
        emb,
        &args.destination.join("screen"),
        &graph_manifest,
        &block,
        &text_path,
        args.chunk,
    )?;
    Ok(())
}

fn normalise_sampler(mut v: Value, notes: &mut Vec<String>) -> Value {
    if let Some(obj) = v.as_object_mut() {
        if !obj.contains_key("greedy_share") {
            obj.insert("greedy_share".into(), 1.0.into());
            if notes.is_empty() {
                notes.push(
                    "an absent greedy_share is read as 1.0 (splits written before amendment 6)"
                        .into(),
                );
            }
        }
    }
    v
}

/// One episode's baseline row: the k-greedy examination order and its
/// per-target registrations, the frozen variant beside it, and the k-oracle's
/// `o_k` with the budget's own verdict on it.
///
/// `k_greedy` is `K_TARGETS_DESIGN.md` §3 item 1's baseline — the key is
/// `(-max over UNREGISTERED t of cos(node, t), counter)` with the whole
/// frontier re-keyed at every registration — and at k = 1 it is v1's
/// `similarity_greedy_trace`, trace field for trace field
/// (`crates/hf-policies/tests/k_targets.rs`).
#[derive(serde::Serialize)]
struct BaselineRow<'a> {
    record_kind: &'static str,
    episode_id: &'a str,
    start_node: &'a str,
    targets: &'a [String],
    /// the RUNG's `n` (the sampler block's `subgraph_size`), not the realised ball
    subgraph_size: u32,
    /// the realised ball, as the visible edge list and node list show it
    ball_nodes: usize,
    b_fix: u32,
    k_greedy_examined: &'a [String],
    k_greedy_registered_at: &'a [Option<u32>],
    k_greedy_expansions: u32,
    k_greedy_recall_at_budget: Option<f64>,
    k_greedy_stop_reason: &'a str,
    k_greedy_frozen_examined: &'a [String],
    k_greedy_frozen_registered_at: &'a [Option<u32>],
    k_greedy_frozen_expansions: u32,
    k_greedy_frozen_recall_at_budget: Option<f64>,
    k_greedy_frozen_stop_reason: &'a str,
    /// `o_k`: the k-oracle's expansion count, minimal only where `k_oracle_exact`
    k_oracle_expansions: u32,
    k_oracle_exact: bool,
    k_oracle_lower_bound: u32,
    k_oracle_upper_bound: u32,
    k_oracle_searched: u64,
    /// `|V'|`: the nodes the k-oracle searches (every node on any surviving
    /// path to any target, plus the start)
    k_oracle_pruned_nodes: usize,
    /// the hidden scalar — at k >= 2 the MAXIMUM of the per-target
    /// single-target overshoots, never `g_k - o_k`, which needs the k-oracle
    greedy_overshoot: Option<i64>,
    greedy_overshoots: Option<&'a Vec<i64>>,
    removed_count: u32,
    /// `|nodes_on_surviving_path|` — the union of the per-target path sets
    nodes_on_surviving_path: usize,
    /// Similarity-greedy walking on the EPISODE'S QUESTION, and the
    /// single-target oracle beside it — written only under
    /// `--query-embeddings-dir`, so a run without it writes v1's row byte for
    /// byte. `question_greedy_overshoot` is greedy's expansions minus the
    /// oracle's, the sampler's own definition of `greedy_overshoot` with the
    /// question in the target's place (`hf-episodes/src/lib.rs`), which is what
    /// makes the stage-1 strata a re-derivation rather than a carried label.
    #[serde(skip_serializing_if = "Option::is_none")]
    question_greedy_expansions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    question_greedy_registered_at: Option<Option<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    question_greedy_stop_reason: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    oracle_expansions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    question_greedy_overshoot: Option<i64>,
}

/// The k baselines of `K_TARGETS_DESIGN.md` §3 on one split directory.
///
/// No model, no checkpoint, no training: every baseline here is model-free and
/// reads the visible adjacency plus the hidden labels `hf-stage0`'s own
/// baseline half already reads. The split directory is taken whole — the
/// caller names the ONE directory to read, and a holdout path is refused on
/// every argument.
fn baselines(args: BaselinesArgs) -> Result<(), HfError> {
    use std::io::Write;
    for path in [&args.split_dir, &args.embeddings, &args.output] {
        refuse_holdout(path)?;
    }
    if let Some(q) = &args.query_embeddings_dir {
        refuse_holdout(q)?;
    }
    let (episodes, artifacts) = hf_io::read_split(&args.split_dir)?;
    let embeddings = hf_embed::EmbeddingMatrix::load(&args.embeddings)?;
    let embedding_manifest = hf_embed::read_manifest(&args.embeddings)?;
    // the question vectors, keyed by episode id, and the encoder agreement
    // every `cos(question, node)` rests on
    let (queries, query_manifest) = match &args.query_embeddings_dir {
        None => (None, Value::Null),
        Some(dir) => {
            let manifest = hf_embed::read_manifest(dir)?;
            hf_embed::same_encoder(&embedding_manifest, &manifest)?;
            (
                Some(hf_embed::EmbeddingMatrix::load(dir)?),
                serde_json::to_value(&manifest).map_err(|e| HfError::Invalid(e.to_string()))?,
            )
        }
    };
    let query_source = if queries.is_some() {
        "episode_query"
    } else {
        "target_embedding"
    };
    let take = if args.limit == 0 {
        episodes.len()
    } else {
        args.limit.min(episodes.len())
    };
    let episodes = &episodes[..take];
    if episodes.is_empty() {
        return Err(HfError::Invalid(format!(
            "{}: no episodes to read",
            args.split_dir.display()
        )));
    }
    // the node coverage the caller demanded, over the DISTINCT visible nodes
    // of the episodes actually read
    let (present, distinct) = hf_embed::coverage(
        episodes
            .iter()
            .flat_map(|e| e.visible.nodes.iter().map(|n| n.node.as_str())),
        &embeddings,
    );
    let node_coverage = if distinct == 0 {
        1.0
    } else {
        present as f64 / distinct as f64
    };
    if let Some(floor) = args.expect_embedding_coverage {
        println!("embedding coverage {node_coverage:.6} ({present} of {distinct} nodes)");
        if node_coverage < floor {
            return Err(HfError::BandH(format!(
                "embedding coverage {node_coverage:.6} ({present} of {distinct} distinct nodes) \
                 is below the --expect-embedding-coverage floor {floor}"
            )));
        }
    }
    // B_fix is ONE number per rung (§4), so a split whose episodes disagree is
    // refused rather than averaged; --b-fix names it explicitly instead.
    let mut derived: BTreeSet<u32> = BTreeSet::new();
    for e in episodes {
        derived.insert(hf_policies::rung_ball_size(e) / 2);
    }
    let b_fix = match args.b_fix {
        Some(b) => b,
        None => {
            if derived.len() != 1 {
                return Err(HfError::BandH(format!(
                    "the split's episodes derive more than one B_fix ({derived:?}); name it with --b-fix"
                )));
            }
            *derived.iter().next().expect("one")
        }
    };
    let parent = args
        .output
        .parent()
        .ok_or_else(|| HfError::Invalid("--output has no parent".into()))?;
    if !parent.as_os_str().is_empty() {
        std::fs::create_dir_all(parent).map_err(|e| HfError::Invalid(e.to_string()))?;
    }
    let mut file =
        std::fs::File::create(&args.output).map_err(|e| HfError::Invalid(e.to_string()))?;
    let mut targets_per_episode: BTreeSet<usize> = BTreeSet::new();
    let mut exact = 0u64;
    for e in episodes {
        let g = hf_policies::EpisodeGraph::from_episode_k(e);
        targets_per_episode.insert(g.target_count());
        let greedy = hf_policies::k_greedy_trace(&g, &embeddings);
        let frozen = hf_policies::k_greedy_frozen_trace(&g, &embeddings);
        let oracle = hf_policies::k_oracle(&g);
        if oracle.exact {
            exact += 1;
        }
        // the question walk: similarity-greedy with the episode's own query in
        // the target's place, against the SINGLE-target oracle the sampler's
        // `greedy_overshoot` is measured against
        let question = match &queries {
            None => None,
            Some(cache) => {
                if g.target_count() != 1 {
                    return Err(HfError::BandH(format!(
                        "{}: --query-embeddings-dir reads a k = 1 split; this episode \
                         carries {} targets",
                        e.episode_id,
                        g.target_count()
                    )));
                }
                let q: Vec<f64> = cache
                    .get(&e.episode_id)
                    .ok_or_else(|| {
                        HfError::BandH(format!(
                            "the query cache holds no vector for episode {}",
                            e.episode_id
                        ))
                    })?
                    .iter()
                    .map(|x| *x as f64)
                    .collect();
                let single = hf_policies::EpisodeGraph::from_episode(e);
                let walk = hf_policies::similarity_greedy_trace(&single, &embeddings, Some(&q));
                let floor = hf_policies::oracle_trace(&single);
                Some((walk, floor))
            }
        };
        let row = BaselineRow {
            record_kind: "k_baseline_row",
            episode_id: &e.episode_id,
            start_node: &e.visible.start_node,
            targets: &g.targets,
            subgraph_size: g.subgraph_size,
            ball_nodes: e.visible.nodes.len(),
            b_fix,
            k_greedy_examined: &greedy.examined,
            k_greedy_registered_at: &greedy.registered_at_by_target,
            k_greedy_expansions: greedy.expansions,
            k_greedy_recall_at_budget: greedy.recall_at_budget(b_fix),
            k_greedy_stop_reason: &greedy.stop_reason,
            k_greedy_frozen_examined: &frozen.examined,
            k_greedy_frozen_registered_at: &frozen.registered_at_by_target,
            k_greedy_frozen_expansions: frozen.expansions,
            k_greedy_frozen_recall_at_budget: frozen.recall_at_budget(b_fix),
            k_greedy_frozen_stop_reason: &frozen.stop_reason,
            k_oracle_expansions: oracle.trace.expansions,
            k_oracle_exact: oracle.exact,
            k_oracle_lower_bound: oracle.lower_bound,
            k_oracle_upper_bound: oracle.upper_bound,
            k_oracle_searched: oracle.searched,
            k_oracle_pruned_nodes: oracle.pruned_nodes,
            greedy_overshoot: e.hidden.greedy_overshoot,
            greedy_overshoots: e.hidden.greedy_overshoots.as_ref(),
            removed_count: e.hidden.removed_count,
            nodes_on_surviving_path: e.hidden.nodes_on_surviving_path.len(),
            question_greedy_expansions: question.as_ref().map(|(w, _)| w.expansions),
            question_greedy_registered_at: question.as_ref().map(|(w, _)| w.registered_at),
            question_greedy_stop_reason: question.as_ref().map(|(w, _)| w.stop_reason.as_str()),
            oracle_expansions: question.as_ref().map(|(_, o)| o.expansions),
            question_greedy_overshoot: question
                .as_ref()
                .map(|(w, o)| w.expansions as i64 - o.expansions as i64),
        };
        let line = serde_json::to_string(&row).map_err(|e| HfError::Invalid(e.to_string()))?;
        writeln!(file, "{line}").map_err(|e| HfError::Invalid(e.to_string()))?;
    }
    file.flush().map_err(|e| HfError::Invalid(e.to_string()))?;
    let manifest_path = parent.join(format!(
        "{}.manifest.json",
        args.output
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "baselines".into())
    ));
    let manifest = json!({
        "record_kind": "k_baseline_manifest",
        "engine": "hippo-13 hf-splits baselines",
        "policies": ["k_greedy", "k_greedy_frozen", "k_oracle"],
        "split_dir": args.split_dir.to_string_lossy(),
        "split_schema_version": artifacts.public.get("schema_version").cloned().unwrap_or(Value::Null),
        "visible_sha256": artifacts.public.get("visible_sha256").cloned().unwrap_or(Value::Null),
        "episode_count_in_split": artifacts.public.get("episode_count").cloned().unwrap_or(Value::Null),
        "episodes_read": episodes.len(),
        "targets_per_episode": targets_per_episode.iter().copied().collect::<Vec<usize>>(),
        "b_fix": b_fix,
        "b_fix_source": if args.b_fix.is_some() { "flag" } else { "rung_ball_size / 2" },
        "oracle_state_budget": hf_policies::ORACLE_STATE_BUDGET,
        "oracle_time_limit_ms": hf_policies::ORACLE_TIME_LIMIT.as_millis() as u64,
        "oracle_exact_episodes": exact,
        "query_source": query_source,
        "query_embeddings": args.query_embeddings_dir.as_ref().map(|p| p.to_string_lossy().to_string()),
        "query_embedding_manifest": query_manifest,
        "embedding_manifest": serde_json::to_value(&embedding_manifest).map_err(|e| HfError::Invalid(e.to_string()))?,
        "embedding_coverage": node_coverage,
        "embedding_coverage_nodes": [present, distinct],
        "expect_embedding_coverage": args.expect_embedding_coverage,
        "embeddings": args.embeddings.to_string_lossy(),
        "rows": args.output.file_name().map(|s| s.to_string_lossy().to_string()),
        "model": Value::Null,
        "checkpoint": Value::Null,
        "training_authorized": false,
    });
    std::fs::write(
        &manifest_path,
        hf_core::files::python_json_pretty(&manifest),
    )
    .map_err(|e| HfError::Invalid(e.to_string()))?;
    println!(
        "baselines: {} episodes, B_fix {b_fix}, k-oracle exact on {exact} -> {}",
        episodes.len(),
        args.output.display()
    );
    Ok(())
}

fn prefix_check(old: &Path, new: &Path, embeddings: Option<&Path>) -> Result<bool, HfError> {
    let mut notes = Vec::new();
    let old_art = hf_io::validate_split_artifacts(old)?;
    let new_art = hf_io::validate_split_artifacts(new)?;
    let n = old_art.public["episode_count"].as_u64().unwrap_or(0) as usize;
    let m = new_art.public["episode_count"].as_u64().unwrap_or(0) as usize;
    if m < n {
        println!("MISMATCH: new carries {m} episodes, fewer than old's {n}");
        return Ok(false);
    }
    for key in ["family", "graph_manifest_sha256"] {
        if old_art.public[key] != new_art.public[key] {
            println!("MISMATCH: manifests differ on {key}");
            return Ok(false);
        }
    }
    let old_s = normalise_sampler(old_art.public["sampler"].clone(), &mut notes);
    let new_s = normalise_sampler(new_art.public["sampler"].clone(), &mut notes);
    if hf_core::canonical_bytes(&old_s)? != hf_core::canonical_bytes(&new_s)? {
        println!("MISMATCH: sampler blocks differ");
        return Ok(false);
    }
    let (old_eps, _) = hf_io::read_split(old)?;
    let (new_eps, _) = hf_io::read_split(new)?;
    for (i, (a, b)) in old_eps.iter().zip(&new_eps).enumerate() {
        if a.episode_id != b.episode_id {
            println!(
                "MISMATCH in visible at ordinal {i}: {} vs {}",
                a.episode_id, b.episode_id
            );
            return Ok(false);
        }
        if a.visible != b.visible {
            println!("MISMATCH in visible at ordinal {i}");
            return Ok(false);
        }
        let ha = normalise_sampler(serde_json::to_value(&a.hidden).unwrap(), &mut notes);
        let hb = normalise_sampler(serde_json::to_value(&b.hidden).unwrap(), &mut notes);
        if hf_core::canonical_bytes(&ha)? != hf_core::canonical_bytes(&hb)? {
            println!("MISMATCH in hidden at ordinal {i}");
            return Ok(false);
        }
    }
    for note in notes {
        println!("note: {note}");
    }
    for (label, dir) in [("old", old), ("new", new)] {
        if let Ok(s) = read_json(&dir.join("sampling.json")) {
            println!(
                "{label} sampling: attempts {} kept {} drops {}",
                s["attempts"], s["kept"], s["drops"]
            );
        }
    }
    if let Some(dir) = embeddings {
        let manifest = hf_embed::read_manifest(dir)?;
        println!("embeddings: {} {}", manifest.model, manifest.model_digest);
    }
    println!("OK: the first {n} records of both streams are equal");
    Ok(true)
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Write(args) => write(args),
        Command::Baselines(args) => baselines(args),
        Command::PrefixCheck {
            old,
            new,
            embeddings,
        } => match prefix_check(&old, &new, embeddings.as_deref()) {
            Ok(true) => Ok(()),
            Ok(false) => std::process::exit(2),
            Err(e) => Err(e),
        },
    };
    if let Err(e) = result {
        hf_core::exit_with("hf-splits", &e);
    }
}
