//! `hf-splits write`: `real_walk_write_splits.py` in Rust — a train split and a
//! screen split under one destination from an adapter's graph directory,
//! structure-only, with the same flags; the greedy-path rule needs an
//! embedding cache; `--greedy-share` applies to the train split only;
//! `--train 0` writes no train split (the screen-extension mode).
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
