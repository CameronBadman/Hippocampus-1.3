//! `hf-embed`: embed a family's `text.tsv` through a running ollama server into
//! the v5 cache, with `real_walk_embed.py`'s flags and guards. It never starts
//! the server. The manifest is written before the first vector, so a
//! half-run never lacks provenance; an existing cache is resumed only when its
//! served digest and character limit match.

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use clap::Parser;
use hf_core::{refuse_holdout, HfError};
use hf_embed::{
    append_vector, nodes_present, read_manifest, truncate_chars, write_manifest, Manifest,
    OllamaEmbedClient,
};

#[derive(Parser, Debug)]
#[command(
    name = "hf-embed",
    about = "embed node texts into the real-walk v5 cache via ollama"
)]
struct Args {
    /// the adapter's text.tsv (node<TAB>text)
    #[arg(long)]
    text: PathBuf,
    /// the cache directory (manifest.json + vectors.jsonl)
    #[arg(long)]
    destination: PathBuf,
    #[arg(long, default_value = "nomic-embed-text")]
    model: String,
    #[arg(long, default_value = "http://127.0.0.1:11434")]
    base_url: String,
    #[arg(long, default_value_t = 16)]
    batch: usize,
    #[arg(long, default_value_t = 600.0)]
    timeout: f64,
    /// characters of text sent per node; longer texts are cut
    #[arg(long, default_value_t = 6000)]
    max_chars: usize,
    /// refuse unless the served model digest equals this
    #[arg(long)]
    expect_digest: Option<String>,
    /// embed at most this many nodes
    #[arg(long)]
    limit: Option<usize>,
    /// a file of node ids, one per line, restricting which nodes are embedded
    #[arg(long)]
    nodes: Option<PathBuf>,
}

fn run(args: Args) -> Result<(), HfError> {
    refuse_holdout(&args.destination)?;
    let client = OllamaEmbedClient::new(&args.base_url, &args.model, args.timeout);
    let digest = client.served_digest()?;
    if let Some(expected) = &args.expect_digest {
        if expected != &digest {
            return Err(HfError::Refused(format!(
                "served digest {digest} != expected {expected}"
            )));
        }
    }
    std::fs::create_dir_all(&args.destination).map_err(|e| HfError::Invalid(e.to_string()))?;
    let (text_bytes, text_sha256) =
        hf_core::sha256_file(&args.text).map_err(|e| HfError::Invalid(e.to_string()))?;
    let _ = text_bytes;
    let wanted: Option<HashSet<String>> = match &args.nodes {
        Some(path) => Some(
            std::fs::read_to_string(path)
                .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))?
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(str::to_string)
                .collect(),
        ),
        None => None,
    };
    let vectors_path = args.destination.join("vectors.jsonl");
    let (mut done, dimension, mut truncated, mut manifest) = if vectors_path.exists() {
        let previous = read_manifest(&args.destination)?;
        if previous.model_digest != digest {
            return Err(HfError::Refused(format!(
                "cache was built with digest {} but the server serves {digest}; refusing to mix",
                previous.model_digest
            )));
        }
        if previous.text_char_limit != args.max_chars as u64 {
            return Err(HfError::Refused(format!(
                "cache was built with text_char_limit {} but --max-chars is {}; refusing to mix",
                previous.text_char_limit, args.max_chars
            )));
        }
        let done: HashSet<String> = nodes_present(&args.destination)?.into_iter().collect();
        let truncated = previous.truncated.clone();
        (done, previous.dimension as usize, truncated, previous)
    } else {
        let probe = client
            .embed(&["dimension probe".to_string()], 4)?
            .map_err(|_| HfError::Invalid("the dimension probe overflowed the context".into()))?;
        let dimension = probe[0].len();
        let manifest = Manifest {
            record_kind: hf_embed::MANIFEST_KIND.into(),
            model: args.model.clone(),
            model_digest: digest.clone(),
            base_url: args.base_url.clone(),
            dimension: dimension as u32,
            count: 0,
            text_char_limit: args.max_chars as u64,
            text_sha256: Some(text_sha256.clone()),
            truncated: HashMap::new(),
            training_authorized: false,
            extra: serde_json::Map::new(),
        };
        write_manifest(&args.destination, &manifest)?;
        (HashSet::new(), dimension, HashMap::new(), manifest)
    };
    let text_file = std::fs::File::open(&args.text)
        .map_err(|e| HfError::Invalid(format!("{}: {e}", args.text.display())))?;
    let mut out = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(&vectors_path)
        .map_err(|e| HfError::Invalid(format!("{}: {e}", vectors_path.display())))?;
    let mut pending: Vec<(String, String)> = Vec::new();
    let mut written = 0usize;
    let mut seen_requested: HashSet<String> = HashSet::new();
    let flush = |pending: &mut Vec<(String, String)>,
                 out: &mut std::fs::File,
                 written: &mut usize,
                 truncated: &mut HashMap<String, u64>|
     -> Result<(), HfError> {
        if pending.is_empty() {
            return Ok(());
        }
        let texts: Vec<String> = pending.iter().map(|(_, t)| t.clone()).collect();
        let vectors = match client.embed(&texts, 4)? {
            Ok(v) => v,
            Err(_) => {
                // overflow: retry each item alone, halving the text until accepted
                let mut all = Vec::with_capacity(pending.len());
                for (node, text) in pending.iter() {
                    let mut text = text.clone();
                    loop {
                        match client.embed(std::slice::from_ref(&text), 4)? {
                            Ok(v) => {
                                if text.chars().count() < pending_len(&texts, node) {
                                    truncated.insert(node.clone(), text.chars().count() as u64);
                                }
                                all.push(v.into_iter().next().expect("one vector"));
                                break;
                            }
                            Err(_) => {
                                let n = text.chars().count();
                                if n <= 64 {
                                    return Err(HfError::Invalid(format!(
                                        "{node}: refused at {n} characters"
                                    )));
                                }
                                text = truncate_chars(&text, n / 2).to_string();
                            }
                        }
                    }
                }
                all
            }
        };
        for ((node, _), vector) in pending.iter().zip(vectors.iter()) {
            if vector.len() != dimension {
                return Err(HfError::BandH(format!(
                    "{node}: dimension {} != {dimension}",
                    vector.len()
                )));
            }
            append_vector(out, node, vector).map_err(|e| HfError::Invalid(e.to_string()))?;
            *written += 1;
        }
        out.flush().map_err(|e| HfError::Invalid(e.to_string()))?;
        pending.clear();
        Ok(())
    };
    for line in BufReader::with_capacity(1 << 20, text_file).lines() {
        let line = line.map_err(|e| HfError::Invalid(e.to_string()))?;
        let (node, body) = line.split_once('\t').unwrap_or((line.as_str(), ""));
        if let Some(wanted) = &wanted {
            if !wanted.contains(node) {
                continue;
            }
            seen_requested.insert(node.to_string());
        }
        if done.contains(node) {
            continue;
        }
        if let Some(limit) = args.limit {
            if written + pending.len() >= limit {
                break;
            }
        }
        let text = if body.is_empty() { node } else { body };
        pending.push((
            node.to_string(),
            truncate_chars(text, args.max_chars).to_string(),
        ));
        done.insert(node.to_string());
        if pending.len() >= args.batch {
            flush(&mut pending, &mut out, &mut written, &mut truncated)?;
        }
    }
    flush(&mut pending, &mut out, &mut written, &mut truncated)?;
    if let Some(wanted) = &wanted {
        let missing = wanted.len() - seen_requested.len();
        if missing > 0 {
            eprintln!("hf-embed: {missing} requested node ids have no text row");
        }
    }
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(&vectors_path, std::fs::Permissions::from_mode(0o644));
    manifest.count = done.len() as u64;
    manifest.truncated = truncated;
    write_manifest(&args.destination, &manifest)?;
    println!(
        "hf-embed: wrote {written} vectors; cache holds {} (dimension {dimension})",
        manifest.count
    );
    Ok(())
}

fn pending_len(texts: &[String], _node: &str) -> usize {
    texts.iter().map(|t| t.chars().count()).max().unwrap_or(0)
}

fn main() {
    let args = Args::parse();
    if let Err(e) = run(args) {
        hf_core::exit_with("hf-embed", &e);
    }
}
