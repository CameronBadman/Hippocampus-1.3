//! The identity gate on real data: sample the first attempts of a pool's
//! configuration and compare with the Python-written pool on disk, record for
//! record. `cargo run --release -p hf-episodes --example pool_check -- <graph edges.tsv> <embedding cache> <split dir> <how many kept episodes>`
use std::time::Instant;

fn peak_rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<f64>().ok())
        })
        .map(|kb| kb / 1024.0)
        .unwrap_or(f64::NAN)
}

fn canon(v: &serde_json::Value) -> String {
    String::from_utf8(hf_core::canonical_bytes(v).unwrap()).unwrap()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (edges, cache, split_dir, count) =
        (&args[1], &args[2], &args[3], args[4].parse::<usize>()?);
    let t0 = Instant::now();
    let graph = hf_graph::RealGraph::from_triples(std::path::Path::new(edges), "wikidata5m", None)?;
    let embeddings = hf_embed::EmbeddingMatrix::load(std::path::Path::new(cache))?;
    println!(
        "loaded graph and cache in {:.1}s",
        t0.elapsed().as_secs_f64()
    );
    let (want, artifacts) = hf_io::read_split(std::path::Path::new(split_dir))?;
    let s = &artifacts.public["sampler"];
    let config = hf_episodes::SamplerConfig {
        family: s["family"].as_str().unwrap().to_string(),
        subgraph_size: s["subgraph_size"].as_u64().unwrap() as u32,
        target_distance: s["target_distance"].as_u64().unwrap() as u32,
        removal_level: s["removal_level"].as_u64().unwrap() as u32,
        cost_epsilon: s["cost_epsilon"].as_f64().unwrap(),
        max_paths: s["max_paths"].as_u64().unwrap() as u32,
        seed_label: s["seed_label"].as_str().unwrap().to_string(),
        hub_degree_cap: s["hub_degree_cap"].as_u64().map(|c| c as u32),
        screen_region: s["screen_region"].as_f64().unwrap_or(0.0),
        targets: s["targets"].as_u64().unwrap_or(1) as u32,
        removal_rule: s["removal_rule"]
            .as_str()
            .unwrap_or("cheapest-first")
            .to_string(),
        greedy_share: s["greedy_share"].as_f64().unwrap_or(1.0),
    };
    println!("config {config:?}");
    let split = artifacts.public["split"].as_str().unwrap().to_string();
    let mut sampler = hf_episodes::Sampler::new(&graph, config)?;
    let t1 = Instant::now();
    sampler.prepare(&split)?;
    println!("start list in {:.1}s", t1.elapsed().as_secs_f64());
    let t2 = Instant::now();
    let sampled = hf_episodes::sample_split(&sampler, &split, count, Some(&embeddings), 256)?;
    let elapsed = t2.elapsed().as_secs_f64();
    let (mut same_id, mut same_visible, mut same_hidden) = (0, 0, 0);
    let mut first_diff = None;
    for (i, (got, want)) in sampled.episodes.iter().zip(&want).enumerate() {
        let mut visible = got.visible.clone();
        for node in visible["nodes"].as_array_mut().unwrap() {
            node["text"] = serde_json::Value::from("");
        }
        let id_ok = got.episode_id == want.episode_id;
        let v_ok = canon(&visible) == canon(&serde_json::to_value(&want.visible)?);
        let h_ok = canon(&got.hidden) == canon(&serde_json::to_value(&want.hidden)?);
        same_id += id_ok as usize;
        same_visible += v_ok as usize;
        same_hidden += h_ok as usize;
        if !(id_ok && v_ok && h_ok) && first_diff.is_none() {
            first_diff = Some((
                i,
                got.episode_id.clone(),
                want.episode_id.clone(),
                id_ok,
                v_ok,
                h_ok,
            ));
        }
    }
    println!(
        "sampled {} kept of {} attempts in {:.1}s ({:.0} attempts/s) | drops {:?} | peak RSS {:.0} MB",
        sampled.episodes.len(), sampled.attempts, elapsed, sampled.attempts as f64 / elapsed, sampled.drops, peak_rss_mb()
    );
    println!("identity vs the Python pool: ids {same_id}/{count}, visible {same_visible}/{count}, hidden {same_hidden}/{count}");
    if let Some(d) = first_diff {
        println!(
            "first difference at ordinal {}: got {} want {} (id {} visible {} hidden {})",
            d.0, d.1, d.2, d.3, d.4, d.5
        );
    }
    Ok(())
}
