//! Bench: read a whole split. `cargo run --release -p hf-io --example load -- <split dir>`
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
fn main() -> anyhow::Result<()> {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("split dir"));
    let t0 = std::time::Instant::now();
    let (episodes, artifacts) = hf_io::read_split(&dir)?;
    let nodes: usize = episodes.iter().map(|e| e.visible.nodes.len()).sum();
    let greedy = episodes
        .iter()
        .filter(|e| e.hidden.removal_recipe() == "greedy-path")
        .count();
    println!(
        "{} episodes ({} greedy-path), {} node records | load {:.2}s | peak RSS {:.0} MB | manifest count {}",
        episodes.len(), greedy, nodes, t0.elapsed().as_secs_f64(), peak_rss_mb(), artifacts.public["episode_count"]
    );
    Ok(())
}
