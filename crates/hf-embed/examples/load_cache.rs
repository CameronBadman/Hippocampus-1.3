//! Bench: load a cache (first run parses vectors.jsonl and builds the sidecar; the
//! second maps it). `cargo run --release -p hf-embed --example load_cache -- <cache dir>`
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
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("cache dir"));
    let sidecar = dir.join("vectors.f32").exists();
    let t0 = std::time::Instant::now();
    let m = hf_embed::EmbeddingMatrix::load(&dir)?;
    println!(
        "{} vectors x {} | sidecar present before: {sidecar} | load {:.2}s | peak RSS {:.0} MB | first node {} cos(self) {:.3}",
        m.len(), m.dimension, t0.elapsed().as_secs_f64(), peak_rss_mb(), m.nodes[0], hf_embed::cosine(m.row(0), m.row(0))
    );
    Ok(())
}
