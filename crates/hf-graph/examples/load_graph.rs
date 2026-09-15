//! Load a graph structure-only and report counts, wall time and peak RSS —
//! the bench for step 2 (`cargo run --release -p hf-graph --example load_graph -- <edges.tsv> [percentile]`).
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

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = std::path::Path::new(args.get(1).expect("edges.tsv path"));
    let percentile: f64 = args.get(2).map(|p| p.parse()).transpose()?.unwrap_or(99.0);
    let t0 = Instant::now();
    let g = hf_graph::RealGraph::from_triples(path, "bench", None)?;
    let loaded = t0.elapsed();
    let cap = g.out_degree_percentile(percentile)?;
    let t1 = Instant::now();
    let start = g
        .nodes()
        .find(|n| g.has_out(*n))
        .expect("a node with out-edges");
    let mut balls = 0usize;
    let mut nodes = 0usize;
    for n in g.nodes().filter(|n| g.has_out(*n)).take(2000) {
        let ball = g.ball(n, 40, Some(cap), 3, None)?;
        balls += 1;
        nodes += ball.len();
    }
    let _ = start;
    println!(
        "nodes {} edges {} typed {} | load {:.1}s | p{percentile} out-degree {cap} | {balls} balls of 40 in {:.2}s (mean size {:.1}) | peak RSS {:.0} MB",
        g.node_count(),
        g.edge_count(),
        g.typed(),
        loaded.as_secs_f64(),
        t1.elapsed().as_secs_f64(),
        nodes as f64 / balls.max(1) as f64,
        peak_rss_mb()
    );
    Ok(())
}
