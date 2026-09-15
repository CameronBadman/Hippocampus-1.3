//! Against the foundation's `graph_v5` on a synthetic fixture graph
//! (`tools/goldens/gen_graph_goldens.py`): node and edge counts, `typed`,
//! degree percentiles, balls with and without the hub cap and an `allow`
//! filter, neighbour order, bounded simple paths against brute force, and
//! distances — every quantity episode identity rests on.

use std::collections::HashMap;
use std::path::PathBuf;

use hf_graph::RealGraph;
use serde_json::Value;

fn goldens_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/goldens")
}

fn names(g: &RealGraph, ids: &[u32]) -> Vec<String> {
    ids.iter().map(|i| g.name(*i).to_string()).collect()
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().to_string())
        .collect()
}

fn check(name: &str) {
    let golden: Value =
        serde_json::from_str(&std::fs::read_to_string(goldens_dir().join("graph.json")).unwrap())
            .unwrap();
    let g_want = &golden[name];
    let mut g =
        RealGraph::from_triples(&goldens_dir().join(format!("{name}.edges.tsv")), name, None)
            .unwrap();
    assert_eq!(
        g.node_count() as u64,
        g_want["node_count"].as_u64().unwrap()
    );
    assert_eq!(
        g.edge_count() as u64,
        g_want["edge_count"].as_u64().unwrap()
    );
    assert_eq!(
        g.typed(),
        g_want["typed"].as_bool().unwrap(),
        "typed quirk: an empty relation column is typed"
    );
    for (p, want) in g_want["percentiles"].as_object().unwrap() {
        let p: f64 = p.parse().unwrap();
        assert_eq!(
            g.out_degree_percentile(p).unwrap() as u64,
            want.as_u64().unwrap(),
            "percentile {p}"
        );
    }
    for case in g_want["balls"].as_array().unwrap() {
        let start = g.id(case["start"].as_str().unwrap()).unwrap();
        let size = case["size"].as_u64().unwrap() as usize;
        let cap = case["hub_cap"].as_u64().map(|c| c as u32);
        let allow_mod3 = case
            .get("allow_mod3")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let allow = |n: u32| -> bool { g.name(n)[1..].parse::<u32>().unwrap() % 3 != 0 };
        let ball = if allow_mod3 {
            g.ball(start, size, cap, 3, Some(&allow)).unwrap()
        } else {
            g.ball(start, size, cap, 3, None).unwrap()
        };
        assert_eq!(names(&g, &ball), strings(&case["ball"]), "ball {case}");
    }
    for (node, want) in g_want["neighbours"].as_object().unwrap() {
        let id = g.id(node).unwrap();
        assert_eq!(
            names(&g, &g.out_neighbours(id)),
            strings(want),
            "neighbours of {node}"
        );
    }
    for case in g_want["paths"].as_array().unwrap() {
        let start = g.id(case["start"].as_str().unwrap()).unwrap();
        if let Some(from) = case.get("distances_from") {
            let ball = g.ball(start, 40, None, 3, None).unwrap();
            let sub = g.induced(&ball);
            let got: HashMap<String, u64> = sub
                .distances_from(start)
                .into_iter()
                .map(|(n, d)| (g.name(n).to_string(), d as u64))
                .collect();
            let want: HashMap<String, u64> = from
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.as_u64().unwrap()))
                .collect();
            assert_eq!(got, want, "distances_from {}", case["start"]);
            let got: HashMap<String, u64> = sub
                .distances_to(start)
                .into_iter()
                .map(|(n, d)| (g.name(n).to_string(), d as u64))
                .collect();
            let want: HashMap<String, u64> = case["distances_to"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.as_u64().unwrap()))
                .collect();
            assert_eq!(got, want, "distances_to {}", case["start"]);
            continue;
        }
        let ball_ids: Vec<u32> = strings(&case["ball"])
            .iter()
            .map(|n| g.id(n).unwrap())
            .collect();
        let sub = g.induced(&ball_ids);
        let target = g.id(case["target"].as_str().unwrap()).unwrap();
        let max_cost = case["max_cost"].as_u64().unwrap() as u32;
        let paths: Vec<Vec<String>> = sub
            .simple_paths(start, target, max_cost)
            .iter()
            .map(|p| names(&g, p))
            .collect();
        let want: Vec<Vec<String>> = case["paths"]
            .as_array()
            .unwrap()
            .iter()
            .map(strings)
            .collect();
        assert_eq!(
            paths, want,
            "paths {} -> {} <= {}",
            case["start"], case["target"], max_cost
        );
        let brute: Vec<Vec<String>> = sub
            .brute_force_paths(start, target, max_cost)
            .iter()
            .map(|p| names(&g, p))
            .collect();
        let want: Vec<Vec<String>> = case["brute"]
            .as_array()
            .unwrap()
            .iter()
            .map(strings)
            .collect();
        assert_eq!(brute, want, "brute force");
        assert_eq!(paths, brute, "pruned == brute force");
    }
    let attached = g
        .load_text(&goldens_dir().join(format!("{name}.text.tsv")))
        .unwrap();
    assert_eq!(
        attached as u64,
        g_want["text_attached"].as_u64().unwrap(),
        "unknown nodes are not attached"
    );
    for (node, want) in g_want["text_sample"].as_object().unwrap() {
        let got = g.text(g.id(node).unwrap()).map(str::to_string);
        assert_eq!(got, want.as_str().map(str::to_string), "text of {node}");
    }
}

#[test]
fn untyped_fixture_matches_graph_v5() {
    check("fixture");
}

#[test]
fn typed_fixture_matches_graph_v5() {
    check("fixture-typed");
}

/// The real vault graph, when the foundation checkout is beside this one:
/// its rung-3 hub cap (p99 → 24) and a ball, read-only. Skipped otherwise.
#[test]
fn vault_quartz_docs_percentile_when_present() {
    let foundation = std::env::var("HF_FOUNDATION")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../hippocampus-foundation")
        });
    let edges = foundation.join("private/real-walk-v1/graphs/vault-quartz-docs/edges.tsv");
    if !edges.exists() {
        eprintln!("skipped: {} not present", edges.display());
        return;
    }
    let g = RealGraph::from_triples(&edges, "vault-quartz-docs", None).unwrap();
    assert_eq!(
        g.out_degree_percentile(99.0).unwrap(),
        24,
        "the recorded rung-3 hub cap"
    );
    assert!(
        g.typed(),
        "the vault's empty relation column reads typed, as in Python"
    );
}
