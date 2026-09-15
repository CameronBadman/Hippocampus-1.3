//! The sampler against the foundation's own output: the three fixture splits
//! under `crates/hf-io/tests/goldens/fixture-split` were written by Python's
//! `sample_episode` + `write_real_split_v5` on `fixture_world(5)`; sampling the
//! same configuration here must give the same episode ids, the same visible
//! and hidden payloads (as canonical bytes), the same attempts and drops —
//! including the greedy-path split with its route-based removals,
//! `greedy_overshoot` and `removal_recipe`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use hf_episodes::fixture::fixture_world;
use hf_episodes::{sample_split, Sampler, SamplerConfig};
use serde_json::Value;

fn canon(v: &Value) -> String {
    String::from_utf8(hf_core::canonical_bytes(v).unwrap()).unwrap()
}

fn check(name: &str, split: &str, count: usize, config: SamplerConfig, greedy: bool, chunk: usize) {
    let goldens =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../hf-io/tests/goldens/fixture-split");
    let (graph, embeddings) = fixture_world(5, 400, 1600, 8);
    let mut sampler = Sampler::new(&graph, config).unwrap();
    sampler.prepare(split).unwrap();
    let sampled = sample_split(
        &sampler,
        split,
        count,
        if greedy { Some(&embeddings) } else { None },
        chunk,
    )
    .unwrap();
    let (want, _) = hf_io::read_split(&goldens.join(name)).unwrap();
    assert_eq!(sampled.episodes.len(), want.len(), "{name}: count");
    for (got, want) in sampled.episodes.iter().zip(&want) {
        assert_eq!(got.episode_id, want.episode_id, "{name}: id");
        let mut visible = got.visible.clone();
        for node in visible["nodes"].as_array_mut().unwrap() {
            node["text"] = Value::from(""); // the writer blanks text; it lives in texts.jsonl
        }
        assert_eq!(
            canon(&visible),
            canon(&serde_json::to_value(&want.visible).unwrap()),
            "{name}: visible {}",
            got.episode_id
        );
        assert_eq!(
            canon(&got.hidden),
            canon(&serde_json::to_value(&want.hidden).unwrap()),
            "{name}: hidden {}",
            got.episode_id
        );
    }
    let expected: Value =
        serde_json::from_str(&std::fs::read_to_string(goldens.join("expected.json")).unwrap())
            .unwrap();
    assert_eq!(
        sampled.attempts,
        expected[name]["attempts"].as_u64().unwrap(),
        "{name}: attempts"
    );
    let drops: BTreeMap<String, u64> = sampled
        .drops
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();
    let want_drops: BTreeMap<String, u64> = expected[name]["dropped"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_u64().unwrap()))
        .collect();
    assert_eq!(drops, want_drops, "{name}: drops");
}

fn fixture_config() -> SamplerConfig {
    let mut c = SamplerConfig::new("fixture", 64, 3, 2);
    c.cost_epsilon = 0.5;
    c
}

#[test]
fn cheapest_first_train_split_is_reproduced() {
    check("train", "train", 12, fixture_config(), false, 4);
}

#[test]
fn screen_split_is_reproduced() {
    check("screen", "screen", 6, fixture_config(), false, 64);
}

#[test]
fn greedy_path_split_with_share_is_reproduced() {
    let mut c = fixture_config();
    c.removal_rule = "greedy-path".into();
    c.greedy_share = 0.5;
    check("train-greedy", "train", 12, c, true, 5);
}

#[test]
fn a_larger_draw_reproduces_the_smaller_as_its_prefix() {
    let (graph, _) = fixture_world(5, 400, 1600, 8);
    let mut sampler = Sampler::new(&graph, fixture_config()).unwrap();
    sampler.prepare("screen").unwrap();
    let small = sample_split(&sampler, "screen", 6, None, 3).unwrap();
    let large = sample_split(&sampler, "screen", 20, None, 7).unwrap();
    let small_ids: Vec<&str> = small
        .episodes
        .iter()
        .map(|e| e.episode_id.as_str())
        .collect();
    let large_ids: Vec<&str> = large
        .episodes
        .iter()
        .map(|e| e.episode_id.as_str())
        .collect();
    assert_eq!(&large_ids[..6], &small_ids[..]);
    assert!(large.attempts >= small.attempts);
}
