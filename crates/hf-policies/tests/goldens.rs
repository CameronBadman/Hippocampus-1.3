//! Every baseline's trace on the fixture splits equals `policies_v5`'s
//! (`tools/goldens/gen_policies_goldens.py`): expansion order, count,
//! registration index, stop reason, parents, route, and the report row.

use std::collections::HashMap;
use std::path::PathBuf;

use hf_policies::{all_traces, policy_row, EpisodeGraph};
use serde_json::Value;

#[test]
fn traces_match_policies_v5_on_every_fixture_episode() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let golden: Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("tests/goldens/policies.json")).unwrap(),
    )
    .unwrap();
    let embeddings: HashMap<String, Vec<f64>> = golden["embeddings"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap())
                    .collect(),
            )
        })
        .collect();
    let splits = root.join("../hf-io/tests/goldens/fixture-split");
    let mut checked = 0;
    for (name, records) in golden["splits"].as_object().unwrap() {
        let (episodes, _) = hf_io::read_split(&splits.join(name)).unwrap();
        for (episode, record) in episodes.iter().zip(records.as_array().unwrap()) {
            assert_eq!(episode.episode_id, record["episode_id"]);
            let g = EpisodeGraph::from_episode(episode);
            for (policy, trace) in all_traces(&g, &embeddings) {
                let want = &record["traces"][policy];
                let examined: Vec<&str> = want["examined"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap())
                    .collect();
                assert_eq!(
                    trace.examined, examined,
                    "{name} {} {policy} order",
                    episode.episode_id
                );
                assert_eq!(
                    trace.expansions as u64,
                    want["expansions"].as_u64().unwrap(),
                    "{policy} expansions"
                );
                assert_eq!(
                    trace.registered_at.map(u64::from),
                    want["registered_at"].as_u64(),
                    "{policy} registered_at"
                );
                assert_eq!(
                    trace.stop_reason,
                    want["stop_reason"].as_str().unwrap(),
                    "{policy} stop"
                );
                let parents: HashMap<String, String> = want["parents"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
                    .collect();
                assert_eq!(trace.parents, parents, "{policy} parents");
                if let Some(route) = want["route"].as_array() {
                    let route: Vec<&str> = route.iter().map(|v| v.as_str().unwrap()).collect();
                    assert_eq!(trace.route(&g.start, &g.target), route, "{policy} route");
                } else {
                    assert_eq!(
                        policy, "bidirectional_bfs",
                        "only bidirectional has no route"
                    );
                }
                let row = serde_json::to_value(policy_row(&g, &trace)).unwrap();
                assert_eq!(row, want["row"], "{policy} row");
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 30 * 4, "12 + 6 + 12 episodes x four policies");
}
