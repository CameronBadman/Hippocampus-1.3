//! `evaluate()` and the report it returns, the per-episode rows and the
//! candidate dump, shaped exactly as `scripts/real_walk_stage0.py` writes
//! them so `scripts/real_walk_stage0_report.py` reads them unchanged.

use std::io::Write;
use std::path::Path;

use hf_core::HfError;
use hf_model::{Model, ModelScorer};
use hf_policies::WalkTrace;
use hf_walk::{
    walk_batch, EpisodeIndex, QuerySource, QueryVectors, StopRule, WalkOptions, WalkResult,
};
use serde_json::{json, Map, Value};

pub const Z_ONE_SIDED_95: f64 = 1.644_853_626_951_472_2;
pub const ALLOWED_STOP_REASONS: [&str; 3] = ["target_registered", "exhausted", "learned_stop"];
/// Episodes walked per scoring batch during evaluation.
pub const EVAL_BATCH: usize = 64;

pub fn wilson_lower_bound(correct: usize, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let z = Z_ONE_SIDED_95;
    let n = total as f64;
    let p = correct as f64 / n;
    let denominator = 1.0 + z * z / n;
    let centre = p + z * z / (2.0 * n);
    let radius = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt();
    (centre - radius) / denominator
}

pub fn wilson_upper_bound(successes: usize, total: usize) -> Result<f64, HfError> {
    if total == 0 || successes > total {
        return Err(HfError::BandH(
            "invalid binomial counts for Wilson bound".into(),
        ));
    }
    let z = Z_ONE_SIDED_95;
    let n = total as f64;
    let p = successes as f64 / n;
    let denominator = 1.0 + z * z / n;
    let centre = p + z * z / (2.0 * n);
    let radius = z * ((p * (1.0 - p) + z * z / (4.0 * n)) / n).sqrt();
    Ok((centre + radius) / denominator)
}

/// `round(x, 6)` as the rows carry margins; `round(x, 5)` for residuals.
pub fn round_to(x: f32, places: i32) -> f64 {
    let scale = 10f64.powi(places);
    (x as f64 * scale).round() / scale
}

#[derive(Clone, Debug)]
struct RuleRow {
    registered: bool,
    expansions: usize,
    expansions_at_registration: Option<usize>,
    stop_reason: &'static str,
    removed_count: u32,
}

/// One evaluation's outputs: the aggregate report, the per-episode rows and
/// the candidate records (when asked for).
pub struct Evaluation {
    pub report: Value,
    pub rows: Vec<Value>,
    pub candidates: Vec<(String, Vec<hf_walk::CandidateRecord>)>,
}

fn summarise(items: &[RuleRow]) -> Result<Value, HfError> {
    let n = items.len();
    let registered = items.iter().filter(|r| r.registered).count();
    let mut exp: Vec<usize> = items.iter().map(|r| r.expansions).collect();
    exp.sort_unstable();
    let stop_reasons: Map<String, Value> = ALLOWED_STOP_REASONS
        .iter()
        .map(|k| {
            (
                (*k).to_string(),
                Value::from(items.iter().filter(|r| r.stop_reason == *k).count()),
            )
        })
        .collect();
    Ok(json!({
        "n": n,
        "registered": if n > 0 { Value::from(registered as f64 / n as f64) } else { Value::Null },
        "registered_wilson_lower_bound": if n > 0 { Value::from(wilson_lower_bound(registered, n)) } else { Value::Null },
        "registered_wilson_upper_bound": if n > 0 { Value::from(wilson_upper_bound(registered, n)?) } else { Value::Null },
        "expansions_mean": if n > 0 { Value::from(exp.iter().sum::<usize>() as f64 / n as f64) } else { Value::Null },
        "expansions_median": if n > 0 { Value::from(exp[n / 2]) } else { Value::Null },
        "stop_reasons": stop_reasons,
    }))
}

fn paired(walked: &[RuleRow], items: &[RuleRow]) -> Value {
    let wins = walked
        .iter()
        .zip(items)
        .filter(|(a, b)| a.registered && a.expansions < b.expansions)
        .count();
    let losses = walked
        .iter()
        .zip(items)
        .filter(|(a, b)| b.registered && (!a.registered || a.expansions > b.expansions))
        .count();
    let discordant = wins + losses;
    json!({
        "wins": wins,
        "losses": losses,
        "ties": walked.len() - wins - losses,
        "win_share_among_discordant": if discordant > 0 { Value::from(wins as f64 / discordant as f64) } else { Value::Null },
        "win_share_wilson_lower_bound": if discordant > 0 { Value::from(wilson_lower_bound(wins, discordant)) } else { Value::Null },
    })
}

fn rule_row(w: &WalkResult, removed_count: u32) -> Result<RuleRow, HfError> {
    if !ALLOWED_STOP_REASONS.contains(&w.stop_reason) {
        return Err(HfError::BandH(format!(
            "unknown stop reason {}",
            w.stop_reason
        )));
    }
    Ok(RuleRow {
        registered: w.registered(),
        expansions: w.expansions(),
        expansions_at_registration: w.registered_at,
        stop_reason: w.stop_reason,
        removed_count,
    })
}

/// The row's key for greedy's overshoot, and its value.
///
/// At stage 0 it is the sampler's hidden `greedy_overshoot`: similarity-greedy
/// following the TARGET's own embedding, minus the single-target oracle
/// (`hf-episodes/src/lib.rs`). Under `episode_query` that label was measured on
/// a query this walk never sees, so carrying it would read the stage-0 strata
/// under another name (Track Q binding revision 1). The row carries
/// `question_greedy_overshoot` instead — the same subtraction with the
/// question in the target's place, from this run's own traces — and NEVER the
/// stage-0 key.
fn overshoot_entry(
    query_source: QuerySource,
    hidden: Option<i64>,
    greedy: &WalkTrace,
    oracle: &WalkTrace,
) -> (&'static str, Value) {
    match query_source {
        QuerySource::TargetEmbedding => (
            "greedy_overshoot",
            hidden.map(Value::from).unwrap_or(Value::Null),
        ),
        QuerySource::EpisodeQuery => (
            "question_greedy_overshoot",
            Value::from(greedy.expansions as i64 - oracle.expansions as i64),
        ),
    }
}

/// Walk every episode under both stop rules and every baseline; assemble the report.
pub fn evaluate(
    model: &Model,
    episodes: &[hf_io::RealEpisode],
    embeddings: &hf_embed::EmbeddingMatrix,
    dim: usize,
    record_candidates: bool,
    queries: QueryVectors<'_>,
) -> Result<Evaluation, HfError> {
    let features = model.features();
    let with_prior = model.config.greedy_prior;
    // a split is drawn at one k; the baselines are v1's at k = 1 and
    // `K_TARGETS_DESIGN.md` §3's at k >= 2, and the per-episode row names them
    let k_split = episodes
        .first()
        .is_some_and(|e| e.hidden.target_set.len() > 1);
    let policy_names: &[&str] = if k_split {
        &hf_policies::K_POLICY_NAMES
    } else {
        &hf_policies::POLICY_NAMES
    };
    // the similarity baseline, whose recall at B_fix defines §4's strata:
    // v1's `similarity_greedy`, k-greedy at k >= 2, which equals it at k = 1
    let greedy_at = policy_names
        .iter()
        .position(|n| *n == "similarity_greedy" || *n == "k_greedy")
        .expect("a similarity baseline in every list");
    // the floor `question_greedy_overshoot` is measured against, found by name
    let oracle_at = policy_names
        .iter()
        .position(|n| *n == "oracle" || *n == "k_oracle")
        .expect("an oracle in every list");
    if queries.source == QuerySource::EpisodeQuery && k_split {
        return Err(HfError::BandH(
            "query_source episode_query is k = 1 only; this split shows more".into(),
        ));
    }
    let mut learned_rows: Vec<RuleRow> = Vec::with_capacity(episodes.len());
    let mut exhaust_rows: Vec<RuleRow> = Vec::with_capacity(episodes.len());
    let mut baseline_rows: Vec<Vec<RuleRow>> = vec![Vec::new(); policy_names.len()];
    let mut rows: Vec<Value> = Vec::with_capacity(episodes.len());
    let mut candidates = Vec::new();
    for chunk in episodes.chunks(EVAL_BATCH) {
        let indexes: Vec<EpisodeIndex> = chunk
            .iter()
            .map(|e| EpisodeIndex::new_with_query(e, embeddings, dim, queries))
            .collect::<Result<_, _>>()?;
        let refs: Vec<&EpisodeIndex> = indexes.iter().collect();
        let mut scorer = ModelScorer { model };
        let learned = walk_batch(
            &refs,
            features.as_ref(),
            &mut scorer,
            WalkOptions {
                stop_rule: StopRule::Learned,
                record_candidates: false,
                keep_items: false,
                with_prior,
            },
        )?;
        let exhaust = walk_batch(
            &refs,
            features.as_ref(),
            &mut scorer,
            WalkOptions {
                stop_rule: StopRule::Exhaust,
                record_candidates,
                keep_items: false,
                with_prior,
            },
        )?;
        for (i, e) in chunk.iter().enumerate() {
            let g = if k_split {
                hf_policies::EpisodeGraph::from_episode_k(e)
            } else {
                hf_policies::EpisodeGraph::from_episode(e)
            };
            // under `episode_query` similarity-greedy walks on the QUESTION —
            // the opponent a stage-1 reading is against; the index has already
            // refused an episode the query cache does not cover
            let query64: Option<Vec<f64>> = match queries.source {
                QuerySource::TargetEmbedding => None,
                QuerySource::EpisodeQuery => Some(
                    indexes[i]
                        .query()
                        .iter()
                        .map(|x| *x as f64)
                        .collect::<Vec<f64>>(),
                ),
            };
            let traces = if k_split {
                hf_policies::all_k_traces(&g, embeddings)
            } else {
                hf_policies::all_traces_with_query(&g, embeddings, query64.as_deref())
            };
            let index = &indexes[i];
            let lw = &learned[i];
            let xw = &exhaust[i];
            learned_rows.push(rule_row(lw, e.hidden.removed_count)?);
            exhaust_rows.push(rule_row(xw, e.hidden.removed_count)?);
            for (k, (_, t)) in traces.iter().enumerate() {
                baseline_rows[k].push(RuleRow {
                    registered: t.registered(),
                    expansions: t.expansions as usize,
                    expansions_at_registration: t.registered_at.map(|r| r as usize),
                    stop_reason: if t.stop_reason == "target_registered" {
                        "target_registered"
                    } else {
                        "exhausted"
                    },
                    removed_count: e.hidden.removed_count,
                });
            }
            let mut row = Map::new();
            row.insert("episode_id".into(), e.episode_id.clone().into());
            row.insert("walk_exhaust".into(), xw.expansions().into());
            row.insert("walk_registered".into(), xw.registered().into());
            row.insert("walk_learned".into(), lw.expansions().into());
            for (name, t) in &traces {
                row.insert((*name).to_string(), t.expansions.into());
            }
            let (overshoot_key, overshoot) = overshoot_entry(
                queries.source,
                e.hidden.greedy_overshoot,
                &traces[greedy_at].1,
                &traces[oracle_at].1,
            );
            row.insert(overshoot_key.into(), overshoot);
            row.insert("removed_count".into(), e.hidden.removed_count.into());
            // K_TARGETS_DESIGN.md §4: recall over k at the fixed budget
            // B_fix = n / 2, read at the walk's own stop under `exhaust`, and
            // the per-target registration the strata and the rank reading need.
            // The STRATA are k-greedy's recall, formed at read time from
            // `similarity_greedy_recall_at_budget`, which §4 names; the key
            // keeps v1's name so a committed reader finds it.
            // n is the RUNG's ball size (the sampler block's `subgraph_size`:
            // 20 at rung 3, 40 at rung 4), not the realised ball, so B_fix is
            // one number per rung as §4 fixes it
            let b_fix = hf_policies::rung_ball_size(e) / 2;
            row.insert("budget_at_recall".into(), b_fix.into());
            row.insert(
                "recall_at_budget".into(),
                xw.recall_at_budget(b_fix as usize)
                    .map(Value::from)
                    .unwrap_or(Value::Null),
            );
            row.insert("registered_targets".into(), xw.registered_targets().into());
            row.insert(
                "registered_at".into(),
                xw.registered_at_by_target
                    .iter()
                    .map(|r| r.map(Value::from).unwrap_or(Value::Null))
                    .collect::<Vec<_>>()
                    .into(),
            );
            row.insert(
                "similarity_greedy_recall_at_budget".into(),
                traces[greedy_at]
                    .1
                    .recall_at_budget(b_fix)
                    .map(Value::from)
                    .unwrap_or(Value::Null),
            );
            row.insert(
                "walk_expanded".into(),
                xw.expanded
                    .iter()
                    .map(|n| Value::from(index.names[*n as usize].clone()))
                    .collect::<Vec<_>>()
                    .into(),
            );
            row.insert(
                "similarity_greedy_examined".into(),
                traces[greedy_at]
                    .1
                    .examined
                    .iter()
                    .map(|n| Value::from(n.clone()))
                    .collect::<Vec<_>>()
                    .into(),
            );
            row.insert(
                "walk_margins".into(),
                xw.margins
                    .iter()
                    .map(|m| {
                        m.map(|x| Value::from(round_to(x, 6)))
                            .unwrap_or(Value::Null)
                    })
                    .collect::<Vec<_>>()
                    .into(),
            );
            row.insert(
                "walk_cosine_margins".into(),
                xw.cosine_margins
                    .iter()
                    .map(|m| {
                        m.map(|x| Value::from(round_to(x, 6)))
                            .unwrap_or(Value::Null)
                    })
                    .collect::<Vec<_>>()
                    .into(),
            );
            if let Some(res) = &xw.residuals {
                row.insert(
                    "walk_residuals".into(),
                    res.iter()
                        .map(|r| {
                            Value::from(
                                r.iter()
                                    .map(|x| Value::from(round_to(*x, 5)))
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect::<Vec<_>>()
                        .into(),
                );
            }
            rows.push(Value::Object(row));
            if let Some(c) = &xw.candidates {
                candidates.push((e.episode_id.clone(), c.clone()));
            }
        }
    }
    let mut report = Map::new();
    report.insert("learned".into(), summarise(&learned_rows)?);
    report.insert("exhaust".into(), summarise(&exhaust_rows)?);
    let mut baselines = Map::new();
    for (k, name) in policy_names.iter().enumerate() {
        baselines.insert((*name).to_string(), summarise(&baseline_rows[k])?);
    }
    report.insert("baselines".into(), Value::Object(baselines));
    for (rule, walked) in [("learned", &learned_rows), ("exhaust", &exhaust_rows)] {
        for (k, name) in policy_names.iter().enumerate() {
            report.insert(
                format!("{rule}_vs_{name}"),
                paired(walked, &baseline_rows[k]),
            );
        }
    }
    report.insert(
        "realised_removal_mean".into(),
        if learned_rows.is_empty() {
            Value::Null
        } else {
            Value::from(
                learned_rows
                    .iter()
                    .map(|r| r.removed_count as f64)
                    .sum::<f64>()
                    / learned_rows.len() as f64,
            )
        },
    );
    let _ = learned_rows.first().map(|r| r.expansions_at_registration);
    Ok(Evaluation {
        report: Value::Object(report),
        rows,
        candidates,
    })
}

/// Append rows to `evaluation_rows.jsonl` with `split` and `update` set (an
/// integer for a training evaluation, the literal `"reeval"` otherwise).
pub fn write_evaluation_rows(
    output: &Path,
    split: &str,
    update: &Value,
    rows: &[Value],
) -> Result<(), HfError> {
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(output.join("evaluation_rows.jsonl"))
        .map_err(|e| HfError::Invalid(e.to_string()))?;
    for row in rows {
        let mut full = Map::new();
        full.insert("split".into(), split.into());
        full.insert("update".into(), update.clone());
        for (k, v) in row.as_object().unwrap() {
            full.insert(k.clone(), v.clone());
        }
        let line = serde_json::to_string(&Value::Object(full))
            .map_err(|e| HfError::Invalid(e.to_string()))?;
        file.write_all(line.as_bytes())
            .and_then(|_| file.write_all(b"\n"))
            .map_err(|e| HfError::Invalid(e.to_string()))?;
    }
    file.flush().map_err(|e| HfError::Invalid(e.to_string()))
}

/// Append candidate records to `candidate_dump.jsonl.gz` (one line per exhaust decision).
pub fn write_candidate_dump(
    output: &Path,
    split: &str,
    update: &Value,
    candidates: &[(String, Vec<hf_walk::CandidateRecord>)],
) -> Result<(), HfError> {
    let file = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(output.join("candidate_dump.jsonl.gz"))
        .map_err(|e| HfError::Invalid(e.to_string()))?;
    let mut gz = flate2::write::GzEncoder::new(file, flate2::Compression::default());
    for (episode_id, records) in candidates {
        for (k, rec) in records.iter().enumerate() {
            let line = json!({
                "split": split,
                "update": update,
                "episode_id": episode_id,
                "decision": k,
                "frontier": rec.frontier,
                "scores": rec.scores.iter().map(|x| round_to(*x, 6)).collect::<Vec<_>>(),
                "cosines": rec.cosines.iter().map(|x| round_to(*x, 6)).collect::<Vec<_>>(),
                "chosen": rec.chosen,
                "parents": rec.parents,
                "depths": rec.depths,
            });
            gz.write_all(line.to_string().as_bytes())
                .and_then(|_| gz.write_all(b"\n"))
                .map_err(|e| HfError::Invalid(e.to_string()))?;
        }
    }
    gz.finish().map_err(|e| HfError::Invalid(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The golden fixture episodes, rewritten as the stage-1 records a teacher
    /// would write (no `target_node`, a `query`), with the node cache and a
    /// question-vector sidecar keyed by episode id.
    fn stage1_world() -> (
        Vec<hf_io::RealEpisode>,
        hf_embed::EmbeddingMatrix,
        hf_embed::EmbeddingMatrix,
    ) {
        let goldens = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../hf-io/tests/goldens/fixture-split/screen");
        let (_, embeddings) = hf_episodes::fixture::fixture_world(5, 400, 1600, 8);
        let mut names: Vec<String> = embeddings.keys().cloned().collect();
        names.sort();
        let data: Vec<f32> = names
            .iter()
            .flat_map(|n| embeddings[n].iter().map(|x| *x as f32))
            .collect();
        let nodes = hf_embed::EmbeddingMatrix::from_rows(names, 8, data);
        let (stage0, _) = hf_io::read_split(&goldens).expect("the golden screen split");
        let stage0: Vec<hf_io::RealEpisode> = stage0.into_iter().take(6).collect();
        let mut ids = Vec::new();
        let mut rows: Vec<f32> = Vec::new();
        let episodes: Vec<hf_io::RealEpisode> = stage0
            .into_iter()
            .enumerate()
            .map(|(i, mut e)| {
                e.visible.stage = hf_io::STAGES[1].into();
                e.visible.target_node = None;
                e.visible.query = Some(Value::from("which node answers this?"));
                // a teacher that FAILED to drop the stage-0 label: the row
                // must still never carry it
                e.hidden.greedy_overshoot = Some(900 + i as i64);
                ids.push(e.episode_id.clone());
                // a question vector that is nobody's node vector
                rows.extend(
                    (0..8).map(|j| (((i + 1) as f32) * 0.37 + (j as f32) * 0.11).sin() * 0.5),
                );
                e
            })
            .collect();
        let queries = hf_embed::EmbeddingMatrix::from_rows(ids, 8, rows);
        (episodes, nodes, queries)
    }

    fn cpu_model() -> hf_model::Model {
        let config = hf_model::ModelConfig::from_value(
            &json!({
                "hidden_dimension": 16, "self_attention_heads": 2,
                "feedforward_multiplier": 2, "score_hidden_dimension": 8,
                "coverage_hidden_dimension": 8, "traversal_blocks": 1,
                "dropout": 0.0, "feature_set": "relational-v6"
            }),
            8,
        )
        .expect("a model config");
        hf_model::Model::new(config, tch::Device::Cpu).expect("a CPU model")
    }

    /// Track Q binding revision 1, on the rows themselves: under
    /// `episode_query` every evaluation row carries `question_greedy_overshoot`
    /// and NOT the stage-0 `greedy_overshoot`, whose value the hidden payload
    /// still holds — so a reader cannot take the stage-0 strata for the
    /// question strata. The value is greedy-on-the-question minus the oracle,
    /// recomputed here from the policies.
    #[test]
    fn a_stage_1_row_carries_the_question_overshoot_and_never_the_stage_0_key() {
        let (episodes, nodes, queries) = stage1_world();
        let model = cpu_model();
        let stage1 = evaluate(
            &model,
            &episodes,
            &nodes,
            8,
            false,
            QueryVectors::episode_query(&queries),
        )
        .expect("a stage-1 evaluation");
        assert_eq!(stage1.rows.len(), episodes.len());
        let mut moved = 0;
        for (row, e) in stage1.rows.iter().zip(&episodes) {
            let object = row.as_object().expect("a row");
            assert!(
                object.get("greedy_overshoot").is_none(),
                "the stage-0 label reached a stage-1 row for {}",
                e.episode_id
            );
            let g = hf_policies::EpisodeGraph::from_episode(e);
            let q: Vec<f64> = queries
                .get(&e.episode_id)
                .expect("a question")
                .iter()
                .map(|x| *x as f64)
                .collect();
            let greedy = hf_policies::similarity_greedy_trace(&g, &nodes, Some(&q));
            let oracle = hf_policies::oracle_trace(&g);
            assert_eq!(
                object["question_greedy_overshoot"]
                    .as_i64()
                    .expect("an int"),
                greedy.expansions as i64 - oracle.expansions as i64,
                "{}",
                e.episode_id
            );
            // the baseline row is the walk on the QUESTION, not on the target
            assert_eq!(
                object["similarity_greedy"].as_u64().expect("expansions"),
                greedy.expansions as u64
            );
            if hf_policies::similarity_greedy_trace(&g, &nodes, None).expansions
                != greedy.expansions
            {
                moved += 1;
            }
            // the hidden payload still holds the stage-0 label; the row simply
            // does not carry it
            assert!(e.hidden.greedy_overshoot.is_some());
        }
        assert!(
            moved > 0,
            "the question changed no greedy walk; the check would be vacuous"
        );
        // and at stage 0 the same rows carry the old key and not the new one
        let stage0: Vec<hf_io::RealEpisode> = {
            let goldens = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../hf-io/tests/goldens/fixture-split/screen");
            hf_io::read_split(&goldens)
                .expect("the golden split")
                .0
                .into_iter()
                .take(6)
                .enumerate()
                .map(|(i, mut e)| {
                    e.hidden.greedy_overshoot = Some(7 + i as i64);
                    e
                })
                .collect()
        };
        let plain = evaluate(
            &model,
            &stage0,
            &nodes,
            8,
            false,
            QueryVectors::target_embedding(),
        )
        .expect("a stage-0 evaluation");
        for (row, e) in plain.rows.iter().zip(&stage0) {
            let object = row.as_object().expect("a row");
            assert!(object.get("question_greedy_overshoot").is_none());
            assert_eq!(
                object["greedy_overshoot"],
                e.hidden
                    .greedy_overshoot
                    .map(Value::from)
                    .unwrap_or(Value::Null)
            );
        }
    }

    /// The key alone, both ways, at the one place the row is written.
    #[test]
    fn the_overshoot_key_is_one_or_the_other_and_never_both() {
        let greedy = hf_policies::WalkTrace {
            expansions: 9,
            ..Default::default()
        };
        let oracle = hf_policies::WalkTrace {
            expansions: 4,
            ..Default::default()
        };
        assert_eq!(
            overshoot_entry(QuerySource::TargetEmbedding, Some(3), &greedy, &oracle),
            ("greedy_overshoot", Value::from(3))
        );
        assert_eq!(
            overshoot_entry(QuerySource::TargetEmbedding, None, &greedy, &oracle),
            ("greedy_overshoot", Value::Null)
        );
        assert_eq!(
            overshoot_entry(QuerySource::EpisodeQuery, Some(3), &greedy, &oracle),
            ("question_greedy_overshoot", Value::from(5)),
            "the hidden label is not read under episode_query"
        );
    }

    /// A dump line keeps every key the readers already consume and gains
    /// `parents` (the discovery parent of each frontier candidate, by name) and
    /// `depths`; `chosen` still indexes `frontier`.
    #[test]
    fn a_dump_line_carries_the_parents_and_depths_beside_the_old_keys() {
        let dir = std::env::temp_dir().join(format!("hf-stage0-dump-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let records = vec![(
            "ep-1".to_string(),
            vec![hf_walk::CandidateRecord {
                frontier: vec!["Q2".into(), "Q3".into(), "Q4".into()],
                scores: vec![0.25, 0.75, 0.5],
                cosines: vec![0.1, 0.2, 0.3],
                chosen: 1,
                parents: vec!["Q1".into(), "Q1".into(), "Q2".into()],
                depths: vec![1, 1, 2],
            }],
        )];
        write_candidate_dump(&dir, "screen", &Value::from("reeval"), &records).unwrap();
        let raw = std::fs::read(dir.join("candidate_dump.jsonl.gz")).unwrap();
        let mut text = String::new();
        std::io::Read::read_to_string(&mut flate2::read::MultiGzDecoder::new(&raw[..]), &mut text)
            .unwrap();
        let line: Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        let keys: Vec<&str> = line
            .as_object()
            .unwrap()
            .keys()
            .map(|k| k.as_str())
            .collect();
        assert_eq!(
            keys,
            [
                "split",
                "update",
                "episode_id",
                "decision",
                "frontier",
                "scores",
                "cosines",
                "chosen",
                "parents",
                "depths",
            ],
            "the existing keys keep their names and their order"
        );
        assert_eq!(line["split"], "screen");
        assert_eq!(line["update"], "reeval");
        assert_eq!(line["episode_id"], "ep-1");
        assert_eq!(line["decision"], 0);
        assert_eq!(line["parents"], serde_json::json!(["Q1", "Q1", "Q2"]));
        assert_eq!(line["depths"], serde_json::json!([1, 1, 2]));
        let frontier = line["frontier"].as_array().unwrap();
        let scores: Vec<f64> = line["scores"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let chosen = line["chosen"].as_u64().unwrap() as usize;
        assert_eq!(frontier.len(), scores.len());
        assert_eq!(line["parents"].as_array().unwrap().len(), frontier.len());
        assert_eq!(line["depths"].as_array().unwrap().len(), frontier.len());
        assert_eq!(frontier[chosen], "Q3");
        assert_eq!(
            scores[chosen],
            scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
