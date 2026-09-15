//! `evaluate()` and the report it returns, the per-episode rows and the
//! candidate dump, shaped exactly as `scripts/real_walk_stage0.py` writes
//! them so `scripts/real_walk_stage0_report.py` reads them unchanged.

use std::io::Write;
use std::path::Path;

use hf_core::HfError;
use hf_model::{Model, ModelScorer};
use hf_walk::{walk_batch, EpisodeIndex, StopRule, WalkOptions, WalkResult};
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

/// Walk every episode under both stop rules and every baseline; assemble the report.
pub fn evaluate(
    model: &Model,
    episodes: &[hf_io::RealEpisode],
    embeddings: &hf_embed::EmbeddingMatrix,
    dim: usize,
    record_candidates: bool,
) -> Result<Evaluation, HfError> {
    let features = model.features();
    let with_prior = model.config.greedy_prior;
    let mut learned_rows: Vec<RuleRow> = Vec::with_capacity(episodes.len());
    let mut exhaust_rows: Vec<RuleRow> = Vec::with_capacity(episodes.len());
    let mut baseline_rows: Vec<Vec<RuleRow>> = vec![Vec::new(); 4];
    let mut rows: Vec<Value> = Vec::with_capacity(episodes.len());
    let mut candidates = Vec::new();
    for chunk in episodes.chunks(EVAL_BATCH) {
        let indexes: Vec<EpisodeIndex> = chunk
            .iter()
            .map(|e| EpisodeIndex::new(e, embeddings, dim))
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
            let g = hf_policies::EpisodeGraph::from_episode(e);
            let traces = hf_policies::all_traces(&g, embeddings);
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
            row.insert(
                "greedy_overshoot".into(),
                e.hidden
                    .greedy_overshoot
                    .map(Value::from)
                    .unwrap_or(Value::Null),
            );
            row.insert("removed_count".into(), e.hidden.removed_count.into());
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
                traces[2]
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
    for (k, name) in hf_policies::POLICY_NAMES.iter().enumerate() {
        baselines.insert((*name).to_string(), summarise(&baseline_rows[k])?);
    }
    report.insert("baselines".into(), Value::Object(baselines));
    for (rule, walked) in [("learned", &learned_rows), ("exhaust", &exhaust_rows)] {
        for (k, name) in hf_policies::POLICY_NAMES.iter().enumerate() {
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
            });
            gz.write_all(line.to_string().as_bytes())
                .and_then(|_| gz.write_all(b"\n"))
                .map_err(|e| HfError::Invalid(e.to_string()))?;
        }
    }
    gz.finish().map_err(|e| HfError::Invalid(e.to_string()))?;
    Ok(())
}
