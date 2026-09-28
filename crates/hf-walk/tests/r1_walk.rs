//! The R1 head's walk (`R1_HEAD_DESIGN.md` §8 ENG-3, and ENG-7 (e)), on a
//! stub scorer: the return item's candidates are `E_t ∖ {s}`, a snapshot is
//! the return item of the walk capped at `t`, the snapshots past a walk's end
//! are taken at its end, nothing hidden reaches an item or a stop row, and
//! `StopRule::LearnedR1` consults its head only at the snapshots, stays under
//! the cap and truncates the capped walk without changing it.

use std::collections::{BTreeSet, HashSet};

use hf_embed::EmbeddingMatrix;
use hf_walk::{
    walk_batch_capped, walk_batch_r1, DecisionBatch, EpisodeIndex, FeatureSet, R1Options, R1Trace,
    RelationalV6Prev, RstopInput, Scored, Scorer, StopRule, WalkOptions, WalkResult,
    R1_STOP_SNAPSHOTS, STOP_DIM,
};
use sha2::{Digest, Sha256};

const EDIM: usize = 8;

fn vector(name: &str) -> Vec<f32> {
    let mut h = Sha256::new();
    h.update(name.as_bytes());
    let d = h.finalize();
    (0..EDIM).map(|i| (d[i] as f32 - 128.0) / 128.0).collect()
}

/// A ring of `n` nodes with chords 1, 3 and 7 ahead, and X's row as the query.
fn ball(n: u32, id: &str) -> EpisodeIndex {
    let names: Vec<String> = (0..n).map(|i| format!("v{i}")).collect();
    let mut edges = Vec::new();
    for i in 0..n {
        for d in [1u32, 3, 7] {
            edges.push((
                edges.len() as u32,
                format!("v{i}"),
                format!("v{}", (i + d) % n),
            ));
        }
    }
    let mut all = names.clone();
    all.push("x".into());
    let data: Vec<f32> = all.iter().flat_map(|v| vector(v)).collect();
    let cache = EmbeddingMatrix::from_rows(all, EDIM, data);
    EpisodeIndex::for_deletion(id, "v0", &names, &edges, &cache, EDIM, &vector("x")).unwrap()
}

/// Scores from a hash of each candidate row; pooled "hidden states" from the
/// rows too; the rstop head fires at `fire_at` and records every `t` it is
/// asked at.
struct Stub {
    cdim: usize,
    fire_at: Option<usize>,
    consulted: Vec<usize>,
}

impl Stub {
    fn new(fire_at: Option<usize>) -> Self {
        Self {
            cdim: RelationalV6Prev.candidate_dim(EDIM),
            fire_at,
            consulted: Vec::new(),
        }
    }
}

impl Scorer for Stub {
    fn score(&mut self, batch: &DecisionBatch) -> Result<Scored, hf_core::HfError> {
        Ok(Scored {
            scores: batch
                .items
                .iter()
                .map(|it| {
                    (0..it.frontier_len)
                        .map(|i| {
                            let row = &it.cand[i * self.cdim..(i + 1) * self.cdim];
                            let mut h = Sha256::new();
                            for v in row {
                                h.update(v.to_le_bytes());
                            }
                            let d = h.finalize();
                            (u32::from_le_bytes([d[0], d[1], d[2], d[3]]) as f32) / u32::MAX as f32
                        })
                        .collect()
                })
                .collect(),
            residuals: None,
        })
    }
    fn stop_logits(&mut self, rows: &[[f32; STOP_DIM]]) -> Result<Vec<f32>, hf_core::HfError> {
        Ok(vec![-1.0; rows.len()])
    }
    fn score_pooled(
        &mut self,
        batch: &DecisionBatch,
    ) -> Result<(Scored, Vec<Vec<f32>>), hf_core::HfError> {
        let scored = self.score(batch)?;
        let pooled = batch
            .items
            .iter()
            .map(|it| vec![it.cand.iter().sum::<f32>(), it.frontier_len as f32])
            .collect();
        Ok((scored, pooled))
    }
    fn rstop_logits(&mut self, inputs: &[RstopInput]) -> Result<Vec<f32>, hf_core::HfError> {
        Ok(inputs
            .iter()
            .map(|i| {
                self.consulted.push(i.t);
                if Some(i.t) == self.fire_at {
                    5.0
                } else {
                    -5.0
                }
            })
            .collect())
    }
}

fn options(rule: StopRule) -> WalkOptions {
    WalkOptions {
        stop_rule: rule,
        record_candidates: false,
        keep_items: false,
        with_prior: false,
    }
}

fn r1_walk(
    index: &EpisodeIndex,
    scorer: &mut Stub,
    rule: StopRule,
    cap: usize,
    return_at: &[usize],
) -> (WalkResult, R1Trace) {
    walk_batch_r1(
        &[index],
        &RelationalV6Prev,
        scorer,
        options(rule),
        Some(cap),
        &R1Options {
            return_at: return_at.to_vec(),
            rstop_at: vec![16, 32, 48, 64],
        },
    )
    .unwrap()
    .remove(0)
}

const S: [usize; 5] = [16, 32, 48, 64, 80];

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// ENG-3 (a): the return item's candidates are `E_t ∖ {s}` with
/// `E_t = {s} ∪ ⋃_{i < t} out(expanded[i])` — the premise's `E_B`,
/// recomputed from the expansion order and the out-lists — the expanded
/// nodes first, in expansion order, the start never among them.
#[test]
fn the_return_items_candidates_are_the_examined_set_without_the_start() {
    let index = ball(120, "fixture-screen-000001-00000000-del-U");
    let (w, trace) = r1_walk(&index, &mut Stub::new(None), StopRule::Exhaust, 80, &S);
    assert_eq!(w.expansions(), 80, "the cap binds on this ball");
    assert_eq!(trace.snapshots.len(), S.len());
    for snap in &trace.snapshots {
        assert_eq!(snap.t_eff, snap.t);
        let t = snap.t;
        let mut e: BTreeSet<u32> = BTreeSet::from([index.start]);
        for x in &w.expanded[..t] {
            e.extend(index.out[*x as usize].iter().copied());
        }
        e.remove(&index.start);
        let got: BTreeSet<u32> = snap.candidates.iter().copied().collect();
        assert_eq!(got, e, "t = {t}");
        assert_eq!(got.len(), snap.candidates.len(), "a candidate listed twice");
        assert!(!snap.candidates.contains(&index.start));
        assert_eq!(&snap.candidates[..t - 1], &w.expanded[1..t]);
        assert_eq!(snap.item.frontier_len, snap.candidates.len());
        assert_eq!(snap.item.context_len, t);
        assert_eq!(snap.extras.len(), 4 * snap.candidates.len());
        for (k, _) in snap.candidates.iter().enumerate() {
            let ex = &snap.extras[4 * k..4 * k + 4];
            let expanded = k < t - 1;
            assert_eq!(ex[0], if expanded { 1.0 } else { 0.0 });
            assert_eq!(ex[1], if expanded { (k + 1) as f32 / 80.0 } else { 0.0 });
            assert!((0.0..=1.0).contains(&ex[3]));
        }
    }
}

/// ENG-3 (b): the snapshot at `t` of the capped walk is the return item of
/// the same walk capped at `t` — candidates, candidate rows, context, pair
/// channel and extras, bit for bit (the prefix property).
#[test]
fn a_snapshot_is_the_return_item_of_the_walk_capped_there() {
    let index = ball(120, "fixture-screen-000002-00000000-del-D3");
    let (_, full) = r1_walk(&index, &mut Stub::new(None), StopRule::Exhaust, 80, &S);
    for (j, &t) in S.iter().enumerate() {
        let (w, short) = r1_walk(&index, &mut Stub::new(None), StopRule::Exhaust, t, &[t]);
        assert_eq!(w.expansions(), t);
        let (a, b) = (&full.snapshots[j], &short.snapshots[0]);
        assert_eq!(a.candidates, b.candidates, "t = {t}");
        assert_eq!(bits(&a.item.cand), bits(&b.item.cand));
        assert_eq!(bits(&a.item.ctx), bits(&b.item.ctx));
        assert_eq!(bits(&a.item.pair), bits(&b.item.pair));
        assert_eq!(bits(&a.extras), bits(&b.extras));
    }
}

/// ENG-7 (e) and ENG-3's end rule: on a ball the walk exhausts before the cap
/// (40 nodes, so 40 expansions), the snapshots at 16 and 32 are taken on the
/// way, and every snapshot past the end is taken AT the end (`t_eff` = the
/// end, the same item each time); no stop input exists past the end.
#[test]
fn snapshots_past_the_walks_end_are_taken_at_its_end() {
    let index = ball(40, "fixture-screen-000003-00000000-del-U");
    let (w, trace) = r1_walk(&index, &mut Stub::new(None), StopRule::Exhaust, 80, &S);
    let end = w.expansions();
    assert!(end < 80, "the ball is exhausted first");
    assert_eq!(w.stop_reason, "exhausted");
    let effs: Vec<usize> = trace.snapshots.iter().map(|s| s.t_eff).collect();
    let want: Vec<usize> = S.iter().map(|t| (*t).min(end)).collect();
    assert_eq!(effs, want);
    let at_end: Vec<&hf_walk::ReturnSnapshot> =
        trace.snapshots.iter().filter(|s| s.t > end).collect();
    assert!(at_end.len() >= 2);
    for s in &at_end {
        assert_eq!(s.candidates, at_end[0].candidates);
        assert_eq!(bits(&s.item.cand), bits(&at_end[0].item.cand));
    }
    let stops: Vec<usize> = trace.rstop.iter().map(|r| r.t).collect();
    let want: Vec<usize> = [16, 32, 48, 64].into_iter().filter(|t| *t < end).collect();
    assert_eq!(stops, want, "a stop input only at a decision the walk made");
}

/// ENG-3 (c) at the walk: garbage in every hidden field of the index (the
/// on-path flags, the distances, a hidden target, the removal count) and a
/// different episode id leave every decision, return item, rstop row and
/// stop consultation bit-identical. (The labels file itself is covered by
/// the CLI tests, which delete it.)
#[test]
fn nothing_hidden_or_in_the_id_reaches_an_item_or_a_stop_row() {
    let clean = ball(120, "fixture-screen-000004-00000000-del-U");
    let mut dirty = ball(120, "garbage-id-with-no-tag");
    for f in dirty.on_path.iter_mut() {
        *f = true;
    }
    for d in dirty.distance.iter_mut() {
        *d = Some(1);
    }
    dirty.hidden_targets = vec![5, 9];
    dirty.removed_count = 77;
    dirty.greedy_overshoot = Some(-3);
    let rule = StopRule::LearnedR1 {
        theta: 0.5,
        snapshots: R1_STOP_SNAPSHOTS,
    };
    let (mut s1, mut s2) = (Stub::new(Some(48)), Stub::new(Some(48)));
    let (wa, ta) = r1_walk(&clean, &mut s1, rule, 80, &S);
    let (wb, tb) = r1_walk(&dirty, &mut s2, rule, 80, &S);
    assert_eq!(wa.expanded, wb.expanded);
    assert_eq!(wa.stop_reason, wb.stop_reason);
    assert_eq!(s1.consulted, s2.consulted);
    for (a, b) in ta.snapshots.iter().zip(&tb.snapshots) {
        assert_eq!(a.candidates, b.candidates);
        assert_eq!(bits(&a.item.cand), bits(&b.item.cand));
        assert_eq!(bits(&a.item.ctx), bits(&b.item.ctx));
        assert_eq!(bits(&a.item.pair), bits(&b.item.pair));
        assert_eq!(bits(&a.extras), bits(&b.extras));
    }
    assert_eq!(ta.rstop, tb.rstop);
    assert_eq!(ta.rstop_logits, tb.rstop_logits);
}

/// ENG-3 (e) and (f): the learned stop consults its head ONLY at the
/// snapshots, fires where the head says, never passes the cap, and the
/// expansion order it leaves is the prefix of the capped walk's — the frozen
/// path of §4. With the head never firing it is the capped walk exactly.
#[test]
fn the_learned_stop_reads_only_at_snapshots_and_truncates_the_capped_walk() {
    let index = ball(120, "fixture-screen-000005-00000000-del-U");
    let (capped, _) = r1_walk(&index, &mut Stub::new(None), StopRule::Exhaust, 80, &S);
    let rule = StopRule::LearnedR1 {
        theta: 0.5,
        snapshots: R1_STOP_SNAPSHOTS,
    };
    for fire in [Some(16), Some(32), Some(48), Some(64), None] {
        let mut stub = Stub::new(fire);
        let (w, trace) = r1_walk(&index, &mut stub, rule, 80, &S);
        let all: HashSet<usize> = R1_STOP_SNAPSHOTS.into_iter().collect();
        assert!(
            stub.consulted.iter().all(|t| all.contains(t)),
            "{:?}",
            stub.consulted
        );
        assert!(w.expansions() <= 80);
        assert_eq!(&capped.expanded[..w.expansions()], &w.expanded[..]);
        match fire {
            Some(t) => {
                assert_eq!(w.expansions(), t);
                assert_eq!(w.stop_reason, "learned_r1_stop");
                assert_eq!(*stub.consulted.last().unwrap(), t);
                assert_eq!(trace.rstop_logits.last().unwrap().0, t);
            }
            None => {
                assert_eq!(w.expanded, capped.expanded);
                assert_eq!(w.stop_reason, "expansion_cap");
                assert_eq!(stub.consulted, vec![16, 32, 48, 64]);
            }
        }
    }
    // the plain capped walk never asks for pooled states or an rstop logit
    let mut stub = Stub::new(Some(16));
    let plain = walk_batch_capped(
        &[&index],
        &RelationalV6Prev,
        &mut stub,
        options(StopRule::Exhaust),
        Some(80),
    )
    .unwrap()
    .remove(0);
    assert_eq!(plain.expanded, capped.expanded);
    assert!(stub.consulted.is_empty());
}
