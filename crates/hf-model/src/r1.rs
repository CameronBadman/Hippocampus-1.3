//! The R1 return/stop head's data path (`experiments/real_walk_v2/
//! R1_HEAD_DESIGN.md` §3–§5, §8 ENG-3, ENG-4 and ENG-7).
//!
//! - [`compute_features`]: the frozen trunk walks a batch of deletion
//!   episodes to the cap under `no_grad` and returns, per record, the return
//!   items' hidden states `h_v` and extras at every snapshot, and the stop
//!   head's 523-wide input at the stop snapshots. This is the ON-THE-FLY
//!   computation. Only tests call it outside [`CacheWriter`]'s build: every
//!   training and evaluation path reads the cache.
//! - [`CacheWriter`] / [`R1Cache`]: the f32 cache (ENG-7). Four files —
//!   `records.jsonl` (ids, names, row offsets), `hidden.f32`, `extras.f32`,
//!   `rstop.f32` (little-endian f32 rows) — and `cache.manifest.json`, which
//!   carries every file's byte count and sha256 and the record count. The
//!   reader memory-maps the three f32 files and refuses a cache whose files
//!   disagree with its manifest.
//! - [`DeletionLabels`]: `T` per episode, read from `labels.jsonl.gz`. Only
//!   the loss holds it; no feature path takes it.
//! - [`r1_losses`] and [`stop_label`]: §5's losses.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};

use hf_core::HfError;
use hf_walk::{
    walk_batch_r1, DecisionItem, EpisodeIndex, R1Options, StopRule, WalkOptions, RETURN_EXTRAS,
    RSTOP_ROW_DIM,
};
use serde_json::{json, Value};
use tch::{Device, Kind, Reduction, Tensor};

use crate::{clip_grad_norm, AdamW, Model, ModelScorer};

pub const CACHE_MANIFEST: &str = "cache.manifest.json";
pub const CACHE_KIND: &str = "r1_walk_cache_manifest_v1";
pub const CACHE_FILES: [&str; 4] = ["records.jsonl", "hidden.f32", "extras.f32", "rstop.f32"];

fn invalid(e: impl std::fmt::Display) -> HfError {
    HfError::Invalid(e.to_string())
}

/// The walk the cache records: the cap (the start counted), the return
/// snapshots (the last is the cap) and the stop snapshots (the others).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1Spec {
    pub cap: usize,
    pub snapshots: Vec<usize>,
}

impl R1Spec {
    /// §3: S = {16, 32, 48, 64, 80}, the cap 80.
    pub fn design() -> Self {
        Self {
            cap: 80,
            snapshots: vec![16, 32, 48, 64, 80],
        }
    }

    pub fn validate(&self) -> Result<(), HfError> {
        let sorted = self.snapshots.windows(2).all(|w| w[0] < w[1]);
        if self.snapshots.is_empty()
            || !sorted
            || self.snapshots[0] == 0
            || self.snapshots.last() != Some(&self.cap)
        {
            return Err(HfError::Invalid(format!(
                "R1 snapshots {:?} must rise strictly, start above 0 and end at the cap {}",
                self.snapshots, self.cap
            )));
        }
        Ok(())
    }

    /// The stop is consulted at every snapshot before the cap.
    pub fn stop_at(&self) -> Vec<usize> {
        self.snapshots[..self.snapshots.len() - 1].to_vec()
    }
}

/// One snapshot's return item, as features.
#[derive(Clone, Debug, PartialEq)]
pub struct SnapshotFeatures {
    pub t: usize,
    pub t_eff: usize,
    pub candidates: Vec<String>,
    /// `n × H` final hidden states of the frozen trunk.
    pub hidden: Vec<f32>,
    /// `n × 4` extras.
    pub extras: Vec<f32>,
}

/// One record's features: everything the heads read, and nothing else.
#[derive(Clone, Debug, PartialEq)]
pub struct R1Record {
    pub episode_id: String,
    pub walk_expanded: Vec<String>,
    pub t_end: usize,
    pub stop_reason: String,
    pub snapshots: Vec<SnapshotFeatures>,
    /// Per stop snapshot `t`, the 523-wide input, or `None` when the walk
    /// ended before its decision at `t`.
    pub rstop: Vec<(usize, Option<Vec<f32>>)>,
}

impl R1Record {
    /// The snapshot asked for at `t`.
    pub fn snapshot(&self, t: usize) -> Option<&SnapshotFeatures> {
        self.snapshots.iter().find(|s| s.t == t)
    }
}

/// One deletion record's visible payload (`r1_deletion_visible_v1`).
pub struct DeletionVisible {
    pub episode_id: String,
    pub start: String,
    pub nodes: Vec<String>,
    pub edges: Vec<(u32, String, String)>,
}

/// Read a deletion split's visible stream (never its labels).
pub fn read_deletion_visible(path: &Path, limit: usize) -> Result<Vec<DeletionVisible>, HfError> {
    let raw = hf_io::read_maybe_gz(path)?;
    let mut out = Vec::new();
    for line in raw.split(|b| *b == b'\n') {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        if limit > 0 && out.len() >= limit {
            break;
        }
        let v: Value = serde_json::from_slice(line).map_err(invalid)?;
        let visible = &v["visible"];
        if visible["record_kind"] != "r1_deletion_visible_v1" {
            return Err(HfError::BandH(format!(
                "{}: not a deletion record ({})",
                path.display(),
                visible["record_kind"]
            )));
        }
        let text = |x: &Value| x.as_str().unwrap_or("").to_string();
        out.push(DeletionVisible {
            episode_id: text(&v["episode_id"]),
            start: text(&visible["start_node"]),
            nodes: visible["nodes"]
                .as_array()
                .into_iter()
                .flatten()
                .map(text)
                .collect(),
            edges: visible["edges"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|e| {
                    (
                        e["edge_id"].as_u64().unwrap_or(0) as u32,
                        text(&e["source"]),
                        text(&e["target"]),
                    )
                })
                .collect(),
        });
    }
    Ok(out)
}

/// §3 and §4's inputs for a batch of deletion episodes, computed on the fly
/// by the model's frozen trunk: one walk to the cap under `no_grad`, the
/// return items' hidden states from ONE trunk forward per snapshot over the
/// batch, in batch order. The batch composition is part of the result on a
/// GPU (kernels can differ with batch shape), which is why the cache records
/// its batching.
pub fn compute_features(
    model: &Model,
    indexes: &[&EpisodeIndex],
    spec: &R1Spec,
) -> Result<Vec<R1Record>, HfError> {
    spec.validate()?;
    let _guard = tch::no_grad_guard();
    let features = model.features();
    let stop_at = spec.stop_at();
    let walked = {
        let mut scorer = ModelScorer { model };
        walk_batch_r1(
            indexes,
            features.as_ref(),
            &mut scorer,
            WalkOptions {
                stop_rule: StopRule::Exhaust,
                record_candidates: false,
                keep_items: false,
                with_prior: model.config.greedy_prior,
            },
            Some(spec.cap),
            &R1Options {
                return_at: spec.snapshots.clone(),
                rstop_at: stop_at.clone(),
            },
        )?
    };
    let hdim = model.hidden_dimension() as usize;
    let mut records: Vec<R1Record> = Vec::with_capacity(indexes.len());
    for (index, (w, _)) in indexes.iter().zip(&walked) {
        if w.registered_at.is_some() {
            return Err(HfError::BandH(format!(
                "{}: an R1 walk registered a target",
                index.episode_id
            )));
        }
        records.push(R1Record {
            episode_id: index.episode_id.clone(),
            walk_expanded: w
                .expanded
                .iter()
                .map(|n| index.names[*n as usize].clone())
                .collect(),
            t_end: w.expanded.len(),
            stop_reason: w.stop_reason.to_string(),
            snapshots: Vec::new(),
            rstop: Vec::new(),
        });
    }
    for j in 0..spec.snapshots.len() {
        let items: Vec<&DecisionItem> =
            walked.iter().map(|(_, tr)| &tr.snapshots[j].item).collect();
        let (h, _) = model.hidden_states(&items);
        let h = h.to_device(Device::Cpu).contiguous();
        let (n_items, f) = (h.size()[0] as usize, h.size()[1] as usize);
        let mut flat = vec![0f32; n_items * f * hdim];
        h.copy_data(&mut flat, n_items * f * hdim);
        for (k, ((index, (_, tr)), record)) in indexes
            .iter()
            .zip(&walked)
            .zip(records.iter_mut())
            .enumerate()
        {
            let snap = &tr.snapshots[j];
            let n = snap.candidates.len();
            let base = k * f * hdim;
            record.snapshots.push(SnapshotFeatures {
                t: snap.t,
                t_eff: snap.t_eff,
                candidates: snap
                    .candidates
                    .iter()
                    .map(|c| index.names[*c as usize].clone())
                    .collect(),
                hidden: flat[base..base + n * hdim].to_vec(),
                extras: snap.extras.clone(),
            });
        }
    }
    for ((_, tr), record) in walked.iter().zip(records.iter_mut()) {
        for &t in &stop_at {
            let input = tr.rstop.iter().find(|r| r.t == t).map(|r| r.flat());
            record.rstop.push((t, input));
        }
    }
    Ok(records)
}

/// `T` per deletion episode, from `labels.jsonl.gz`. Held by the loss only.
pub type DeletionLabels = HashMap<String, HashSet<String>>;

pub fn read_deletion_labels(path: &Path) -> Result<DeletionLabels, HfError> {
    let raw = hf_io::read_maybe_gz(path)?;
    let mut out = DeletionLabels::new();
    for line in raw.split(|b| *b == b'\n') {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let v: Value = serde_json::from_slice(line).map_err(invalid)?;
        let id = v["episode_id"]
            .as_str()
            .ok_or_else(|| HfError::BandH("a label row without an episode_id".into()))?;
        let t: HashSet<String> = v["targets"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|n| n["node"].as_str().map(str::to_string))
            .collect();
        if out.insert(id.to_string(), t).is_some() {
            return Err(HfError::BandH(format!("{id}: two label rows")));
        }
    }
    Ok(out)
}

// --------------------------------------------------------------------------
// The cache (ENG-7)

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(values.len() * 4);
    for x in values {
        v.extend_from_slice(&x.to_le_bytes());
    }
    v
}

/// Writes a cache directory record by record.
pub struct CacheWriter {
    dir: PathBuf,
    hdim: usize,
    records: std::io::BufWriter<std::fs::File>,
    hidden: std::io::BufWriter<std::fs::File>,
    extras: std::io::BufWriter<std::fs::File>,
    rstop: std::io::BufWriter<std::fs::File>,
    hidden_rows: u64,
    rstop_rows: u64,
    count: u64,
}

impl CacheWriter {
    /// A new cache at `dir`, which must not exist.
    pub fn create(dir: &Path, hdim: usize) -> Result<Self, HfError> {
        if dir.exists() {
            return Err(HfError::Refused(format!(
                "{} exists; a cache is written once",
                dir.display()
            )));
        }
        std::fs::create_dir_all(dir).map_err(invalid)?;
        let open = |name: &str| -> Result<std::io::BufWriter<std::fs::File>, HfError> {
            Ok(std::io::BufWriter::new(
                std::fs::File::create(dir.join(name)).map_err(invalid)?,
            ))
        };
        Ok(Self {
            dir: dir.to_path_buf(),
            hdim,
            records: open(CACHE_FILES[0])?,
            hidden: open(CACHE_FILES[1])?,
            extras: open(CACHE_FILES[2])?,
            rstop: open(CACHE_FILES[3])?,
            hidden_rows: 0,
            rstop_rows: 0,
            count: 0,
        })
    }

    pub fn append(&mut self, r: &R1Record) -> Result<(), HfError> {
        let rstop_width = RSTOP_ROW_DIM + 2 * self.hdim;
        let mut snaps = Vec::new();
        for s in &r.snapshots {
            let n = s.candidates.len();
            if s.hidden.len() != n * self.hdim || s.extras.len() != n * RETURN_EXTRAS {
                return Err(HfError::BandH(format!(
                    "{} t={}: features do not match {n} candidates",
                    r.episode_id, s.t
                )));
            }
            snaps.push(json!({
                "t": s.t, "t_eff": s.t_eff, "candidates": s.candidates, "row": self.hidden_rows,
            }));
            self.hidden
                .write_all(&f32_bytes(&s.hidden))
                .map_err(invalid)?;
            self.extras
                .write_all(&f32_bytes(&s.extras))
                .map_err(invalid)?;
            self.hidden_rows += n as u64;
        }
        let mut stops = Vec::new();
        for (t, input) in &r.rstop {
            match input {
                Some(v) => {
                    if v.len() != rstop_width {
                        return Err(HfError::BandH(format!(
                            "{} t={t}: an rstop input of {} values, not {rstop_width}",
                            r.episode_id,
                            v.len()
                        )));
                    }
                    stops.push(json!({"t": t, "row": self.rstop_rows}));
                    self.rstop.write_all(&f32_bytes(v)).map_err(invalid)?;
                    self.rstop_rows += 1;
                }
                None => stops.push(json!({"t": t, "row": null})),
            }
        }
        let line = json!({
            "episode_id": r.episode_id,
            "walk_expanded": r.walk_expanded,
            "t_end": r.t_end,
            "stop_reason": r.stop_reason,
            "snapshots": snaps,
            "rstop": stops,
        });
        self.records
            .write_all(serde_json::to_string(&line).map_err(invalid)?.as_bytes())
            .and_then(|_| self.records.write_all(b"\n"))
            .map_err(invalid)?;
        self.count += 1;
        Ok(())
    }

    /// Flush every file and write the manifest: `extra` (provenance) plus the
    /// counts and each file's size and sha256.
    pub fn finish(mut self, extra: Value) -> Result<Value, HfError> {
        for w in [
            &mut self.records,
            &mut self.hidden,
            &mut self.extras,
            &mut self.rstop,
        ] {
            w.flush().map_err(invalid)?;
            w.get_ref().sync_all().map_err(invalid)?;
        }
        let mut files = serde_json::Map::new();
        for name in CACHE_FILES {
            let (bytes, sha) = hf_core::sha256_file(&self.dir.join(name)).map_err(invalid)?;
            files.insert(name.into(), json!({"bytes": bytes, "sha256": sha}));
        }
        let mut manifest = json!({
            "record_kind": CACHE_KIND,
            "governed_by": "experiments/real_walk_v2/R1_HEAD_DESIGN.md",
            "dtype": "f32 little-endian",
            "hidden_dimension": self.hdim,
            "extras_width": RETURN_EXTRAS,
            "rstop_width": RSTOP_ROW_DIM + 2 * self.hdim,
            "records": self.count,
            "hidden_rows": self.hidden_rows,
            "rstop_rows": self.rstop_rows,
            "files": files,
            "labels_opened": false,
            "training_authorized": false,
        });
        if let Some(obj) = extra.as_object() {
            for (k, v) in obj {
                if manifest.get(k).is_some() {
                    return Err(HfError::Invalid(format!(
                        "the cache manifest key {k} is the writer's own"
                    )));
                }
                manifest[k] = v.clone();
            }
        }
        std::fs::write(
            self.dir.join(CACHE_MANIFEST),
            hf_core::files::python_json_pretty(&manifest),
        )
        .map_err(invalid)?;
        Ok(manifest)
    }
}

struct IndexEntry {
    episode_id: String,
    walk_expanded: Vec<String>,
    t_end: usize,
    stop_reason: String,
    snapshots: Vec<(usize, usize, Vec<String>, u64)>,
    rstop: Vec<(usize, Option<u64>)>,
}

/// A cache, opened and checked: its manifest, every file's size and sha256,
/// the record count and the row counts. The f32 files are memory-mapped, so
/// a record is copied out of the page cache when it is read, never loaded
/// whole.
pub struct R1Cache {
    pub dir: PathBuf,
    pub manifest: Value,
    hdim: usize,
    index: Vec<IndexEntry>,
    hidden: memmap2::Mmap,
    extras: memmap2::Mmap,
    rstop: memmap2::Mmap,
}

fn map(path: &Path) -> Result<memmap2::Mmap, HfError> {
    let file =
        std::fs::File::open(path).map_err(|e| invalid(format!("{}: {e}", path.display())))?;
    // SAFETY: the cache is written once and never modified while it is read
    // (the runner refuses to open one without a complete manifest, and the
    // writer refuses an existing directory); a concurrent writer to these
    // files would be a protocol violation the per-file sha check at open
    // catches before any row is read.
    unsafe { memmap2::Mmap::map(&file) }.map_err(|e| invalid(format!("{}: {e}", path.display())))
}

impl R1Cache {
    pub fn open(dir: &Path) -> Result<Self, HfError> {
        hf_core::refuse_holdout(dir)?;
        let mpath = dir.join(CACHE_MANIFEST);
        let manifest: Value = serde_json::from_str(
            &std::fs::read_to_string(&mpath)
                .map_err(|e| HfError::BandH(format!("{}: {e}", mpath.display())))?,
        )
        .map_err(|e| HfError::BandH(format!("{}: {e}", mpath.display())))?;
        if manifest["record_kind"] != CACHE_KIND
            || manifest["training_authorized"] != Value::Bool(false)
        {
            return Err(HfError::BandH(format!(
                "{} is not an R1 walk cache manifest",
                mpath.display()
            )));
        }
        for name in CACHE_FILES {
            let path = dir.join(name);
            let (bytes, sha) = hf_core::sha256_file(&path)
                .map_err(|e| HfError::BandH(format!("{}: {e}", path.display())))?;
            let want = &manifest["files"][name];
            if want["bytes"].as_u64() != Some(bytes)
                || want["sha256"].as_str() != Some(sha.as_str())
            {
                return Err(HfError::BandH(format!(
                    "{}: {bytes} bytes {sha}, the manifest says {} bytes {}: a truncated or \
                     partial cache",
                    path.display(),
                    want["bytes"],
                    want["sha256"]
                )));
            }
        }
        let hdim = manifest["hidden_dimension"]
            .as_u64()
            .ok_or_else(|| HfError::BandH("the cache manifest has no hidden_dimension".into()))?
            as usize;
        let text = std::fs::read_to_string(dir.join(CACHE_FILES[0])).map_err(invalid)?;
        let mut index = Vec::new();
        for line in text.lines() {
            let v: Value = serde_json::from_str(line).map_err(|e| HfError::BandH(e.to_string()))?;
            let names = |x: &Value| -> Vec<String> {
                x.as_array()
                    .into_iter()
                    .flatten()
                    .map(|n| n.as_str().unwrap_or("").to_string())
                    .collect()
            };
            index.push(IndexEntry {
                episode_id: v["episode_id"].as_str().unwrap_or("").to_string(),
                walk_expanded: names(&v["walk_expanded"]),
                t_end: v["t_end"].as_u64().unwrap_or(0) as usize,
                stop_reason: v["stop_reason"].as_str().unwrap_or("").to_string(),
                snapshots: v["snapshots"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|s| {
                        (
                            s["t"].as_u64().unwrap_or(0) as usize,
                            s["t_eff"].as_u64().unwrap_or(0) as usize,
                            names(&s["candidates"]),
                            s["row"].as_u64().unwrap_or(0),
                        )
                    })
                    .collect(),
                rstop: v["rstop"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|s| (s["t"].as_u64().unwrap_or(0) as usize, s["row"].as_u64()))
                    .collect(),
            });
        }
        if manifest["records"].as_u64() != Some(index.len() as u64) {
            return Err(HfError::BandH(format!(
                "the cache holds {} records, its manifest says {}",
                index.len(),
                manifest["records"]
            )));
        }
        let hidden = map(&dir.join(CACHE_FILES[1]))?;
        let extras = map(&dir.join(CACHE_FILES[2]))?;
        let rstop = map(&dir.join(CACHE_FILES[3]))?;
        let rows = manifest["hidden_rows"].as_u64().unwrap_or(u64::MAX) as usize;
        let rstop_rows = manifest["rstop_rows"].as_u64().unwrap_or(u64::MAX) as usize;
        let rstop_width = RSTOP_ROW_DIM + 2 * hdim;
        if hidden.len() != rows * hdim * 4
            || extras.len() != rows * RETURN_EXTRAS * 4
            || rstop.len() != rstop_rows * rstop_width * 4
        {
            return Err(HfError::BandH(
                "the cache's f32 files disagree with its row counts".into(),
            ));
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            manifest,
            hdim,
            index,
            hidden,
            extras,
            rstop,
        })
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    pub fn episode_id(&self, i: usize) -> &str {
        &self.index[i].episode_id
    }

    fn floats(map: &memmap2::Mmap, start: usize, count: usize) -> Result<Vec<f32>, HfError> {
        let bytes = map
            .get(start * 4..(start + count) * 4)
            .ok_or_else(|| HfError::BandH("a cache row lies past its file's end".into()))?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }

    /// Record `i`, copied out of the maps.
    pub fn get(&self, i: usize) -> Result<R1Record, HfError> {
        let e = &self.index[i];
        let mut snapshots = Vec::with_capacity(e.snapshots.len());
        for (t, t_eff, candidates, row) in &e.snapshots {
            let n = candidates.len();
            let row = *row as usize;
            snapshots.push(SnapshotFeatures {
                t: *t,
                t_eff: *t_eff,
                candidates: candidates.clone(),
                hidden: Self::floats(&self.hidden, row * self.hdim, n * self.hdim)?,
                extras: Self::floats(&self.extras, row * RETURN_EXTRAS, n * RETURN_EXTRAS)?,
            });
        }
        let width = RSTOP_ROW_DIM + 2 * self.hdim;
        let mut rstop = Vec::with_capacity(e.rstop.len());
        for (t, row) in &e.rstop {
            rstop.push((
                *t,
                match row {
                    Some(r) => Some(Self::floats(&self.rstop, *r as usize * width, width)?),
                    None => None,
                },
            ));
        }
        Ok(R1Record {
            episode_id: e.episode_id.clone(),
            walk_expanded: e.walk_expanded.clone(),
            t_end: e.t_end,
            stop_reason: e.stop_reason.clone(),
            snapshots,
            rstop,
        })
    }
}

/// Two records hold the same features bit for bit: ids, walks, every
/// snapshot's candidates, `h_v` and extras, every rstop input.
pub fn bit_equal(a: &R1Record, b: &R1Record) -> bool {
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    a.episode_id == b.episode_id
        && a.walk_expanded == b.walk_expanded
        && a.t_end == b.t_end
        && a.snapshots.len() == b.snapshots.len()
        && a.snapshots.iter().zip(&b.snapshots).all(|(x, y)| {
            x.t == y.t
                && x.t_eff == y.t_eff
                && x.candidates == y.candidates
                && bits(&x.hidden) == bits(&y.hidden)
                && bits(&x.extras) == bits(&y.extras)
        })
        && a.rstop.len() == b.rstop.len()
        && a.rstop
            .iter()
            .zip(&b.rstop)
            .all(|((t1, x), (t2, y))| t1 == t2 && x.as_deref().map(bits) == y.as_deref().map(bits))
}

// --------------------------------------------------------------------------
// The losses (§5)

/// §5's stop label at snapshot `t`: 1 iff every target the capped walk
/// examines is already in `E_t` — `T ∩ E_cap ⊆ E_t`. The start is in both
/// sets and in no `T`, so the candidate lists (which exclude it) suffice.
pub fn stop_label(targets: &HashSet<String>, cand_t: &[String], cand_cap: &[String]) -> f32 {
    let at_t: HashSet<&str> = cand_t.iter().map(String::as_str).collect();
    let all = cand_cap
        .iter()
        .filter(|c| targets.contains(*c))
        .all(|c| at_t.contains(c.as_str()));
    if all {
        1.0
    } else {
        0.0
    }
}

pub struct R1Losses {
    pub ret: Tensor,
    pub stop: Tensor,
    pub total: Tensor,
    /// Positive candidates over every item of the batch (a count).
    pub positives: usize,
    pub candidates: usize,
    /// Mean `|E_t ∖ {s}|` over the batch's cap items.
    pub mean_examined: f64,
    pub mean_expansions: f64,
}

/// §5: `L_ret` — per item, BCE(ℓ_v, [v ∈ T]) averaged over its candidates;
/// items averaged within the episode; episodes within the batch — and
/// `L_stop` — BCE(rstop_t, y_t) averaged over the episode's stop decisions,
/// then over episodes; `L = L_ret + L_stop`. Unweighted.
pub fn r1_losses(
    model: &Model,
    records: &[&R1Record],
    targets: &[&HashSet<String>],
    spec: &R1Spec,
) -> Result<R1Losses, HfError> {
    if records.len() != targets.len() {
        return Err(HfError::Invalid("records and labels disagree".into()));
    }
    let device = model.device();
    let hdim = model.hidden_dimension() as usize;
    let width = hdim + RETURN_EXTRAS;
    let mut rows: Vec<f32> = Vec::new();
    let mut goals: Vec<f32> = Vec::new();
    // (episode, start row, n) per non-empty item
    let mut spans: Vec<(usize, i64, i64)> = Vec::new();
    let mut stop_rows: Vec<f32> = Vec::new();
    let mut stop_goals: Vec<f32> = Vec::new();
    let mut stop_owner: Vec<usize> = Vec::new();
    let (mut positives, mut examined, mut expansions) = (0usize, 0usize, 0usize);
    for (e, (r, t)) in records.iter().zip(targets).enumerate() {
        let cap = r.snapshot(spec.cap).ok_or_else(|| {
            HfError::BandH(format!("{}: no item at the cap {}", r.episode_id, spec.cap))
        })?;
        examined += cap.candidates.len();
        expansions += r.t_end;
        for s in &r.snapshots {
            let n = s.candidates.len();
            if n == 0 {
                continue;
            }
            spans.push((e, (rows.len() / width) as i64, n as i64));
            for (k, c) in s.candidates.iter().enumerate() {
                rows.extend_from_slice(&s.hidden[k * hdim..(k + 1) * hdim]);
                rows.extend_from_slice(&s.extras[k * RETURN_EXTRAS..(k + 1) * RETURN_EXTRAS]);
                let y = if t.contains(c) { 1.0 } else { 0.0 };
                positives += y as usize;
                goals.push(y);
            }
        }
        for (st, input) in &r.rstop {
            let Some(input) = input else { continue };
            let at = r.snapshot(*st).ok_or_else(|| {
                HfError::BandH(format!(
                    "{}: no item at the stop snapshot {st}",
                    r.episode_id
                ))
            })?;
            stop_rows.extend_from_slice(input);
            stop_goals.push(stop_label(t, &at.candidates, &cap.candidates));
            stop_owner.push(e);
        }
    }
    let n_rows = goals.len();
    let zero = || Tensor::zeros([], (Kind::Float, device));
    let ret = if n_rows == 0 {
        zero()
    } else {
        let x = Tensor::from_slice(&rows)
            .view([n_rows as i64, width as i64])
            .to_device(device);
        let y = Tensor::from_slice(&goals).to_device(device);
        let per = model
            .return_head_logits(&x)?
            .binary_cross_entropy_with_logits::<Tensor>(&y, None, None, Reduction::None);
        let mut by_episode: BTreeMap<usize, Vec<Tensor>> = BTreeMap::new();
        for (e, start, n) in &spans {
            by_episode
                .entry(*e)
                .or_default()
                .push(per.narrow(0, *start, *n).mean(Kind::Float));
        }
        let episode_means: Vec<Tensor> = by_episode
            .into_values()
            .map(|items| Tensor::stack(&items, 0).mean(Kind::Float))
            .collect();
        Tensor::stack(&episode_means, 0).mean(Kind::Float)
    };
    let stop = if stop_goals.is_empty() {
        zero()
    } else {
        let w = stop_rows.len() / stop_goals.len();
        let x = Tensor::from_slice(&stop_rows)
            .view([stop_goals.len() as i64, w as i64])
            .to_device(device);
        let y = Tensor::from_slice(&stop_goals).to_device(device);
        let per = model
            .rstop_head_logits(&x)?
            .binary_cross_entropy_with_logits::<Tensor>(&y, None, None, Reduction::None);
        let mut by_episode: BTreeMap<usize, Vec<i64>> = BTreeMap::new();
        for (k, e) in stop_owner.iter().enumerate() {
            by_episode.entry(*e).or_default().push(k as i64);
        }
        let episode_means: Vec<Tensor> = by_episode
            .into_values()
            .map(|ks| {
                per.index_select(0, &Tensor::from_slice(&ks).to_device(device))
                    .mean(Kind::Float)
            })
            .collect();
        Tensor::stack(&episode_means, 0).mean(Kind::Float)
    };
    let total = &ret + &stop;
    Ok(R1Losses {
        ret,
        stop,
        total,
        positives,
        candidates: n_rows,
        mean_examined: examined as f64 / records.len().max(1) as f64,
        mean_expansions: expansions as f64 / records.len().max(1) as f64,
    })
}

/// One update's numbers.
pub struct StepOut {
    pub ret: f64,
    pub stop: f64,
    pub total: f64,
    pub grad_norm: f64,
    pub finite: bool,
    pub positives: usize,
    pub candidates: usize,
    pub mean_examined: f64,
    pub mean_expansions: f64,
}

/// One R1 update: losses, backward, clip, and — only when every number is
/// finite — the optimiser step. `inject_nan` multiplies the total by NaN
/// (the fixture-only halt test).
pub fn train_step(
    model: &Model,
    optimiser: &mut AdamW,
    records: &[&R1Record],
    targets: &[&HashSet<String>],
    spec: &R1Spec,
    clip: Option<f64>,
    inject_nan: bool,
) -> Result<StepOut, HfError> {
    optimiser.zero_grad();
    let mut losses = r1_losses(model, records, targets, spec)?;
    if inject_nan {
        losses.total = f64::NAN * &losses.total;
    }
    if losses.total.requires_grad() {
        losses.total.backward();
    }
    let grad_norm = clip_grad_norm(&model.vs, clip);
    let v = |t: &Tensor| f64::try_from(t.detach()).unwrap_or(f64::NAN);
    let (ret, stop, total) = (v(&losses.ret), v(&losses.stop), v(&losses.total));
    let finite = ret.is_finite() && stop.is_finite() && total.is_finite() && grad_norm.is_finite();
    if finite {
        optimiser.step();
    }
    Ok(StepOut {
        ret,
        stop,
        total,
        grad_norm,
        finite,
        positives: losses.positives,
        candidates: losses.candidates,
        mean_examined: losses.mean_examined,
        mean_expansions: losses.mean_expansions,
    })
}

/// The eval row of one record (ENG-4 `--r1-eval`): logits only — the reader
/// applies τ and θ. `return_logits[j]` is snapshot j's per-candidate logits.
pub fn eval_logits(model: &Model, r: &R1Record) -> Result<Value, HfError> {
    let _guard = tch::no_grad_guard();
    let device = model.device();
    let hdim = model.hidden_dimension() as usize;
    let mut snaps = Vec::new();
    for s in &r.snapshots {
        let n = s.candidates.len();
        let logits: Vec<f32> = if n == 0 {
            Vec::new()
        } else {
            let mut x = Vec::with_capacity(n * (hdim + RETURN_EXTRAS));
            for k in 0..n {
                x.extend_from_slice(&s.hidden[k * hdim..(k + 1) * hdim]);
                x.extend_from_slice(&s.extras[k * RETURN_EXTRAS..(k + 1) * RETURN_EXTRAS]);
            }
            let t = Tensor::from_slice(&x)
                .view([n as i64, (hdim + RETURN_EXTRAS) as i64])
                .to_device(device);
            let out = model.return_head_logits(&t)?.to_device(Device::Cpu);
            let mut v = vec![0f32; n];
            out.copy_data(&mut v, n);
            v
        };
        snaps.push(json!({
            "t": s.t, "t_eff": s.t_eff, "candidates": s.candidates, "return_logits": logits,
        }));
    }
    let mut stops = Vec::new();
    for (t, input) in &r.rstop {
        let logit = match input {
            Some(v) => {
                let x = Tensor::from_slice(v)
                    .view([1, v.len() as i64])
                    .to_device(device);
                let out = model.rstop_head_logits(&x)?.to_device(Device::Cpu);
                let mut one = [0f32; 1];
                out.copy_data(&mut one, 1);
                Some(one[0])
            }
            None => None,
        };
        stops.push(json!({"t": t, "rstop_logit": logit}));
    }
    Ok(json!({
        "episode_id": r.episode_id,
        "walk_expanded": r.walk_expanded,
        "t_end": r.t_end,
        "stop_reason": r.stop_reason,
        "snapshots": snaps,
        "rstop": stops,
    }))
}
