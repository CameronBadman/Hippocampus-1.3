//! `hf-splits merge-deletions`: join deletion splits drawn by `hf-splits
//! deletions --source-ordinals` over contiguous blocks of ONE source into the
//! split a single draw over their union would have written.
//!
//! Every start is drawn independently — its X by `hash_int([label, id, tag])`,
//! its rotation tag by its ABSOLUTE source position — so the one-go draw's
//! streams are the parts' streams concatenated in ordinal order. The merge
//! re-encodes each gzip stream through the writer's own encoder, line by line,
//! so the bytes are the one-go draw's, and rebuilds the manifest from the first
//! part's: the per-start counts are summed, the uncached-node set is united,
//! `source_ordinals` is the union's, and ONE key is added at the end —
//! `chunks`, the parts' provenance. With `chunks` removed the manifest is
//! byte for byte the one-go draw's run with `--source-ordinals <first>..<last>`.
//!
//! Refused (exit 2, nothing written): parts whose manifests differ in anything
//! but the summed counts, the uncached set and `source_ordinals` (the source,
//! the graph, the sampler config, the cache and its state, the draw label, the
//! ranges, the per-start rule, the mode); a part without `source_ordinals`;
//! ranges that overlap or leave a gap; a part whose streams disagree with its
//! own record count; an episode id in two parts; an existing destination.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use clap::Parser;
use hf_core::{refuse_holdout, HfError};
use serde_json::{json, Map, Value};

#[derive(Parser, Debug)]
pub struct MergeArgs {
    /// a part: a deletion split written with --source-ordinals (repeat; any order)
    #[arg(long = "part", required = true, num_args = 1)]
    parts: Vec<PathBuf>,
    /// the merged split; refused when it exists
    #[arg(long)]
    destination: PathBuf,
}

/// Summed over the parts (per-start counts).
const SUMMED: [&str; 9] = [
    "records_written",
    "source_episodes_read",
    "candidates_with_neighbour",
    "candidates_without_neighbour",
    "candidates_without_vector",
    "candidates_isolating_the_start",
    "draws_computed",
    "records_per_draw_tag",
    "drops",
];
/// Rebuilt, not compared: the uncached count (a union), the block, the
/// rotation's fall-backs (a sum per tag).
const REBUILT: [&str; 3] = ["uncached_nodes", "source_ordinals", "rotate_fallbacks_to_u"];
const GZ_STREAMS: [&str; 3] = ["visible.jsonl.gz", "visible_h.jsonl.gz", "labels.jsonl.gz"];
const DELETION_KIND: &str = "r1_deletion_split_manifest_v1";
const COVERAGE_KIND: &str = "r1_deletion_coverage_manifest_v1";

fn invalid(e: impl std::fmt::Display) -> HfError {
    HfError::Invalid(e.to_string())
}

fn read_json(path: &Path) -> Result<Value, HfError> {
    serde_json::from_str(
        &std::fs::read_to_string(path)
            .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))?,
    )
    .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))
}

struct Part {
    dir: PathBuf,
    manifest: Value,
    first: u64,
    last: u64,
}

fn load_part(dir: &Path) -> Result<Part, HfError> {
    refuse_holdout(dir)?;
    let manifest = read_json(&dir.join("deletions.manifest.json"))?;
    let kind = manifest["record_kind"].as_str().unwrap_or("");
    if kind != DELETION_KIND && kind != COVERAGE_KIND {
        return Err(HfError::Refused(format!(
            "{}: not a deletion split ({kind:?})",
            dir.display()
        )));
    }
    if manifest["training_authorized"] != Value::Bool(false) {
        return Err(HfError::BandH(format!(
            "{}: the manifest does not set training_authorized false",
            dir.display()
        )));
    }
    let so = &manifest["source_ordinals"];
    let (Some(first), Some(last), Some(count)) = (
        so["first"].as_u64(),
        so["last"].as_u64(),
        so["count"].as_u64(),
    ) else {
        return Err(HfError::Refused(format!(
            "{}: drawn without --source-ordinals, so its block of the source is not recorded",
            dir.display()
        )));
    };
    if count != last - first + 1 || manifest["source_episodes_read"].as_u64() != Some(count) {
        return Err(HfError::BandH(format!(
            "{}: source_ordinals {first}..{last} disagrees with its counts",
            dir.display()
        )));
    }
    Ok(Part {
        dir: dir.to_path_buf(),
        manifest,
        first,
        last,
    })
}

/// A manifest with the summed and rebuilt keys removed: what must be equal.
fn comparable(m: &Value) -> Value {
    let mut m = m.as_object().cloned().unwrap_or_default();
    for k in SUMMED.iter().chain(REBUILT.iter()) {
        m.remove(*k);
    }
    Value::Object(m)
}

/// Lines of a gzip stream, streamed (a labels stream can be large).
fn gz_lines(path: &Path) -> Result<impl Iterator<Item = std::io::Result<String>>, HfError> {
    let file =
        std::fs::File::open(path).map_err(|e| invalid(format!("{}: {e}", path.display())))?;
    Ok(
        BufReader::with_capacity(1 << 20, flate2::read::GzDecoder::new(file))
            .lines()
            .filter(|l| l.as_ref().map(|s| !s.trim().is_empty()).unwrap_or(true)),
    )
}

fn text_lines(path: &Path) -> Result<Vec<String>, HfError> {
    Ok(std::fs::read_to_string(path)
        .map_err(|e| invalid(format!("{}: {e}", path.display())))?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(str::to_string)
        .collect())
}

fn sum_maps(parts: &[Part], key: &str) -> Result<Value, HfError> {
    let mut out: BTreeMap<String, u64> = BTreeMap::new();
    for p in parts {
        let m = &p.manifest[key];
        if m.is_null() {
            continue;
        }
        let obj = m
            .as_object()
            .ok_or_else(|| HfError::BandH(format!("{key} is not a map")))?;
        for (k, v) in obj {
            *out.entry(k.clone()).or_default() += v
                .as_u64()
                .ok_or_else(|| HfError::BandH(format!("{key}.{k} is not a count")))?;
        }
    }
    serde_json::to_value(out).map_err(invalid)
}

pub fn merge_deletions(args: MergeArgs) -> Result<(), HfError> {
    refuse_holdout(&args.destination)?;
    if args.destination.exists() {
        return Err(HfError::Refused(format!(
            "{} exists; a merged split is written once",
            args.destination.display()
        )));
    }
    if args.parts.len() < 2 {
        return Err(HfError::Invalid(
            "merge-deletions needs at least two parts".into(),
        ));
    }
    let mut parts: Vec<Part> = args
        .parts
        .iter()
        .map(|d| load_part(d))
        .collect::<Result<_, _>>()?;
    parts.sort_by_key(|p| p.first);
    // one source, one graph, one config, one cache state, one label: every
    // key but the per-start counts must agree
    let base = comparable(&parts[0].manifest);
    for p in &parts[1..] {
        let other = comparable(&p.manifest);
        if other != base {
            let differ: Vec<&String> = base
                .as_object()
                .unwrap()
                .keys()
                .chain(other.as_object().unwrap().keys())
                .filter(|k| base.get(*k) != other.get(*k))
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            return Err(HfError::Refused(format!(
                "{} and {} were not drawn alike: {differ:?} differ",
                parts[0].dir.display(),
                p.dir.display()
            )));
        }
    }
    // contiguous blocks, no overlap, no gap
    for w in parts.windows(2) {
        if w[1].first != w[0].last + 1 {
            return Err(HfError::Refused(format!(
                "the parts' source blocks {}..{} and {}..{} {}",
                w[0].first,
                w[0].last,
                w[1].first,
                w[1].last,
                if w[1].first <= w[0].last {
                    "overlap"
                } else {
                    "leave a gap"
                }
            )));
        }
    }
    let coverage = parts[0].manifest["record_kind"] == COVERAGE_KIND;
    // each part's streams agree with its own count, and no id is in two parts
    let mut seen_ids: HashSet<String> = HashSet::new();
    let mut part_files: Vec<Map<String, Value>> = Vec::new();
    for p in &parts {
        let mut files = Map::new();
        let names: &[&str] = if coverage {
            &["uncached_nodes.txt", "deletions.manifest.json"]
        } else {
            &[
                "visible.jsonl.gz",
                "visible_h.jsonl.gz",
                "labels.jsonl.gz",
                "queries/vectors.jsonl",
                "queries/manifest.json",
                "uncached_nodes.txt",
                "deletions.manifest.json",
            ]
        };
        for n in names {
            let (_, sha) = hf_core::sha256_file(&p.dir.join(n))
                .map_err(|e| HfError::BandH(format!("{}: {n}: {e}", p.dir.display())))?;
            files.insert((*n).into(), sha.into());
        }
        part_files.push(files);
        if coverage {
            continue;
        }
        let want = p.manifest["records_written"].as_u64().unwrap_or(u64::MAX) as usize;
        for stream in GZ_STREAMS {
            let mut n = 0usize;
            for line in gz_lines(&p.dir.join(stream))? {
                let line = line.map_err(invalid)?;
                if stream == "labels.jsonl.gz" {
                    let v: Value = serde_json::from_str(&line).map_err(invalid)?;
                    let id = v["episode_id"].as_str().unwrap_or("").to_string();
                    if !seen_ids.insert(id.clone()) {
                        return Err(HfError::Refused(format!("{id} is in two parts")));
                    }
                }
                n += 1;
            }
            if n != want {
                return Err(HfError::BandH(format!(
                    "{}: {stream} holds {n} records, its manifest says {want}",
                    p.dir.display()
                )));
            }
        }
        let q = text_lines(&p.dir.join("queries/vectors.jsonl"))?.len();
        let qm = read_json(&p.dir.join("queries/manifest.json"))?;
        if q != want || qm["count"].as_u64() != Some(want as u64) {
            return Err(HfError::BandH(format!(
                "{}: the query sidecar holds {q} rows, its manifest {}, the split {want}",
                p.dir.display(),
                qm["count"]
            )));
        }
    }
    // the query sidecars' manifests must agree but for their counts
    if !coverage {
        let strip = |dir: &Path| -> Result<Value, HfError> {
            let mut m = read_json(&dir.join("queries/manifest.json"))?;
            m["count"] = Value::Null;
            Ok(m)
        };
        let q0 = strip(&parts[0].dir)?;
        for p in &parts[1..] {
            if strip(&p.dir)? != q0 {
                return Err(HfError::Refused(format!(
                    "{}: its query sidecar's manifest is not {}'s",
                    p.dir.display(),
                    parts[0].dir.display()
                )));
            }
        }
    }
    // the uncached set, united
    let mut uncached: BTreeSet<String> = BTreeSet::new();
    for p in &parts {
        uncached.extend(text_lines(&p.dir.join("uncached_nodes.txt"))?);
    }
    // ---- write
    let dest = &args.destination;
    std::fs::create_dir_all(dest).map_err(invalid)?;
    if !coverage {
        for stream in GZ_STREAMS {
            let mut out = crate::deletions::Gz::create(&dest.join(stream))?;
            for p in &parts {
                for line in gz_lines(&p.dir.join(stream))? {
                    out.raw_line(line.map_err(invalid)?.as_bytes())?;
                }
            }
            out.finish()?;
        }
        let qdir = dest.join("queries");
        std::fs::create_dir_all(&qdir).map_err(invalid)?;
        let mut q = std::fs::File::create(qdir.join("vectors.jsonl")).map_err(invalid)?;
        let mut total = 0u64;
        for p in &parts {
            let mut src =
                std::fs::File::open(p.dir.join("queries/vectors.jsonl")).map_err(invalid)?;
            std::io::copy(&mut src, &mut q).map_err(invalid)?;
            total += p.manifest["records_written"].as_u64().unwrap_or(0);
        }
        q.sync_all().map_err(invalid)?;
        let mut qm = read_json(&parts[0].dir.join("queries/manifest.json"))?;
        qm["count"] = total.into();
        let path = qdir.join("manifest.json");
        std::fs::write(&path, hf_core::files::python_json_pretty(&qm)).map_err(invalid)?;
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).map_err(invalid)?;
    }
    let mut list = String::new();
    for n in &uncached {
        list.push_str(n);
        list.push('\n');
    }
    std::fs::write(dest.join("uncached_nodes.txt"), list).map_err(invalid)?;
    // the manifest: the first part's, in its own key order, the per-start
    // counts summed, the union's block, then the parts' provenance
    let mut manifest = parts[0].manifest.clone();
    for key in SUMMED {
        if manifest.get(key).is_none() {
            continue;
        }
        manifest[key] = if manifest[key].is_object() {
            sum_maps(&parts, key)?
        } else {
            let mut total = 0u64;
            for p in &parts {
                total += p.manifest[key]
                    .as_u64()
                    .ok_or_else(|| HfError::BandH(format!("{key} is not a count")))?;
            }
            total.into()
        };
    }
    manifest["rotate_fallbacks_to_u"] = sum_maps(&parts, "rotate_fallbacks_to_u")?;
    manifest["uncached_nodes"] = uncached.len().into();
    let (first, last) = (parts[0].first, parts[parts.len() - 1].last);
    manifest["source_ordinals"] = json!({
        "declared": format!("{first}..{last}"),
        "first": first,
        "last": last,
        "count": last - first + 1,
    });
    manifest["chunks"] = json!({
        "merged_by": "hippo-13 hf-splits merge-deletions",
        "note": "with this key removed the manifest is the one-go draw's over source_ordinals",
        "parts": parts.iter().zip(&part_files).map(|(p, files)| json!({
            "dir": p.dir.to_string_lossy(),
            "source_ordinals": p.manifest["source_ordinals"]["declared"],
            "first": p.first,
            "last": p.last,
            "records_written": p.manifest["records_written"],
            "files_sha256": files,
        })).collect::<Vec<_>>(),
    });
    std::fs::write(
        dest.join("deletions.manifest.json"),
        hf_core::files::python_json_pretty(&manifest),
    )
    .map_err(invalid)?;
    println!(
        "merge-deletions: {} parts, source {first}..{last}, {} records -> {}",
        parts.len(),
        manifest["records_written"],
        dest.display()
    );
    std::io::stdout().flush().ok();
    Ok(())
}
