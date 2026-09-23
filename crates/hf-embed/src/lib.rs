//! The real-walk embedding cache, as `real_walk_embed.py` writes it and
//! `embeddings_v5` reads it: `manifest.json` (record kind
//! `real_walk_embedding_manifest_v5`, `training_authorized: false`, the served
//! model digest, `dimension`, `count`, `text_char_limit`, `text_sha256`,
//! `truncated`) beside an append-only `vectors.jsonl` of
//! `{"node": …, "vector": […]}` lines. Nothing in this crate starts a daemon.
//!
//! A binary sidecar (`vectors.f32`, row-major float32, plus
//! `vectors.index.json`) is built on first use and memory-mapped afterwards;
//! it is bound to the size and SHA-256 of the `vectors.jsonl` it came from and
//! refused when that file has moved. It is regenerable and never evidence.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use hf_core::canonical::sha256_file;
use hf_core::{python_json_compact, HfError};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const MANIFEST_KIND: &str = "real_walk_embedding_manifest_v5";
pub const SIDECAR_KIND: &str = "hippo13_embedding_sidecar_v1";

/// The cache manifest; unknown fields are preserved through `extra`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Manifest {
    pub record_kind: String,
    pub model: String,
    pub model_digest: String,
    pub base_url: String,
    pub dimension: u32,
    pub count: u64,
    pub text_char_limit: u64,
    #[serde(default)]
    pub text_sha256: Option<String>,
    #[serde(default)]
    pub truncated: HashMap<String, u64>,
    pub training_authorized: bool,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

/// `read_manifest`: the kind must match and `training_authorized` must be false.
pub fn read_manifest(directory: &Path) -> Result<Manifest, HfError> {
    let path = directory.join("manifest.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))?;
    let manifest: Manifest = serde_json::from_str(&text)
        .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))?;
    if manifest.record_kind != MANIFEST_KIND {
        return Err(HfError::BandH(format!(
            "{} is not an embedding manifest",
            path.display()
        )));
    }
    if manifest.training_authorized {
        return Err(HfError::BandH(format!(
            "{} does not set training_authorized false",
            path.display()
        )));
    }
    Ok(manifest)
}

/// `write_manifest`: forces the kind and `training_authorized: false`;
/// `json.dumps(indent=2, sort_keys=True)` without a trailing newline, mode 0644.
pub fn write_manifest(directory: &Path, manifest: &Manifest) -> Result<(), HfError> {
    let mut value = serde_json::to_value(manifest).map_err(|e| HfError::Invalid(e.to_string()))?;
    value["record_kind"] = Value::from(MANIFEST_KIND);
    value["training_authorized"] = Value::Bool(false);
    let path = directory.join("manifest.json");
    std::fs::write(&path, hf_core::files::python_json_pretty(&value))
        .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))?;
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644))
        .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))?;
    Ok(())
}

#[derive(Deserialize)]
struct VectorLine {
    node: String,
    vector: Vec<f32>,
}

/// The node ids already present in `vectors.jsonl` (the resume guard's `done`).
pub fn nodes_present(directory: &Path) -> Result<Vec<String>, HfError> {
    let path = directory.join("vectors.jsonl");
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = std::fs::File::open(&path)
        .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))?;
    let mut out = Vec::new();
    for line in BufReader::with_capacity(1 << 20, file).lines() {
        let line = line.map_err(|e| HfError::Invalid(e.to_string()))?;
        if line.trim().is_empty() {
            continue;
        }
        // only the node id is needed; a full parse of every vector would cost seconds
        let v: Value = serde_json::from_str(&line)
            .map_err(|e| HfError::Invalid(format!("vectors.jsonl: {e}")))?;
        out.push(
            v.get("node")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
        );
    }
    Ok(out)
}

/// Append one vector line exactly as Python does (`json.dumps` default separators).
pub fn append_vector(file: &mut impl Write, node: &str, vector: &[f64]) -> std::io::Result<()> {
    let line = python_json_compact(&serde_json::json!({"node": node, "vector": vector}));
    file.write_all(line.as_bytes())?;
    file.write_all(b"\n")
}

#[derive(Serialize, Deserialize)]
struct SidecarIndex {
    record_kind: String,
    dimension: u32,
    count: u64,
    source_bytes: u64,
    source_sha256: String,
    nodes: Vec<String>,
    training_authorized: bool,
}

/// Every vector of a cache, row-major float32, with unit-normalised rows beside.
pub struct EmbeddingMatrix {
    pub nodes: Vec<String>,
    pub dimension: usize,
    index: HashMap<String, u32>,
    data: Data,
    unit: Vec<f32>,
}

enum Data {
    Owned(Vec<f32>),
    Mapped(memmap2::Mmap),
}

impl Data {
    fn as_slice(&self) -> &[f32] {
        match self {
            Data::Owned(v) => v.as_slice(),
            Data::Mapped(m) => {
                // the sidecar is written by this crate in native byte order and 4-byte aligned
                let bytes: &[u8] = m;
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) }
            }
        }
    }
}

impl EmbeddingMatrix {
    /// `load_embedding_matrix`: from the sidecar when it matches `vectors.jsonl`,
    /// else by parsing `vectors.jsonl` (and writing the sidecar when the
    /// directory is writable). Row order is file order; dimension and count
    /// are checked against the manifest as Python checks them.
    pub fn load(directory: &Path) -> Result<Self, HfError> {
        let manifest = read_manifest(directory)?;
        let source = directory.join("vectors.jsonl");
        let (source_bytes, source_sha256) = sha256_file(&source)
            .map_err(|e| HfError::Invalid(format!("{}: {e}", source.display())))?;
        if let Some(loaded) =
            Self::from_sidecar(directory, &manifest, source_bytes, &source_sha256)?
        {
            return Ok(loaded);
        }
        let (nodes, data) = parse_vectors(&source, manifest.dimension as usize)?;
        if nodes.len() as u64 != manifest.count {
            return Err(HfError::BandH(format!(
                "vectors.jsonl carries {} vectors but the manifest says {}",
                nodes.len(),
                manifest.count
            )));
        }
        let _ = Self::write_sidecar(
            directory,
            &manifest,
            &nodes,
            &data,
            source_bytes,
            &source_sha256,
        );
        Ok(Self::build(
            nodes,
            manifest.dimension as usize,
            Data::Owned(data),
        ))
    }

    /// A matrix from rows already in memory (the fixture world's embeddings).
    pub fn from_rows(nodes: Vec<String>, dimension: usize, data: Vec<f32>) -> Self {
        assert_eq!(
            data.len(),
            nodes.len() * dimension,
            "rows do not match the dimension"
        );
        Self::build(nodes, dimension, Data::Owned(data))
    }

    fn build(nodes: Vec<String>, dimension: usize, data: Data) -> Self {
        let index = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i as u32))
            .collect();
        let rows = data.as_slice();
        let mut unit = vec![0f32; rows.len()];
        for (r, row) in rows.chunks_exact(dimension.max(1)).enumerate() {
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for (j, x) in row.iter().enumerate() {
                unit[r * dimension + j] = x / norm;
            }
        }
        Self {
            nodes,
            dimension,
            index,
            data,
            unit,
        }
    }

    fn sidecar_paths(directory: &Path) -> (PathBuf, PathBuf) {
        (
            directory.join("vectors.f32"),
            directory.join("vectors.index.json"),
        )
    }

    fn from_sidecar(
        directory: &Path,
        manifest: &Manifest,
        source_bytes: u64,
        source_sha256: &str,
    ) -> Result<Option<Self>, HfError> {
        let (data_path, index_path) = Self::sidecar_paths(directory);
        if !data_path.exists() || !index_path.exists() {
            return Ok(None);
        }
        let index: SidecarIndex = serde_json::from_str(
            &std::fs::read_to_string(&index_path).map_err(|e| HfError::Invalid(e.to_string()))?,
        )
        .map_err(|e| HfError::Invalid(format!("{}: {e}", index_path.display())))?;
        if index.record_kind != SIDECAR_KIND
            || index.source_bytes != source_bytes
            || index.source_sha256 != source_sha256
            || index.dimension != manifest.dimension
            || index.count != manifest.count
        {
            return Err(HfError::Refused(format!(
                "{} does not match vectors.jsonl (rebuild it by deleting the sidecar)",
                index_path.display()
            )));
        }
        let file = std::fs::File::open(&data_path).map_err(|e| HfError::Invalid(e.to_string()))?;
        let map =
            unsafe { memmap2::Mmap::map(&file) }.map_err(|e| HfError::Invalid(e.to_string()))?;
        if map.len() != index.count as usize * index.dimension as usize * 4 {
            return Err(HfError::Refused(format!(
                "{} has the wrong size",
                data_path.display()
            )));
        }
        Ok(Some(Self::build(
            index.nodes,
            index.dimension as usize,
            Data::Mapped(map),
        )))
    }

    fn write_sidecar(
        directory: &Path,
        manifest: &Manifest,
        nodes: &[String],
        data: &[f32],
        source_bytes: u64,
        source_sha256: &str,
    ) -> Result<(), HfError> {
        let (data_path, index_path) = Self::sidecar_paths(directory);
        let tmp = data_path.with_extension("f32.tmp");
        {
            let mut file =
                std::fs::File::create(&tmp).map_err(|e| HfError::Invalid(e.to_string()))?;
            let bytes =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
            file.write_all(bytes)
                .and_then(|_| file.sync_all())
                .map_err(|e| HfError::Invalid(e.to_string()))?;
        }
        std::fs::rename(&tmp, &data_path).map_err(|e| HfError::Invalid(e.to_string()))?;
        let index = SidecarIndex {
            record_kind: SIDECAR_KIND.into(),
            dimension: manifest.dimension,
            count: nodes.len() as u64,
            source_bytes,
            source_sha256: source_sha256.to_string(),
            nodes: nodes.to_vec(),
            training_authorized: false,
        };
        std::fs::write(
            &index_path,
            serde_json::to_string(&index).map_err(|e| HfError::Invalid(e.to_string()))?,
        )
        .map_err(|e| HfError::Invalid(e.to_string()))
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn contains(&self, node: &str) -> bool {
        self.index.contains_key(node)
    }

    pub fn row_of(&self, node: &str) -> Option<u32> {
        self.index.get(node).copied()
    }

    /// The raw vector of a node.
    pub fn get(&self, node: &str) -> Option<&[f32]> {
        self.index.get(node).map(|r| self.row(*r))
    }

    pub fn row(&self, r: u32) -> &[f32] {
        let d = self.dimension;
        &self.data.as_slice()[r as usize * d..(r as usize + 1) * d]
    }

    /// The unit-normalised vector of a row (a zero vector stays zero).
    pub fn unit_row(&self, r: u32) -> &[f32] {
        let d = self.dimension;
        &self.unit[r as usize * d..(r as usize + 1) * d]
    }

    pub fn unit(&self, node: &str) -> Option<&[f32]> {
        self.index.get(node).map(|r| self.unit_row(*r))
    }
}

/// Refuse a query sidecar that did not come from the node cache's encoder.
///
/// Every stage-1 channel is `cos(question, node)`, which is meaningless unless
/// both vectors were written by the same model at the same width: a 384-wide
/// question against 768-wide nodes, or a question from another encoder, would
/// read as a walk whose compass is noise rather than as a failure. The two
/// manifests must agree on `dimension` and on `model_digest`.
pub fn same_encoder(nodes: &Manifest, queries: &Manifest) -> Result<(), HfError> {
    if nodes.dimension != queries.dimension {
        return Err(HfError::BandH(format!(
            "the query cache is {} wide and the node cache {}",
            queries.dimension, nodes.dimension
        )));
    }
    if nodes.model_digest != queries.model_digest {
        return Err(HfError::BandH(format!(
            "the query cache was written by {} ({}) and the node cache by {} ({}); \
             a cosine between two encoders' vectors is not a similarity",
            queries.model, queries.model_digest, nodes.model, nodes.model_digest
        )));
    }
    Ok(())
}

/// How many of the given ids the cache holds, over how many DISTINCT ids were
/// given: `(present, total)`. A node the cache lacks is a silent zero vector
/// wherever an episode is indexed, which is what `--expect-embedding-coverage`
/// exists to catch before a run reads the result as evidence.
pub fn coverage<'a>(
    ids: impl IntoIterator<Item = &'a str>,
    matrix: &EmbeddingMatrix,
) -> (usize, usize) {
    // a pool of 40,000 episodes names about 1.6 M nodes; the set is the
    // distinct ones, and hashing beats ordering here since only the count is read
    let distinct: HashMap<&str, ()> = ids.into_iter().map(|n| (n, ())).collect();
    let present = distinct.keys().filter(|n| matrix.contains(n)).count();
    (present, distinct.len())
}

/// Cosine between two vectors, `0.0` when either norm is zero — the Python `cosine`.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn parse_vectors(path: &Path, dimension: usize) -> Result<(Vec<String>, Vec<f32>), HfError> {
    let file = std::fs::File::open(path)
        .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))?;
    let mut nodes = Vec::new();
    let mut data = Vec::new();
    for (i, line) in BufReader::with_capacity(1 << 20, file).lines().enumerate() {
        let line = line.map_err(|e| HfError::Invalid(e.to_string()))?;
        if line.trim().is_empty() {
            continue;
        }
        let record: VectorLine = serde_json::from_str(&line)
            .map_err(|e| HfError::Invalid(format!("vectors.jsonl line {}: {e}", i + 1)))?;
        if record.vector.len() != dimension {
            return Err(HfError::BandH(format!(
                "vector for {} has {} values, manifest dimension {dimension}",
                record.node,
                record.vector.len()
            )));
        }
        nodes.push(record.node);
        data.extend_from_slice(&record.vector);
    }
    Ok((nodes, data))
}

/// What `merge_caches` wrote.
#[derive(Clone, Debug)]
pub struct MergeReport {
    pub manifest: Manifest,
    pub written: u64,
    pub duplicates_identical: u64,
}

#[derive(Deserialize)]
struct VectorLine64 {
    node: String,
    vector: Vec<f64>,
}

fn vector_fingerprint(vector: &[f64]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for x in vector {
        h.update(x.to_bits().to_le_bytes());
    }
    h.finalize().into()
}

/// ENG-2: several v5 caches — in practice the teacher's per-destination
/// `queries/` caches, whose ids are EPISODE ids — merged into ONE cache under
/// `into` with one manifest, so a single `--query-embeddings-dir` covers
/// train, screen, screen2 and the vault. (A destination's `discarded/`
/// sibling has no cache of its own: its vectors are in the destination's
/// `queries/`, so naming that cache covers the discards.)
///
/// Every source must agree with the first on the encoder
/// (`same_encoder`: `dimension` and `model_digest`), on `text_char_limit` and
/// on `keyed_by` (so a node cache cannot be folded into a query cache), and
/// must hold exactly its manifest's `count` of lines, each `dimension` wide.
/// An id in two sources is kept once when the two vectors are bit-identical as
/// f64 and REFUSED otherwise (the test-writer's and the teacher's copies of
/// one episode are different questions). Lines are copied as the sources wrote
/// them, sources in the order given, file order within each. The manifest
/// records each source's path, count, manifest (whole, and its canonical
/// sha256) and `vectors.jsonl` size and sha256. Written under a temporary
/// name and renamed; an existing `into` is refused; no sidecar is copied.
pub fn merge_caches(sources: &[PathBuf], into: &Path) -> Result<MergeReport, HfError> {
    hf_core::refuse_holdout(into)?;
    if sources.is_empty() {
        return Err(HfError::Invalid(
            "merge needs at least one source cache".into(),
        ));
    }
    if into.exists() || into.symlink_metadata().is_ok() {
        return Err(HfError::Refused(format!(
            "{} already exists",
            into.display()
        )));
    }
    let mut named = std::collections::HashSet::new();
    let mut manifests = Vec::with_capacity(sources.len());
    for source in sources {
        hf_core::refuse_holdout(source)?;
        let canonical = std::fs::canonicalize(source)
            .map_err(|e| HfError::Invalid(format!("{}: {e}", source.display())))?;
        if !named.insert(canonical) {
            return Err(HfError::Refused(format!(
                "{} is named twice",
                source.display()
            )));
        }
        manifests.push(read_manifest(source)?);
    }
    let first = manifests[0].clone();
    for (source, m) in sources.iter().zip(&manifests).skip(1) {
        same_encoder(&first, m)
            .map_err(|e| HfError::BandH(format!("{}: {e}", source.display())))?;
        if m.text_char_limit != first.text_char_limit {
            return Err(HfError::BandH(format!(
                "{}: text_char_limit {} differs from the first source's {}",
                source.display(),
                m.text_char_limit,
                first.text_char_limit
            )));
        }
        if m.extra.get("keyed_by") != first.extra.get("keyed_by") {
            return Err(HfError::BandH(format!(
                "{}: keyed_by {:?} differs from the first source's {:?}",
                source.display(),
                m.extra.get("keyed_by"),
                first.extra.get("keyed_by")
            )));
        }
    }
    let parent = into
        .parent()
        .ok_or_else(|| HfError::Invalid("--into has no parent".into()))?;
    if !parent.as_os_str().is_empty() {
        std::fs::create_dir_all(parent)
            .map_err(|e| HfError::Invalid(format!("{}: {e}", parent.display())))?;
    }
    let temporary = parent.join(format!(
        ".tmp-{}-{}",
        into.file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default(),
        std::process::id()
    ));
    std::fs::create_dir(&temporary)
        .map_err(|e| HfError::Invalid(format!("{}: {e}", temporary.display())))?;
    let result = (|| -> Result<MergeReport, HfError> {
        let out_path = temporary.join("vectors.jsonl");
        let mut out = std::io::BufWriter::new(
            std::fs::File::create(&out_path)
                .map_err(|e| HfError::Invalid(format!("{}: {e}", out_path.display())))?,
        );
        let mut seen: HashMap<String, ([u8; 32], usize)> = HashMap::new();
        let mut truncated: HashMap<String, u64> = HashMap::new();
        let mut records = Vec::with_capacity(sources.len());
        let mut written = 0u64;
        let mut duplicates = 0u64;
        for (i, (source, m)) in sources.iter().zip(&manifests).enumerate() {
            let path = source.join("vectors.jsonl");
            let (bytes, sha) = sha256_file(&path)
                .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))?;
            let file = std::fs::File::open(&path)
                .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))?;
            let mut lines = 0u64;
            let mut kept = 0u64;
            let mut dup_here = 0u64;
            for (n, line) in BufReader::with_capacity(1 << 20, file).lines().enumerate() {
                let line = line.map_err(|e| HfError::Invalid(e.to_string()))?;
                if line.trim().is_empty() {
                    continue;
                }
                lines += 1;
                let record: VectorLine64 = serde_json::from_str(&line).map_err(|e| {
                    HfError::Invalid(format!("{} line {}: {e}", path.display(), n + 1))
                })?;
                if record.vector.len() != first.dimension as usize {
                    return Err(HfError::BandH(format!(
                        "{}: the vector for {} has {} values, the dimension is {}",
                        path.display(),
                        record.node,
                        record.vector.len(),
                        first.dimension
                    )));
                }
                let print = vector_fingerprint(&record.vector);
                match seen.get(&record.node) {
                    Some((earlier, _)) if *earlier == print => {
                        dup_here += 1;
                        continue;
                    }
                    Some((_, from)) => {
                        return Err(HfError::BandH(format!(
                            "{} is in {} and in {} with DIFFERENT vectors; refusing to pick one",
                            record.node,
                            sources[*from].display(),
                            source.display()
                        )));
                    }
                    None => {}
                }
                seen.insert(record.node.clone(), (print, i));
                out.write_all(line.as_bytes())
                    .and_then(|_| out.write_all(b"\n"))
                    .map_err(|e| HfError::Invalid(e.to_string()))?;
                kept += 1;
            }
            if lines != m.count {
                return Err(HfError::BandH(format!(
                    "{}: vectors.jsonl carries {lines} vectors but the manifest says {}",
                    source.display(),
                    m.count
                )));
            }
            for (node, n) in &m.truncated {
                if let Some(prev) = truncated.insert(node.clone(), *n) {
                    if prev != *n {
                        return Err(HfError::BandH(format!(
                            "{node} is truncated at {prev} and at {n} characters in two sources"
                        )));
                    }
                }
            }
            let manifest_value =
                serde_json::to_value(m).map_err(|e| HfError::Invalid(e.to_string()))?;
            records.push(serde_json::json!({
                "path": source.to_string_lossy(),
                "count": m.count,
                "written": kept,
                "duplicates_identical": dup_here,
                "vectors_bytes": bytes,
                "vectors_sha256": sha,
                "manifest_sha256": hf_core::canonical_sha256(&manifest_value)?,
                "manifest": manifest_value,
            }));
            written += kept;
            duplicates += dup_here;
        }
        out.flush().map_err(|e| HfError::Invalid(e.to_string()))?;
        out.get_ref()
            .sync_all()
            .map_err(|e| HfError::Invalid(e.to_string()))?;
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&out_path, std::fs::Permissions::from_mode(0o644))
            .map_err(|e| HfError::Invalid(e.to_string()))?;
        let mut extra = serde_json::Map::new();
        if let Some(k) = first.extra.get("keyed_by") {
            extra.insert("keyed_by".into(), k.clone());
        }
        extra.insert("merged_from".into(), Value::Array(records));
        extra.insert("duplicates_identical".into(), duplicates.into());
        extra.insert(
            "merge_note".into(),
            Value::from(
                "hf-embed merge: the sources' lines copied in the order given; an id in \
                 two sources is kept once when its vectors are bit-identical and refused \
                 otherwise; model and base_url are the first source's, every source's \
                 own manifest is under merged_from",
            ),
        );
        let manifest = Manifest {
            record_kind: MANIFEST_KIND.into(),
            model: first.model.clone(),
            model_digest: first.model_digest.clone(),
            base_url: first.base_url.clone(),
            dimension: first.dimension,
            count: written,
            text_char_limit: first.text_char_limit,
            text_sha256: None,
            truncated,
            training_authorized: false,
            extra,
        };
        write_manifest(&temporary, &manifest)?;
        Ok(MergeReport {
            manifest,
            written,
            duplicates_identical: duplicates,
        })
    })();
    match result {
        Ok(report) => {
            std::fs::rename(&temporary, into)
                .map_err(|e| HfError::Invalid(format!("rename to {}: {e}", into.display())))?;
            Ok(report)
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&temporary);
            Err(e)
        }
    }
}

/// Raised when the server refuses an input for its context length.
#[derive(Debug)]
pub struct ContextOverflow;

/// `OllamaEmbedClient`: `/api/tags` for the served digest, `/api/embed` for vectors.
pub struct OllamaEmbedClient {
    agent: ureq::Agent,
    base_url: String,
    model: String,
}

impl OllamaEmbedClient {
    pub fn new(base_url: &str, model: &str, timeout_seconds: f64) -> Self {
        let config = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_global(Some(std::time::Duration::from_secs_f64(timeout_seconds)))
            .build();
        Self {
            agent: config.into(),
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
        }
    }

    /// The digest the server reports for the model (`name` or `model` equal to
    /// the tag, or `<tag>:latest` when the tag has no colon).
    pub fn served_digest(&self) -> Result<String, HfError> {
        let mut response = self
            .agent
            .get(format!("{}/api/tags", self.base_url))
            .call()
            .map_err(|e| HfError::Invalid(format!("ollama /api/tags: {e}")))?;
        let body: Value = response
            .body_mut()
            .read_json()
            .map_err(|e| HfError::Invalid(format!("ollama /api/tags: {e}")))?;
        let mut wanted = vec![self.model.clone()];
        if !self.model.contains(':') {
            wanted.push(format!("{}:latest", self.model));
        }
        let mut served = Vec::new();
        for m in body
            .get("models")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let name = m.get("name").and_then(Value::as_str).unwrap_or("");
            let model = m.get("model").and_then(Value::as_str).unwrap_or("");
            served.push(name.to_string());
            if wanted.iter().any(|w| w == name || w == model) {
                if let Some(d) = m.get("digest").and_then(Value::as_str) {
                    return Ok(d.to_string());
                }
            }
        }
        Err(HfError::Refused(format!(
            "model {} is not served; served: {served:?}",
            self.model
        )))
    }

    /// Embed a batch; `Err(Overflow)` on a 400 whose body names the context length.
    pub fn embed(
        &self,
        texts: &[String],
        retries: u32,
    ) -> Result<Result<Vec<Vec<f64>>, ContextOverflow>, HfError> {
        let mut attempt = 0;
        loop {
            let sent = self
                .agent
                .post(format!("{}/api/embed", self.base_url))
                .send_json(serde_json::json!({"model": self.model, "input": texts}));
            match sent {
                Ok(mut response) => {
                    let status = response.status().as_u16();
                    let text = response.body_mut().read_to_string().unwrap_or_default();
                    if status == 400 && text.contains("context length") {
                        return Ok(Err(ContextOverflow));
                    }
                    if status >= 400 {
                        return Err(HfError::Invalid(format!(
                            "ollama /api/embed: HTTP {status}: {text}"
                        )));
                    }
                    let body: Value = serde_json::from_str(&text)
                        .map_err(|e| HfError::Invalid(format!("ollama /api/embed: {e}")))?;
                    let vectors: Vec<Vec<f64>> = serde_json::from_value(body["embeddings"].clone())
                        .map_err(|e| HfError::Invalid(format!("ollama /api/embed: {e}")))?;
                    if vectors.len() != texts.len() {
                        return Err(HfError::Invalid(format!(
                            "ollama returned {} embeddings for {} texts",
                            vectors.len(),
                            texts.len()
                        )));
                    }
                    return Ok(Ok(vectors));
                }
                Err(e) => {
                    attempt += 1;
                    if attempt > retries {
                        return Err(HfError::Invalid(format!(
                            "ollama unreachable after {retries} retries: {e}"
                        )));
                    }
                    std::thread::sleep(std::time::Duration::from_secs_f64(1.5 * attempt as f64));
                }
            }
        }
    }
}

/// Python's `text[:n]` — by code point.
pub fn truncate_chars(text: &str, n: usize) -> &str {
    match text.char_indices().nth(n) {
        Some((i, _)) => &text[..i],
        None => text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_matches_python_edge_cases() {
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 2.0]), 0.0);
        assert!((cosine(&[1.0, 0.0], &[1.0, 1.0]) - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn truncation_is_by_code_point() {
        assert_eq!(truncate_chars("héllo", 2), "hé");
        assert_eq!(truncate_chars("hi", 5), "hi");
    }
}
