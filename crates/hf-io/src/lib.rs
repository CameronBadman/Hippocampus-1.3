//! Split directories in the foundation's v5 layout
//! (`hippocampus_foundation.read_run.io_v5`), read and written with the same
//! framing, digests and file modes, so a Rust-written split validates under
//! the Python `validate_real_split_artifacts_v5` and a Python-written split
//! reads here.
//!
//! Layout: `visible.jsonl.gz` (0644) and `hidden.jsonl.gz` (0600), one
//! canonical-JSON line per episode (`{"episode_id": …, "visible": …}` /
//! `{"episode_id": …, "hidden": …}`) plus `\n`, gzip level 6 with mtime 0 and no
//! filename; `manifest.public.json` (0644), `manifest.private.json` (0600),
//! `graph.manifest.json` (0644) pretty-printed; the directory itself 0700,
//! written under a temporary name and renamed into place. Sidecars written
//! beside by the split writer: `texts.jsonl`, `sampling.json`, `nodes.txt`.
//!
//! Records are never re-canonicalised at load: the line bytes on disk are the
//! canonical bytes, and a reader that keeps them keeps the digest material.

use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

use hf_core::canonical::{canonical_bytes, canonical_sha256, sha256_file};
use hf_core::files::{create_exclusive, dump_pretty, MODE_HIDDEN, MODE_SPLIT_DIR, MODE_VISIBLE};
use hf_core::{refuse_holdout, HfError};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub const SCHEMA_VERSION_V5: &str = "5.0.0";
pub const STAGES: [&str; 2] = ["stage0_known_target", "stage1_described_target"];
pub const SPLITS: [&str; 3] = ["train", "screen", "test-fixture"];
pub const SPLIT_MANIFEST_KIND: &str = "real_walk_split_manifest_v5";
pub const VISIBLE_KIND: &str = "real_walk_episode_visible";
pub const HIDDEN_KIND: &str = "real_walk_episode_hidden";
/// The largest record line the reader accepts, newline included.
pub const MAX_JSON_LINE_BYTES: usize = 16 * 1024 * 1024;

const VISIBLE_ALLOWLIST: [&str; 11] = [
    "schema_version",
    "record_kind",
    "family",
    "stage",
    "start_node",
    "target_node",
    "query",
    "subgraph_size",
    "removal_level",
    "nodes",
    "edges",
];
const NODE_ALLOWLIST: [&str; 2] = ["node", "text"];
const EDGE_ALLOWLIST: [&str; 4] = ["edge_id", "relation", "source", "target"]; // sorted
const LEAK_KEYS: [&str; 5] = [
    "path_set",
    "removal_set",
    "distances",
    "target_set",
    "label",
];

/// A visible node record.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VisibleNode {
    pub node: String,
    pub text: String,
}

/// A visible edge record; `edge_id` order is the walk's child order.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VisibleEdge {
    pub edge_id: u32,
    pub source: String,
    pub target: String,
    pub relation: Option<String>,
}

/// The model-visible payload, typed; the allow-list is enforced on the raw value.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Visible {
    pub schema_version: String,
    pub record_kind: String,
    pub family: String,
    pub stage: String,
    pub start_node: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_node: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query: Option<Value>,
    pub subgraph_size: u32,
    pub removal_level: u32,
    pub nodes: Vec<VisibleNode>,
    pub edges: Vec<VisibleEdge>,
}

/// The hidden payload, typed where the engine reads it; anything else is kept in `extra`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Hidden {
    pub schema_version: String,
    pub record_kind: String,
    pub family: String,
    pub stage: String,
    pub split: String,
    pub index: u64,
    pub start_node: String,
    pub target_set: Vec<String>,
    pub target_distance: u32,
    pub cost_bound: u32,
    pub path_set: Vec<Vec<String>>,
    pub surviving_paths: Vec<Vec<String>>,
    pub removal_set: Vec<Vec<String>>,
    pub removed_count: u32,
    pub unremovable_count: u32,
    pub nodes_on_surviving_path: Vec<String>,
    pub distance_to_target: BTreeMap<String, u32>,
    pub sampler: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub greedy_overshoot: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub removal_recipe: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl Hidden {
    /// `removal_recipe` with the readers' default for splits written before amendment 6.
    pub fn removal_recipe(&self) -> &str {
        self.removal_recipe.as_deref().unwrap_or("cheapest-first")
    }
}

/// One episode: id, the typed payloads, and the canonical line bytes they came from.
#[derive(Clone, Debug)]
pub struct RealEpisode {
    pub episode_id: String,
    pub visible: Visible,
    pub hidden: Hidden,
}

/// Python's `validate_visible_v5` on the raw value.
pub fn validate_visible(value: &Value) -> Result<(), HfError> {
    let obj = value
        .as_object()
        .ok_or_else(|| HfError::BandH("visible payload is not an object".into()))?;
    let get = |k: &str| obj.get(k).and_then(Value::as_str);
    if get("schema_version") != Some(SCHEMA_VERSION_V5) {
        return Err(HfError::BandH(
            "visible payload schema version is not 5.0.0".into(),
        ));
    }
    if get("record_kind") != Some(VISIBLE_KIND) {
        return Err(HfError::BandH(
            "visible payload record kind is wrong".into(),
        ));
    }
    let stage = get("stage").unwrap_or("");
    if !STAGES.contains(&stage) {
        return Err(HfError::BandH(format!("unknown stage {stage:?}")));
    }
    let has_target = obj.contains_key("target_node");
    let has_query = obj.contains_key("query");
    if has_target == has_query {
        return Err(HfError::BandH(
            "visible payload must carry exactly one of target_node / query".into(),
        ));
    }
    if stage == STAGES[0] && !has_target {
        return Err(HfError::BandH(
            "stage 0 visible payload must carry target_node".into(),
        ));
    }
    if stage == STAGES[1] && !has_query {
        return Err(HfError::BandH(
            "stage 1 visible payload must carry query".into(),
        ));
    }
    for key in obj.keys() {
        if !VISIBLE_ALLOWLIST.contains(&key.as_str()) {
            return Err(HfError::BandH(format!(
                "visible payload carries a disallowed key {key:?}"
            )));
        }
    }
    let nodes = obj
        .get("nodes")
        .and_then(Value::as_array)
        .ok_or_else(|| HfError::BandH("nodes missing".into()))?;
    for node in nodes {
        let keys: Vec<&str> = node
            .as_object()
            .map(|o| o.keys().map(String::as_str).collect())
            .unwrap_or_default();
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        if sorted != NODE_ALLOWLIST {
            return Err(HfError::BandH(format!(
                "node record keys {keys:?} are not exactly {NODE_ALLOWLIST:?}"
            )));
        }
    }
    let edges = obj
        .get("edges")
        .and_then(Value::as_array)
        .ok_or_else(|| HfError::BandH("edges missing".into()))?;
    for edge in edges {
        let keys: Vec<&str> = edge
            .as_object()
            .map(|o| o.keys().map(String::as_str).collect())
            .unwrap_or_default();
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        if sorted != EDGE_ALLOWLIST {
            return Err(HfError::BandH(format!(
                "edge record keys {keys:?} are not exactly {EDGE_ALLOWLIST:?}"
            )));
        }
    }
    for leak in LEAK_KEYS {
        if obj.contains_key(leak) {
            return Err(HfError::BandH(format!("visible payload leaks {leak}")));
        }
    }
    Ok(())
}

/// The three manifests of a split directory, validated as Python validates them.
#[derive(Clone, Debug)]
pub struct SplitArtifacts {
    pub public: Value,
    pub private: Value,
    pub graph: Value,
}

fn read_json(path: &Path) -> Result<Value, HfError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))
}

/// `validate_real_split_artifacts_v5`: record kinds, `training_authorized`
/// exactly false, the graph manifest's canonical digest, and both streams'
/// container bytes and digests.
pub fn validate_split_artifacts(root: &Path) -> Result<SplitArtifacts, HfError> {
    refuse_holdout(root)?;
    let public = read_json(&root.join("manifest.public.json"))?;
    let private = read_json(&root.join("manifest.private.json"))?;
    let graph = read_json(&root.join("graph.manifest.json"))?;
    for (name, manifest) in [("public", &public), ("private", &private)] {
        if manifest.get("record_kind").and_then(Value::as_str) != Some(SPLIT_MANIFEST_KIND) {
            return Err(HfError::BandH(format!(
                "{name} manifest record kind is wrong"
            )));
        }
        if manifest.get("training_authorized") != Some(&Value::Bool(false)) {
            return Err(HfError::BandH(format!(
                "{name} manifest does not set training_authorized false"
            )));
        }
    }
    if public
        .get("family")
        .and_then(Value::as_str)
        .unwrap_or("")
        .is_empty()
    {
        return Err(HfError::BandH("public manifest has no family".into()));
    }
    let want = public
        .get("graph_manifest_sha256")
        .and_then(Value::as_str)
        .unwrap_or("");
    if want.is_empty() {
        return Err(HfError::BandH(
            "public manifest has no graph manifest digest".into(),
        ));
    }
    if canonical_sha256(&graph)? != want {
        return Err(HfError::BandH(
            "graph manifest digest does not match the split's record".into(),
        ));
    }
    for (file, manifest, bytes_key, sha_key) in [
        (
            "visible.jsonl.gz",
            &public,
            "visible_bytes",
            "visible_sha256",
        ),
        ("hidden.jsonl.gz", &private, "hidden_bytes", "hidden_sha256"),
    ] {
        let (size, digest) =
            sha256_file(&root.join(file)).map_err(|e| HfError::Invalid(format!("{file}: {e}")))?;
        if manifest.get(bytes_key).and_then(Value::as_u64) != Some(size)
            || manifest.get(sha_key).and_then(Value::as_str) != Some(digest.as_str())
        {
            return Err(HfError::BandH(format!(
                "{file} does not match its manifest digest"
            )));
        }
    }
    Ok(SplitArtifacts {
        public,
        private,
        graph,
    })
}

fn gz_lines(path: &Path) -> Result<impl Iterator<Item = Result<Vec<u8>, HfError>>, HfError> {
    let file = std::fs::File::open(path)
        .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))?;
    let reader = BufReader::with_capacity(1 << 20, flate2::read::MultiGzDecoder::new(file));
    let name = path.display().to_string();
    Ok(reader.split(b'\n').map(move |line| {
        let line = line.map_err(|e| HfError::Invalid(format!("{name}: {e}")))?;
        if line.len() + 1 > MAX_JSON_LINE_BYTES {
            return Err(HfError::BandH(format!(
                "{name}: a record exceeds {MAX_JSON_LINE_BYTES} bytes"
            )));
        }
        Ok(line)
    }))
}

#[derive(Deserialize)]
struct VisibleLine {
    episode_id: String,
    visible: Value,
}

#[derive(Deserialize)]
struct HiddenLine {
    episode_id: String,
    hidden: Hidden,
}

/// `read_real_split_v5`: validate, materialise the hidden stream, then stream
/// the visible one, validating every visible payload. The callback receives
/// each episode with its canonical visible line bytes (without the newline).
pub fn for_each_episode(
    root: &Path,
    mut f: impl FnMut(RealEpisode, &[u8]) -> Result<(), HfError>,
) -> Result<SplitArtifacts, HfError> {
    let artifacts = validate_split_artifacts(root)?;
    let mut hidden_by_id: HashMap<String, Hidden> = HashMap::new();
    for line in gz_lines(&root.join("hidden.jsonl.gz"))? {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let record: HiddenLine = serde_json::from_slice(&line)
            .map_err(|e| HfError::BandH(format!("hidden record: {e}")))?;
        hidden_by_id.insert(record.episode_id, record.hidden);
    }
    let mut count = 0usize;
    for line in gz_lines(&root.join("visible.jsonl.gz"))? {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let record: VisibleLine = serde_json::from_slice(&line)
            .map_err(|e| HfError::BandH(format!("visible record: {e}")))?;
        validate_visible(&record.visible)?;
        let hidden = hidden_by_id.remove(&record.episode_id).ok_or_else(|| {
            HfError::BandH(format!(
                "episode {} has no hidden record",
                record.episode_id
            ))
        })?;
        let visible: Visible = serde_json::from_value(record.visible)
            .map_err(|e| HfError::BandH(format!("visible payload: {e}")))?;
        f(
            RealEpisode {
                episode_id: record.episode_id,
                visible,
                hidden,
            },
            &line,
        )?;
        count += 1;
    }
    if count == 0 {
        return Err(HfError::BandH("dataset stream is empty".into()));
    }
    Ok(artifacts)
}

/// Every episode of a split, materialised.
pub fn read_split(root: &Path) -> Result<(Vec<RealEpisode>, SplitArtifacts), HfError> {
    let mut episodes = Vec::new();
    let artifacts = for_each_episode(root, |episode, _| {
        episodes.push(episode);
        Ok(())
    })?;
    Ok((episodes, artifacts))
}

/// One episode to write: the id and both payloads as JSON values (the writer
/// canonicalises them; the visible one is validated first).
#[derive(Clone, Debug)]
pub struct EpisodeOut {
    pub episode_id: String,
    pub visible: Value,
    pub hidden: Value,
}

struct GzStream {
    encoder: flate2::write::GzEncoder<std::fs::File>,
}

impl GzStream {
    fn create(path: &Path, mode: u32) -> Result<Self, HfError> {
        let file = create_exclusive(path, mode)
            .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))?;
        let encoder = flate2::GzBuilder::new()
            .mtime(0)
            .write(file, flate2::Compression::new(6));
        Ok(Self { encoder })
    }

    fn write_record(&mut self, bytes: &[u8]) -> Result<(), HfError> {
        self.encoder
            .write_all(bytes)
            .and_then(|_| self.encoder.write_all(b"\n"))
            .map_err(|e| HfError::Invalid(e.to_string()))
    }

    fn finish(self) -> Result<(), HfError> {
        let mut file = self
            .encoder
            .finish()
            .map_err(|e| HfError::Invalid(e.to_string()))?;
        file.flush()
            .and_then(|_| file.sync_all())
            .map_err(|e| HfError::Invalid(e.to_string()))
    }
}

fn write_json_exclusive(path: &Path, mode: u32, value: &Value) -> Result<(), HfError> {
    let mut file = create_exclusive(path, mode)
        .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))?;
    file.write_all(dump_pretty(value).as_bytes())
        .and_then(|_| file.sync_all())
        .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))
}

fn fsync_dir(path: &Path) -> Result<(), HfError> {
    std::fs::File::open(path)
        .and_then(|d| d.sync_all())
        .map_err(|e| HfError::Invalid(format!("fsync {}: {e}", path.display())))
}

/// `write_real_split_v5`: atomically write both streams and the three manifests.
/// Returns `(public, private)` manifests.
pub fn write_split(
    family: &str,
    stage: &str,
    split: &str,
    destination: &Path,
    episodes: impl IntoIterator<Item = Result<EpisodeOut, HfError>>,
    graph_manifest: &Value,
    sampler: &Value,
) -> Result<(Value, Value), HfError> {
    if !STAGES.contains(&stage) {
        return Err(HfError::Invalid(format!("unknown stage {stage:?}")));
    }
    if !SPLITS.contains(&split) {
        return Err(HfError::Invalid(format!("unknown split {split:?}")));
    }
    refuse_holdout(destination)?;
    if destination.exists() || destination.symlink_metadata().is_ok() {
        return Err(HfError::Refused(format!(
            "{} already exists",
            destination.display()
        )));
    }
    let parent = destination
        .parent()
        .ok_or_else(|| HfError::Invalid("destination has no parent".into()))?;
    std::fs::create_dir_all(parent)
        .map_err(|e| HfError::Invalid(format!("{}: {e}", parent.display())))?;
    let temporary: PathBuf = parent.join(format!(
        ".tmp-{}-{}",
        destination
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default(),
        std::process::id()
    ));
    {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new()
            .mode(MODE_SPLIT_DIR)
            .create(&temporary)
            .map_err(|e| HfError::Invalid(format!("{}: {e}", temporary.display())))?;
    }
    let result = write_split_into(
        family,
        stage,
        split,
        &temporary,
        episodes,
        graph_manifest,
        sampler,
    );
    match result {
        Ok(manifests) => {
            fsync_dir(&temporary)?;
            std::fs::rename(&temporary, destination).map_err(|e| {
                HfError::Invalid(format!("rename to {}: {e}", destination.display()))
            })?;
            fsync_dir(parent)?;
            Ok(manifests)
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&temporary);
            Err(e)
        }
    }
}

fn write_split_into(
    family: &str,
    stage: &str,
    split: &str,
    dir: &Path,
    episodes: impl IntoIterator<Item = Result<EpisodeOut, HfError>>,
    graph_manifest: &Value,
    sampler: &Value,
) -> Result<(Value, Value), HfError> {
    let visible_path = dir.join("visible.jsonl.gz");
    let hidden_path = dir.join("hidden.jsonl.gz");
    let mut visible = GzStream::create(&visible_path, MODE_VISIBLE)?;
    let mut hidden = GzStream::create(&hidden_path, MODE_HIDDEN)?;
    let mut count = 0u64;
    let mut removal_levels: BTreeMap<u64, u64> = BTreeMap::new();
    for episode in episodes {
        let episode = episode?;
        validate_visible(&episode.visible)?;
        let level = episode
            .visible
            .get("removal_level")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        *removal_levels.entry(level).or_default() += 1;
        let v = serde_json::json!({"episode_id": episode.episode_id, "visible": episode.visible});
        let h = serde_json::json!({"episode_id": episode.episode_id, "hidden": episode.hidden});
        visible.write_record(&canonical_bytes(&v)?)?;
        hidden.write_record(&canonical_bytes(&h)?)?;
        count += 1;
    }
    visible.finish()?;
    hidden.finish()?;
    if count == 0 {
        return Err(HfError::Invalid("split produced no episodes".into()));
    }
    let (visible_bytes, visible_sha256) =
        sha256_file(&visible_path).map_err(|e| HfError::Invalid(e.to_string()))?;
    let (hidden_bytes, hidden_sha256) =
        sha256_file(&hidden_path).map_err(|e| HfError::Invalid(e.to_string()))?;
    let levels: Map<String, Value> = removal_levels
        .into_iter()
        .map(|(k, v)| (k.to_string(), Value::from(v)))
        .collect();
    let mut common = Map::new();
    common.insert("schema_version".into(), SCHEMA_VERSION_V5.into());
    common.insert("record_kind".into(), SPLIT_MANIFEST_KIND.into());
    common.insert("family".into(), family.into());
    common.insert("stage".into(), stage.into());
    common.insert("split".into(), split.into());
    common.insert("episode_count".into(), count.into());
    common.insert("removal_levels".into(), Value::Object(levels));
    common.insert(
        "graph_manifest_sha256".into(),
        canonical_sha256(graph_manifest)?.into(),
    );
    common.insert("sampler".into(), sampler.clone());
    common.insert("visible_bytes".into(), visible_bytes.into());
    common.insert("visible_sha256".into(), visible_sha256.into());
    common.insert("training_authorized".into(), Value::Bool(false));
    let mut public = common.clone();
    public.insert("disclosure".into(), "model_visible_only".into());
    let mut private = common;
    private.insert("disclosure".into(), "private_labels_and_oracle".into());
    private.insert("hidden_bytes".into(), hidden_bytes.into());
    private.insert("hidden_sha256".into(), hidden_sha256.into());
    let (public, private) = (Value::Object(public), Value::Object(private));
    write_json_exclusive(&dir.join("manifest.public.json"), MODE_VISIBLE, &public)?;
    write_json_exclusive(&dir.join("manifest.private.json"), MODE_HIDDEN, &private)?;
    write_json_exclusive(
        &dir.join("graph.manifest.json"),
        MODE_VISIBLE,
        graph_manifest,
    )?;
    Ok((public, private))
}

/// The split writer's sidecars: `texts.jsonl` (one `{"node", "text"}` line per
/// node in the given order), `sampling.json` and `nodes.txt` (sorted, one per line).
pub fn write_sidecars(
    dir: &Path,
    texts: impl IntoIterator<Item = (String, String)>,
    sampling: &Value,
    nodes: &[String],
) -> Result<usize, HfError> {
    let mut file = std::fs::File::create(dir.join("texts.jsonl"))
        .map_err(|e| HfError::Invalid(e.to_string()))?;
    let mut written = 0;
    for (node, text) in texts {
        let line = hf_core::python_json_compact(&serde_json::json!({"node": node, "text": text}));
        file.write_all(line.as_bytes())
            .and_then(|_| file.write_all(b"\n"))
            .map_err(|e| HfError::Invalid(e.to_string()))?;
        written += 1;
    }
    file.flush().map_err(|e| HfError::Invalid(e.to_string()))?;
    // sampling.json is json.dumps(indent=2, sort_keys=True) without a trailing newline
    let pretty = hf_core::files::python_json_pretty(sampling);
    std::fs::write(dir.join("sampling.json"), pretty)
        .map_err(|e| HfError::Invalid(e.to_string()))?;
    let mut sorted: Vec<&String> = nodes.iter().collect();
    sorted.sort_unstable();
    let mut text = sorted
        .iter()
        .map(|n| n.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    text.push('\n');
    std::fs::write(dir.join("nodes.txt"), text).map_err(|e| HfError::Invalid(e.to_string()))?;
    Ok(written)
}

/// The `texts.jsonl` sidecar, node → text.
pub fn read_texts(dir: &Path) -> Result<HashMap<String, String>, HfError> {
    let path = dir.join("texts.jsonl");
    let file = std::fs::File::open(&path)
        .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))?;
    let mut out = HashMap::new();
    for line in BufReader::new(file).lines() {
        let line = line.map_err(|e| HfError::Invalid(e.to_string()))?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(&line)
            .map_err(|e| HfError::Invalid(format!("texts.jsonl: {e}")))?;
        let node = v
            .get("node")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let text = v
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        out.insert(node, text);
    }
    Ok(out)
}

/// Read a whole file through gzip if named `.gz` (for sidecar-sized inputs).
pub fn read_maybe_gz(path: &Path) -> Result<Vec<u8>, HfError> {
    let file = std::fs::File::open(path)
        .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))?;
    let mut out = Vec::new();
    if path.to_string_lossy().ends_with(".gz") {
        flate2::read::MultiGzDecoder::new(file).read_to_end(&mut out)
    } else {
        std::io::BufReader::new(file).read_to_end(&mut out)
    }
    .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))?;
    Ok(out)
}
