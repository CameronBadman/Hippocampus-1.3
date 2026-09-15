//! The real-walk model of `model_v5.build_read_model_v5` on libtorch, with
//! the parameter names of the Python `state_dict` so a Python checkpoint
//! exported to safetensors loads here, and the redesign's relational feature
//! set behind the same architecture (a learned query token in place of the
//! query encoder). Attention is written out — `tch` has no
//! `MultiheadAttention` module — with PyTorch's semantics: the per-head
//! additive pair bias, a float key-padding mask of `-inf`, `nan_to_num` on
//! fully masked rows. The optimiser is a hand-written AdamW whose moments are
//! saved and loaded, so a resumed run is exact.

use std::collections::HashMap;
use std::path::Path;

use hf_core::HfError;
use hf_walk::{
    DecisionBatch, DecisionItem, FeatureSet, RawV5, RelationalV6, Scored, Scorer, WalkResult,
    STOP_DIM,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tch::nn::{self, Module};
use tch::{Device, Kind, Reduction, Tensor};

/// The `model` block of a training config, plus the engine's feature-set switch.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelConfig {
    pub hidden_dimension: i64,
    pub self_attention_heads: i64,
    pub feedforward_multiplier: i64,
    pub score_hidden_dimension: i64,
    pub coverage_hidden_dimension: i64,
    pub traversal_blocks: i64,
    #[serde(default)]
    pub dropout: f64,
    pub embedding_dimension: i64,
    #[serde(default)]
    pub greedy_prior: bool,
    #[serde(default = "default_prior_scale")]
    pub greedy_prior_scale: f64,
    /// `"raw-v5"` (the Python layout) or `"relational-v6"` (the redesign).
    #[serde(default = "default_feature_set")]
    pub feature_set: String,
    #[serde(default)]
    pub zero_embedding_blocks: Vec<String>,
    #[serde(default = "default_score_chunk")]
    pub score_chunk: usize,
}

fn default_prior_scale() -> f64 {
    10.0
}
fn default_feature_set() -> String {
    "raw-v5".into()
}
fn default_score_chunk() -> usize {
    32
}

impl ModelConfig {
    /// From a training config's `model` object (the runner injects `embedding_dimension`).
    pub fn from_value(model: &serde_json::Value, edim: i64) -> Result<Self, HfError> {
        let mut v = model.clone();
        if let Some(obj) = v.as_object_mut() {
            obj.retain(|k, _| k != "note" && k != "stated_capacity");
        }
        v["embedding_dimension"] = edim.into();
        serde_json::from_value(v).map_err(|e| HfError::Invalid(format!("model config: {e}")))
    }

    pub fn features(&self) -> Result<Box<dyn FeatureSet>, HfError> {
        match self.feature_set.as_str() {
            "raw-v5" => Ok(Box::new(RawV5)),
            "relational-v6" => Ok(Box::new(RelationalV6)),
            other => Err(HfError::Invalid(format!("unknown feature set {other:?}"))),
        }
    }
}

struct Attention {
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    out_proj: nn::Linear,
    heads: i64,
    hidden: i64,
}

impl Attention {
    fn new(p: &nn::Path, hidden: i64, heads: i64) -> Self {
        // PyTorch's MultiheadAttention: xavier_uniform in_proj, zero biases
        let bound = (6.0 / (4.0 * hidden as f64)).sqrt();
        let in_proj_weight = p.var(
            "in_proj_weight",
            &[3 * hidden, hidden],
            nn::Init::Uniform {
                lo: -bound,
                up: bound,
            },
        );
        let in_proj_bias = p.var("in_proj_bias", &[3 * hidden], nn::Init::Const(0.0));
        let out_proj = nn::linear(
            p / "out_proj",
            hidden,
            hidden,
            nn::LinearConfig {
                bs_init: Some(nn::Init::Const(0.0)),
                ..Default::default()
            },
        );
        Self {
            in_proj_weight,
            in_proj_bias,
            out_proj,
            heads,
            hidden,
        }
    }

    /// `query [B, F, H]`, `kv [B, C, H]`, optional additive `bias [B, heads, F, C]`,
    /// optional `key_padding [B, C]` (true = ignore). Out-of-place adds keep
    /// autograd's view of the graph simple.
    #[allow(clippy::assign_op_pattern)]
    fn forward(
        &self,
        query: &Tensor,
        kv: &Tensor,
        bias: Option<&Tensor>,
        key_padding: Option<&Tensor>,
    ) -> Tensor {
        let (b, f, _) = query.size3().unwrap();
        let c = kv.size()[1];
        let h = self.hidden;
        let hd = h / self.heads;
        let w = self.in_proj_weight.split(h, 0);
        let bs = self.in_proj_bias.split(h, 0);
        let q = query.matmul(&w[0].transpose(0, 1)) + &bs[0];
        let k = kv.matmul(&w[1].transpose(0, 1)) + &bs[1];
        let v = kv.matmul(&w[2].transpose(0, 1)) + &bs[2];
        let q = q.view([b, f, self.heads, hd]).transpose(1, 2);
        let k = k.view([b, c, self.heads, hd]).transpose(1, 2);
        let v = v.view([b, c, self.heads, hd]).transpose(1, 2);
        let mut scores = q.matmul(&k.transpose(2, 3)) / (hd as f64).sqrt(); // [B, heads, F, C]
        if let Some(bias) = bias {
            scores = scores + bias;
        }
        if let Some(pad) = key_padding {
            let mask = Tensor::zeros_like(pad)
                .to_kind(Kind::Float)
                .masked_fill(pad, f64::NEG_INFINITY);
            scores = scores + mask.view([b, 1, 1, c]);
        }
        let attn = scores.softmax(-1, Kind::Float);
        let out = attn.matmul(&v).transpose(1, 2).contiguous().view([b, f, h]);
        self.out_proj.forward(&out)
    }
}

struct Block {
    norm_context: nn::LayerNorm,
    context_attention: Attention,
    context_bias: nn::Linear,
    norm_query: nn::LayerNorm,
    query_attention: Attention,
    norm_ff: nn::LayerNorm,
    ff0: nn::Linear,
    ff2: nn::Linear,
}

impl Block {
    fn new(p: &nn::Path, hidden: i64, heads: i64, multiplier: i64, pair_dim: i64) -> Self {
        let no_bias = nn::LinearConfig {
            bias: false,
            ..Default::default()
        };
        Self {
            norm_context: nn::layer_norm(p / "norm_context", vec![hidden], Default::default()),
            context_attention: Attention::new(&(p / "context_attention"), hidden, heads),
            context_bias: nn::linear(p / "context_bias", pair_dim, heads, no_bias),
            norm_query: nn::layer_norm(p / "norm_query", vec![hidden], Default::default()),
            query_attention: Attention::new(&(p / "query_attention"), hidden, heads),
            norm_ff: nn::layer_norm(p / "norm_ff", vec![hidden], Default::default()),
            ff0: nn::linear(
                p / "feedforward" / "0",
                hidden,
                multiplier * hidden,
                Default::default(),
            ),
            ff2: nn::linear(
                p / "feedforward" / "2",
                multiplier * hidden,
                hidden,
                Default::default(),
            ),
        }
    }

    fn forward(
        &self,
        cand: &Tensor,
        ctx: &Tensor,
        query: &Tensor,
        pair: &Tensor,
        ctx_mask: &Tensor,
    ) -> Tensor {
        let normed = self.norm_context.forward(cand);
        let bias = self.context_bias.forward(pair).permute([0, 3, 1, 2]); // [B, heads, F, C]
        let attended = self
            .context_attention
            .forward(&normed, ctx, Some(&bias), Some(ctx_mask));
        let cand = cand + attended.nan_to_num(0.0, None, None);
        let normed = self.norm_query.forward(&cand);
        let attended = self.query_attention.forward(&normed, query, None, None);
        let cand = cand + attended.nan_to_num(0.0, None, None);
        let ff = self
            .ff2
            .forward(&self.ff0.forward(&self.norm_ff.forward(&cand)).gelu("none"));
        cand + ff
    }
}

/// `RealWalkModel`.
pub struct Model {
    pub vs: nn::VarStore,
    pub config: ModelConfig,
    candidate_encoder: nn::Linear,
    candidate_norm: nn::LayerNorm,
    context_encoder: nn::Linear,
    context_norm: nn::LayerNorm,
    query_encoder: Option<nn::Linear>,
    query_token: Option<Tensor>,
    blocks: Vec<Block>,
    score0: nn::Linear,
    score2: nn::Linear,
    greedy_tau: Option<Tensor>,
    stop0: nn::Linear,
    stop2: nn::Linear,
    _device_probe: Tensor,
    cdim: i64,
    ctx_dim: i64,
    pair_dim: i64,
    cosine_column: i64,
    edim: i64,
    /// Runtime override of `config.zero_embedding_blocks` (the ablation flag).
    pub zero_blocks: Vec<String>,
}

impl Model {
    pub fn new(config: ModelConfig, device: Device) -> Result<Self, HfError> {
        let features = config.features()?;
        let edim = config.embedding_dimension as usize;
        let cdim = features.candidate_dim(edim) as i64;
        let ctx_dim = features.context_dim(edim) as i64;
        let pair_dim = features.pair_dim() as i64;
        let cosine_column = features.cosine_column(edim) as i64;
        let hidden = config.hidden_dimension;
        let vs = nn::VarStore::new(device);
        let p = vs.root();
        let candidate_encoder =
            nn::linear(&p / "candidate_encoder", cdim, hidden, Default::default());
        let candidate_norm =
            nn::layer_norm(&p / "candidate_norm", vec![hidden], Default::default());
        let context_encoder =
            nn::linear(&p / "context_encoder", ctx_dim, hidden, Default::default());
        let context_norm = nn::layer_norm(&p / "context_norm", vec![hidden], Default::default());
        let (query_encoder, query_token) = match features.query_dim(edim) {
            Some(qdim) => (
                Some(nn::linear(
                    &p / "query_encoder",
                    qdim as i64,
                    hidden,
                    Default::default(),
                )),
                None,
            ),
            None => (
                None,
                Some(p.var(
                    "query_token",
                    &[1, 1, hidden],
                    nn::Init::Randn {
                        mean: 0.0,
                        stdev: 0.02,
                    },
                )),
            ),
        };
        let blocks = (0..config.traversal_blocks)
            .map(|i| {
                Block::new(
                    &(&p / "blocks" / i),
                    hidden,
                    config.self_attention_heads,
                    config.feedforward_multiplier,
                    pair_dim,
                )
            })
            .collect();
        let score0 = nn::linear(
            &p / "score_head" / "0",
            hidden,
            config.score_hidden_dimension,
            Default::default(),
        );
        let score2 = nn::linear(
            &p / "score_head" / "2",
            config.score_hidden_dimension,
            1,
            Default::default(),
        );
        let greedy_tau = if config.greedy_prior {
            // the residual is exactly zero at initialisation: the untrained walk is greedy
            tch::no_grad(|| {
                let mut ws = score2.ws.shallow_clone();
                let _ = ws.zero_();
                if let Some(bs) = &score2.bs {
                    let mut bs = bs.shallow_clone();
                    let _ = bs.zero_();
                }
            });
            Some(p.var(
                "greedy_tau",
                &[],
                nn::Init::Const(config.greedy_prior_scale),
            ))
        } else {
            None
        };
        let stop0 = nn::linear(
            &p / "stop_head" / "0",
            STOP_DIM as i64,
            config.coverage_hidden_dimension,
            Default::default(),
        );
        let stop2 = nn::linear(
            &p / "stop_head" / "2",
            config.coverage_hidden_dimension,
            1,
            Default::default(),
        );
        let device_probe = p.zeros_no_train("_device_probe", &[1]);
        Ok(Self {
            zero_blocks: config.zero_embedding_blocks.clone(),
            vs,
            config,
            candidate_encoder,
            candidate_norm,
            context_encoder,
            context_norm,
            query_encoder,
            query_token,
            blocks,
            score0,
            score2,
            greedy_tau,
            stop0,
            stop2,
            _device_probe: device_probe,
            cdim,
            ctx_dim,
            pair_dim,
            cosine_column,
            edim: edim as i64,
        })
    }

    pub fn device(&self) -> Device {
        self.vs.device()
    }

    pub fn features(&self) -> Box<dyn FeatureSet> {
        self.config.features().expect("validated at construction")
    }

    /// The trained `greedy_tau`, when the prior is on.
    pub fn greedy_tau(&self) -> Option<f64> {
        self.greedy_tau.as_ref().and_then(|t| f64::try_from(t).ok())
    }

    /// `trainable_parameter_count_v5`.
    pub fn trainable_parameter_count(&self) -> i64 {
        self.vs
            .trainable_variables()
            .iter()
            .map(|t| t.numel() as i64)
            .sum()
    }

    /// The runner's `_state_digest` over every state tensor (buffers included),
    /// sorted by name: name, `str(dtype)`, `str(tuple(shape))`, raw bytes.
    pub fn state_digest(&self, exclude: &[&str]) -> String {
        let mut names: Vec<(String, Tensor)> = self.vs.variables().into_iter().collect();
        names.sort_by(|a, b| a.0.cmp(&b.0));
        let mut h = Sha256::new();
        for (name, t) in names {
            if exclude.contains(&name.as_str()) {
                continue;
            }
            h.update(name.as_bytes());
            h.update(kind_name(t.kind()).as_bytes());
            let shape: Vec<String> = t.size().iter().map(|d| d.to_string()).collect();
            let shape_repr = match shape.len() {
                0 => "()".to_string(),
                1 => format!("({},)", shape[0]),
                _ => format!("({})", shape.join(", ")),
            };
            h.update(shape_repr.as_bytes());
            let t = t.detach().to_device(Device::Cpu).contiguous();
            let n = t.numel();
            match t.kind() {
                Kind::Float => {
                    let mut buf = vec![0f32; n];
                    t.copy_data(&mut buf, n);
                    for x in buf {
                        h.update(x.to_le_bytes());
                    }
                }
                other => {
                    let mut buf = vec![0u8; n * kind_bytes(other)];
                    t.copy_data_u8(&mut buf, n);
                    h.update(&buf);
                }
            }
        }
        format!("sha256:{}", hex::encode(h.finalize()))
    }

    /// `score_decisions`: one padded batch → scores `[N, F]` and residuals `[N, F]`
    /// (`-inf` on padding); the residual equals the score without a prior.
    pub fn score_decisions(&self, items: &[&DecisionItem]) -> (Tensor, Tensor) {
        let n = items.len() as i64;
        let f = items
            .iter()
            .map(|it| it.frontier_len)
            .max()
            .unwrap_or(0)
            .max(1) as i64;
        let c = items
            .iter()
            .map(|it| it.context_len)
            .max()
            .unwrap_or(0)
            .max(1) as i64;
        let (cdim, ctx_dim, pdim) = (
            self.cdim as usize,
            self.ctx_dim as usize,
            self.pair_dim as usize,
        );
        let (fu, cu) = (f as usize, c as usize);
        let mut cand = vec![0f32; n as usize * fu * cdim];
        let mut cmask = vec![true; n as usize * fu];
        let mut ctx = vec![0f32; n as usize * cu * ctx_dim];
        let mut ctx_mask = vec![true; n as usize * cu];
        let mut pair = vec![0f32; n as usize * fu * cu * pdim];
        let mut query = vec![0f32; n as usize * self.edim as usize];
        for (r, it) in items.iter().enumerate() {
            let fl = it.frontier_len;
            cand[r * fu * cdim..r * fu * cdim + fl * cdim].copy_from_slice(&it.cand);
            for i in 0..fl {
                cmask[r * fu + i] = false;
            }
            if self.query_encoder.is_some() {
                let e = self.edim as usize;
                query[r * e..(r + 1) * e].copy_from_slice(&it.query);
            }
            let cl = it.context_len;
            if cl > 0 {
                ctx[r * cu * ctx_dim..r * cu * ctx_dim + cl * ctx_dim].copy_from_slice(&it.ctx);
                for j in 0..cl {
                    ctx_mask[r * cu + j] = false;
                }
                for i in 0..fl {
                    let src = &it.pair[i * cl * pdim..(i + 1) * cl * pdim];
                    let dst = ((r * fu + i) * cu) * pdim;
                    pair[dst..dst + cl * pdim].copy_from_slice(src);
                }
            } else {
                ctx_mask[r * cu] = false; // one null context slot so attention is defined
            }
        }
        if !self.zero_blocks.is_empty() && self.config.feature_set == "raw-v5" {
            let edim = self.edim as usize;
            let ranges: HashMap<&str, (usize, usize)> = HashMap::from([
                ("candidate", (0, edim)),
                ("query", (edim, 2 * edim)),
                ("path_mean", (2 * edim, 3 * edim)),
                ("parent", (3 * edim, 4 * edim)),
            ]);
            for name in &self.zero_blocks {
                if name == "pair" {
                    for k in 0..(n as usize * fu * cu) {
                        pair[k * pdim] = 0.0;
                    }
                } else if let Some((lo, hi)) = ranges.get(name.as_str()) {
                    for row in 0..(n as usize * fu) {
                        for col in *lo..*hi {
                            cand[row * cdim + col] = 0.0;
                        }
                    }
                }
            }
        }
        let dev = self.device();
        let cand_t = Tensor::from_slice(&cand)
            .view([n, f, self.cdim])
            .to_device(dev);
        let cmask_t = Tensor::from_slice(&cmask).view([n, f]).to_device(dev);
        let ctx_t = Tensor::from_slice(&ctx)
            .view([n, c, self.ctx_dim])
            .to_device(dev);
        let ctx_mask_t = Tensor::from_slice(&ctx_mask).view([n, c]).to_device(dev);
        let pair_t = Tensor::from_slice(&pair)
            .view([n, f, c, self.pair_dim])
            .to_device(dev);
        let h = self
            .candidate_norm
            .forward(&self.candidate_encoder.forward(&cand_t));
        let ctx_h = self
            .context_norm
            .forward(&self.context_encoder.forward(&ctx_t));
        let q = match (&self.query_encoder, &self.query_token) {
            (Some(enc), _) => enc.forward(
                &Tensor::from_slice(&query)
                    .view([n, 1, self.edim])
                    .to_device(dev),
            ),
            (None, Some(token)) => token.expand([n, 1, self.config.hidden_dimension], false),
            _ => unreachable!(),
        };
        let mut h = h;
        for block in &self.blocks {
            h = block.forward(&h, &ctx_h, &q, &pair_t, &ctx_mask_t);
        }
        let head = self
            .score2
            .forward(&self.score0.forward(&h).gelu("none"))
            .squeeze_dim(-1);
        let scores = match &self.greedy_tau {
            Some(tau) => &head + tau * cand_t.select(2, self.cosine_column),
            None => head.shallow_clone(),
        };
        (
            scores.masked_fill(&cmask_t, f64::NEG_INFINITY),
            head.masked_fill(&cmask_t, f64::NEG_INFINITY),
        )
    }

    /// The stop head on `[N, 8]` rows → `[N]` logits.
    pub fn stop_logits(&self, rows: &[[f32; STOP_DIM]]) -> Tensor {
        if rows.is_empty() {
            return Tensor::zeros([0], (Kind::Float, self.device()));
        }
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let t = Tensor::from_slice(&flat)
            .view([rows.len() as i64, STOP_DIM as i64])
            .to_device(self.device());
        self.stop2
            .forward(&self.stop0.forward(&t).gelu("none"))
            .squeeze_dim(-1)
    }

    /// The gradient pass over recorded decisions (the walk's second pass):
    /// per episode, per decision, the score row and residual row (frontier
    /// width), scored in chunks of `score_chunk`; and one stop-logit tensor
    /// per episode.
    pub fn forward_training(&self, walks: &[WalkResult]) -> Result<TrainingOut, HfError> {
        let mut items: Vec<&DecisionItem> = Vec::new();
        let mut owner: Vec<usize> = Vec::new();
        for (e, w) in walks.iter().enumerate() {
            for dec in &w.decisions {
                let item = dec.item.as_ref().ok_or_else(|| {
                    HfError::Invalid(
                        "the gradient pass needs the walk's decision items (keep_items)".into(),
                    )
                })?;
                items.push(item);
                owner.push(e);
            }
        }
        let chunk = if self.config.score_chunk == 0 {
            items.len().max(1)
        } else {
            self.config.score_chunk
        };
        let mut scores: Vec<Vec<Tensor>> = walks.iter().map(|_| Vec::new()).collect();
        let mut residuals: Vec<Vec<Tensor>> = walks.iter().map(|_| Vec::new()).collect();
        for (ci, chunk_items) in items.chunks(chunk).enumerate() {
            let start = ci * chunk;
            let (s, r) = self.score_decisions(chunk_items);
            for (k, item) in chunk_items.iter().enumerate() {
                let e = owner[start + k];
                let w = item.frontier_len as i64;
                scores[e].push(s.get(k as i64).narrow(0, 0, w));
                residuals[e].push(r.get(k as i64).narrow(0, 0, w));
            }
        }
        let stop_rows: Vec<[f32; STOP_DIM]> = walks
            .iter()
            .flat_map(|w| w.decisions.iter().map(|d| d.stop_features))
            .collect();
        let all_stop = self.stop_logits(&stop_rows);
        let mut stop = Vec::new();
        let mut cursor = 0i64;
        for w in walks {
            let n = w.decisions.len() as i64;
            stop.push(all_stop.narrow(0, cursor, n));
            cursor += n;
        }
        Ok(TrainingOut {
            scores,
            residuals,
            stop,
        })
    }
}

fn kind_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Float => "torch.float32",
        Kind::Double => "torch.float64",
        Kind::Half => "torch.float16",
        Kind::BFloat16 => "torch.bfloat16",
        Kind::Int64 => "torch.int64",
        Kind::Int => "torch.int32",
        Kind::Bool => "torch.bool",
        Kind::Uint8 => "torch.uint8",
        _ => "torch.unknown",
    }
}

fn kind_bytes(kind: Kind) -> usize {
    match kind {
        Kind::Double | Kind::Int64 => 8,
        Kind::Float | Kind::Int => 4,
        Kind::Half | Kind::BFloat16 => 2,
        _ => 1,
    }
}

/// The gradient pass's output.
pub struct TrainingOut {
    pub scores: Vec<Vec<Tensor>>,
    pub residuals: Vec<Vec<Tensor>>,
    pub stop: Vec<Tensor>,
}

/// The model as the walk's scorer (no gradients).
pub struct ModelScorer<'a> {
    pub model: &'a Model,
}

impl Scorer for ModelScorer<'_> {
    fn score(&mut self, batch: &DecisionBatch) -> Result<Scored, HfError> {
        let _guard = tch::no_grad_guard();
        let items: Vec<&DecisionItem> = batch.items.iter().collect();
        if items.is_empty() {
            return Ok(Scored::default());
        }
        let (s, r) = self.model.score_decisions(&items);
        let s = s.to_device(Device::Cpu);
        let r = r.to_device(Device::Cpu);
        let f = s.size()[1] as usize;
        let mut sv = vec![0f32; items.len() * f];
        let mut rv = vec![0f32; items.len() * f];
        s.copy_data(&mut sv, items.len() * f);
        r.copy_data(&mut rv, items.len() * f);
        let scores: Vec<Vec<f32>> = items
            .iter()
            .enumerate()
            .map(|(k, it)| sv[k * f..k * f + it.frontier_len].to_vec())
            .collect();
        let residuals: Vec<Vec<f32>> = items
            .iter()
            .enumerate()
            .map(|(k, it)| rv[k * f..k * f + it.frontier_len].to_vec())
            .collect();
        Ok(Scored {
            scores,
            residuals: if self.model.config.greedy_prior {
                Some(residuals)
            } else {
                None
            },
        })
    }

    fn stop_logits(&mut self, rows: &[[f32; STOP_DIM]]) -> Result<Vec<f32>, HfError> {
        let _guard = tch::no_grad_guard();
        let t = self.model.stop_logits(rows).to_device(Device::Cpu);
        let mut out = vec![0f32; rows.len()];
        if !rows.is_empty() {
            t.copy_data(&mut out, rows.len());
        }
        Ok(out)
    }
}

/// `walk_losses`' knobs.
#[derive(Clone, Copy, Debug)]
pub struct LossConfig {
    pub distance_weight: f64,
    pub stop_weight: f64,
    pub residual_penalty: f64,
    pub residual_penalty_margin: Option<f64>,
}

impl Default for LossConfig {
    fn default() -> Self {
        Self {
            distance_weight: 0.5,
            stop_weight: 0.25,
            residual_penalty: 0.0,
            residual_penalty_margin: None,
        }
    }
}

pub struct Losses {
    pub edge: Tensor,
    pub distance: Tensor,
    pub stop: Tensor,
    pub residual: Tensor,
    pub total: Tensor,
}

impl Losses {
    pub fn values(&self) -> [f64; 5] {
        [
            &self.edge,
            &self.distance,
            &self.stop,
            &self.residual,
            &self.total,
        ]
        .map(|t| f64::try_from(t.detach()).unwrap_or(f64::NAN))
    }
}

/// `training_v5.walk_losses`, reduced decision → episode → batch.
#[allow(clippy::assign_op_pattern)]
pub fn walk_losses(
    out: &TrainingOut,
    walks: &[WalkResult],
    indexes: &[&hf_walk::EpisodeIndex],
    with_prior: bool,
    cfg: LossConfig,
) -> Result<Losses, HfError> {
    if walks.len() != indexes.len() {
        return Err(HfError::Invalid(
            "walks and episodes disagree on the batch size".into(),
        ));
    }
    if cfg.residual_penalty > 0.0 && !with_prior {
        return Err(HfError::Invalid(
            "the residual penalty needs the greedy prior's residuals".into(),
        ));
    }
    if let (true, Some(m)) = (cfg.residual_penalty > 0.0, cfg.residual_penalty_margin) {
        if m <= 0.0 {
            return Err(HfError::Invalid(
                "residual_penalty_margin must be positive".into(),
            ));
        }
    }
    let device = out.stop.first().map(|t| t.device()).unwrap_or(Device::Cpu);
    let mut edge_terms = Vec::new();
    let mut dist_terms = Vec::new();
    let mut stop_terms = Vec::new();
    let mut residual_terms = Vec::new();
    for (e, walk) in walks.iter().enumerate() {
        if walk.decisions.is_empty() {
            continue;
        }
        if cfg.residual_penalty > 0.0 {
            let mut r_terms = Vec::new();
            for (d, r) in out.residuals[e].iter().enumerate() {
                let keep = match cfg.residual_penalty_margin {
                    None => true,
                    Some(m) => walk
                        .cosine_margins
                        .get(d)
                        .copied()
                        .flatten()
                        .map(|cm| (cm as f64) < m)
                        .unwrap_or(false),
                };
                if keep && r.numel() > 0 {
                    let centred = r - r.mean(Kind::Float);
                    r_terms.push(centred.pow_tensor_scalar(2).mean(Kind::Float));
                }
            }
            if !r_terms.is_empty() {
                residual_terms.push(Tensor::stack(&r_terms, 0).mean(Kind::Float));
            }
        }
        let (on_rows, dist_rows) = hf_walk::candidate_labels(indexes[e], &walk.decisions);
        let mut e_terms = Vec::new();
        let mut d_terms = Vec::new();
        for (d, row) in out.scores[e].iter().enumerate() {
            let goal = Tensor::from_slice(&on_rows[d]).to_device(device);
            e_terms.push(row.binary_cross_entropy_with_logits::<Tensor>(
                &goal,
                None,
                None,
                Reduction::Mean,
            ));
            let mask: Vec<bool> = dist_rows[d].iter().map(|v| *v >= 0.0).collect();
            if mask.iter().any(|m| *m) {
                let mask_t = Tensor::from_slice(&mask).to_device(device);
                let target = Tensor::from_slice(&dist_rows[d]).to_device(device);
                let neg = -row;
                d_terms.push(neg.masked_select(&mask_t).smooth_l1_loss(
                    &target.masked_select(&mask_t),
                    Reduction::Mean,
                    1.0,
                ));
            }
        }
        edge_terms.push(Tensor::stack(&e_terms, 0).mean(Kind::Float));
        if !d_terms.is_empty() {
            dist_terms.push(Tensor::stack(&d_terms, 0).mean(Kind::Float));
        }
        let labels = Tensor::from_slice(&hf_walk::stop_labels(&walk.decisions)).to_device(device);
        stop_terms.push(out.stop[e].binary_cross_entropy_with_logits::<Tensor>(
            &labels,
            None,
            None,
            Reduction::Mean,
        ));
    }
    let zero = || Tensor::zeros([], (Kind::Float, device)).set_requires_grad(true);
    let mean_or_zero = |terms: Vec<Tensor>| {
        if terms.is_empty() {
            zero()
        } else {
            Tensor::stack(&terms, 0).mean(Kind::Float)
        }
    };
    let edge = mean_or_zero(edge_terms);
    let distance = mean_or_zero(dist_terms);
    let stop = mean_or_zero(stop_terms);
    let residual = mean_or_zero(residual_terms);
    let mut total = &edge + cfg.distance_weight * &distance + cfg.stop_weight * &stop;
    if cfg.residual_penalty > 0.0 {
        total = total + cfg.residual_penalty * &residual;
    }
    Ok(Losses {
        edge,
        distance,
        stop,
        residual,
        total,
    })
}

/// `torch.nn.utils.clip_grad_norm_`: returns the pre-clip total norm.
pub fn clip_grad_norm(vs: &nn::VarStore, max_norm: f64) -> f64 {
    tch::no_grad(|| {
        let vars = vs.trainable_variables();
        let norms: Vec<Tensor> = vars
            .iter()
            .filter(|v| v.grad().defined())
            .map(|v| v.grad().norm())
            .collect();
        if norms.is_empty() {
            return 0.0;
        }
        let total = f64::try_from(Tensor::stack(&norms, 0).norm()).unwrap_or(0.0);
        let coef = max_norm / (total + 1e-6);
        if coef < 1.0 {
            for v in &vars {
                let mut g = v.grad();
                if g.defined() {
                    let _ = g.g_mul_scalar_(coef);
                }
            }
        }
        total
    })
}

/// AdamW as PyTorch applies it (decoupled decay first, then the Adam step),
/// with its moments held here so they can be saved and restored exactly.
pub struct AdamW {
    pub lr: f64,
    pub weight_decay: f64,
    pub betas: (f64, f64),
    pub eps: f64,
    pub step_count: i64,
    /// Names exempt from weight decay (`tau`, norms, biases, when asked).
    pub no_decay: Vec<String>,
    params: Vec<(String, Tensor)>,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
}

impl AdamW {
    pub fn new(vs: &nn::VarStore, lr: f64, weight_decay: f64) -> Self {
        let mut params: Vec<(String, Tensor)> = vs
            .variables()
            .into_iter()
            .filter(|(_, t)| t.requires_grad())
            .collect();
        params.sort_by(|a, b| a.0.cmp(&b.0));
        let m = params.iter().map(|(_, t)| Tensor::zeros_like(t)).collect();
        let v = params.iter().map(|(_, t)| Tensor::zeros_like(t)).collect();
        Self {
            lr,
            weight_decay,
            betas: (0.9, 0.999),
            eps: 1e-8,
            step_count: 0,
            no_decay: Vec::new(),
            params,
            m,
            v,
        }
    }

    pub fn zero_grad(&mut self) {
        for (_, p) in &mut self.params {
            p.zero_grad();
        }
    }

    pub fn step(&mut self) {
        self.step_count += 1;
        let t = self.step_count as f64;
        let (b1, b2) = self.betas;
        let bc1 = 1.0 - b1.powf(t);
        let bc2 = 1.0 - b2.powf(t);
        tch::no_grad(|| {
            for (i, (name, p)) in self.params.iter_mut().enumerate() {
                let g = p.grad();
                if !g.defined() {
                    continue;
                }
                if self.weight_decay > 0.0 && !self.no_decay.contains(name) {
                    let _ = p.g_mul_scalar_(1.0 - self.lr * self.weight_decay);
                }
                let _ = self.m[i].g_mul_scalar_(b1).g_add_(&(&g * (1.0 - b1)));
                let _ = self.v[i].g_mul_scalar_(b2).g_add_(&(&g * &g * (1.0 - b2)));
                let denom = (&self.v[i] / bc2).sqrt() + self.eps;
                let update = (&self.m[i] / bc1) / denom * (-self.lr);
                let _ = p.g_add_(&update);
            }
        });
    }

    /// The moments and step count, as safetensors.
    pub fn save(&self, path: &Path) -> Result<(), HfError> {
        let mut named: Vec<(String, Tensor)> = Vec::new();
        for (i, (name, _)) in self.params.iter().enumerate() {
            named.push((format!("m.{name}"), self.m[i].to_device(Device::Cpu)));
            named.push((format!("v.{name}"), self.v[i].to_device(Device::Cpu)));
        }
        named.push(("step_count".into(), Tensor::from_slice(&[self.step_count])));
        let refs: Vec<(&str, Tensor)> = named
            .iter()
            .map(|(n, t)| (n.as_str(), t.shallow_clone()))
            .collect();
        Tensor::write_safetensors(&refs, path)
            .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))
    }

    pub fn load(&mut self, path: &Path) -> Result<(), HfError> {
        let tensors: HashMap<String, Tensor> = Tensor::read_safetensors(path)
            .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))?
            .into_iter()
            .collect();
        for (i, (name, _)) in self.params.iter().enumerate() {
            let m = tensors
                .get(&format!("m.{name}"))
                .ok_or_else(|| HfError::BandH(format!("optimiser state lacks {name}")))?;
            let v = tensors
                .get(&format!("v.{name}"))
                .ok_or_else(|| HfError::BandH(format!("optimiser state lacks {name}")))?;
            self.m[i] = m.to_device(self.m[i].device());
            self.v[i] = v.to_device(self.v[i].device());
        }
        let step = tensors
            .get("step_count")
            .ok_or_else(|| HfError::BandH("optimiser state lacks step_count".into()))?;
        self.step_count = step.int64_value(&[0]);
        Ok(())
    }
}
