//! The real-walk model of `model_v5.build_read_model_v5` on libtorch, with
//! the parameter names of the Python `state_dict` so a Python checkpoint
//! exported to safetensors loads here, and the redesign's relational feature
//! set behind the same architecture (a learned query token in place of the
//! query encoder). Attention is written out — `tch` has no
//! `MultiheadAttention` module — with PyTorch's semantics: the per-head
//! additive pair bias, a float key-padding mask of `-inf`, `nan_to_num` on
//! fully masked rows. The optimiser is a hand-written AdamW whose moments are
//! saved and loaded, so a resumed run is exact, and whose weight decay can be
//! waived for named parameters — `training.decay_exempt`, a list of globs over
//! the parameter names, empty in every config written before it existed. The
//! gradient clip is likewise a value and no longer a constant —
//! `training.clip_max_norm`, absent meaning the 1.0 every run so far used —
//! and the pre-clip norm it returns is computed whether or not it clips.

use std::collections::HashMap;
use std::path::Path;

pub mod r1;

use hf_core::HfError;
use hf_walk::{
    DecisionBatch, DecisionItem, FeatureSet, RawV5, RelationalV6, RelationalV6K, RelationalV6Prev,
    Scored, Scorer, WalkResult, STOP_DIM,
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
    /// `"raw-v5"` (the Python layout), `"relational-v6"` (the redesign),
    /// `"relational-v6-prev"` (the redesign with the previous-node channels)
    /// or `"relational-v6-k"` (the redesign with `K_TARGETS_DESIGN.md` §2's
    /// max-over-unregistered reduction and its three extra columns).
    #[serde(default = "default_feature_set")]
    pub feature_set: String,
    #[serde(default)]
    pub zero_embedding_blocks: Vec<String>,
    #[serde(default = "default_score_chunk")]
    pub score_chunk: usize,
    /// The R1 heads (`R1_HEAD_DESIGN.md` §8 ENG-2): `return_head` and
    /// `rstop_head`, built after every other parameter. False — every config
    /// written before the key existed — builds exactly today's model.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub return_head: bool,
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
            "relational-v6-prev" => Ok(Box::new(RelationalV6Prev)),
            "relational-v6-k" => Ok(Box::new(RelationalV6K)),
            other => Err(HfError::Invalid(format!("unknown feature set {other:?}"))),
        }
    }
}

/// The bound PyTorch gives a linear weight of this fan-in:
/// `kaiming_uniform_(a=sqrt(5))` is uniform on ±1/sqrt(fan_in).
fn torch_uniform(in_dim: i64) -> nn::Init {
    let bound = 1.0 / (in_dim as f64).sqrt();
    nn::Init::Uniform {
        lo: -bound,
        up: bound,
    }
}

/// A linear layer initialised as `torch.nn.Linear` initialises one, which
/// `tch`'s `LinearConfig::default()` does not: its default is Kaiming-uniform
/// with the ReLU gain, a bound of sqrt(6/fan_in) — sqrt(6) = 2.449 times
/// PyTorch's. The bias default already agrees (`tch` derives ±1/sqrt(fan_in)
/// from `in_dim`, as PyTorch does).
fn torch_linear(p: nn::Path, in_dim: i64, out_dim: i64, bias: bool) -> nn::Linear {
    nn::linear(
        p,
        in_dim,
        out_dim,
        nn::LinearConfig {
            ws_init: torch_uniform(in_dim),
            bs_init: None,
            bias,
        },
    )
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
                ws_init: torch_uniform(hidden),
                bs_init: Some(nn::Init::Const(0.0)),
                bias: true,
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
        Self {
            norm_context: nn::layer_norm(p / "norm_context", vec![hidden], Default::default()),
            context_attention: Attention::new(&(p / "context_attention"), hidden, heads),
            context_bias: torch_linear(p / "context_bias", pair_dim, heads, false),
            norm_query: nn::layer_norm(p / "norm_query", vec![hidden], Default::default()),
            query_attention: Attention::new(&(p / "query_attention"), hidden, heads),
            norm_ff: nn::layer_norm(p / "norm_ff", vec![hidden], Default::default()),
            ff0: torch_linear(p / "feedforward" / "0", hidden, multiplier * hidden, true),
            ff2: torch_linear(p / "feedforward" / "2", multiplier * hidden, hidden, true),
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

/// Which R1 heads a model carries. `from_config` is both or neither
/// (`model.return_head`); `rstop_only` is `--trunk-plus-rstop`'s model: the
/// stage-0 config unchanged plus `rstop_head` alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct R1Heads {
    pub return_head: bool,
    pub rstop_head: bool,
}

impl R1Heads {
    pub fn from_config(config: &ModelConfig) -> Self {
        Self {
            return_head: config.return_head,
            rstop_head: config.return_head,
        }
    }

    pub fn rstop_only() -> Self {
        Self {
            return_head: false,
            rstop_head: true,
        }
    }
}

/// The parameter-name patterns of the R1 heads.
pub const RETURN_HEAD_PATTERN: &str = "return_head.*";
pub const RSTOP_HEAD_PATTERN: &str = "rstop_head.*";
pub const R1_HEAD_PATTERNS: [&str; 2] = [RETURN_HEAD_PATTERN, RSTOP_HEAD_PATTERN];

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
    return_head: Option<(nn::Linear, nn::Linear)>,
    rstop_head: Option<(nn::Linear, nn::Linear)>,
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
        let heads = R1Heads::from_config(&config);
        Self::new_with_heads(config, device, heads)
    }

    /// The model with the R1 heads named by `heads`, created after every
    /// other parameter so the existing parameters' initialisation, and so a
    /// seed's trunk, is unchanged.
    pub fn new_with_heads(
        config: ModelConfig,
        device: Device,
        heads: R1Heads,
    ) -> Result<Self, HfError> {
        let features = config.features()?;
        let edim = config.embedding_dimension as usize;
        let cdim = features.candidate_dim(edim) as i64;
        let ctx_dim = features.context_dim(edim) as i64;
        let pair_dim = features.pair_dim() as i64;
        let cosine_column = features.cosine_column(edim) as i64;
        let hidden = config.hidden_dimension;
        let vs = nn::VarStore::new(device);
        let p = vs.root();
        let candidate_encoder = torch_linear(&p / "candidate_encoder", cdim, hidden, true);
        let candidate_norm =
            nn::layer_norm(&p / "candidate_norm", vec![hidden], Default::default());
        let context_encoder = torch_linear(&p / "context_encoder", ctx_dim, hidden, true);
        let context_norm = nn::layer_norm(&p / "context_norm", vec![hidden], Default::default());
        let (query_encoder, query_token) = match features.query_dim(edim) {
            Some(qdim) => (
                Some(torch_linear(
                    &p / "query_encoder",
                    qdim as i64,
                    hidden,
                    true,
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
        let score0 = torch_linear(
            &p / "score_head" / "0",
            hidden,
            config.score_hidden_dimension,
            true,
        );
        let score2 = torch_linear(
            &p / "score_head" / "2",
            config.score_hidden_dimension,
            1,
            true,
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
        let stop0 = torch_linear(
            &p / "stop_head" / "0",
            STOP_DIM as i64,
            config.coverage_hidden_dimension,
            true,
        );
        let stop2 = torch_linear(
            &p / "stop_head" / "2",
            config.coverage_hidden_dimension,
            1,
            true,
        );
        let device_probe = p.zeros_no_train("_device_probe", &[1]);
        // the R1 heads (`R1_HEAD_DESIGN.md` §3, §4): shaped like score_head
        // and stop_head, over `[h ; 4 extras]` and `[rstop_row ; mean h ; max h]`
        let return_head = if heads.return_head {
            let width = hidden + hf_walk::RETURN_EXTRAS as i64;
            Some((
                torch_linear(
                    &p / "return_head" / "0",
                    width,
                    config.score_hidden_dimension,
                    true,
                ),
                torch_linear(
                    &p / "return_head" / "2",
                    config.score_hidden_dimension,
                    1,
                    true,
                ),
            ))
        } else {
            None
        };
        let rstop_head = if heads.rstop_head {
            let width = hf_walk::RSTOP_ROW_DIM as i64 + 2 * hidden;
            Some((
                torch_linear(
                    &p / "rstop_head" / "0",
                    width,
                    config.coverage_hidden_dimension,
                    true,
                ),
                torch_linear(
                    &p / "rstop_head" / "2",
                    config.coverage_hidden_dimension,
                    1,
                    true,
                ),
            ))
        } else {
            None
        };
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
            return_head,
            rstop_head,
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
        let t = self.trunk(items);
        self.score_from(&t)
    }

    /// `score_decisions` and, from the same forward pass, each item's final
    /// hidden states over its valid candidates, mean-pooled and max-pooled:
    /// `[N, 2H]`, detached (`R1_HEAD_DESIGN.md` §4 item 3).
    pub fn score_decisions_with_hidden(&self, items: &[&DecisionItem]) -> (Tensor, Tensor, Tensor) {
        let t = self.trunk(items);
        let (scores, residuals) = self.score_from(&t);
        let h = t.h.detach();
        let pad = t.cmask.unsqueeze(-1); // [N, F, 1], true on padding
        let valid = pad.logical_not().to_kind(Kind::Float);
        let count = valid.sum_dim_intlist(1, false, Kind::Float).clamp_min(1.0); // [N, 1]
        let mean = (&h * &valid).sum_dim_intlist(1, false, Kind::Float) / count;
        let max = h
            .masked_fill(&pad, f64::NEG_INFINITY)
            .amax(1, false)
            .nan_to_num(0.0, Some(0.0), Some(0.0));
        (scores, residuals, Tensor::cat(&[mean, max], 1))
    }

    /// The trunk's final hidden states `[N, F, H]` over a padded batch of
    /// items (the return items' `h_v`, §3), and the padding mask `[N, F]`.
    pub fn hidden_states(&self, items: &[&DecisionItem]) -> (Tensor, Tensor) {
        let t = self.trunk(items);
        (t.h, t.cmask)
    }

    fn score_from(&self, t: &Trunk) -> (Tensor, Tensor) {
        let head = self
            .score2
            .forward(&self.score0.forward(&t.h).gelu("none"))
            .squeeze_dim(-1);
        let scores = match &self.greedy_tau {
            Some(tau) => &head + tau * t.cand.select(2, self.cosine_column),
            None => head.shallow_clone(),
        };
        (
            scores.masked_fill(&t.cmask, f64::NEG_INFINITY),
            head.masked_fill(&t.cmask, f64::NEG_INFINITY),
        )
    }

    /// `return_head` on `[n, H + 4]` rows (`[h_v ; extras]`) → `[n]` logits.
    pub fn return_head_logits(&self, x: &Tensor) -> Result<Tensor, HfError> {
        let (l0, l2) = self
            .return_head
            .as_ref()
            .ok_or_else(|| HfError::Invalid("this model has no return_head".into()))?;
        Ok(l2.forward(&l0.forward(x).gelu("none")).squeeze_dim(-1))
    }

    /// `rstop_head` on `[n, 11 + 2H]` rows → `[n]` logits.
    pub fn rstop_head_logits(&self, x: &Tensor) -> Result<Tensor, HfError> {
        let (l0, l2) = self
            .rstop_head
            .as_ref()
            .ok_or_else(|| HfError::Invalid("this model has no rstop_head".into()))?;
        Ok(l2.forward(&l0.forward(x).gelu("none")).squeeze_dim(-1))
    }

    /// `return_logits(item, extras)`: the trunk forward over the return
    /// items, then `return_head` on `[h ; extras]`, one logit tensor per item
    /// (its candidates' logits). The graph runs through the trunk, so a
    /// gradient could reach it were it not frozen (ENG-2 test (d)).
    pub fn return_logits(
        &self,
        items: &[&DecisionItem],
        extras: &[&[f32]],
    ) -> Result<Vec<Tensor>, HfError> {
        let (h, _) = self.hidden_states(items);
        let mut out = Vec::with_capacity(items.len());
        for (k, (item, ex)) in items.iter().zip(extras).enumerate() {
            let n = item.frontier_len as i64;
            if ex.len() != item.frontier_len * hf_walk::RETURN_EXTRAS {
                return Err(HfError::BandH(format!(
                    "{} extras for {} candidates",
                    ex.len(),
                    item.frontier_len
                )));
            }
            let hv = h.get(k as i64).narrow(0, 0, n);
            let ext = Tensor::from_slice(ex)
                .view([n, hf_walk::RETURN_EXTRAS as i64])
                .to_device(self.device());
            out.push(self.return_head_logits(&Tensor::cat(&[hv, ext], 1))?);
        }
        Ok(out)
    }

    /// Whether the model carries each R1 head.
    pub fn r1_heads(&self) -> R1Heads {
        R1Heads {
            return_head: self.return_head.is_some(),
            rstop_head: self.rstop_head.is_some(),
        }
    }

    /// The hidden width `H`.
    pub fn hidden_dimension(&self) -> i64 {
        self.config.hidden_dimension
    }

    /// Load a safetensors checkpoint BY NAME, strictly: every tensor in the
    /// file must be a model parameter of the same shape, unless its name
    /// matches `extra_ok` (then it is read and ignored); every model
    /// parameter absent from the file must match `missing_ok` (it keeps its
    /// initialisation). Anything else is band H, and nothing is copied until
    /// every check has passed. Returns the ignored names and the missing ones.
    pub fn load_strict(
        &mut self,
        path: &Path,
        missing_ok: &[&str],
        extra_ok: &[&str],
    ) -> Result<(Vec<String>, Vec<String>), HfError> {
        let file: HashMap<String, Tensor> = Tensor::read_safetensors(path)
            .map_err(|e| HfError::Invalid(format!("{}: {e}", path.display())))?
            .into_iter()
            .collect();
        let vars = self.vs.variables();
        let mut ignored = Vec::new();
        let mut missing = Vec::new();
        let mut names: Vec<&String> = file.keys().collect();
        names.sort();
        for name in names {
            match vars.get(name) {
                Some(var) => {
                    if var.size() != file[name].size() {
                        return Err(HfError::BandH(format!(
                            "{}: {name} has shape {:?}, the model's is {:?}",
                            path.display(),
                            file[name].size(),
                            var.size()
                        )));
                    }
                }
                None if extra_ok.iter().any(|p| glob_match(p, name)) => ignored.push(name.clone()),
                None => {
                    return Err(HfError::BandH(format!(
                        "{}: {name} is not a parameter of this model",
                        path.display()
                    )))
                }
            }
        }
        let mut var_names: Vec<&String> = vars.keys().collect();
        var_names.sort();
        for name in var_names {
            if !file.contains_key(name) {
                if missing_ok.iter().any(|p| glob_match(p, name)) {
                    missing.push(name.clone());
                } else {
                    return Err(HfError::BandH(format!(
                        "{}: the checkpoint lacks {name}",
                        path.display()
                    )));
                }
            }
        }
        tch::no_grad(|| {
            for (name, var) in &vars {
                if let Some(src) = file.get(name) {
                    let mut dst = var.shallow_clone();
                    dst.copy_(&src.to_device(var.device()).to_kind(var.kind()));
                }
            }
        });
        Ok((ignored, missing))
    }

    /// `training.trainable`: every parameter whose name matches none of
    /// `patterns` gets `requires_grad(false)`, so an optimiser built AFTER
    /// this call never holds it (and never decays it). A pattern matching no
    /// parameter is refused. Returns the trainable names, sorted.
    pub fn freeze_except(&self, patterns: &[String]) -> Result<Vec<String>, HfError> {
        let vars = self.vs.variables();
        for p in patterns {
            if !vars.keys().any(|n| glob_match(p, n)) {
                return Err(HfError::Invalid(format!(
                    "training.trainable: {p:?} matches no parameter"
                )));
            }
        }
        let mut kept = Vec::new();
        for (name, var) in &vars {
            let keep = patterns.iter().any(|p| glob_match(p, name));
            let _ = var.set_requires_grad(keep);
            if keep {
                kept.push(name.clone());
            }
        }
        kept.sort();
        Ok(kept)
    }

    /// The parameter count of the tensors whose names match `patterns`.
    pub fn parameter_count_matching(&self, patterns: &[&str]) -> i64 {
        self.vs
            .variables()
            .iter()
            .filter(|(n, _)| patterns.iter().any(|p| glob_match(p, n)))
            .map(|(_, t)| t.numel() as i64)
            .sum()
    }

    /// The names of the variables matching `patterns`, sorted: what a
    /// `state_digest` exclusion list is built from.
    pub fn names_matching(&self, patterns: &[&str]) -> Vec<String> {
        let mut v: Vec<String> = self
            .vs
            .variables()
            .into_keys()
            .filter(|n| patterns.iter().any(|p| glob_match(p, n)))
            .collect();
        v.sort();
        v
    }

    fn trunk(&self, items: &[&DecisionItem]) -> Trunk {
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
        Trunk {
            h,
            cand: cand_t,
            cmask: cmask_t,
        }
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

/// The trunk's output over one padded batch.
struct Trunk {
    h: Tensor,
    cand: Tensor,
    cmask: Tensor,
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
        Ok(scored_rows(&items, &s, &r, self.model.config.greedy_prior))
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

    fn score_pooled(&mut self, batch: &DecisionBatch) -> Result<(Scored, Vec<Vec<f32>>), HfError> {
        let _guard = tch::no_grad_guard();
        let items: Vec<&DecisionItem> = batch.items.iter().collect();
        if items.is_empty() {
            return Ok((Scored::default(), Vec::new()));
        }
        let (s, r, pooled) = self.model.score_decisions_with_hidden(&items);
        let scored = scored_rows(&items, &s, &r, self.model.config.greedy_prior);
        Ok((scored, tensor_rows(&pooled)))
    }

    fn rstop_logits(&mut self, inputs: &[hf_walk::RstopInput]) -> Result<Vec<f32>, HfError> {
        let _guard = tch::no_grad_guard();
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let width = hf_walk::RSTOP_ROW_DIM + 2 * self.model.hidden_dimension() as usize;
        let flat: Vec<f32> = inputs.iter().flat_map(|i| i.flat()).collect();
        if flat.len() != inputs.len() * width {
            return Err(HfError::BandH(format!(
                "rstop inputs are {} values for {} rows of {width}",
                flat.len(),
                inputs.len()
            )));
        }
        let x = Tensor::from_slice(&flat)
            .view([inputs.len() as i64, width as i64])
            .to_device(self.model.device());
        let t = self.model.rstop_head_logits(&x)?.to_device(Device::Cpu);
        let mut out = vec![0f32; inputs.len()];
        t.copy_data(&mut out, inputs.len());
        Ok(out)
    }
}

/// A `[N, F]` score and residual pair cut back to each item's frontier.
fn scored_rows(items: &[&DecisionItem], s: &Tensor, r: &Tensor, with_prior: bool) -> Scored {
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
    Scored {
        scores,
        residuals: if with_prior { Some(residuals) } else { None },
    }
}

/// A `[N, W]` tensor as N rows of W f32s.
pub fn tensor_rows(t: &Tensor) -> Vec<Vec<f32>> {
    let t = t.detach().to_device(Device::Cpu).contiguous();
    let (n, w) = (t.size()[0] as usize, t.size()[1] as usize);
    let mut flat = vec![0f32; n * w];
    if n * w > 0 {
        t.copy_data(&mut flat, n * w);
    }
    flat.chunks(w.max(1)).take(n).map(|c| c.to_vec()).collect()
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

/// `torch.nn.utils.clip_grad_norm_`: returns the pre-clip total norm, having
/// scaled every gradient by `max_norm / (norm + 1e-6)` when that is below one.
///
/// `None` is `training.clip_max_norm: null` — no clipping at all. The total
/// norm is still computed and returned, by the same reduction and in the same
/// order, so a run without a clip still logs the true norm in `updates.jsonl`;
/// nothing is scaled. `Some(1.0)` is what the runner passed unconditionally
/// before the key existed, and is what an absent key still means.
pub fn clip_grad_norm(vs: &nn::VarStore, max_norm: Option<f64>) -> f64 {
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
        if let Some(max_norm) = max_norm {
            let coef = max_norm / (total + 1e-6);
            if coef < 1.0 {
                for v in &vars {
                    let mut g = v.grad();
                    if g.defined() {
                        let _ = g.g_mul_scalar_(coef);
                    }
                }
            }
        }
        total
    })
}

/// A simple glob over a parameter's FULL name, both ends anchored: `*` stands
/// for any run of characters, the dots included, and every other character is
/// literal. So `greedy_tau` matches that name and nothing else, `*bias` matches
/// every name ENDING in `bias` (`…out_proj.bias` and `…in_proj_bias` alike, but
/// not `blocks.0.context_bias.weight`), and `*norm*.weight` matches every
/// LayerNorm gain (`candidate_norm.weight` as well as `blocks.0.norm_ff.weight`
/// — note that `*.norm*.weight`, with the dot, misses the first).
pub fn glob_match(pattern: &str, name: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == name;
    }
    let mut rest = match name.strip_prefix(parts[0]) {
        Some(r) => r,
        None => return false,
    };
    let last = parts.len() - 1;
    for part in &parts[1..last] {
        match rest.find(part) {
            Some(i) => rest = &rest[i + part.len()..],
            None => return false,
        }
    }
    rest.ends_with(parts[last])
}

/// AdamW as PyTorch applies it (decoupled decay first, then the Adam step),
/// with its moments held here so they can be saved and restored exactly.
pub struct AdamW {
    pub lr: f64,
    pub weight_decay: f64,
    pub betas: (f64, f64),
    pub eps: f64,
    pub step_count: i64,
    /// Parameter names exempt from weight decay (`greedy_tau`, the LayerNorm
    /// gains, the biases — when asked). Empty unless `set_decay_exempt` fills
    /// it from the config's `training.decay_exempt`.
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

    /// Fill `no_decay` from `patterns` (`training.decay_exempt` in the config),
    /// resolved against THIS optimiser's own parameter names — the ones `step`
    /// consults, already filtered to the trainable ones. Returns the matched
    /// names in the optimiser's order and the patterns that matched nothing,
    /// which the runner records and prints: a pattern matching nothing is not
    /// an error (a config may name `greedy_tau` for a model built without the
    /// prior), but it is never silent.
    pub fn set_decay_exempt(&mut self, patterns: &[String]) -> (Vec<String>, Vec<String>) {
        let matched: Vec<String> = self
            .params
            .iter()
            .filter(|(name, _)| patterns.iter().any(|p| glob_match(p, name)))
            .map(|(name, _)| name.clone())
            .collect();
        let unmatched: Vec<String> = patterns
            .iter()
            .filter(|p| !self.params.iter().any(|(name, _)| glob_match(p, name)))
            .cloned()
            .collect();
        self.no_decay = matched.clone();
        (matched, unmatched)
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
