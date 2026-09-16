//! Initialisation: every linear weight must land inside PyTorch's bound.
//!
//! `torch.nn.Linear` initialises its weight with `kaiming_uniform_(a=sqrt(5))`
//! — uniform on ±1/sqrt(fan_in) — while `tch`'s `LinearConfig::default()` is
//! Kaiming-uniform with the ReLU gain, ±sqrt(6/fan_in), sqrt(6) = 2.449 times
//! wider. The twelve linear weights of the fixture model (eleven named sites;
//! each block holds two attentions, so `out_proj` appears twice) are checked
//! against PyTorch's bound, and the hand-initialised `in_proj_weight` against
//! the xavier bound it has always had.
//!
//! The model is built with `greedy_prior: false` — the fixture golden config
//! sets it true, which zeroes `score_head.2` after construction and would hide
//! that weight's initialiser. The prior's own invariants (`score_head.2` all
//! zero, `greedy_tau` = 10) are asserted in `goldens.rs`.
//!
//! Values are pooled over `SEEDS` instantiations before the standard deviation
//! is judged: `context_bias` is 2x2 in this config, and four samples cannot
//! resolve a 20 % band (the relative sampling error of s is about 0.447/sqrt(n)).

use std::collections::BTreeMap;
use std::path::PathBuf;

use hf_model::{Model, ModelConfig};
use serde_json::Value;
use tch::{Device, Tensor};

const SEEDS: i64 = 128;
const FIXED_SEED: i64 = 20260917;

/// Every linear weight of the one-block fixture model, `in_proj_weight` apart.
const LINEAR_WEIGHTS: [&str; 12] = [
    "candidate_encoder.weight",
    "context_encoder.weight",
    "query_encoder.weight",
    "blocks.0.context_attention.out_proj.weight",
    "blocks.0.query_attention.out_proj.weight",
    "blocks.0.context_bias.weight",
    "blocks.0.feedforward.0.weight",
    "blocks.0.feedforward.2.weight",
    "score_head.0.weight",
    "score_head.2.weight",
    "stop_head.0.weight",
    "stop_head.2.weight",
];

const IN_PROJ: [&str; 2] = [
    "blocks.0.context_attention.in_proj_weight",
    "blocks.0.query_attention.in_proj_weight",
];

fn fixture_config() -> ModelConfig {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/model.json");
    let g: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut m = g["model_config"].clone();
    m["greedy_prior"] = Value::Bool(false);
    ModelConfig::from_value(&m, 8).unwrap()
}

fn values(t: &Tensor) -> Vec<f64> {
    let flat = t.detach().to_device(Device::Cpu).reshape([-1]);
    Vec::<f32>::try_from(flat)
        .unwrap()
        .into_iter()
        .map(|x| x as f64)
        .collect()
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |a, x| a.max(x.abs()))
}

/// The population standard deviation about zero, which is what the uniform's
/// theoretical std is: the mean is zero by construction.
fn std_about_zero(v: &[f64]) -> f64 {
    (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
}

#[test]
fn linear_weights_use_torchs_bound_not_tchs_kaiming_relu_default() {
    let config = fixture_config();
    // the fixed-seed model: the bound, weight by weight
    tch::manual_seed(FIXED_SEED);
    let model = Model::new(config.clone(), Device::Cpu).unwrap();
    let vars = model.vs.variables();
    // the enumerated names are exactly the two-dimensional `.weight` tensors
    let found: Vec<String> = {
        let mut v: Vec<String> = vars
            .iter()
            .filter(|(n, t)| n.ends_with(".weight") && t.size().len() == 2)
            .map(|(n, _)| n.clone())
            .collect();
        v.sort();
        v
    };
    let mut want: Vec<String> = LINEAR_WEIGHTS.iter().map(|s| s.to_string()).collect();
    want.sort();
    assert_eq!(found, want, "the set of linear weights changed");
    for name in LINEAR_WEIGHTS {
        let t = &vars[name];
        let fan_in = t.size()[1] as f64;
        let bound = 1.0 / fan_in.sqrt();
        let m = max_abs(&values(t));
        assert!(
            m <= bound + 1e-6,
            "{name}: max |w| {m} > PyTorch's bound {bound} (fan_in {fan_in})"
        );
    }
    // and the xavier `in_proj_weight`, whose bound is unchanged
    for name in IN_PROJ {
        let t = &vars[name];
        let hidden = t.size()[1] as f64;
        let bound = (6.0 / (4.0 * hidden)).sqrt();
        let m = max_abs(&values(t));
        assert!(
            m <= bound + 1e-6,
            "{name}: max |w| {m} > the xavier bound {bound}"
        );
    }

    // the spread, pooled over seeds so even a 2x2 weight has enough samples
    let mut pooled: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let mut fan_ins: BTreeMap<&str, f64> = BTreeMap::new();
    for seed in 1..=SEEDS {
        tch::manual_seed(seed);
        let model = Model::new(config.clone(), Device::Cpu).unwrap();
        let vars = model.vs.variables();
        for name in LINEAR_WEIGHTS.iter().chain(IN_PROJ.iter()) {
            let t = &vars[*name];
            fan_ins.insert(name, t.size()[1] as f64);
            pooled.entry(name).or_default().extend(values(t));
        }
    }
    for name in LINEAR_WEIGHTS {
        let fan_in = fan_ins[name];
        let bound = 1.0 / fan_in.sqrt();
        let v = &pooled[name];
        assert!(v.len() >= 256, "{name}: {} pooled samples", v.len());
        let m = max_abs(v);
        assert!(m <= bound + 1e-6, "{name}: pooled max |w| {m} > {bound}");
        assert!(
            m >= 0.5 * bound,
            "{name}: pooled max |w| {m} is far below the bound {bound}; is fan_in right?"
        );
        let want = 1.0 / (3.0 * fan_in).sqrt();
        let got = std_about_zero(v);
        assert!(
            (got - want).abs() <= 0.2 * want,
            "{name}: std {got} is not within 20 % of {want} (fan_in {fan_in})"
        );
    }
    for name in IN_PROJ {
        let hidden = fan_ins[name];
        let bound = (6.0 / (4.0 * hidden)).sqrt();
        let v = &pooled[name];
        let m = max_abs(v);
        assert!(m <= bound + 1e-6, "{name}: pooled max |w| {m} > {bound}");
        assert!(m >= 0.95 * bound, "{name}: the xavier bound narrowed");
        let want = bound / 3f64.sqrt();
        let got = std_about_zero(v);
        assert!(
            (got - want).abs() <= 0.2 * want,
            "{name}: std {got} is not within 20 % of {want}"
        );
    }
}
