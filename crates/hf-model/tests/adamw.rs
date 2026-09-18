//! `training.decay_exempt`: AdamW's weight-decay exemption by parameter-name
//! glob (`hf_model::glob_match`, `AdamW::set_decay_exempt`).
//!
//! What is asserted here:
//! 1. the glob is anchored at both ends and `*` crosses the dots;
//! 2. **the empty switch is today's optimiser, bit for bit** — one step and two
//!    steps of the fixture model reproduce digests measured at commit `1fd3733`,
//!    before the switch existed (the command is named below the constants);
//! 3. a hand-computed decoupled-decay step on a tiny `VarStore`, both with the
//!    switch empty (every parameter decayed) and with it set (`greedy_tau`, the
//!    LayerNorm gains and the biases NOT decayed, a linear weight decayed),
//!    asserted on both sides: within tolerance of the right prediction AND far
//!    from the wrong one;
//! 4. the names the pattern value Part B intends actually matches, on the
//!    fixture model — including the four it does NOT (see the test);
//! 5. the optimiser state round-trips across a save/load with the switch set:
//!    the moments are bit-identical, a resumed run continues bit-identically,
//!    and a resume that forgets to re-apply the exemption is detectably
//!    different (the exemption lives in the config, not in the safetensors).

use std::path::PathBuf;

use hf_model::{glob_match, AdamW, Model, ModelConfig};
use serde_json::{json, Value};
use tch::nn::{self, VarStore};
use tch::{Device, Kind, Tensor};

/// The digests of the fixture model after one and two AdamW steps at
/// `lr = 1e-3`, `weight_decay = 0.01`, on the synthetic gradients of
/// `backward_synthetic`, measured on the code at commit `1fd3733` — the
/// optimiser before `training.decay_exempt` existed, whose `no_decay` was
/// unconditionally empty. To re-measure: check out that commit's
/// `crates/hf-model/src/lib.rs`, copy the helpers above into a test that calls
/// `AdamW::new` and `step` alone, and read the two digests it prints.
const ONE_STEP_DIGEST: &str =
    "sha256:79eebd00a57232da46318045bb780a2b5e5d2a0e5622f12edb8eda2c0f8e20de";
const TWO_STEP_DIGEST: &str =
    "sha256:b0f30e0569797932367d9069f7db560e63357ea56d42e76e076fec747d2e487a";

/// What Part B's preregistration proposed, verbatim.
fn intended_patterns() -> Vec<String> {
    ["greedy_tau", "*.norm*.weight", "*.norm*.bias", "*.bias"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// The same intent, written so the dots do not swallow the names: `*norm*` also
/// reaches `candidate_norm` / `context_norm` (no dot before `norm`) and `*bias`
/// also reaches the attention `in_proj_bias` (no dot before `bias`).
fn corrected_patterns() -> Vec<String> {
    ["greedy_tau", "*norm*.weight", "*bias"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

fn goldens() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/goldens")
}

fn fixture_model() -> Model {
    let g: Value =
        serde_json::from_str(&std::fs::read_to_string(goldens().join("model.json")).unwrap())
            .unwrap();
    let config = ModelConfig::from_value(&g["model_config"], 8).unwrap();
    let mut model = Model::new(config, Device::Cpu).unwrap();
    model
        .vs
        .load(goldens().join("fixture.safetensors"))
        .unwrap();
    model
}

fn sorted_trainable(vs: &VarStore) -> Vec<(String, Tensor)> {
    let mut named: Vec<(String, Tensor)> = vs
        .variables()
        .into_iter()
        .filter(|(_, t)| t.requires_grad())
        .collect();
    named.sort_by(|a, b| a.0.cmp(&b.0));
    named
}

/// `sum_i c_i * (p_i . p_i)` over every trainable parameter, sorted by name:
/// a gradient for EVERY parameter (an undefined one would be skipped by `step`
/// and silently drop out of the golden), `2 * c_i * p_i`, deterministic.
fn backward_synthetic(vs: &VarStore) {
    let mut loss: Option<Tensor> = None;
    for (i, (_, t)) in sorted_trainable(vs).iter().enumerate() {
        let term = (t * t).sum(Kind::Float) * (1.0 + (i as f64) * 0.1);
        loss = Some(match loss {
            Some(l) => l + term,
            None => term,
        });
    }
    loss.expect("the model has trainable parameters").backward();
}

fn values(vs: &VarStore, name: &str) -> Vec<f64> {
    let t = vs
        .variables()
        .remove(name)
        .unwrap_or_else(|| panic!("{name} is not a parameter"));
    Vec::<f32>::try_from(t.detach().reshape([-1]))
        .unwrap()
        .into_iter()
        .map(|x| x as f64)
        .collect()
}

// ---------------------------------------------------------------- the glob

#[test]
fn the_glob_is_anchored_at_both_ends_and_star_crosses_the_dots() {
    // a pattern without a wildcard is the whole name, never a substring
    assert!(glob_match("greedy_tau", "greedy_tau"));
    assert!(!glob_match("greedy_tau", "greedy_tau.weight"));
    assert!(!glob_match("greedy_tau", "model.greedy_tau"));
    assert!(!glob_match("tau", "greedy_tau"));
    // `*` spans the dots
    assert!(glob_match(
        "*.out_proj.bias",
        "blocks.0.context_attention.out_proj.bias"
    ));
    assert!(glob_match(
        "blocks.*.weight",
        "blocks.7.feedforward.0.weight"
    ));
    assert!(glob_match("*", "anything.at.all"));
    // the tail is anchored: `*bias` is every name ENDING in bias
    assert!(glob_match("*bias", "candidate_encoder.bias"));
    assert!(glob_match(
        "*bias",
        "blocks.0.context_attention.in_proj_bias"
    ));
    // ... and does not catch the linear layer whose NAME contains bias
    assert!(!glob_match("*bias", "blocks.0.context_bias.weight"));
    assert!(!glob_match("*bias", "blocks.0.context_bias.weight.extra"));
    // the head is anchored
    assert!(!glob_match("blocks.*", "score_head.0.weight"));
    assert!(glob_match("score_head.*", "score_head.0.weight"));
    // the dot in the brief's `*.norm*.weight` is literal, so it misses the
    // encoders' norms — the finding this file records
    assert!(glob_match("*.norm*.weight", "blocks.0.norm_ff.weight"));
    assert!(!glob_match("*.norm*.weight", "candidate_norm.weight"));
    assert!(glob_match("*norm*.weight", "candidate_norm.weight"));
}

// ------------------------------------------- (a) the empty switch is today

#[test]
fn the_empty_switch_reproduces_the_step_of_the_optimiser_before_it() {
    let model = fixture_model();
    let mut opt = AdamW::new(&model.vs, 1e-3, 0.01);
    let (matched, unmatched) = opt.set_decay_exempt(&[]);
    assert!(matched.is_empty() && unmatched.is_empty());
    assert!(opt.no_decay.is_empty());
    opt.zero_grad();
    backward_synthetic(&model.vs);
    opt.step();
    assert_eq!(
        model.state_digest(&[]),
        ONE_STEP_DIGEST,
        "one step with the switch empty must be the pre-switch step, bit for bit"
    );
    opt.zero_grad();
    backward_synthetic(&model.vs);
    opt.step();
    assert_eq!(model.state_digest(&[]), TWO_STEP_DIGEST);

    // the golden discriminates: with the switch set the same two steps differ
    let other = fixture_model();
    let mut opt = AdamW::new(&other.vs, 1e-3, 0.01);
    let (matched, _) = opt.set_decay_exempt(&corrected_patterns());
    assert!(matched.contains(&"greedy_tau".to_string()));
    for _ in 0..2 {
        opt.zero_grad();
        backward_synthetic(&other.vs);
        opt.step();
    }
    assert_ne!(other.state_digest(&[]), TWO_STEP_DIGEST);
    // and only the exempted parameters moved: the rest are bit-identical
    let re = fixture_model();
    let mut plain = AdamW::new(&re.vs, 1e-3, 0.01);
    for _ in 0..2 {
        plain.zero_grad();
        backward_synthetic(&re.vs);
        plain.step();
    }
    let exempt: Vec<&str> = matched.iter().map(String::as_str).collect();
    assert_eq!(other.state_digest(&exempt), re.state_digest(&exempt));
}

// ------------------------------------- (b) the arithmetic, computed by hand

/// A `VarStore` with the model's own naming: a scalar `greedy_tau`, LayerNorm
/// gains and biases, an attention `in_proj_bias`, the `context_bias` LINEAR
/// weight (whose name ends in `.weight`, not `bias`) and two plain weights.
fn tiny_store() -> (VarStore, Vec<(&'static str, f64)>) {
    let inits: Vec<(&'static str, f64)> = vec![
        ("greedy_tau", 10.0),
        ("candidate_norm.weight", 2.0),
        ("candidate_norm.bias", -1.0),
        ("blocks.0.norm_ff.weight", 3.0),
        ("blocks.0.context_attention.in_proj_bias", -2.0),
        ("blocks.0.context_bias.weight", 4.0),
        ("candidate_encoder.weight", 5.0),
        ("query_token", 6.0),
    ];
    let vs = VarStore::new(Device::Cpu);
    for (name, value) in &inits {
        let parts: Vec<&str> = name.split('.').collect();
        let mut path = vs.root();
        for p in &parts[..parts.len() - 1] {
            path = &path / *p;
        }
        let dims: Vec<i64> = match *name {
            "greedy_tau" => vec![],
            "candidate_encoder.weight" | "blocks.0.context_bias.weight" => vec![2, 2],
            "query_token" => vec![1, 1, 2],
            _ => vec![2],
        };
        let _ = path.var(parts[parts.len() - 1], &dims, nn::Init::Const(*value));
    }
    (vs, inits)
}

/// One AdamW step from rest: the decoupled decay multiplies the parameter, then
/// the Adam term at step 1 is `-lr * g / (|g| + eps)` (the bias corrections
/// cancel exactly at `t = 1`). `g = 2 * c * p` for the synthetic loss.
fn expected(p: f64, g: f64, lr: f64, wd: f64, decayed: bool) -> f64 {
    let after_decay = if decayed { p * (1.0 - lr * wd) } else { p };
    after_decay - lr * g / (g.abs() + 1e-8)
}

const LR: f64 = 0.1;
const WD: f64 = 0.5;

/// `(name, value before, value after, gradient)` after one step at LR/WD on
/// `tiny_store`, beside the names the switch exempted.
type Rows = Vec<(String, f64, f64, f64)>;
fn one_tiny_step(patterns: &[String]) -> (Rows, Vec<String>) {
    let (vs, inits) = tiny_store();
    let mut opt = AdamW::new(&vs, LR, WD);
    let (matched, _) = opt.set_decay_exempt(patterns);
    opt.zero_grad();
    backward_synthetic(&vs);
    opt.step();
    let sorted: Vec<String> = sorted_trainable(&vs).into_iter().map(|(n, _)| n).collect();
    let rows = inits
        .iter()
        .map(|(name, p0)| {
            let i = sorted.iter().position(|n| n == name).unwrap();
            let c = 1.0 + (i as f64) * 0.1;
            (name.to_string(), *p0, values(&vs, name)[0], 2.0 * c * p0)
        })
        .collect();
    (rows, matched)
}

#[test]
fn the_empty_switch_decays_every_parameter_by_the_decoupled_rule() {
    let (rows, matched) = one_tiny_step(&[]);
    assert!(matched.is_empty());
    for (name, p0, got, g) in rows {
        let with = expected(p0, g, LR, WD, true);
        let without = expected(p0, g, LR, WD, false);
        assert!(
            (got - with).abs() < 1e-4,
            "{name}: {got} is not the decayed step {with}"
        );
        assert!(
            (got - without).abs() > 1e-2,
            "{name}: the decay term is too small to discriminate ({got} vs {without})"
        );
    }
}

#[test]
fn the_switch_exempts_tau_and_the_norms_and_the_biases_but_not_a_weight() {
    let (rows, matched) = one_tiny_step(&corrected_patterns());
    let exempt = [
        "blocks.0.context_attention.in_proj_bias",
        "blocks.0.norm_ff.weight",
        "candidate_norm.bias",
        "candidate_norm.weight",
        "greedy_tau",
    ];
    assert_eq!(
        matched,
        exempt.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        "the exempt set, in the optimiser's own (sorted) order"
    );
    for (name, p0, got, g) in rows {
        let decayed = !exempt.contains(&name.as_str());
        let right = expected(p0, g, LR, WD, decayed);
        let wrong = expected(p0, g, LR, WD, !decayed);
        assert!(
            (got - right).abs() < 1e-4,
            "{name} (decayed={decayed}): {got} vs {right}"
        );
        assert!(
            (got - wrong).abs() > 1e-2,
            "{name}: the two predictions are too close to tell apart"
        );
    }
}

// ------------------------------------- what the intended value really matches

#[test]
fn the_intended_pattern_value_leaves_two_norm_gains_and_the_in_proj_biases_decayed() {
    let model = fixture_model();
    let mut opt = AdamW::new(&model.vs, 1e-3, 0.01);
    let (literal, unmatched) = opt.set_decay_exempt(&intended_patterns());
    assert!(unmatched.is_empty(), "{unmatched:?}");
    let (corrected, unmatched) = opt.set_decay_exempt(&corrected_patterns());
    assert!(unmatched.is_empty(), "{unmatched:?}");
    let missed: Vec<&String> = corrected.iter().filter(|n| !literal.contains(n)).collect();
    assert_eq!(
        missed,
        vec![
            "blocks.0.context_attention.in_proj_bias",
            "blocks.0.query_attention.in_proj_bias",
            "candidate_norm.weight",
            "context_norm.weight",
        ],
        "the dots in `*.norm*.weight` / `*.bias` are literal"
    );
    assert_eq!(literal.len(), 20);
    assert_eq!(corrected.len(), 24);
    // neither value touches a weight matrix, `context_bias` included
    for set in [&literal, &corrected] {
        for name in set.iter() {
            assert!(
                name == "greedy_tau" || name.ends_with("bias") || name.contains("norm"),
                "{name} is not a tau, a norm or a bias"
            );
        }
        assert!(!set.contains(&"blocks.0.context_bias.weight".to_string()));
        assert!(!set.contains(&"candidate_encoder.weight".to_string()));
        assert!(!set.contains(&"score_head.0.weight".to_string()));
    }
}

#[test]
fn relational_v6_adds_a_query_token_that_no_pattern_matches() {
    let config = ModelConfig::from_value(
        &json!({
            "hidden_dimension": 8, "self_attention_heads": 2,
            "feedforward_multiplier": 2, "score_hidden_dimension": 4,
            "coverage_hidden_dimension": 4, "traversal_blocks": 1,
            "dropout": 0.0, "greedy_prior": true, "feature_set": "relational-v6"
        }),
        8,
    )
    .unwrap();
    let model = Model::new(config, Device::Cpu).unwrap();
    let names: Vec<String> = sorted_trainable(&model.vs)
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert!(names.contains(&"query_token".to_string()));
    assert!(!names.contains(&"query_encoder.weight".to_string()));
    let mut opt = AdamW::new(&model.vs, 1e-3, 0.01);
    for patterns in [intended_patterns(), corrected_patterns()] {
        let (matched, _) = opt.set_decay_exempt(&patterns);
        assert!(
            !matched.contains(&"query_token".to_string()),
            "the learned query token is decayed by both pattern values; Part B \
             names it or it is decayed"
        );
    }
}

#[test]
fn a_pattern_matching_nothing_is_reported_and_not_an_error() {
    // the fixture config builds the model WITHOUT the greedy prior, so it has
    // no `greedy_tau` for the pattern to match
    let g: Value =
        serde_json::from_str(&std::fs::read_to_string(goldens().join("model.json")).unwrap())
            .unwrap();
    let mut m = g["model_config"].clone();
    m["greedy_prior"] = Value::Bool(false);
    let model = Model::new(ModelConfig::from_value(&m, 8).unwrap(), Device::Cpu).unwrap();
    let mut opt = AdamW::new(&model.vs, 1e-3, 0.01);
    let (matched, unmatched) = opt.set_decay_exempt(&corrected_patterns());
    assert_eq!(unmatched, vec!["greedy_tau".to_string()]);
    assert!(!matched.is_empty());
    assert!(!matched.contains(&"greedy_tau".to_string()));
}

// --------------------------------------------------------- (c) the resume

#[test]
fn the_optimiser_state_round_trips_with_the_switch_and_the_resume_keeps_it() {
    let path = std::env::temp_dir().join(format!("hf-adamw-{}.safetensors", std::process::id()));
    let patterns = corrected_patterns();

    // A: two steps without interruption
    let (straight, _) = tiny_store();
    let mut opt = AdamW::new(&straight, LR, WD);
    opt.set_decay_exempt(&patterns);
    for _ in 0..2 {
        opt.zero_grad();
        backward_synthetic(&straight);
        opt.step();
    }

    // B: one step, save, a fresh optimiser that loads and re-applies the switch
    let (resumed, _) = tiny_store();
    let mut opt = AdamW::new(&resumed, LR, WD);
    opt.set_decay_exempt(&patterns);
    opt.zero_grad();
    backward_synthetic(&resumed);
    opt.step();
    opt.save(&path).unwrap();
    let mut fresh = AdamW::new(&resumed, LR, WD);
    fresh.set_decay_exempt(&patterns);
    fresh.load(&path).unwrap();
    assert_eq!(fresh.step_count, 1);
    assert_eq!(fresh.no_decay, opt.no_decay, "load must not clear no_decay");
    fresh.zero_grad();
    backward_synthetic(&resumed);
    fresh.step();
    for (name, _) in tiny_store().1 {
        assert_eq!(
            values(&resumed, name),
            values(&straight, name),
            "{name}: the resumed run is not the unbroken one"
        );
    }

    // C: the same resume that forgets the switch decays tau again — the state
    // file does not carry the exemption, the config does
    let (forgotten, _) = tiny_store();
    let mut opt = AdamW::new(&forgotten, LR, WD);
    opt.set_decay_exempt(&patterns);
    opt.zero_grad();
    backward_synthetic(&forgotten);
    opt.step();
    opt.save(&path).unwrap();
    let mut fresh = AdamW::new(&forgotten, LR, WD);
    fresh.load(&path).unwrap();
    assert!(fresh.no_decay.is_empty());
    fresh.zero_grad();
    backward_synthetic(&forgotten);
    fresh.step();
    let tau_kept = values(&straight, "greedy_tau")[0];
    let tau_lost = values(&forgotten, "greedy_tau")[0];
    assert!(
        (tau_kept - tau_lost).abs() > 1e-2,
        "the test cannot tell a lost exemption apart ({tau_kept} vs {tau_lost})"
    );

    // the moments themselves are unchanged by the switch: at step 1 they depend
    // only on the gradients, which the decay does not enter
    let (with, _) = tiny_store();
    let mut a = AdamW::new(&with, LR, WD);
    a.set_decay_exempt(&patterns);
    a.zero_grad();
    backward_synthetic(&with);
    a.step();
    a.save(&path).unwrap();
    let saved_a: std::collections::HashMap<String, Tensor> = Tensor::read_safetensors(&path)
        .unwrap()
        .into_iter()
        .collect();
    let (without, _) = tiny_store();
    let mut b = AdamW::new(&without, LR, WD);
    b.zero_grad();
    backward_synthetic(&without);
    b.step();
    b.save(&path).unwrap();
    let saved_b: std::collections::HashMap<String, Tensor> = Tensor::read_safetensors(&path)
        .unwrap()
        .into_iter()
        .collect();
    assert_eq!(saved_a.len(), saved_b.len());
    for (key, ta) in &saved_a {
        assert!(
            ta.equal(&saved_b[key]),
            "{key}: the exemption changed a moment"
        );
    }
    let _ = std::fs::remove_file(&path);
}
