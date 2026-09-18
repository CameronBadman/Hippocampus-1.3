//! `training.clip_max_norm`: the gradient clip as a VALUE — `clip_grad_norm`'s
//! `Option<f64>` — rather than the constant `1.0` the runner passed at every
//! update before the key existed.
//!
//! What is asserted here:
//! 1. **the absent key is today's step, bit for bit** — `Some(1.0)` on the
//!    fixture model reproduces the digests measured at commit `f85da2d`, before
//!    the switch existed (the command is named below the constants), and the
//!    clip demonstrably fires there: the pre-clip norm is 134.8, so the
//!    coefficient is 0.0074, not 1;
//! 2. `None` over the same two steps reaches a DIFFERENT state, and that state
//!    is the un-clipped two-step golden `tests/adamw.rs` measured for the
//!    optimiser alone — an independently written test whose constant this file
//!    reproduces by removing the clip;
//! 3. the clip's arithmetic hand-computed on a tiny `VarStore` and asserted on
//!    the GRADIENTS, where a clipped and an unclipped step genuinely differ:
//!    every gradient multiplied by `min(1, max_norm / (norm + 1e-6))` — the
//!    `1e-6` is the implementation's, and `torch.nn.utils.clip_grad_norm_`'s,
//!    not the idealised `max_norm / norm`;
//! 4. `None` scales nothing — every gradient bit-identical — and still returns
//!    the true norm, which is the number `updates.jsonl` logs as `grad_norm`;
//! 5. a clip above the norm also scales nothing, so the norm a run records is
//!    the same number whether the clip fired, did not fire, or was off.
//!
//! **A caveat recorded here rather than left for a reader to trip over.** AdamW's
//! step is nearly invariant under a uniform rescaling of the gradient — at
//! `t = 1` the update is `-lr * g / (|g| + 1e-8)`, and rescaling `g` cancels in
//! that ratio but for the epsilon — so a PARAMETER-level assertion after one
//! step does not distinguish a clipped run from an unclipped one; the test
//! below measures that it does not. The gradient assertions are this file's
//! wiring proof. The fixture digests discriminate only because the ratio is not
//! exactly invariant and f32 rounding over 170 parameters accumulates.

use std::path::PathBuf;

use hf_model::{clip_grad_norm, AdamW, Model, ModelConfig};
use serde_json::Value;
use tch::nn::{self, VarStore};
use tch::{Device, Kind, Tensor};

/// The digests of the fixture model after one and two AdamW steps at
/// `lr = 1e-3`, `weight_decay = 0.01` on the synthetic gradients of
/// `backward_synthetic`, **each step clipped at 1.0** — the runner's constant —
/// measured on the code at commit `f85da2d`, where `clip_grad_norm` took a bare
/// `f64` and the runner passed `1.0`. To re-measure: check that commit out and
/// run this file with `clip_grad_norm(&vs, 1.0)` in place of
/// `clip_grad_norm(&vs, Some(1.0))`, printing the two digests.
const ONE_STEP_CLIPPED: &str =
    "sha256:0da83052c34413f3606292d762ed113f52900986ae3cf3a032ece9ea0f8e112c";
const TWO_STEP_CLIPPED: &str =
    "sha256:6893dd99735e4d7d53599ba39edb3e192d1c6db06f9d14149cc3b90f45955e8a";
/// The same two steps with NO clip. This is `TWO_STEP_DIGEST` of
/// `tests/adamw.rs` verbatim — that file exercises the optimiser without ever
/// calling `clip_grad_norm`, so the two agree only if `None` really is "the
/// optimiser, untouched".
const TWO_STEP_UNCLIPPED: &str =
    "sha256:b0f30e0569797932367d9069f7db560e63357ea56d42e76e076fec747d2e487a";
/// The fixture's pre-clip norm at the first step, measured at the same commit:
/// far above 1.0, so the golden above is a golden of a clip that FIRED.
const FIXTURE_FIRST_NORM: f64 = 134.82879638671875;

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

/// `sum_i c_i * (p_i . p_i)` over every trainable parameter, sorted by name —
/// the same synthetic loss `tests/adamw.rs` uses, so the digests of the two
/// files are comparable: a gradient for EVERY parameter, `2 * c_i * p_i`,
/// deterministic.
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

fn flat(t: &Tensor) -> Vec<f64> {
    Vec::<f32>::try_from(t.detach().reshape([-1]))
        .unwrap()
        .into_iter()
        .map(|x| x as f64)
        .collect()
}

/// Every gradient, by name, in sorted order — read back exactly as f32 widened
/// to f64, so two reads of the same unscaled gradient compare equal.
fn grads(vs: &VarStore) -> Vec<(String, Vec<f64>)> {
    sorted_trainable(vs)
        .iter()
        .map(|(n, t)| (n.clone(), flat(&t.grad())))
        .collect()
}

/// The total norm as `torch.nn.utils.clip_grad_norm_` defines it, computed here
/// in f64 from the gradient values themselves rather than by the two-level
/// reduction the implementation uses: an independent arithmetic, agreeing only
/// to f32 precision, which is the point.
fn hand_norm(g: &[(String, Vec<f64>)]) -> f64 {
    g.iter()
        .flat_map(|(_, v)| v.iter())
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt()
}

// -------------------------------------- (1, 2) the fixture model's two steps

#[test]
fn the_absent_key_reproduces_the_clipped_step_of_the_runner_before_it() {
    let model = fixture_model();
    let mut opt = AdamW::new(&model.vs, 1e-3, 0.01);
    opt.zero_grad();
    backward_synthetic(&model.vs);
    let norm = clip_grad_norm(&model.vs, Some(1.0));
    assert!(
        (norm - FIXTURE_FIRST_NORM).abs() < 1e-3,
        "the pre-clip norm is {norm}, measured {FIXTURE_FIRST_NORM}"
    );
    assert!(
        norm > 1.0,
        "the golden must be a golden of a clip that fired (coefficient {})",
        1.0 / (norm + 1e-6)
    );
    opt.step();
    assert_eq!(
        model.state_digest(&[]),
        ONE_STEP_CLIPPED,
        "an absent training.clip_max_norm must be the clip of 1.0 the runner \
         always passed, bit for bit"
    );
    opt.zero_grad();
    backward_synthetic(&model.vs);
    let second = clip_grad_norm(&model.vs, Some(1.0));
    assert!(second > 1.0);
    opt.step();
    assert_eq!(model.state_digest(&[]), TWO_STEP_CLIPPED);
}

#[test]
fn null_reaches_the_unclipped_golden_that_the_optimisers_own_test_measured() {
    let model = fixture_model();
    let mut opt = AdamW::new(&model.vs, 1e-3, 0.01);
    let mut norms = Vec::new();
    for _ in 0..2 {
        opt.zero_grad();
        backward_synthetic(&model.vs);
        // the norm is computed and returned under None: it is what the run logs
        norms.push(clip_grad_norm(&model.vs, None));
        opt.step();
    }
    assert!(
        (norms[0] - FIXTURE_FIRST_NORM).abs() < 1e-3,
        "the pre-clip norm under null is {}, measured {FIXTURE_FIRST_NORM}",
        norms[0]
    );
    assert_eq!(
        model.state_digest(&[]),
        TWO_STEP_UNCLIPPED,
        "null must be the optimiser untouched — adamw.rs's own two-step golden"
    );
    assert_ne!(
        model.state_digest(&[]),
        TWO_STEP_CLIPPED,
        "and must not be the clipped state"
    );
    // a clip far above the norm is the same thing: nothing is scaled
    let other = fixture_model();
    let mut opt = AdamW::new(&other.vs, 1e-3, 0.01);
    for _ in 0..2 {
        opt.zero_grad();
        backward_synthetic(&other.vs);
        let n = clip_grad_norm(&other.vs, Some(1e6));
        assert!(
            n < 1e6,
            "the premise: {n} is below the clip, so it cannot fire"
        );
        opt.step();
    }
    assert_eq!(other.state_digest(&[]), TWO_STEP_UNCLIPPED);
}

// ------------------------------------- (3, 4, 5) the arithmetic, by hand

/// A `VarStore` with the model's own naming and values large enough that the
/// synthetic gradient's norm is far above the clip of 10 this file tests (it is
/// ≈ 51), so `min(1, 10 / norm)` is a coefficient near 0.19 and not near 1.
fn tiny_store() -> VarStore {
    let inits: Vec<(&'static str, f64)> = vec![
        ("greedy_tau", 10.0),
        ("candidate_norm.weight", 2.0),
        ("candidate_norm.bias", -1.0),
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
    vs
}

/// A tiny store with its synthetic gradients already computed, beside a copy of
/// those gradients taken before any clip touched them.
fn tiny_with_grads() -> (VarStore, Vec<(String, Vec<f64>)>) {
    let vs = tiny_store();
    backward_synthetic(&vs);
    let before = grads(&vs);
    (vs, before)
}

const MAX: f64 = 10.0;

#[test]
fn a_clip_below_the_norm_multiplies_every_gradient_by_max_over_norm() {
    let (vs, before) = tiny_with_grads();
    let want_norm = hand_norm(&before);
    assert!(
        want_norm > MAX,
        "the premise of this test: {want_norm} must exceed the clip {MAX}"
    );
    let returned = clip_grad_norm(&vs, Some(MAX));
    assert!(
        (returned - want_norm).abs() < 1e-5 * want_norm,
        "the returned pre-clip norm {returned} vs the hand norm {want_norm}"
    );
    // torch's coefficient, the 1e-6 included; `min(1, ...)` is not reached here
    let coef = MAX / (want_norm + 1e-6);
    assert!(coef < 1.0);
    let after = grads(&vs);
    let mut checked = 0;
    for ((name, b), (name2, a)) in before.iter().zip(after.iter()) {
        assert_eq!(name, name2);
        for (b, a) in b.iter().zip(a.iter()) {
            assert!(
                (a - b * coef).abs() <= 1e-5 * (1.0 + b.abs()),
                "{name}: {a} is not the clipped {} of {b}",
                b * coef
            );
            // and far from the wrong prediction — the unclipped gradient
            if b.abs() > 1e-6 {
                assert!(
                    (a - b).abs() > 0.5 * b.abs(),
                    "{name}: {a} is indistinguishable from the unclipped {b}"
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 17, "every gradient of the tiny store is non-zero");
}

#[test]
fn null_scales_nothing_and_still_returns_the_true_norm() {
    let (vs, before) = tiny_with_grads();
    let want_norm = hand_norm(&before);
    let returned = clip_grad_norm(&vs, None);
    assert_eq!(
        grads(&vs),
        before,
        "null must leave every gradient bit-identical"
    );
    assert!(
        (returned - want_norm).abs() < 1e-5 * want_norm,
        "the norm logged under null is {returned}, the true norm {want_norm}"
    );
    // the same number the clip returns from the same gradients: what a run
    // records in updates.jsonl does not depend on whether it clipped
    let (other, other_before) = tiny_with_grads();
    assert_eq!(other_before, before);
    assert_eq!(clip_grad_norm(&other, Some(MAX)), returned);
    // a clip above the norm is the third case, and also scales nothing
    let (third, third_before) = tiny_with_grads();
    assert_eq!(clip_grad_norm(&third, Some(1e6)), returned);
    assert_eq!(grads(&third), third_before);
}

const LR: f64 = 0.1;
const WD: f64 = 0.5;

/// One AdamW step from rest: the decoupled decay multiplies the parameter, then
/// the Adam term at step 1 is `-lr * g / (|g| + eps)` (the bias corrections
/// cancel exactly at `t = 1`), as `tests/adamw.rs` computes it.
fn expected(p: f64, g: f64, decayed: bool) -> f64 {
    let after_decay = if decayed { p * (1.0 - LR * WD) } else { p };
    after_decay - LR * g / (g.abs() + 1e-8)
}

#[test]
fn null_moves_the_parameters_by_the_unclipped_update() {
    let (vs, before) = tiny_with_grads();
    let mut opt = AdamW::new(&vs, LR, WD);
    // the gradients are already on the parameters; `zero_grad` would discard them
    let p0: Vec<(String, Vec<f64>)> = sorted_trainable(&vs)
        .iter()
        .map(|(n, t)| (n.clone(), flat(t)))
        .collect();
    clip_grad_norm(&vs, None);
    opt.step();
    let p1 = sorted_trainable(&vs);
    for (i, (name, t)) in p1.iter().enumerate() {
        let got = flat(t);
        for (j, got) in got.iter().enumerate() {
            let want = expected(p0[i].1[j], before[i].1[j], true);
            assert!(
                (got - want).abs() <= 1e-6 * (1.0 + want.abs()),
                "{name}[{j}]: {got} vs the unclipped prediction {want}"
            );
        }
    }
    // THE CAVEAT, measured rather than asserted from theory: the same step with
    // the clip at 10 lands on the same parameters to f32, because AdamW's step-1
    // update is the scale-invariant ratio `g / |g|`. A parameter-level test
    // therefore cannot tell a clipped step from an unclipped one; the gradient
    // assertions above are what pin the clip.
    let (clipped, _) = tiny_with_grads();
    let mut opt = AdamW::new(&clipped, LR, WD);
    clip_grad_norm(&clipped, Some(MAX));
    opt.step();
    for ((name, a), (_, b)) in sorted_trainable(&vs)
        .iter()
        .map(|(n, t)| (n.clone(), flat(t)))
        .zip(
            sorted_trainable(&clipped)
                .iter()
                .map(|(n, t)| (n.clone(), flat(t))),
        )
    {
        for (a, b) in a.iter().zip(b.iter()) {
            assert!(
                (a - b).abs() <= 1e-6 * (1.0 + a.abs()),
                "{name}: one step of AdamW is not scale-invariant after all \
                 ({a} vs {b}) — if this fails, the caveat in this file's header \
                 is wrong and the parameter-level test above has real force"
            );
        }
    }
}
