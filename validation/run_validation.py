#!/usr/bin/env python3
"""
Validation suite for bayes-tree-eng.py / bayes_engine.
Runs test YAMLs and verifies expected mathematical properties.

Usage: python validation/run_validation.py
"""

import sys
import os
import math
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
from bayes_engine import (
    sim_root, sts, validate_node, bayes_upd,
    to_lo, from_lo, post_to_lr, sample_lr, pct,
    collect, NodeResult, run_simulation,
)

VALDIR = os.path.dirname(os.path.abspath(__file__))
N_SIM = 10_000
random.seed(42)


def load(name):
    with open(os.path.join(VALDIR, name), encoding="utf-8") as f:
        return yaml.safe_load(f)


def simulate(data, n=N_SIM):
    results = [sim_root(data) for _ in range(n)]
    posteriors = [r[0] for r in results]
    return sts(posteriors), posteriors


def check(condition, msg):
    status = "\033[92mPASS\033[0m" if condition else "\033[91mFAIL\033[0m"
    print(f"  [{status}] {msg}")
    return condition


# ── Individual test functions ─────────────────────────────────────────────────

def test_01():
    """Point LR for: prior=0.5, LR=2 → posterior = 2/3"""
    d = load("test_01_point_lr_for.yaml")
    s, _ = simulate(d)
    expected = 2 / 3
    return check(abs(s["median"] - expected) < 0.005,
                 f"median={s['median']:.4f}, expected={expected:.4f}")


def test_02():
    """Point LR against: prior=0.5, LR=0.5 → posterior = 1/3"""
    d = load("test_02_point_lr_against.yaml")
    s, _ = simulate(d)
    expected = 1 / 3
    return check(abs(s["median"] - expected) < 0.005,
                 f"median={s['median']:.4f}, expected={expected:.4f}")


def test_03():
    """Point LR cancel: LR=2 + LR=0.5 → posterior = 0.5"""
    d = load("test_03_point_lr_cancel.yaml")
    s, _ = simulate(d)
    return check(abs(s["median"] - 0.5) < 0.005,
                 f"median={s['median']:.4f}, expected=0.5000")


def test_04():
    """Point LR combined: 2 × LR=2 → posterior = 4/5"""
    d = load("test_04_point_lr_combined.yaml")
    s, _ = simulate(d)
    expected = 4 / 5
    return check(abs(s["median"] - expected) < 0.005,
                 f"median={s['median']:.4f}, expected={expected:.4f}")


def test_05():
    """Point LR three: LR=3 + LR=4 + LR=0.25 → posterior = 3/4"""
    d = load("test_05_point_lr_three.yaml")
    s, _ = simulate(d)
    expected = 3 / 4
    return check(abs(s["median"] - expected) < 0.005,
                 f"median={s['median']:.4f}, expected={expected:.4f}")


def test_06():
    """Low prior: prior=0.01, LR=10 → posterior ≈ 0.0917"""
    d = load("test_06_prior_low.yaml")
    s, _ = simulate(d)
    expected = 0.01 * 10 / (0.01 * 10 + 0.99)
    return check(abs(s["median"] - expected) < 0.005,
                 f"median={s['median']:.4f}, expected={expected:.4f}")


def test_07():
    """High prior: prior=0.99, LR=0.1 → posterior ≈ 0.9083"""
    d = load("test_07_prior_high.yaml")
    s, _ = simulate(d)
    expected = 0.99 * 0.1 / (0.99 * 0.1 + 0.01)
    return check(abs(s["median"] - expected) < 0.005,
                 f"median={s['median']:.4f}, expected={expected:.4f}")


def test_08():
    """All-for evidence: posterior should be well above 0.5"""
    d = load("test_08_all_for.yaml")
    s, _ = simulate(d)
    return check(s["median"] > 0.85,
                 f"median={s['median']:.4f} > 0.85 (all evidence supporting)")


def test_09():
    """All-against evidence: posterior should be well below 0.5"""
    d = load("test_09_all_against.yaml")
    s, _ = simulate(d)
    return check(s["median"] < 0.01,
                 f"median={s['median']:.4f} < 0.01 (all evidence against)")


def test_10():
    """Neutral-only: posterior should stay near prior (0.5)"""
    d = load("test_10_neutral_only.yaml")
    s, _ = simulate(d)
    return check(abs(s["median"] - 0.5) < 0.10,
                 f"median={s['median']:.4f} ≈ 0.50 (neutral evidence)")


def test_11():
    """Uniform distribution: runs without error, produces valid results"""
    d = load("test_11_dist_uniform.yaml")
    s, samples = simulate(d)
    valid = all(0.0 <= x <= 1.0 for x in samples)
    return check(valid and s["std"] > 0,
                 f"uniform dist: median={s['median']:.4f}, std={s['std']:.4f}")


def test_12():
    """Beta distribution: runs without error, produces valid results"""
    d = load("test_12_dist_beta.yaml")
    s, samples = simulate(d)
    valid = all(0.0 <= x <= 1.0 for x in samples)
    return check(valid and s["std"] > 0,
                 f"beta dist: median={s['median']:.4f}, std={s['std']:.4f}")


def test_13():
    """Deep nesting: engine handles 4 levels without error"""
    d = load("test_13_deep_nesting.yaml")
    s, samples = simulate(d)
    valid = all(0.0 <= x <= 1.0 for x in samples)
    return check(valid and len(samples) == N_SIM,
                 f"deep nesting: median={s['median']:.4f}, {len(samples)} samples")


def test_14():
    """Many children: engine handles 8 branches"""
    d = load("test_14_many_children.yaml")
    s, samples = simulate(d)
    valid = all(0.0 <= x <= 1.0 for x in samples)
    return check(valid and len(samples) == N_SIM,
                 f"8 branches: median={s['median']:.4f}, {len(samples)} samples")


def test_15():
    """Wide range: extreme LR uncertainty produces high variance"""
    d = load("test_15_wide_range.yaml")
    s, _ = simulate(d)
    spread = s["p95"] - s["p5"]
    return check(spread > 0.30,
                 f"wide range: 90% CI spread={spread:.4f} > 0.30")


def test_16():
    """Narrow range: near-deterministic LR → low variance, ≈ cancels"""
    d = load("test_16_narrow_range.yaml")
    s, _ = simulate(d)
    return check(s["std"] < 0.01 and abs(s["median"] - 0.5) < 0.02,
                 f"narrow range: median={s['median']:.4f}, std={s['std']:.6f}")


def test_17():
    """Warning: evidence_type conflicts with LR → 2 warnings"""
    d = load("test_17_warn_type_conflict.yaml")
    warnings = validate_node(d)
    return check(len(warnings) == 2,
                 f"type-LR conflict: {len(warnings)} warnings (expected 2)")


def test_18():
    """Warning: lr_min > lr_max → warning about order"""
    d = load("test_18_warn_lr_order.yaml")
    warnings = validate_node(d)
    order_warns = [w for w in warnings if "order" in w.lower() or "lr_min" in w]
    return check(len(order_warns) >= 1,
                 f"lr_min > lr_max: {len(order_warns)} order warning(s)")


def test_19():
    """Warning: lr_min ≤ 0 → warning about non-positive LR"""
    d = load("test_19_warn_lr_zero.yaml")
    warnings = validate_node(d)
    zero_warns = [w for w in warnings if "0" in w or "zero" in w.lower() or "negative" in w.lower()]
    return check(len(zero_warns) >= 1,
                 f"lr_min=0: {len(zero_warns)} zero/negative warning(s)")


def test_20():
    """Default prior: no prior in YAML → defaults to 0.5, LR=2 → 2/3"""
    d = load("test_20_prior_default.yaml")
    s, _ = simulate(d)
    expected = 2 / 3
    return check(abs(s["median"] - expected) < 0.005,
                 f"median={s['median']:.4f}, expected={expected:.4f} (default prior)")


# ── Numerical stability tests (21–28) ────────────────────────────────────────

def test_21():
    """Prior=0.0 → clamp to 1e-12, posterior stays near 0."""
    d = load("test_21_prior_zero.yaml")
    s, samples = simulate(d)
    ok1 = all(0.0 <= x <= 1.0 for x in samples)
    ok2 = s["median"] < 0.01
    d2 = load("test_21b_prior_one.yaml")
    s2, samples2 = simulate(d2)
    ok3 = all(0.0 <= x <= 1.0 for x in samples2)
    ok4 = s2["median"] > 0.99
    return check(ok1 and ok2 and ok3 and ok4,
                 f"prior=0→{s['median']:.6f}, prior=1→{s2['median']:.6f}")


def test_22():
    """LR=0 and LR=−1 clamped via max(lr, 1e-12) → no crash."""
    d = load("test_22_lr_zero_negative.yaml")
    s, samples = simulate(d)
    valid = all(0.0 <= x <= 1.0 for x in samples)
    return check(valid and s["median"] < 0.01,
                 f"zero/neg LR: median={s['median']:.6f}, no crash")


def test_23():
    """Extreme LR: 1e-300 → posterior≈0, 1e+300 → posterior≈1."""
    d_small = load("test_23a_lr_extreme_small.yaml")
    s_small, _ = simulate(d_small)
    d_large = load("test_23b_lr_extreme_large.yaml")
    s_large, _ = simulate(d_large)
    return check(s_small["median"] < 1e-6 and s_large["median"] > 1.0 - 1e-6,
                 f"LR=1e-300→{s_small['median']:.2e}, "
                 f"LR=1e+300→{s_large['median']:.10f}")


def test_24():
    """LR=1.0 exactly → posterior equals prior (0.30)."""
    d = load("test_24_lr_identity.yaml")
    s, _ = simulate(d)
    return check(abs(s["median"] - 0.30) < 0.005,
                 f"LR=1 identity: median={s['median']:.4f}, expected=0.3000")


def test_25():
    """Round-trip: from_lo(to_lo(p)) ≈ p for many values."""
    test_vals = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
    max_err = max(abs(from_lo(to_lo(p)) - p) for p in test_vals)
    return check(max_err < 1e-10,
                 f"to_lo/from_lo round-trip: max error={max_err:.2e}")


def test_26():
    """Round-trip: post_to_lr(prior, bayes_upd(prior, lr)) ≈ lr."""
    combos = [(0.5, 2.0), (0.3, 5.0), (0.7, 0.1), (0.01, 10.0), (0.99, 0.01)]
    max_err = 0
    for prior, lr in combos:
        recovered = post_to_lr(prior, bayes_upd(prior, lr))
        max_err = max(max_err, abs(recovered - lr) / lr)
    return check(max_err < 1e-8,
                 f"post_to_lr round-trip: max relative error={max_err:.2e}")


def test_27():
    """from_lo at clamp boundaries: ±700, ±1000."""
    ok1 = from_lo(700) > 1.0 - 1e-10
    ok2 = from_lo(-700) < 1e-10
    ok3 = from_lo(1000) > 1.0 - 1e-10
    ok4 = from_lo(-1000) < 1e-10
    ok5 = 0.0 <= from_lo(700) <= 1.0
    ok6 = 0.0 <= from_lo(-700) <= 1.0
    return check(ok1 and ok2 and ok3 and ok4 and ok5 and ok6,
                 f"from_lo(700)={from_lo(700):.10f}, "
                 f"from_lo(-700)={from_lo(-700):.2e}")


def test_28():
    """Accumulation overflow/underflow: 20× LR=100 and 20× LR=0.01."""
    d_over = load("test_28a_accumulation_overflow.yaml")
    s_over, samp_over = simulate(d_over)
    d_under = load("test_28b_accumulation_underflow.yaml")
    s_under, samp_under = simulate(d_under)
    valid = (all(0.0 <= x <= 1.0 for x in samp_over) and
             all(0.0 <= x <= 1.0 for x in samp_under))
    return check(valid and s_over["median"] > 0.99 and s_under["median"] < 0.01,
                 f"overflow→{s_over['median']:.6f}, "
                 f"underflow→{s_under['median']:.2e}")


# ── Input validation & robustness tests (29–35) ──────────────────────────────

def test_29():
    """Empty children list → posterior equals prior."""
    d = load("test_29_empty_children.yaml")
    s, _ = simulate(d)
    return check(abs(s["median"] - 0.35) < 0.005,
                 f"empty children: median={s['median']:.4f}, expected=0.3500")


def test_30():
    """Prior out of [0,1] silently clamped, produces valid results."""
    d_above = load("test_30a_prior_above_one.yaml")
    s_above, samp_above = simulate(d_above)
    d_neg = load("test_30b_prior_negative.yaml")
    s_neg, samp_neg = simulate(d_neg)
    valid = (all(0.0 <= x <= 1.0 for x in samp_above) and
             all(0.0 <= x <= 1.0 for x in samp_neg))
    return check(valid,
                 f"prior=1.5→{s_above['median']:.4f}, "
                 f"prior=-0.5→{s_neg['median']:.6f} (clamped, valid)")


def test_31():
    """lr_min == lr_max behaves like point LR; prior=0.5, LR=2 → 2/3."""
    d = load("test_31_lr_degenerate.yaml")
    s, _ = simulate(d)
    expected = 2 / 3
    return check(abs(s["median"] - expected) < 0.005 and s["std"] < 0.005,
                 f"degenerate range: median={s['median']:.4f}, std={s['std']:.6f}")


def test_32():
    """Partial range spec: only lr_min or only lr_max set → runs without crash."""
    d_min = load("test_32a_partial_lr_min.yaml")
    s_min, samp_min = simulate(d_min)
    d_max = load("test_32b_partial_lr_max.yaml")
    s_max, samp_max = simulate(d_max)
    valid = (all(0.0 <= x <= 1.0 for x in samp_min) and
             all(0.0 <= x <= 1.0 for x in samp_max))
    return check(valid,
                 f"partial lr_min→{s_min['median']:.4f}, "
                 f"partial lr_max→{s_max['median']:.4f}")


def test_33():
    """Unknown lr_dist silently treated as log_uniform → valid results."""
    d = load("test_33_unknown_dist.yaml")
    s, samples = simulate(d)
    valid = all(0.0 <= x <= 1.0 for x in samples) and s["std"] > 0
    return check(valid,
                 f"unknown dist: median={s['median']:.4f}, std={s['std']:.4f}")


def test_34():
    """Point LR takes precedence over lr_min/lr_max → exact 2/3."""
    d = load("test_34_point_lr_precedence.yaml")
    s, _ = simulate(d)
    expected = 2 / 3
    return check(abs(s["median"] - expected) < 0.005 and s["std"] < 0.005,
                 f"point LR wins: median={s['median']:.4f}, std={s['std']:.6f}")


def test_35():
    """Validation warning in deeply nested child (3 levels deep)."""
    d = load("test_35_nested_warning.yaml")
    warnings = validate_node(d)
    nested = [w for w in warnings if "Bad grandchild" in w]
    return check(len(nested) == 1,
                 f"nested warning: {len(nested)} found in grandchild")


# ── Distribution edge cases (36–38) ──────────────────────────────────────────

def test_36():
    """Beta(1,1) ≈ uniform: compare medians within tolerance."""
    d_beta = load("test_36_beta_uniform_equiv.yaml")
    d_unif = {
        "node": "Uniform equiv", "prior": 0.5,
        "children": [{"node": "U", "lr_min": 1.0, "lr_max": 5.0,
                       "lr_dist": "uniform", "evidence_type": "for"}]
    }
    random.seed(42)
    s_beta, _ = simulate(d_beta, n=20_000)
    random.seed(42)
    s_unif, _ = simulate(d_unif, n=20_000)
    diff = abs(s_beta["median"] - s_unif["median"])
    return check(diff < 0.03,
                 f"beta(1,1) vs uniform: diff={diff:.4f} < 0.03")


def test_37():
    """Beta extreme params: U-shape and concentrated both produce valid output."""
    d = load("test_37_beta_extreme.yaml")
    s, samples = simulate(d)
    valid = all(0.0 <= x <= 1.0 for x in samples) and s["std"] > 0
    return check(valid,
                 f"beta extreme: median={s['median']:.4f}, std={s['std']:.4f}")


def test_38():
    """Log-uniform with lr_min=1e-15 → clamp guards work, no crash."""
    d = load("test_38_log_uniform_near_zero.yaml")
    s, samples = simulate(d)
    valid = all(0.0 <= x <= 1.0 for x in samples)
    return check(valid and s["median"] < 0.5,
                 f"near-zero lr_min: median={s['median']:.4f}")


# ── Mathematical property tests (39–43) ──────────────────────────────────────

def test_39():
    """Monotonicity: higher LR → higher posterior, higher prior → higher posterior."""
    lr_mono = (bayes_upd(0.5, 0.5) < bayes_upd(0.5, 1.0) < bayes_upd(0.5, 2.0)
               < bayes_upd(0.5, 10.0))
    prior_mono = (bayes_upd(0.1, 2.0) < bayes_upd(0.3, 2.0) < bayes_upd(0.5, 2.0)
                  < bayes_upd(0.7, 2.0) < bayes_upd(0.9, 2.0))
    return check(lr_mono and prior_mono,
                 f"LR monotone: {lr_mono}, prior monotone: {prior_mono}")


def test_40():
    """Symmetry: bayes_upd(0.5, LR) = 1 − bayes_upd(0.5, 1/LR)."""
    lrs = [0.1, 0.5, 2.0, 5.0, 10.0, 100.0]
    max_err = max(abs(bayes_upd(0.5, lr) - (1.0 - bayes_upd(0.5, 1.0 / lr)))
                  for lr in lrs)
    return check(max_err < 1e-10,
                 f"symmetry: max error={max_err:.2e}")


def test_41():
    """Commutativity: children order doesn't change posterior distribution."""
    base = {
        "node": "Test", "prior": 0.5,
        "children": [
            {"node": "A", "likelihood_ratio": 3.0, "evidence_type": "for"},
            {"node": "B", "lr_min": 0.1, "lr_max": 0.5, "evidence_type": "against"},
            {"node": "C", "lr_min": 1.5, "lr_max": 4.0, "evidence_type": "for"},
        ]
    }
    swapped = {
        "node": "Test", "prior": 0.5,
        "children": [base["children"][2], base["children"][0], base["children"][1]]
    }
    random.seed(99)
    s1, _ = simulate(base, n=5000)
    random.seed(99)
    s2, _ = simulate(swapped, n=5000)
    diff = abs(s1["median"] - s2["median"])
    return check(diff < 0.02,
                 f"commutativity: diff={diff:.4f} < 0.02")


def test_42():
    """Log-odds additivity: 2×LR=3 ≡ 1×LR=9."""
    d_two = {
        "node": "Two", "prior": 0.5,
        "children": [
            {"node": "A", "likelihood_ratio": 3.0, "evidence_type": "for"},
            {"node": "B", "likelihood_ratio": 3.0, "evidence_type": "for"},
        ]
    }
    d_one = {
        "node": "One", "prior": 0.5,
        "children": [
            {"node": "A", "likelihood_ratio": 9.0, "evidence_type": "for"},
        ]
    }
    s_two, _ = simulate(d_two)
    s_one, _ = simulate(d_one)
    return check(abs(s_two["median"] - s_one["median"]) < 0.005,
                 f"2×LR=3={s_two['median']:.4f}, "
                 f"1×LR=9={s_one['median']:.4f}")


def test_43():
    """Neutral identity: adding LR=1 child doesn't change posterior."""
    d_base = {
        "node": "Base", "prior": 0.5,
        "children": [
            {"node": "A", "likelihood_ratio": 2.0, "evidence_type": "for"},
        ]
    }
    d_with_neutral = {
        "node": "WithNeutral", "prior": 0.5,
        "children": [
            {"node": "A", "likelihood_ratio": 2.0, "evidence_type": "for"},
            {"node": "N", "likelihood_ratio": 1.0, "evidence_type": "neutral"},
        ]
    }
    s_base, _ = simulate(d_base)
    s_neut, _ = simulate(d_with_neutral)
    return check(abs(s_base["median"] - s_neut["median"]) < 0.005,
                 f"base={s_base['median']:.4f}, "
                 f"with LR=1={s_neut['median']:.4f}")


# ── run_simulation / collect / NodeResult tests (44–47) ──────────────────────

def test_44():
    """run_simulation full pipeline returns expected keys and valid stats."""
    d = load("test_01_point_lr_for.yaml")
    result = run_simulation(d, n_sim=2000)
    keys = {'warnings', 'posteriors', 'eff_lrs', 'stats', 'lr_stats',
            'tree', 'sensitivity', 'importance', 'baseline', 'prior', 'n_sim'}
    has_keys = keys.issubset(result.keys())
    valid_stats = (0.0 <= result['stats']['median'] <= 1.0 and
                   result['n_sim'] == 2000 and
                   len(result['posteriors']) == 2000)
    return check(has_keys and valid_stats,
                 f"run_simulation: keys={has_keys}, "
                 f"median={result['stats']['median']:.4f}")


def test_45():
    """run_simulation progress_callback is called correctly."""
    d = load("test_01_point_lr_for.yaml")
    calls = []
    def cb(current, total):
        calls.append((current, total))
    run_simulation(d, n_sim=2000, progress_callback=cb)
    has_final = any(c == t for c, t in calls)
    all_valid = all(0 <= c <= t for c, t in calls)
    return check(len(calls) > 0 and has_final and all_valid,
                 f"callback: {len(calls)} calls, final=(2000,2000): {has_final}")


def test_46():
    """collect() returns valid NodeResult tree with correct structure."""
    d = load("test_13_deep_nesting.yaml")
    prior = d.get('prior', 0.5)
    tree = collect(d, 1000, prior, is_root=True)
    ok_type = isinstance(tree, NodeResult)
    ok_name = tree.name == d['node']
    ok_children = len(tree.children) == len(d['children'])
    ok_range = 0.0 <= tree.med <= 1.0 and 0.0 <= tree.p5 <= tree.p95 <= 1.0
    # Check recursion: first child should have its own children
    ok_depth = len(tree.children[0].children) > 0 if tree.children else False
    return check(ok_type and ok_name and ok_children and ok_range and ok_depth,
                 f"collect tree: {len(tree.children)} children, "
                 f"median={tree.med:.4f}, depth verified")


def test_47():
    """run_simulation importance ranking sorted by |delta|, correct count."""
    d = load("test_14_many_children.yaml")
    result = run_simulation(d, n_sim=2000)
    imp = result['importance']
    ok_count = len(imp) == len(d['children'])
    ok_sorted = all(abs(imp[i]['delta']) >= abs(imp[i+1]['delta'])
                    for i in range(len(imp)-1))
    ok_fields = all('name' in e and 'delta' in e and 'evidence_type' in e
                    and 'median_without' in e for e in imp)
    return check(ok_count and ok_sorted and ok_fields,
                 f"importance: {len(imp)} items, sorted={ok_sorted}")


# ── Stress & degenerate tests (48–50) ────────────────────────────────────────

def test_48():
    """50 branches → no crash, valid posterior in [0,1]."""
    d = load("test_48_many_children_stress.yaml")
    s, samples = simulate(d, n=2000)
    valid = all(0.0 <= x <= 1.0 for x in samples)
    n_children = len(d['children'])
    return check(valid and n_children == 50,
                 f"50 branches: median={s['median']:.4f}, valid={valid}")


def test_49():
    """Single-child LOO: data_without has empty children → posterior=prior."""
    d = load("test_49_single_child_loo.yaml")
    result = run_simulation(d, n_sim=2000)
    imp = result['importance']
    ok_count = len(imp) == 1
    ok_without = abs(imp[0]['median_without'] - 0.5) < 0.02
    return check(ok_count and ok_without,
                 f"single-child LOO: without={imp[0]['median_without']:.4f}≈prior")


def test_50():
    """All point LRs → zero variance, identical posteriors."""
    d = load("test_50_all_point_zero_var.yaml")
    s, samples = simulate(d)
    all_same = all(abs(x - samples[0]) < 1e-12 for x in samples)
    return check(all_same and s["std"] < 1e-12,
                 f"all-point: std={s['std']:.2e}, all identical={all_same}")


# ── Runner ────────────────────────────────────────────────────────────────────

TESTS = [
    # Original suite (01–20)
    ("01 Point LR — single for",                test_01),
    ("02 Point LR — single against",            test_02),
    ("03 Point LR — cancelling evidence",       test_03),
    ("04 Point LR — combined for",              test_04),
    ("05 Point LR — three mixed",               test_05),
    ("06 Prior edge — very low (0.01)",          test_06),
    ("07 Prior edge — very high (0.99)",         test_07),
    ("08 All evidence FOR",                      test_08),
    ("09 All evidence AGAINST",                  test_09),
    ("10 Neutral only — posterior ≈ prior",      test_10),
    ("11 Distribution — uniform",               test_11),
    ("12 Distribution — beta",                   test_12),
    ("13 Structure — deep nesting (4 levels)",   test_13),
    ("14 Structure — many children (8)",         test_14),
    ("15 Edge — wide LR uncertainty",            test_15),
    ("16 Edge — narrow LR (near-deterministic)", test_16),
    ("17 Warning — type vs LR conflict",         test_17),
    ("18 Warning — lr_min > lr_max",             test_18),
    ("19 Warning — lr_min ≤ 0",                  test_19),
    ("20 Default prior (none specified)",         test_20),
    # Numerical stability (21–28)
    ("21 Numerical — prior=0 and prior=1",       test_21),
    ("22 Numerical — LR=0 and LR=−1",           test_22),
    ("23 Numerical — extreme LR 1e±300",         test_23),
    ("24 Numerical — LR=1 identity",             test_24),
    ("25 Numerical — to_lo/from_lo round-trip",  test_25),
    ("26 Numerical — post_to_lr round-trip",     test_26),
    ("27 Numerical — from_lo clamp ±700/±1000",  test_27),
    ("28 Numerical — accumulation overflow",     test_28),
    # Input validation (29–35)
    ("29 Input — empty children list",           test_29),
    ("30 Input — prior out of [0,1]",            test_30),
    ("31 Input — lr_min == lr_max degenerate",   test_31),
    ("32 Input — partial range (lr_min or lr_max only)", test_32),
    ("33 Input — unknown lr_dist",               test_33),
    ("34 Input — point LR precedence",           test_34),
    ("35 Input — nested validation warning",     test_35),
    # Distribution edges (36–38)
    ("36 Dist — beta(1,1) ≈ uniform",           test_36),
    ("37 Dist — beta extreme params",            test_37),
    ("38 Dist — log_uniform near-zero clamp",    test_38),
    # Mathematical properties (39–43)
    ("39 Math — monotonicity (LR & prior)",      test_39),
    ("40 Math — symmetry around 0.5",            test_40),
    ("41 Math — commutativity of children order", test_41),
    ("42 Math — log-odds additivity 2×3=9",      test_42),
    ("43 Math — neutral identity (LR=1)",        test_43),
    # run_simulation / collect (44–47)
    ("44 API — run_simulation full pipeline",    test_44),
    ("45 API — progress_callback",               test_45),
    ("46 API — collect() NodeResult tree",        test_46),
    ("47 API — importance ranking",              test_47),
    # Stress & degenerate (48–50)
    ("48 Stress — 50 branches",                  test_48),
    ("49 Stress — single-child LOO",             test_49),
    ("50 Stress — all point LRs zero variance",  test_50),
]


def main():
    print()
    print("\033[1m\033[96mBAYES-TREE ENGINE — VALIDATION SUITE\033[0m")
    print(f"\033[90mSimulations per test: {N_SIM:,}   Seed: 42\033[0m")
    print("─" * 55)

    passed = 0
    failed = 0
    failures = []

    for label, fn in TESTS:
        print(f"\n\033[1m{label}\033[0m")
        try:
            if fn():
                passed += 1
            else:
                failed += 1
                failures.append(label)
        except Exception as e:
            failed += 1
            failures.append(label)
            print(f"  [\033[91mERROR\033[0m] {e}")

    print()
    print("═" * 55)
    total = passed + failed
    color = "\033[92m" if failed == 0 else "\033[91m"
    print(f"{color}\033[1m{passed}/{total} passed, {failed} failed\033[0m")

    if failures:
        print("\nFailed tests:")
        for f in failures:
            print(f"  \033[91m✗\033[0m {f}")

    print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
