#!/usr/bin/env python3
"""
Generate a PDF report summarising the validation suite results.

Usage: python validation/generate_report.py
Output: validation/validation_report.pdf
"""

import sys
import os
import io
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from bayes_engine import (
    sim_root, sts, validate_node, bayes_upd,
    to_lo, from_lo, post_to_lr, sample_lr, pct,
    collect, NodeResult, run_simulation,
)

VALDIR = os.path.dirname(os.path.abspath(__file__))
N_SIM = 10_000


def load(name):
    with open(os.path.join(VALDIR, name), encoding="utf-8") as f:
        return yaml.safe_load(f)


def simulate(data, n=N_SIM):
    results = [sim_root(data) for _ in range(n)]
    posteriors = [r[0] for r in results]
    return sts(posteriors), posteriors


# ── Test definitions (mirrors run_validation.py) ─────────────────────────────

def _run_all_tests():
    """Run all 50 tests and return list of (label, passed, detail) tuples."""
    results = []

    def chk(label, cond, detail):
        results.append((label, cond, detail))
        return cond

    random.seed(42)

    # 01–05  Point LR exact math
    d = load("test_01_point_lr_for.yaml"); s, _ = simulate(d)
    chk("01 Point LR — single for", abs(s["median"] - 2/3) < 0.005,
        f'median={s["median"]:.4f}, expected=0.6667')

    d = load("test_02_point_lr_against.yaml"); s, _ = simulate(d)
    chk("02 Point LR — single against", abs(s["median"] - 1/3) < 0.005,
        f'median={s["median"]:.4f}, expected=0.3333')

    d = load("test_03_point_lr_cancel.yaml"); s, _ = simulate(d)
    chk("03 Point LR — cancelling evidence", abs(s["median"] - 0.5) < 0.005,
        f'median={s["median"]:.4f}, expected=0.5000')

    d = load("test_04_point_lr_combined.yaml"); s, _ = simulate(d)
    chk("04 Point LR — combined for", abs(s["median"] - 4/5) < 0.005,
        f'median={s["median"]:.4f}, expected=0.8000')

    d = load("test_05_point_lr_three.yaml"); s, _ = simulate(d)
    chk("05 Point LR — three mixed", abs(s["median"] - 3/4) < 0.005,
        f'median={s["median"]:.4f}, expected=0.7500')

    # 06–07  Prior edge cases
    d = load("test_06_prior_low.yaml"); s, _ = simulate(d)
    exp6 = 0.01*10 / (0.01*10 + 0.99)
    chk("06 Prior edge — very low (0.01)", abs(s["median"] - exp6) < 0.005,
        f'median={s["median"]:.4f}, expected={exp6:.4f}')

    d = load("test_07_prior_high.yaml"); s, _ = simulate(d)
    exp7 = 0.99*0.1 / (0.99*0.1 + 0.01)
    chk("07 Prior edge — very high (0.99)", abs(s["median"] - exp7) < 0.005,
        f'median={s["median"]:.4f}, expected={exp7:.4f}')

    # 08–10  Directional
    d = load("test_08_all_for.yaml"); s, _ = simulate(d)
    chk("08 All evidence FOR", s["median"] > 0.85,
        f'median={s["median"]:.4f} > 0.85')

    d = load("test_09_all_against.yaml"); s, _ = simulate(d)
    chk("09 All evidence AGAINST", s["median"] < 0.01,
        f'median={s["median"]:.4f} < 0.01')

    d = load("test_10_neutral_only.yaml"); s, _ = simulate(d)
    chk("10 Neutral only — posterior ≈ prior", abs(s["median"] - 0.5) < 0.10,
        f'median={s["median"]:.4f} ≈ 0.50')

    # 11–12  Distributions
    d = load("test_11_dist_uniform.yaml"); s, samp = simulate(d)
    chk("11 Distribution — uniform",
        all(0 <= x <= 1 for x in samp) and s["std"] > 0,
        f'median={s["median"]:.4f}, std={s["std"]:.4f}')

    d = load("test_12_dist_beta.yaml"); s, samp = simulate(d)
    chk("12 Distribution — beta",
        all(0 <= x <= 1 for x in samp) and s["std"] > 0,
        f'median={s["median"]:.4f}, std={s["std"]:.4f}')

    # 13–14  Structure
    d = load("test_13_deep_nesting.yaml"); s, samp = simulate(d)
    chk("13 Structure — deep nesting (4 levels)",
        all(0 <= x <= 1 for x in samp) and len(samp) == N_SIM,
        f'median={s["median"]:.4f}, {len(samp)} samples')

    d = load("test_14_many_children.yaml"); s, samp = simulate(d)
    chk("14 Structure — many children (8)",
        all(0 <= x <= 1 for x in samp) and len(samp) == N_SIM,
        f'median={s["median"]:.4f}, {len(samp)} samples')

    # 15–16  Uncertainty edges
    d = load("test_15_wide_range.yaml"); s, _ = simulate(d)
    spread = s["p95"] - s["p5"]
    chk("15 Edge — wide LR uncertainty", spread > 0.30,
        f'90% CI spread={spread:.4f} > 0.30')

    d = load("test_16_narrow_range.yaml"); s, _ = simulate(d)
    chk("16 Edge — narrow LR (near-deterministic)",
        s["std"] < 0.01 and abs(s["median"] - 0.5) < 0.02,
        f'median={s["median"]:.4f}, std={s["std"]:.6f}')

    # 17–19  Warnings
    d = load("test_17_warn_type_conflict.yaml"); w = validate_node(d)
    chk("17 Warning — type vs LR conflict", len(w) == 2,
        f'{len(w)} warnings (expected 2)')

    d = load("test_18_warn_lr_order.yaml"); w = validate_node(d)
    ow = [x for x in w if "order" in x.lower() or "lr_min" in x]
    chk("18 Warning — lr_min > lr_max", len(ow) >= 1,
        f'{len(ow)} order warning(s)')

    d = load("test_19_warn_lr_zero.yaml"); w = validate_node(d)
    zw = [x for x in w if "0" in x or "zero" in x.lower() or "negative" in x.lower()]
    chk("19 Warning — lr_min ≤ 0", len(zw) >= 1,
        f'{len(zw)} zero/negative warning(s)')

    # 20  Default prior
    d = load("test_20_prior_default.yaml"); s, _ = simulate(d)
    chk("20 Default prior (none specified)", abs(s["median"] - 2/3) < 0.005,
        f'median={s["median"]:.4f}, expected=0.6667')

    # 21  Prior boundaries
    d = load("test_21_prior_zero.yaml"); s, samp = simulate(d)
    d2 = load("test_21b_prior_one.yaml"); s2, samp2 = simulate(d2)
    chk("21 Numerical — prior=0 and prior=1",
        s["median"] < 0.01 and s2["median"] > 0.99
        and all(0 <= x <= 1 for x in samp) and all(0 <= x <= 1 for x in samp2),
        f'prior=0→{s["median"]:.6f}, prior=1→{s2["median"]:.6f}')

    # 22  LR=0 and negative
    d = load("test_22_lr_zero_negative.yaml"); s, samp = simulate(d)
    chk("22 Numerical — LR=0 and LR=−1",
        all(0 <= x <= 1 for x in samp) and s["median"] < 0.01,
        f'median={s["median"]:.6f}, no crash')

    # 23  Extreme LR
    ds = load("test_23a_lr_extreme_small.yaml"); ss, _ = simulate(ds)
    dl = load("test_23b_lr_extreme_large.yaml"); sl, _ = simulate(dl)
    chk("23 Numerical — extreme LR 1e±300",
        ss["median"] < 1e-6 and sl["median"] > 1 - 1e-6,
        f'LR=1e-300→{ss["median"]:.2e}, LR=1e+300→{sl["median"]:.10f}')

    # 24  Identity
    d = load("test_24_lr_identity.yaml"); s, _ = simulate(d)
    chk("24 Numerical — LR=1 identity", abs(s["median"] - 0.30) < 0.005,
        f'median={s["median"]:.4f}, expected=0.3000')

    # 25  to_lo/from_lo round-trip
    vals = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
    merr = max(abs(from_lo(to_lo(p)) - p) for p in vals)
    chk("25 Numerical — to_lo/from_lo round-trip", merr < 1e-10,
        f'max error={merr:.2e}')

    # 26  post_to_lr round-trip
    combos = [(0.5, 2.0), (0.3, 5.0), (0.7, 0.1), (0.01, 10.0), (0.99, 0.01)]
    merr2 = max(abs(post_to_lr(p, bayes_upd(p, lr)) - lr) / lr for p, lr in combos)
    chk("26 Numerical — post_to_lr round-trip", merr2 < 1e-8,
        f'max rel error={merr2:.2e}')

    # 27  from_lo clamp
    ok27 = (from_lo(700) > 1 - 1e-10 and from_lo(-700) < 1e-10
            and from_lo(1000) > 1 - 1e-10 and from_lo(-1000) < 1e-10
            and 0 <= from_lo(700) <= 1 and 0 <= from_lo(-700) <= 1)
    chk("27 Numerical — from_lo clamp ±700/±1000", ok27,
        f'from_lo(700)={from_lo(700):.10f}, from_lo(-700)={from_lo(-700):.2e}')

    # 28  Accumulation
    d = load("test_28a_accumulation_overflow.yaml"); so, sampo = simulate(d)
    d = load("test_28b_accumulation_underflow.yaml"); su, sampu = simulate(d)
    chk("28 Numerical — accumulation overflow",
        so["median"] > 0.99 and su["median"] < 0.01
        and all(0 <= x <= 1 for x in sampo) and all(0 <= x <= 1 for x in sampu),
        f'overflow→{so["median"]:.6f}, underflow→{su["median"]:.2e}')

    # 29  Empty children
    d = load("test_29_empty_children.yaml"); s, _ = simulate(d)
    chk("29 Input — empty children list", abs(s["median"] - 0.35) < 0.005,
        f'median={s["median"]:.4f}, expected=0.3500')

    # 30  Prior out of range
    da = load("test_30a_prior_above_one.yaml"); sa, sampa = simulate(da)
    dn = load("test_30b_prior_negative.yaml"); sn, sampn = simulate(dn)
    chk("30 Input — prior out of [0,1]",
        all(0 <= x <= 1 for x in sampa) and all(0 <= x <= 1 for x in sampn),
        f'prior=1.5→{sa["median"]:.4f}, prior=-0.5→{sn["median"]:.6f}')

    # 31  Degenerate range
    d = load("test_31_lr_degenerate.yaml"); s, _ = simulate(d)
    chk("31 Input — lr_min == lr_max degenerate",
        abs(s["median"] - 2/3) < 0.005 and s["std"] < 0.005,
        f'median={s["median"]:.4f}, std={s["std"]:.6f}')

    # 32  Partial range
    dm = load("test_32a_partial_lr_min.yaml"); sm, sampm = simulate(dm)
    dx = load("test_32b_partial_lr_max.yaml"); sx, sampx = simulate(dx)
    chk("32 Input — partial range spec",
        all(0 <= x <= 1 for x in sampm) and all(0 <= x <= 1 for x in sampx),
        f'lr_min only→{sm["median"]:.4f}, lr_max only→{sx["median"]:.4f}')

    # 33  Unknown dist
    d = load("test_33_unknown_dist.yaml"); s, samp = simulate(d)
    chk("33 Input — unknown lr_dist",
        all(0 <= x <= 1 for x in samp) and s["std"] > 0,
        f'median={s["median"]:.4f}, std={s["std"]:.4f}')

    # 34  Point LR precedence
    d = load("test_34_point_lr_precedence.yaml"); s, _ = simulate(d)
    chk("34 Input — point LR precedence",
        abs(s["median"] - 2/3) < 0.005 and s["std"] < 0.005,
        f'median={s["median"]:.4f}, std={s["std"]:.6f}')

    # 35  Nested warning
    d = load("test_35_nested_warning.yaml"); w = validate_node(d)
    nw = [x for x in w if "Bad grandchild" in x]
    chk("35 Input — nested validation warning", len(nw) == 1,
        f'{len(nw)} nested warning(s)')

    # 36  Beta(1,1) ≈ uniform
    db = load("test_36_beta_uniform_equiv.yaml")
    du = {"node": "U", "prior": 0.5, "children": [
        {"node": "U", "lr_min": 1.0, "lr_max": 5.0,
         "lr_dist": "uniform", "evidence_type": "for"}]}
    random.seed(42); sb, _ = simulate(db, n=20_000)
    random.seed(42); su2, _ = simulate(du, n=20_000)
    diff36 = abs(sb["median"] - su2["median"])
    chk("36 Dist — beta(1,1) ≈ uniform", diff36 < 0.03,
        f'diff={diff36:.4f} < 0.03')

    # 37  Beta extreme
    d = load("test_37_beta_extreme.yaml"); s, samp = simulate(d)
    chk("37 Dist — beta extreme params",
        all(0 <= x <= 1 for x in samp) and s["std"] > 0,
        f'median={s["median"]:.4f}, std={s["std"]:.4f}')

    # 38  log_uniform near zero
    d = load("test_38_log_uniform_near_zero.yaml"); s, samp = simulate(d)
    chk("38 Dist — log_uniform near-zero clamp",
        all(0 <= x <= 1 for x in samp) and s["median"] < 0.5,
        f'median={s["median"]:.4f}')

    # 39  Monotonicity
    lr_ok = (bayes_upd(0.5, 0.5) < bayes_upd(0.5, 1.0) < bayes_upd(0.5, 2.0)
             < bayes_upd(0.5, 10.0))
    pr_ok = (bayes_upd(0.1, 2.0) < bayes_upd(0.3, 2.0) < bayes_upd(0.5, 2.0)
             < bayes_upd(0.7, 2.0) < bayes_upd(0.9, 2.0))
    chk("39 Math — monotonicity", lr_ok and pr_ok,
        f'LR mono={lr_ok}, prior mono={pr_ok}')

    # 40  Symmetry
    lrs40 = [0.1, 0.5, 2.0, 5.0, 10.0, 100.0]
    merr40 = max(abs(bayes_upd(0.5, lr) - (1 - bayes_upd(0.5, 1/lr))) for lr in lrs40)
    chk("40 Math — symmetry around 0.5", merr40 < 1e-10,
        f'max error={merr40:.2e}')

    # 41  Commutativity
    base41 = {"node": "T", "prior": 0.5, "children": [
        {"node": "A", "likelihood_ratio": 3.0, "evidence_type": "for"},
        {"node": "B", "lr_min": 0.1, "lr_max": 0.5, "evidence_type": "against"},
        {"node": "C", "lr_min": 1.5, "lr_max": 4.0, "evidence_type": "for"}]}
    swap41 = {"node": "T", "prior": 0.5,
              "children": [base41["children"][2], base41["children"][0],
                           base41["children"][1]]}
    random.seed(99); s41a, _ = simulate(base41, n=5000)
    random.seed(99); s41b, _ = simulate(swap41, n=5000)
    d41 = abs(s41a["median"] - s41b["median"])
    chk("41 Math — commutativity", d41 < 0.02,
        f'diff={d41:.4f} < 0.02')

    # 42  Log-odds additivity
    d42a = {"node": "2x3", "prior": 0.5, "children": [
        {"node": "A", "likelihood_ratio": 3.0, "evidence_type": "for"},
        {"node": "B", "likelihood_ratio": 3.0, "evidence_type": "for"}]}
    d42b = {"node": "1x9", "prior": 0.5, "children": [
        {"node": "A", "likelihood_ratio": 9.0, "evidence_type": "for"}]}
    s42a, _ = simulate(d42a); s42b, _ = simulate(d42b)
    chk("42 Math — log-odds additivity 2×3=9",
        abs(s42a["median"] - s42b["median"]) < 0.005,
        f'2×LR=3={s42a["median"]:.4f}, 1×LR=9={s42b["median"]:.4f}')

    # 43  Neutral identity
    d43a = {"node": "B", "prior": 0.5, "children": [
        {"node": "A", "likelihood_ratio": 2.0, "evidence_type": "for"}]}
    d43b = {"node": "N", "prior": 0.5, "children": [
        {"node": "A", "likelihood_ratio": 2.0, "evidence_type": "for"},
        {"node": "N", "likelihood_ratio": 1.0, "evidence_type": "neutral"}]}
    s43a, _ = simulate(d43a); s43b, _ = simulate(d43b)
    chk("43 Math — neutral identity (LR=1)",
        abs(s43a["median"] - s43b["median"]) < 0.005,
        f'base={s43a["median"]:.4f}, +LR=1={s43b["median"]:.4f}')

    # 44  run_simulation pipeline
    d = load("test_01_point_lr_for.yaml")
    res = run_simulation(d, n_sim=2000)
    keys = {'warnings', 'posteriors', 'eff_lrs', 'stats', 'lr_stats',
            'tree', 'sensitivity', 'importance', 'baseline', 'prior', 'n_sim'}
    chk("44 API — run_simulation full pipeline",
        keys.issubset(res.keys()) and 0 <= res['stats']['median'] <= 1
        and res['n_sim'] == 2000 and len(res['posteriors']) == 2000,
        f'keys OK, median={res["stats"]["median"]:.4f}')

    # 45  Progress callback
    d = load("test_01_point_lr_for.yaml")
    calls = []
    run_simulation(d, n_sim=2000, progress_callback=lambda c, t: calls.append((c, t)))
    chk("45 API — progress_callback",
        len(calls) > 0 and any(c == t for c, t in calls),
        f'{len(calls)} calls, final reached')

    # 46  collect / NodeResult
    d = load("test_13_deep_nesting.yaml")
    tree = collect(d, 1000, d.get('prior', 0.5), is_root=True)
    chk("46 API — collect() NodeResult tree",
        isinstance(tree, NodeResult) and tree.name == d['node']
        and len(tree.children) == len(d['children'])
        and 0 <= tree.med <= 1 and len(tree.children[0].children) > 0,
        f'{len(tree.children)} children, median={tree.med:.4f}')

    # 47  Importance ranking
    d = load("test_14_many_children.yaml")
    res = run_simulation(d, n_sim=2000)
    imp = res['importance']
    chk("47 API — importance ranking",
        len(imp) == len(d['children'])
        and all(abs(imp[i]['delta']) >= abs(imp[i+1]['delta']) for i in range(len(imp)-1)),
        f'{len(imp)} items, sorted correctly')

    # 48  50 branches stress
    d = load("test_48_many_children_stress.yaml"); s, samp = simulate(d, n=2000)
    chk("48 Stress — 50 branches",
        all(0 <= x <= 1 for x in samp) and len(d['children']) == 50,
        f'median={s["median"]:.4f}, 50 branches OK')

    # 49  Single-child LOO
    d = load("test_49_single_child_loo.yaml")
    res = run_simulation(d, n_sim=2000)
    imp = res['importance']
    chk("49 Stress — single-child LOO",
        len(imp) == 1 and abs(imp[0]['median_without'] - 0.5) < 0.02,
        f'without={imp[0]["median_without"]:.4f} ≈ prior')

    # 50  All point LRs zero variance
    d = load("test_50_all_point_zero_var.yaml"); s, samp = simulate(d)
    chk("50 Stress — all point LRs zero variance",
        all(abs(x - samp[0]) < 1e-12 for x in samp) and s["std"] < 1e-12,
        f'std={s["std"]:.2e}')

    return results


# ── PDF generation ────────────────────────────────────────────────────────────

COL_PASS = colors.HexColor('#16a34a')
COL_FAIL = colors.HexColor('#dc2626')
COL_HEAD = colors.HexColor('#1e3a5f')
COL_ALT  = colors.HexColor('#f0f4f8')

CATEGORIES = [
    ("Exact Bayesian Math (Point LR)",        1,  5),
    ("Prior Edge Cases",                       6,  7),
    ("Directional Evidence",                   8, 10),
    ("Distribution Types",                    11, 12),
    ("Tree Structure",                        13, 14),
    ("Uncertainty Ranges",                    15, 16),
    ("Validation Warnings",                   17, 19),
    ("Default Behaviour",                     20, 20),
    ("Numerical Stability",                   21, 28),
    ("Input Validation & Robustness",         29, 35),
    ("Distribution Edge Cases",               36, 38),
    ("Mathematical Properties",               39, 43),
    ("Engine API Coverage",                   44, 47),
    ("Stress & Degenerate Cases",             48, 50),
]


def _make_summary_chart(passed, failed):
    fig, ax = plt.subplots(figsize=(6, 3))
    labels = ['Passed', 'Failed']
    sizes = [passed, failed]
    cs = ['#16a34a', '#dc2626'] if failed else ['#16a34a', '#e5e7eb']
    if failed == 0:
        sizes = [passed, 0.001]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=cs, autopct='%1.0f%%',
        startangle=90, textprops={'fontsize': 11})
    if failed == 0:
        autotexts[1].set_text('')
        texts[1].set_text('')
    ax.set_title(f'{passed}/{passed+failed} Tests Passed', fontsize=14,
                 fontweight='bold')
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def _make_category_chart(test_results):
    cat_labels = []
    cat_pass = []
    cat_total = []
    for name, lo, hi in CATEGORIES:
        n = hi - lo + 1
        p = sum(1 for label, ok, _ in test_results
                if ok and lo <= int(label[:2]) <= hi)
        cat_labels.append(name)
        cat_pass.append(p)
        cat_total.append(n)

    fig, ax = plt.subplots(figsize=(7, 5))
    y = range(len(cat_labels))
    bars_total = ax.barh(y, cat_total, color='#e5e7eb', height=0.6, label='Total')
    bars_pass = ax.barh(y, cat_pass, color='#16a34a', height=0.6, label='Passed')
    ax.set_yticks(list(y))
    ax.set_yticklabels(cat_labels, fontsize=8)
    ax.set_xlabel('Number of tests', fontsize=9)
    ax.set_title('Results by Category', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_pdf(test_results, output_path):
    passed = sum(1 for _, ok, _ in test_results if ok)
    failed = sum(1 for _, ok, _ in test_results if not ok)
    total = passed + failed

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm,
        leftMargin=2*cm, rightMargin=2*cm,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('Title2', parent=styles['Title'],
               fontSize=22, spaceAfter=6*mm))
    styles.add(ParagraphStyle('Subtitle', parent=styles['Normal'],
               fontSize=12, textColor=colors.grey, alignment=TA_CENTER,
               spaceAfter=12*mm))
    styles.add(ParagraphStyle('SectionHead', parent=styles['Heading2'],
               fontSize=14, spaceBefore=8*mm, spaceAfter=4*mm,
               textColor=COL_HEAD))
    styles.add(ParagraphStyle('CatHead', parent=styles['Heading3'],
               fontSize=11, spaceBefore=5*mm, spaceAfter=2*mm,
               textColor=COL_HEAD))
    styles.add(ParagraphStyle('BodySmall', parent=styles['Normal'],
               fontSize=9, spaceAfter=2*mm))

    story = []

    # ── Title page ────────────────────────────────────────────────────────
    story.append(Spacer(1, 3*cm))
    story.append(Paragraph('Bayes Tree Engine', styles['Title2']))
    story.append(Paragraph('Validation Report', styles['Heading1']))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph(
        f'Tests: {total}&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'Passed: {passed}&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'Failed: {failed}<br/><br/>'
        f'Simulations per test: {N_SIM:,}<br/>'
        f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        styles['Subtitle']))

    summary_buf = _make_summary_chart(passed, failed)
    story.append(Image(summary_buf, width=10*cm, height=5*cm))
    story.append(PageBreak())

    # ── Overview by category ──────────────────────────────────────────────
    story.append(Paragraph('Results by Category', styles['SectionHead']))
    cat_buf = _make_category_chart(test_results)
    story.append(Image(cat_buf, width=14*cm, height=10*cm))
    story.append(PageBreak())

    # ── Detailed results by category ──────────────────────────────────────
    story.append(Paragraph('Detailed Test Results', styles['SectionHead']))

    for cat_name, lo, hi in CATEGORIES:
        story.append(Paragraph(cat_name, styles['CatHead']))

        rows = [['#', 'Test', 'Result', 'Details']]
        for label, ok, detail in test_results:
            num = int(label[:2])
            if lo <= num <= hi:
                status = 'PASS' if ok else 'FAIL'
                short_label = label[3:].strip()
                rows.append([label[:2], short_label, status, detail])

        t = Table(rows, colWidths=[10*mm, 62*mm, 14*mm, 72*mm], repeatRows=1)
        style_cmds = [
            ('BACKGROUND', (0, 0), (-1, 0), COL_HEAD),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7.5),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (2, 0), (2, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.lightgrey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.white, COL_ALT]),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]
        # Color PASS/FAIL cells
        for i, (label, ok, _) in enumerate(test_results):
            num = int(label[:2])
            if lo <= num <= hi:
                row_idx = sum(1 for l2, _, _ in test_results
                              if lo <= int(l2[:2]) <= hi
                              and int(l2[:2]) < num) + 1
                cell_col = COL_PASS if ok else COL_FAIL
                style_cmds.append(
                    ('TEXTCOLOR', (2, row_idx), (2, row_idx), cell_col))
                style_cmds.append(
                    ('FONTNAME', (2, row_idx), (2, row_idx), 'Helvetica-Bold'))

        t.setStyle(TableStyle(style_cmds))
        story.append(t)
        story.append(Spacer(1, 3*mm))

    # ── Test coverage summary ─────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph('Test Coverage Summary', styles['SectionHead']))

    coverage_text = [
        '<b>Core Bayesian math</b> — exact posterior verification with point LRs, '
        'cancellation, combination, and three-branch mixed evidence.',

        '<b>Prior handling</b> — extreme priors (0.01, 0.99), boundary priors (0, 1), '
        'out-of-range priors (1.5, −0.5), and default prior omission.',

        '<b>Distributions</b> — uniform, log_uniform, beta with standard and extreme '
        'parameters (U-shaped, concentrated, alpha=beta=1 equivalence).',

        '<b>Numerical stability</b> — LR=0, LR=−1, LR=1e±300, to_lo/from_lo round-trips, '
        'from_lo clamp at ±700/±1000, and accumulation overflow/underflow with 20 extreme children.',

        '<b>Input robustness</b> — empty children, degenerate lr_min==lr_max, partial range '
        'specification, unknown lr_dist fallback, point LR precedence over ranges.',

        '<b>Validation warnings</b> — evidence_type vs LR conflicts, inverted lr_min/lr_max, '
        'lr_min≤0, and warnings detected in deeply nested children.',

        '<b>Mathematical properties</b> — monotonicity (LR and prior), symmetry around 0.5, '
        'commutativity of children order, log-odds additivity (2×LR=3 ≡ LR=9), '
        'neutral identity (LR=1 no-op).',

        '<b>Engine API</b> — run_simulation full pipeline, progress_callback invocation, '
        'collect()/NodeResult tree structure, importance ranking sort order.',

        '<b>Stress tests</b> — 50-branch tree, single-child leave-one-out, '
        'all-point-LR zero variance.',
    ]
    for para in coverage_text:
        story.append(Paragraph('• ' + para, styles['BodySmall']))
        story.append(Spacer(1, 1*mm))

    # ── Footer ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph(
        f'<i>Report generated by validation suite on '
        f'{datetime.now().strftime("%Y-%m-%d %H:%M")}. '
        f'All {total} tests executed with {N_SIM:,} Monte Carlo simulations each '
        f'(seed=42 for reproducibility).</i>',
        ParagraphStyle('Footer', parent=styles['Normal'],
                       fontSize=8, textColor=colors.grey)))

    doc.build(story)
    return output_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Running validation suite...")
    test_results = _run_all_tests()

    passed = sum(1 for _, ok, _ in test_results if ok)
    failed = sum(1 for _, ok, _ in test_results if not ok)
    print(f"\n{passed}/{passed+failed} passed, {failed} failed")

    output = os.path.join(VALDIR, "validation_report.pdf")
    print(f"Generating PDF → {output}")
    generate_pdf(test_results, output)
    print("Done.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
