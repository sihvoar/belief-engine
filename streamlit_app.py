#!/usr/bin/env python3
"""
Bayes Tree — Streamlit Web Demo

A zero-install interactive demo of the Bayesian evidence tree engine.
Deploy on HuggingFace Spaces or run locally: streamlit run streamlit_app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import yaml
import random

from bayes_tree import run_simulation, validate_node, sim_root, sts

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Bayes Tree",
    page_icon="🌳",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("🌳 Bayes Tree")
st.sidebar.markdown(
    "Structure arguments as Bayesian evidence trees and let Monte Carlo "
    "tell you what to believe."
)

n_sim = st.sidebar.slider("Simulations", 1000, 50000, 10000, step=1000)

# Example selector
EXAMPLES = {
    "Custom (paste YAML below)": None,
    "Napoleon poisoned?": "examples/napoleon.yaml",
    "God exists?": "examples/god.yaml",
    "Shroud of Turin authentic?": "examples/shroud.yaml",
    "Empty tomb is legend?": "examples/empty_grave.yaml",
    "Hitler murdered?": "examples/hitler.yaml",
    "Moses historical?": "examples/moses.yaml",
}

selected = st.sidebar.selectbox("Load example", list(EXAMPLES.keys()))

# ── Main content ─────────────────────────────────────────────────────────────

st.title("🌳 Bayesian Evidence Tree")
st.markdown("Paste a YAML evidence tree or select an example from the sidebar.")

# Load default YAML
default_yaml = """\
node: "Should I take this job offer?"
prior: 0.50
children:
  - node: "50% salary increase"
    lr_min: 2.0
    lr_max: 5.0
    evidence_type: for
  - node: "Longer commute (1.5 hours)"
    lr_min: 0.3
    lr_max: 0.7
    evidence_type: against
  - node: "Strong company growth"
    lr_min: 1.5
    lr_max: 3.0
    evidence_type: for
  - node: "Would lose current team"
    lr_min: 0.4
    lr_max: 0.8
    evidence_type: against
"""

if selected != "Custom (paste YAML below)" and EXAMPLES[selected]:
    try:
        with open(EXAMPLES[selected], encoding="utf-8") as f:
            default_yaml = f.read()
    except FileNotFoundError:
        st.sidebar.warning(f"Example file not found: {EXAMPLES[selected]}")

yaml_input = st.text_area("YAML Evidence Tree", value=default_yaml, height=350)

# ── Run simulation ───────────────────────────────────────────────────────────

if st.button("🎲 Run Simulation", type="primary"):
    try:
        data = yaml.safe_load(yaml_input)
    except yaml.YAMLError as e:
        st.error(f"YAML parse error: {e}")
        st.stop()

    if not data or "node" not in data:
        st.error("YAML must have a 'node' field at the top level.")
        st.stop()

    # Warnings
    warnings = validate_node(data)
    if warnings:
        with st.expander("⚠️ Warnings", expanded=True):
            for w in warnings:
                st.warning(w)

    # Run
    with st.spinner(f"Running {n_sim:,} Monte Carlo simulations..."):
        results = run_simulation(data, n_sim=n_sim)

    s = results["stats"]
    lr = results["lr_stats"]

    # ── Results ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"📊 Results: {data['node']}")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Prior", f"{results['prior']:.1%}")
    col2.metric("Posterior (median)", f"{s['median']:.2%}")
    col3.metric("90% CI", f"[{s['p5']:.2%} – {s['p95']:.2%}]")
    col4.metric("Effective LR", f"{lr['median']:.4f}")

    # Histogram
    st.subheader("Posterior Distribution")
    import numpy as np
    hist_data = np.array(results["posteriors"])
    st.bar_chart(
        np.histogram(hist_data, bins=30)[0],
        use_container_width=True,
    )

    # Sensitivity
    col_a, col_b, col_c = st.columns(3)
    sens = results["sensitivity"]
    col_a.metric("P(post < 5%)", f"{sens['p_lt_5']:.1%}")
    col_b.metric("P(post < 10%)", f"{sens['p_lt_10']:.1%}")
    col_c.metric("P(post > 50%)", f"{sens['p_gt_50']:.1%}")

    # Importance ranking
    if results["importance"]:
        st.subheader("🏆 Importance Ranking (leave-one-out)")
        for rank, item in enumerate(results["importance"], 1):
            arrow = "↑ raises" if item["delta"] > 0 else "↓ lowers"
            icon = "🟢" if item["delta"] > 0 else "🔴"
            st.markdown(
                f"**{rank}.** {icon} {arrow} by {abs(item['delta']):.4%} — "
                f"*{item['name']}*"
            )

    # Prior sweep
    st.subheader("📈 Prior Sensitivity Sweep")
    priors_range = [round(0.05 + i * 0.05, 2) for i in range(19)]
    sweep_medians = []
    sweep_p5 = []
    sweep_p95 = []
    for p in priors_range:
        sweep_data = {**data, "prior": p}
        posteriors = [sim_root(sweep_data)[0] for _ in range(2000)]
        ss = sts(posteriors)
        sweep_medians.append(ss["median"])
        sweep_p5.append(ss["p5"])
        sweep_p95.append(ss["p95"])

    import pandas as pd
    sweep_df = pd.DataFrame({
        "Prior": priors_range,
        "Posterior (median)": sweep_medians,
        "Lower 90% CI": sweep_p5,
        "Upper 90% CI": sweep_p95,
    }).set_index("Prior")
    st.line_chart(sweep_df, use_container_width=True)

    # JSON output
    with st.expander("📋 JSON Output"):
        from bayes_tree.cli import _format_json
        st.code(_format_json(results, data), language="json")

# ── Footer ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**[GitHub](https://github.com/sihvoar/belief-engine)** · "
    "MIT License · © 2026 Ari-Pekka Sihvonen"
)
