#!/usr/bin/env python3
"""
Bayesian Decision Tree + Monte Carlo Simulation
Copyright (c) 2026 Ari-Pekka Sihvonen
MIT License — see LICENSE file

Usage: python bayes-tree-eng.py [file.yaml] [simulations]

Architecture:
  - Only leaf nodes carry evidence (LR intervals)
  - Internal nodes are pure groupers — no LR allowed
  - All leaf log-LRs are summed once and applied to the root prior
  - Internal nodes display their subtree's aggregated contribution
  - The root prior is the only prior used

YAML structure:
  lr_min: 0.01        # uncertainty interval (recommended)
  lr_max: 0.10
  lr_dist: log_uniform (default) | uniform | beta
  likelihood_ratio: 0.05  # or exact point value
  evidence_type: against | for | neutral
"""

import sys, os, yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from bayes_tree import (
    to_lo, from_lo, bayes_upd, post_to_lr, sample_lr,
    validate_node, collect_leaves, sim_root, pct, sts, collect, NodeResult,
)

class C:
    RESET="\033[0m"; BOLD="\033[1m"; RED="\033[91m"; GREEN="\033[92m"
    YELLOW="\033[93m"; CYAN="\033[96m"; GRAY="\033[90m"; WHITE="\033[97m"

# ── Histogram ─────────────────────────────────────────────────────────────────
def histogram(samples, bins=20, width=38):
    """Print ASCII histogram of posterior distribution."""
    lo, hi = min(samples), max(samples)
    if abs(hi-lo) < 1e-9:
        col = C.GREEN if lo>=0.5 else (C.YELLOW if lo>=0.25 else C.RED)
        print(f"  All simulations: {col}{lo:.3%}{C.RESET}")
        return
    step = (hi-lo)/bins; counts=[0]*bins
    for s in samples:
        counts[min(int((s-lo)/step), bins-1)] += 1
    mx = max(counts)
    print(f"\n  {'':>9} {'0%':<4}{'':>{width//2-3}}{'100%':>5}")
    print(f"  {'':>9} {'─'*width}")
    for i,c in enumerate(counts):
        bw = int(c/mx*width) if mx else 0
        lb = lo+i*step; hb=lb+step; mid=(lb+hb)/2
        col= C.GREEN if mid>=0.5 else (C.YELLOW if mid>=0.25 else C.RED)
        bar= col+"█"*bw+C.RESET+"░"*(width-bw)
        print(f"  {lb:>6.2%}–{hb:<6.2%} {bar} {c}")

# ── Output ────────────────────────────────────────────────────────────────────
def pc(p):
    """Color code for probability."""
    if p>=0.75: return C.GREEN
    if p>=0.50: return C.YELLOW
    if p>=0.25: return C.RED
    return C.RED+C.BOLD

def es(et):
    """Evidence type symbol."""
    return {'for':     C.GREEN+'(+)'+C.RESET,
            'against': C.RED +'(-)'+C.RESET}.get(et, C.GRAY+'( )'+C.RESET)

def print_tree(n, prefix="", is_last=True, is_root=True):
    """Print the decision tree with statistics."""
    con  = "" if is_root else ("└── " if is_last else "├── ")
    cpfx = "" if is_root else ("    " if is_last else "│   ")
    col  = pc(n.med)

    if is_root:
        lr_tag = (C.CYAN + f"LR=[{n.lr_min:.4f}–{n.lr_max:.4f}]" +
                  C.GRAY + " (computed from children's distribution)" + C.RESET)
        print(f"{C.BOLD}{C.WHITE}{n.name}{C.RESET}")
        print(f"  Prior:    {C.CYAN}{n.prior:.1%}{C.RESET}")
        print(f"  Eff. LR:  {lr_tag}")
        print(f"  Median:   {col}{n.med:.3%}{C.RESET}  "
              f"{C.GRAY}90% CI [{n.p5:.3%}–{n.p95:.3%}]{C.RESET}")
        print()
    else:
        if n.lr_derived:
            lr_s = C.CYAN+f"LR=[{n.lr_min:.4f}–{n.lr_max:.4f}] (subtree contribution)"+C.RESET
        elif n.lr_pt is not None:
            lr_s = C.GRAY+f"LR={n.lr_pt:.2f}"+C.RESET
        else:
            lr_s = C.GRAY+f"LR=[{n.lr_min:.2f}–{n.lr_max:.2f}]"+C.RESET
        print(f"{prefix}{con}{C.BOLD}{n.name}{C.RESET} {es(n.etype)} {lr_s}")
        label = "subtree" if n.lr_derived else "prior"
        print(f"{prefix}{cpfx}  {n.prior:.2%} → {col}{n.med:.2%}{C.RESET} "
              f"{C.GRAY}[{n.p5:.2%}–{n.p95:.2%}] ({label}){C.RESET}")

    for i,child in enumerate(n.children):
        print_tree(child, prefix+cpfx, i==len(n.children)-1, False)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    fname = sys.argv[1] if len(sys.argv)>1 else "example.yaml"
    n_sim = int(sys.argv[2]) if len(sys.argv)>2 else 10_000

    try:
        with open(fname, encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"File not found: {fname}"); sys.exit(1)

    print()
    print(C.BOLD+C.CYAN+"BAYESIAN DECISION TREE  +  MONTE CARLO"+C.RESET)
    print(C.GRAY+f"File: {fname}   Simulations: {n_sim:,}"+C.RESET)
    print("─"*55)

    # 0. Validation
    try:
        warnings = validate_node(data)
    except ValueError as e:
        print(f"\n{C.RED}{C.BOLD}ERROR: {e}{C.RESET}")
        sys.exit(1)
    if warnings:
        print()
        print(C.YELLOW+C.BOLD+"WARNINGS — evidence_type vs LR conflict:"+C.RESET)
        for w in warnings:
            print(C.YELLOW+w+C.RESET)
        print()

    # 1. Simulate and collect effective LRs
    print()
    print(C.BOLD+"Simulating combined posterior..."+C.RESET)
    results    = [sim_root(data) for _ in range(n_sim)]
    posteriors = [r[0] for r in results]
    eff_lrs    = [r[1] for r in results]

    s    = sts(posteriors)
    s_lr = sts(eff_lrs)
    col  = pc(s['median'])

    print(f"\n  Median   : {col}{s['median']:.3%}{C.RESET}")
    print(f"  Mean     : {col}{s['mean']:.3%}{C.RESET}")
    print(f"  Std      : {C.GRAY}{s['std']:.3%}{C.RESET}")
    print(f"  90% CI   : {C.GRAY}[{s['p5']:.3%}–{s['p95']:.3%}]{C.RESET}")
    print(f"  Range    : {C.GRAY}[{s['min']:.3%}–{s['max']:.3%}]{C.RESET}")
    print(f"\n  Effective LR (90% CI): "
          f"{C.CYAN}[{s_lr['p5']:.4f}–{s_lr['p95']:.4f}]{C.RESET}")
    print(f"  Effective LR median:   {C.CYAN}{s_lr['median']:.4f}{C.RESET}")
    histogram(posteriors)

    # 2. Tree — root gets derived LR
    print()
    print("─"*55)
    print(C.BOLD+"TREE  (leaves carry evidence; internal nodes show subtree contribution)"+C.RESET)
    print("─"*55)
    print()
    prior   = data.get('prior', 0.5)
    root_nr = collect(data, 2000, prior, is_root=True)
    print_tree(root_nr)

    # 3. Sensitivity analysis
    print()
    print("─"*55)
    print(C.BOLD+"SENSITIVITY ANALYSIS"+C.RESET)
    print("─"*55)
    p5  = sum(x<0.05  for x in posteriors)/n_sim
    p10 = sum(x<0.10  for x in posteriors)/n_sim
    p50 = sum(x>0.50  for x in posteriors)/n_sim
    print(f"\n  P(posterior < 5%)  = {C.RED}{p5:.1%}{C.RESET}")
    print(f"  P(posterior < 10%) = {C.YELLOW}{p10:.1%}{C.RESET}")
    print(f"  P(posterior > 50%) = {C.GREEN}{p50:.1%}{C.RESET}")
    print()

    # 4. Leave-one-out importance ranking
    print("─"*55)
    print(C.BOLD+"IMPORTANCE RANKING  (leave-one-out)"+C.RESET)
    print("─"*55)
    print(C.GRAY+"How much does the posterior change if a branch is removed?"+C.RESET)

    baseline = s["median"]
    n_loo    = max(2000, n_sim // 5)
    impacts  = []

    children = data.get("children", [])
    for i, child in enumerate(children):
        data_without = {**data, "children": [c for j,c in enumerate(children) if j != i]}
        samples_without = [sim_root(data_without)[0] for _ in range(n_loo)]
        med_without = sts(samples_without)["median"]
        delta = med_without - baseline
        impacts.append((child["node"], delta, child.get("evidence_type","neutral"), med_without))

    impacts.sort(key=lambda x: abs(x[1]), reverse=True)

    print()
    max_delta = max(abs(d) for _,d,_,_ in impacts) or 1
    bar_w = 25

    for rank, (name, delta, etype, med_wo) in enumerate(impacts, 1):
        filled  = int(abs(delta) / max_delta * bar_w)
        bar_col = C.GREEN if delta > 0 else C.RED
        bar     = bar_col + "█"*filled + C.RESET + "░"*(bar_w-filled)
        arrow   = (C.GREEN+"↑ raises "+C.RESET) if delta > 0 else (C.RED+"↓ lowers "+C.RESET)
        esymbol = {"for":     C.GREEN+"(+)"+C.RESET,
                   "against": C.RED  +"(-)"+C.RESET}.get(etype, C.GRAY+"( )"+C.RESET)
        short = name if len(name) <= 38 else name[:35]+"..."
        print(f"  {rank:>2}. {bar} {arrow}{abs(delta):.4%}")
        print(f"      {esymbol} {C.BOLD}{short}{C.RESET}")
        print(f"      {C.GRAY}Without this: {med_wo:.4%}  (baseline: {baseline:.4%}){C.RESET}")
        print()

if __name__ == "__main__":
    main()
