#!/usr/bin/env python3
"""
Bayesian Decision Tree + Monte Carlo Simulation
Copyright (c) 2026 Ari-Pekka Sihvonen
MIT License — see LICENSE file

Usage: python bayes-tree-eng.py [file.yaml] [simulations]

Architecture:
  - Root combines direct children as independent evidence (log-odds sum)
  - Root lr_min/lr_max is computed automatically from children's distribution
  - Each branch uses ONLY its own LR in the root combination
  - Children represent drill-down/explanation, not additions to the combination
  - Tree view shows chaining: parent's posterior -> child's posterior

YAML structure:
  lr_min: 0.01        # uncertainty interval (recommended)
  lr_max: 0.10
  lr_dist: log_uniform (default) | uniform | beta
  likelihood_ratio: 0.05  # or exact point value
  evidence_type: against | for | neutral
"""

import sys, math, random, yaml
from dataclasses import dataclass, field
from typing import Optional

class C:
    RESET="\033[0m"; BOLD="\033[1m"; RED="\033[91m"; GREEN="\033[92m"
    YELLOW="\033[93m"; CYAN="\033[96m"; GRAY="\033[90m"; WHITE="\033[97m"

# ── Bayes ─────────────────────────────────────────────────────────────────────
def to_lo(p):
    """Convert probability to log-odds."""
    p = max(1e-12, min(1-1e-12, p))
    return math.log(p / (1-p))

def from_lo(lo):
    """Convert log-odds to probability."""
    return 1.0 / (1.0 + math.exp(-max(-700, min(700, lo))))

def bayes_upd(prior, lr):
    """Bayesian update: apply likelihood ratio to prior."""
    return from_lo(to_lo(prior) + math.log(max(lr, 1e-12)))

def post_to_lr(prior, posterior):
    """Inverse: given prior and posterior, what LR produced this?"""
    return math.exp(to_lo(posterior) - to_lo(prior))

def sample_lr(d):
    """Sample LR from point value or interval."""
    if 'likelihood_ratio' in d:
        return float(d['likelihood_ratio'])
    lo   = float(d.get('lr_min', 1.0))
    hi   = float(d.get('lr_max', 1.0))
    dist = d.get('lr_dist', 'log_uniform')
    if dist == 'uniform':
        return random.uniform(lo, hi)
    elif dist == 'beta':
        t = random.betavariate(float(d.get('lr_alpha',2)), float(d.get('lr_beta',2)))
        return lo + t*(hi-lo)
    else:  # log_uniform – best for uncertain small values
        return math.exp(random.uniform(
            math.log(max(lo, 1e-12)),
            math.log(max(hi, 1e-12))
        ))

# ── Validation ────────────────────────────────────────────────────────────────
def validate_node(data, path="root"):
    """Validate node for logical consistency between evidence_type and LR."""
    warnings = []
    et   = data.get('evidence_type', 'neutral')
    lrpt = data.get('likelihood_ratio', None)
    lrlo = data.get('lr_min', None)
    lrhi = data.get('lr_max', None)

    if lrpt is not None:
        lr_center = float(lrpt)
    elif lrlo is not None and lrhi is not None:
        lr_center = math.exp((math.log(max(float(lrlo),1e-12)) +
                               math.log(max(float(lrhi),1e-12))) / 2)
    else:
        lr_center = None

    if lr_center is not None:
        if et == 'for' and lr_center < 1.0:
            warnings.append(
                f"  ⚠  '{data['node']}'\n"
                f"     evidence_type=for but LR mean={lr_center:.3f} < 1.0\n"
                f"     → LR below 1.0 means counter-evidence"
            )
        elif et == 'against' and lr_center > 1.0:
            warnings.append(
                f"  ⚠  '{data['node']}'\n"
                f"     evidence_type=against but LR mean={lr_center:.3f} > 1.0\n"
                f"     → LR above 1.0 means supporting evidence"
            )

    if lrlo is not None and lrhi is not None:
        if float(lrlo) > float(lrhi):
            warnings.append(
                f"  ⚠  '{data['node']}'\n"
                f"     lr_min={lrlo} > lr_max={lrhi} — order is wrong"
            )
    if lrlo is not None and float(lrlo) <= 0:
        warnings.append(
            f"  ⚠  '{data['node']}'\n"
            f"     lr_min={lrlo} ≤ 0 — LR cannot be negative or zero"
        )

    for child in data.get('children', []):
        warnings.extend(validate_node(child, path + " → " + data['node']))
    return warnings

# ── Simulation ────────────────────────────────────────────────────────────────
def sim_root(data):
    """
    Combine all top-level branches as independent evidence (log-odds sum).
    Returns (posterior, effective_lr) to compute root's LR distribution.
    """
    prior    = data.get('prior', 0.5)
    base_lo  = to_lo(prior)
    total_lo = base_lo
    for child in data.get('children', []):
        lr          = sample_lr(child)
        branch_post = bayes_upd(prior, lr)
        total_lo   += to_lo(branch_post) - base_lo
    posterior = from_lo(total_lo)
    # Effective LR: what single LR would produce the same posterior from prior
    eff_lr = post_to_lr(prior, posterior)
    return posterior, eff_lr

# ── Statistics ────────────────────────────────────────────────────────────────
def pct(data, p):
    """Compute percentile from sorted data."""
    sd = sorted(data); idx = (len(sd)-1)*p/100
    lo, hi = int(idx), math.ceil(idx)
    return sd[lo] if lo==hi else sd[lo]*(hi-idx)+sd[hi]*(idx-lo)

def sts(samples):
    """Compute summary statistics."""
    n=len(samples); m=sum(samples)/n
    return dict(mean=m, median=pct(samples,50),
                std=math.sqrt(sum((x-m)**2 for x in samples)/n),
                p5=pct(samples,5), p95=pct(samples,95),
                min=min(samples), max=max(samples))

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

# ── Tree view node statistics ─────────────────────────────────────────────────
@dataclass
class NR:
    name:str; etype:str
    lr_min:float; lr_max:float; lr_pt:Optional[float]
    lr_derived:bool          # True = computed from children's distribution
    prior:float; med:float; p5:float; p95:float
    children:list = field(default_factory=list)

def collect(data, n_sim, prior, is_root=False):
    """Collect statistics for tree view."""
    lrpt = data.get('likelihood_ratio', None)

    if is_root:
        # Root: run full simulation, collect effective LRs
        results   = [sim_root(data) for _ in range(n_sim)]
        posteriors = [r[0] for r in results]
        eff_lrs    = [r[1] for r in results]
        s      = sts(posteriors)
        lr_s   = sts(eff_lrs)
        # Use 5%–95% percentiles as LR interval
        derived_min = lr_s['p5']
        derived_max = lr_s['p95']
        nr = NR(
            name=data['node'], etype='neutral',
            lr_min=derived_min, lr_max=derived_max, lr_pt=None,
            lr_derived=True,
            prior=prior, med=s['median'], p5=s['p5'], p95=s['p95']
        )
        for child in data.get('children', []):
            nr.children.append(collect(child, n_sim, prior, is_root=False))
    else:
        lrlo = float(data.get('lr_min', lrpt or 1.0))
        lrhi = float(data.get('lr_max', lrpt or 1.0))
        samples = [bayes_upd(prior, sample_lr(data)) for _ in range(n_sim)]
        s = sts(samples)
        nr = NR(
            name=data['node'], etype=data.get('evidence_type','neutral'),
            lr_min=lrlo, lr_max=lrhi, lr_pt=lrpt,
            lr_derived=False,
            prior=prior, med=s['median'], p5=s['p5'], p95=s['p95']
        )
        for child in data.get('children', []):
            nr.children.append(collect(child, n_sim, s['median'], is_root=False))
    return nr

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
            lr_s = C.CYAN+f"LR=[{n.lr_min:.4f}–{n.lr_max:.4f}] (derived)"+C.RESET
        elif n.lr_pt is not None:
            lr_s = C.GRAY+f"LR={n.lr_pt:.2f}"+C.RESET
        else:
            lr_s = C.GRAY+f"LR=[{n.lr_min:.2f}–{n.lr_max:.2f}]"+C.RESET
        print(f"{prefix}{con}{C.BOLD}{n.name}{C.RESET} {es(n.etype)} {lr_s}")
        print(f"{prefix}{cpfx}  {n.prior:.2%} → {col}{n.med:.2%}{C.RESET} "
              f"{C.GRAY}[{n.p5:.2%}–{n.p95:.2%}]{C.RESET}")

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
    warnings = validate_node(data)
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
    print(C.BOLD+"TREE  (root LR computed from children's distribution)"+C.RESET)
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
