"""
Bayesian Decision Tree Engine
Core logic for Bayesian evidence tree simulation.

Copyright (c) 2026 Ari-Pekka Sihvonen
MIT License — see LICENSE file
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# Type alias for parsed YAML node dicts
NodeDict = dict[str, Any]


# ── Bayes math ────────────────────────────────────────────────────────────────

def to_lo(p: float) -> float:
    """Convert probability to log-odds."""
    p = max(1e-12, min(1 - 1e-12, p))
    return math.log(p / (1 - p))


def from_lo(lo: float) -> float:
    """Convert log-odds to probability."""
    return 1.0 / (1.0 + math.exp(-max(-700, min(700, lo))))


def bayes_upd(prior: float, lr: float) -> float:
    """Bayesian update: apply likelihood ratio to prior."""
    return from_lo(to_lo(prior) + math.log(max(lr, 1e-12)))


def post_to_lr(prior: float, posterior: float) -> float:
    """Inverse: given prior and posterior, what LR produced this?"""
    return math.exp(to_lo(posterior) - to_lo(prior))


def sample_lr(d: NodeDict) -> float:
    """Sample LR from point value or interval."""
    if 'likelihood_ratio' in d:
        return float(d['likelihood_ratio'])
    lo = float(d.get('lr_min', 1.0))
    hi = float(d.get('lr_max', 1.0))
    dist = d.get('lr_dist', 'log_uniform')
    if dist == 'uniform':
        return random.uniform(lo, hi)
    elif dist == 'beta':
        t = random.betavariate(float(d.get('lr_alpha', 2)),
                               float(d.get('lr_beta', 2)))
        return lo + t * (hi - lo)
    else:  # log_uniform
        return math.exp(random.uniform(
            math.log(max(lo, 1e-12)),
            math.log(max(hi, 1e-12))
        ))


# ── Validation ────────────────────────────────────────────────────────────────

def validate_node(data: NodeDict, path: str = "root") -> list[str]:
    """Validate node for logical consistency.

    Raises ValueError if a node has both children and LR parameters
    (internal nodes must not carry evidence — only leaves do).
    """
    warnings: list[str] = []
    has_children = bool(data.get('children'))
    has_lr = any(k in data for k in ('lr_min', 'lr_max', 'likelihood_ratio'))

    if has_children and has_lr:
        raise ValueError(
            f"Node '{data['node']}' has children and an LR — internal nodes "
            f"must not carry evidence.  Move the LR to a leaf, or remove "
            f"the children."
        )

    if not has_children:
        et = data.get('evidence_type', 'neutral')
        lrpt = data.get('likelihood_ratio', None)
        lrlo = data.get('lr_min', None)
        lrhi = data.get('lr_max', None)

        if lrpt is not None:
            lr_center: Optional[float] = float(lrpt)
        elif lrlo is not None and lrhi is not None:
            lr_center = math.exp((math.log(max(float(lrlo), 1e-12)) +
                                  math.log(max(float(lrhi), 1e-12))) / 2)
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


# ── Leaf collection ───────────────────────────────────────────────────────────

def collect_leaves(data: NodeDict) -> list[NodeDict]:
    """Recursively collect all leaf nodes (nodes without children)."""
    children = data.get('children', [])
    if not children:
        return [data]
    leaves: list[NodeDict] = []
    for child in children:
        leaves.extend(collect_leaves(child))
    return leaves


def _collect_subtree_leaves(data: NodeDict) -> list[NodeDict]:
    """Collect leaf nodes from a subtree (excluding the root's own node)."""
    return collect_leaves(data)


# ── Simulation ────────────────────────────────────────────────────────────────

def sim_root(data: NodeDict) -> tuple[float, float]:
    """
    Flat leaf-only posterior computation.

    Collects all leaf nodes, samples one LR per leaf, sums log-LRs,
    and applies once to the root prior.  Internal nodes are pure
    groupers and carry no evidence.

    Supports correlation_group on leaves for dependent evidence
    (Gaussian copula).

    Returns (posterior, effective_lr).
    """
    prior = data.get('prior', 0.5)
    leaves = collect_leaves(data)
    if not leaves:
        return prior, 1.0

    base_lo = to_lo(prior)

    # Check for correlation groups among leaves
    groups: dict[str, list[int]] = {}
    for i, leaf in enumerate(leaves):
        grp = leaf.get('correlation_group')
        if grp is not None:
            groups.setdefault(str(grp), []).append(i)

    corr_groups = {k: v for k, v in groups.items() if len(v) >= 2}

    if not corr_groups:
        # Fast path: all independent
        total_log_lr = sum(
            math.log(max(sample_lr(leaf), 1e-12)) for leaf in leaves
        )
        posterior = from_lo(base_lo + total_log_lr)
        return posterior, math.exp(total_log_lr)

    # Correlated path: Gaussian copula for grouped leaves
    n = len(leaves)
    z = [random.gauss(0, 1) for _ in range(n)]

    for _grp_name, indices in corr_groups.items():
        rho = float(leaves[indices[0]].get('rho', 0.5))
        anchor = indices[0]
        for member in indices[1:]:
            z[member] = (rho * z[anchor]
                         + math.sqrt(max(0, 1 - rho * rho)) * z[member])

    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

    total_log_lr = 0.0
    for i, leaf in enumerate(leaves):
        grp = leaf.get('correlation_group')
        if grp is not None and str(grp) in corr_groups:
            lr = _sample_lr_quantile(leaf, _norm_cdf(z[i]))
        else:
            lr = sample_lr(leaf)
        total_log_lr += math.log(max(lr, 1e-12))

    posterior = from_lo(base_lo + total_log_lr)
    return posterior, math.exp(total_log_lr)


def _sample_lr_quantile(d: NodeDict, u: float) -> float:
    """Sample LR using a uniform quantile u ∈ [0,1] instead of random."""
    if 'likelihood_ratio' in d:
        return float(d['likelihood_ratio'])
    lo = float(d.get('lr_min', 1.0))
    hi = float(d.get('lr_max', 1.0))
    dist = d.get('lr_dist', 'log_uniform')
    if dist == 'uniform':
        return lo + u * (hi - lo)
    elif dist == 'beta':
        alpha = float(d.get('lr_alpha', 2))
        beta_p = float(d.get('lr_beta', 2))
        t = _beta_inv(u, alpha, beta_p)
        return lo + t * (hi - lo)
    else:  # log_uniform
        log_lo = math.log(max(lo, 1e-12))
        log_hi = math.log(max(hi, 1e-12))
        return math.exp(log_lo + u * (log_hi - log_lo))


def _beta_inv(u: float, a: float, b: float) -> float:
    """Approximate beta quantile via bisection."""
    u = max(1e-10, min(1 - 1e-10, u))
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if _beta_cdf_approx(mid, a, b) < u:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _beta_cdf_approx(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta via simple numerical integration."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    n_steps = 200
    dx = x / n_steps
    total = 0.0
    for i in range(n_steps):
        t = (i + 0.5) * dx
        total += t ** (a - 1) * (1 - t) ** (b - 1) * dx
    # Normalize by B(a, b)
    b_full = 0.0
    dx_full = 1.0 / n_steps
    for i in range(n_steps):
        t = (i + 0.5) * dx_full
        b_full += t ** (a - 1) * (1 - t) ** (b - 1) * dx_full
    return total / max(b_full, 1e-30)


# ── Statistics ────────────────────────────────────────────────────────────────

def pct(data: list[float], p: float) -> float:
    """Compute percentile from sorted data."""
    sd = sorted(data)
    idx = (len(sd) - 1) * p / 100
    lo, hi = int(idx), math.ceil(idx)
    return sd[lo] if lo == hi else sd[lo] * (hi - idx) + sd[hi] * (idx - lo)


def sts(samples: list[float]) -> dict[str, float]:
    """Compute summary statistics."""
    n = len(samples)
    m = sum(samples) / n
    return dict(
        mean=m,
        median=pct(samples, 50),
        std=math.sqrt(sum((x - m) ** 2 for x in samples) / n),
        p5=pct(samples, 5),
        p95=pct(samples, 95),
        min=min(samples),
        max=max(samples),
    )


# ── Node result data class ───────────────────────────────────────────────────

@dataclass
class NodeResult:
    """Statistics for a single node in the evidence tree."""
    name: str
    etype: str
    lr_min: float
    lr_max: float
    lr_pt: Optional[float]
    lr_derived: bool
    prior: float
    med: float
    p5: float
    p95: float
    children: list['NodeResult'] = field(default_factory=list)


def collect(data: NodeDict, n_sim: int, prior: float,
            is_root: bool = False) -> NodeResult:
    """Collect statistics for tree view.

    - Root: runs full sim_root (all leaves).
    - Internal nodes: compute subtree contribution (their leaves only)
      with root prior as input — purely for human insight.
    - Leaf nodes: single-LR Bayesian update from root prior.
    """
    lrpt = data.get('likelihood_ratio', None)
    has_children = bool(data.get('children'))

    if is_root:
        results = [sim_root(data) for _ in range(n_sim)]
        posteriors = [r[0] for r in results]
        eff_lrs = [r[1] for r in results]
        s = sts(posteriors)
        lr_s = sts(eff_lrs)
        derived_min = lr_s['p5']
        derived_max = lr_s['p95']
        nr = NodeResult(
            name=data['node'], etype='neutral',
            lr_min=derived_min, lr_max=derived_max, lr_pt=None,
            lr_derived=True,
            prior=prior, med=s['median'], p5=s['p5'], p95=s['p95']
        )
        for child in data.get('children', []):
            nr.children.append(collect(child, n_sim, prior, is_root=False))

    elif has_children:
        # Internal node — subtree contribution
        leaves = collect_leaves(data)
        base_lo = to_lo(prior)
        posteriors = []
        eff_lrs = []
        for _ in range(n_sim):
            total_log_lr = sum(
                math.log(max(sample_lr(lf), 1e-12)) for lf in leaves
            )
            posteriors.append(from_lo(base_lo + total_log_lr))
            eff_lrs.append(math.exp(total_log_lr))
        s = sts(posteriors)
        lr_s = sts(eff_lrs)
        nr = NodeResult(
            name=data['node'],
            etype=data.get('evidence_type', 'neutral'),
            lr_min=lr_s['p5'], lr_max=lr_s['p95'], lr_pt=None,
            lr_derived=True,
            prior=prior, med=s['median'], p5=s['p5'], p95=s['p95']
        )
        for child in data.get('children', []):
            nr.children.append(collect(child, n_sim, prior, is_root=False))

    else:
        # Leaf node — direct evidence
        lrlo = float(data.get('lr_min', lrpt or 1.0))
        lrhi = float(data.get('lr_max', lrpt or 1.0))
        samples = [bayes_upd(prior, sample_lr(data)) for _ in range(n_sim)]
        s = sts(samples)
        nr = NodeResult(
            name=data['node'], etype=data.get('evidence_type', 'neutral'),
            lr_min=lrlo, lr_max=lrhi, lr_pt=lrpt,
            lr_derived=False,
            prior=prior, med=s['median'], p5=s['p5'], p95=s['p95']
        )

    return nr


# ── Full simulation run ──────────────────────────────────────────────────────

def run_simulation(data: NodeDict, n_sim: int = 10000,
                   progress_callback: Optional[Callable[[int, int], None]] = None
                   ) -> dict[str, Any]:
    """
    Run a complete simulation and return all results.

    Args:
        data: Parsed YAML data dict
        n_sim: Number of Monte Carlo simulations
        progress_callback: Optional callable(current, total) for progress updates

    Returns dict with keys:
        posteriors, eff_lrs, stats, lr_stats, tree, sensitivity, importance
    """
    # Validate
    warnings = validate_node(data)

    # Simulate
    posteriors = []
    eff_lrs = []
    for i in range(n_sim):
        post, eff_lr = sim_root(data)
        posteriors.append(post)
        eff_lrs.append(eff_lr)
        if progress_callback and i % 500 == 0:
            progress_callback(i, n_sim)

    if progress_callback:
        progress_callback(n_sim, n_sim)

    s = sts(posteriors)
    s_lr = sts(eff_lrs)

    # Tree
    prior = data.get('prior', 0.5)
    tree = collect(data, min(2000, n_sim), prior, is_root=True)

    # Sensitivity
    sensitivity = {
        'p_lt_5': sum(x < 0.05 for x in posteriors) / n_sim,
        'p_lt_10': sum(x < 0.10 for x in posteriors) / n_sim,
        'p_gt_50': sum(x > 0.50 for x in posteriors) / n_sim,
    }

    # Leave-one-out importance
    baseline = s['median']
    n_loo = max(2000, n_sim // 5)
    children = data.get('children', [])
    importance = []
    for i, child in enumerate(children):
        data_without = {**data, 'children': [c for j, c in enumerate(children) if j != i]}
        samples_without = [sim_root(data_without)[0] for _ in range(n_loo)]
        med_without = sts(samples_without)['median']
        delta = med_without - baseline
        importance.append({
            'name': child['node'],
            'delta': delta,
            'evidence_type': child.get('evidence_type', 'neutral'),
            'median_without': med_without,
        })
    importance.sort(key=lambda x: abs(x['delta']), reverse=True)

    return {
        'warnings': warnings,
        'posteriors': posteriors,
        'eff_lrs': eff_lrs,
        'stats': s,
        'lr_stats': s_lr,
        'tree': tree,
        'sensitivity': sensitivity,
        'importance': importance,
        'baseline': baseline,
        'prior': prior,
        'n_sim': n_sim,
    }
