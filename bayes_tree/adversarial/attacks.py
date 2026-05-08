"""
Attack classes for adversarial analysis of Bayesian evidence trees.

Each attack modifies the tree or simulation to challenge a specific assumption,
then measures the impact on the posterior.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional

from bayes_tree.engine import (
    NodeDict, run_simulation, sim_root, sts,
    to_lo, from_lo, bayes_upd, sample_lr, post_to_lr,
)


@dataclass
class AttackResult:
    """Result of applying a single attack."""
    attack_type: str
    description: str
    target: str
    original_median: float
    attacked_median: float
    delta: float
    plausibility: float
    details: dict[str, Any] = field(default_factory=dict)
    defenses: list[str] = field(default_factory=list)

    @property
    def flipped(self) -> bool:
        """Did the attack flip the conclusion across 50%?"""
        return (self.original_median >= 0.5) != (self.attacked_median >= 0.5)

    @property
    def severity(self) -> str:
        """Human-readable severity label."""
        d = abs(self.delta)
        if d > 0.20:
            return "critical"
        elif d > 0.10:
            return "high"
        elif d > 0.05:
            return "moderate"
        else:
            return "low"


class Attack:
    """Base class for adversarial attacks."""

    attack_type: str = "base"

    def generate(self, data: NodeDict, baseline_median: float,
                 n_sim: int = 5000) -> list[AttackResult]:
        """Generate all plausible attack variants and return results."""
        raise NotImplementedError


# ── CorrelationAttack ────────────────────────────────────────────────────────

def _sim_root_correlated(
    data: NodeDict,
    corr_pairs: list[tuple[int, int]],
    rho: float,
) -> tuple[float, float]:
    """
    Like sim_root but introduces correlation between specified branch pairs.

    Uses a Gaussian copula: generate correlated normals, transform to
    uniform, use as quantile inputs for LR sampling.
    """
    prior = data.get('prior', 0.5)
    children = data.get('children', [])
    n = len(children)
    if n == 0:
        return prior, 1.0

    # Generate independent standard normals
    z = [random.gauss(0, 1) for _ in range(n)]

    # Inject correlation: for each pair (i, j), mix z values
    for i, j in corr_pairs:
        if i < n and j < n:
            # Cholesky-like mixing: z_j' = rho * z_i + sqrt(1-rho^2) * z_j
            z[j] = rho * z[i] + math.sqrt(max(0, 1 - rho * rho)) * z[j]

    # Convert normals to uniform [0,1] via standard normal CDF approximation
    def norm_cdf(x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    uniforms = [norm_cdf(zi) for zi in z]

    # Sample LRs using the correlated uniforms as quantile inputs
    base_lo = to_lo(prior)
    total_lo = base_lo
    for idx, child in enumerate(children):
        u = uniforms[idx]
        lr = _sample_lr_from_quantile(child, u)
        branch_post = bayes_upd(prior, lr)
        total_lo += to_lo(branch_post) - base_lo

    posterior = from_lo(total_lo)
    eff_lr = post_to_lr(prior, posterior)
    return posterior, eff_lr


def _sample_lr_from_quantile(d: NodeDict, u: float) -> float:
    """Sample LR using a uniform quantile u ∈ [0,1] instead of random."""
    if 'likelihood_ratio' in d:
        return float(d['likelihood_ratio'])

    lo = float(d.get('lr_min', 1.0))
    hi = float(d.get('lr_max', 1.0))
    dist = d.get('lr_dist', 'log_uniform')

    if dist == 'uniform':
        return lo + u * (hi - lo)
    elif dist == 'beta':
        # Beta quantile approximation: use inverse beta via bisection
        alpha = float(d.get('lr_alpha', 2))
        beta_p = float(d.get('lr_beta', 2))
        t = _beta_quantile(u, alpha, beta_p)
        return lo + t * (hi - lo)
    else:  # log_uniform
        log_lo = math.log(max(lo, 1e-12))
        log_hi = math.log(max(hi, 1e-12))
        return math.exp(log_lo + u * (log_hi - log_lo))


def _beta_quantile(u: float, alpha: float, beta: float) -> float:
    """Approximate beta quantile via bisection on the regularized beta CDF."""
    u = max(1e-10, min(1 - 1e-10, u))
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if _beta_cdf(mid, alpha, beta) < u:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _beta_cdf(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function approximation using MC."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    # Use numerical integration (trapezoidal)
    n_steps = 200
    total = 0.0
    dx = x / n_steps
    for i in range(n_steps):
        t = (i + 0.5) * dx
        total += t ** (a - 1) * (1 - t) ** (b - 1) * dx
    # Normalize by B(a,b)
    full = 0.0
    dx_full = 1.0 / n_steps
    for i in range(n_steps):
        t = (i + 0.5) * dx_full
        full += t ** (a - 1) * (1 - t) ** (b - 1) * dx_full
    return total / max(full, 1e-30)


class CorrelationAttack(Attack):
    """
    Challenges the independence assumption between evidence branches.

    Tests what happens if specified pairs of evidence are correlated
    (e.g., they share an information source or methodology).
    """

    attack_type = "correlation"

    def generate(self, data: NodeDict, baseline_median: float,
                 n_sim: int = 5000) -> list[AttackResult]:
        children = data.get('children', [])
        n = len(children)
        if n < 2:
            return []

        results = []
        rho_values = [0.3, 0.5, 0.7, 0.9]

        # Test all unique pairs
        for i in range(n):
            for j in range(i + 1, n):
                name_i = children[i].get('node', f'Branch {i}')
                name_j = children[j].get('node', f'Branch {j}')

                # Only test pairs of same evidence_type (most plausible)
                et_i = children[i].get('evidence_type', 'neutral')
                et_j = children[j].get('evidence_type', 'neutral')

                for rho in rho_values:
                    posteriors = [
                        _sim_root_correlated(data, [(i, j)], rho)[0]
                        for _ in range(n_sim)
                    ]
                    s = sts(posteriors)
                    delta = s['median'] - baseline_median

                    # Only report if meaningful impact
                    if abs(delta) < 0.005:
                        continue

                    results.append(AttackResult(
                        attack_type=self.attack_type,
                        description=(
                            f"If '{_short(name_i)}' and '{_short(name_j)}' "
                            f"are correlated (ρ={rho})"
                        ),
                        target=f"{name_i} ↔ {name_j}",
                        original_median=baseline_median,
                        attacked_median=s['median'],
                        delta=delta,
                        plausibility=0.0,  # set by plausibility scorer
                        details={
                            'branch_i': i,
                            'branch_j': j,
                            'rho': rho,
                            'same_direction': et_i == et_j,
                            'p5': s['p5'],
                            'p95': s['p95'],
                        },
                    ))

        return results


# ── LRCalibrationAttack ──────────────────────────────────────────────────────

class LRCalibrationAttack(Attack):
    """
    Questions the likelihood ratio estimates.

    Two modes:
    - Shrink toward neutrality (LR → 1)
    - Expand uncertainty interval
    """

    attack_type = "lr_calibration"

    def generate(self, data: NodeDict, baseline_median: float,
                 n_sim: int = 5000) -> list[AttackResult]:
        children = data.get('children', [])
        results = []

        shrink_factors = [0.25, 0.50, 0.75]

        for idx, child in enumerate(children):
            name = child.get('node', f'Branch {idx}')

            if 'likelihood_ratio' in child:
                lr_pt = float(child['likelihood_ratio'])
                log_lr = math.log(max(lr_pt, 1e-12))

                for sf in shrink_factors:
                    new_lr = math.exp(log_lr * sf)
                    mod_data = _modify_child(data, idx,
                                             {'likelihood_ratio': new_lr})
                    posteriors = [sim_root(mod_data)[0]
                                 for _ in range(n_sim)]
                    s = sts(posteriors)
                    delta = s['median'] - baseline_median
                    if abs(delta) < 0.005:
                        continue
                    results.append(AttackResult(
                        attack_type=self.attack_type,
                        description=(
                            f"LR for '{_short(name)}' shrunk {sf:.0%} "
                            f"toward neutral: {lr_pt:.3g} → {new_lr:.3g}"
                        ),
                        target=name,
                        original_median=baseline_median,
                        attacked_median=s['median'],
                        delta=delta,
                        plausibility=0.0,
                        details={
                            'branch_idx': idx,
                            'mode': 'shrink_point',
                            'shrink_factor': sf,
                            'original_lr': lr_pt,
                            'attacked_lr': new_lr,
                        },
                    ))
            else:
                lr_min = float(child.get('lr_min', 1.0))
                lr_max = float(child.get('lr_max', 1.0))
                log_min = math.log(max(lr_min, 1e-12))
                log_max = math.log(max(lr_max, 1e-12))

                # Shrink toward neutrality
                for sf in shrink_factors:
                    new_min = math.exp(log_min * sf)
                    new_max = math.exp(log_max * sf)
                    mod_data = _modify_child(data, idx, {
                        'lr_min': new_min, 'lr_max': new_max,
                    })
                    posteriors = [sim_root(mod_data)[0]
                                 for _ in range(n_sim)]
                    s = sts(posteriors)
                    delta = s['median'] - baseline_median
                    if abs(delta) < 0.005:
                        continue
                    results.append(AttackResult(
                        attack_type=self.attack_type,
                        description=(
                            f"LR for '{_short(name)}' shrunk {sf:.0%} toward "
                            f"neutral: [{lr_min:.3g}–{lr_max:.3g}] → "
                            f"[{new_min:.3g}–{new_max:.3g}]"
                        ),
                        target=name,
                        original_median=baseline_median,
                        attacked_median=s['median'],
                        delta=delta,
                        plausibility=0.0,
                        details={
                            'branch_idx': idx,
                            'mode': 'shrink_interval',
                            'shrink_factor': sf,
                            'original_lr_min': lr_min,
                            'original_lr_max': lr_max,
                            'attacked_lr_min': new_min,
                            'attacked_lr_max': new_max,
                        },
                    ))

                # Expand uncertainty
                for expansion in [2.0, 4.0]:
                    log_center = (log_min + log_max) / 2
                    log_half = (log_max - log_min) / 2
                    new_half = log_half * expansion
                    new_min_e = math.exp(log_center - new_half)
                    new_max_e = math.exp(log_center + new_half)
                    mod_data = _modify_child(data, idx, {
                        'lr_min': new_min_e, 'lr_max': new_max_e,
                    })
                    posteriors = [sim_root(mod_data)[0]
                                 for _ in range(n_sim)]
                    s = sts(posteriors)
                    delta = s['median'] - baseline_median
                    if abs(delta) < 0.005:
                        continue
                    results.append(AttackResult(
                        attack_type=self.attack_type,
                        description=(
                            f"Uncertainty for '{_short(name)}' expanded "
                            f"{expansion}×: [{lr_min:.3g}–{lr_max:.3g}] → "
                            f"[{new_min_e:.3g}–{new_max_e:.3g}]"
                        ),
                        target=name,
                        original_median=baseline_median,
                        attacked_median=s['median'],
                        delta=delta,
                        plausibility=0.0,
                        details={
                            'branch_idx': idx,
                            'mode': 'expand',
                            'expansion': expansion,
                            'attacked_lr_min': new_min_e,
                            'attacked_lr_max': new_max_e,
                        },
                    ))

        return results


# ── MisspecificationAttack ───────────────────────────────────────────────────

class MisspecificationAttack(Attack):
    """
    Argues the wrong distribution was chosen for LR sampling.

    Tests alternatives: log_uniform ↔ uniform, beta with different shapes.
    """

    attack_type = "misspecification"

    def generate(self, data: NodeDict, baseline_median: float,
                 n_sim: int = 5000) -> list[AttackResult]:
        children = data.get('children', [])
        results = []

        alt_dists = [
            ('uniform', {}),
            ('beta', {'lr_alpha': 1, 'lr_beta': 1}),  # uniform-like
            ('beta', {'lr_alpha': 0.5, 'lr_beta': 0.5}),  # U-shaped
            ('beta', {'lr_alpha': 2, 'lr_beta': 5}),  # left-skewed
            ('beta', {'lr_alpha': 5, 'lr_beta': 2}),  # right-skewed
        ]

        for idx, child in enumerate(children):
            if 'likelihood_ratio' in child:
                continue  # point LR — no distribution to attack

            name = child.get('node', f'Branch {idx}')
            current_dist = child.get('lr_dist', 'log_uniform')

            for alt_dist, alt_params in alt_dists:
                if alt_dist == current_dist and not alt_params:
                    continue

                mods: dict[str, Any] = {'lr_dist': alt_dist}
                mods.update(alt_params)
                mod_data = _modify_child(data, idx, mods)

                posteriors = [sim_root(mod_data)[0] for _ in range(n_sim)]
                s = sts(posteriors)
                delta = s['median'] - baseline_median

                if abs(delta) < 0.005:
                    continue

                param_str = ""
                if alt_params:
                    param_str = f" (α={alt_params.get('lr_alpha', '?')}, β={alt_params.get('lr_beta', '?')})"

                results.append(AttackResult(
                    attack_type=self.attack_type,
                    description=(
                        f"Distribution for '{_short(name)}' changed: "
                        f"{current_dist} → {alt_dist}{param_str}"
                    ),
                    target=name,
                    original_median=baseline_median,
                    attacked_median=s['median'],
                    delta=delta,
                    plausibility=0.0,
                    details={
                        'branch_idx': idx,
                        'original_dist': current_dist,
                        'attacked_dist': alt_dist,
                        'attacked_params': alt_params,
                    },
                ))

        return results


# ── PriorBiasAttack ──────────────────────────────────────────────────────────

class PriorBiasAttack(Attack):
    """
    Attacks the prior probability assumption.

    Tests robustness across a range of priors and finds the prior
    needed to flip the conclusion.
    """

    attack_type = "prior_bias"

    def generate(self, data: NodeDict, baseline_median: float,
                 n_sim: int = 5000) -> list[AttackResult]:
        original_prior = data.get('prior', 0.5)
        results = []

        # Test specific priors
        test_priors = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                       0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
        test_priors = [p for p in test_priors
                       if abs(p - original_prior) > 0.02]

        sweep_data = []
        for p in test_priors:
            mod_data = {**data, 'prior': p}
            posteriors = [sim_root(mod_data)[0] for _ in range(n_sim)]
            s = sts(posteriors)
            sweep_data.append((p, s['median']))

        # Find flip prior via bisection
        flip_prior = self._find_flip_prior(data, baseline_median, n_sim)

        # Report the most impactful prior shifts
        for p, med in sweep_data:
            delta = med - baseline_median
            if abs(delta) < 0.005:
                continue

            direction = "higher" if p > original_prior else "lower"
            results.append(AttackResult(
                attack_type=self.attack_type,
                description=(
                    f"Prior shifted {direction}: "
                    f"{original_prior:.0%} → {p:.0%}"
                ),
                target="prior",
                original_median=baseline_median,
                attacked_median=med,
                delta=delta,
                plausibility=0.0,
                details={
                    'original_prior': original_prior,
                    'attacked_prior': p,
                    'flip_prior': flip_prior,
                    'sweep': [(sp, sm) for sp, sm in sweep_data],
                },
            ))

        return results

    def _find_flip_prior(self, data: NodeDict, baseline_median: float,
                         n_sim: int) -> Optional[float]:
        """Binary search for the prior that flips the conclusion past 50%."""
        above = baseline_median >= 0.5
        lo, hi = 0.01, 0.99

        # Check if flip is even possible
        for extreme in [lo, hi]:
            mod = {**data, 'prior': extreme}
            posteriors = [sim_root(mod)[0] for _ in range(n_sim // 2)]
            med = sts(posteriors)['median']
            if (med >= 0.5) != above:
                break
        else:
            return None  # can't flip even at extremes

        for _ in range(20):
            mid = (lo + hi) / 2
            mod = {**data, 'prior': mid}
            posteriors = [sim_root(mod)[0] for _ in range(n_sim // 2)]
            med = sts(posteriors)['median']
            if (med >= 0.5) == above:
                if above:
                    hi = mid
                else:
                    lo = mid
            else:
                if above:
                    lo = mid
                else:
                    hi = mid

        return round((lo + hi) / 2, 4)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _modify_child(data: NodeDict, child_idx: int,
                  mods: dict[str, Any]) -> NodeDict:
    """Return a deep copy of data with modifications to one child."""
    new_data = copy.deepcopy(data)
    child = new_data['children'][child_idx]
    child.update(mods)
    return new_data


def _short(name: str, max_len: int = 40) -> str:
    """Truncate name for display."""
    return name if len(name) <= max_len else name[:max_len - 1] + "…"
