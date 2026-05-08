"""
Defense generation for adversarial attacks.

For each successful attack, suggests concrete defenses the analyst can take
to strengthen their argument.
"""

from __future__ import annotations

import math
from typing import Any

from bayes_tree.adversarial.attacks import AttackResult
from bayes_tree.engine import NodeDict


def generate_defenses(result: AttackResult, data: NodeDict) -> list[str]:
    """Generate defense suggestions for an attack result."""
    if result.attack_type == "correlation":
        return _defend_correlation(result, data)
    elif result.attack_type == "lr_calibration":
        return _defend_lr_calibration(result, data)
    elif result.attack_type == "misspecification":
        return _defend_misspecification(result, data)
    elif result.attack_type == "prior_bias":
        return _defend_prior_bias(result, data)
    return []


def _defend_correlation(result: AttackResult, data: NodeDict) -> list[str]:
    """Defenses against correlation attacks."""
    defenses = []
    details = result.details
    rho = details.get('rho', 0.5)
    children = data.get('children', [])
    i = details.get('branch_i', 0)
    j = details.get('branch_j', 0)

    name_i = children[i].get('node', '?') if i < len(children) else '?'
    name_j = children[j].get('node', '?') if j < len(children) else '?'

    defenses.append(
        f"Document why '{_short(name_i)}' and '{_short(name_j)}' are "
        f"independently sourced (different methodologies, different data)."
    )
    defenses.append(
        "Add a correlation parameter explicitly to the model to show "
        "robustness even under partial dependence."
    )

    if rho >= 0.5:
        defenses.append(
            "Consider merging these branches into a single combined branch "
            "with adjusted LR bounds to avoid the independence question."
        )

    # Suggest corroborating evidence
    if abs(result.delta) > 0.10:
        target_lr = 1.0 / max(abs(result.delta), 0.01)  # compensating LR
        direction = "supporting" if result.delta < 0 else "opposing"
        defenses.append(
            f"Add independent corroborating {direction} evidence with "
            f"LR ≈ {target_lr:.1f} to compensate for potential correlation."
        )

    return defenses


def _defend_lr_calibration(result: AttackResult, data: NodeDict) -> list[str]:
    """Defenses against LR calibration attacks."""
    defenses = []
    details = result.details
    idx = details.get('branch_idx', 0)
    children = data.get('children', [])
    name = children[idx].get('node', '?') if idx < len(children) else '?'
    mode = details.get('mode', '')

    if mode.startswith('shrink'):
        defenses.append(
            f"Cite specific studies or data supporting the LR estimate for "
            f"'{_short(name)}'. Peer-reviewed sources are strongest."
        )
        defenses.append(
            "Compute LR from base rates: P(evidence|H) / P(evidence|¬H) "
            "using documented frequencies."
        )

    if mode == 'expand':
        defenses.append(
            f"Narrow the uncertainty interval for '{_short(name)}' by "
            f"collecting additional data or citing meta-analyses."
        )
        defenses.append(
            "Use a calibrated expert elicitation protocol (e.g., SHELF) "
            "to produce defensible bounds."
        )

    # General
    sf = details.get('shrink_factor', 0.5)
    if sf <= 0.5:
        lr_min = details.get('original_lr_min', details.get('original_lr', 1))
        lr_max = details.get('original_lr_max', details.get('original_lr', 1))
        defenses.append(
            f"Run a sensitivity analysis showing the conclusion holds even "
            f"if the LR for '{_short(name)}' is halved."
        )

    return defenses


def _defend_misspecification(result: AttackResult, data: NodeDict) -> list[str]:
    """Defenses against distribution misspecification attacks."""
    defenses = []
    details = result.details
    idx = details.get('branch_idx', 0)
    children = data.get('children', [])
    name = children[idx].get('node', '?') if idx < len(children) else '?'
    orig_dist = details.get('original_dist', 'log_uniform')

    defenses.append(
        f"Justify why {orig_dist} is appropriate for '{_short(name)}'. "
        f"Log-uniform is standard when uncertainty spans orders of magnitude."
    )
    defenses.append(
        "Show the conclusion is robust to distribution choice by running "
        "with all three distributions and comparing posteriors."
    )

    lr_min = float(children[idx].get('lr_min', 1.0)) if idx < len(children) else 1.0
    lr_max = float(children[idx].get('lr_max', 1.0)) if idx < len(children) else 1.0
    ratio = lr_max / max(lr_min, 1e-12)
    if ratio < 5:
        defenses.append(
            f"The interval [{lr_min:.2g}–{lr_max:.2g}] is narrow enough "
            f"that distribution choice has minimal impact. Document this."
        )

    return defenses


def _defend_prior_bias(result: AttackResult, data: NodeDict) -> list[str]:
    """Defenses against prior bias attacks."""
    defenses = []
    details = result.details
    original = details.get('original_prior', 0.5)
    flip_prior = details.get('flip_prior')

    defenses.append(
        f"Justify the prior of {original:.0%} with reference to base rates, "
        f"domain expertise, or a systematic literature review."
    )

    if flip_prior is not None:
        defenses.append(
            f"The conclusion only flips if the prior exceeds {flip_prior:.0%}. "
            f"Argue why priors above this threshold are unreasonable."
        )

    defenses.append(
        "Include a prior sensitivity sweep (--prior-sweep) in your report "
        "to show readers the conclusion's robustness."
    )

    if abs(original - 0.5) < 0.05:
        defenses.append(
            "Consider whether a more informative prior (based on base rates) "
            "would strengthen the analysis."
        )

    return defenses


def _short(name: str, max_len: int = 40) -> str:
    return name if len(name) <= max_len else name[:max_len - 1] + "…"
