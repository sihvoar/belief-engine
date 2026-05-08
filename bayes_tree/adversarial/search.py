"""
Attack combination search.

Finds the minimal set of plausible attacks that can flip a conclusion,
using greedy search with optional beam width.
"""

from __future__ import annotations

import copy
import math
from typing import Any

from bayes_tree.adversarial.attacks import (
    AttackResult,
    CorrelationAttack,
    LRCalibrationAttack,
    MisspecificationAttack,
    PriorBiasAttack,
    _modify_child,
    _sim_root_correlated,
)
from bayes_tree.engine import NodeDict, sim_root, sts


def greedy_search(
    data: NodeDict,
    candidates: list[AttackResult],
    baseline_median: float,
    max_attacks: int = 3,
    threshold: float = 0.50,
    n_sim: int = 5000,
) -> list[AttackResult]:
    """
    Greedy search for the minimal combination of attacks that flips the
    conclusion past the threshold.

    1. Pick the single most damaging plausible attack.
    2. Apply it and re-evaluate remaining candidates.
    3. Repeat until conclusion flips or budget exhausted.

    Returns the selected attack combination (may be fewer than max_attacks
    if the conclusion flips early).
    """
    if not candidates:
        return []

    above_threshold = baseline_median >= threshold
    selected: list[AttackResult] = []
    remaining = list(candidates)
    current_data = copy.deepcopy(data)
    current_median = baseline_median

    for _ in range(max_attacks):
        if not remaining:
            break

        # Already flipped?
        if (current_median >= threshold) != above_threshold:
            break

        # Evaluate each remaining candidate on the current (possibly modified) tree
        best: AttackResult | None = None
        best_score = -1.0

        for candidate in remaining:
            # Score = impact × plausibility (weighted)
            impact = abs(candidate.delta)
            plaus = candidate.plausibility
            score = impact * (0.5 + 0.5 * plaus / 10.0)

            if score > best_score:
                best_score = score
                best = candidate

        if best is None:
            break

        selected.append(best)
        remaining.remove(best)

        # Apply the attack to current_data for next iteration
        current_data = _apply_attack(current_data, best)

        # Re-simulate to get new median
        posteriors = [sim_root(current_data)[0] for _ in range(n_sim)]
        current_median = sts(posteriors)['median']

    # Update the last attack's result to show cumulative effect
    if selected:
        combined_delta = current_median - baseline_median
        selected[-1] = AttackResult(
            attack_type="combined" if len(selected) > 1 else selected[-1].attack_type,
            description=(
                f"Combined effect of {len(selected)} attack(s)"
                if len(selected) > 1 else selected[-1].description
            ),
            target="combined",
            original_median=baseline_median,
            attacked_median=current_median,
            delta=combined_delta,
            plausibility=min(a.plausibility for a in selected),
            details={
                'attacks_applied': len(selected),
                'individual_attacks': [
                    {'type': a.attack_type, 'target': a.target, 'delta': a.delta}
                    for a in selected[:-1]
                ] + [{'type': selected[-1].attack_type if len(selected) == 1 else "combined",
                      'target': 'combined', 'delta': combined_delta}],
                'flipped': (current_median >= threshold) != above_threshold,
            },
        )

    return selected


def _apply_attack(data: NodeDict, attack: AttackResult) -> NodeDict:
    """Apply an attack's modifications to a tree, returning modified copy."""
    new_data = copy.deepcopy(data)
    details = attack.details

    if attack.attack_type == "lr_calibration":
        idx = details.get('branch_idx', 0)
        mode = details.get('mode', '')
        if mode == 'shrink_point':
            new_data['children'][idx]['likelihood_ratio'] = details['attacked_lr']
        elif mode == 'shrink_interval':
            new_data['children'][idx]['lr_min'] = details['attacked_lr_min']
            new_data['children'][idx]['lr_max'] = details['attacked_lr_max']
        elif mode == 'expand':
            new_data['children'][idx]['lr_min'] = details['attacked_lr_min']
            new_data['children'][idx]['lr_max'] = details['attacked_lr_max']

    elif attack.attack_type == "misspecification":
        idx = details.get('branch_idx', 0)
        new_data['children'][idx]['lr_dist'] = details['attacked_dist']
        for k, v in details.get('attacked_params', {}).items():
            new_data['children'][idx][k] = v

    elif attack.attack_type == "prior_bias":
        new_data['prior'] = details.get('attacked_prior', data.get('prior', 0.5))

    # correlation attacks don't modify the tree — they change the sim process
    # so we skip them here (they're handled separately in the auditor)

    return new_data
