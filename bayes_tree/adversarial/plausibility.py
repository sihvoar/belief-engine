"""
Plausibility scoring for adversarial attacks.

Assigns a 0–10 plausibility score to each attack based on how realistic
the attack scenario is. Filters out pathological attacks.
"""

from __future__ import annotations

import math
from typing import Any

from bayes_tree.adversarial.attacks import AttackResult
from bayes_tree.engine import NodeDict


def score_attack(result: AttackResult, data: NodeDict) -> float:
    """
    Assign a plausibility score (0–10) to an attack result.

    Higher = more plausible criticism a peer reviewer might raise.
    """
    if result.attack_type == "correlation":
        return _score_correlation(result, data)
    elif result.attack_type == "lr_calibration":
        return _score_lr_calibration(result, data)
    elif result.attack_type == "misspecification":
        return _score_misspecification(result, data)
    elif result.attack_type == "prior_bias":
        return _score_prior_bias(result, data)
    return 5.0


def _score_correlation(result: AttackResult, data: NodeDict) -> float:
    """
    Correlation attacks are more plausible when:
    - Branches share similar evidence types (same direction)
    - Lower rho values (mild correlation is very common)
    - Both are testimonial/qualitative rather than quantitative
    """
    score = 5.0
    details = result.details
    rho = details.get('rho', 0.5)

    # Moderate correlations (0.3–0.5) are very plausible
    if rho <= 0.3:
        score += 2.0
    elif rho <= 0.5:
        score += 1.0
    elif rho >= 0.9:
        score -= 2.0

    # Same-direction evidence more likely to share sources
    if details.get('same_direction', False):
        score += 1.5

    # Check if branches look like they could share an information source
    children = data.get('children', [])
    i, j = details.get('branch_i', 0), details.get('branch_j', 0)
    if i < len(children) and j < len(children):
        name_i = children[i].get('node', '').lower()
        name_j = children[j].get('node', '').lower()
        # Same methodology keywords suggest shared source
        shared_keywords = _count_shared_keywords(name_i, name_j)
        score += min(shared_keywords * 0.5, 2.0)

    return max(0.0, min(10.0, score))


def _score_lr_calibration(result: AttackResult, data: NodeDict) -> float:
    """
    LR calibration attacks are more plausible when:
    - Original LR is extreme (far from 1)
    - Shrink factor is moderate (not total)
    - Evidence is qualitative/subjective
    """
    score = 5.0
    details = result.details
    idx = details.get('branch_idx', 0)
    children = data.get('children', [])

    if idx < len(children):
        child = children[idx]
        lr_min = float(child.get('lr_min', child.get('likelihood_ratio', 1.0)))
        lr_max = float(child.get('lr_max', child.get('likelihood_ratio', 1.0)))
        geo_mean = math.sqrt(max(lr_min, 1e-12) * max(lr_max, 1e-12))

        # Extreme LRs are more attackable (harder to justify)
        extremity = abs(math.log(max(geo_mean, 1e-12)))
        if extremity > 4:  # LR > ~55 or < ~0.02
            score += 2.5
        elif extremity > 2:  # LR > ~7 or < ~0.14
            score += 1.5
        elif extremity > 1:
            score += 0.5

        # Qualitative evidence is more attackable
        name = child.get('node', '').lower()
        if _is_qualitative(name):
            score += 1.5
        elif _is_quantitative(name):
            score -= 1.0

    # Moderate shrinks are more plausible than total
    mode = details.get('mode', '')
    sf = details.get('shrink_factor', 0.5)
    if mode.startswith('shrink'):
        if 0.4 <= sf <= 0.6:
            score += 1.0
        elif sf <= 0.25:
            score -= 1.0

    # Uncertainty expansions are always plausible
    if mode == 'expand':
        expansion = details.get('expansion', 2.0)
        if expansion <= 2.0:
            score += 1.0
        elif expansion >= 4.0:
            score -= 0.5

    return max(0.0, min(10.0, score))


def _score_misspecification(result: AttackResult, data: NodeDict) -> float:
    """
    Distribution misspecification is moderately plausible for any branch
    that spans more than one order of magnitude.
    """
    score = 4.0
    details = result.details
    idx = details.get('branch_idx', 0)
    children = data.get('children', [])

    if idx < len(children):
        child = children[idx]
        lr_min = float(child.get('lr_min', 1.0))
        lr_max = float(child.get('lr_max', 1.0))
        ratio = lr_max / max(lr_min, 1e-12)

        # Wider intervals → distribution choice matters more
        if ratio > 100:
            score += 3.0
        elif ratio > 10:
            score += 2.0
        elif ratio > 3:
            score += 1.0
        else:
            score -= 1.0  # narrow interval → dist barely matters

    # Switching to uniform from log_uniform is very plausible
    orig = details.get('original_dist', 'log_uniform')
    alt = details.get('attacked_dist', '')
    if orig == 'log_uniform' and alt == 'uniform':
        score += 1.0
    elif alt == 'beta':
        params = details.get('attacked_params', {})
        # U-shaped beta is unusual
        if params.get('lr_alpha', 2) < 1 and params.get('lr_beta', 2) < 1:
            score -= 1.0

    return max(0.0, min(10.0, score))


def _score_prior_bias(result: AttackResult, data: NodeDict) -> float:
    """
    Prior attacks are more plausible when:
    - The shift is small
    - The original prior has no strong empirical basis
    """
    score = 5.0
    details = result.details
    original = details.get('original_prior', 0.5)
    attacked = details.get('attacked_prior', 0.5)
    shift = abs(attacked - original)

    # Small shifts are more plausible
    if shift <= 0.10:
        score += 2.0
    elif shift <= 0.20:
        score += 1.0
    elif shift >= 0.40:
        score -= 2.0

    # 50% prior is the weakest (maximally uncertain) — easy to argue for shift
    if abs(original - 0.5) < 0.05:
        score += 1.0

    return max(0.0, min(10.0, score))


# ── Keyword helpers ──────────────────────────────────────────────────────────

_QUALITATIVE_TERMS = {
    'testimony', 'witness', 'report', 'argument', 'tradition',
    'tradition', 'opinion', 'narrative', 'claim', 'historical',
    'ancient', 'literary', 'theological', 'philosophical',
    'subjective', 'anecdotal', 'folk', 'legend',
}

_QUANTITATIVE_TERMS = {
    'dna', 'isotope', 'radiocarbon', 'carbon-14', 'c-14',
    'spectroscopy', 'analysis', 'measurement', 'statistical',
    'laboratory', 'experiment', 'test', 'assay', 'data',
    'forensic', 'autopsy', 'sample',
}


def _is_qualitative(name: str) -> bool:
    words = set(name.split())
    return bool(words & _QUALITATIVE_TERMS)


def _is_quantitative(name: str) -> bool:
    words = set(name.split())
    return bool(words & _QUANTITATIVE_TERMS)


def _count_shared_keywords(name_i: str, name_j: str) -> int:
    """Count semantically meaningful words shared between two branch names."""
    stop = {'the', 'a', 'an', 'is', 'was', 'of', 'in', 'to', 'and', 'or',
            'for', 'with', 'from', 'by', 'not', 'no', 'but', 'that', 'this'}
    words_i = set(name_i.split()) - stop
    words_j = set(name_j.split()) - stop
    return len(words_i & words_j)
