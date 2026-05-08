"""
Adversarial Auditor — main orchestrator for adversarial analysis.

Runs all attack types, scores plausibility, searches for attack combinations,
generates defenses, and produces structured audit reports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from bayes_tree.engine import NodeDict, run_simulation, sts
from bayes_tree.adversarial.attacks import (
    AttackResult,
    CorrelationAttack,
    LRCalibrationAttack,
    MisspecificationAttack,
    PriorBiasAttack,
)
from bayes_tree.adversarial.plausibility import score_attack
from bayes_tree.adversarial.defense import generate_defenses
from bayes_tree.adversarial.search import greedy_search


@dataclass
class AuditResult:
    """Complete adversarial audit of a Bayesian evidence tree."""
    hypothesis: str
    original_prior: float
    original_median: float
    original_p5: float
    original_p95: float
    n_sim: int

    # All individual attacks (filtered by plausibility)
    attacks: list[AttackResult] = field(default_factory=list)

    # Best attack combo found by search
    best_combo: list[AttackResult] = field(default_factory=list)

    # Metadata
    flip_prior: Optional[float] = None
    can_flip: bool = False
    most_vulnerable: Optional[str] = None

    def summary(self) -> str:
        """Human-readable summary of the audit."""
        return format_audit_text(self)

    def to_json(self) -> str:
        """JSON-serializable audit report."""
        return format_audit_json(self)


def run_audit(
    data: NodeDict,
    n_sim: int = 5000,
    max_attacks: int = 3,
    plausibility_threshold: float = 3.0,
    flip_threshold: float = 0.50,
) -> AuditResult:
    """
    Run a complete adversarial audit on an evidence tree.

    Args:
        data: Parsed YAML evidence tree
        n_sim: Simulations per attack variant
        max_attacks: Maximum attacks in combination search
        plausibility_threshold: Minimum plausibility to report (0–10)
        flip_threshold: Posterior threshold defining "conclusion flip"

    Returns:
        AuditResult with all attacks, scores, defenses, and combos
    """
    # Get baseline
    results = run_simulation(data, n_sim=n_sim)
    baseline_median = results['stats']['median']
    hypothesis = data.get('node', 'Unknown')
    prior = data.get('prior', 0.5)

    # Run all attack types
    attack_classes = [
        CorrelationAttack(),
        LRCalibrationAttack(),
        MisspecificationAttack(),
        PriorBiasAttack(),
    ]

    all_attacks: list[AttackResult] = []
    for attacker in attack_classes:
        variants = attacker.generate(data, baseline_median, n_sim=n_sim)
        all_attacks.extend(variants)

    # Score plausibility and generate defenses
    for attack in all_attacks:
        attack.plausibility = score_attack(attack, data)
        attack.defenses = generate_defenses(attack, data)

    # Filter by plausibility
    plausible = [a for a in all_attacks if a.plausibility >= plausibility_threshold]

    # Sort by impact × plausibility
    plausible.sort(
        key=lambda a: abs(a.delta) * (0.5 + 0.5 * a.plausibility / 10.0),
        reverse=True,
    )

    # Deduplicate: keep only the best variant per (attack_type, target) pair
    seen: set[tuple[str, str]] = set()
    deduped: list[AttackResult] = []
    for a in plausible:
        key = (a.attack_type, a.target)
        if key not in seen:
            seen.add(key)
            deduped.append(a)

    # Search for best attack combination
    best_combo = greedy_search(
        data, deduped, baseline_median,
        max_attacks=max_attacks,
        threshold=flip_threshold,
        n_sim=n_sim,
    )

    # Find flip prior (from PriorBiasAttack details)
    flip_prior = None
    for a in all_attacks:
        if a.attack_type == "prior_bias" and 'flip_prior' in a.details:
            flip_prior = a.details['flip_prior']
            break

    # Determine most vulnerable branch
    most_vulnerable = None
    if deduped:
        most_vulnerable = deduped[0].target

    can_flip = any(
        a.flipped for a in deduped
    ) or (best_combo and best_combo[-1].details.get('flipped', False))

    return AuditResult(
        hypothesis=hypothesis,
        original_prior=prior,
        original_median=baseline_median,
        original_p5=results['stats']['p5'],
        original_p95=results['stats']['p95'],
        n_sim=n_sim,
        attacks=deduped,
        best_combo=best_combo,
        flip_prior=flip_prior,
        can_flip=can_flip,
        most_vulnerable=most_vulnerable,
    )


# ── Text formatting ──────────────────────────────────────────────────────────

def format_audit_text(audit: AuditResult) -> str:
    """Format audit as human-readable terminal report."""
    lines: list[str] = []
    B = "\033[1m"
    R = "\033[0m"
    RED = "\033[91m"
    GRN = "\033[92m"
    YEL = "\033[93m"
    CYN = "\033[96m"
    GRY = "\033[90m"

    lines.append("")
    lines.append(f"{B}{CYN}⚔️  ADVERSARIAL AUDIT{R}")
    lines.append("═" * 60)
    lines.append(f"  Hypothesis: {B}{audit.hypothesis}{R}")
    lines.append(f"  Prior: {CYN}{audit.original_prior:.1%}{R}")
    lines.append(f"  Baseline posterior: {CYN}{audit.original_median:.2%}{R} "
                 f"{GRY}[{audit.original_p5:.2%}–{audit.original_p95:.2%}]{R}")
    lines.append("")

    # Vulnerability summary
    if audit.can_flip:
        lines.append(f"  {RED}⚠ CONCLUSION CAN BE FLIPPED by plausible attacks{R}")
    else:
        lines.append(f"  {GRN}✓ Conclusion appears robust to individual attacks{R}")

    if audit.flip_prior is not None:
        lines.append(f"  {GRY}Flip prior: {audit.flip_prior:.1%} "
                     f"(prior needed to reverse conclusion){R}")

    if audit.most_vulnerable:
        lines.append(f"  {YEL}Most vulnerable: {audit.most_vulnerable}{R}")

    lines.append("")

    # Individual attacks
    if audit.attacks:
        lines.append(f"{B}INDIVIDUAL ATTACKS{R} "
                     f"{GRY}(plausibility ≥ 3.0, sorted by impact){R}")
        lines.append("─" * 60)

        for rank, attack in enumerate(audit.attacks[:15], 1):
            sev_col = {
                'critical': RED, 'high': RED,
                'moderate': YEL, 'low': GRY,
            }.get(attack.severity, GRY)

            sev_icon = {
                'critical': '🔴', 'high': '🟠',
                'moderate': '🟡', 'low': '⚪',
            }.get(attack.severity, '⚪')

            flip_mark = f" {RED}← FLIPS{R}" if attack.flipped else ""
            plaus_bar = "●" * int(attack.plausibility) + "○" * (10 - int(attack.plausibility))

            lines.append(
                f"\n  {sev_icon} {B}#{rank}{R} {sev_col}[{attack.severity.upper()}]{R}"
                f"  Δ = {sev_col}{attack.delta:+.2%}{R}{flip_mark}"
            )
            lines.append(f"     {attack.description}")
            lines.append(f"     Plausibility: {plaus_bar} ({attack.plausibility:.1f}/10)")
            lines.append(f"     Posterior: {audit.original_median:.2%} → "
                         f"{attack.attacked_median:.2%}")

            if attack.defenses:
                lines.append(f"     {GRN}Defenses:{R}")
                for defense in attack.defenses[:2]:
                    lines.append(f"       • {defense}")

    # Attack combination
    if audit.best_combo and len(audit.best_combo) > 1:
        lines.append("")
        lines.append(f"{B}ATTACK COMBINATION{R} {GRY}(greedy search){R}")
        lines.append("─" * 60)
        combo = audit.best_combo[-1]
        flipped = combo.details.get('flipped', False)
        status = f"{RED}FLIPS CONCLUSION{R}" if flipped else f"{YEL}weakens but doesn't flip{R}"
        lines.append(f"  {len(audit.best_combo)} attacks combined: {status}")
        lines.append(f"  {audit.original_median:.2%} → {combo.attacked_median:.2%} "
                     f"(Δ = {combo.delta:+.2%})")
        for i, a in enumerate(audit.best_combo[:-1], 1):
            lines.append(f"    {i}. [{a.attack_type}] {a.target}")
        if len(audit.best_combo) > 1:
            lines.append(f"    {len(audit.best_combo)}. combined effect")

    # Robustness verdict
    lines.append("")
    lines.append("─" * 60)
    n_critical = sum(1 for a in audit.attacks if a.severity in ('critical', 'high'))
    n_moderate = sum(1 for a in audit.attacks if a.severity == 'moderate')
    n_low = sum(1 for a in audit.attacks if a.severity == 'low')

    if n_critical == 0 and not audit.can_flip:
        lines.append(f"  {GRN}{B}VERDICT: ROBUST{R} — no high-severity vulnerabilities found")
    elif n_critical <= 2 and not audit.can_flip:
        lines.append(f"  {YEL}{B}VERDICT: MODERATELY ROBUST{R} — "
                     f"{n_critical} high-severity attacks, but conclusion holds")
    else:
        lines.append(f"  {RED}{B}VERDICT: VULNERABLE{R} — "
                     f"{n_critical} high-severity attacks found"
                     + (", conclusion can be flipped" if audit.can_flip else ""))

    lines.append(f"  {GRY}Attacks tested: {len(audit.attacks)} plausible "
                 f"({n_critical} critical/high, {n_moderate} moderate, {n_low} low){R}")
    lines.append("")

    return "\n".join(lines)


# ── JSON formatting ──────────────────────────────────────────────────────────

def format_audit_json(audit: AuditResult) -> str:
    """Format audit as JSON."""
    output = {
        "hypothesis": audit.hypothesis,
        "prior": audit.original_prior,
        "baseline_posterior": audit.original_median,
        "baseline_ci": [audit.original_p5, audit.original_p95],
        "n_sim": audit.n_sim,
        "can_flip": audit.can_flip,
        "flip_prior": audit.flip_prior,
        "most_vulnerable": audit.most_vulnerable,
        "verdict": _verdict(audit),
        "attacks": [
            {
                "rank": i + 1,
                "type": a.attack_type,
                "description": a.description,
                "target": a.target,
                "original_median": a.original_median,
                "attacked_median": a.attacked_median,
                "delta": a.delta,
                "severity": a.severity,
                "plausibility": a.plausibility,
                "flipped": a.flipped,
                "defenses": a.defenses,
            }
            for i, a in enumerate(audit.attacks)
        ],
        "best_combo": {
            "n_attacks": len(audit.best_combo),
            "final_median": audit.best_combo[-1].attacked_median if audit.best_combo else None,
            "flipped": (audit.best_combo[-1].details.get('flipped', False)
                        if audit.best_combo else False),
            "steps": [
                {"type": a.attack_type, "target": a.target, "delta": a.delta}
                for a in audit.best_combo
            ],
        } if audit.best_combo else None,
    }
    return json.dumps(output, indent=2, ensure_ascii=False)


def _verdict(audit: AuditResult) -> str:
    n_critical = sum(1 for a in audit.attacks if a.severity in ('critical', 'high'))
    if n_critical == 0 and not audit.can_flip:
        return "robust"
    elif n_critical <= 2 and not audit.can_flip:
        return "moderately_robust"
    else:
        return "vulnerable"
