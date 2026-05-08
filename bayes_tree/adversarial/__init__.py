"""
Adversarial Attack Mode for Bayes Tree.

Systematically challenges the assumptions behind a Bayesian evidence tree
by applying plausible attacks and reporting vulnerabilities.

Usage::

    from bayes_tree.adversarial import run_audit

    audit = run_audit(tree_data, n_sim=10_000)
    print(audit.summary())
"""

from bayes_tree.adversarial.auditor import run_audit, AuditResult
from bayes_tree.adversarial.attacks import (
    Attack,
    AttackResult,
    CorrelationAttack,
    LRCalibrationAttack,
    MisspecificationAttack,
    PriorBiasAttack,
)

__all__ = [
    "run_audit",
    "AuditResult",
    "Attack",
    "AttackResult",
    "CorrelationAttack",
    "LRCalibrationAttack",
    "MisspecificationAttack",
    "PriorBiasAttack",
]
