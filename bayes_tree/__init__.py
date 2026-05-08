"""
Bayes Tree — Bayesian evidence trees with Monte Carlo uncertainty propagation.

Make your assumptions explicit. Combine uncertain evidence into a posterior
distribution with calibrated uncertainty propagation.

Basic usage::

    from bayes_tree import run_simulation
    import yaml

    with open("my_analysis.yaml") as f:
        data = yaml.safe_load(f)

    results = run_simulation(data, n_sim=10_000)
    print(f"Posterior median: {results['stats']['median']:.1%}")
"""

__version__ = "1.1.0"

from bayes_tree.engine import (
    to_lo,
    from_lo,
    bayes_upd,
    post_to_lr,
    sample_lr,
    validate_node,
    collect_leaves,
    sim_root,
    pct,
    sts,
    collect,
    run_simulation,
    NodeResult,
    NodeDict,
)

__all__ = [
    "to_lo",
    "from_lo",
    "bayes_upd",
    "post_to_lr",
    "sample_lr",
    "validate_node",
    "collect_leaves",
    "sim_root",
    "pct",
    "sts",
    "collect",
    "run_simulation",
    "NodeResult",
    "NodeDict",
    "__version__",
    "adversarial",
]
