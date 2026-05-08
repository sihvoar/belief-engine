"""
Bayes Tree — Bayesian evidence trees with Monte Carlo uncertainty propagation.

Turn messy debates into math. Structure arguments as Bayesian evidence trees
and let Monte Carlo tell you what to believe.

Basic usage::

    from bayes_tree import run_simulation
    import yaml

    with open("my_analysis.yaml") as f:
        data = yaml.safe_load(f)

    results = run_simulation(data, n_sim=10_000)
    print(f"Posterior median: {results['stats']['median']:.1%}")
"""

__version__ = "1.0.0"

from bayes_tree.engine import (
    to_lo,
    from_lo,
    bayes_upd,
    post_to_lr,
    sample_lr,
    validate_node,
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
