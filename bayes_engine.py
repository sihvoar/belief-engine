"""
Bayesian Decision Tree Engine — backward-compatibility shim.

Canonical source: bayes_tree/engine.py
This file re-exports everything so existing scripts (bayes-tree-eng.py, GUI, etc.)
continue to work with `from bayes_engine import ...`.
"""

from bayes_tree.engine import *  # noqa: F401,F403
from bayes_tree.engine import NodeResult, NodeDict  # noqa: F401 (explicit for type checkers)
