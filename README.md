# 🌳 Bayes Tree

**Turn messy debates into math.** Structure arguments as Bayesian evidence trees and let Monte Carlo tell you what to believe.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)

---

## What It Does

You have a yes/no question. You have evidence — some supporting, some against, all uncertain. Bayes Tree lets you:

1. **Structure** evidence as a tree with likelihood-ratio intervals
2. **Simulate** thousands of Monte Carlo runs to propagate uncertainty
3. **See** a posterior distribution, sensitivity analysis, and importance ranking

No more gut feelings about complex questions. No more "I read both sides and I'm confused." Quantify it.

## Quick Start

```bash
pip install -r requirements.txt
python bayes-tree-eng.py examples/shroud.yaml
```

## Sample Output

```
BAYESIAN DECISION TREE  +  MONTE CARLO
File: shroud.yaml   Simulations: 10,000
───────────────────────────────────────────────────────

Simulating combined posterior...

  Median   : 1.613%
  Mean     : 2.037%
  Std      : 1.507%
  90% CI   : [0.490%–5.153%]
  Range    : [0.199%–10.679%]

  Effective LR (90% CI): [0.0049–0.0543]
  Effective LR median:   0.0164

            0%                   100%
            ──────────────────────────────────────
   0.20%–0.72%  ███████████████████████░░░░░░░░░░░░░░░ 684
   0.72%–1.25%  ██████████████████████████████████████ 1124
   1.25%–1.77%  ███████████████████████████████░░░░░░░ 944
   1.77%–2.29%  ███████████████████████░░░░░░░░░░░░░░░ 695
   2.29%–2.82%  ███████████████░░░░░░░░░░░░░░░░░░░░░░░ 454
   2.82%–3.34%  ███████████░░░░░░░░░░░░░░░░░░░░░░░░░░░ 335
   ...

───────────────────────────────────────────────────────
TREE  (root LR computed from children's distribution)
───────────────────────────────────────────────────────

Is the Shroud of Turin authentic?
  Prior:    50.0%
  Eff. LR:  LR=[0.0051–0.0519] (computed from children's distribution)
  Median:   1.599%  90% CI [0.510%–4.933%]

├── Radiocarbon dating (-) LR=[0.01–0.05]
│     50.00% → 2.20% [1.08%–4.39%]
│   ├── Oxford laboratory (-) LR=[0.05–0.15]
│   │     2.20% → 0.19% [0.12%–0.32%]
│   ├── Zürich laboratory (-) LR=[0.05–0.15]
│   │     2.20% → 0.20% [0.12%–0.32%]
│   ...
├── Historical sources (-) LR=[0.05–0.20]
│     50.00% → 6.42% [3.05%–13.33%]
├── Image properties (+) LR=[2.00–6.00]
│     50.00% → 77.28% [66.79%–85.53%]
└── Wound anatomy (-) LR=[1.50–3.00]
      50.00% → 67.99% [61.00%–74.26%]

───────────────────────────────────────────────────────
IMPORTANCE RANKING  (leave-one-out)
───────────────────────────────────────────────────────
How much does the posterior change if a branch is removed?

   1. █████████████████████████ ↑ raises 40.9170%
      (-) Radiocarbon dating
      Without this: 42.5325%  (baseline: 1.6156%)

   2. ███████░░░░░░░░░░░░░░░░░░ ↑ raises 12.4242%
      (-) Historical sources
      Without this: 14.0398%  (baseline: 1.6156%)
```

## Create Your Own

Model any yes/no question in 5 lines:

```yaml
node: "Should I take this job offer?"
prior: 0.50
children:
  - node: "50% salary increase"
    lr_min: 2.0
    lr_max: 5.0
    evidence_type: for
  - node: "Longer commute (1.5 hours)"
    lr_min: 0.3
    lr_max: 0.7
    evidence_type: against
  - node: "Company has strong growth trajectory"
    lr_min: 1.5
    lr_max: 3.0
    evidence_type: for
  - node: "Would lose current team I enjoy"
    lr_min: 0.4
    lr_max: 0.8
    evidence_type: against
```

Then run it:

```bash
python bayes-tree-eng.py my_decision.yaml
```

## YAML Format

```yaml
node: "Is the hypothesis true?"
prior: 0.50

children:
  - node: "Supporting evidence A"
    lr_min: 1.5          # likelihood ratio lower bound
    lr_max: 4.0          # likelihood ratio upper bound
    lr_dist: log_uniform # log_uniform (default) | uniform | beta
    evidence_type: for   # for | against | neutral

    children:            # optional sub-evidence (chains sequentially)
      - node: "Counterpoint to A"
        lr_min: 0.3
        lr_max: 0.7
        evidence_type: against

  - node: "Counter-evidence B"
    lr_min: 0.05
    lr_max: 0.25
    evidence_type: against
```

### Fields

| Field | Description |
|-------|-------------|
| `node` | Human-readable label for this evidence |
| `prior` | Prior probability (root node only) |
| `lr_min`, `lr_max` | Likelihood ratio uncertainty interval |
| `lr_dist` | Sampling distribution: `log_uniform` (default), `uniform`, `beta` |
| `likelihood_ratio` | Exact point LR (alternative to min/max interval) |
| `evidence_type` | `for` (LR > 1), `against` (LR < 1), or `neutral` |
| `children` | Sub-evidence nodes (optional) |

### How to Think About Likelihood Ratios

| LR | Meaning | Example |
|----|---------|---------|
| 10+ | Strong support | DNA match at crime scene |
| 2–5 | Moderate support | Witness places suspect nearby |
| 1 | Neutral — no update | Irrelevant information |
| 0.2–0.5 | Moderate counter-evidence | Alibi from one friend |
| < 0.1 | Strong counter-evidence | Verified alibi with video |

## Architecture

- **Root level**: children are combined as independent evidence via log-odds summation
- **Sub-children**: represent drill-down/refinement — parent's posterior becomes child's prior
- **Monte Carlo**: each run samples LRs from their intervals, producing a posterior distribution
- **Effective LR**: the single likelihood ratio that would produce the same combined posterior

## Output Sections

1. **Combined posterior** — median, mean, std, 90% CI, histogram
2. **Evidence tree** — each branch showing prior → posterior with confidence intervals
3. **Sensitivity analysis** — P(posterior < 5%), P(posterior > 50%), etc.
4. **Importance ranking** — leave-one-out: which branch matters most?

## Examples

| File | Hypothesis |
|------|-----------|
| [`god.yaml`](examples/god.yaml) | Does a theistic God exist? |
| [`shroud.yaml`](examples/shroud.yaml) | Is the Shroud of Turin authentic? |
| [`empty_grave.yaml`](examples/empty_grave.yaml) | Is the empty tomb later legend? |
| [`J_grave_historical.yaml`](examples/J_grave_historical.yaml) | Was Jesus's tomb historical and empty? |
| [`jesus_historicity.yaml`](examples/jesus_historicity.yaml) | Did a historical Jesus of Nazareth exist? (prior: 0.33) |
| [`jesus_objective.yaml`](examples/jesus_objective.yaml) | Did a historical Jesus of Nazareth exist? (prior: 0.50) |
| [`moses.yaml`](examples/moses.yaml) | Is there a historical person behind the Moses legend? |
| [`hitler.yaml`](examples/hitler.yaml) | Was Hitler murdered rather than suicide? |
| [`napoleon.yaml`](examples/napoleon.yaml) | Was Napoleon deliberately poisoned with arsenic? |

## Theory

The mathematical foundations (Bayesian updating, log-odds combination, Monte Carlo propagation, limitations) are documented in [`bayes_tree_theory.tex`](bayes_tree_theory.tex).

## License

MIT © 2026 Ari-Pekka Sihvonen
