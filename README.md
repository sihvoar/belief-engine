# Belief Engine

**Make your assumptions explicit. Combine uncertain evidence into a posterior distribution with calibrated uncertainty propagation.**

*Not an AI that thinks for you — a calculator for belief revision that forces you to show your work.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-green.svg)](https://python.org)

---

## Abstract

Belief Engine provides a formal method for evaluating complex hypotheses by decomposing them into hierarchical evidence trees. Each branch carries a likelihood ratio (LR) interval representing the diagnosticity of a piece of evidence. The framework propagates uncertainty through Monte Carlo simulation, yielding posterior probability distributions, sensitivity analyses, and quantitative importance rankings.

The approach bridges informal argumentation and formal probabilistic reasoning — making the logical structure of multi-evidence problems explicit, auditable, and reproducible.

> **Honesty note.** Belief Engine does not eliminate subjectivity — it makes subjective judgments explicit, auditable, and testable. The output is only as good as the likelihood ratios you supply. The tool's value is in *transparency*: it forces you to state your assumptions where others can inspect, challenge, and improve them. Use the sensitivity analysis and adversarial audit to stress-test your inputs.

> **Note on naming**: The Python import package is `bayes_tree` for historical reasons. The PyPI distribution name is `belief-engine` to avoid confusion with the unrelated phylogenetic "BayesTree" method in bioinformatics.

## Motivation

Many important questions in science, history, forensics, and decision-making involve synthesising heterogeneous evidence of varying strength and reliability. Traditional approaches either:

- Rely on qualitative narrative weighing (subjective, non-reproducible), or
- Require full probabilistic graphical models (high expertise barrier).

Belief Engine occupies a practical middle ground: it requires only that the analyst estimate *how diagnostic* each piece of evidence is (expressed as a likelihood ratio interval), then handles the combination and uncertainty propagation computationally. The result is a structured, transparent argument where every assumption is visible and every disagreement can be localized to a specific likelihood ratio.

## Method

### Formal Model

Given a binary hypothesis *H* with prior probability *P(H)*, and *n* conditionally independent evidence branches *E₁, …, Eₙ*, each characterised by a likelihood ratio interval *[LR_min, LR_max]*:

1. **Sampling**: For each simulation run, sample *LRᵢ* from the specified distribution (log-uniform by default) over *[LR_min_i, LR_max_i]*.

2. **Combination**: Combine evidence via log-odds summation:

   ```
   log-odds(posterior) = log-odds(prior) + Σᵢ log(LRᵢ)
   ```

3. **Propagation**: Repeat for *N* Monte Carlo iterations (default: 10,000) to obtain the posterior distribution.

4. **Analysis**: Compute summary statistics, sensitivity bounds, and leave-one-out importance rankings.

### Hierarchical Structure

- **Internal nodes** are pure groupers that organise evidence into logical categories. They must not carry likelihood ratios.
- **Leaf nodes** are the sole carriers of evidence. Each leaf specifies a likelihood ratio interval.
- **The root prior** is the only prior used. All leaf log-LRs are summed once:

  ```
  log-odds(posterior) = log-odds(prior) + Σ_leaves log(LRᵢ)
  ```

- **Subtree contribution** is computed for each internal node (its descendant leaves' combined LR) purely for display — it is not an input to the posterior calculation.

### Semantics

Only leaves contribute evidence to the posterior. Internal nodes exist to provide logical structure and human-readable grouping. The engine validates that no internal node carries an LR; a YAML file with `lr_min` or `lr_max` on a node that also has `children` is rejected with a clear error message.

This design enforces a **single counting rule**: each piece of evidence enters the posterior exactly once, through its leaf node. Duplication that previously arose from internal nodes carrying their own LR alongside children is eliminated.

> **Conditional independence warning.** The log-sum combination assumes that leaf nodes are conditionally independent given *H* and given *¬H*. Sibling leaves that are facets of the same underlying observation (e.g., three laboratories running C14 on samples from the same cloth) violate this assumption. Mitigations: (a) merge such leaves into a single combined leaf; (b) keep only the strongest and drop the rest; or (c) use `correlation_group` and `rho` to model the dependence explicitly via Gaussian copula.

### Supported Distributions

| Distribution | Use case |
|-------------|----------|
| `log_uniform` (default) | Appropriate when LR uncertainty spans orders of magnitude |
| `uniform` | When LR uncertainty is symmetric on a linear scale |
| `beta` | When the analyst has shape information (parameterised via `lr_alpha`, `lr_beta`). **Caveat:** the beta distribution is linearly scaled into `[lr_min, lr_max]`. On wide intervals (e.g., `[0.01, 100]`), this puts most probability mass near `lr_min` in log terms, which is rarely what users want. Prefer `log_uniform` or `log_beta` for wide intervals. |
| `log_beta` | Beta sampling in log-space: `LR = exp(log(lr_min) + Beta(α, β) × (log(lr_max) − log(lr_min)))`. Appropriate when the analyst has shape information and the LR interval spans orders of magnitude. |

## Installation

```bash
pip install belief-engine
```

Or from source:

```bash
git clone https://github.com/sihvoar/belief-engine.git
cd belief-engine
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `pyyaml` | YAML evidence tree parsing |
| `matplotlib` | Visualisation (optional) |
| `reportlab` | PDF report generation (optional) |
| `numpy` | Numerical support (optional) |

## Usage

### Command-Line Interface

```bash
# Standard terminal output with histogram and tree
belief-engine examples/napoleon.yaml

# Machine-readable JSON output
belief-engine examples/napoleon.yaml --format json

# Prior sensitivity sweep
belief-engine examples/napoleon.yaml --prior-sweep

# Adversarial robustness audit
belief-engine examples/napoleon.yaml --adversarial

# Specify simulation count
belief-engine examples/napoleon.yaml -n 50000
```

> **Alias**: `bayes-tree` also works as a CLI command for backward compatibility.

### Python API

```python
from bayes_tree import run_simulation
import yaml

with open("my_analysis.yaml") as f:
    data = yaml.safe_load(f)

results = run_simulation(data, n_sim=10_000)
print(f"Posterior median: {results['stats']['median']:.2%}")
print(f"90% CI: [{results['stats']['p5']:.2%} – {results['stats']['p95']:.2%}]")
```

### GUI

```bash
python scripts/bayes_tree_gui.py                        # interactive editor
python scripts/bayes_tree_gui.py examples/shroud.yaml   # open existing analysis
```

### Web Demo

```bash
streamlit run scripts/streamlit_app.py
```

## YAML Specification

```yaml
node: "Is the hypothesis H true?"
prior: 0.50                    # P(H) — prior probability

children:
  - node: "Evidence group A"   # internal node (grouper, no LR)
    evidence_type: for         # for display only

    children:                  # leaf children carry the actual evidence
      - node: "Supporting observation"
        lr_min: 1.5
        lr_max: 4.0
        lr_dist: log_uniform   # log_uniform | uniform | beta | log_beta
        evidence_type: for     # for (LR > 1) | against (LR < 1) | neutral

      - node: "Counterpoint weakening A"
        lr_min: 0.3
        lr_max: 0.7
        evidence_type: against

  - node: "C14 date lab 1"    # leaf node (carries evidence)
    lr_min: 5.0
    lr_max: 20.0
    evidence_type: for
    correlation_group: c14     # shared flaw → correlated sampling
    rho: 0.7                   # Pearson correlation (0–1)

  - node: "C14 date lab 2"
    lr_min: 5.0
    lr_max: 20.0
    evidence_type: for
    correlation_group: c14     # same group = Gaussian copula
    rho: 0.7

  - node: "Evidence B against H"
    lr_min: 0.05
    lr_max: 0.25
    evidence_type: against
```

> **Rule**: A node with `children` must not specify `lr_min`, `lr_max`, or `likelihood_ratio`. Internal nodes carry only `node` (label), `evidence_type` (for display), and `children`. This constraint is enforced at parse time to prevent double-counting: if an internal node carried its own LR alongside its children's LRs, that evidence would enter the posterior twice.

### Correlation Groups

When multiple evidence branches share a common methodological flaw (e.g., two radiocarbon dates from the same lab, or witness testimonies from the same source), tag them with a `correlation_group` and `rho` parameter. The engine uses a Gaussian copula to correlate their LR samples, widening the posterior uncertainty to reflect that they do not provide fully independent information.

### Likelihood Ratio Interpretation

| LR | Evidential strength | Equivalent |
|----|--------------------:|-----------|
| > 100 | Decisive for *H* | Near-certain proof |
| 10–100 | Strong for *H* | DNA match, verified document |
| 2–10 | Moderate for *H* | Credible testimony, correlational data |
| 1 | Neutral | No evidential value |
| 0.1–0.5 | Moderate against *H* | Disconfirming witness, failed prediction |
| 0.01–0.1 | Strong against *H* | Robust replication failure |
| < 0.01 | Decisive against *H* | Definitive refutation |

## Output

The framework produces:

1. **Posterior distribution** — full Monte Carlo sample with median, mean, standard deviation, and credible intervals. *Note: this distribution reflects uncertainty in the posterior induced by the user-specified LR intervals and distribution families. It is not a hierarchical Bayesian model over the likelihood ratios themselves — the LR intervals are treated as fixed inputs, and Monte Carlo sampling propagates that interval uncertainty into a distribution over possible posteriors.*
2. **Effective likelihood ratio** — the single LR equivalent to the combined evidence.
3. **Evidence tree** — per-branch prior → posterior transformation with uncertainty.
4. **Sensitivity analysis** — threshold exceedance probabilities.
5. **Importance ranking** — leave-one-out analysis identifying which evidence most influences the conclusion.
6. **Adversarial audit** — systematic robustness testing against plausible attacks on assumptions (prior bias, LR calibration, distribution misspecification, evidence correlation).

### Example Output (Napoleon poisoning hypothesis)

```
Posterior: median 6.6%, mean 8.6%, 90% CI [1.8%–22.2%]
Effective LR: 0.071 (strong evidence against deliberate poisoning)

Importance ranking:
  1. Scheele's Green wallpaper explains arsenic   Δ = +20.3%
  2. Autopsy showed stomach cancer                Δ = +15.6%
  3. DNA/isotope analyses (2021) support natural   Δ = +15.0%
```

## Case Studies

### Data-grounded examples

These examples use likelihood ratios derived from empirical data (test performance, experimental results). They are the best starting point for learning the tool.

| Analysis | Hypothesis | Branches | Posterior |
|----------|-----------|:--------:|-----------|
| [`medical_test.yaml`](examples/medical_test.yaml) | Patient has disease X given positive screen | 7 | varies |
| [`ab_test.yaml`](examples/ab_test.yaml) | A/B treatment genuinely improves conversion | 7 | varies |
| [`napoleon.yaml`](examples/napoleon.yaml) | Napoleon was deliberately poisoned | 8 | ~7% |
| [`hitler.yaml`](examples/hitler.yaml) | Hitler was murdered (not suicide) | 6 | varies |

### Illustrative / advanced — contested hypotheses

The examples below apply the framework to contested religious and historical hypotheses where **likelihood ratios are necessarily subjective**. They are useful for exploring the tool's capabilities and for practicing structured disagreement, but the LRs should be treated as a starting point for argument — not a conclusion. Reasonable people will assign very different LRs to these branches; the value is in making those disagreements explicit and quantifiable.

| Analysis | Hypothesis | Branches | Posterior |
|----------|-----------|:--------:|-----------|
| [`god.yaml`](examples/god.yaml) | A theistic God exists | 10 | varies |
| [`shroud.yaml`](examples/shroud.yaml) | Shroud of Turin is authentic | 6 | ~2% |
| [`moses.yaml`](examples/moses.yaml) | Historical person behind Moses legend | 7 | varies |
| [`jesus_historicity.yaml`](examples/jesus_historicity.yaml) | Historical Jesus existed | 8 | varies |
| [`empty_grave.yaml`](examples/empty_grave.yaml) | Empty tomb is later legend | 5 | varies |

## Theoretical Foundations

The mathematical framework — Bayesian updating, log-odds combination, conditional independence assumptions, Monte Carlo uncertainty propagation, and known limitations — is documented in [`doc/bayes_tree_theory.tex`](doc/bayes_tree_theory.tex).

### Key Properties

- **Commutativity**: Evidence order does not affect the posterior.
- **Associativity**: Grouping of branches is irrelevant to the combined result.
- **Monotonicity**: Adding evidence with LR > 1 strictly increases the posterior; LR < 1 strictly decreases it.
- **Prior sensitivity**: The prior sweep analysis quantifies how robust conclusions are to prior specification.

### Assumptions and Limitations

- **Conditional independence**: Leaf nodes are assumed conditionally independent given *H* and *¬H*. Evidence sharing a common methodological flaw should be either merged into a single leaf, or tagged with `correlation_group` and `rho` for copula-based correlation modeling. The adversarial audit can quantify the impact of unmodeled correlations.
- **Leaves-only evidence**: Only leaf nodes carry likelihood ratios. Internal nodes are pure groupers. This design prevents double-counting that arises when both a parent node and its children carry LRs.
- **Analyst calibration**: Results are only as good as the LR estimates. The tool makes reasoning transparent but cannot verify inputs.
- **Binary hypotheses**: The current framework evaluates *H* vs *¬H*. Multi-hypothesis extensions are planned.

### Independence — the assumption you're most likely to get wrong

The log-odds combination assumes that leaf evidence nodes are conditionally independent given *H* and *¬H*. In practice, this is the hardest assumption to satisfy because evidence can share **latent common causes** that are not immediately obvious:

- **Shared information source**: Three "independent" eyewitnesses who all heard the same rumor before testifying are not independent — their testimonies are correlated through the rumor.
- **Shared dataset**: Multiple published studies that drew from the same underlying patient registry or data warehouse produce correlated results, even if the authors were unaware of each other.
- **Shared methodology**: Two archaeological dating methods that both rely on the same calibration curve share a systematic error component.

The `correlation_group` / `rho` mechanism helps when you *know* about the dependence. But latent common causes — the ones you don't know about — are the real risk. Mitigations:

1. **Ask explicitly**: For every pair of strong evidence branches, ask "Could these share a common information source, dataset, or methodological assumption?"
2. **Run the adversarial correlation attack** (`--adversarial`): it systematically injects pairwise correlations and reports how much the posterior shifts. Large shifts indicate fragile independence assumptions.
3. **When in doubt, merge**: If two pieces of evidence might be facets of the same underlying observation, combine them into a single leaf with a conservative LR.

## Validation

The framework includes a comprehensive 50-test validation suite covering:

- Exact Bayesian update arithmetic
- Numerical stability at extreme values
- Input validation and error handling
- Distribution sampling correctness
- Mathematical properties (commutativity, monotonicity, symmetry)
- API contract verification
- Stress testing (50+ branches)

```bash
python validation/run_validation.py    # run all tests
python validation/generate_report.py   # generate PDF report
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## Citation

If you use Belief Engine in academic work, please cite:

```bibtex
@software{sihvonen2026beliefengine,
  author  = {Sihvonen, Ari-Pekka},
  title   = {Belief Engine: Structured Bayesian Argumentation with Monte Carlo Uncertainty Propagation},
  year    = {2026},
  url     = {https://github.com/sihvoar/belief-engine},
  license = {MIT}
}
```

## License

MIT © 2026 Ari-Pekka Sihvonen
