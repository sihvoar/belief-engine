#!/usr/bin/env python3
"""
Bayes Tree CLI — command-line interface with JSON output and prior sweep.

Usage:
    bayes-tree analysis.yaml                    # default text output
    bayes-tree analysis.yaml --format json      # JSON output
    bayes-tree analysis.yaml --prior-sweep      # sensitivity to prior
    bayes-tree analysis.yaml -n 50000           # more simulations
    bayes-tree analysis.yaml --verbose          # full precision output
    bayes-tree calibrate                        # show reference LR table
"""

from __future__ import annotations

import argparse
import json
import sys
import math

import yaml

from bayes_tree.engine import (
    run_simulation, sim_root, sts, validate_node,
    bayes_upd, to_lo, from_lo, post_to_lr, NodeResult,
)


# ── Sig-fig formatting ────────────────────────────────────────────────────────

def _sigfig_pct(value: float, n: int = 2) -> str:
    """Format a probability as a percentage with n significant figures."""
    pct_val = value * 100
    if pct_val == 0:
        return "0%"
    if pct_val >= 100:
        return "100%"
    magnitude = math.floor(math.log10(abs(pct_val)))
    decimals = max(0, n - 1 - magnitude)
    return f"{pct_val:.{decimals}f}%"


def _sigfig(value: float, n: int = 2) -> str:
    """Format a number with n significant figures."""
    if value == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(value)))
    decimals = max(0, n - 1 - magnitude)
    return f"{value:.{decimals}f}"


# ── Calibration reference table ───────────────────────────────────────────────

CALIBRATION_TABLE = [
    ("Forensic DNA match (single locus)", "~100–10,000", "for"),
    ("Full DNA profile match", "~10⁶–10⁹", "for"),
    ("Mammogram positive (cancer screening)", "~7–15", "for"),
    ("Rapid antigen test positive (COVID-like)", "~10–30", "for"),
    ("Single credible eyewitness ID", "~2–10", "for"),
    ("Fingerprint match (verified examiner)", "~50–500", "for"),
    ("Failed replication of key prediction", "~0.05–0.2", "against"),
    ("Alibi confirmed by independent CCTV", "~0.01–0.05", "against"),
    ("Controlled RCT, significant result (p<0.01)", "~5–50", "for"),
    ("Published meta-analysis (strong effect)", "~10–100", "for"),
    ("Anecdotal report / single testimony", "~1.5–3", "for"),
    ("Expert opinion without data", "~1.5–5", "for/against"),
]

# Threshold for extreme log-odds shift warning (3 orders of magnitude)
_EXTREME_LOG_ODDS_THRESHOLD = math.log(1000)  # ~6.9


def _node_to_dict(nr: NodeResult) -> dict:
    """Convert NodeResult tree to JSON-serializable dict."""
    return {
        "name": nr.name,
        "evidence_type": nr.etype,
        "lr_min": nr.lr_min,
        "lr_max": nr.lr_max,
        "lr_point": nr.lr_pt,
        "lr_derived": nr.lr_derived,
        "prior": nr.prior,
        "posterior_median": nr.med,
        "posterior_p5": nr.p5,
        "posterior_p95": nr.p95,
        "children": [_node_to_dict(c) for c in nr.children],
    }


def _format_json(results: dict, data: dict) -> str:
    """Format results as JSON."""
    output = {
        "hypothesis": data.get("node", "Unknown"),
        "prior": results["prior"],
        "n_sim": results["n_sim"],
        "posterior": {
            "median": results["stats"]["median"],
            "mean": results["stats"]["mean"],
            "std": results["stats"]["std"],
            "p5": results["stats"]["p5"],
            "p95": results["stats"]["p95"],
            "min": results["stats"]["min"],
            "max": results["stats"]["max"],
        },
        "effective_lr": {
            "median": results["lr_stats"]["median"],
            "p5": results["lr_stats"]["p5"],
            "p95": results["lr_stats"]["p95"],
        },
        "sensitivity": results["sensitivity"],
        "importance": results["importance"],
        "warnings": results["warnings"],
        "tree": _node_to_dict(results["tree"]),
    }
    return json.dumps(output, indent=2, ensure_ascii=False)


def _format_csv(results: dict, data: dict) -> str:
    """Format summary as CSV row."""
    s = results["stats"]
    lr = results["lr_stats"]
    lines = [
        "hypothesis,prior,n_sim,median,mean,std,p5,p95,lr_median,lr_p5,lr_p95",
        (f'"{data.get("node", "")}",{results["prior"]},{results["n_sim"]},'
         f'{s["median"]:.6f},{s["mean"]:.6f},{s["std"]:.6f},'
         f'{s["p5"]:.6f},{s["p95"]:.6f},'
         f'{lr["median"]:.6f},{lr["p5"]:.6f},{lr["p95"]:.6f}'),
    ]
    return "\n".join(lines)


def _prior_sweep(data: dict, n_sim: int, steps: int = 19) -> dict:
    """Run simulation across a range of priors."""
    priors = [round(0.05 + i * 0.05, 2) for i in range(steps)]
    sweep_results = []
    for prior in priors:
        sweep_data = {**data, "prior": prior}
        posteriors = [sim_root(sweep_data)[0] for _ in range(n_sim)]
        s = sts(posteriors)
        sweep_results.append({
            "prior": prior,
            "median": s["median"],
            "mean": s["mean"],
            "p5": s["p5"],
            "p95": s["p95"],
        })
    return {"sweep": sweep_results}


def _format_prior_sweep_text(sweep: dict) -> str:
    """Format prior sweep as ASCII table."""
    lines = [
        "",
        "PRIOR SENSITIVITY SWEEP",
        "═" * 55,
        f"{'Prior':>7}  {'Median':>8}  {'90% CI':>20}  {'Bar'}",
        "─" * 55,
    ]
    for row in sweep["sweep"]:
        bar_len = int(row["median"] * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        lines.append(
            f"  {row['prior']:>5.0%}  {row['median']:>7.3%}  "
            f"[{row['p5']:.3%}–{row['p95']:.3%}]  {bar}"
        )
    lines.append("")
    return "\n".join(lines)


def _format_prior_sweep_json(sweep: dict, data: dict) -> str:
    """Format prior sweep as JSON."""
    return json.dumps({
        "hypothesis": data.get("node", "Unknown"),
        "prior_sweep": sweep["sweep"],
    }, indent=2)


# ── Text output (matches original bayes-tree-eng.py style) ───────────────────

def _format_text(results: dict, data: dict, verbose: bool = False) -> str:
    """Format results as colored terminal text.

    Default: 2 significant figures for readability.
    Verbose: full precision (3+ decimal places).
    """
    s = results["stats"]
    lr = results["lr_stats"]
    lines = [
        "",
        "\033[1m\033[96mBAYESIAN DECISION TREE  +  MONTE CARLO\033[0m",
        f"\033[90mSimulations: {results['n_sim']:,}\033[0m",
        "─" * 55,
    ]

    # Warnings
    if results["warnings"]:
        lines.append("")
        lines.append("\033[93m\033[1mWARNINGS:\033[0m")
        for w in results["warnings"]:
            lines.append(f"\033[93m{w}\033[0m")
        lines.append("")

    # Extreme log-odds shift warning
    median_lr = lr["median"]
    if median_lr > 0:
        abs_log_shift = abs(math.log(max(median_lr, 1e-12)))
        if abs_log_shift > _EXTREME_LOG_ODDS_THRESHOLD:
            lines.append("")
            lines.append(
                "\033[93m  ⚠  Total evidence shift exceeds 3 orders of "
                "magnitude (effective LR"
            )
            lines.append(
                f"     = {_sigfig(median_lr)}). Verify that inputs are "
                "not double-counted or overconfident.\033[0m"
            )
            lines.append("")

    # Posterior
    col = "\033[92m" if s["median"] >= 0.5 else (
        "\033[93m" if s["median"] >= 0.25 else "\033[91m")
    lines.append("")
    lines.append(f"  Hypothesis: \033[1m{data.get('node', 'Unknown')}\033[0m")

    if verbose:
        lines.append(f"  Prior:      \033[96m{results['prior']:.1%}\033[0m")
        lines.append(f"  Median:     {col}{s['median']:.3%}\033[0m")
        lines.append(f"  Mean:       {col}{s['mean']:.3%}\033[0m")
        lines.append(f"  Std:        \033[90m{s['std']:.3%}\033[0m")
        lines.append(f"  90% CI:     \033[90m[{s['p5']:.3%}–{s['p95']:.3%}]\033[0m")
        lines.append(f"  Eff. LR:    \033[96m{lr['median']:.4f} "
                     f"[{lr['p5']:.4f}–{lr['p95']:.4f}]\033[0m")
    else:
        lines.append(f"  Prior:      \033[96m{_sigfig_pct(results['prior'])}\033[0m")
        lines.append(f"  Median:     {col}{_sigfig_pct(s['median'])}\033[0m")
        lines.append(f"  Mean:       {col}{_sigfig_pct(s['mean'])}\033[0m")
        lines.append(f"  Std:        \033[90m{_sigfig_pct(s['std'])}\033[0m")
        lines.append(f"  90% CI:     \033[90m[{_sigfig_pct(s['p5'])}–"
                     f"{_sigfig_pct(s['p95'])}]\033[0m")
        lines.append(f"  Eff. LR:    \033[96m{_sigfig(lr['median'])} "
                     f"[{_sigfig(lr['p5'])}–{_sigfig(lr['p95'])}]\033[0m")

    # Sensitivity
    sens = results["sensitivity"]
    lines.append("")
    lines.append("─" * 55)
    lines.append("\033[1mSENSITIVITY\033[0m")
    lines.append(f"  P(posterior < 5%)  = \033[91m{sens['p_lt_5']:.1%}\033[0m")
    lines.append(f"  P(posterior < 10%) = \033[93m{sens['p_lt_10']:.1%}\033[0m")
    lines.append(f"  P(posterior > 50%) = \033[92m{sens['p_gt_50']:.1%}\033[0m")

    # Importance
    if results["importance"]:
        lines.append("")
        lines.append("─" * 55)
        lines.append("\033[1mIMPORTANCE RANKING (leave-one-out)\033[0m")
        baseline = results["baseline"]
        for rank, item in enumerate(results["importance"], 1):
            arrow = "↑" if item["delta"] > 0 else "↓"
            col2 = "\033[92m" if item["delta"] > 0 else "\033[91m"
            if verbose:
                delta_str = f"{abs(item['delta']):.4%}"
            else:
                delta_str = _sigfig_pct(abs(item["delta"]))
            lines.append(
                f"  {rank:>2}. {col2}{arrow} {delta_str}\033[0m  "
                f"{item['name']}"
            )

    lines.append("")
    return "\n".join(lines)


def _format_calibrate() -> str:
    """Format the calibration reference table."""
    lines = [
        "",
        "\033[1m\033[96mCALIBRATION REFERENCE — Likelihood Ratios from Known Domains\033[0m",
        "",
        "Use these as anchors when estimating LRs for your evidence branches.",
        "A good LR estimate should be defensible by comparison to a reference class.",
        "",
        "─" * 70,
        f"  {'Evidence type':<48} {'LR range':<14} {'Dir'}",
        "─" * 70,
    ]
    for desc, lr_range, direction in CALIBRATION_TABLE:
        col = "\033[92m" if "for" in direction else "\033[91m"
        lines.append(f"  {desc:<48} {col}{lr_range:<14}\033[0m {direction}")
    lines.append("─" * 70)
    lines.append("")
    lines.append("  \033[90mRemember: LR=2 is weak, LR=10 is moderate, LR=100 is strong.")
    lines.append("  Most real-world non-forensic evidence falls in the 1.5–10 range.\033[0m")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        prog="bayes-tree",
        description="Bayesian evidence trees with Monte Carlo simulation",
    )
    parser.add_argument("file", nargs="?", default=None,
                        help="YAML evidence tree file, or 'calibrate' for reference LRs")
    parser.add_argument("-n", "--simulations", type=int, default=10_000,
                        help="Number of Monte Carlo simulations (default: 10000)")
    parser.add_argument("-f", "--format", choices=["text", "json", "csv"],
                        default="text", help="Output format (default: text)")
    parser.add_argument("--prior-sweep", action="store_true",
                        help="Run sensitivity sweep across priors 0.05–0.95")
    parser.add_argument("--adversarial", action="store_true",
                        help="Run adversarial audit to find vulnerabilities")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show full-precision output (default: 2 sig figs)")
    parser.add_argument("--version", action="version",
                        version="%(prog)s 1.1.0")
    args = parser.parse_args()

    # Calibrate subcommand
    if args.file == "calibrate":
        print(_format_calibrate())
        return

    if args.file is None:
        parser.print_help()
        sys.exit(1)

    # Load YAML
    try:
        with open(args.file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: invalid YAML: {e}", file=sys.stderr)
        sys.exit(1)

    # Adversarial audit mode
    if args.adversarial:
        from bayes_tree.adversarial import run_audit
        audit = run_audit(data, n_sim=min(args.simulations, 5000))
        if args.format == "json":
            print(audit.to_json())
        else:
            print(audit.summary())
        return

    # Prior sweep mode
    if args.prior_sweep:
        sweep = _prior_sweep(data, args.simulations)
        if args.format == "json":
            print(_format_prior_sweep_json(sweep, data))
        elif args.format == "csv":
            print("prior,median,mean,p5,p95")
            for row in sweep["sweep"]:
                print(f'{row["prior"]},{row["median"]:.6f},{row["mean"]:.6f},'
                      f'{row["p5"]:.6f},{row["p95"]:.6f}')
        else:
            print(_format_prior_sweep_text(sweep))
        return

    # Normal simulation
    try:
        results = run_simulation(data, n_sim=args.simulations)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.format == "json":
        print(_format_json(results, data))
    elif args.format == "csv":
        print(_format_csv(results, data))
    else:
        print(_format_text(results, data, verbose=args.verbose))


if __name__ == "__main__":
    main()
