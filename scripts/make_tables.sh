#!/bin/bash
# ==============================================================================
# Generate LaTeX Tables from Results
# ==============================================================================
# Processes CSV results and generates LaTeX table code for the paper.
#
# Usage:
#   ./scripts/make_tables.sh [process]
#
# Arguments:
#   process     Process to use: p1, p3, p12 (default: p3)
#
# Output:
#   - paper/tables/tab_offline_leaderboard.tex
#   - paper/tables/tab_online_performance.tex
#   - paper/tables/tab_attack_robustness.tex
#   - paper/tables/tab_cross_version.tex (if available)
# ==============================================================================

set -e

PROCESS=${1:-p3}

echo "=============================================="
echo "Generating LaTeX Tables"
echo "Process: $PROCESS"
echo "=============================================="

# Create output directory
mkdir -p paper/tables

# Python script for table generation
python3 << 'PYTHON_SCRIPT'
import sys
import pandas as pd
import numpy as np
from pathlib import Path

process = "${PROCESS}" if "${PROCESS}" != "" else "p3"

# Helper to format number with std
def fmt_mean_std(mean, std, precision=2):
    return f"${mean:.{precision}f} \\pm {std:.{precision}f}$"

def fmt_number(val, precision=2):
    return f"${val:.{precision}f}$"

# =============================================================================
# Table 1: Offline Leaderboard (OPE Results)
# =============================================================================
print("Generating Table 1: Offline Leaderboard...")

leaderboard_path = f"results/{process}_offline_leaderboard.csv"
if Path(leaderboard_path).exists():
    df = pd.read_csv(leaderboard_path)
    
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Off-Policy Evaluation Leaderboard. FQE estimates with 95\% bootstrap confidence intervals. Policies are admitted if lower CI $> 0$ and exceeds BC baseline.}")
    latex.append(r"\label{tab:offline_leaderboard}")
    latex.append(r"\begin{tabular}{lcccc}")
    latex.append(r"\toprule")
    latex.append(r"Algorithm & FQE Estimate & 95\% CI & WIS & Admit \\")
    latex.append(r"\midrule")
    
    for _, row in df.iterrows():
        algo = row['algo'].upper()
        fqe = fmt_number(row['fqe_mean'])
        ci = f"[{row['fqe_lo']:.2f}, {row['fqe_hi']:.2f}]"
        wis = fmt_number(row['wis'])
        admit = r"\cmark" if row['admit'] else r"\xmark"
        
        latex.append(f"{algo} & {fqe} & {ci} & {wis} & {admit} \\\\")
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    with open(f"paper/tables/tab_offline_leaderboard.tex", "w") as f:
        f.write("\n".join(latex))
    print(f"  Saved: paper/tables/tab_offline_leaderboard.tex")
else:
    print(f"  Warning: {leaderboard_path} not found, skipping")

# =============================================================================
# Table 2: Online Performance (Nominal)
# =============================================================================
print("Generating Table 2: Online Performance...")

nominal_path = f"results/{process}_nominal.csv"
if Path(nominal_path).exists():
    df = pd.read_csv(nominal_path)
    
    # Aggregate by algorithm
    agg = df.groupby('algo').agg({
        'total_reward': ['mean', 'std'],
        'itae': ['mean', 'std'],
        'violations': ['mean', 'std'],
        'interventions': ['mean', 'std'],
        'latency_mean': ['mean'],
    }).reset_index()
    
    # Flatten column names
    agg.columns = ['_'.join(col).strip('_') for col in agg.columns.values]
    
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Online evaluation under nominal conditions. Metrics averaged over 50 episodes.}")
    latex.append(r"\label{tab:online_performance}")
    latex.append(r"\begin{tabular}{lccccc}")
    latex.append(r"\toprule")
    latex.append(r"Algorithm & Return & ITAE & Violations & Shield Int. & Latency (ms) \\")
    latex.append(r"\midrule")
    
    for _, row in agg.iterrows():
        algo = row['algo'].upper()
        ret = fmt_mean_std(row['total_reward_mean'], row['total_reward_std'])
        itae = fmt_mean_std(row['itae_mean'], row['itae_std'])
        viol = fmt_mean_std(row['violations_mean'], row['violations_std'], 1)
        interv = fmt_mean_std(row['interventions_mean'], row['interventions_std'], 1)
        lat = fmt_number(row['latency_mean_mean'], 2)
        
        latex.append(f"{algo} & {ret} & {itae} & {viol} & {interv} & {lat} \\\\")
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    with open(f"paper/tables/tab_online_performance.tex", "w") as f:
        f.write("\n".join(latex))
    print(f"  Saved: paper/tables/tab_online_performance.tex")
else:
    print(f"  Warning: {nominal_path} not found, skipping")

# =============================================================================
# Table 3: Attack Robustness
# =============================================================================
print("Generating Table 3: Attack Robustness...")

attacks_path = f"results/{process}_attacks.csv"
if Path(attacks_path).exists():
    df = pd.read_csv(attacks_path)
    
    # Pivot by algorithm and attack type
    pivot = df.pivot_table(
        values=['total_reward', 'violations', 'intervention_rate'],
        index='algo',
        columns='attack_type',
        aggfunc='mean'
    ).round(2)
    
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Performance under adversarial attacks. Shield intervention rate in parentheses.}")
    latex.append(r"\label{tab:attack_robustness}")
    latex.append(r"\resizebox{\columnwidth}{!}{%")
    latex.append(r"\begin{tabular}{lcccc}")
    latex.append(r"\toprule")
    latex.append(r"Algorithm & Hostile & Flood & Bias & Delay \\")
    latex.append(r"\midrule")
    
    for algo in pivot.index:
        row_data = [algo.upper()]
        for attack in ['hostile', 'flood', 'bias', 'delay']:
            try:
                ret = pivot.loc[algo, ('total_reward', attack)]
                rate = pivot.loc[algo, ('intervention_rate', attack)]
                row_data.append(f"${ret:.1f}$ ({rate:.1%})")
            except KeyError:
                row_data.append("--")
        latex.append(" & ".join(row_data) + r" \\")
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}}")
    latex.append(r"\end{table}")
    
    with open(f"paper/tables/tab_attack_robustness.tex", "w") as f:
        f.write("\n".join(latex))
    print(f"  Saved: paper/tables/tab_attack_robustness.tex")
else:
    print(f"  Warning: {attacks_path} not found, skipping")

# =============================================================================
# Table 4: Cross-Version Transfer (if available)
# =============================================================================
print("Generating Table 4: Cross-Version Transfer...")

cross_path = f"results/{process}_cross_version.csv"
if Path(cross_path).exists():
    df = pd.read_csv(cross_path)
    
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Cross-version transfer performance. Policies trained on v1, evaluated on both versions.}")
    latex.append(r"\label{tab:cross_version}")
    latex.append(r"\begin{tabular}{lccc}")
    latex.append(r"\toprule")
    latex.append(r"Algorithm & Same Version & Cross Version & $\Delta$ (\%) \\")
    latex.append(r"\midrule")
    
    for algo in df['algo'].unique():
        subset = df[df['algo'] == algo]
        same = subset[subset['version'] == 'same']['total_reward'].mean()
        cross = subset[subset['version'] == 'cross']['total_reward'].mean()
        delta = (cross - same) / same * 100
        
        latex.append(f"{algo.upper()} & {same:.2f} & {cross:.2f} & {delta:+.1f}\\% \\\\")
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    with open(f"paper/tables/tab_cross_version.tex", "w") as f:
        f.write("\n".join(latex))
    print(f"  Saved: paper/tables/tab_cross_version.tex")
else:
    print(f"  Info: {cross_path} not found (cross-version evaluation not run)")

print("\nDone!")
PYTHON_SCRIPT

# Substitute PROCESS variable
sed -i "s/\${PROCESS}/$PROCESS/g" paper/tables/*.tex 2>/dev/null || true

echo ""
echo "=============================================="
echo "Table Generation Complete!"
echo "=============================================="
echo ""
echo "Generated tables in paper/tables/:"
ls -la paper/tables/*.tex 2>/dev/null || echo "  (no tables generated - run evaluation first)"
echo ""
echo "Include in LaTeX with:"
echo "  \\input{tables/tab_offline_leaderboard}"
echo "  \\input{tables/tab_online_performance}"
echo "  \\input{tables/tab_attack_robustness}"
echo ""
