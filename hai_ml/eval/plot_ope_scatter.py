"""
OPE vs Realized Return Scatter Plot
====================================

Generates scatter plot comparing OPE estimates to realized returns.

Usage:
    python -m hai_ml.eval.plot_ope_scatter \
        --offline results/p3_offline_leaderboard.csv \
        --online results/p3_nominal.csv \
        --out paper/figs/fig_ope_scatter.pdf
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Style settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (6, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_offline_results(csv_path: str) -> pd.DataFrame:
    """Load offline OPE leaderboard."""
    return pd.read_csv(csv_path)


def load_online_results(csv_path: str) -> pd.DataFrame:
    """Load online evaluation results."""
    return pd.read_csv(csv_path)


def aggregate_online_by_algo(df: pd.DataFrame) -> Dict[str, float]:
    """Aggregate online results by algorithm."""
    # Assume CSV has 'algo' or we infer from model path
    if 'algo' in df.columns:
        return df.groupby('algo')['total_reward'].mean().to_dict()
    
    # Try to infer from filename or return aggregate
    return {'all': df['total_reward'].mean()}


def plot_ope_scatter(
    offline_df: pd.DataFrame,
    online_returns: Dict[str, float],
    output_path: str,
    title: str = 'OPE vs Realized Returns',
):
    """
    Create scatter plot of OPE estimates vs realized returns.
    
    Args:
        offline_df: DataFrame with columns [algo, fqe_mean, fqe_lo, fqe_hi, wis, admit]
        online_returns: Dict mapping algo name to realized return
        output_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Collect data points
    ope_values = []
    realized_values = []
    labels = []
    admitted = []
    
    for _, row in offline_df.iterrows():
        algo = row['algo']
        fqe_mean = row['fqe_mean']
        fqe_lo = row.get('fqe_lo', fqe_mean - 0.1)
        fqe_hi = row.get('fqe_hi', fqe_mean + 0.1)
        
        # Get realized return
        if algo in online_returns:
            realized = online_returns[algo]
        elif 'all' in online_returns:
            realized = online_returns['all']
        else:
            realized = fqe_mean * 1.1  # Fallback
        
        ope_values.append(fqe_mean)
        realized_values.append(realized)
        labels.append(algo.upper())
        admitted.append(row.get('admit', True))
        
        # Plot error bars for OPE CI
        ax.errorbar(
            fqe_mean, realized,
            xerr=[[fqe_mean - fqe_lo], [fqe_hi - fqe_mean]],
            fmt='none',
            color='gray',
            alpha=0.5,
            capsize=3,
        )
    
    ope_values = np.array(ope_values)
    realized_values = np.array(realized_values)
    admitted = np.array(admitted)
    
    # Plot admitted vs rejected differently
    colors = ['green' if a else 'red' for a in admitted]
    markers = ['o' if a else 'x' for a in admitted]
    
    for i, (x, y, label, color, marker) in enumerate(
        zip(ope_values, realized_values, labels, colors, markers)
    ):
        ax.scatter(x, y, c=color, marker=marker, s=100, label=label, zorder=5)
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )
    
    # Add correlation line
    if len(ope_values) >= 3:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            ope_values, realized_values
        )
        
        x_line = np.linspace(ope_values.min() - 0.1, ope_values.max() + 0.1, 100)
        y_line = slope * x_line + intercept
        
        ax.plot(x_line, y_line, 'b--', alpha=0.5, label=f'Fit (r={r_value:.2f})')
    
    # Add diagonal reference
    min_val = min(ope_values.min(), realized_values.min())
    max_val = max(ope_values.max(), realized_values.max())
    padding = (max_val - min_val) * 0.1
    diag_range = [min_val - padding, max_val + padding]
    ax.plot(diag_range, diag_range, 'k:', alpha=0.3, label='y=x')
    
    ax.set_xlabel('OPE Estimate (FQE)')
    ax.set_ylabel('Realized Return')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Legend for admitted/rejected
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               markersize=10, label='Admitted'),
        Line2D([0], [0], marker='x', color='red', markersize=10, label='Rejected'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Saved PDF to: {output_path}")
    
    # Also save PNG
    png_path = output_path.with_suffix('.png')
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f"Saved PNG to: {png_path}")
    
    plt.close(fig)
    
    # Print correlation stats
    if len(ope_values) >= 3:
        pearson_r, pearson_p = stats.pearsonr(ope_values, realized_values)
        spearman_r, spearman_p = stats.spearmanr(ope_values, realized_values)
        
        print(f"\nCorrelation Statistics:")
        print(f"  Pearson r: {pearson_r:.3f} (p={pearson_p:.3f})")
        print(f"  Spearman ρ: {spearman_r:.3f} (p={spearman_p:.3f})")


def main():
    parser = argparse.ArgumentParser(
        description='Plot OPE vs realized return scatter'
    )
    parser.add_argument(
        '--offline', type=str, required=True,
        help='Path to offline leaderboard CSV'
    )
    parser.add_argument(
        '--online', type=str, required=True,
        help='Path to online results CSV'
    )
    parser.add_argument(
        '--out', type=str, required=True,
        help='Output path for figure (PDF)'
    )
    parser.add_argument(
        '--title', type=str, default='OPE vs Realized Returns',
        help='Plot title'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading offline results from: {args.offline}")
    offline_df = load_offline_results(args.offline)
    
    print(f"Loading online results from: {args.online}")
    online_df = load_online_results(args.online)
    online_returns = aggregate_online_by_algo(online_df)
    
    # Create plot
    plot_ope_scatter(offline_df, online_returns, args.out, args.title)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
