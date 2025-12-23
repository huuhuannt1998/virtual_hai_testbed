#!/usr/bin/env python3
"""
Visualization Script for LLM Attack Defense Evaluation

Generates the "money plot" showing:
- Water Level trajectory over time
- CUSUM Score accumulation
- Detection point marker
- Comparison between Baseline vs Defense runs

Author: HAI Testbed Research
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np


# =============================================================================
# Data Loading
# =============================================================================

@dataclass
class ExperimentData:
    """Loaded experiment data from CSV."""
    name: str
    steps: List[int]
    elapsed_sec: List[float]
    water_level: List[float]
    cusum_score: List[float]
    cusum_alarm: List[bool]
    safe_hold_active: List[bool]
    shield_action: List[str]
    llm_proposed: List[float]
    shield_applied: List[float]
    first_alarm_idx: Optional[int] = None


def load_csv(filepath: str) -> ExperimentData:
    """Load experiment data from CSV file."""
    steps = []
    elapsed_sec = []
    water_level = []
    cusum_score = []
    cusum_alarm = []
    safe_hold_active = []
    shield_action = []
    llm_proposed = []
    shield_applied = []
    first_alarm_idx = None
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            steps.append(int(row['Step']))
            elapsed_sec.append(float(row['Elapsed_Sec']))
            water_level.append(float(row['Water_Level']))
            cusum_score.append(float(row['CUSUM_Score']))
            alarm = row['CUSUM_Alarm'].lower() == 'true'
            cusum_alarm.append(alarm)
            safe_hold_active.append(row['Safe_Hold_Active'].lower() == 'true')
            shield_action.append(row['Shield_Action'])
            llm_proposed.append(float(row['LLM_Proposed']))
            shield_applied.append(float(row['Shield_Applied']))
            
            if alarm and first_alarm_idx is None:
                first_alarm_idx = i
    
    name = os.path.basename(filepath).replace('_log.csv', '')
    
    return ExperimentData(
        name=name,
        steps=steps,
        elapsed_sec=elapsed_sec,
        water_level=water_level,
        cusum_score=cusum_score,
        cusum_alarm=cusum_alarm,
        safe_hold_active=safe_hold_active,
        shield_action=shield_action,
        llm_proposed=llm_proposed,
        shield_applied=shield_applied,
        first_alarm_idx=first_alarm_idx,
    )


def load_summary(filepath: str) -> Dict:
    """Load A/B test summary JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_money_graph(
    baseline: ExperimentData,
    defense: ExperimentData,
    summary: Optional[Dict] = None,
    output_path: str = "money_plot.png",
    title: str = "LLM Attack Defense Evaluation",
):
    """
    Generate the "money plot" comparing baseline vs defense.
    
    X-Axis: Time (Seconds)
    Y-Axis (Left): Water Level (%)
    Y-Axis (Right): CUSUM Score
    
    Shows:
    - Baseline water level trajectory (dashed line - shows attack success)
    - Defense water level trajectory (solid line - shows mitigation)
    - CUSUM score accumulation
    - Detection point marker (vertical line with annotation)
    """
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Color scheme
    COLOR_BASELINE_LEVEL = '#e74c3c'   # Red
    COLOR_DEFENSE_LEVEL = '#27ae60'    # Green
    COLOR_CUSUM = '#3498db'            # Blue
    COLOR_DANGER = '#c0392b'           # Dark red
    COLOR_DETECTION = '#f39c12'        # Orange
    COLOR_SAFE_HOLD = '#9b59b6'        # Purple
    
    # =========================================================================
    # Left Y-Axis: Water Level
    # =========================================================================
    
    # Baseline trajectory (attack without CUSUM)
    line_baseline = ax1.plot(
        baseline.elapsed_sec,
        baseline.water_level,
        color=COLOR_BASELINE_LEVEL,
        linestyle='--',
        linewidth=2.5,
        alpha=0.8,
        label='Water Level (Baseline - Shield Only)'
    )
    
    # Defense trajectory (with CUSUM)
    line_defense = ax1.plot(
        defense.elapsed_sec,
        defense.water_level,
        color=COLOR_DEFENSE_LEVEL,
        linestyle='-',
        linewidth=2.5,
        label='Water Level (Defense - Shield + CUSUM)'
    )
    
    # Danger zone (above 85%)
    ax1.axhspan(85, 100, color=COLOR_DANGER, alpha=0.15, label='Danger Zone (>85%)')
    ax1.axhline(y=90, color=COLOR_DANGER, linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.text(
        baseline.elapsed_sec[-1] * 0.02, 91,
        'Attack Success Threshold (90%)',
        color=COLOR_DANGER,
        fontsize=9,
        fontweight='bold'
    )
    
    # Safe zone reference
    ax1.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax1.text(
        baseline.elapsed_sec[-1] * 0.02, 51,
        'Nominal Setpoint (50%)',
        color='gray',
        fontsize=8
    )
    
    ax1.set_xlabel('Time (Seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Water Level (%)', fontsize=12, fontweight='bold', color='black')
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    
    # =========================================================================
    # Right Y-Axis: CUSUM Score
    # =========================================================================
    
    ax2 = ax1.twinx()
    
    # CUSUM score (defense run only, since baseline has CUSUM disabled)
    line_cusum = ax2.fill_between(
        defense.elapsed_sec,
        0,
        defense.cusum_score,
        color=COLOR_CUSUM,
        alpha=0.25,
        label='CUSUM Score'
    )
    ax2.plot(
        defense.elapsed_sec,
        defense.cusum_score,
        color=COLOR_CUSUM,
        linewidth=2,
        alpha=0.8
    )
    
    # CUSUM threshold line
    cusum_threshold = 4.0  # From experiment config
    ax2.axhline(
        y=cusum_threshold,
        color=COLOR_CUSUM,
        linestyle='--',
        linewidth=2,
        alpha=0.7
    )
    max_cusum = max(defense.cusum_score) if defense.cusum_score else cusum_threshold
    ax2.text(
        baseline.elapsed_sec[-1] * 0.98,
        cusum_threshold + 0.3,
        f'CUSUM Threshold (h={cusum_threshold})',
        color=COLOR_CUSUM,
        fontsize=9,
        fontweight='bold',
        ha='right'
    )
    
    ax2.set_ylabel('CUSUM Score', fontsize=12, fontweight='bold', color=COLOR_CUSUM)
    ax2.set_ylim(0, max(max_cusum * 1.2, cusum_threshold * 1.5))
    ax2.tick_params(axis='y', labelcolor=COLOR_CUSUM)
    
    # =========================================================================
    # Detection Point Marker
    # =========================================================================
    
    if defense.first_alarm_idx is not None:
        detection_time = defense.elapsed_sec[defense.first_alarm_idx]
        detection_level = defense.water_level[defense.first_alarm_idx]
        detection_cusum = defense.cusum_score[defense.first_alarm_idx]
        
        # Vertical line at detection
        ax1.axvline(
            x=detection_time,
            color=COLOR_DETECTION,
            linestyle='-',
            linewidth=3,
            alpha=0.8
        )
        
        # Annotation arrow pointing to detection
        ax1.annotate(
            f'CUSUM ALARM\nDetected at t={detection_time:.1f}s\nLevel={detection_level:.1f}%',
            xy=(detection_time, detection_level),
            xytext=(detection_time + baseline.elapsed_sec[-1] * 0.1, detection_level + 15),
            fontsize=10,
            fontweight='bold',
            color=COLOR_DETECTION,
            arrowprops=dict(
                arrowstyle='->',
                color=COLOR_DETECTION,
                lw=2
            ),
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='white',
                edgecolor=COLOR_DETECTION,
                linewidth=2
            )
        )
        
        # Mark where safe hold begins
        for i, safe_hold in enumerate(defense.safe_hold_active):
            if safe_hold and (i == 0 or not defense.safe_hold_active[i-1]):
                ax1.axvspan(
                    defense.elapsed_sec[i],
                    defense.elapsed_sec[-1],
                    color=COLOR_SAFE_HOLD,
                    alpha=0.1,
                    label='Safe Hold Active'
                )
                break
    
    # =========================================================================
    # Title and Legend
    # =========================================================================
    
    # Title with summary stats
    if summary:
        baseline_max = summary.get('baseline', {}).get('max_level', max(baseline.water_level))
        defense_max = summary.get('defense', {}).get('max_level', max(defense.water_level))
        reduction = baseline_max - defense_max
        
        title_text = f"{title}\n"
        title_text += f"Baseline Max: {baseline_max:.1f}% | Defense Max: {defense_max:.1f}% | "
        title_text += f"Level Reduction: {reduction:.1f}%"
    else:
        title_text = title
    
    ax1.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
    
    # Custom legend
    legend_elements = [
        Line2D([0], [0], color=COLOR_BASELINE_LEVEL, linestyle='--', linewidth=2.5,
               label='Water Level (Baseline - No CUSUM)'),
        Line2D([0], [0], color=COLOR_DEFENSE_LEVEL, linestyle='-', linewidth=2.5,
               label='Water Level (Defense - With CUSUM)'),
        mpatches.Patch(color=COLOR_CUSUM, alpha=0.3, label='CUSUM Score'),
        Line2D([0], [0], color=COLOR_DETECTION, linestyle='-', linewidth=3,
               label='Detection Point'),
        mpatches.Patch(color=COLOR_DANGER, alpha=0.15, label='Danger Zone (>85%)'),
        mpatches.Patch(color=COLOR_SAFE_HOLD, alpha=0.2, label='Safe Hold Active'),
    ]
    
    ax1.legend(
        handles=legend_elements,
        loc='upper left',
        fontsize=9,
        framealpha=0.9
    )
    
    # =========================================================================
    # Save and Show
    # =========================================================================
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved plot to: {output_path}")
    
    # Also save as PDF for paper
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"✅ Saved PDF to: {pdf_path}")
    
    plt.show()
    
    return fig


def plot_detailed_analysis(
    defense: ExperimentData,
    output_path: str = "detailed_analysis.png",
):
    """
    Generate detailed analysis plot for the defense run.
    
    Shows:
    - Water level with LLM proposed vs Shield applied setpoints
    - CUSUM score with threshold
    - Shield action types over time
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Colors
    COLOR_LEVEL = '#27ae60'
    COLOR_LLM = '#e74c3c'
    COLOR_SHIELD = '#3498db'
    COLOR_CUSUM = '#9b59b6'
    
    # =========================================================================
    # Panel 1: Water Level and Setpoints
    # =========================================================================
    ax1 = axes[0]
    
    ax1.plot(defense.elapsed_sec, defense.water_level, 
             color=COLOR_LEVEL, linewidth=2, label='Actual Water Level')
    ax1.plot(defense.elapsed_sec, defense.llm_proposed,
             color=COLOR_LLM, linewidth=1.5, linestyle=':', alpha=0.7,
             label='LLM Proposed Setpoint')
    ax1.plot(defense.elapsed_sec, defense.shield_applied,
             color=COLOR_SHIELD, linewidth=1.5, linestyle='--', alpha=0.7,
             label='Shield Applied Setpoint')
    
    # Mark detection point
    if defense.first_alarm_idx:
        t = defense.elapsed_sec[defense.first_alarm_idx]
        ax1.axvline(x=t, color='orange', linewidth=2, label='Detection Point')
    
    ax1.axhline(y=90, color='red', linestyle=':', alpha=0.5)
    ax1.set_ylabel('Level / Setpoint (%)', fontsize=11)
    ax1.set_ylim(40, 100)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Water Level vs LLM Attack Attempts', fontsize=12, fontweight='bold')
    
    # =========================================================================
    # Panel 2: CUSUM Score
    # =========================================================================
    ax2 = axes[1]
    
    ax2.fill_between(defense.elapsed_sec, 0, defense.cusum_score,
                     color=COLOR_CUSUM, alpha=0.3)
    ax2.plot(defense.elapsed_sec, defense.cusum_score,
             color=COLOR_CUSUM, linewidth=2, label='CUSUM Score')
    ax2.axhline(y=4.0, color=COLOR_CUSUM, linestyle='--', 
                linewidth=2, label='Threshold (h=4)')
    
    if defense.first_alarm_idx:
        t = defense.elapsed_sec[defense.first_alarm_idx]
        ax2.axvline(x=t, color='orange', linewidth=2)
        ax2.scatter([t], [defense.cusum_score[defense.first_alarm_idx]],
                   color='orange', s=100, zorder=5, marker='*')
    
    ax2.set_ylabel('CUSUM Score', fontsize=11)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('CUSUM Anomaly Detection Score', fontsize=12, fontweight='bold')
    
    # =========================================================================
    # Panel 3: Shield Actions
    # =========================================================================
    ax3 = axes[2]
    
    # Convert shield actions to numeric for plotting
    action_map = {'ALLOWED': 1, 'CLIPPED': 2, 'BLOCKED': 3}
    action_colors = {'ALLOWED': 'green', 'CLIPPED': 'orange', 'BLOCKED': 'red'}
    
    for action_type, level in action_map.items():
        indices = [i for i, a in enumerate(defense.shield_action) if a == action_type]
        times = [defense.elapsed_sec[i] for i in indices]
        levels = [level] * len(times)
        ax3.scatter(times, levels, color=action_colors[action_type], 
                   s=30, alpha=0.7, label=action_type)
    
    ax3.set_yticks([1, 2, 3])
    ax3.set_yticklabels(['ALLOWED', 'CLIPPED', 'BLOCKED'])
    ax3.set_ylabel('Shield Action', fontsize=11)
    ax3.set_xlabel('Time (Seconds)', fontsize=11)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.set_title('Safety Shield Actions Over Time', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved detailed analysis to: {output_path}")
    
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize LLM Attack Defense Experiment Results"
    )
    
    parser.add_argument(
        "--results-dir", "-d",
        type=str,
        required=True,
        help="Directory containing experiment CSV files and summary JSON"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for the plot (default: results_dir/money_plot.png)"
    )
    
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Also generate detailed analysis plot"
    )
    
    args = parser.parse_args()
    
    # Find CSV files
    baseline_csv = os.path.join(args.results_dir, "baseline_shield_only_log.csv")
    defense_csv = os.path.join(args.results_dir, "defense_shield_cusum_log.csv")
    summary_json = os.path.join(args.results_dir, "ab_test_summary.json")
    
    if not os.path.exists(baseline_csv):
        print(f"❌ Baseline CSV not found: {baseline_csv}")
        sys.exit(1)
    
    if not os.path.exists(defense_csv):
        print(f"❌ Defense CSV not found: {defense_csv}")
        sys.exit(1)
    
    # Load data
    print(f"📂 Loading baseline data from: {baseline_csv}")
    baseline = load_csv(baseline_csv)
    
    print(f"📂 Loading defense data from: {defense_csv}")
    defense = load_csv(defense_csv)
    
    summary = None
    if os.path.exists(summary_json):
        print(f"📂 Loading summary from: {summary_json}")
        summary = load_summary(summary_json)
    
    # Generate plots
    output_path = args.output or os.path.join(args.results_dir, "money_plot.png")
    
    print("\n📊 Generating money plot...")
    plot_money_graph(baseline, defense, summary, output_path)
    
    if args.detailed:
        detailed_path = output_path.replace('.png', '_detailed.png')
        print("\n📊 Generating detailed analysis...")
        plot_detailed_analysis(defense, detailed_path)
    
    print("\n✨ Visualization complete!")


if __name__ == "__main__":
    main()
