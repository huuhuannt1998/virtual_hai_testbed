#!/usr/bin/env python3
"""
LLM Model Comparison Visualization

Generates comparison plots for LLM attack capabilities:
1. Water level trajectories (all models overlaid)
2. Response time comparison
3. Attack effectiveness bar chart
4. Combined dashboard

Author: HAI Testbed Research
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np


# =============================================================================
# Data Loading
# =============================================================================

@dataclass
class ModelData:
    """Loaded data for a single model."""
    model: str
    short_name: str
    steps: List[int]
    elapsed_sec: List[float]
    water_level: List[float]
    llm_proposed: List[float]
    shield_applied: List[float]
    response_time_ms: List[float]
    llm_success: List[bool]
    max_level: float = 0.0
    final_level: float = 0.0
    total_drift: float = 0.0
    avg_response_ms: float = 0.0


def load_model_csv(filepath: str) -> ModelData:
    """Load model data from CSV file."""
    steps = []
    elapsed_sec = []
    water_level = []
    llm_proposed = []
    shield_applied = []
    response_time_ms = []
    llm_success = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['Step']))
            elapsed_sec.append(float(row['Elapsed_Sec']))
            water_level.append(float(row['Water_Level']))
            llm_proposed.append(float(row['LLM_Proposed']))
            shield_applied.append(float(row['Shield_Applied']))
            response_time_ms.append(float(row['LLM_Response_Time_ms']))
            llm_success.append(row['LLM_Success'].lower() == 'true')
    
    # Extract model name from filename
    filename = os.path.basename(filepath).replace('_log.csv', '')
    short_name = filename.split('_')[-1] if '_' in filename else filename
    
    # Compute metrics
    max_level = max(water_level) if water_level else 0
    final_level = water_level[-1] if water_level else 0
    total_drift = final_level - 50.0
    avg_response = sum(response_time_ms) / len(response_time_ms) if response_time_ms else 0
    
    return ModelData(
        model=filename,
        short_name=short_name,
        steps=steps,
        elapsed_sec=elapsed_sec,
        water_level=water_level,
        llm_proposed=llm_proposed,
        shield_applied=shield_applied,
        response_time_ms=response_time_ms,
        llm_success=llm_success,
        max_level=max_level,
        final_level=final_level,
        total_drift=total_drift,
        avg_response_ms=avg_response,
    )


def load_comparison_json(filepath: str) -> Dict:
    """Load comparison results JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


# =============================================================================
# Plotting Functions
# =============================================================================

# Color palette for models
MODEL_COLORS = [
    '#e74c3c',  # Red
    '#3498db',  # Blue
    '#27ae60',  # Green
    '#9b59b6',  # Purple
    '#f39c12',  # Orange
    '#1abc9c',  # Teal
]


def plot_water_level_comparison(
    models: List[ModelData],
    output_path: str = "water_level_comparison.png",
):
    """Plot water level trajectories for all models."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot each model
    for i, model in enumerate(models):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax.plot(
            model.elapsed_sec,
            model.water_level,
            color=color,
            linewidth=2.5,
            label=f"{model.short_name} (max: {model.max_level:.1f}%)",
            alpha=0.8,
        )
    
    # Reference lines
    ax.axhline(y=90, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Attack Success (90%)')
    ax.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Initial Level (50%)')
    
    # Danger zone
    ax.axhspan(85, 100, color='red', alpha=0.1)
    
    ax.set_xlabel('Time (Seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Water Level (%)', fontsize=12, fontweight='bold')
    ax.set_title('LLM Attack Effectiveness Comparison\nWater Level Trajectory by Model', fontsize=14, fontweight='bold')
    ax.set_ylim(45, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    return fig


def plot_response_time_comparison(
    models: List[ModelData],
    output_path: str = "response_time_comparison.png",
):
    """Plot response time distribution for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Box plot
    ax1 = axes[0]
    data = [m.response_time_ms for m in models]
    labels = [m.short_name for m in models]
    colors = [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(models))]
    
    bp = ax1.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Response Time (ms)', fontsize=11)
    ax1.set_title('Response Time Distribution', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)
    
    # Right: Average bar chart
    ax2 = axes[1]
    avg_times = [m.avg_response_ms for m in models]
    bars = ax2.bar(labels, avg_times, color=colors, alpha=0.8)
    
    # Add value labels
    for bar, val in zip(bars, avg_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:.0f}ms', ha='center', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Average Response Time (ms)', fontsize=11)
    ax2.set_title('Average Response Time by Model', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    return fig


def plot_effectiveness_comparison(
    models: List[ModelData],
    output_path: str = "effectiveness_comparison.png",
):
    """Plot attack effectiveness metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    labels = [m.short_name for m in models]
    colors = [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(models))]
    
    # 1. Max Water Level
    ax1 = axes[0]
    max_levels = [m.max_level for m in models]
    bars1 = ax1.bar(labels, max_levels, color=colors, alpha=0.8)
    ax1.axhline(y=90, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.set_ylabel('Max Water Level (%)', fontsize=11)
    ax1.set_title('Peak Attack Level', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, max_levels):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Total Drift
    ax2 = axes[1]
    drifts = [m.total_drift for m in models]
    bars2 = ax2.bar(labels, drifts, color=colors, alpha=0.8)
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax2.set_ylabel('Total Drift (%)', fontsize=11)
    ax2.set_title('Cumulative Level Increase', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, drifts):
        y_pos = bar.get_height() + 1 if val >= 0 else bar.get_height() - 3
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{val:+.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. LLM Success Rate
    ax3 = axes[2]
    success_rates = [sum(m.llm_success)/len(m.llm_success)*100 for m in models]
    bars3 = ax3.bar(labels, success_rates, color=colors, alpha=0.8)
    ax3.set_ylabel('LLM Success Rate (%)', fontsize=11)
    ax3.set_title('API Response Success', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 110)
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars3, success_rates):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', fontsize=10, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    return fig


def plot_dashboard(
    models: List[ModelData],
    output_path: str = "llm_comparison_dashboard.png",
):
    """Generate comprehensive dashboard."""
    fig = plt.figure(figsize=(18, 12))
    
    # Layout: 2 rows
    # Top row: Water level comparison (full width)
    # Bottom row: 3 metrics charts
    
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=3)  # Top row
    ax2 = plt.subplot2grid((2, 3), (1, 0))  # Bottom left
    ax3 = plt.subplot2grid((2, 3), (1, 1))  # Bottom middle
    ax4 = plt.subplot2grid((2, 3), (1, 2))  # Bottom right
    
    labels = [m.short_name for m in models]
    colors = [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(models))]
    
    # ==========================================================================
    # Top: Water Level Trajectories
    # ==========================================================================
    for i, model in enumerate(models):
        ax1.plot(
            model.elapsed_sec,
            model.water_level,
            color=colors[i],
            linewidth=2.5,
            label=f"{model.short_name}",
            alpha=0.8,
        )
    
    ax1.axhline(y=90, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax1.axhspan(85, 100, color='red', alpha=0.1)
    ax1.set_xlabel('Time (Seconds)', fontsize=11)
    ax1.set_ylabel('Water Level (%)', fontsize=11)
    ax1.set_title('Water Level Trajectory Comparison', fontsize=13, fontweight='bold')
    ax1.set_ylim(45, 100)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    
    # ==========================================================================
    # Bottom Left: Max Water Level
    # ==========================================================================
    max_levels = [m.max_level for m in models]
    bars2 = ax2.bar(labels, max_levels, color=colors, alpha=0.8)
    ax2.axhline(y=90, color='red', linestyle='--', linewidth=2)
    ax2.set_ylabel('Max Level (%)', fontsize=10)
    ax2.set_title('Peak Attack Level', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    # ==========================================================================
    # Bottom Middle: Response Time
    # ==========================================================================
    avg_times = [m.avg_response_ms for m in models]
    bars3 = ax3.bar(labels, avg_times, color=colors, alpha=0.8)
    ax3.set_ylabel('Avg Response (ms)', fontsize=10)
    ax3.set_title('LLM Response Speed', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=45)
    
    # ==========================================================================
    # Bottom Right: Success Rate
    # ==========================================================================
    success_rates = [sum(m.llm_success)/len(m.llm_success)*100 for m in models]
    bars4 = ax4.bar(labels, success_rates, color=colors, alpha=0.8)
    ax4.set_ylabel('Success Rate (%)', fontsize=10)
    ax4.set_title('API Reliability', fontsize=11, fontweight='bold')
    ax4.set_ylim(0, 110)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.tick_params(axis='x', rotation=45)
    
    # Main title
    fig.suptitle('LLM Attack Capability Comparison Dashboard', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    # Also save as PDF
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    print(f"✅ Saved dashboard: {output_path}")
    print(f"✅ Saved PDF: {pdf_path}")
    
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize LLM model comparison results"
    )
    
    parser.add_argument(
        "--results-dir", "-d",
        type=str,
        required=True,
        help="Directory containing model CSV logs"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for plots (default: same as results-dir)"
    )
    
    args = parser.parse_args()
    
    output_dir = args.output or args.results_dir
    
    # Find all CSV files
    csv_files = [f for f in os.listdir(args.results_dir) if f.endswith('_log.csv')]
    
    if not csv_files:
        print(f"❌ No CSV log files found in: {args.results_dir}")
        sys.exit(1)
    
    print(f"📂 Found {len(csv_files)} model logs")
    
    # Load data
    models: List[ModelData] = []
    for csv_file in sorted(csv_files):
        filepath = os.path.join(args.results_dir, csv_file)
        print(f"  Loading: {csv_file}")
        model_data = load_model_csv(filepath)
        models.append(model_data)
    
    # Generate plots
    print("\n📊 Generating visualizations...")
    
    plot_water_level_comparison(
        models,
        os.path.join(output_dir, "water_level_comparison.png")
    )
    
    plot_response_time_comparison(
        models,
        os.path.join(output_dir, "response_time_comparison.png")
    )
    
    plot_effectiveness_comparison(
        models,
        os.path.join(output_dir, "effectiveness_comparison.png")
    )
    
    plot_dashboard(
        models,
        os.path.join(output_dir, "llm_comparison_dashboard.png")
    )
    
    print("\n✨ Visualization complete!")
    print(f"📁 Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
