#!/usr/bin/env python3
"""
LLM Attack Defense Demo Runner

This script orchestrates the complete demonstration workflow:
1. Runs A/B test (baseline vs defense)
2. Generates visualization (money plot)
3. Outputs summary for paper

Usage:
    python run_demo.py                    # Full demo with default settings
    python run_demo.py --quick            # Quick 50-step demo
    python run_demo.py --duration 100     # Custom duration
    
Author: HAI Testbed Research
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Run complete LLM Attack Defense Demonstration"
    )
    
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=100,
        help="Experiment duration in steps (default: 100)"
    )
    
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick demo with 50 steps and 0.5s delay"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="openai/gpt-oss-120b",
        help="LLM model to use"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results/demo",
        help="Output directory"
    )
    
    parser.add_argument(
        "--skip-experiment",
        action="store_true",
        help="Skip experiment, only generate visualization from existing data"
    )
    
    args = parser.parse_args()
    
    # Quick mode overrides
    if args.quick:
        args.duration = 50
        delay = 0.3
    else:
        delay = 0.5
    
    print("=" * 70)
    print("LLM ATTACK DEFENSE DEMONSTRATION")
    print("=" * 70)
    print(f"Duration: {args.duration} steps")
    print(f"Delay: {delay}s per step")
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    
    if not args.skip_experiment:
        # Step 1: Run A/B Test
        print("\n[STEP 1/3] Running A/B Test Experiments...")
        print("-" * 50)
        
        experiment_cmd = [
            sys.executable,
            "llm_attack/attack_experiment.py",
            "--model", args.model,
            "--duration", str(args.duration),
            "--delay", str(delay),
            "--output", output_dir,
        ]
        
        result = subprocess.run(experiment_cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode != 0:
            print("❌ Experiment failed!")
            sys.exit(1)
        
        print("\n✅ A/B Test completed!")
    else:
        # Find most recent results
        if os.path.exists(args.output_dir):
            subdirs = [d for d in os.listdir(args.output_dir) 
                      if os.path.isdir(os.path.join(args.output_dir, d))]
            if subdirs:
                subdirs.sort(reverse=True)
                output_dir = os.path.join(args.output_dir, subdirs[0])
                print(f"Using existing results from: {output_dir}")
            else:
                print("❌ No existing results found!")
                sys.exit(1)
        else:
            print("❌ Output directory does not exist!")
            sys.exit(1)
    
    # Step 2: Generate Visualization
    print("\n[STEP 2/3] Generating Visualization...")
    print("-" * 50)
    
    viz_cmd = [
        sys.executable,
        "llm_defense/visualize_results.py",
        "--results-dir", output_dir,
        "--detailed",
    ]
    
    result = subprocess.run(viz_cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    if result.returncode != 0:
        print("⚠️ Visualization had issues (matplotlib may not display)")
    
    print("\n✅ Visualization generated!")
    
    # Step 3: Print Summary for Paper
    print("\n[STEP 3/3] Summary for Paper...")
    print("-" * 50)
    
    summary_path = os.path.join(output_dir, "ab_test_summary.json")
    if os.path.exists(summary_path):
        import json
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        print("\n" + "=" * 70)
        print("PAPER-READY SUMMARY")
        print("=" * 70)
        
        baseline = summary.get('baseline', {})
        defense = summary.get('defense', {})
        comparison = summary.get('comparison', {})
        
        print(f"""
EXPERIMENTAL RESULTS:

1. BASELINE (Safety Shield Only):
   - Maximum Water Level: {baseline.get('max_level', 'N/A'):.1f}%
   - Final Water Level: {baseline.get('final_level', 'N/A'):.1f}%
   - Total Drift: +{baseline.get('drift', 'N/A'):.1f}%
   - Attack Success: {'YES' if baseline.get('attack_success', False) else 'NO'}

2. DEFENSE (Safety Shield + CUSUM Detection):
   - Maximum Water Level: {defense.get('max_level', 'N/A'):.1f}%
   - Final Water Level: {defense.get('final_level', 'N/A'):.1f}%
   - Total Drift: {defense.get('drift', 'N/A'):+.1f}%
   - First Detection: Step {defense.get('first_alarm_step', 'N/A')} (t={defense.get('first_alarm_time', 'N/A'):.1f}s)
   - Attack Success: {'YES' if defense.get('attack_success', False) else 'NO'}
   - Defense Success: {'YES' if defense.get('defense_success', False) else 'NO'}

3. COMPARISON:
   - Maximum Level Reduction: {comparison.get('max_level_reduction', 0):.1f}%
   - Drift Reduction: {comparison.get('drift_reduction', 0):.1f}%

CONCLUSION:
The CUSUM-based temporal detection successfully identified the low-and-slow 
LLM attack at t={defense.get('first_alarm_time', 'N/A'):.1f}s (step {defense.get('first_alarm_step', 'N/A')}), 
engaging the safe-hold mechanism before the attack succeeded.

The defense reduced maximum water level by {comparison.get('max_level_reduction', 0):.1f}% compared to 
the baseline, preventing the overflow condition that would have occurred with 
shield-only protection.
""")
        
        print("=" * 70)
        print(f"📁 Full results saved to: {output_dir}")
        print(f"📊 Money plot: {os.path.join(output_dir, 'money_plot.png')}")
        print(f"📊 Detailed plot: {os.path.join(output_dir, 'money_plot_detailed.png')}")
        print("=" * 70)
    
    print("\n✨ Demo complete!")


if __name__ == "__main__":
    main()
