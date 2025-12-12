"""
Generate Attack Evaluation Tables for Paper
============================================

Creates LaTeX tables showing:
- Table 7: Attack scenario comparison (before/after defense)
- Detection performance per attack type
- Safety improvement metrics

Usage:
    python scripts/generate_attack_tables.py
"""

import json
from pathlib import Path


def load_latest_attack_results():
    """Load the latest attack evaluation results."""
    results_dir = Path("results")
    attack_files = list(results_dir.glob("attack_eval_*.json"))
    
    if not attack_files:
        print("No attack evaluation results found!")
        print("Run: python scripts/evaluate_attacks.py --all")
        return None
    
    # Get latest
    latest = sorted(attack_files)[-1]
    print(f"Loading: {latest.name}\n")
    
    with open(latest) as f:
        return json.load(f)


def generate_table7_attack_scenarios(results):
    """Generate Table 7: Attack Scenarios."""
    print("="*70)
    print("TABLE 7: Attack Scenario Evaluation")
    print("="*70)
    print()
    print("\\begin{table*}[t]")
    print("\\centering")
    print("\\caption{Attack scenario evaluation comparing baseline (no defense) vs. protected system (detection + shield). Violation Reduction shows safety improvement.}")
    print("\\label{tab:attack_scenarios}")
    print("\\begin{tabular}{llcccccc}")
    print("\\toprule")
    print("Process & Attack Type & \\multicolumn{3}{c}{Detection} & \\multicolumn{3}{c}{Safety} \\\\")
    print("\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}")
    print("& & Prec$\\uparrow$ & Rec$\\uparrow$ & F1$\\uparrow$ & Baseline Viol. & Protected Viol. & Reduction \\\\")
    print("\\midrule")
    
    processes = ["p1", "p2", "p3", "p4"]
    attacks = {
        "sensor_spoofing": "Sensor Spoofing",
        "actuator_injection": "Actuator Injection",
        "replay_attack": "Replay Attack",
        "dos_attack": "DoS Attack",
        "adversarial": "Adversarial",
        "combined": "Combined Attack"
    }
    
    all_results = results.get("results", {})
    
    for process in processes:
        if process not in all_results:
            continue
        
        process_data = all_results[process]
        
        print(f"\\multirow{{{len(attacks)}}}{{*}}{{{process.upper()}}}")
        
        first = True
        for attack_key, attack_name in attacks.items():
            if attack_key in process_data:
                r = process_data[attack_key]
                det = r["detection"]
                base = r["baseline"]
                prot = r["protected"]
                imp = r["improvement"]
                
                if first:
                    prefix = ""
                    first = False
                else:
                    prefix = "  "
                
                # Highlight good detection (F1 > 0.8)
                f1_str = f"\\textbf{{{det['f1']:.3f}}}" if det['f1'] > 0.8 else f"{det['f1']:.3f}"
                
                # Highlight zero violations in protected
                prot_viol_str = f"\\textbf{{{prot['violations']}}}" if prot['violations'] == 0 else f"{prot['violations']}"
                
                print(f"{prefix}& {attack_name:<18} & {det['precision']:.3f} & {det['recall']:.3f} & "
                      f"{f1_str} & {base['violations']} & {prot_viol_str} & "
                      f"{imp['violation_reduction_pct']:.1f}\\% \\\\")
        
        if process != "p4":
            print("\\midrule")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table*}")


def generate_detection_summary_table(results):
    """Generate compact detection performance table."""
    print("\n" + "="*70)
    print("TABLE 4: Detection Performance Summary (Compact)")
    print("="*70)
    print()
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Attack detection performance across all attack types.}")
    print("\\label{tab:detection_summary}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Process & Precision$\\uparrow$ & Recall$\\uparrow$ & F1$\\uparrow$ & FPR$\\downarrow$ \\\\")
    print("\\midrule")
    
    all_results = results.get("results", {})
    
    for process in ["p1", "p2", "p3", "p4"]:
        if process not in all_results:
            print(f"{process.upper()} & [--] & [--] & [--] & [--] \\\\")
            continue
        
        process_data = all_results[process]
        
        # Average across all attacks
        precisions = []
        recalls = []
        f1s = []
        fprs = []
        
        for attack_key in process_data:
            if "detection" in process_data[attack_key]:
                det = process_data[attack_key]["detection"]
                precisions.append(det["precision"])
                recalls.append(det["recall"])
                f1s.append(det["f1"])
                fprs.append(det["fpr"])
        
        if precisions:
            avg_prec = sum(precisions) / len(precisions)
            avg_rec = sum(recalls) / len(recalls)
            avg_f1 = sum(f1s) / len(f1s)
            avg_fpr = sum(fprs) / len(fprs)
            
            # Highlight good performance
            f1_str = f"\\textbf{{{avg_f1:.3f}}}" if avg_f1 > 0.8 else f"{avg_f1:.3f}"
            
            print(f"{process.upper()} & {avg_prec:.3f} & {avg_rec:.3f} & {f1_str} & {avg_fpr:.4f} \\\\")
        else:
            print(f"{process.upper()} & [--] & [--] & [--] & [--] \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def generate_safety_comparison(results):
    """Generate safety improvement summary for paper text."""
    print("\n" + "="*70)
    print("SAFETY IMPROVEMENT STATISTICS (for paper text)")
    print("="*70)
    
    all_results = results.get("results", {})
    
    total_baseline_viol = 0
    total_protected_viol = 0
    all_f1_scores = []
    
    for process in all_results:
        for attack in all_results[process]:
            r = all_results[process][attack]
            total_baseline_viol += r["baseline"]["violations"]
            total_protected_viol += r["protected"]["violations"]
            all_f1_scores.append(r["detection"]["f1"])
    
    print(f"\n- Total baseline violations: {total_baseline_viol}")
    print(f"- Total protected violations: {total_protected_viol}")
    print(f"- Violation reduction: {total_baseline_viol - total_protected_viol} ({(total_baseline_viol - total_protected_viol)/max(total_baseline_viol, 1)*100:.1f}%)")
    
    if all_f1_scores:
        avg_f1 = sum(all_f1_scores) / len(all_f1_scores)
        min_f1 = min(all_f1_scores)
        max_f1 = max(all_f1_scores)
        print(f"- Average detection F1: {avg_f1:.3f}")
        print(f"- Detection F1 range: {min_f1:.3f} - {max_f1:.3f}")
    
    print("\n\\textbf{For Abstract/Results Section:}")
    print(f"Our detection system achieves an average F1-score of {avg_f1:.2f} across")
    print(f"six attack types, reducing safety violations by {(total_baseline_viol - total_protected_viol)/max(total_baseline_viol, 1)*100:.0f}%")
    print(f"(from {total_baseline_viol} to {total_protected_viol} violations).")


def main():
    print("="*70)
    print("ATTACK EVALUATION TABLE GENERATOR")
    print("="*70)
    print()
    
    results = load_latest_attack_results()
    
    if not results:
        return
    
    generate_detection_summary_table(results)
    generate_table7_attack_scenarios(results)
    generate_safety_comparison(results)
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("""
1. Copy tables to paper_updated.tex
2. Update abstract with detection F1 and violation reduction
3. Add attack scenario discussion to Results section
4. Consider adding per-attack-type analysis

For complete evaluation across all processes:
    python scripts/evaluate_attacks.py --all
""")


if __name__ == "__main__":
    main()
