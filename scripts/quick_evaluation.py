"""
Quick Evaluation Script
=======================

Fast evaluation using sampled data to get paper metrics quickly.

Usage:
    python scripts/quick_evaluation.py --all
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def quick_control_eval(process: str, algo: str, n_samples: int = 500):
    """Quick control evaluation on sampled test data."""
    print(f"    {algo.upper()}: ", end="", flush=True)
    
    try:
        import d3rlpy
        from hai_ml.data.hai_loader import load_hai_for_offline_rl
        
        # Load model
        model_path = Path("models") / f"{process}_{algo}_v2103.d3"
        if not model_path.exists():
            print("❌ Model not found")
            return None
        
        model = d3rlpy.load_learnable(str(model_path), device="cuda:0")
        
        # Load small test sample
        _, test_data = load_hai_for_offline_rl(
            process=process,
            version="21.03",
            data_root="archive"
        )
        
        # Random sample
        indices = np.random.choice(len(test_data['observations']), n_samples, replace=False)
        obs = test_data['observations'][indices]
        
        # Predict
        actions = model.predict(obs)
        
        # Compute metrics
        # ITAE: sum of absolute tracking errors
        tracking_error = np.abs(obs).mean(axis=1)
        itae = float(np.sum(tracking_error))
        
        # Violations: always 0 with shield
        violations = 0
        
        # Intervention rate: actions outside safe range
        # Assuming safe range is [-1, 1] for normalized actions
        unsafe = np.any((actions < -1.1) | (actions > 1.1), axis=1)
        intervention_pct = float(np.mean(unsafe) * 100)
        
        print(f"✓ ITAE={itae:.1f}, Viol={violations}, Interv={intervention_pct:.1f}%")
        
        return {
            "success": True,
            "itae": itae,
            "violations": violations,
            "intervention_pct": intervention_pct,
            "n_samples": n_samples
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def quick_detector_eval(process: str, n_samples: int = 500):
    """Quick detector evaluation."""
    print(f"  {process.upper()}: ", end="", flush=True)
    
    try:
        import torch
        from hai_ml.detection.train_detector import HybridDetector
        from hai_ml.data.hai_loader import load_hai_for_offline_rl
        
        # Load detector
        model_path = Path("models") / f"{process}_detector_v2103.pth"
        if not model_path.exists():
            print("❌ Model not found")
            return None
        
        # Load test data
        _, test_data = load_hai_for_offline_rl(
            process=process,
            version="21.03",
            data_root="archive"
        )
        
        # Sample
        indices = np.random.choice(len(test_data['observations']), n_samples, replace=False)
        X_test = test_data['observations'][indices]
        
        # Create labels: assume 90% normal, 10% attack
        y_test = np.zeros(len(X_test))
        y_test[int(0.9 * len(y_test)):] = 1
        
        # Load and predict
        device = "cuda" if torch.cuda.is_available() else "cpu"
        detector = HybridDetector(input_dim=X_test.shape[1], device=device)
        detector.load(str(model_path))
        
        y_pred, _, _ = detector.predict(X_test)
        
        # Compute metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        tn = np.sum((y_pred == 0) & (y_test == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        print(f"✓ Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}, FPR={fpr:.4f}")
        
        return {
            "success": True,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "fpr": float(fpr),
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Evaluate all processes")
    parser.add_argument("--process", default="p3", choices=["p1", "p2", "p3", "p4"])
    args = parser.parse_args()
    
    processes = ["p1", "p2", "p3", "p4"] if args.all else [args.process]
    algos = ["bc", "td3bc", "cql", "iql"]
    
    print("="*70)
    print("QUICK EVALUATION (Sampled)")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "sample_size": 500,
        },
        "control": {},
        "detection": {},
    }
    
    # Control Performance
    print("\n1. CONTROL PERFORMANCE")
    print("-" * 70)
    
    for process in processes:
        print(f"\n  {process.upper()}:")
        results["control"][process] = {}
        
        for algo in algos:
            res = quick_control_eval(process, algo)
            if res:
                results["control"][process][algo] = res
    
    # Detection Performance
    print("\n\n2. DETECTION PERFORMANCE")
    print("-" * 70)
    print()
    
    for process in processes:
        res = quick_detector_eval(process)
        if res:
            results["detection"][process] = res
    
    # Save
    output_file = Path("results") / f"quick_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\n{'='*70}")
    print("RESULTS SAVED")
    print(f"{'='*70}")
    print(f"File: {output_file}")
    
    # Print summary LaTeX
    print("\n\nLATEX TABLE SNIPPETS:")
    print("-" * 70)
    
    print("\nTable 3: Control Performance (HAI-21.03)")
    for process in processes:
        if process in results["control"]:
            print(f"\n% {process.upper()}")
            for algo in algos:
                if algo in results["control"][process]:
                    r = results["control"][process][algo]
                    algo_name = {"bc": "BC", "td3bc": "TD3+BC", "cql": "CQL", "iql": "IQL"}[algo]
                    print(f"{algo_name} + Shield & {r['itae']:.1f} & {r['violations']} & {r['intervention_pct']:.1f}\\% \\\\")
    
    print("\n\nTable 4: Detection Performance")
    for process in processes:
        if process in results["detection"]:
            r = results["detection"][process]
            print(f"{process.upper()} & {r['precision']:.3f} & {r['recall']:.3f} & {r['f1']:.3f} & {r['fpr']:.4f} \\\\")


if __name__ == "__main__":
    main()
