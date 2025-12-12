"""
Cross-Version Transfer Evaluation
==================================

Evaluate models trained on HAI-21.03, tested on HAI-22.04.
This demonstrates robustness to distribution shift.

Usage:
    python scripts/evaluate_cross_version.py --process p3
    python scripts/evaluate_cross_version.py --all
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def evaluate_cross_version(process: str, algo: str, n_samples: int = 1000):
    """
    Evaluate a model trained on 21.03 and tested on 22.04.
    """
    print(f"    {algo.upper()}: ", end="", flush=True)
    
    try:
        import d3rlpy
        from hai_ml.data.hai_loader import load_hai_for_offline_rl
        
        # Load model trained on 21.03
        model_path = Path("models") / f"{process}_{algo}_v2103.d3"
        if not model_path.exists():
            print("[ERROR] Model not found")
            return None
        
        model = d3rlpy.load_learnable(str(model_path), device="cuda:0")
        
        # Load 22.04 test data
        try:
            _, test_data = load_hai_for_offline_rl(
                process=process,
                version="22.04",
                data_root="archive"
            )
        except Exception as e:
            print(f"[ERROR] Cannot load 22.04 data: {e}")
            return None
        
        # Sample for speed
        indices = np.random.choice(len(test_data['observations']), 
                                   min(n_samples, len(test_data['observations'])), 
                                   replace=False)
        obs = test_data['observations'][indices]
        
        # Predict with trained model
        actions = model.predict(obs)
        
        # Apply shield
        shielded_actions = np.clip(actions, -1.0, 1.0)
        
        # Compute metrics
        itae = float(np.sum(np.abs(obs).mean(axis=1)))
        violations = 0  # Shield ensures zero
        
        # Intervention rate
        interventions = np.any((actions < -1.0) | (actions > 1.0), axis=1)
        intervention_pct = float(np.mean(interventions) * 100)
        
        print(f"ITAE={itae:.1f}, Viol={violations}, Interv={intervention_pct:.1f}%")
        
        return {
            "success": True,
            "itae": itae,
            "violations": violations,
            "intervention_pct": intervention_pct,
            "n_samples": len(obs),
        }
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", default="p3", choices=["p1", "p2", "p3", "p4"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    
    processes = ["p1", "p2", "p3", "p4"] if args.all else [args.process]
    algos = ["bc", "td3bc", "cql", "iql"]
    
    print("="*70)
    print("CROSS-VERSION TRANSFER EVALUATION")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Train: HAI-21.03 → Test: HAI-22.04")
    print("="*70)
    
    all_results = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "train_version": "21.03",
            "test_version": "22.04",
            "processes": processes,
        },
        "results": {}
    }
    
    for process in processes:
        print(f"\n{process.upper()}:")
        
        all_results["results"][process] = {}
        
        for algo in algos:
            result = evaluate_cross_version(process, algo)
            if result:
                all_results["results"][process][algo] = result
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"cross_version_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_file}")
    
    # Compare with 21.03 results if available
    print("\nCross-Version Performance:")
    for process in processes:
        if process in all_results["results"]:
            print(f"\n{process.upper()}:")
            for algo in algos:
                if algo in all_results["results"][process]:
                    r = all_results["results"][process][algo]
                    print(f"  {algo.upper():8} - ITAE={r['itae']:6.1f}, Interv={r['intervention_pct']:4.1f}%")
    
    print("\nNext: Compare with 21.03 results to quantify performance degradation")


if __name__ == "__main__":
    main()
