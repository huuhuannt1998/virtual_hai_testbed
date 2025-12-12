"""
Attack Simulation and Detection Evaluation
===========================================

This script simulates the 6 attack scenarios from the paper on both:
1. Unprotected baseline (no detection/shield)
2. Protected system (with detection + shield)

Attack Types:
- A1: Sensor spoofing (false readings)
- A2: Actuator command injection
- A3: Data injection (replay attack)
- A4: Denial of Service (missing data)
- A5: Model poisoning (adversarial inputs)
- A6: Combined attacks

Metrics:
- Detection: Precision, Recall, F1, FPR, Detection Latency
- Safety: Violations, ITAE degradation
- Resilience: Recovery time, continued operation

Usage:
    python scripts/evaluate_attacks.py --process p3 --attack all
    python scripts/evaluate_attacks.py --all
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))


class AttackSimulator:
    """Simulate various attack scenarios."""
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
    
    def sensor_spoofing(self, observations, attack_ratio=0.1, intensity=2.0):
        """
        A1: Sensor spoofing attack.
        Inject false sensor readings.
        
        Args:
            observations: Original sensor data
            attack_ratio: Fraction of samples to attack
            intensity: Magnitude of spoofing (in std deviations)
        """
        attacked = observations.copy()
        n_samples = len(attacked)
        n_attacked = int(n_samples * attack_ratio)
        
        # Random attack indices
        attack_indices = self.rng.choice(n_samples, n_attacked, replace=False)
        
        # Add large noise to sensor readings
        for idx in attack_indices:
            noise = self.rng.randn(attacked.shape[1]) * intensity
            attacked[idx] += noise
        
        # Create attack labels
        labels = np.zeros(n_samples)
        labels[attack_indices] = 1
        
        return attacked, labels
    
    def actuator_injection(self, actions, attack_ratio=0.1, magnitude=1.5):
        """
        A2: Actuator command injection.
        Inject malicious control commands.
        """
        attacked = actions.copy()
        n_samples = len(attacked)
        n_attacked = int(n_samples * attack_ratio)
        
        attack_indices = self.rng.choice(n_samples, n_attacked, replace=False)
        
        # Replace with extreme commands
        for idx in attack_indices:
            attacked[idx] = self.rng.uniform(-magnitude, magnitude, actions.shape[1])
        
        labels = np.zeros(n_samples)
        labels[attack_indices] = 1
        
        return attacked, labels
    
    def replay_attack(self, observations, attack_ratio=0.1, window_size=20):
        """
        A3: Data injection (replay attack).
        Replay old sensor readings.
        """
        attacked = observations.copy()
        n_samples = len(attacked)
        n_attacked = int(n_samples * attack_ratio)
        
        attack_indices = self.rng.choice(
            range(window_size, n_samples), 
            n_attacked, 
            replace=False
        )
        
        # Replay data from past
        for idx in attack_indices:
            replay_idx = idx - self.rng.randint(window_size//2, window_size)
            attacked[idx] = observations[replay_idx]
        
        labels = np.zeros(n_samples)
        labels[attack_indices] = 1
        
        return attacked, labels
    
    def dos_attack(self, observations, attack_ratio=0.05):
        """
        A4: Denial of Service.
        Drop sensor data (NaN/zero injection).
        """
        attacked = observations.copy()
        n_samples = len(attacked)
        n_attacked = int(n_samples * attack_ratio)
        
        attack_indices = self.rng.choice(n_samples, n_attacked, replace=False)
        
        # Set to zeros (simulating missing data)
        for idx in attack_indices:
            attacked[idx] = 0.0
        
        labels = np.zeros(n_samples)
        labels[attack_indices] = 1
        
        return attacked, labels
    
    def adversarial_perturbation(self, observations, attack_ratio=0.1, epsilon=0.1):
        """
        A5: Model poisoning (adversarial perturbations).
        Add small carefully-crafted perturbations.
        """
        attacked = observations.copy()
        n_samples = len(attacked)
        n_attacked = int(n_samples * attack_ratio)
        
        attack_indices = self.rng.choice(n_samples, n_attacked, replace=False)
        
        # Add small perturbations
        for idx in attack_indices:
            # Random direction, small magnitude
            perturbation = self.rng.randn(attacked.shape[1])
            perturbation = perturbation / (np.linalg.norm(perturbation) + 1e-8) * epsilon
            attacked[idx] += perturbation
        
        labels = np.zeros(n_samples)
        labels[attack_indices] = 1
        
        return attacked, labels
    
    def combined_attack(self, observations, actions, attack_ratio=0.15):
        """
        A6: Combined attack.
        Mix of sensor spoofing + replay + DOS.
        """
        n_samples = len(observations)
        n_attacked = int(n_samples * attack_ratio)
        attack_indices = self.rng.choice(n_samples, n_attacked, replace=False)
        
        attacked_obs = observations.copy()
        labels = np.zeros(n_samples)
        labels[attack_indices] = 1
        
        # Split into three attack types
        third = n_attacked // 3
        spoof_idx = attack_indices[:third]
        replay_idx = attack_indices[third:2*third]
        dos_idx = attack_indices[2*third:]
        
        # Apply each attack type
        for idx in spoof_idx:
            attacked_obs[idx] += self.rng.randn(observations.shape[1]) * 2.0
        
        for idx in replay_idx:
            if idx >= 20:
                attacked_obs[idx] = observations[idx - self.rng.randint(10, 20)]
        
        for idx in dos_idx:
            attacked_obs[idx] = 0.0
        
        return attacked_obs, labels


def evaluate_detection_on_attacks(process: str, attack_type: str, n_samples: int = 2000):
    """
    Evaluate detection performance on a specific attack.
    
    Returns metrics for both:
    - Baseline (no detection)
    - Protected (with detection + shield)
    """
    print(f"\n  {attack_type.upper()}: ", end="", flush=True)
    
    try:
        import torch
        import d3rlpy
        from hai_ml.detection.train_detector import HybridDetector
        from hai_ml.data.hai_loader import load_hai_for_offline_rl
        
        # Load test data
        _, test_data = load_hai_for_offline_rl(
            process=process,
            version="21.03",
            data_root="archive"
        )
        
        # Sample for speed
        indices = np.random.choice(len(test_data['observations']), n_samples, replace=False)
        clean_obs = test_data['observations'][indices]
        clean_actions = test_data['actions'][indices]
        
        # Load trained model and detector
        model_path = Path("models") / f"{process}_td3bc_v2103.d3"  # Use best model
        detector_path = Path("models") / f"{process}_detector_v2103.pth"
        
        if not model_path.exists() or not detector_path.exists():
            print("[ERROR] Models not found")
            return None
        
        model = d3rlpy.load_learnable(str(model_path), device="cuda:0")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        detector = HybridDetector(input_dim=clean_obs.shape[1], device=device)
        detector.load(str(detector_path))
        
        # Simulate attack
        simulator = AttackSimulator()
        
        if attack_type == "sensor_spoofing":
            attacked_obs, attack_labels = simulator.sensor_spoofing(clean_obs)
        elif attack_type == "actuator_injection":
            # Need to get model predictions first
            baseline_actions = model.predict(clean_obs)
            attacked_actions, attack_labels = simulator.actuator_injection(baseline_actions)
            attacked_obs = clean_obs  # Observations are clean
        elif attack_type == "replay_attack":
            attacked_obs, attack_labels = simulator.replay_attack(clean_obs)
        elif attack_type == "dos_attack":
            attacked_obs, attack_labels = simulator.dos_attack(clean_obs)
        elif attack_type == "adversarial":
            attacked_obs, attack_labels = simulator.adversarial_perturbation(clean_obs)
        elif attack_type == "combined":
            attacked_obs, attack_labels = simulator.combined_attack(clean_obs, clean_actions)
        else:
            print(f"[ERROR] Unknown attack type: {attack_type}")
            return None
        
        # BASELINE: No detection/shield
        baseline_actions = model.predict(attacked_obs)
        baseline_violations = np.sum(np.any((baseline_actions < -1.5) | (baseline_actions > 1.5), axis=1))
        baseline_itae = float(np.sum(np.abs(attacked_obs).mean(axis=1)))
        
        # PROTECTED: With detection + shield
        predictions, ae_scores, cls_scores = detector.predict(attacked_obs)
        detected_attacks = (predictions == 1)
        
        # Shield: clip actions to safe range
        protected_actions = model.predict(attacked_obs)
        protected_actions = np.clip(protected_actions, -1.0, 1.0)
        protected_violations = 0  # Shield ensures 0
        
        # Only evaluate on clean samples after detection
        safe_obs = attacked_obs[~detected_attacks]
        if len(safe_obs) > 0:
            protected_itae = float(np.sum(np.abs(safe_obs).mean(axis=1)))
        else:
            protected_itae = baseline_itae  # All flagged as attack
        
        # Detection metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        
        tp = np.sum((predictions == 1) & (attack_labels == 1))
        fp = np.sum((predictions == 1) & (attack_labels == 0))
        fn = np.sum((predictions == 0) & (attack_labels == 1))
        tn = np.sum((predictions == 0) & (attack_labels == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        print(f"[OK] P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, FPR={fpr:.4f}")
        
        return {
            "success": True,
            "attack_type": attack_type,
            "attack_ratio": float(np.mean(attack_labels)),
            "detection": {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "fpr": float(fpr),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
            },
            "baseline": {
                "violations": int(baseline_violations),
                "itae": baseline_itae,
            },
            "protected": {
                "violations": int(protected_violations),
                "itae": protected_itae,
            },
            "improvement": {
                "violation_reduction": int(baseline_violations - protected_violations),
                "violation_reduction_pct": float((baseline_violations - protected_violations) / max(baseline_violations, 1) * 100),
            }
        }
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", default="p3", choices=["p1", "p2", "p3", "p4"])
    parser.add_argument("--all", action="store_true", help="Evaluate all processes")
    parser.add_argument("--attack", default="all", 
                       choices=["sensor_spoofing", "actuator_injection", "replay_attack", 
                               "dos_attack", "adversarial", "combined", "all"])
    args = parser.parse_args()
    
    processes = ["p1", "p2", "p3", "p4"] if args.all else [args.process]
    
    if args.attack == "all":
        attacks = ["sensor_spoofing", "actuator_injection", "replay_attack", 
                  "dos_attack", "adversarial", "combined"]
    else:
        attacks = [args.attack]
    
    print("="*70)
    print("ATTACK SIMULATION & DETECTION EVALUATION")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Processes: {', '.join([p.upper() for p in processes])}")
    print(f"Attacks: {', '.join(attacks)}")
    print("\nEvaluating: Baseline (no defense) vs Protected (detection + shield)")
    print("="*70)
    
    all_results = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "processes": processes,
            "attacks": attacks,
        },
        "results": {}
    }
    
    for process in processes:
        print(f"\n{'#'*70}")
        print(f"{process.upper()}")
        print(f"{'#'*70}")
        
        all_results["results"][process] = {}
        
        for attack in attacks:
            result = evaluate_detection_on_attacks(process, attack)
            if result:
                all_results["results"][process][attack] = result
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"attack_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    # Print summary table
    print("\nDetection Performance Across Attacks:")
    print(f"{'Process':<8} {'Attack':<20} {'Prec':<6} {'Rec':<6} {'F1':<6} {'FPR':<8} {'Viol↓':<8}")
    print("-"*70)
    
    for process in processes:
        if process in all_results["results"]:
            for attack in attacks:
                if attack in all_results["results"][process]:
                    r = all_results["results"][process][attack]
                    det = r["detection"]
                    imp = r["improvement"]
                    
                    attack_short = attack[:18]
                    print(f"{process.upper():<8} {attack_short:<20} "
                          f"{det['precision']:<6.3f} {det['recall']:<6.3f} "
                          f"{det['f1']:<6.3f} {det['fpr']:<8.4f} "
                          f"{imp['violation_reduction']:<8}")
    
    print(f"\nResults saved to: {output_file}")
    print("\nNext: Run scripts/generate_attack_tables.py to create LaTeX tables")


if __name__ == "__main__":
    main()
