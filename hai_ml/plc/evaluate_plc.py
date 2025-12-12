"""
Real PLC Evaluation Module
==========================

Runs trained RL policies against the real Siemens CPU 1212FC PLC.

SAFETY NOTICE:
- This module sends real commands to physical hardware
- Ensure all safety interlocks are active before running
- Shield is ALWAYS enabled during real-world testing
- Emergency stop must be accessible

Prerequisites:
1. PLC connected via Ethernet (192.168.0.1)
2. TIA Portal project loaded with HAI DB
3. Raspberry Pi or Windows PC on same network
4. python-snap7 installed

Usage:
    # Dry run (no writes to PLC)
    python -m hai_ml.plc.evaluate_plc --dry-run --method td3bc
    
    # Real evaluation (CAUTION!)
    python -m hai_ml.plc.evaluate_plc --method td3bc --episodes 10
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hai.io import PLCClient
from hai.config import OFFSETS, PLC_IP


# P3 Water Treatment Tags
P3_SENSOR_TAGS = [
    "P3_FIT01",   # Flow indicator transmitter
    "P3_LIT01",   # Level indicator transmitter
    "P3_PIT01",   # Pressure indicator transmitter
]

P3_ACTUATOR_TAGS = [
    "P3_LCV01D",  # Level control valve demand
    "P3_LCP01D",  # Level control pump demand
]

# Safety limits for P3 (from HAI documentation)
P3_LIMITS = {
    "P3_LIT01": {"min": 5000, "max": 25000, "unit": "raw"},  # Level
    "P3_PIT01": {"min": 0, "max": 5000, "unit": "raw"},      # Pressure
    "P3_LCV01D": {"min": -1000, "max": 1000, "unit": "raw"}, # Valve
}


class PLCEvaluator:
    """
    Evaluates RL policies on real PLC hardware.
    
    Safety features:
    - All actions pass through the safety shield
    - Rate limiting on actuator changes
    - Emergency bounds checking
    - Dry-run mode for testing
    """
    
    def __init__(
        self,
        process: str = "p3",
        dry_run: bool = True,
        sample_rate_hz: float = 1.0,
    ):
        self.process = process
        self.dry_run = dry_run
        self.sample_period = 1.0 / sample_rate_hz
        
        self.plc = PLCClient()
        self.connected = False
        
        # Normalization stats (from training data)
        self.obs_mean = None
        self.obs_std = None
        
        # Load shield
        from hai_ml.safety.shield import Shield
        self.shield = Shield(schema_path=f'hai_ml/schemas/{process}.yaml')
        
        # Action rate limiter
        self.last_action = None
        self.max_action_rate = 0.1  # Max change per step
    
    def connect(self) -> bool:
        """Connect to PLC."""
        if self.dry_run:
            print("[DRY RUN] Simulating PLC connection")
            self.connected = True
            return True
        
        print(f"Connecting to PLC at {PLC_IP}...")
        self.connected = self.plc.connect()
        
        if self.connected:
            print("  ✓ Connected to PLC")
        else:
            print("  ✗ Failed to connect")
        
        return self.connected
    
    def disconnect(self):
        """Disconnect from PLC."""
        if not self.dry_run:
            self.plc.disconnect()
        self.connected = False
    
    def read_observation(self) -> np.ndarray:
        """Read current state from PLC sensors."""
        if self.dry_run:
            # Return synthetic observation
            return np.random.randn(len(P3_SENSOR_TAGS)).astype(np.float32)
        
        obs = []
        for tag in P3_SENSOR_TAGS:
            value = self.plc.read_tag_real(tag)
            obs.append(value)
        
        obs = np.array(obs, dtype=np.float32)
        
        # Normalize if stats available
        if self.obs_mean is not None and self.obs_std is not None:
            obs = (obs - self.obs_mean) / self.obs_std
        
        return obs
    
    def write_action(self, action: np.ndarray) -> Dict:
        """Write action to PLC actuators with safety checks."""
        info = {
            "action_raw": action.tolist(),
            "action_sent": None,
            "clipped": False,
            "rate_limited": False,
        }
        
        # Clip to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Rate limiting
        if self.last_action is not None:
            delta = action - self.last_action
            if np.max(np.abs(delta)) > self.max_action_rate:
                action = self.last_action + np.clip(delta, -self.max_action_rate, self.max_action_rate)
                info["rate_limited"] = True
        
        self.last_action = action.copy()
        info["action_sent"] = action.tolist()
        
        if self.dry_run:
            print(f"  [DRY RUN] Would write: {dict(zip(P3_ACTUATOR_TAGS, action))}")
            return info
        
        # Write to PLC
        for i, tag in enumerate(P3_ACTUATOR_TAGS):
            if i < len(action):
                # Denormalize action to raw PLC values
                raw_value = float(action[i]) * 1000  # Scale to raw
                
                # Bounds check
                limits = P3_LIMITS.get(tag, {"min": -10000, "max": 10000})
                raw_value = np.clip(raw_value, limits["min"], limits["max"])
                
                self.plc.write_tag_real(tag, raw_value)
        
        return info
    
    def run_episode(
        self,
        policy,
        max_steps: int = 3600,
        log_interval: int = 100,
    ) -> Dict:
        """
        Run one evaluation episode on real PLC.
        
        Args:
            policy: Trained policy (must have .predict(obs) method)
            max_steps: Maximum steps per episode
            log_interval: Print progress every N steps
            
        Returns:
            Episode results dictionary
        """
        print(f"\n  Starting episode ({max_steps} steps)...")
        
        observations = []
        actions = []
        interventions = []
        latencies = []
        violations = []
        
        for step in range(max_steps):
            t_start = time.perf_counter()
            
            # Read state
            obs = self.read_observation()
            observations.append(obs)
            
            t_read = time.perf_counter()
            
            # Get policy action
            if policy is not None:
                action = policy.predict(obs.reshape(1, -1))[0]
            else:
                action = np.zeros(len(P3_ACTUATOR_TAGS))
            
            t_policy = time.perf_counter()
            
            # Apply safety shield
            safe_action, shield_info = self.shield.project(obs, action)
            
            t_shield = time.perf_counter()
            
            # Write to PLC
            write_info = self.write_action(safe_action)
            
            t_write = time.perf_counter()
            
            # Record
            actions.append(safe_action)
            interventions.append(shield_info['intervened'])
            violations.append(shield_info.get('violation', False))
            
            latencies.append({
                "step": step,
                "read_ms": (t_read - t_start) * 1000,
                "policy_ms": (t_policy - t_read) * 1000,
                "shield_ms": (t_shield - t_policy) * 1000,
                "write_ms": (t_write - t_shield) * 1000,
                "total_ms": (t_write - t_start) * 1000,
            })
            
            # Progress logging
            if (step + 1) % log_interval == 0:
                avg_lat = np.mean([l["total_ms"] for l in latencies[-log_interval:]])
                interv_rate = np.mean(interventions[-log_interval:]) * 100
                print(f"    Step {step+1}/{max_steps}: lat={avg_lat:.1f}ms, interv={interv_rate:.1f}%")
            
            # Maintain sample rate
            elapsed = time.perf_counter() - t_start
            if elapsed < self.sample_period:
                time.sleep(self.sample_period - elapsed)
        
        # Compute metrics
        from hai_ml.eval.metrics import itae, ise, wear
        
        obs_array = np.array(observations)
        act_array = np.array(actions)
        lat_totals = [l["total_ms"] for l in latencies]
        
        errors = obs_array[:, 0]  # First sensor as error
        
        results = {
            "steps": max_steps,
            "ITAE": float(itae(errors)),
            "ISE": float(ise(errors)),
            "wear": float(wear(act_array)),
            "violations": int(sum(violations)),
            "violation_rate": sum(violations) / max_steps,
            "interventions": int(sum(interventions)),
            "intervention_rate": sum(interventions) / max_steps,
            "latency_ms_p50": float(np.percentile(lat_totals, 50)),
            "latency_ms_p95": float(np.percentile(lat_totals, 95)),
            "latency_ms_p99": float(np.percentile(lat_totals, 99)),
            "latency_max_ms": float(max(lat_totals)),
            "success": sum(violations) == 0,
        }
        
        return results, latencies
    
    def evaluate(
        self,
        policy,
        n_episodes: int = 5,
        episode_length: int = 600,
        output_dir: str = "results/plc",
    ) -> pd.DataFrame:
        """
        Run full evaluation on real PLC.
        
        Args:
            policy: Trained policy
            n_episodes: Number of episodes
            episode_length: Steps per episode
            output_dir: Directory for results
            
        Returns:
            DataFrame with episode results
        """
        if not self.connected:
            if not self.connect():
                raise RuntimeError("Failed to connect to PLC")
        
        results = []
        all_latencies = []
        
        for ep in range(n_episodes):
            print(f"\n{'='*60}")
            print(f"Episode {ep+1}/{n_episodes}")
            print(f"{'='*60}")
            
            ep_results, ep_latencies = self.run_episode(
                policy,
                max_steps=episode_length,
            )
            
            ep_results["episode"] = ep
            ep_results["timestamp"] = datetime.now().isoformat()
            results.append(ep_results)
            all_latencies.extend(ep_latencies)
            
            print(f"\n  Episode {ep+1} results:")
            print(f"    ITAE: {ep_results['ITAE']:.2f}")
            print(f"    Violations: {ep_results['violations']}")
            print(f"    Intervention rate: {ep_results['intervention_rate']*100:.1f}%")
            print(f"    Latency p99: {ep_results['latency_ms_p99']:.1f}ms")
            
            # Cool-down between episodes
            if ep < n_episodes - 1:
                print("\n  Cooling down for 10 seconds...")
                time.sleep(10)
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(output_path / f"plc_eval_{timestamp}.csv", index=False)
        
        df_lat = pd.DataFrame(all_latencies)
        df_lat.to_csv(output_path / f"plc_latency_{timestamp}.csv", index=False)
        
        print(f"\n{'='*60}")
        print("Evaluation Complete!")
        print(f"{'='*60}")
        print(f"\nSummary over {n_episodes} episodes:")
        print(f"  ITAE: {df['ITAE'].mean():.2f} ± {df['ITAE'].std():.2f}")
        print(f"  Violations: {df['violations'].sum()}")
        print(f"  Success rate: {df['success'].mean()*100:.1f}%")
        print(f"  Latency p99: {df['latency_ms_p99'].mean():.1f}ms")
        print(f"\nResults saved to: {output_path}/")
        
        return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL policy on real PLC")
    parser.add_argument("--method", default="pid", choices=["pid", "bc", "td3bc", "cql", "iql"])
    parser.add_argument("--process", default="p3")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-length", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true", help="Don't write to PLC")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--output-dir", default="results/plc")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PLC Evaluation")
    print("=" * 60)
    print(f"Method: {args.method.upper()}")
    print(f"Process: {args.process.upper()}")
    print(f"Episodes: {args.episodes}")
    print(f"Dry run: {args.dry_run}")
    
    # Load policy
    policy = None
    if args.method != "pid":
        try:
            import d3rlpy
            model_path = Path(args.model_dir) / f"{args.process}_{args.method}_v2103.d3"
            if model_path.exists():
                policy = d3rlpy.load_learnable(str(model_path))
                print(f"Loaded policy from {model_path}")
            else:
                print(f"WARNING: Model not found at {model_path}, using PID")
        except ImportError:
            print("WARNING: d3rlpy not available, using PID")
    
    # Create evaluator
    evaluator = PLCEvaluator(
        process=args.process,
        dry_run=args.dry_run,
    )
    
    try:
        df = evaluator.evaluate(
            policy=policy,
            n_episodes=args.episodes,
            episode_length=args.episode_length,
            output_dir=args.output_dir,
        )
    finally:
        evaluator.disconnect()


if __name__ == "__main__":
    main()
