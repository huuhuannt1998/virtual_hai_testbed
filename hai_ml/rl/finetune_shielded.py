"""
Shielded Online Fine-Tuning
===========================

Fine-tunes admitted policies with safety shield in online environment.

Usage:
    python -m hai_ml.rl.finetune_shielded \
        --task p3 \
        --admitted runs/p3_td3bc_s0.d3 \
        --episodes 20 \
        --max_interv 0.1 \
        --config hai_ml/configs/p3_baseline.yml \
        --out results/p3_finetune.csv
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ruamel.yaml import YAML

# Import local modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from hai_ml.envs.hai_gym import make_env, HaiEnv
from hai_ml.safety.shield import create_shield, Shield
from hai_ml.rl.ope_gate import PolicyWrapper


class ShieldedPolicy:
    """Policy wrapper that applies safety shield to actions."""
    
    def __init__(
        self,
        policy: PolicyWrapper,
        shield: Shield,
        action_names: List[str],
    ):
        self.policy = policy
        self.shield = shield
        self.action_names = action_names
    
    def predict(
        self,
        obs: np.ndarray,
        state_dict: Dict[str, float],
    ) -> tuple:
        """
        Predict shielded action.
        
        Returns:
            (safe_action, intervention_info)
        """
        # Get raw policy action
        raw_action = self.policy.predict(obs)
        
        # Apply shield
        safe_action, shield_info = self.shield.project(state_dict, raw_action)
        
        return safe_action, shield_info


def finetune_shielded(
    task: str,
    model_path: str,
    n_episodes: int = 20,
    max_intervention_rate: float = 0.1,
    config_path: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run shielded online fine-tuning/evaluation.
    
    Args:
        task: Task name (p1, p3, p12)
        model_path: Path to admitted model
        n_episodes: Number of episodes to run
        max_intervention_rate: Abort episode if exceeded
        config_path: Path to config file
        seed: Random seed
        
    Returns:
        Dictionary with episode metrics
    """
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
    
    # Load config
    config = {}
    if config_path and Path(config_path).exists():
        yaml = YAML(typ='safe')
        with open(config_path, 'r') as f:
            config = yaml.load(f)
    
    max_steps = config.get('max_steps', 500)
    
    # Create environment and shield
    print(f"Creating environment for task: {task}")
    env = make_env(task, mode='sim', max_steps=max_steps, seed=seed)
    shield = create_shield(task)
    
    # Load policy
    print(f"Loading policy from: {model_path}")
    policy = PolicyWrapper(model_path)
    shielded_policy = ShieldedPolicy(policy, shield, env.get_action_names())
    
    # Episode metrics storage
    episode_metrics = []
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        shield.reset()
        
        episode_reward = 0.0
        episode_steps = 0
        episode_violations = 0
        episode_interventions = 0
        tracking_errors = []
        action_changes = []
        prev_action = None
        
        done = False
        truncated = False
        aborted = False
        
        while not done and not truncated and not aborted:
            # Get state dict for shield
            state_dict = info.get('raw_state', {})
            
            # Get shielded action
            safe_action, shield_info = shielded_policy.predict(obs, state_dict)
            
            if shield_info['intervened']:
                episode_interventions += 1
            
            # Check intervention rate
            current_intervention_rate = episode_interventions / (episode_steps + 1)
            if current_intervention_rate > max_intervention_rate and episode_steps > 10:
                print(f"  Episode {ep+1}: Aborted - intervention rate {current_intervention_rate:.2%} > {max_intervention_rate:.2%}")
                aborted = True
                break
            
            # Step environment
            next_obs, reward, done, truncated, info = env.step(safe_action)
            
            episode_reward += reward
            episode_steps += 1
            episode_violations += info.get('violations', 0)
            
            # Track metrics
            if env.track_targets:
                raw_state = info.get('raw_state', {})
                error = 0.0
                for tag, target in env.track_targets.items():
                    if tag in raw_state:
                        error += (raw_state[tag] - target) ** 2
                tracking_errors.append(np.sqrt(error))
            
            if prev_action is not None:
                action_changes.append(np.sum(np.abs(safe_action - prev_action)))
            prev_action = safe_action.copy()
            
            obs = next_obs
        
        # Compute episode metrics
        intervention_rate = episode_interventions / max(episode_steps, 1)
        avg_tracking_error = np.mean(tracking_errors) if tracking_errors else 0.0
        wear = np.sum(action_changes) if action_changes else 0.0
        success = not aborted and episode_violations == 0
        
        metrics = {
            'episode': ep + 1,
            'reward': episode_reward,
            'steps': episode_steps,
            'violations': episode_violations,
            'interventions': episode_interventions,
            'intervention_rate': intervention_rate,
            'tracking_error': avg_tracking_error,
            'wear': wear,
            'success': success,
            'aborted': aborted,
        }
        
        episode_metrics.append(metrics)
        
        print(f"  Episode {ep+1}: reward={episode_reward:.2f}, steps={episode_steps}, "
              f"interv_rate={intervention_rate:.2%}, success={success}")
    
    env.close()
    
    # Aggregate statistics
    success_rate = np.mean([m['success'] for m in episode_metrics])
    avg_reward = np.mean([m['reward'] for m in episode_metrics])
    avg_intervention_rate = np.mean([m['intervention_rate'] for m in episode_metrics])
    
    summary = {
        'task': task,
        'model': model_path,
        'n_episodes': n_episodes,
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_intervention_rate': avg_intervention_rate,
        'episodes': episode_metrics,
    }
    
    print(f"\nSummary: success_rate={success_rate:.2%}, avg_reward={avg_reward:.2f}, "
          f"avg_interv_rate={avg_intervention_rate:.2%}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Shielded online fine-tuning'
    )
    parser.add_argument(
        '--task', type=str, required=True,
        choices=['p1', 'p3', 'p12'],
        help='Task name'
    )
    parser.add_argument(
        '--admitted', type=str, required=True,
        help='Path to admitted model'
    )
    parser.add_argument(
        '--episodes', type=int, default=20,
        help='Number of episodes'
    )
    parser.add_argument(
        '--max_interv', type=float, default=0.1,
        help='Maximum intervention rate before abort'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to config file'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--out', type=str, required=True,
        help='Output CSV path'
    )
    
    args = parser.parse_args()
    
    # Run fine-tuning
    results = finetune_shielded(
        task=args.task,
        model_path=args.admitted,
        n_episodes=args.episodes,
        max_intervention_rate=args.max_interv,
        config_path=args.config,
        seed=args.seed,
    )
    
    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'episode', 'reward', 'steps', 'violations', 'interventions',
            'intervention_rate', 'tracking_error', 'wear', 'success', 'aborted'
        ])
        writer.writeheader()
        for m in results['episodes']:
            writer.writerow(m)
    
    print(f"Saved results to: {args.out}")
    
    # Save JSON summary
    json_path = out_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump({
            'task': results['task'],
            'model': results['model'],
            'n_episodes': results['n_episodes'],
            'success_rate': results['success_rate'],
            'avg_reward': results['avg_reward'],
            'avg_intervention_rate': results['avg_intervention_rate'],
        }, f, indent=2)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
