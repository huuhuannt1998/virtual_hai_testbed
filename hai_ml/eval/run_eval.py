"""
Evaluation Harness
==================

Main evaluation harness for nominal, attack, and cross-version runs.

Usage:
    python -m hai_ml.eval.run_eval \
        --task p3 \
        --policy td3bc --model runs/p3_td3bc_s0.d3 \
        --schema hai_ml/schemas/p3.yaml \
        --episodes 20 \
        --attack none \
        --config hai_ml/configs/p3_baseline.yml \
        --out results/p3_nominal.csv

Attack modes:
    none     - Normal operation
    hostile  - Attempt unsafe setpoints
    flood    - Rapid step commands
    bias     - Add sensor bias
    delay    - Add observation/command latency
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from ruamel.yaml import YAML

# Import local modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hai_ml.envs.hai_gym import make_env, HaiEnv
from hai_ml.safety.shield import create_shield, Shield
from hai_ml.eval.metrics import (
    MetricCollector, itae, ise, wear, energy,
    violation_rate, intervention_rate, compute_latency_stats,
    aggregate_episode_metrics
)


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    task: str = 'p3'
    policy: str = 'td3bc'
    model_path: str = ''
    schema_path: str = ''
    n_episodes: int = 20
    max_steps: int = 500
    attack: str = 'none'
    attack_strength: float = 0.1
    attack_start: int = 100
    attack_duration: int = 200
    seed: int = 42
    version_train: str = ''
    version_test: str = ''


class Attack:
    """Base class for attack injection."""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.active = False
        self.n_blocked = 0
        self.n_attempts = 0
    
    def should_attack(self, step: int) -> bool:
        """Check if attack should be active at this step."""
        start = self.config.attack_start
        end = start + self.config.attack_duration
        return start <= step < end
    
    def inject_observation(
        self,
        obs: np.ndarray,
        step: int,
    ) -> np.ndarray:
        """Modify observation (for sensor attacks)."""
        return obs
    
    def inject_action(
        self,
        action: np.ndarray,
        step: int,
    ) -> np.ndarray:
        """Modify action (for actuator attacks)."""
        return action
    
    def on_blocked(self):
        """Called when attack is blocked by shield."""
        self.n_blocked += 1
    
    def on_attempt(self):
        """Called when attack is attempted."""
        self.n_attempts += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get attack statistics."""
        return {
            'n_attempts': self.n_attempts,
            'n_blocked': self.n_blocked,
            'blocked_pct': (self.n_blocked / max(self.n_attempts, 1)) * 100,
        }


class HostileAttack(Attack):
    """
    Hostile attack: inject unsafe setpoints.
    
    Attempts to drive actuators to dangerous values.
    """
    
    def inject_action(
        self,
        action: np.ndarray,
        step: int,
    ) -> np.ndarray:
        if not self.should_attack(step):
            return action
        
        self.on_attempt()
        
        # Try to set all actions to extreme values
        hostile_action = action.copy()
        for i in range(len(action)):
            # Alternate between max and min
            if i % 2 == 0:
                hostile_action[i] = 1.0  # Max
            else:
                hostile_action[i] = -1.0  # Min
        
        return hostile_action


class FloodAttack(Attack):
    """
    Flood attack: rapid step commands.
    
    Sends rapidly changing commands to stress rate limits.
    """
    
    def inject_action(
        self,
        action: np.ndarray,
        step: int,
    ) -> np.ndarray:
        if not self.should_attack(step):
            return action
        
        self.on_attempt()
        
        # Oscillate rapidly
        flood_action = action.copy()
        amplitude = self.config.attack_strength
        freq = step * 0.5
        
        for i in range(len(action)):
            flood_action[i] += amplitude * np.sin(freq + i * np.pi/2)
        
        return np.clip(flood_action, -1.0, 1.0)


class BiasAttack(Attack):
    """
    Bias attack: add constant bias to sensors.
    
    Simulates compromised sensors reporting incorrect values.
    """
    
    def inject_observation(
        self,
        obs: np.ndarray,
        step: int,
    ) -> np.ndarray:
        if not self.should_attack(step):
            return obs
        
        self.on_attempt()
        
        # Add bias to first few sensors
        biased_obs = obs.copy()
        bias = self.config.attack_strength * 3  # Scaled bias
        
        n_biased = min(3, len(obs))
        biased_obs[:n_biased] += bias
        
        return biased_obs


class DelayAttack(Attack):
    """
    Delay attack: add latency to observations/commands.
    
    Simulates network delays or DoS conditions.
    """
    
    def __init__(self, config: EvalConfig):
        super().__init__(config)
        self.obs_buffer: List[np.ndarray] = []
        self.delay_steps = int(config.attack_strength * 10)  # 0.1 -> 1 step delay
    
    def inject_observation(
        self,
        obs: np.ndarray,
        step: int,
    ) -> np.ndarray:
        if not self.should_attack(step):
            self.obs_buffer = []
            return obs
        
        self.on_attempt()
        
        # Buffer observations and return delayed version
        self.obs_buffer.append(obs.copy())
        
        if len(self.obs_buffer) > self.delay_steps:
            delayed = self.obs_buffer.pop(0)
        else:
            delayed = self.obs_buffer[0]
        
        return delayed


def create_attack(attack_type: str, config: EvalConfig) -> Attack:
    """Factory function for attack creation."""
    attacks = {
        'none': Attack,
        'hostile': HostileAttack,
        'flood': FloodAttack,
        'bias': BiasAttack,
        'delay': DelayAttack,
    }
    
    if attack_type not in attacks:
        print(f"Unknown attack type: {attack_type}, using none")
        attack_type = 'none'
    
    return attacks[attack_type](config)


class PolicyLoader:
    """Load and wrap different policy types."""
    
    @staticmethod
    def load(
        policy_type: str,
        model_path: str,
        schema_path: Optional[str] = None,
    ):
        """Load policy from file."""
        if policy_type == 'bc':
            from hai_ml.il.train_bc import BCTrainer
            trainer = BCTrainer.load(model_path)
            return trainer
        
        elif policy_type in ['td3bc', 'cql', 'iql']:
            from hai_ml.rl.ope_gate import PolicyWrapper
            return PolicyWrapper(model_path, algo=policy_type)
        
        elif policy_type == 'mpc':
            from hai_ml.mpc.plan_cem import MPCPolicy
            return MPCPolicy(model_path, schema_path)
        
        elif policy_type == 'pid':
            from hai_ml.mpc.plan_cem import TunedPIDPolicy
            # Get dims from schema
            if schema_path:
                yaml = YAML(typ='safe')
                with open(schema_path, 'r') as f:
                    schema = yaml.load(f)
                n_states = len(schema['state_tags'])
                n_actions = len(schema['action_tags'])
            else:
                n_states = 12
                n_actions = 6
            return TunedPIDPolicy(n_states, n_actions)
        
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")


def run_episode(
    env: HaiEnv,
    shield: Shield,
    policy: Any,
    attack: Attack,
    config: EvalConfig,
    episode_idx: int,
) -> Dict[str, Any]:
    """
    Run a single evaluation episode.
    
    Returns episode metrics.
    """
    np.random.seed(config.seed + episode_idx)
    
    obs, info = env.reset(seed=config.seed + episode_idx)
    shield.reset()
    attack.obs_buffer = [] if hasattr(attack, 'obs_buffer') else None
    
    if hasattr(policy, 'reset'):
        policy.reset()
    
    collector = MetricCollector()
    latencies = []
    
    done = False
    truncated = False
    step = 0
    total_reward = 0.0
    
    all_actions_original = []
    all_actions_shielded = []
    
    while not done and not truncated and step < config.max_steps:
        step_start = time.perf_counter()
        
        # Apply observation attack
        attacked_obs = attack.inject_observation(obs, step)
        
        # Get policy action
        t_policy_start = time.perf_counter()
        if hasattr(policy, 'predict'):
            raw_action = policy.predict(attacked_obs)
        else:
            raw_action = np.zeros(env.action_space.shape)
        t_policy = time.perf_counter() - t_policy_start
        
        # Apply action attack
        attacked_action = attack.inject_action(raw_action, step)
        
        # Apply shield
        t_shield_start = time.perf_counter()
        state_dict = info.get('raw_state', {})
        safe_action, shield_info = shield.project(state_dict, attacked_action)
        t_shield = time.perf_counter() - t_shield_start
        
        # Track if attack was blocked
        if attack.should_attack(step) and shield_info['intervened']:
            attack.on_blocked()
        
        all_actions_original.append(attacked_action.copy())
        all_actions_shielded.append(safe_action.copy())
        
        # Step environment
        t_env_start = time.perf_counter()
        next_obs, reward, done, truncated, info = env.step(safe_action)
        t_env = time.perf_counter() - t_env_start
        
        # Total step latency
        step_latency = (time.perf_counter() - step_start) * 1000  # ms
        latencies.append(step_latency)
        
        # Compute tracking error
        error = np.linalg.norm(next_obs)  # Assuming centered at 0
        
        # Record metrics
        collector.step(
            error=error,
            action=safe_action,
            state=next_obs,
            reward=reward,
            violation=info.get('violations', 0) > 0,
            intervention=shield_info['intervened'],
            latency=step_latency,
        )
        
        total_reward += reward
        obs = next_obs
        step += 1
    
    # Compute episode summary
    summary = collector.compute_summary(dt=env.dt)
    
    # Add episode-level metrics
    summary['episode'] = episode_idx
    summary['total_reward'] = total_reward
    summary['n_steps'] = step
    summary['completed'] = not done and step >= config.max_steps
    summary['success'] = summary['n_violations'] == 0 and summary['completed']
    
    # Attack stats
    attack_stats = attack.get_stats()
    summary['attack_attempts'] = attack_stats['n_attempts']
    summary['attack_blocked'] = attack_stats['n_blocked']
    summary['attack_blocked_pct'] = attack_stats['blocked_pct']
    
    # Intervention rate from raw vs shielded actions
    if all_actions_original:
        orig = np.array(all_actions_original)
        shield_act = np.array(all_actions_shielded)
        summary['intervention_rate'] = intervention_rate(orig, shield_act)
    
    # Latency stats
    lat_stats = compute_latency_stats(np.array(latencies))
    for k, v in lat_stats.items():
        summary[f'latency_{k}'] = v
    
    return summary


def run_evaluation(config: EvalConfig) -> Dict[str, Any]:
    """
    Run full evaluation with multiple episodes.
    
    Returns aggregated results.
    """
    print(f"=" * 60)
    print(f"Evaluation: {config.task} / {config.policy} / attack={config.attack}")
    print(f"=" * 60)
    
    # Create environment
    schema_path = config.schema_path
    if not schema_path:
        schema_path = f'hai_ml/schemas/{config.task}.yaml'
    
    env = make_env(
        config.task,
        mode='sim',
        max_steps=config.max_steps,
        seed=config.seed,
    )
    
    # Create shield
    shield = create_shield(config.task)
    
    # Load policy
    policy = PolicyLoader.load(
        config.policy,
        config.model_path,
        schema_path,
    )
    
    # Run episodes
    all_metrics = []
    all_latencies = []
    
    for ep in range(config.n_episodes):
        # Create fresh attack for each episode
        attack = create_attack(config.attack, config)
        
        metrics = run_episode(
            env=env,
            shield=shield,
            policy=policy,
            attack=attack,
            config=config,
            episode_idx=ep,
        )
        
        all_metrics.append(metrics)
        
        # Collect latencies for detailed logging
        all_latencies.extend(metrics.get('latencies', []))
        
        print(f"  Episode {ep+1}/{config.n_episodes}: "
              f"reward={metrics['total_reward']:.2f}, "
              f"violations={metrics['n_violations']}, "
              f"interv_rate={metrics['intervention_rate']:.2%}, "
              f"lat_p99={metrics.get('latency_p99', 0):.1f}ms")
    
    env.close()
    
    # Aggregate metrics
    aggregated = aggregate_episode_metrics(all_metrics)
    
    # Add config info
    aggregated['task'] = config.task
    aggregated['policy'] = config.policy
    aggregated['attack'] = config.attack
    aggregated['n_episodes'] = config.n_episodes
    
    print(f"\nSummary:")
    print(f"  Success rate: {aggregated.get('success_rate', 0):.2%}")
    print(f"  Avg reward: {aggregated.get('total_reward_mean', 0):.2f}")
    print(f"  Avg violations: {aggregated.get('n_violations_mean', 0):.2f}")
    print(f"  Avg intervention rate: {aggregated.get('intervention_rate_mean', 0):.2%}")
    print(f"  P99 latency: {aggregated.get('latency_p99_mean', 0):.1f}ms")
    
    return {
        'config': vars(config),
        'episodes': all_metrics,
        'summary': aggregated,
        'all_latencies': all_latencies,
    }


def save_results(
    results: Dict[str, Any],
    output_path: str,
    latency_path: Optional[str] = None,
):
    """Save evaluation results to files."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save episode CSV
    episodes = results['episodes']
    if episodes:
        fieldnames = list(episodes[0].keys())
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for ep in episodes:
                writer.writerow(ep)
        print(f"Saved episode results to: {out_path}")
    
    # Save JSON with full results
    json_path = out_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        json.dump(convert(results), f, indent=2)
    print(f"Saved JSON results to: {json_path}")
    
    # Save latency CSV
    if latency_path and results.get('all_latencies'):
        lat_path = Path(latency_path)
        lat_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(lat_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'latency_ms'])
            for i, lat in enumerate(results['all_latencies']):
                writer.writerow([i, lat])
        print(f"Saved latency data to: {lat_path}")


def main():
    parser = argparse.ArgumentParser(
        description='HAI-ML Evaluation Harness'
    )
    parser.add_argument(
        '--task', type=str, required=True,
        choices=['p1', 'p3', 'p12'],
        help='Task name'
    )
    parser.add_argument(
        '--policy', type=str, required=True,
        choices=['bc', 'td3bc', 'cql', 'iql', 'mpc', 'pid'],
        help='Policy type'
    )
    parser.add_argument(
        '--model', type=str, required=True,
        help='Path to model file'
    )
    parser.add_argument(
        '--schema', type=str, default=None,
        help='Path to schema YAML'
    )
    parser.add_argument(
        '--episodes', type=int, default=20,
        help='Number of episodes'
    )
    parser.add_argument(
        '--max-steps', type=int, default=500,
        help='Max steps per episode'
    )
    parser.add_argument(
        '--attack', type=str, default='none',
        choices=['none', 'hostile', 'flood', 'bias', 'delay'],
        help='Attack type'
    )
    parser.add_argument(
        '--attack-strength', type=float, default=0.1,
        help='Attack strength (0-1)'
    )
    parser.add_argument(
        '--attack-start', type=int, default=100,
        help='Step to start attack'
    )
    parser.add_argument(
        '--attack-duration', type=int, default=200,
        help='Attack duration in steps'
    )
    parser.add_argument(
        '--version', type=str, default=None,
        help='Version string (train:X,test:Y)'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to config YAML'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--out', type=str, required=True,
        help='Output CSV path'
    )
    parser.add_argument(
        '--latency-out', type=str, default=None,
        help='Output path for latency CSV'
    )
    
    args = parser.parse_args()
    
    # Parse version
    version_train = ''
    version_test = ''
    if args.version:
        parts = args.version.split(',')
        for p in parts:
            if p.startswith('train:'):
                version_train = p.split(':')[1]
            elif p.startswith('test:'):
                version_test = p.split(':')[1]
    
    # Create config
    config = EvalConfig(
        task=args.task,
        policy=args.policy,
        model_path=args.model,
        schema_path=args.schema or f'hai_ml/schemas/{args.task}.yaml',
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        attack=args.attack,
        attack_strength=args.attack_strength,
        attack_start=args.attack_start,
        attack_duration=args.attack_duration,
        seed=args.seed,
        version_train=version_train,
        version_test=version_test,
    )
    
    # Load additional config from YAML
    if args.config and Path(args.config).exists():
        yaml = YAML(typ='safe')
        with open(args.config, 'r') as f:
            yaml_config = yaml.load(f)
        # Override with YAML values (if not set via CLI)
        if 'max_steps' in yaml_config and args.max_steps == 500:
            config.max_steps = yaml_config['max_steps']
    
    # Run evaluation
    results = run_evaluation(config)
    
    # Determine latency output path
    latency_path = args.latency_out
    if latency_path is None:
        out_stem = Path(args.out).stem
        latency_path = str(Path(args.out).parent / f"{out_stem}_latency.csv")
    
    # Save results
    save_results(results, args.out, latency_path)
    
    # Check quality gates
    summary = results['summary']
    p99_lat = summary.get('latency_p99_mean', 0)
    
    print(f"\n{'='*60}")
    print("Quality Gates:")
    if p99_lat < 100:
        print(f"  ✓ P99 latency: {p99_lat:.1f}ms < 100ms")
    else:
        print(f"  ✗ P99 latency: {p99_lat:.1f}ms >= 100ms")
    
    success_rate = summary.get('success_rate', 0)
    if success_rate >= 0.8:
        print(f"  ✓ Success rate: {success_rate:.1%} >= 80%")
    else:
        print(f"  ✗ Success rate: {success_rate:.1%} < 80%")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
