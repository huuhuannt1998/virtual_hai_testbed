"""
OPE Gating Module
=================

Off-Policy Evaluation with FQE and WIS, plus bootstrap CIs for gating.

Usage:
    python -m hai_ml.rl.ope_gate \
        --npz hai_ml/data/p3_21_03.npz \
        --models runs/p3_td3bc_s0.d3 runs/p3_cql_s0.d3 runs/p3_iql_s0.d3 \
        --baseline bc \
        --n_bootstrap 50 \
        --out results/p3_ope.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import d3rlpy
    from d3rlpy.dataset import MDPDataset
    from d3rlpy.ope import FQE, FQEConfig
    HAS_D3RLPY = True
except ImportError:
    HAS_D3RLPY = False


class PolicyWrapper:
    """Wrapper to load and predict from saved policies."""
    
    def __init__(self, model_path: str, algo: str = 'unknown'):
        self.model_path = model_path
        self.algo = algo
        self.policy = None
        self.meta = None
        self._load()
    
    def _load(self):
        """Load model from file."""
        path = Path(self.model_path)
        
        # Try to load d3rlpy model
        if HAS_D3RLPY and path.suffix == '.d3':
            try:
                # Check for algo type from meta
                meta_path = path.with_suffix('.meta.json')
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        self.meta = json.load(f)
                    self.algo = self.meta.get('algo', self.algo)
                
                # Load d3rlpy model
                if self.algo == 'td3bc':
                    from d3rlpy.algos import TD3PlusBCConfig
                    self.policy = TD3PlusBCConfig().create()
                elif self.algo == 'cql':
                    from d3rlpy.algos import CQLConfig
                    self.policy = CQLConfig().create()
                elif self.algo == 'iql':
                    from d3rlpy.algos import IQLConfig
                    self.policy = IQLConfig().create()
                else:
                    from d3rlpy.algos import TD3PlusBCConfig
                    self.policy = TD3PlusBCConfig().create()
                
                self.policy.load_model(str(path))
                return
            except Exception as e:
                print(f"Failed to load d3rlpy model: {e}")
        
        # Try to load PyTorch model
        if HAS_TORCH:
            try:
                checkpoint = torch.load(str(path), map_location='cpu')
                self.meta = checkpoint
                
                obs_dim = checkpoint.get('obs_dim', 12)
                action_dim = checkpoint.get('action_dim', 6)
                
                # Create simple policy network
                class SimplePolicy(nn.Module):
                    def __init__(self, obs_dim, action_dim):
                        super().__init__()
                        self.net = nn.Sequential(
                            nn.Linear(obs_dim, 256),
                            nn.ReLU(),
                            nn.Linear(256, 256),
                            nn.ReLU(),
                            nn.Linear(256, action_dim),
                            nn.Tanh(),
                        )
                    
                    def forward(self, x):
                        return self.net(x)
                
                self.policy = SimplePolicy(obs_dim, action_dim)
                if 'policy_state' in checkpoint:
                    self.policy.load_state_dict(checkpoint['policy_state'])
                self.policy.eval()
                return
            except Exception as e:
                print(f"Failed to load PyTorch model: {e}")
        
        print(f"Warning: Could not load model from {self.model_path}")
    
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict action from observation."""
        if self.policy is None:
            # Return random action
            return np.random.randn(6) * 0.1
        
        if HAS_D3RLPY and hasattr(self.policy, 'predict'):
            return self.policy.predict(obs)
        
        if HAS_TORCH and isinstance(self.policy, nn.Module):
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs)
                if obs_t.dim() == 1:
                    obs_t = obs_t.unsqueeze(0)
                action = self.policy(obs_t)
                return action.cpu().numpy().squeeze()
        
        return np.random.randn(6) * 0.1


class FQEvaluator:
    """
    Fitted Q Evaluation (FQE) for off-policy evaluation.
    
    Estimates expected return of a target policy using the offline dataset.
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        n_epochs: int = 100,
        batch_size: int = 256,
        hidden_dims: List[int] = [256, 256],
    ):
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.hidden_dims = hidden_dims
    
    def evaluate(
        self,
        policy: PolicyWrapper,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        terminals: np.ndarray,
    ) -> float:
        """
        Evaluate policy using FQE.
        
        Returns estimated expected return.
        """
        if HAS_D3RLPY and hasattr(policy.policy, 'predict'):
            return self._evaluate_d3rlpy(
                policy, observations, actions, rewards,
                next_observations, terminals
            )
        
        return self._evaluate_simple(
            policy, observations, actions, rewards,
            next_observations, terminals
        )
    
    def _evaluate_d3rlpy(
        self,
        policy: PolicyWrapper,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        terminals: np.ndarray,
    ) -> float:
        """Use d3rlpy's FQE implementation."""
        try:
            dataset = MDPDataset(
                observations=observations,
                actions=actions,
                rewards=rewards,
                terminals=terminals.astype(bool),
            )
            
            fqe_config = FQEConfig(
                learning_rate=1e-3,
                gamma=self.gamma,
                batch_size=self.batch_size,
            )
            fqe = fqe_config.create()
            fqe.build_with_dataset(dataset)
            
            # Fit FQE
            fqe.fit(
                dataset,
                n_steps=self.n_epochs * len(dataset),
                n_steps_per_epoch=len(dataset),
                show_progress=False,
            )
            
            # Estimate initial state values
            initial_obs = observations[terminals.astype(bool).nonzero()[0] - 1]
            if len(initial_obs) == 0:
                initial_obs = observations[:100]
            
            values = fqe.predict_value(initial_obs, policy.policy.predict(initial_obs))
            return float(np.mean(values))
        except Exception as e:
            print(f"d3rlpy FQE failed: {e}")
            return self._evaluate_simple(
                policy, observations, actions, rewards,
                next_observations, terminals
            )
    
    def _evaluate_simple(
        self,
        policy: PolicyWrapper,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        terminals: np.ndarray,
    ) -> float:
        """
        Simple Monte Carlo estimation using behavior policy data.
        
        This is an approximation when FQE is not available.
        """
        # Compute discounted returns from the dataset
        n_samples = len(rewards)
        returns = np.zeros(n_samples)
        
        # Backward pass to compute returns
        G = 0.0
        for i in range(n_samples - 1, -1, -1):
            if terminals[i]:
                G = 0.0
            G = rewards[i] + self.gamma * G
            returns[i] = G
        
        # Weight by importance ratio (simplified: assume similar behavior)
        # In practice, would need behavior policy log probs
        # Here we use a simple average as baseline
        return float(np.mean(returns))


class WISEvaluator:
    """
    Weighted Importance Sampling (WIS) evaluator.
    
    Reweights offline trajectories by importance ratio.
    """
    
    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
    
    def evaluate(
        self,
        policy: PolicyWrapper,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        behavior_actions: Optional[np.ndarray] = None,
    ) -> float:
        """
        Evaluate using WIS.
        
        Returns estimated expected return.
        """
        # Simplified WIS: weight by action similarity
        # In practice, need log probs from both policies
        
        n_samples = len(rewards)
        
        # Compute policy actions
        policy_actions = np.zeros_like(actions)
        batch_size = 1000
        for i in range(0, n_samples, batch_size):
            end_i = min(i + batch_size, n_samples)
            policy_actions[i:end_i] = policy.predict(observations[i:end_i])
        
        # Compute action similarity as proxy for importance ratio
        # Higher similarity = higher weight
        action_diff = np.sum((actions - policy_actions) ** 2, axis=1)
        action_similarity = np.exp(-0.5 * action_diff)  # Gaussian kernel
        
        # Compute weighted returns
        returns = np.zeros(n_samples)
        weights = np.ones(n_samples)
        
        G = 0.0
        w = 1.0
        for i in range(n_samples - 1, -1, -1):
            if terminals[i]:
                G = 0.0
                w = 1.0
            G = rewards[i] + self.gamma * G
            w *= action_similarity[i]
            returns[i] = G
            weights[i] = w
        
        # Normalized weighted average
        if np.sum(weights) > 1e-8:
            return float(np.sum(weights * returns) / np.sum(weights))
        return float(np.mean(returns))


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 50,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.
    
    Returns (mean, ci_lower, ci_upper).
    """
    n_samples = len(values)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        bootstrap_means.append(np.mean(values[indices]))
    
    bootstrap_means = np.array(bootstrap_means)
    
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    mean = np.mean(values)
    
    return mean, ci_lower, ci_upper


def ope_gate(
    npz_path: str,
    model_paths: List[str],
    baseline_path: Optional[str] = None,
    n_bootstrap: int = 50,
    gate_threshold: float = 0.0,
) -> Dict[str, Any]:
    """
    Run OPE on multiple models and gate based on CIs.
    
    Args:
        npz_path: Path to dataset
        model_paths: List of model paths to evaluate
        baseline_path: Path to baseline model (BC)
        n_bootstrap: Number of bootstrap samples
        gate_threshold: Minimum CI lower bound for admission
        
    Returns:
        Dictionary with OPE results for each model
    """
    # Load dataset
    data = np.load(npz_path, allow_pickle=True)
    observations = data['observations'].astype(np.float32)
    actions = data['actions'].astype(np.float32)
    rewards = data['rewards'].astype(np.float32)
    next_observations = data['next_observations'].astype(np.float32)
    terminals = data['terminals'].astype(np.float32)
    
    # Normalize
    s_mean = data['s_mean']
    s_std = data['s_std']
    observations = (observations - s_mean) / s_std
    next_observations = (next_observations - s_mean) / s_std
    
    # Normalize actions
    a_min = actions.min(axis=0)
    a_max = actions.max(axis=0)
    a_range = np.where(a_max - a_min < 1e-6, 1.0, a_max - a_min)
    actions = 2.0 * (actions - a_min) / a_range - 1.0
    
    # Create evaluators
    fqe = FQEvaluator()
    wis = WISEvaluator()
    
    results = []
    
    # Evaluate baseline first
    bc_ci_lower = None
    if baseline_path:
        print(f"Evaluating baseline: {baseline_path}")
        bc_policy = PolicyWrapper(baseline_path, algo='bc')
        
        # Bootstrap FQE
        bc_values = []
        for b in range(n_bootstrap):
            indices = np.random.choice(len(observations), len(observations), replace=True)
            val = fqe.evaluate(
                bc_policy,
                observations[indices], actions[indices], rewards[indices],
                next_observations[indices], terminals[indices]
            )
            bc_values.append(val)
        
        bc_mean, bc_lo, bc_hi = bootstrap_ci(np.array(bc_values), n_bootstrap)
        bc_wis = wis.evaluate(bc_policy, observations, actions, rewards, terminals)
        bc_ci_lower = bc_lo
        
        results.append({
            'model': baseline_path,
            'algo': 'bc',
            'fqe_mean': bc_mean,
            'fqe_ci': [bc_lo, bc_hi],
            'wis': bc_wis,
            'admit': True,  # Baseline always admitted
        })
        
        print(f"  BC: FQE={bc_mean:.4f} [{bc_lo:.4f}, {bc_hi:.4f}], WIS={bc_wis:.4f}")
    
    # Evaluate each model
    for model_path in model_paths:
        print(f"Evaluating: {model_path}")
        
        # Infer algo from path
        path_str = str(model_path).lower()
        if 'td3bc' in path_str:
            algo = 'td3bc'
        elif 'cql' in path_str:
            algo = 'cql'
        elif 'iql' in path_str:
            algo = 'iql'
        else:
            algo = 'unknown'
        
        policy = PolicyWrapper(model_path, algo=algo)
        
        # Bootstrap FQE
        fqe_values = []
        for b in range(n_bootstrap):
            indices = np.random.choice(len(observations), len(observations), replace=True)
            val = fqe.evaluate(
                policy,
                observations[indices], actions[indices], rewards[indices],
                next_observations[indices], terminals[indices]
            )
            fqe_values.append(val)
        
        fqe_mean, fqe_lo, fqe_hi = bootstrap_ci(np.array(fqe_values), n_bootstrap)
        wis_val = wis.evaluate(policy, observations, actions, rewards, terminals)
        
        # Admission gate
        admit = fqe_lo > gate_threshold
        if bc_ci_lower is not None:
            admit = admit and (fqe_lo > bc_ci_lower)
        
        results.append({
            'model': model_path,
            'algo': algo,
            'fqe_mean': fqe_mean,
            'fqe_ci': [fqe_lo, fqe_hi],
            'wis': wis_val,
            'admit': admit,
        })
        
        status = "ADMIT" if admit else "REJECT"
        print(f"  {algo.upper()}: FQE={fqe_mean:.4f} [{fqe_lo:.4f}, {fqe_hi:.4f}], WIS={wis_val:.4f} -> {status}")
    
    return {
        'results': results,
        'bc_ci_lower': bc_ci_lower,
        'gate_threshold': gate_threshold,
        'n_bootstrap': n_bootstrap,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Off-Policy Evaluation with gating'
    )
    parser.add_argument(
        '--npz', type=str, required=True,
        help='Path to NPZ dataset file'
    )
    parser.add_argument(
        '--models', type=str, nargs='+', required=True,
        help='Paths to model files to evaluate'
    )
    parser.add_argument(
        '--baseline', type=str, default=None,
        help='Path to baseline (BC) model'
    )
    parser.add_argument(
        '--n_bootstrap', type=int, default=50,
        help='Number of bootstrap samples'
    )
    parser.add_argument(
        '--gate_threshold', type=float, default=0.0,
        help='Minimum CI lower bound for admission'
    )
    parser.add_argument(
        '--out', type=str, required=True,
        help='Output path for JSON results'
    )
    parser.add_argument(
        '--csv', type=str, default=None,
        help='Output path for CSV leaderboard'
    )
    
    args = parser.parse_args()
    
    # Run OPE
    results = ope_gate(
        npz_path=args.npz,
        model_paths=args.models,
        baseline_path=args.baseline,
        n_bootstrap=args.n_bootstrap,
        gate_threshold=args.gate_threshold,
    )
    
    # Save JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved OPE results to: {args.out}")
    
    # Save CSV leaderboard
    csv_path = args.csv
    if csv_path is None:
        csv_path = out_path.with_name(out_path.stem.replace('ope', 'offline_leaderboard')).with_suffix('.csv')
    
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['algo', 'fqe_mean', 'fqe_lo', 'fqe_hi', 'wis', 'admit'])
        for r in results['results']:
            writer.writerow([
                r['algo'],
                f"{r['fqe_mean']:.4f}",
                f"{r['fqe_ci'][0]:.4f}",
                f"{r['fqe_ci'][1]:.4f}",
                f"{r['wis']:.4f}",
                str(r['admit']).lower(),
            ])
    
    print(f"Saved leaderboard to: {csv_path}")
    
    # Print summary
    admitted = [r for r in results['results'] if r['admit']]
    print(f"\nAdmitted models: {len(admitted)}/{len(results['results'])}")
    for r in admitted:
        print(f"  - {r['algo']}: FQE CI = [{r['fqe_ci'][0]:.4f}, {r['fqe_ci'][1]:.4f}]")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
