"""
Offline RL Trainers
===================

Trains offline RL policies using d3rlpy (TD3+BC, CQL, IQL).

Usage:
    python -m hai_ml.rl.train_offline \
        --npz hai_ml/data/p3_21_03.npz \
        --algo td3bc \
        --steps 300000 \
        --seed 0 \
        --out runs/p3_td3bc_s0.d3
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

# d3rlpy imports
try:
    import d3rlpy
    from d3rlpy.dataset import MDPDataset
    from d3rlpy.algos import TD3PlusBCConfig, CQLConfig, IQLConfig
    from d3rlpy.metrics import TDErrorEvaluator, AverageValueEstimationEvaluator
    HAS_D3RLPY = True
except ImportError:
    HAS_D3RLPY = False
    print("Warning: d3rlpy not installed. Install with: pip install d3rlpy>=2.6")


def load_dataset(npz_path: str) -> Dict[str, np.ndarray]:
    """Load dataset from NPZ file."""
    data = np.load(npz_path, allow_pickle=True)
    return {
        'observations': data['observations'].astype(np.float32),
        'actions': data['actions'].astype(np.float32),
        'rewards': data['rewards'].astype(np.float32),
        'next_observations': data['next_observations'].astype(np.float32),
        'terminals': data['terminals'].astype(np.float32),
        's_mean': data['s_mean'],
        's_std': data['s_std'],
        'a_mean': data.get('a_mean', None),
        'a_std': data.get('a_std', None),
    }


def normalize_observations(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Z-score normalize observations."""
    s_mean = data['s_mean']
    s_std = data['s_std']
    
    data['observations'] = (data['observations'] - s_mean) / s_std
    data['next_observations'] = (data['next_observations'] - s_mean) / s_std
    
    return data


def normalize_actions(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Normalize actions to [-1, 1] range."""
    actions = data['actions']
    a_min = actions.min(axis=0)
    a_max = actions.max(axis=0)
    a_range = a_max - a_min
    a_range = np.where(a_range < 1e-6, 1.0, a_range)
    
    data['actions'] = 2.0 * (actions - a_min) / a_range - 1.0
    data['a_min'] = a_min
    data['a_max'] = a_max
    
    return data


def create_mdp_dataset(data: Dict[str, np.ndarray]) -> 'MDPDataset':
    """Create d3rlpy MDPDataset from data dict."""
    if not HAS_D3RLPY:
        raise RuntimeError("d3rlpy is required for offline RL training")
    
    # d3rlpy expects terminals as boolean
    terminals = data['terminals'].astype(bool)
    
    # Create dataset
    dataset = MDPDataset(
        observations=data['observations'],
        actions=data['actions'],
        rewards=data['rewards'],
        terminals=terminals,
    )
    
    return dataset


def create_algorithm(algo_name: str, **kwargs) -> Any:
    """
    Create d3rlpy algorithm by name.
    
    Supported: td3bc, cql, iql
    """
    if not HAS_D3RLPY:
        raise RuntimeError("d3rlpy is required for offline RL training")
    
    algo_name = algo_name.lower()
    
    if algo_name == 'td3bc':
        config = TD3PlusBCConfig(
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            alpha=2.5,  # BC regularization weight
        )
    elif algo_name == 'cql':
        config = CQLConfig(
            actor_learning_rate=1e-4,
            critic_learning_rate=3e-4,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            alpha=5.0,  # CQL regularization
        )
    elif algo_name == 'iql':
        config = IQLConfig(
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            expectile=0.7,
            weight_temp=3.0,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}. Supported: td3bc, cql, iql")
    
    return config.create()


def train_offline(
    npz_path: str,
    algo_name: str,
    n_steps: int = 300000,
    seed: int = 0,
    save_path: Optional[str] = None,
    eval_interval: int = 10000,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train offline RL algorithm.
    
    Args:
        npz_path: Path to NPZ dataset
        algo_name: Algorithm name (td3bc, cql, iql)
        n_steps: Number of training steps
        seed: Random seed
        save_path: Path to save trained model
        eval_interval: Steps between evaluations
        verbose: Print progress
        
    Returns:
        Training metrics dictionary
    """
    if not HAS_D3RLPY:
        print("d3rlpy not available. Creating placeholder model.")
        return _create_placeholder_model(npz_path, algo_name, save_path, seed)
    
    # Set seeds
    np.random.seed(seed)
    d3rlpy.seed(seed)
    
    # Load and preprocess data
    if verbose:
        print(f"Loading dataset from: {npz_path}")
    data = load_dataset(npz_path)
    data = normalize_observations(data)
    data = normalize_actions(data)
    
    if verbose:
        print(f"  Observations: {data['observations'].shape}")
        print(f"  Actions: {data['actions'].shape}")
        print(f"  Rewards: min={data['rewards'].min():.2f}, max={data['rewards'].max():.2f}")
    
    # Create dataset
    dataset = create_mdp_dataset(data)
    
    # Create algorithm
    if verbose:
        print(f"Creating {algo_name.upper()} algorithm...")
    algo = create_algorithm(algo_name)
    
    # Build model
    algo.build_with_dataset(dataset)
    
    # Training loop
    if verbose:
        print(f"Training for {n_steps} steps...")
    
    metrics_history = []
    best_value = float('-inf')
    
    n_epochs = n_steps // len(dataset) + 1
    steps_per_epoch = len(dataset)
    
    for epoch in range(n_epochs):
        current_step = epoch * steps_per_epoch
        if current_step >= n_steps:
            break
        
        # Train one epoch
        results = algo.fit(
            dataset,
            n_steps=min(steps_per_epoch, n_steps - current_step),
            n_steps_per_epoch=steps_per_epoch,
            show_progress=False,
            save_interval=0,
        )
        
        # Log metrics
        if verbose and (epoch + 1) % (eval_interval // steps_per_epoch + 1) == 0:
            print(f"  Step {current_step + steps_per_epoch}: epoch completed")
    
    # Save model
    if save_path:
        save_dir = Path(save_path)
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        algo.save(str(save_dir))
        
        # Save metadata
        meta_path = save_dir.with_suffix('.meta.json')
        meta = {
            'algo': algo_name,
            'n_steps': n_steps,
            'seed': seed,
            'obs_dim': data['observations'].shape[1],
            'action_dim': data['actions'].shape[1],
            's_mean': data['s_mean'].tolist(),
            's_std': data['s_std'].tolist(),
            'a_min': data['a_min'].tolist(),
            'a_max': data['a_max'].tolist(),
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        if verbose:
            print(f"Saved model to: {save_path}")
    
    return {
        'algo': algo_name,
        'n_steps': n_steps,
        'seed': seed,
    }


def _create_placeholder_model(
    npz_path: str,
    algo_name: str,
    save_path: Optional[str],
    seed: int,
) -> Dict[str, Any]:
    """Create a placeholder model when d3rlpy is not available."""
    import torch
    import torch.nn as nn
    
    # Load data for dimensions
    data = load_dataset(npz_path)
    obs_dim = data['observations'].shape[1]
    action_dim = data['actions'].shape[1]
    
    # Simple MLP policy
    class PlaceholderPolicy(nn.Module):
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
    
    policy = PlaceholderPolicy(obs_dim, action_dim)
    
    # Train with simple BC
    data = normalize_observations(data)
    data = normalize_actions(data)
    
    obs = torch.FloatTensor(data['observations'])
    act = torch.FloatTensor(data['actions'])
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    for _ in range(100):
        pred = policy(obs)
        loss = nn.functional.mse_loss(pred, act)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Save
    if save_path:
        save_dir = Path(save_path)
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'policy_state': policy.state_dict(),
            'obs_dim': obs_dim,
            'action_dim': action_dim,
            's_mean': data['s_mean'].tolist(),
            's_std': data['s_std'].tolist(),
            'a_min': data['a_min'].tolist(),
            'a_max': data['a_max'].tolist(),
            'algo': algo_name,
        }, str(save_dir))
        
        print(f"Saved placeholder model to: {save_path}")
    
    return {'algo': algo_name, 'n_steps': 0, 'seed': seed, 'placeholder': True}


def main():
    parser = argparse.ArgumentParser(
        description='Train offline RL policy (TD3+BC, CQL, IQL)'
    )
    parser.add_argument(
        '--npz', type=str, required=True,
        help='Path to NPZ dataset file'
    )
    parser.add_argument(
        '--algo', type=str, required=True,
        choices=['td3bc', 'cql', 'iql'],
        help='Algorithm to train'
    )
    parser.add_argument(
        '--steps', type=int, default=300000,
        help='Number of training steps'
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed'
    )
    parser.add_argument(
        '--out', type=str, required=True,
        help='Output path for saved model'
    )
    parser.add_argument(
        '--eval-interval', type=int, default=10000,
        help='Steps between evaluations'
    )
    
    args = parser.parse_args()
    
    metrics = train_offline(
        npz_path=args.npz,
        algo_name=args.algo,
        n_steps=args.steps,
        seed=args.seed,
        save_path=args.out,
        eval_interval=args.eval_interval,
        verbose=True,
    )
    
    print(f"Training complete: {metrics}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
