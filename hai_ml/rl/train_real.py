"""
Train Offline RL Policies on Real HAI Data
===========================================

This script trains actual offline RL policies using d3rlpy on the HAI dataset.

Supported algorithms:
- BC (Behavioral Cloning) - baseline
- TD3+BC - Conservative policy learning
- CQL (Conservative Q-Learning) - OOD action penalty
- IQL (Implicit Q-Learning) - Expectile regression

Usage:
    python -m hai_ml.rl.train_real --process p3 --algo td3bc --epochs 100
    
Output:
    models/p3_td3bc_v21.03.d3
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import json
import numpy as np

# Check for d3rlpy
try:
    import d3rlpy
    from d3rlpy.algos import (
        BCConfig,
        TD3PlusBCConfig,
        CQLConfig,
        IQLConfig,
    )
    from d3rlpy.dataset import MDPDataset
    from d3rlpy.metrics import TDErrorEvaluator, DiscountedSumOfAdvantageEvaluator
    HAS_D3RLPY = True
except ImportError:
    HAS_D3RLPY = False
    print("WARNING: d3rlpy not installed. Install with: pip install d3rlpy")


def create_algorithm(algo_name: str, obs_shape: tuple, action_size: int, device: str = "cuda:0"):
    """Create d3rlpy algorithm instance with GPU support."""
    
    if algo_name == "bc":
        config = BCConfig(
            batch_size=512,
            learning_rate=3e-4,
        )
    elif algo_name == "td3bc":
        config = TD3PlusBCConfig(
            batch_size=512,
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            alpha=2.5,  # BC regularization weight
        )
    elif algo_name == "cql":
        config = CQLConfig(
            batch_size=512,
            actor_learning_rate=1e-4,
            critic_learning_rate=3e-4,
            conservative_weight=5.0,
        )
    elif algo_name == "iql":
        config = IQLConfig(
            batch_size=512,
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            expectile=0.7,
            weight_temp=3.0,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    return config.create(device=device)



def train_offline_rl(
    process: str = "p3",
    version: str = "21.03",
    algo: str = "td3bc",
    epochs: int = 100,
    eval_episodes: int = 10,
    save_dir: str = "models",
    data_root: str = "archive",
):
    """
    Train offline RL policy on real HAI data.
    
    Args:
        process: HAI process (p1, p2, p3, p4)
        version: HAI version
        algo: Algorithm name
        epochs: Training epochs
        eval_episodes: Episodes for OPE evaluation
        save_dir: Directory to save models
        data_root: Path to HAI data
    """
    if not HAS_D3RLPY:
        print("ERROR: d3rlpy required for training")
        return None
    
    print("=" * 60)
    print(f"Training {algo.upper()} on HAI-{version} Process {process.upper()}")
    print("=" * 60)
    
    # Load data
    from hai_ml.data.hai_loader import load_hai_for_offline_rl
    
    print("\n[1/5] Loading HAI dataset...")
    train_data, test_data = load_hai_for_offline_rl(
        process=process,
        version=version,
        data_root=data_root,
    )
    
    n_train = train_data['observations'].shape[0]
    n_test = test_data['observations'].shape[0]
    obs_dim = train_data['observations'].shape[1]
    act_dim = train_data['actions'].shape[1]
    
    print(f"  Train: {n_train:,} transitions")
    print(f"  Test: {n_test:,} transitions")
    print(f"  Obs dim: {obs_dim}, Action dim: {act_dim}")
    
    # Create d3rlpy dataset
    print("\n[2/5] Creating MDPDataset...")
    dataset = MDPDataset(
        observations=train_data['observations'],
        actions=train_data['actions'],
        rewards=train_data['rewards'],
        terminals=train_data['terminals'],
    )
    
    # Count episodes
    n_episodes = len(list(dataset.episodes))
    print(f"  Episodes: {n_episodes}")
    
    # Check GPU
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n[3/5] Creating {algo.upper()} algorithm on {device}...")
    
    algorithm = create_algorithm(
        algo,
        obs_shape=(obs_dim,),
        action_size=act_dim,
        device=device,
    )
    
    # Train
    print(f"\n[4/5] Training for {epochs} epochs...")
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    model_name = f"{process}_{algo}_v{version.replace('.', '')}"

    # Training with progress (d3rlpy 2.x API)
    from d3rlpy.logging import FileAdapterFactory
    
    algorithm.fit(
        dataset,
        n_steps=epochs * 1000,  # Steps per epoch
        n_steps_per_epoch=1000,
        experiment_name=model_name,
        with_timestamp=False,
        logger_adapter=FileAdapterFactory(root_dir=str(save_path / "logs")),
        save_interval=10,
    )
    
    # Save final model
    model_path = save_path / f"{model_name}.d3"
    algorithm.save(str(model_path))
    print(f"\n[5/5] Saved model to {model_path}")
    
    # Save training metadata
    metadata = {
        "process": process,
        "version": version,
        "algorithm": algo,
        "epochs": epochs,
        "train_transitions": n_train,
        "obs_dim": obs_dim,
        "action_dim": act_dim,
        "n_episodes": n_episodes,
        "trained_at": datetime.now().isoformat(),
    }
    
    with open(save_path / f"{model_name}_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\nTraining complete!")
    return algorithm


def main():
    parser = argparse.ArgumentParser(description="Train offline RL on HAI data")
    parser.add_argument("--process", default="p3", choices=["p1", "p2", "p3", "p4"])
    parser.add_argument("--version", default="21.03")
    parser.add_argument("--algo", default="td3bc", choices=["bc", "td3bc", "cql", "iql"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save-dir", default="models")
    parser.add_argument("--data-root", default="archive")
    
    args = parser.parse_args()
    
    train_offline_rl(
        process=args.process,
        version=args.version,
        algo=args.algo,
        epochs=args.epochs,
        save_dir=args.save_dir,
        data_root=args.data_root,
    )


if __name__ == "__main__":
    main()
