"""
Behavior Cloning Trainer (Track A Baseline)
============================================

Trains a simple BC policy by supervised learning on expert demonstrations.

Usage:
    python -m hai_ml.il.train_bc \
        --npz hai_ml/data/p3_21_03.npz \
        --epochs 30 \
        --seed 0 \
        --out runs/p3_bc_s0.pt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class BCPolicy(nn.Module):
    """
    Behavior Cloning Policy Network.
    
    Simple MLP that maps observations to actions.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = 'relu',
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Build MLP layers
        layers = []
        prev_dim = obs_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            prev_dim = h_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())  # Actions in [-1, 1]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass: obs -> action."""
        return self.network(obs)
    
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict action from numpy observation."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
            action = self.forward(obs_t)
            return action.cpu().numpy().squeeze()


class BCTrainer:
    """
    Behavior Cloning Trainer.
    
    Trains policy via supervised learning to minimize MSE between
    predicted and expert actions.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = 'auto',
    ):
        # Auto-detect device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.policy = BCPolicy(obs_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
    
    def train(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        epochs: int = 30,
        batch_size: int = 256,
        val_split: float = 0.1,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train BC policy.
        
        Args:
            observations: (N, obs_dim) array
            actions: (N, action_dim) array
            epochs: Number of training epochs
            batch_size: Batch size
            val_split: Fraction for validation
            verbose: Print progress
            
        Returns:
            Dictionary with 'train_loss' and 'val_loss' lists
        """
        # Normalize actions to [-1, 1] if needed
        action_min = actions.min(axis=0)
        action_max = actions.max(axis=0)
        action_range = action_max - action_min
        action_range = np.where(action_range < 1e-6, 1.0, action_range)
        
        # Store normalization params
        self.action_min = action_min
        self.action_max = action_max
        
        # Normalize to [-1, 1]
        actions_norm = 2.0 * (actions - action_min) / action_range - 1.0
        
        # Split data
        n_samples = len(observations)
        n_val = int(n_samples * val_split)
        indices = np.random.permutation(n_samples)
        
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        
        # Create datasets
        train_obs = torch.FloatTensor(observations[train_idx]).to(self.device)
        train_act = torch.FloatTensor(actions_norm[train_idx]).to(self.device)
        val_obs = torch.FloatTensor(observations[val_idx]).to(self.device)
        val_act = torch.FloatTensor(actions_norm[val_idx]).to(self.device)
        
        train_dataset = TensorDataset(train_obs, train_act)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        self.train_losses = []
        self.val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.policy.train()
            epoch_loss = 0.0
            n_batches = 0
            
            for obs_batch, act_batch in train_loader:
                self.optimizer.zero_grad()
                pred = self.policy(obs_batch)
                loss = self.criterion(pred, act_batch)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            train_loss = epoch_loss / n_batches
            self.train_losses.append(train_loss)
            
            # Validation
            self.policy.eval()
            with torch.no_grad():
                val_pred = self.policy(val_obs)
                val_loss = self.criterion(val_pred, val_act).item()
            self.val_losses.append(val_loss)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
        }
    
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """
        Predict action from observation.
        
        Returns action in original scale.
        """
        self.policy.eval()
        action_norm = self.policy.predict(obs)
        
        # Denormalize
        action = (action_norm + 1.0) * 0.5 * (self.action_max - self.action_min) + self.action_min
        return action
    
    def save(self, path: str):
        """Save model and normalization parameters."""
        save_dict = {
            'policy_state': self.policy.state_dict(),
            'obs_dim': self.policy.obs_dim,
            'action_dim': self.policy.action_dim,
            'action_min': self.action_min.tolist(),
            'action_max': self.action_max.tolist(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        torch.save(save_dict, path)
        print(f"Saved BC model to: {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'auto') -> 'BCTrainer':
        """Load model from file."""
        checkpoint = torch.load(path, map_location='cpu')
        
        trainer = cls(
            obs_dim=checkpoint['obs_dim'],
            action_dim=checkpoint['action_dim'],
            device=device,
        )
        
        trainer.policy.load_state_dict(checkpoint['policy_state'])
        trainer.action_min = np.array(checkpoint['action_min'])
        trainer.action_max = np.array(checkpoint['action_max'])
        trainer.train_losses = checkpoint.get('train_losses', [])
        trainer.val_losses = checkpoint.get('val_losses', [])
        
        return trainer


def normalize_observations(obs: np.ndarray, s_mean: np.ndarray, s_std: np.ndarray) -> np.ndarray:
    """Z-score normalize observations."""
    return (obs - s_mean) / s_std


def main():
    parser = argparse.ArgumentParser(
        description='Train Behavior Cloning policy (Track A baseline)'
    )
    parser.add_argument(
        '--npz', type=str, required=True,
        help='Path to NPZ dataset file'
    )
    parser.add_argument(
        '--epochs', type=int, default=30,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=256,
        help='Batch size'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--hidden', type=int, nargs='+', default=[256, 256],
        help='Hidden layer dimensions'
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
        '--device', type=str, default='auto',
        help='Device (auto/cpu/cuda)'
    )
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load dataset
    print(f"Loading dataset from: {args.npz}")
    data = np.load(args.npz, allow_pickle=True)
    
    observations = data['observations']
    actions = data['actions']
    s_mean = data['s_mean']
    s_std = data['s_std']
    
    print(f"  Observations: {observations.shape}")
    print(f"  Actions: {actions.shape}")
    
    # Normalize observations
    observations = normalize_observations(observations, s_mean, s_std)
    
    # Create trainer
    obs_dim = observations.shape[1]
    action_dim = actions.shape[1]
    
    trainer = BCTrainer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=args.hidden,
        lr=args.lr,
        device=args.device,
    )
    
    print(f"Training BC policy...")
    print(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")
    print(f"  Hidden: {args.hidden}")
    print(f"  Device: {trainer.device}")
    
    # Train
    metrics = trainer.train(
        observations=observations,
        actions=actions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=True,
    )
    
    # Save model
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(out_path))
    
    # Save metrics
    metrics_path = out_path.with_suffix('.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'final_train_loss': float(metrics['train_loss'][-1]),
            'final_val_loss': float(metrics['val_loss'][-1]),
            'epochs': args.epochs,
            'seed': args.seed,
        }, f, indent=2)
    
    print(f"Final train loss: {metrics['train_loss'][-1]:.6f}")
    print(f"Final val loss: {metrics['val_loss'][-1]:.6f}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
