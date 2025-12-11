"""
Learned Dynamics Model
======================

Trains an MLP to predict state transitions: Δs = f(s, a)

Usage:
    python -m hai_ml.mpc.dyn_model \
        --npz hai_ml/data/p3_21_03.npz \
        --epochs 20 \
        --out runs/p3_dyn.pt
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


class DynamicsModel(nn.Module):
    """
    Neural network dynamics model.
    
    Predicts state change: Δs = f(s, a)
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = 'relu',
        use_residual: bool = True,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.use_residual = use_residual
        
        # Input: [state, action]
        input_dim = obs_dim + action_dim
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'elu':
                layers.append(nn.ELU())
            else:
                layers.append(nn.ReLU())
            prev_dim = h_dim
        
        # Output: Δs (same dim as obs)
        layers.append(nn.Linear(prev_dim, obs_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Optional: output scaling
        self.output_scale = nn.Parameter(torch.ones(obs_dim))
    
    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: (s, a) -> Δs
        
        Args:
            obs: (batch, obs_dim)
            action: (batch, action_dim)
            
        Returns:
            delta_s: (batch, obs_dim)
        """
        x = torch.cat([obs, action], dim=-1)
        delta_s = self.network(x) * self.output_scale
        return delta_s
    
    def predict_next(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next state: s' = s + Δs"""
        delta_s = self.forward(obs, action)
        if self.use_residual:
            return obs + delta_s
        return delta_s
    
    def predict(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Predict next state from numpy inputs."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs)
            action_t = torch.FloatTensor(action)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
                action_t = action_t.unsqueeze(0)
            next_obs = self.predict_next(obs_t, action_t)
            return next_obs.cpu().numpy().squeeze()


class EnsembleDynamicsModel(nn.Module):
    """
    Ensemble of dynamics models for uncertainty estimation.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_models: int = 5,
        hidden_dims: List[int] = [256, 256],
    ):
        super().__init__()
        
        self.n_models = n_models
        self.models = nn.ModuleList([
            DynamicsModel(obs_dim, action_dim, hidden_dims)
            for _ in range(n_models)
        ])
    
    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ensemble.
        
        Returns (mean, std) of predictions.
        """
        predictions = torch.stack([
            m.predict_next(obs, action) for m in self.models
        ], dim=0)  # (n_models, batch, obs_dim)
        
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return mean, std
    
    def predict(self, obs: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty from numpy inputs."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs)
            action_t = torch.FloatTensor(action)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
                action_t = action_t.unsqueeze(0)
            mean, std = self.forward(obs_t, action_t)
            return mean.cpu().numpy().squeeze(), std.cpu().numpy().squeeze()


class DynamicsTrainer:
    """Trainer for dynamics model."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        n_ensemble: int = 1,
        device: str = 'auto',
    ):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        if n_ensemble > 1:
            self.model = EnsembleDynamicsModel(
                obs_dim, action_dim, n_ensemble, hidden_dims
            ).to(self.device)
        else:
            self.model = DynamicsModel(
                obs_dim, action_dim, hidden_dims
            ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
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
        next_observations: np.ndarray,
        epochs: int = 20,
        batch_size: int = 256,
        val_split: float = 0.1,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train dynamics model.
        
        Args:
            observations: (N, obs_dim)
            actions: (N, action_dim)
            next_observations: (N, obs_dim)
            epochs: Number of epochs
            batch_size: Batch size
            val_split: Validation fraction
            verbose: Print progress
            
        Returns:
            Training metrics
        """
        # Compute targets (delta or absolute)
        if isinstance(self.model, DynamicsModel) and self.model.use_residual:
            targets = next_observations - observations
        else:
            targets = next_observations
        
        # Split data
        n_samples = len(observations)
        n_val = int(n_samples * val_split)
        indices = np.random.permutation(n_samples)
        
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        
        # Create tensors
        train_obs = torch.FloatTensor(observations[train_idx]).to(self.device)
        train_act = torch.FloatTensor(actions[train_idx]).to(self.device)
        train_tgt = torch.FloatTensor(targets[train_idx]).to(self.device)
        
        val_obs = torch.FloatTensor(observations[val_idx]).to(self.device)
        val_act = torch.FloatTensor(actions[val_idx]).to(self.device)
        val_tgt = torch.FloatTensor(targets[val_idx]).to(self.device)
        
        train_dataset = TensorDataset(train_obs, train_act, train_tgt)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        self.train_losses = []
        self.val_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            
            for obs_batch, act_batch, tgt_batch in train_loader:
                self.optimizer.zero_grad()
                
                if isinstance(self.model, EnsembleDynamicsModel):
                    # Train each model in ensemble
                    loss = 0.0
                    for m in self.model.models:
                        pred = m(obs_batch, act_batch)
                        loss += self.criterion(pred, tgt_batch)
                    loss /= len(self.model.models)
                else:
                    pred = self.model(obs_batch, act_batch)
                    loss = self.criterion(pred, tgt_batch)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            train_loss = epoch_loss / n_batches
            self.train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                if isinstance(self.model, EnsembleDynamicsModel):
                    val_pred, _ = self.model(val_obs, val_act)
                    val_loss = self.criterion(val_pred, val_tgt + val_obs).item()
                else:
                    val_pred = self.model(val_obs, val_act)
                    val_loss = self.criterion(val_pred, val_tgt).item()
            
            self.val_losses.append(val_loss)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
        }
    
    def save(self, path: str):
        """Save model."""
        save_dict = {
            'model_state': self.model.state_dict(),
            'obs_dim': self.model.obs_dim if hasattr(self.model, 'obs_dim') else self.model.models[0].obs_dim,
            'action_dim': self.model.action_dim if hasattr(self.model, 'action_dim') else self.model.models[0].action_dim,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        torch.save(save_dict, path)
        print(f"Saved dynamics model to: {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'auto') -> 'DynamicsTrainer':
        """Load model from file."""
        checkpoint = torch.load(path, map_location='cpu')
        
        trainer = cls(
            obs_dim=checkpoint['obs_dim'],
            action_dim=checkpoint['action_dim'],
            device=device,
        )
        trainer.model.load_state_dict(checkpoint['model_state'])
        trainer.train_losses = checkpoint.get('train_losses', [])
        trainer.val_losses = checkpoint.get('val_losses', [])
        
        return trainer


def main():
    parser = argparse.ArgumentParser(
        description='Train dynamics model for MPC'
    )
    parser.add_argument(
        '--npz', type=str, required=True,
        help='Path to NPZ dataset file'
    )
    parser.add_argument(
        '--epochs', type=int, default=20,
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
        '--ensemble', type=int, default=1,
        help='Number of ensemble models'
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed'
    )
    parser.add_argument(
        '--out', type=str, required=True,
        help='Output path for saved model'
    )
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load dataset
    print(f"Loading dataset from: {args.npz}")
    data = np.load(args.npz, allow_pickle=True)
    
    observations = data['observations'].astype(np.float32)
    actions = data['actions'].astype(np.float32)
    next_observations = data['next_observations'].astype(np.float32)
    s_mean = data['s_mean']
    s_std = data['s_std']
    
    # Normalize
    observations = (observations - s_mean) / s_std
    next_observations = (next_observations - s_mean) / s_std
    
    # Normalize actions
    a_min = actions.min(axis=0)
    a_max = actions.max(axis=0)
    a_range = np.where(a_max - a_min < 1e-6, 1.0, a_max - a_min)
    actions = 2.0 * (actions - a_min) / a_range - 1.0
    
    print(f"  Observations: {observations.shape}")
    print(f"  Actions: {actions.shape}")
    
    # Create trainer
    obs_dim = observations.shape[1]
    action_dim = actions.shape[1]
    
    trainer = DynamicsTrainer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=args.hidden,
        lr=args.lr,
        n_ensemble=args.ensemble,
    )
    
    print(f"Training dynamics model...")
    
    metrics = trainer.train(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=True,
    )
    
    # Save
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
