"""
CEM Planner for Model Predictive Control
========================================

Cross-Entropy Method planner using learned dynamics model.

Usage:
    Integrated with run_eval.py as a policy wrapper.
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class CEMPlanner:
    """
    Cross-Entropy Method (CEM) planner for MPC.
    
    Uses learned dynamics model to plan optimal action sequences.
    """
    
    def __init__(
        self,
        dynamics_model: nn.Module,
        obs_dim: int,
        action_dim: int,
        horizon: int = 10,
        n_samples: int = 500,
        elite_frac: float = 0.1,
        n_iterations: int = 5,
        action_low: float = -1.0,
        action_high: float = 1.0,
        reward_fn: Optional[Callable] = None,
        device: str = 'cpu',
    ):
        """
        Initialize CEM planner.
        
        Args:
            dynamics_model: Learned dynamics model
            obs_dim: Observation dimension
            action_dim: Action dimension
            horizon: Planning horizon
            n_samples: Number of action sequences to sample
            elite_frac: Fraction of elite samples
            n_iterations: Number of CEM iterations
            action_low: Lower action bound
            action_high: Upper action bound
            reward_fn: Custom reward function (obs, action) -> reward
            device: Compute device
        """
        self.dynamics = dynamics_model
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_elite = max(1, int(n_samples * elite_frac))
        self.n_iterations = n_iterations
        self.action_low = action_low
        self.action_high = action_high
        self.reward_fn = reward_fn or self._default_reward
        self.device = torch.device(device)
        
        # Previous solution for warm-starting
        self._prev_mean: Optional[np.ndarray] = None
        self._prev_std: Optional[np.ndarray] = None
    
    def _default_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Default reward: negative L2 norm of observation (minimize deviation)."""
        return -np.sum(obs ** 2) - 0.01 * np.sum(action ** 2)
    
    def plan(
        self,
        current_obs: np.ndarray,
        state_dict: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Plan optimal action for current observation.
        
        Args:
            current_obs: Current normalized observation
            state_dict: Optional raw state for reward computation
            
        Returns:
            Optimal first action
        """
        # Initialize mean and std
        if self._prev_mean is not None:
            # Warm start: shift previous solution
            mean = np.zeros((self.horizon, self.action_dim))
            mean[:-1] = self._prev_mean[1:]
            mean[-1] = self._prev_mean[-1]
            std = self._prev_std.copy()
        else:
            mean = np.zeros((self.horizon, self.action_dim))
            std = np.ones((self.horizon, self.action_dim)) * 0.5
        
        # CEM iterations
        for iteration in range(self.n_iterations):
            # Sample action sequences
            # Shape: (n_samples, horizon, action_dim)
            noise = np.random.randn(self.n_samples, self.horizon, self.action_dim)
            action_sequences = mean + noise * std
            action_sequences = np.clip(action_sequences, self.action_low, self.action_high)
            
            # Evaluate each sequence
            returns = self._evaluate_sequences(current_obs, action_sequences)
            
            # Select elite samples
            elite_indices = np.argsort(returns)[-self.n_elite:]
            elite_sequences = action_sequences[elite_indices]
            
            # Update mean and std
            mean = np.mean(elite_sequences, axis=0)
            std = np.std(elite_sequences, axis=0) + 1e-5
        
        # Store for warm-starting
        self._prev_mean = mean
        self._prev_std = std
        
        # Return first action
        return mean[0]
    
    def _evaluate_sequences(
        self,
        initial_obs: np.ndarray,
        action_sequences: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate returns for action sequences.
        
        Args:
            initial_obs: Starting observation
            action_sequences: (n_samples, horizon, action_dim)
            
        Returns:
            returns: (n_samples,)
        """
        n_samples = action_sequences.shape[0]
        returns = np.zeros(n_samples)
        
        # Rollout each sequence
        with torch.no_grad():
            for i in range(n_samples):
                obs = initial_obs.copy()
                total_reward = 0.0
                gamma = 0.99
                
                for t in range(self.horizon):
                    action = action_sequences[i, t]
                    
                    # Get reward
                    reward = self.reward_fn(obs, action)
                    total_reward += (gamma ** t) * reward
                    
                    # Predict next obs using dynamics
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    action_t = torch.FloatTensor(action).unsqueeze(0).to(self.device)
                    
                    if hasattr(self.dynamics, 'predict_next'):
                        next_obs = self.dynamics.predict_next(obs_t, action_t)
                    else:
                        next_obs = self.dynamics(obs_t, action_t)
                    
                    obs = next_obs.cpu().numpy().squeeze()
                
                returns[i] = total_reward
        
        return returns
    
    def reset(self):
        """Reset planner state (call at episode start)."""
        self._prev_mean = None
        self._prev_std = None


class MPCPolicy:
    """
    MPC policy wrapper that uses CEM planner.
    """
    
    def __init__(
        self,
        dynamics_model_path: str,
        schema_path: Optional[str] = None,
        horizon: int = 10,
        n_samples: int = 500,
        elite_frac: float = 0.1,
        device: str = 'cpu',
    ):
        """
        Initialize MPC policy.
        
        Args:
            dynamics_model_path: Path to saved dynamics model
            schema_path: Path to schema YAML
            horizon: Planning horizon
            n_samples: CEM samples
            elite_frac: CEM elite fraction
            device: Compute device
        """
        self.device = device
        
        # Load dynamics model
        checkpoint = torch.load(dynamics_model_path, map_location='cpu')
        obs_dim = checkpoint['obs_dim']
        action_dim = checkpoint['action_dim']
        
        from hai_ml.mpc.dyn_model import DynamicsModel
        
        self.dynamics = DynamicsModel(obs_dim, action_dim)
        self.dynamics.load_state_dict(checkpoint['model_state'])
        self.dynamics.eval()
        self.dynamics.to(device)
        
        # Load schema for reward function
        self.schema = None
        self.track_targets = {}
        if schema_path:
            from ruamel.yaml import YAML
            yaml = YAML(typ='safe')
            with open(schema_path, 'r') as f:
                self.schema = yaml.load(f)
            self.track_targets = self.schema.get('reward', {}).get('track_targets', {})
        
        # Create planner
        self.planner = CEMPlanner(
            dynamics_model=self.dynamics,
            obs_dim=obs_dim,
            action_dim=action_dim,
            horizon=horizon,
            n_samples=n_samples,
            elite_frac=elite_frac,
            reward_fn=self._reward_fn,
            device=device,
        )
        
        # Store normalization if available
        self.s_mean = checkpoint.get('s_mean', np.zeros(obs_dim))
        self.s_std = checkpoint.get('s_std', np.ones(obs_dim))
    
    def _reward_fn(self, obs: np.ndarray, action: np.ndarray) -> float:
        """
        Reward function for MPC planning.
        
        Minimizes tracking error and action effort.
        """
        # Tracking error (assuming obs is z-scored)
        tracking_error = np.sum(obs ** 2)  # Minimize deviation from zero (setpoint)
        
        # Action effort
        action_effort = np.sum(action ** 2)
        
        # Action smoothness would require previous action
        
        reward = -tracking_error - 0.1 * action_effort
        return reward
    
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """
        Predict action for given observation.
        
        Args:
            obs: Normalized observation
            
        Returns:
            Action in [-1, 1]
        """
        return self.planner.plan(obs)
    
    def reset(self):
        """Reset planner state."""
        self.planner.reset()


class TunedPIDPolicy:
    """
    Simple tuned PID controller as baseline.
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        kp: float = 0.5,
        ki: float = 0.1,
        kd: float = 0.05,
    ):
        """
        Initialize PID controller.
        
        Assumes states 0:n_actions are controlled by corresponding actions.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self._integral = np.zeros(n_actions)
        self._prev_error = np.zeros(n_actions)
    
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute PID control action.
        
        Args:
            obs: Normalized observation (error from setpoint)
            
        Returns:
            Action in [-1, 1]
        """
        # Use first n_actions states as errors
        error = -obs[:self.n_actions]  # Negative because we want to reduce deviation
        
        # Pad if needed
        if len(error) < self.n_actions:
            error = np.pad(error, (0, self.n_actions - len(error)))
        
        # PID terms
        p_term = self.kp * error
        self._integral += error
        i_term = self.ki * self._integral
        d_term = self.kd * (error - self._prev_error)
        
        self._prev_error = error.copy()
        
        # Compute action
        action = p_term + i_term + d_term
        
        # Clip to [-1, 1]
        action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def reset(self):
        """Reset PID state."""
        self._integral = np.zeros(self.n_actions)
        self._prev_error = np.zeros(self.n_actions)
