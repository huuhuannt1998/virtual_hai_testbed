"""
HAI Gymnasium Environment Wrapper
=================================

Wraps the HAI testbed simulator as a Gymnasium environment for RL training
and evaluation. Supports both real PLC and simulation modes.

Usage:
    from hai_ml.envs.hai_gym import HaiEnv
    env = HaiEnv(schema_path='hai_ml/schemas/p3.yaml', mode='sim')
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ruamel.yaml import YAML


class HaiSim:
    """
    Minimal simulator interface for HAI processes.
    
    This can be replaced with the actual HAI plant simulator or PLC connection.
    """
    
    def __init__(
        self,
        state_names: List[str],
        action_names: List[str],
        state_ranges: List[Tuple[float, float]],
        action_ranges: List[Tuple[float, float]],
        rate_limits: Dict[str, float],
        dt: float = 1.0,
        seed: int = 42
    ):
        self.state_names = state_names
        self.action_names = action_names
        self.state_ranges = state_ranges
        self.action_ranges = action_ranges
        self.rate_limits = rate_limits
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        
        self.n_states = len(state_names)
        self.n_actions = len(action_names)
        
        # Current state
        self._state: Dict[str, float] = {}
        self._prev_action: Optional[np.ndarray] = None
        
        # Simple dynamics model coefficients (learned or configured)
        self._A = np.eye(self.n_states) * 0.99  # State transition
        self._B = np.zeros((self.n_states, self.n_actions))  # Action influence
        
        # Set up simple action-to-state coupling
        for i in range(min(self.n_actions, self.n_states)):
            self._B[i, i] = 0.1
        
        # Process noise
        self._noise_std = 0.5
    
    def reset(self, seed: Optional[int] = None) -> Dict[str, float]:
        """Reset simulator to initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Initialize to middle of nominal range with some noise
        self._state = {}
        for i, name in enumerate(self.state_names):
            low, high = self.state_ranges[i]
            mid = (low + high) / 2
            noise = self.rng.normal(0, (high - low) * 0.05)
            self._state[name] = np.clip(mid + noise, low, high)
        
        self._prev_action = None
        return self._state.copy()
    
    def observe(self) -> Dict[str, float]:
        """Get current state observation."""
        return self._state.copy()
    
    def step(self, action_dict: Dict[str, float]) -> Tuple[Dict[str, float], float, bool, Dict[str, Any]]:
        """
        Execute one step with the given action.
        
        Args:
            action_dict: Dictionary mapping action tag names to values
            
        Returns:
            (next_state_dict, reward, done, info)
        """
        # Convert action dict to array
        action = np.array([
            action_dict.get(name, (self.action_ranges[i][0] + self.action_ranges[i][1]) / 2)
            for i, name in enumerate(self.action_names)
        ])
        
        # Apply rate limits
        if self._prev_action is not None:
            for i, name in enumerate(self.action_names):
                limit = self.rate_limits.get(name, float('inf'))
                delta = action[i] - self._prev_action[i]
                if abs(delta) > limit * self.dt:
                    action[i] = self._prev_action[i] + np.sign(delta) * limit * self.dt
        
        # Clip actions to bounds
        for i, (low, high) in enumerate(self.action_ranges):
            action[i] = np.clip(action[i], low, high)
        
        # Get current state as array
        state = np.array([self._state[name] for name in self.state_names])
        
        # Simple dynamics: x_{t+1} = A*x_t + B*u_t + noise
        noise = self.rng.normal(0, self._noise_std, self.n_states)
        next_state = self._A @ state + self._B @ (action - 50.0) + noise
        
        # Clip to extended ranges (allow some violation)
        for i, (low, high) in enumerate(self.state_ranges):
            next_state[i] = np.clip(next_state[i], low * 0.5, high * 1.5)
        
        # Update state dict
        for i, name in enumerate(self.state_names):
            self._state[name] = float(next_state[i])
        
        self._prev_action = action.copy()
        
        # Compute reward (simplified)
        reward = -np.mean(np.abs(next_state - state)) * 0.1
        
        # Check termination (severe violations)
        done = False
        for i, name in enumerate(self.state_names):
            low, high = self.state_ranges[i]
            if self._state[name] < low * 0.3 or self._state[name] > high * 1.7:
                done = True
                reward -= 10.0
                break
        
        info = {
            'action_applied': action.tolist(),
            'violations': 0,
        }
        
        return self._state.copy(), reward, done, info


class HaiEnv(gym.Env):
    """
    Gymnasium environment wrapper for HAI testbed.
    
    Supports:
    - Simulation mode (built-in dynamics)
    - PLC mode (real Siemens S7-1200)
    - External simulator (user-provided HaiSim instance)
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(
        self,
        schema_path: str,
        mode: str = 'sim',
        plc_ip: Optional[str] = None,
        plc_rack: int = 0,
        plc_slot: int = 0,
        plc_db: int = 2,
        external_sim: Optional[HaiSim] = None,
        normalize: bool = True,
        s_mean: Optional[np.ndarray] = None,
        s_std: Optional[np.ndarray] = None,
        max_steps: int = 1000,
        seed: int = 42,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize HAI environment.
        
        Args:
            schema_path: Path to YAML schema file
            mode: 'sim' for simulation, 'plc' for real PLC, 'external' for user sim
            plc_ip: PLC IP address (for mode='plc')
            plc_rack: PLC rack number
            plc_slot: PLC slot number
            plc_db: PLC data block number
            external_sim: External HaiSim instance (for mode='external')
            normalize: Whether to z-score normalize observations
            s_mean: State means for normalization
            s_std: State stds for normalization
            max_steps: Maximum steps per episode
            seed: Random seed
            render_mode: Rendering mode ('human' or 'ansi')
        """
        super().__init__()
        
        self.schema_path = schema_path
        self.mode = mode
        self.normalize = normalize
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Load schema
        yaml = YAML(typ='safe')
        with open(schema_path, 'r') as f:
            self.schema = yaml.load(f)
        
        # Extract configuration
        self.process = self.schema.get('process', 'unknown')
        self.rate_hz = self.schema.get('rate_hz', 1.0)
        self.dt = 1.0 / self.rate_hz
        
        # State space
        self.state_names = [tag['name'] for tag in self.schema['state_tags']]
        self.state_ranges = [
            (tag.get('nominal_min', 0), tag.get('nominal_max', 100))
            for tag in self.schema['state_tags']
        ]
        self.n_states = len(self.state_names)
        
        # Action space
        self.action_names = [tag['name'] for tag in self.schema['action_tags']]
        self.action_ranges = [
            (tag.get('min', 0), tag.get('max', 100))
            for tag in self.schema['action_tags']
        ]
        self.rate_limits = {
            tag['name']: tag.get('rate_limit', float('inf'))
            for tag in self.schema['action_tags']
        }
        self.n_actions = len(self.action_names)
        
        # Normalization stats
        if s_mean is not None:
            self.s_mean = np.array(s_mean, dtype=np.float32)
        else:
            self.s_mean = np.array([
                (low + high) / 2 for low, high in self.state_ranges
            ], dtype=np.float32)
        
        if s_std is not None:
            self.s_std = np.array(s_std, dtype=np.float32)
        else:
            self.s_std = np.array([
                (high - low) / 4 for low, high in self.state_ranges
            ], dtype=np.float32)
        self.s_std = np.where(self.s_std < 1e-6, 1.0, self.s_std)
        
        # Gymnasium spaces
        if normalize:
            # Normalized observations in roughly [-3, 3]
            obs_low = -5.0 * np.ones(self.n_states, dtype=np.float32)
            obs_high = 5.0 * np.ones(self.n_states, dtype=np.float32)
        else:
            obs_low = np.array([low for low, _ in self.state_ranges], dtype=np.float32)
            obs_high = np.array([high for _, high in self.state_ranges], dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )
        
        # Action space: normalized to [-1, 1] for RL algorithms
        self.action_space = spaces.Box(
            low=-1.0 * np.ones(self.n_actions, dtype=np.float32),
            high=1.0 * np.ones(self.n_actions, dtype=np.float32),
            dtype=np.float32
        )
        
        # Initialize simulator
        self.sim: Optional[HaiSim] = None
        self.plc_client = None
        
        if mode == 'sim':
            self.sim = HaiSim(
                state_names=self.state_names,
                action_names=self.action_names,
                state_ranges=self.state_ranges,
                action_ranges=self.action_ranges,
                rate_limits=self.rate_limits,
                dt=self.dt,
                seed=seed
            )
        elif mode == 'plc':
            self._init_plc(plc_ip, plc_rack, plc_slot, plc_db)
        elif mode == 'external':
            self.sim = external_sim
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Episode state
        self._step_count = 0
        self._current_obs: Optional[np.ndarray] = None
        self._prev_action: Optional[np.ndarray] = None
        
        # Reward configuration
        self.reward_cfg = self.schema.get('reward', {})
        self.track_targets = self.reward_cfg.get('track_targets', {})
        
        # RNG
        self._np_random: Optional[np.random.Generator] = None
        self.seed_value = seed
    
    def _init_plc(self, ip: str, rack: int, slot: int, db: int):
        """Initialize PLC connection."""
        try:
            import snap7
            self.plc_client = snap7.client.Client()
            self.plc_client.connect(ip, rack, slot)
            self.plc_db = db
            print(f"Connected to PLC at {ip}")
        except Exception as e:
            print(f"Failed to connect to PLC: {e}")
            print("Falling back to simulation mode")
            self.mode = 'sim'
            self.sim = HaiSim(
                state_names=self.state_names,
                action_names=self.action_names,
                state_ranges=self.state_ranges,
                action_ranges=self.action_ranges,
                rate_limits=self.rate_limits,
                dt=self.dt,
                seed=self.seed_value
            )
    
    def _denormalize_action(self, action: np.ndarray) -> Dict[str, float]:
        """Convert normalized [-1, 1] action to actual values."""
        action_dict = {}
        for i, name in enumerate(self.action_names):
            low, high = self.action_ranges[i]
            # Map [-1, 1] to [low, high]
            value = low + (action[i] + 1.0) * 0.5 * (high - low)
            action_dict[name] = float(np.clip(value, low, high))
        return action_dict
    
    def _normalize_obs(self, state_dict: Dict[str, float]) -> np.ndarray:
        """Convert state dict to normalized observation array."""
        state = np.array([
            state_dict.get(name, self.s_mean[i])
            for i, name in enumerate(self.state_names)
        ], dtype=np.float32)
        
        if self.normalize:
            state = (state - self.s_mean) / self.s_std
        
        return state
    
    def _compute_reward(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        info: Dict[str, Any]
    ) -> float:
        """Compute reward based on schema configuration."""
        w_track = self.reward_cfg.get('w_track', 1.0)
        w_wear = self.reward_cfg.get('w_wear', 0.1)
        w_energy = self.reward_cfg.get('w_energy', 0.05)
        penalty_violation = self.reward_cfg.get('penalty_violation', 10.0)
        
        # De-normalize for target comparison if needed
        if self.normalize:
            state = next_obs * self.s_std + self.s_mean
        else:
            state = next_obs
        
        # Tracking error
        tracking_error = 0.0
        for tag, target in self.track_targets.items():
            if tag in self.state_names:
                idx = self.state_names.index(tag)
                tracking_error += (state[idx] - target) ** 2
        tracking_error = np.sqrt(tracking_error / max(len(self.track_targets), 1))
        
        # Wear (action change magnitude)
        if self._prev_action is not None:
            wear = np.sum(np.abs(action - self._prev_action))
        else:
            wear = 0.0
        
        # Energy (action magnitude)
        energy = np.mean(np.abs(action))
        
        # Violations from info
        violations = info.get('violations', 0)
        
        reward = (
            -w_track * tracking_error
            - w_wear * wear
            - w_energy * energy
            - penalty_violation * violations
        )
        
        return float(reward)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
            self.seed_value = seed
        
        self._step_count = 0
        self._prev_action = None
        
        # Reset simulator
        if self.sim is not None:
            state_dict = self.sim.reset(seed=seed)
        else:
            # PLC mode - just observe current state
            state_dict = self._read_plc_state()
        
        self._current_obs = self._normalize_obs(state_dict)
        
        info = {
            'step': 0,
            'raw_state': state_dict,
        }
        
        return self._current_obs.copy(), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        self._step_count += 1
        
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Convert to actual values
        action_dict = self._denormalize_action(action)
        
        # Execute step
        if self.sim is not None:
            next_state_dict, sim_reward, done, info = self.sim.step(action_dict)
        else:
            next_state_dict, sim_reward, done, info = self._step_plc(action_dict)
        
        # Get observation
        next_obs = self._normalize_obs(next_state_dict)
        
        # Compute reward
        reward = self._compute_reward(self._current_obs, next_obs, action, info)
        
        # Check truncation
        truncated = self._step_count >= self.max_steps
        terminated = done
        
        # Update state
        self._current_obs = next_obs
        self._prev_action = action.copy()
        
        info.update({
            'step': self._step_count,
            'raw_state': next_state_dict,
            'action_dict': action_dict,
        })
        
        return next_obs.copy(), reward, terminated, truncated, info
    
    def _read_plc_state(self) -> Dict[str, float]:
        """Read current state from PLC."""
        # Placeholder - implement actual PLC reading
        # This would use snap7 to read from DB
        return {name: 50.0 for name in self.state_names}
    
    def _step_plc(self, action_dict: Dict[str, float]) -> Tuple[Dict[str, float], float, bool, Dict[str, Any]]:
        """Execute step on real PLC."""
        # Placeholder - implement actual PLC write/read
        # This would use snap7 to write actions and read next state
        import time
        time.sleep(self.dt)
        next_state = self._read_plc_state()
        return next_state, 0.0, False, {'violations': 0}
    
    def render(self):
        """Render current state."""
        if self.render_mode == 'ansi':
            lines = [f"HAI Environment - {self.process}"]
            lines.append(f"Step: {self._step_count}")
            if self._current_obs is not None:
                lines.append("States:")
                for i, name in enumerate(self.state_names[:5]):  # Show first 5
                    val = self._current_obs[i]
                    if self.normalize:
                        val = val * self.s_std[i] + self.s_mean[i]
                    lines.append(f"  {name}: {val:.2f}")
            return '\n'.join(lines)
        return None
    
    def close(self):
        """Clean up resources."""
        if self.plc_client is not None:
            try:
                self.plc_client.disconnect()
            except:
                pass
    
    def get_state_names(self) -> List[str]:
        """Get list of state tag names."""
        return self.state_names.copy()
    
    def get_action_names(self) -> List[str]:
        """Get list of action tag names."""
        return self.action_names.copy()


def make_env(
    task: str,
    mode: str = 'sim',
    **kwargs
) -> HaiEnv:
    """
    Factory function to create HAI environment.
    
    Args:
        task: Task name ('p1', 'p3', 'p12')
        mode: 'sim' or 'plc'
        **kwargs: Additional arguments for HaiEnv
        
    Returns:
        HaiEnv instance
    """
    schema_paths = {
        'p1': 'hai_ml/schemas/p1.yaml',
        'p3': 'hai_ml/schemas/p3.yaml',
        'p12': 'hai_ml/schemas/p12.yaml',
    }
    
    if task not in schema_paths:
        raise ValueError(f"Unknown task: {task}. Available: {list(schema_paths.keys())}")
    
    schema_path = schema_paths[task]
    
    # Try to find schema relative to this file
    base_path = Path(__file__).parent.parent
    full_path = base_path / 'schemas' / f'{task}.yaml'
    
    if full_path.exists():
        schema_path = str(full_path)
    
    return HaiEnv(schema_path=schema_path, mode=mode, **kwargs)
