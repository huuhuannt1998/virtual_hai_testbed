"""
Dataset Builder for HAI-ML
==========================

Converts HAI CSV logs into offline RL datasets (NPZ format).
Builds (s, a, r, s', done) tuples with z-score normalization.

Usage:
    python -m hai_ml.data.build_dataset \
        --csv data/p3_21_03.csv \
        --schema hai_ml/schemas/p3.yaml \
        --out hai_ml/data/p3_21_03.npz
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from ruamel.yaml import YAML


def load_schema(schema_path: str) -> dict:
    """Load and parse YAML schema file."""
    yaml = YAML(typ='safe')
    with open(schema_path, 'r') as f:
        return yaml.load(f)


def extract_tag_names(schema: dict) -> Tuple[List[str], List[str]]:
    """Extract state and action tag names from schema."""
    state_names = [tag['name'] for tag in schema['state_tags']]
    action_names = [tag['name'] for tag in schema['action_tags']]
    return state_names, action_names


def get_action_bounds(schema: dict) -> Dict[str, Tuple[float, float]]:
    """Get action bounds from schema."""
    bounds = {}
    for tag in schema['action_tags']:
        bounds[tag['name']] = (tag['min'], tag['max'])
    return bounds


def get_rate_limits(schema: dict) -> Dict[str, float]:
    """Get action rate limits from schema."""
    limits = {}
    for tag in schema['action_tags']:
        limits[tag['name']] = tag.get('rate_limit', float('inf'))
    return limits


def compute_reward(
    state: np.ndarray,
    next_state: np.ndarray,
    action: np.ndarray,
    prev_action: Optional[np.ndarray],
    schema: dict,
    state_names: List[str],
    action_names: List[str]
) -> float:
    """
    Compute reward based on schema reward configuration.
    
    reward = -w_track * tracking_error - w_wear * wear - w_energy * energy - penalty * violations
    """
    reward_cfg = schema['reward']
    track_targets = reward_cfg['track_targets']
    w_track = reward_cfg['w_track']
    w_wear = reward_cfg['w_wear']
    w_energy = reward_cfg['w_energy']
    penalty_violation = reward_cfg['penalty_violation']
    
    # Tracking error (MSE against targets)
    tracking_error = 0.0
    for tag, target in track_targets.items():
        if tag in state_names:
            idx = state_names.index(tag)
            tracking_error += (next_state[idx] - target) ** 2
    tracking_error = np.sqrt(tracking_error / len(track_targets)) if track_targets else 0.0
    
    # Wear proxy (L1 norm of action changes)
    if prev_action is not None:
        wear = np.sum(np.abs(action - prev_action))
    else:
        wear = 0.0
    
    # Energy proxy (L1 norm of actions - higher actions = more energy)
    energy = np.sum(np.abs(action)) / len(action) if len(action) > 0 else 0.0
    
    # Violation count (check alarm limits)
    violations = 0
    for tag_def in schema['state_tags']:
        tag_name = tag_def['name']
        if tag_name in state_names:
            idx = state_names.index(tag_name)
            val = next_state[idx]
            if val < tag_def.get('alarm_lo', float('-inf')):
                violations += 1
            if val > tag_def.get('alarm_hi', float('inf')):
                violations += 1
    
    # Compute total reward (negative because we minimize tracking error, wear, etc.)
    reward = (
        -w_track * tracking_error
        - w_wear * wear
        - w_energy * energy
        - penalty_violation * violations
    )
    
    return reward


def build_dataset(
    csv_path: str,
    schema: dict,
    rate_hz: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Build offline RL dataset from CSV log.
    
    Returns:
        Dict with keys: observations, actions, rewards, next_observations, terminals,
                       s_mean, s_std, a_mean, a_std, state_names, action_names
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Get tag names
    state_names, action_names = extract_tag_names(schema)
    
    # Validate columns exist (case-insensitive matching)
    df_cols_lower = {c.lower(): c for c in df.columns}
    
    def get_col(name: str) -> str:
        """Get actual column name from dataframe."""
        if name in df.columns:
            return name
        if name.lower() in df_cols_lower:
            return df_cols_lower[name.lower()]
        raise ValueError(f"Column '{name}' not found in CSV. Available: {list(df.columns)[:20]}...")
    
    # Extract state columns
    state_cols = []
    missing_state = []
    for name in state_names:
        try:
            state_cols.append(get_col(name))
        except ValueError:
            missing_state.append(name)
    
    if missing_state:
        print(f"Warning: Missing state columns: {missing_state}")
        # Remove missing from state_names
        state_names = [n for n in state_names if n not in missing_state]
        state_cols = [get_col(n) for n in state_names]
    
    # Extract action columns
    action_cols = []
    missing_action = []
    for name in action_names:
        try:
            action_cols.append(get_col(name))
        except ValueError:
            missing_action.append(name)
    
    if missing_action:
        print(f"Warning: Missing action columns: {missing_action}")
        action_names = [n for n in action_names if n not in missing_action]
        action_cols = [get_col(n) for n in action_names]
    
    # Extract data
    states = df[state_cols].values.astype(np.float32)
    actions = df[action_cols].values.astype(np.float32)
    
    # Handle NaN values
    states = np.nan_to_num(states, nan=0.0)
    actions = np.nan_to_num(actions, nan=0.0)
    
    n_samples = len(states) - 1  # -1 because we need next_state
    
    # Compute statistics for normalization
    s_mean = np.mean(states, axis=0)
    s_std = np.std(states, axis=0)
    s_std = np.where(s_std < 1e-6, 1.0, s_std)  # Avoid division by zero
    
    a_mean = np.mean(actions, axis=0)
    a_std = np.std(actions, axis=0)
    a_std = np.where(a_std < 1e-6, 1.0, a_std)
    
    # Build transition tuples
    observations = states[:-1]
    next_observations = states[1:]
    actions_out = actions[:-1]
    
    # Compute rewards
    rewards = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        prev_action = actions_out[i-1] if i > 0 else None
        rewards[i] = compute_reward(
            observations[i],
            next_observations[i],
            actions_out[i],
            prev_action,
            schema,
            state_names,
            action_names
        )
    
    # Terminals: mark end of each trajectory
    # For continuous logs, we mark as terminal every N steps or at actual boundaries
    terminals = np.zeros(n_samples, dtype=np.bool_)
    terminals[-1] = True  # Mark last as terminal
    
    # Also mark as terminal if there are large jumps (potential log boundaries)
    state_diff = np.abs(next_observations - observations)
    large_jumps = np.any(state_diff > 5 * s_std, axis=1)
    terminals = terminals | large_jumps
    
    # Compute wear proxy for each transition
    wear_proxy = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        if i > 0:
            wear_proxy[i] = np.sum(np.abs(actions_out[i] - actions_out[i-1]))
    
    print(f"Built dataset with {n_samples} transitions")
    print(f"  States: {len(state_names)} dims, Actions: {len(action_names)} dims")
    print(f"  Reward range: [{rewards.min():.2f}, {rewards.max():.2f}]")
    print(f"  Terminal rate: {terminals.sum() / n_samples * 100:.2f}%")
    
    return {
        'observations': observations,
        'actions': actions_out,
        'rewards': rewards,
        'next_observations': next_observations,
        'terminals': terminals,
        's_mean': s_mean,
        's_std': s_std,
        'a_mean': a_mean,
        'a_std': a_std,
        'state_names': np.array(state_names),
        'action_names': np.array(action_names),
        'wear_proxy': wear_proxy,
    }


def create_synthetic_dataset(
    schema: dict,
    n_episodes: int = 100,
    episode_length: int = 200,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Create a synthetic dataset for testing when no real CSV is available.
    Generates plausible trajectories based on schema definitions.
    """
    np.random.seed(seed)
    
    state_names, action_names = extract_tag_names(schema)
    n_states = len(state_names)
    n_actions = len(action_names)
    
    # Get nominal ranges from schema
    state_ranges = []
    for tag in schema['state_tags']:
        low = tag.get('nominal_min', 0.0)
        high = tag.get('nominal_max', 100.0)
        state_ranges.append((low, high))
    
    action_ranges = []
    for tag in schema['action_tags']:
        low = tag.get('min', 0.0)
        high = tag.get('max', 100.0)
        action_ranges.append((low, high))
    
    all_observations = []
    all_actions = []
    all_rewards = []
    all_next_observations = []
    all_terminals = []
    all_wear = []
    
    for ep in range(n_episodes):
        # Initialize state randomly within nominal ranges
        state = np.array([
            np.random.uniform(low, high) 
            for low, high in state_ranges
        ], dtype=np.float32)
        
        prev_action = None
        
        for t in range(episode_length):
            # Random action within bounds
            action = np.array([
                np.random.uniform(low, high) 
                for low, high in action_ranges
            ], dtype=np.float32)
            
            # Simple dynamics: state drifts based on action
            noise = np.random.randn(n_states).astype(np.float32) * 0.5
            next_state = state + noise
            
            # Influence of actions on state (simplified coupling)
            for i, aval in enumerate(action):
                # Actions affect corresponding states (modular)
                state_idx = i % n_states
                next_state[state_idx] += (aval - 50.0) * 0.01
            
            # Clip to reasonable ranges
            for i, (low, high) in enumerate(state_ranges):
                next_state[i] = np.clip(next_state[i], low * 0.5, high * 1.5)
            
            # Compute reward
            reward = compute_reward(
                state, next_state, action, prev_action,
                schema, state_names, action_names
            )
            
            # Wear
            wear = np.sum(np.abs(action - prev_action)) if prev_action is not None else 0.0
            
            # Store transition
            all_observations.append(state)
            all_actions.append(action)
            all_rewards.append(reward)
            all_next_observations.append(next_state)
            all_terminals.append(t == episode_length - 1)
            all_wear.append(wear)
            
            state = next_state
            prev_action = action
    
    observations = np.array(all_observations, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    rewards = np.array(all_rewards, dtype=np.float32)
    next_observations = np.array(all_next_observations, dtype=np.float32)
    terminals = np.array(all_terminals, dtype=np.bool_)
    wear_proxy = np.array(all_wear, dtype=np.float32)
    
    # Statistics
    s_mean = np.mean(observations, axis=0)
    s_std = np.std(observations, axis=0)
    s_std = np.where(s_std < 1e-6, 1.0, s_std)
    
    a_mean = np.mean(actions, axis=0)
    a_std = np.std(actions, axis=0)
    a_std = np.where(a_std < 1e-6, 1.0, a_std)
    
    print(f"Created synthetic dataset with {len(observations)} transitions")
    print(f"  {n_episodes} episodes × {episode_length} steps")
    
    return {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'next_observations': next_observations,
        'terminals': terminals,
        's_mean': s_mean,
        's_std': s_std,
        'a_mean': a_mean,
        'a_std': a_std,
        'state_names': np.array(state_names),
        'action_names': np.array(action_names),
        'wear_proxy': wear_proxy,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Build offline RL dataset from HAI CSV logs'
    )
    parser.add_argument(
        '--csv', type=str, default=None,
        help='Path to input CSV file (if not provided, generates synthetic data)'
    )
    parser.add_argument(
        '--schema', type=str, required=True,
        help='Path to YAML schema file'
    )
    parser.add_argument(
        '--out', type=str, required=True,
        help='Output path for NPZ file'
    )
    parser.add_argument(
        '--synthetic-episodes', type=int, default=100,
        help='Number of episodes for synthetic data'
    )
    parser.add_argument(
        '--synthetic-length', type=int, default=200,
        help='Episode length for synthetic data'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for synthetic data'
    )
    
    args = parser.parse_args()
    
    # Load schema
    schema = load_schema(args.schema)
    print(f"Loaded schema for process: {schema.get('process', 'unknown')}")
    
    # Build dataset
    if args.csv and Path(args.csv).exists():
        dataset = build_dataset(args.csv, schema)
    else:
        if args.csv:
            print(f"CSV file '{args.csv}' not found, generating synthetic data...")
        else:
            print("No CSV provided, generating synthetic data...")
        dataset = create_synthetic_dataset(
            schema,
            n_episodes=args.synthetic_episodes,
            episode_length=args.synthetic_length,
            seed=args.seed
        )
    
    # Create output directory
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as NPZ
    np.savez(
        args.out,
        **dataset
    )
    
    print(f"Saved dataset to: {args.out}")
    print(f"  File size: {out_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
