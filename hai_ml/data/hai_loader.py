"""
HAI Dataset Loader
==================

Loads and preprocesses the real HAI dataset for offline RL training.

The HAI dataset contains:
- P1: Boiler process
- P2: Turbine process  
- P3: Water treatment (our focus)
- P4: Auxiliary systems

Data structure:
- train*.csv: Normal operation (no attacks)
- test*.csv: Contains attack periods (attack=1)

References:
- HAI Dataset: https://github.com/icsdataset/hai
- Technical Details: archive/hai_dataset_technical_details.pdf
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class HAIDataConfig:
    """Configuration for HAI data loading."""
    # Process selection
    process: str = "p3"  # p1, p2, p3, p4, or "all"
    
    # Data version
    version: str = "21.03"
    
    # Paths
    data_root: str = "archive"
    
    # Preprocessing
    normalize: bool = True
    remove_constant: bool = True
    
    # For RL dataset
    window_size: int = 10  # History window for state
    action_cols: Optional[List[str]] = None  # If None, infer from setpoints
    
    # Train/test split
    test_ratio: float = 0.2
    

# Column mappings for each process
PROCESS_COLUMNS = {
    "p1": {
        "sensors": [
            "P1_FT01", "P1_FT02", "P1_FT03",  # Flow transmitters
            "P1_LIT01",  # Level
            "P1_PIT01", "P1_PIT02",  # Pressure
            "P1_TIT01", "P1_TIT02",  # Temperature
        ],
        "actuators": [
            "P1_FCV01D", "P1_FCV02D", "P1_FCV03D",  # Flow control valves
            "P1_LCV01D",  # Level control valve
            "P1_PCV01D", "P1_PCV02D",  # Pressure control valves
        ],
        "setpoints": [
            "P1_FCV01Z", "P1_FCV02Z", "P1_FCV03Z",
            "P1_LCV01Z",
            "P1_PCV01Z", "P1_PCV02Z",
        ],
    },
    "p2": {
        "sensors": [
            "P2_SIT01", "P2_SIT02",  # Speed/vibration
            "P2_VT01",  # Vibration
            "P2_VTR01", "P2_VTR02", "P2_VTR03", "P2_VTR04",
        ],
        "actuators": [
            "P2_AutoGO", "P2_ManualGO",
            "P2_OnOff",
        ],
        "setpoints": [
            "P2_CO_rpm",
        ],
    },
    "p3": {
        "sensors": [
            "P3_FIT01",  # Flow indicator transmitter
            "P3_LIT01",  # Level indicator transmitter
            "P3_PIT01",  # Pressure indicator transmitter
            "P3_LH",     # Level high
            "P3_LL",     # Level low
        ],
        "actuators": [
            "P3_LCV01D",  # Level control valve (demand)
            "P3_LCP01D",  # Level control pump
        ],
        "setpoints": [],  # P3 uses implicit setpoints
    },
    "p4": {
        "sensors": [
            "P4_ST_PT01",  # Steam pressure
            "P4_ST_TT01",  # Steam temperature
        ],
        "actuators": [
            "P4_HT_FD", "P4_HT_LD", "P4_HT_PO", "P4_HT_PS",
            "P4_ST_FD", "P4_ST_GOV", "P4_ST_LD", "P4_ST_PO", "P4_ST_PS",
        ],
        "setpoints": [],
    },
}


class HAIDataLoader:
    """
    Load and preprocess HAI dataset for offline RL.
    
    Usage:
        loader = HAIDataLoader(HAIDataConfig(process="p3", version="21.03"))
        train_data, test_data = loader.load()
        
        # For d3rlpy
        dataset = loader.to_d3rlpy_dataset(train_data)
    """
    
    def __init__(self, config: HAIDataConfig):
        self.config = config
        self.data_path = Path(config.data_root) / f"hai-{config.version}"
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"HAI data not found at {self.data_path}")
        
        # Get column definitions
        self.process_cols = PROCESS_COLUMNS.get(config.process.lower(), {})
        
        # Statistics for normalization
        self.mean: Optional[pd.Series] = None
        self.std: Optional[pd.Series] = None
        
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load all train and test data.
        
        Returns:
            train_df: Normal operation data
            test_df: Data with attacks
        """
        train_dfs = []
        test_dfs = []
        
        for f in sorted(os.listdir(self.data_path)):
            if not f.endswith('.csv'):
                continue
            
            filepath = self.data_path / f
            print(f"Loading {filepath}...")
            
            df = pd.read_csv(filepath)
            
            if 'train' in f:
                train_dfs.append(df)
            elif 'test' in f:
                test_dfs.append(df)
        
        train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
        test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
        
        print(f"Loaded {len(train_df):,} train rows, {len(test_df):,} test rows")
        
        return train_df, test_df
    
    def get_process_columns(self) -> Tuple[List[str], List[str]]:
        """Get sensor and actuator columns for the configured process."""
        sensors = self.process_cols.get("sensors", [])
        actuators = self.process_cols.get("actuators", [])
        return sensors, actuators
    
    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Preprocess data: select columns, normalize, handle missing values.
        
        Args:
            df: Raw dataframe
            fit: If True, compute normalization stats (use for train data)
            
        Returns:
            Preprocessed dataframe
        """
        # Select relevant columns
        sensors, actuators = self.get_process_columns()
        cols = sensors + actuators
        
        # Add attack labels if present
        label_cols = [c for c in ['attack', f'attack_{self.config.process.upper()}'] if c in df.columns]
        
        # Filter to existing columns
        existing_cols = [c for c in cols if c in df.columns]
        if not existing_cols:
            raise ValueError(f"No columns found for process {self.config.process}")
        
        result = df[existing_cols + label_cols].copy()
        
        # Handle missing/infinite values
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.ffill().bfill().fillna(0)
        
        # Remove constant columns
        if self.config.remove_constant:
            non_const = result.std() > 1e-6
            result = result.loc[:, non_const]
        
        # Normalize
        if self.config.normalize:
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            numeric_cols = [c for c in numeric_cols if c not in label_cols]
            
            if fit:
                self.mean = result[numeric_cols].mean()
                self.std = result[numeric_cols].std().replace(0, 1)
            
            if self.mean is not None and self.std is not None:
                result[numeric_cols] = (result[numeric_cols] - self.mean) / self.std
        
        return result
    
    def to_rl_dataset(
        self,
        df: pd.DataFrame,
        reward_col: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Convert preprocessed data to offline RL format.
        
        For HAI, we infer:
        - States: sensor readings (with history window)
        - Actions: actuator commands (changes from previous step)
        - Rewards: negative of control error or custom reward
        
        Returns:
            Dictionary with keys: observations, actions, rewards, terminals
        """
        sensors, actuators = self.get_process_columns()
        
        # Filter to existing columns
        sensor_cols = [c for c in sensors if c in df.columns]
        actuator_cols = [c for c in actuators if c in df.columns]
        
        if not sensor_cols:
            # Fallback: use all numeric non-actuator columns
            sensor_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                          if c not in actuator_cols and 'attack' not in c.lower()]
        
        print(f"Using {len(sensor_cols)} sensor columns, {len(actuator_cols)} actuator columns")
        
        # Extract arrays
        observations = df[sensor_cols].values.astype(np.float32)
        
        if actuator_cols:
            actions = df[actuator_cols].values.astype(np.float32)
            # Compute action deltas (what the controller changed)
            action_deltas = np.diff(actions, axis=0, prepend=actions[:1])
            
            # Ensure at least 2D actions for continuous action space
            # d3rlpy treats 1D actions as discrete, so add small noise if needed
            if action_deltas.shape[1] == 1:
                # Add a dummy action dimension with small random noise
                # This ensures d3rlpy detects continuous action space
                np.random.seed(42)
                dummy_action = np.random.randn(len(action_deltas), 1).astype(np.float32) * 0.001
                action_deltas = np.hstack([action_deltas, dummy_action])
        else:
            # No explicit actuators - use 2D small random actions (minimum for continuous)
            np.random.seed(42)
            action_deltas = np.random.randn(len(df), 2).astype(np.float32) * 0.001
        
        # Compute rewards (negative of deviation from mean = "tracking setpoint")
        # For a real paper, you'd define task-specific rewards
        if reward_col and reward_col in df.columns:
            rewards = df[reward_col].values.astype(np.float32)
        else:
            # Proxy reward: negative L2 distance from normalized origin
            # (i.e., staying near normal operating point is good)
            rewards = -np.sum(observations ** 2, axis=1) / observations.shape[1]
            rewards = rewards.astype(np.float32)
        
        # Terminals: end of each trajectory
        terminals = np.zeros(len(df), dtype=bool)
        terminals[-1] = True  # Last step is terminal
        
        # If attacks are present, mark attack onset as "bad terminal"
        if 'attack' in df.columns:
            attack_starts = (df['attack'].diff() == 1).values
            terminals = terminals | attack_starts
        
        return {
            'observations': observations,
            'actions': action_deltas,
            'rewards': rewards,
            'terminals': terminals,
        }
    
    def to_d3rlpy_dataset(self, rl_data: Dict[str, np.ndarray]):
        """
        Convert to d3rlpy MDPDataset format.
        
        Returns:
            d3rlpy.dataset.MDPDataset
        """
        try:
            from d3rlpy.dataset import MDPDataset
        except ImportError:
            raise ImportError("d3rlpy is required. Install with: pip install d3rlpy")
        
        dataset = MDPDataset(
            observations=rl_data['observations'],
            actions=rl_data['actions'],
            rewards=rl_data['rewards'],
            terminals=rl_data['terminals'],
        )
        
        print(f"Created MDPDataset with {len(dataset)} transitions")
        print(f"  Observation shape: {dataset.observations.shape}")
        print(f"  Action shape: {dataset.actions.shape}")
        
        return dataset


def load_hai_for_offline_rl(
    process: str = "p3",
    version: str = "21.03",
    data_root: str = "archive",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Convenience function to load HAI data for offline RL.
    
    Args:
        process: Which process (p1, p2, p3, p4)
        version: HAI version (21.03, 22.04)
        data_root: Path to archive folder
        
    Returns:
        train_data, test_data: Dictionaries with observations, actions, rewards, terminals
    """
    config = HAIDataConfig(
        process=process,
        version=version,
        data_root=data_root,
    )
    
    loader = HAIDataLoader(config)
    train_df, test_df = loader.load()
    
    train_df = loader.preprocess(train_df, fit=True)
    test_df = loader.preprocess(test_df, fit=False)
    
    train_data = loader.to_rl_dataset(train_df)
    test_data = loader.to_rl_dataset(test_df)
    
    return train_data, test_data


if __name__ == "__main__":
    # Demo usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", default="p3", choices=["p1", "p2", "p3", "p4"])
    parser.add_argument("--version", default="21.03")
    parser.add_argument("--data-root", default="archive")
    args = parser.parse_args()
    
    print(f"Loading HAI-{args.version} for process {args.process.upper()}")
    train_data, test_data = load_hai_for_offline_rl(
        process=args.process,
        version=args.version,
        data_root=args.data_root,
    )
    
    print(f"\nTrain data:")
    for k, v in train_data.items():
        print(f"  {k}: {v.shape}")
    
    print(f"\nTest data:")
    for k, v in test_data.items():
        print(f"  {k}: {v.shape}")
