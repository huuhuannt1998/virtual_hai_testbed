"""
Check HAI Dataset Dimensions Across Versions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hai_ml.data.hai_loader import load_hai_for_offline_rl

print("="*70)
print("HAI DATASET DIMENSION COMPARISON")
print("="*70)

processes = ["p1", "p2", "p3", "p4"]

for process in processes:
    print(f"\n{process.upper()}:")
    
    # Check 21.03
    try:
        _, test_2103 = load_hai_for_offline_rl(
            process=process,
            version="21.03",
            data_root="archive"
        )
        obs_dim_2103 = test_2103['observations'].shape[1]
        act_dim_2103 = test_2103['actions'].shape[1]
        print(f"  21.03: obs_dim={obs_dim_2103}, action_dim={act_dim_2103}")
    except Exception as e:
        print(f"  21.03: ERROR - {e}")
    
    # Check 22.04
    try:
        _, test_2204 = load_hai_for_offline_rl(
            process=process,
            version="22.04",
            data_root="archive"
        )
        obs_dim_2204 = test_2204['observations'].shape[1]
        act_dim_2204 = test_2204['actions'].shape[1]
        print(f"  22.04: obs_dim={obs_dim_2204}, action_dim={act_dim_2204}")
        
        if obs_dim_2103 != obs_dim_2204:
            print(f"  ⚠ DIMENSION MISMATCH! 21.03 has {obs_dim_2103} features, 22.04 has {obs_dim_2204}")
    except Exception as e:
        print(f"  22.04: ERROR - {e}")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
print("If dimensions differ between versions, cross-version transfer is NOT valid.")
print("The model architecture is fixed to the training data dimensions.")
print("\nOptions:")
print("1. Skip cross-version evaluation (mention as limitation)")
print("2. Note that HAI-22.04 has different schema (not compatible)")
print("3. Focus on 21.03 evaluation only")
