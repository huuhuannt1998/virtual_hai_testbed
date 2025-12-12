"""
Investigate P2 Detection Issues
================================

Analyzes why P2 detector performs poorly and suggests fixes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import torch
from hai_ml.detection.train_detector import HybridDetector
from hai_ml.data.hai_loader import load_hai_for_offline_rl

print("="*70)
print("P2 DETECTION ANALYSIS")
print("="*70)

# Load P2 detector metadata
meta_path = Path("models/p2_detector_v2103_meta.json")
with open(meta_path) as f:
    meta = json.load(f)

print(f"\nDetector Metadata:")
print(f"  Input dim: {meta['input_dim']}")
print(f"  AE threshold: {meta['ae_threshold']}")
print(f"  Train samples: {meta['train_samples']:,}")
print(f"  Train attack samples: {meta['train_attack_samples']}")

# Load detector
device = "cuda" if torch.cuda.is_available() else "cpu"
detector = HybridDetector(input_dim=meta['input_dim'], device=device)
detector_path = Path("models/p2_detector_v2103.pth")
detector.load(str(detector_path))

print(f"\n✓ Detector loaded")

# Load P2 test data
print("\nLoading P2 test data...")
_, test_data = load_hai_for_offline_rl(
    process="p2",
    version="21.03",
    data_root="archive"
)

print(f"  Test samples: {len(test_data['observations']):,}")
print(f"  Obs shape: {test_data['observations'].shape}")
print(f"  Obs range: [{test_data['observations'].min():.3f}, {test_data['observations'].max():.3f}]")
print(f"  Obs mean: {test_data['observations'].mean():.3f}")
print(f"  Obs std: {test_data['observations'].std():.3f}")

# Check for NaN
has_nan = np.isnan(test_data['observations']).any()
print(f"  Has NaN: {has_nan}")

if has_nan:
    nan_count = np.isnan(test_data['observations']).sum()
    print(f"  NaN count: {nan_count}")
    print(f"\n⚠ WARNING: P2 test data contains NaN values!")
    print("  This explains the NaN ITAE in attack evaluation.")

# Sample and test detection
print("\nTesting detection on clean samples...")
sample_indices = np.random.choice(len(test_data['observations']), 1000, replace=False)
X_sample = test_data['observations'][sample_indices]

# Remove NaN if present
if np.isnan(X_sample).any():
    print("  Removing NaN samples...")
    valid_mask = ~np.isnan(X_sample).any(axis=1)
    X_sample = X_sample[valid_mask]
    print(f"  Valid samples: {len(X_sample)}")

predictions, ae_scores, cls_scores = detector.predict(X_sample)

print(f"\nDetection Results on Clean Data:")
print(f"  Flagged as attack: {np.sum(predictions == 1)}/{len(predictions)} ({np.mean(predictions)*100:.1f}%)")
print(f"  AE scores range: [{ae_scores.min():.6f}, {ae_scores.max():.6f}]")
print(f"  AE threshold: {detector.ae_threshold:.6f}")
print(f"  Classifier scores range: [{cls_scores.min():.3f}, {cls_scores.max():.3f}]")

# Check if threshold is appropriate
ae_above_threshold = (ae_scores > detector.ae_threshold).sum()
print(f"\n  Samples above AE threshold: {ae_above_threshold}/{len(ae_scores)} ({ae_above_threshold/len(ae_scores)*100:.1f}%)")

print("\n" + "="*70)
print("FINDINGS:")
print("="*70)

if has_nan:
    print("❌ CRITICAL: P2 data contains NaN values")
    print("   → Fix data preprocessing in hai_loader.py")
    print("   → The random noise fix for P2 might be causing issues")

if ae_above_threshold / len(ae_scores) < 0.01:
    print("❌ AE threshold too high - detector flags almost nothing")
    print(f"   → Current: {detector.ae_threshold:.6f}")
    print(f"   → Suggested: {np.percentile(ae_scores, 95):.6f} (95th percentile)")

print("\n" + "="*70)
print("RECOMMENDATIONS:")
print("="*70)
print("1. Fix NaN values in P2 data preprocessing")
print("2. Retrain P2 detector with cleaned data")
print("3. Consider lowering AE threshold for better sensitivity")
print("4. Add actual attack samples from test set for classifier training")
