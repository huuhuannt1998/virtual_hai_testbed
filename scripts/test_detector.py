"""
Quick test to verify detector loading and basic functionality.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from hai_ml.detection.train_detector import HybridDetector

print("Testing detector loading...")

# Test P3 detector
detector_path = Path("models/p3_detector_v2103.pth")
print(f"Detector path: {detector_path}")
print(f"Exists: {detector_path.exists()}")

if detector_path.exists():
    try:
        # Load detector
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        
        # Load metadata to get input_dim
        import json
        meta_path = Path("models/p3_detector_v2103_meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        
        print(f"Metadata: {meta}")
        
        input_dim = meta.get("input_dim", 3)
        print(f"Input dim: {input_dim}")
        
        detector = HybridDetector(input_dim=input_dim, device=device)
        detector.load(str(detector_path))
        
        print("✓ Detector loaded successfully!")
        
        # Test prediction
        test_input = np.random.randn(10, input_dim).astype(np.float32)
        print(f"Test input shape: {test_input.shape}")
        
        predictions, ae_scores, cls_scores = detector.predict(test_input)
        print(f"Predictions shape: {predictions.shape}")
        print(f"Predictions: {predictions}")
        print(f"✓ Detector predictions work!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("❌ Detector file not found!")
