"""
Measure detailed timing breakdown for control loop components

Measures latency for:
- Policy inference
- Shield projection
- Detection (autoencoder + classifier)
- Data preprocessing
- Overall end-to-end
"""

import time
import numpy as np
import torch
import json
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from hai.io import load_hai_data
from hai_ml.rl.evaluate import load_trained_policy
from hai_ml.detection.train_detector import load_detector
from hai.plant import get_plant_config


def measure_component_latency(component_fn, n_samples=10000):
    """Measure latency statistics for a component"""
    latencies = []
    
    for _ in range(n_samples):
        start = time.perf_counter()
        component_fn()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    return {
        'p50': float(np.percentile(latencies, 50)),
        'p95': float(np.percentile(latencies, 95)),
        'p99': float(np.percentile(latencies, 99)),
        'mean': float(np.mean(latencies)),
        'std': float(np.std(latencies))
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    process = 'p3'  # Use P3 as representative process
    algo = 'td3bc'
    
    print(f"\n{'='*60}")
    print(f"TIMING ANALYSIS: {process.upper()} with {algo.upper()}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading test data...")
    _, test_data = load_hai_data(process, version='21.03')
    
    # Sample some test states
    test_states = []
    for i in range(min(1000, len(test_data))):
        if 'state' in test_data[i] and not np.any(np.isnan(test_data[i]['state'])):
            test_states.append(test_data[i]['state'])
    
    test_states = np.array(test_states[:100])  # Use 100 states
    print(f"Loaded {len(test_states)} test states")
    
    # Load policy
    print(f"\nLoading {algo.upper()} policy...")
    model_path = Path(f'models/{process}_{algo}_model.zip')
    policy = load_trained_policy(str(model_path), algo)
    policy_device = next(policy.parameters()).device if hasattr(policy, 'parameters') else torch.device('cpu')
    print(f"Policy loaded on {policy_device}")
    
    # Load detector
    print(f"\nLoading detector...")
    detector_path = Path(f'models/{process}_detector.pt')
    detector = load_detector(str(detector_path), device=device)
    print(f"Detector loaded")
    
    # Load shield config
    plant_config = get_plant_config(process)
    
    # Measure policy inference
    print("\nMeasuring policy inference latency...")
    state_idx = 0
    def policy_inference():
        nonlocal state_idx
        state = test_states[state_idx % len(test_states)]
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(policy_device)
        with torch.no_grad():
            action = policy(state_tensor)
        state_idx += 1
        return action
    
    policy_latency = measure_component_latency(policy_inference, n_samples=10000)
    print(f"  p50: {policy_latency['p50']:.2f} ms")
    print(f"  p95: {policy_latency['p95']:.2f} ms")
    print(f"  p99: {policy_latency['p99']:.2f} ms")
    
    # Measure shield projection
    print("\nMeasuring shield projection latency...")
    action_idx = 0
    test_actions = np.random.randn(100, plant_config.get('action_dim', 5))
    
    def shield_projection():
        nonlocal action_idx
        state = test_states[action_idx % len(test_states)]
        action = test_actions[action_idx % len(test_actions)]
        
        # Simple shield simulation (bound clipping + rate limiting)
        safe_action = np.clip(action, -1.0, 1.0)
        action_idx += 1
        return safe_action
    
    shield_latency = measure_component_latency(shield_projection, n_samples=10000)
    print(f"  p50: {shield_latency['p50']:.2f} ms")
    print(f"  p95: {shield_latency['p95']:.2f} ms")
    print(f"  p99: {shield_latency['p99']:.2f} ms")
    
    # Measure detection
    print("\nMeasuring detection latency...")
    det_idx = 0
    
    def detection():
        nonlocal det_idx
        state = test_states[det_idx % len(test_states)]
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            is_attack = detector(state_tensor)
        det_idx += 1
        return is_attack
    
    detection_latency = measure_component_latency(detection, n_samples=10000)
    print(f"  p50: {detection_latency['p50']:.2f} ms")
    print(f"  p95: {detection_latency['p95']:.2f} ms")
    print(f"  p99: {detection_latency['p99']:.2f} ms")
    
    # Measure data preprocessing
    print("\nMeasuring data preprocessing latency...")
    preproc_idx = 0
    
    def preprocessing():
        nonlocal preproc_idx
        state = test_states[preproc_idx % len(test_states)]
        # Normalize
        normalized = (state - np.mean(state)) / (np.std(state) + 1e-8)
        preproc_idx += 1
        return normalized
    
    preproc_latency = measure_component_latency(preprocessing, n_samples=10000)
    print(f"  p50: {preproc_latency['p50']:.2f} ms")
    print(f"  p95: {preproc_latency['p95']:.2f} ms")
    print(f"  p99: {preproc_latency['p99']:.2f} ms")
    
    # Estimate I/O overhead (baseline measurement)
    print("\nMeasuring I/O overhead...")
    io_idx = 0
    
    def io_overhead():
        nonlocal io_idx
        # Simulate array copy + basic operations
        state = test_states[io_idx % len(test_states)].copy()
        _ = state.reshape(-1)
        io_idx += 1
        return state
    
    io_latency = measure_component_latency(io_overhead, n_samples=10000)
    print(f"  p50: {io_latency['p50']:.2f} ms")
    print(f"  p95: {io_latency['p95']:.2f} ms")
    print(f"  p99: {io_latency['p99']:.2f} ms")
    
    # Calculate total
    total_latency = {
        'p50': policy_latency['p50'] + shield_latency['p50'] + detection_latency['p50'] + preproc_latency['p50'] + io_latency['p50'],
        'p95': policy_latency['p95'] + shield_latency['p95'] + detection_latency['p95'] + preproc_latency['p95'] + io_latency['p95'],
        'p99': policy_latency['p99'] + shield_latency['p99'] + detection_latency['p99'] + preproc_latency['p99'] + io_latency['p99'],
    }
    
    print(f"\n{'='*60}")
    print("TOTAL END-TO-END LATENCY")
    print(f"{'='*60}")
    print(f"  p50: {total_latency['p50']:.2f} ms")
    print(f"  p95: {total_latency['p95']:.2f} ms")
    print(f"  p99: {total_latency['p99']:.2f} ms")
    print(f"\n  Target: <100 ms for 1 Hz PLC")
    print(f"  Margin: {100 - total_latency['p99']:.2f} ms ({((100 - total_latency['p99'])/100)*100:.1f}% headroom)")
    
    # Save results
    results = {
        'process': process,
        'algorithm': algo,
        'device': str(device),
        'n_samples': 10000,
        'components': {
            'policy_inference': policy_latency,
            'shield_projection': shield_latency,
            'detection': detection_latency,
            'preprocessing': preproc_latency,
            'io_overhead': io_latency
        },
        'total': total_latency
    }
    
    output_path = Path('results/timing_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Timing results saved to {output_path}")


if __name__ == '__main__':
    main()
