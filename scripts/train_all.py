"""Train all RL algorithms and detection models for all HAI processes."""
import sys
import subprocess
from pathlib import Path
import time
from datetime import datetime

# Training configurations
PROCESSES = ["p1", "p2", "p3", "p4"]
RL_ALGORITHMS = ["bc", "td3bc", "cql", "iql"]
VERSION = "21.03"
RL_EPOCHS = 100
DETECTION_EPOCHS = 50

def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "="*70)
    print(description)
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True,
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed/60:.1f} minutes")
        return True, elapsed
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Failed after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
        return False, elapsed
    except KeyboardInterrupt:
        print(f"\n✗ Interrupted by user")
        raise


def train_rl_model(process: str, algo: str):
    """Train a single RL model."""
    cmd = [
        "python", "-m", "hai_ml.rl.train_real",
        "--algo", algo,
        "--process", process,
        "--version", VERSION,
        "--epochs", str(RL_EPOCHS),
    ]
    
    description = f"Training RL: {algo.upper()} on {process.upper()} (HAI-{VERSION})"
    return run_command(cmd, description)


def train_detection_model(process: str):
    """Train detection model for a process."""
    cmd = [
        "python", "-m", "hai_ml.detection.train_detector",
        "--process", process,
        "--version", VERSION,
        "--epochs", str(DETECTION_EPOCHS),
    ]
    
    description = f"Training Detection: {process.upper()} (HAI-{VERSION})"
    return run_command(cmd, description)


def check_existing_models():
    """Check what models already exist."""
    models_dir = Path("models/logs")
    if not models_dir.exists():
        return set()
    
    existing = set()
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            # Parse names like "p3_td3bc_v2103" or "p3_bc_v2103"
            parts = model_dir.name.split('_')
            if len(parts) >= 2:
                process = parts[0]  # e.g., "p3"
                algo = parts[1]     # e.g., "td3bc", "bc"
                existing.add((process, algo))
    
    # Check for detection models
    detector_files = Path("models").glob("detector_p*.pt")
    for f in detector_files:
        # e.g., "detector_p3.pt" -> "p3"
        process = f.stem.replace("detector_", "")
        existing.add((process, "detector"))
    
    return existing


def main():
    """Train all RL algorithms and detection models for all processes."""
    print("="*70)
    print("HAI COMPLETE TRAINING PIPELINE")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Processes: {', '.join(PROCESSES)}")
    print(f"  RL Algorithms: {', '.join(RL_ALGORITHMS)}")
    print(f"  Detection: Autoencoder + Classifier + Hybrid")
    print(f"  Version: {VERSION}")
    print(f"  RL Epochs: {RL_EPOCHS}")
    print(f"  Detection Epochs: {DETECTION_EPOCHS}")
    
    # Calculate totals
    total_rl = len(PROCESSES) * len(RL_ALGORITHMS)
    total_detection = len(PROCESSES)
    total_models = total_rl + total_detection
    
    print(f"\nTotal models to train:")
    print(f"  RL models: {total_rl} ({len(PROCESSES)} processes × {len(RL_ALGORITHMS)} algorithms)")
    print(f"  Detection models: {total_detection} (1 per process)")
    print(f"  Total: {total_models}")
    print(f"\nEstimated time: ~{total_rl * 20 + total_detection * 15} minutes")
    print(f"  ≈ {(total_rl * 20 + total_detection * 15) / 60:.1f} hours")
    
    # Check existing models
    existing = check_existing_models()
    if existing:
        print(f"\n⚠ Found {len(existing)} existing models:")
        for process, algo in sorted(existing):
            print(f"  - {process.upper()}-{algo.upper()}")
        print("\nThese will be SKIPPED to avoid retraining.")
    
    response = input("\nPress Enter to start training (or Ctrl+C to cancel)...")
    
    results = []
    start_all = time.time()
    current = 0
    
    # Train RL models
    print("\n" + "#"*70)
    print("PHASE 1: TRAINING RL MODELS")
    print("#"*70)
    
    for process in PROCESSES:
        for algo in RL_ALGORITHMS:
            current += 1
            
            # Skip if already exists
            if (process, algo) in existing:
                print(f"\n[{current}/{total_models}] Skipping {algo.upper()} on {process.upper()} (already exists)")
                results.append({
                    'type': 'rl',
                    'process': process,
                    'algo': algo,
                    'success': True,
                    'time': 0,
                    'skipped': True,
                })
                continue
            
            print(f"\n{'#'*70}")
            print(f"Progress: [{current}/{total_models}] RL Training")
            print(f"{'#'*70}")
            
            success, elapsed = train_rl_model(process, algo)
            
            results.append({
                'type': 'rl',
                'process': process,
                'algo': algo,
                'success': success,
                'time': elapsed,
                'skipped': False,
            })
    
    # Train Detection models
    print("\n" + "#"*70)
    print("PHASE 2: TRAINING DETECTION MODELS")
    print("#"*70)
    
    for process in PROCESSES:
        current += 1
        
        # Skip if already exists
        if (process, "detector") in existing:
            print(f"\n[{current}/{total_models}] Skipping detector on {process.upper()} (already exists)")
            results.append({
                'type': 'detection',
                'process': process,
                'algo': 'detector',
                'success': True,
                'time': 0,
                'skipped': True,
            })
            continue
        
        print(f"\n{'#'*70}")
        print(f"Progress: [{current}/{total_models}] Detection Training")
        print(f"{'#'*70}")
        
        success, elapsed = train_detection_model(process)
        
        results.append({
            'type': 'detection',
            'process': process,
            'algo': 'detector',
            'success': success,
            'time': elapsed,
            'skipped': False,
        })
    
    # Print summary
    elapsed_all = time.time() - start_all
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    completed = sum(1 for r in results if r['success'])
    failed = sum(1 for r in results if not r['success'])
    skipped = sum(1 for r in results if r.get('skipped', False))
    trained = completed - skipped
    
    print(f"\nTotal time: {elapsed_all/60:.1f} minutes ({elapsed_all/3600:.2f} hours)")
    print(f"Total models: {total_models}")
    print(f"  Trained: {trained}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    
    # RL Models
    rl_results = [r for r in results if r['type'] == 'rl']
    rl_success = sum(1 for r in rl_results if r['success'])
    rl_failed = sum(1 for r in rl_results if not r['success'])
    
    print(f"\nRL Models ({rl_success}/{len(rl_results)} successful):")
    for process in PROCESSES:
        process_results = [r for r in rl_results if r['process'] == process]
        success_algos = [r['algo'].upper() for r in process_results if r['success']]
        if success_algos:
            status = "✓" if len(success_algos) == len(RL_ALGORITHMS) else "⚠"
            print(f"  {status} {process.upper()}: {', '.join(success_algos)}")
    
    # Detection Models
    det_results = [r for r in results if r['type'] == 'detection']
    det_success = sum(1 for r in det_results if r['success'])
    det_failed = sum(1 for r in det_results if not r['success'])
    
    print(f"\nDetection Models ({det_success}/{len(det_results)} successful):")
    for r in det_results:
        status = "✓" if r['success'] else "✗"
        skipped_tag = " (skipped)" if r.get('skipped') else ""
        print(f"  {status} {r['process'].upper()}{skipped_tag}")
    
    if failed > 0:
        print(f"\n⚠ {failed} models failed:")
        for r in results:
            if not r['success']:
                print(f"  ✗ {r['process'].upper()}-{r['algo'].upper()}")
    
    print("\n" + "="*70)
    print("Models saved in:")
    print("  RL models: models/logs/")
    print("  Detection models: models/detector_*.pt")
    print("="*70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
