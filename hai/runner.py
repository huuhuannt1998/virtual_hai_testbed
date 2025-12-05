"""Main loop to run the virtual HAI testbed.

This script:
1. Connects to the PLC (TIA Portal / PLCSIM Advanced)
2. Runs the virtual plant simulation at 1 Hz
3. Exchanges data with PLC (sensors -> PLC, actuators <- PLC)
4. Optionally logs all tag values to CSV (HAI-style dataset)

The PLC should contain the control logic (PID loops, safety interlocks).
This script simulates the physical processes (boiler, turbine, water, HIL).

Usage:
    python -m hai.runner

Reference: https://github.com/icsdataset/hai
"""

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from .io import PLCClient, ensure_connected
from .plant import VirtualHAIPlant
from .config import SIM_DT_SEC, LOG_ENABLED, LOG_FILE, OFFSETS


class HAILogger:
    """CSV logger for HAI-style time series data."""
    
    def __init__(self, filepath: str = LOG_FILE):
        self.filepath = Path(filepath)
        self.file = None
        self.writer = None
        self._fieldnames = ["timestamp"] + list(OFFSETS.keys()) + ["Attack"]
    
    def start(self):
        """Open the log file and write header."""
        self.file = open(self.filepath, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self._fieldnames)
        self.writer.writeheader()
        print(f"[Logger] Logging to {self.filepath}")
    
    def log(self, values: dict, attack: int = 0):
        """Log one row of data."""
        if self.writer is None:
            return
        
        row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        for tag in OFFSETS.keys():
            row[tag] = values.get(tag, 0.0)
        row["Attack"] = attack
        
        self.writer.writerow(row)
        self.file.flush()
    
    def stop(self):
        """Close the log file."""
        if self.file:
            self.file.close()
            self.file = None
            self.writer = None


def main(log_enabled: bool = LOG_ENABLED, max_steps: Optional[int] = None):
    """Main simulation loop.
    
    Args:
        log_enabled: Whether to log data to CSV
        max_steps: Maximum simulation steps (None = run forever)
    """
    plc = PLCClient()
    
    print("[Runner] Virtual HAI Testbed")
    print(f"[Runner] Connecting to PLC at {plc.client}...")
    
    if not ensure_connected(plc, retry_interval=2.0, max_retries=10):
        print("[Runner] Could not connect to PLC. Exiting.")
        return
    
    plant = VirtualHAIPlant(plc)
    logger = HAILogger() if log_enabled else None
    
    if logger:
        logger.start()
    
    print(f"[Runner] Starting simulation loop at {SIM_DT_SEC} Hz")
    print("[Runner] Press Ctrl+C to stop")
    
    step_count = 0
    try:
        while True:
            t0 = time.time()
            
            # Step the plant simulation
            plant.step()
            step_count += 1
            
            # Log all values
            if logger:
                try:
                    all_values = plc.read_all_tags()
                    logger.log(all_values, attack=0)
                except Exception as e:
                    print(f"[Runner] Logging error: {e}")
            
            # Print status every 60 steps
            if step_count % 60 == 0:
                p1 = plant.state.p1
                p2 = plant.state.p2
                p3 = plant.state.p3
                print(f"[Step {step_count}] "
                      f"P1: L={p1.level:.1f}% T={p1.temp1:.1f}°C P={p1.pressure1:.2f}bar | "
                      f"P2: {p2.speed:.0f}RPM {'RUN' if p2.is_running else 'STOP'} | "
                      f"P3: L={p3.level:.1f}%")
            
            # Check for max steps
            if max_steps and step_count >= max_steps:
                print(f"[Runner] Reached {max_steps} steps. Stopping.")
                break
            
            # Maintain loop timing
            elapsed = time.time() - t0
            sleep_time = max(0.0, SIM_DT_SEC - elapsed)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n[Runner] Stopped by user")
    except Exception as e:
        print(f"[Runner] Error: {e}")
    finally:
        if logger:
            logger.stop()
        plc.disconnect()
        print(f"[Runner] Disconnected. Total steps: {step_count}")


if __name__ == "__main__":
    main()
