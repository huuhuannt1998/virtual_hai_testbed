"""
HAI Testbed - Attack Simulation Demo

This script runs the HAI virtual testbed with various attack scenarios.
"""

import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hai.plant import VirtualHAIPlant
from hai.attacks import (
    AttackSimulator, 
    AttackConfig, 
    AttackType,
    AttackScenarioManager,
    create_p1_pressure_bias_attack,
    create_p1_level_replay_attack,
    create_p2_speed_ramp_attack,
)
from hai.io import PLCClient
from hai.config import PLC_IP, RACK, SLOT, LOG_FILE
from hai.runner import HAILogger


def run_attack_simulation(duration_seconds=120):
    """
    Run simulation with attack scenarios.
    
    Timeline:
    - 0-30s: Normal operation
    - 30-60s: P1 pressure bias attack (+5 bar)
    - 60-90s: Normal operation  
    - 90-120s: P1 level replay attack
    """
    
    print("=" * 60)
    print("HAI Testbed - Attack Simulation")
    print("=" * 60)
    
    # Initialize components
    print("\n[1/4] Connecting to PLC...")
    plc = PLCClient()
    plc.connect()  # Uses PLC_IP, RACK, SLOT from config.py
    print(f"      Connected to {PLC_IP}")
    
    print("\n[2/4] Initializing plant simulation...")
    plant = VirtualHAIPlant(plc=plc)
    
    print("\n[3/4] Setting up attack scenarios...")
    attack_sim = AttackSimulator()
    
    # Attack 1: Pressure bias injection (30-60s)
    attack1 = AttackConfig(
        attack_type=AttackType.BIAS_INJECTION,
        target_tags=['P1_PIT01'],
        bias_value=5.0,  # Add 5 bar to pressure reading
        start_time=30.0,
        duration=30.0
    )
    attack_sim.add_attack(attack1)
    print("      - Attack 1: P1 Pressure Bias (+5 bar) @ 30-60s")
    
    # Attack 2: Level replay attack (90-120s)
    attack2 = AttackConfig(
        attack_type=AttackType.REPLAY,
        target_tags=['P1_LIT01'],
        replay_delay=20,  # Replay 20-second old values
        replay_buffer_size=30,
        start_time=90.0,
        duration=30.0
    )
    attack_sim.add_attack(attack2)
    print("      - Attack 2: P1 Level Replay (20s delay) @ 90-120s")
    
    print("\n[4/4] Starting simulation...")
    print(f"      Duration: {duration_seconds} seconds")
    print("      Press Ctrl+C to stop early\n")
    
    # Logger
    log_file = "hai_attack_log.csv"
    logger = HAILogger(log_file)
    logger.start()  # Must call start() to open file
    
    print("\n" + "-" * 60)
    print(f"{'Time':>6} | {'Attack':^20} | {'P1_PIT01':>10} | {'P1_LIT01':>10}")
    print("-" * 60)
    
    try:
        start_time = time.time()
        step = 0
        
        while step < duration_seconds:
            loop_start = time.time()
            
            # Advance attack timing
            attack_sim.step(dt=1.0)
            
            # Step the plant simulation
            plant.step()  # No dt argument
            
            # Get current sensor values
            sensors = {
                'P1_PIT01': plant.state.p1.pressure1,
                'P1_LIT01': plant.state.p1.level,
            }
            
            # Apply attacks to sensor values (what gets written to PLC)
            attacked_sensors = attack_sim.apply_to_sensors(sensors)
            
            # Get attack status
            attack_label = attack_sim.attack_label
            attack_name = "Normal" if attack_label == 0 else f"Attack {attack_label}"
            
            # Print status every 5 seconds
            if step % 5 == 0:
                print(f"{step:>5}s | {attack_name:^20} | {attacked_sensors['P1_PIT01']:>10.2f} | {attacked_sensors['P1_LIT01']:>10.2f}")
            
            # Log data (with attack label)
            log_data = plant.get_all_values()
            logger.log(log_data, attack=attack_label)
            
            step += 1
            
            # Maintain 1 Hz loop
            elapsed = time.time() - loop_start
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)
                
    except KeyboardInterrupt:
        print("\n\n[!] Simulation stopped by user")
    finally:
        logger.stop()  # Use stop() not close()
        plc.disconnect()
        
    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {log_file}")
    print(f"Total samples: {step}")
    print(f"Attack samples: Check 'Attack' column (0=normal, 1+=attack)")
    

if __name__ == '__main__':
    duration = 120  # 2 minutes
    if len(sys.argv) > 1:
        duration = int(sys.argv[1])
    
    run_attack_simulation(duration)
