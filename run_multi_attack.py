"""
HAI Testbed - Multi-Process Attack Demo

Attack scenarios for all 4 processes:
- P1: Boiler (pressure, level, temperature)
- P2: Turbine (speed, vibration)
- P3: Water Treatment (level, flow)
- P4: HIL Power Grid (load, power)

Usage:
    python run_multi_attack.py [process] [duration]
    
Examples:
    python run_multi_attack.py p1 30      # Attack P1 Boiler
    python run_multi_attack.py p2 30      # Attack P2 Turbine
    python run_multi_attack.py p3 30      # Attack P3 Water
    python run_multi_attack.py p4 30      # Attack P4 HIL
    python run_multi_attack.py all 60     # Attack all processes
"""

import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hai.io import PLCClient
from hai.config import PLC_IP
from hai.attacks import AttackSimulator, AttackConfig, AttackType


# =============================================================================
# Attack Scenarios for Each Process
# =============================================================================

def get_p1_attacks():
    """P1 Boiler Process Attacks"""
    return [
        AttackConfig(
            attack_type=AttackType.BIAS_INJECTION,
            target_tags=['P1_PIT01'],
            bias_value=5.0,  # Add 5 bar to pressure
            start_time=5.0,
            duration=10.0
        ),
        AttackConfig(
            attack_type=AttackType.BIAS_INJECTION,
            target_tags=['P1_LIT01'],
            bias_value=-20.0,  # Reduce level reading by 20%
            start_time=20.0,
            duration=10.0
        ),
        AttackConfig(
            attack_type=AttackType.BIAS_INJECTION,
            target_tags=['P1_TIT01'],
            bias_value=15.0,  # Add 15°C to temperature
            start_time=35.0,
            duration=10.0
        ),
    ]


def get_p2_attacks():
    """P2 Turbine Process Attacks"""
    return [
        AttackConfig(
            attack_type=AttackType.BIAS_INJECTION,
            target_tags=['P2_SIT01'],
            bias_value=500.0,  # Add 500 RPM to speed reading
            start_time=5.0,
            duration=10.0
        ),
        AttackConfig(
            attack_type=AttackType.BIAS_INJECTION,
            target_tags=['P2_VIBTR01', 'P2_VIBTR02'],
            bias_value=10.0,  # Add 10 mm/s to vibration
            start_time=20.0,
            duration=10.0
        ),
    ]


def get_p3_attacks():
    """P3 Water Treatment Process Attacks"""
    return [
        AttackConfig(
            attack_type=AttackType.BIAS_INJECTION,
            target_tags=['P3_LIT01'],
            bias_value=30.0,  # Add 30% to level reading
            start_time=5.0,
            duration=10.0
        ),
        AttackConfig(
            attack_type=AttackType.BIAS_INJECTION,
            target_tags=['P3_FIT01'],
            bias_value=-15.0,  # Reduce flow reading
            start_time=20.0,
            duration=10.0
        ),
    ]


def get_p4_attacks():
    """P4 HIL Power Grid Attacks"""
    return [
        AttackConfig(
            attack_type=AttackType.BIAS_INJECTION,
            target_tags=['P4_ST_PT01'],
            bias_value=50.0,  # Add 50 units to steam turbine pressure
            start_time=5.0,
            duration=10.0
        ),
        AttackConfig(
            attack_type=AttackType.BIAS_INJECTION,
            target_tags=['P4_LD'],
            bias_value=20.0,  # Add 20% to load demand
            start_time=20.0,
            duration=10.0
        ),
    ]


# =============================================================================
# Main Attack Runner
# =============================================================================

def run_attack(process: str, duration: int):
    """Run attack on specified process."""
    
    process = process.lower()
    
    # Get attacks for selected process
    attacks = []
    monitor_tags = []
    
    if process == 'p1' or process == 'all':
        attacks.extend(get_p1_attacks())
        monitor_tags.extend(['P1_PIT01', 'P1_LIT01', 'P1_TIT01'])
    if process == 'p2' or process == 'all':
        attacks.extend(get_p2_attacks())
        monitor_tags.extend(['P2_SIT01', 'P2_VIBTR01'])
    if process == 'p3' or process == 'all':
        attacks.extend(get_p3_attacks())
        monitor_tags.extend(['P3_LIT01', 'P3_FIT01'])
    if process == 'p4' or process == 'all':
        attacks.extend(get_p4_attacks())
        monitor_tags.extend(['P4_ST_PT01', 'P4_LD'])
    
    if not attacks:
        print(f"Unknown process: {process}")
        print("Valid options: p1, p2, p3, p4, all")
        return
    
    print("=" * 70)
    print(f"HAI Testbed - Attack on {process.upper()}")
    print("=" * 70)
    
    # Connect to PLC
    print(f"\n[1/3] Connecting to PLC at {PLC_IP}...")
    plc = PLCClient()
    plc.connect()
    print("      ✓ Connected")
    
    # Setup attacks
    print("\n[2/3] Setting up attacks...")
    attack_sim = AttackSimulator()
    for i, attack in enumerate(attacks):
        attack_sim.add_attack(attack)
        tags_str = ', '.join(attack.target_tags)
        print(f"      Attack {i+1}: {tags_str} ({attack.attack_type.name}) @ {attack.start_time}-{attack.start_time + attack.duration}s")
    
    # Run simulation
    print(f"\n[3/3] Running for {duration} seconds...")
    print("      Attacks will be INJECTED to real PLC!")
    print("      Press Ctrl+C to stop\n")
    
    # Header
    header = "Time | Attack |"
    for tag in monitor_tags[:4]:  # Show max 4 tags
        header += f" {tag:>12} |"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    try:
        step = 0
        while step < duration:
            loop_start = time.time()
            
            # Advance attack timing
            attack_sim.step(dt=1.0)
            
            # Read current PLC values
            plc_values = plc.read_all_tags()
            
            # Get values for monitored tags
            sensor_values = {tag: plc_values.get(tag, 0.0) for tag in monitor_tags}
            
            # Apply attacks
            attacked_values = attack_sim.apply_to_sensors(sensor_values)
            
            # INJECT attacked values to PLC
            if attack_sim.is_under_attack:
                plc.write_multiple_tags(attacked_values)
            
            # Print status
            attack_label = attack_sim.attack_label
            attack_str = f"ATK {attack_label}" if attack_label > 0 else "Normal"
            injected = " [INJECTED]" if attack_sim.is_under_attack else ""
            
            row = f"{step:>4}s | {attack_str:>6} |"
            for tag in monitor_tags[:4]:
                val = attacked_values.get(tag, 0.0)
                row += f" {val:>12.2f} |"
            print(row + injected)
            
            step += 1
            elapsed = time.time() - loop_start
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)
                
    except KeyboardInterrupt:
        print("\n\n[!] Attack stopped by user")
    finally:
        plc.disconnect()
    
    print("\n" + "=" * 70)
    print("Attack simulation complete")
    print("=" * 70)


def print_help():
    print(__doc__)
    print("\nAvailable attack scenarios:")
    print()
    print("  P1 (Boiler):")
    print("    - P1_PIT01: Pressure bias +5 bar @ 5-15s")
    print("    - P1_LIT01: Level bias -20% @ 20-30s")
    print("    - P1_TIT01: Temperature bias +15°C @ 35-45s")
    print()
    print("  P2 (Turbine):")
    print("    - P2_SIT01: Speed bias +500 RPM @ 5-15s")
    print("    - P2_VIBTR01/02: Vibration bias +10 mm/s @ 20-30s")
    print()
    print("  P3 (Water Treatment):")
    print("    - P3_LIT01: Level bias +30% @ 5-15s")
    print("    - P3_FIT01: Flow bias -15 L/min @ 20-30s")
    print()
    print("  P4 (HIL Power Grid):")
    print("    - P4_ST_PT01: Steam pressure bias +50 @ 5-15s")
    print("    - P4_LD: Load demand bias +20% @ 20-30s")


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help']:
        print_help()
        sys.exit(0)
    
    process = sys.argv[1]
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    
    run_attack(process, duration)
