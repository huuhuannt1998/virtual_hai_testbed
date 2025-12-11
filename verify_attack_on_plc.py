"""
Verify Attack Injection on Real PLC

This script reads values directly from the PLC to confirm
that attack values are being written to the real device.

Run this in a SEPARATE terminal while run_attack_demo.py is running.
"""

import time
import sys
sys.path.insert(0, '.')

from hai.io import PLCClient
from hai.config import PLC_IP, RACK, SLOT, DB_HAI_NUM, OFFSETS


def read_plc_values():
    """Read key sensor values directly from PLC."""
    plc = PLCClient()
    
    print("=" * 70)
    print("PLC Value Monitor - Reading DIRECTLY from PLC")
    print("=" * 70)
    print(f"PLC IP: {PLC_IP}")
    print(f"DB: {DB_HAI_NUM}")
    print()
    
    try:
        plc.connect()
        print("✓ Connected to PLC\n")
        
        # Tags to monitor
        tags = ['P1_PIT01', 'P1_LIT01', 'P1_PIT02', 'P1_TIT01']
        
        print("-" * 70)
        print(f"{'Time':>10} | {'P1_PIT01 (bar)':>14} | {'P1_LIT01 (%)':>12} | {'P1_PIT02':>10} | {'P1_TIT01':>10}")
        print("-" * 70)
        
        # Normal pressure should be ~5 bar
        # During attack with +5 bias, it should show ~10 bar
        
        while True:
            values = plc.read_all_tags()
            
            timestamp = time.strftime("%H:%M:%S")
            p1_pit01 = values.get('P1_PIT01', 0)
            p1_lit01 = values.get('P1_LIT01', 0)
            p1_pit02 = values.get('P1_PIT02', 0)
            p1_tit01 = values.get('P1_TIT01', 0)
            
            # Highlight if pressure is abnormally high (attack detected)
            alert = ""
            if p1_pit01 > 8.0:  # Normal is ~5 bar, attack adds +5
                alert = " ⚠️ ATTACK DETECTED!"
            
            print(f"{timestamp:>10} | {p1_pit01:>14.2f} | {p1_lit01:>12.2f} | {p1_pit02:>10.2f} | {p1_tit01:>10.2f}{alert}")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
    finally:
        plc.disconnect()


if __name__ == '__main__':
    read_plc_values()
