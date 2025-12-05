"""
Quick PLC Connection Test for HAI Testbed

Run this script to verify Snap7 can communicate with your PLC
before running the full simulation.
"""

import sys
try:
    import snap7
except ImportError:
    print("ERROR: snap7 not installed. Run: pip install python-snap7")
    sys.exit(1)

from hai.config import PLC_IP, RACK, SLOT, DB_HAI_NUM, DB_SIZE_BYTES


def test_connection():
    print("=" * 60)
    print("HAI Testbed - PLC Connection Test")
    print("=" * 60)
    print(f"\nTarget PLC: {PLC_IP}")
    print(f"Rack: {RACK}, Slot: {SLOT}")
    print(f"Data Block: DB{DB_HAI_NUM} ({DB_SIZE_BYTES} bytes)")
    print()

    client = snap7.client.Client()
    
    # Step 1: Connect
    print("[1/4] Connecting to PLC...")
    try:
        # Try PG connection type (programming device) for V4.5 firmware
        client.set_connection_type(1)  # 1 = PG connection
        client.connect(PLC_IP, RACK, SLOT)
        print("      ✓ Connected successfully!")
    except Exception as e:
        print(f"      ✗ Connection failed: {e}")
        print("\n  Troubleshooting:")
        print("  - Check PLC IP address in hai/config.py")
        print("  - Ensure PLC is in RUN mode")
        print("  - Verify network connectivity (ping the PLC)")
        print("  - Check Windows Firewall settings")
        return False

    # Step 2: Check PLC state
    print("\n[2/4] Checking PLC state...")
    try:
        state = client.get_cpu_state()
        state_name = {0: "Unknown", 4: "STOP", 8: "RUN"}.get(state, f"Code {state}")
        print(f"      CPU State: {state_name}")
        if state != 8:
            print("      ⚠ Warning: PLC not in RUN mode")
    except Exception as e:
        print(f"      ⚠ Could not read state: {e}")

    # Step 3: Read DB
    print(f"\n[3/4] Reading DB{DB_HAI_NUM}...")
    try:
        data = client.db_read(DB_HAI_NUM, 0, 4)  # Read first 4 bytes
        print(f"      ✓ DB read successful (first 4 bytes: {data.hex()})")
    except Exception as e:
        print(f"      ✗ DB read failed: {e}")
        print("\n  Troubleshooting:")
        print(f"  - Ensure DB{DB_HAI_NUM} exists in the PLC program")
        print("  - Check 'Optimized block access' is DISABLED for the DB")
        print("  - Verify PUT/GET communication is enabled in PLC settings")
        client.disconnect()
        return False

    # Step 4: Write test
    print(f"\n[4/4] Write test (P1_PIT01 at offset 0)...")
    try:
        import struct
        test_value = 50.0
        data = bytearray(struct.pack('>f', test_value))
        client.db_write(DB_HAI_NUM, 0, data)
        
        # Read back
        read_data = client.db_read(DB_HAI_NUM, 0, 4)
        read_value = struct.unpack('>f', read_data)[0]
        
        if abs(read_value - test_value) < 0.01:
            print(f"      ✓ Write/read verified: {read_value:.2f}")
        else:
            print(f"      ⚠ Value mismatch: wrote {test_value}, read {read_value}")
    except Exception as e:
        print(f"      ✗ Write test failed: {e}")
        client.disconnect()
        return False

    client.disconnect()
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! Ready to run simulation.")
    print("=" * 60)
    print("\nNext steps:")
    print("  python -m hai.runner")
    print()
    return True


if __name__ == '__main__':
    success = test_connection()
    sys.exit(0 if success else 1)
