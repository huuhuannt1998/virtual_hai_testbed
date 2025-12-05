"""
OPC UA Connection Test for HAI Testbed

For S7-1200 V4.5+ firmware which blocks classic S7 protocol.
Uses OPC UA instead of Snap7.
"""

import sys
try:
    from opcua import Client
except ImportError:
    print("ERROR: opcua not installed. Run: pip install opcua")
    sys.exit(1)


def test_opcua_connection():
    print("=" * 60)
    print("HAI Testbed - OPC UA Connection Test")
    print("=" * 60)
    
    # OPC UA endpoint for S7-1200
    endpoint = "opc.tcp://192.168.0.1:4840"
    print(f"\nTarget: {endpoint}")
    print()
    
    client = Client(endpoint)
    
    # Step 1: Connect
    print("[1/3] Connecting to OPC UA server...")
    try:
        client.connect()
        print("      ✓ Connected successfully!")
    except Exception as e:
        print(f"      ✗ Connection failed: {e}")
        return False
    
    # Step 2: Browse for DB_HAI
    print("\n[2/3] Browsing for DB_HAI nodes...")
    try:
        root = client.get_root_node()
        objects = client.get_objects_node()
        
        # Browse the tree
        print("      Browsing server namespace...")
        
        # Try to find the PLC data
        for child in objects.get_children():
            name = child.get_browse_name().Name
            print(f"      Found: {name}")
            
            if "PLC" in name or "Data" in name or "S7" in name:
                for subchild in child.get_children():
                    subname = subchild.get_browse_name().Name
                    print(f"        - {subname}")
                    
    except Exception as e:
        print(f"      ⚠ Browse error: {e}")
    
    # Step 3: Try to read a specific node
    print("\n[3/3] Attempting to read DB_HAI.P1_B2004...")
    try:
        # Node ID format for S7-1200: "ns=3;s=\"DB_HAI\".\"P1_B2004\""
        node_id = 'ns=3;s="DB_HAI"."P1_B2004"'
        node = client.get_node(node_id)
        value = node.get_value()
        print(f"      ✓ Read successful: P1_B2004 = {value}")
    except Exception as e:
        print(f"      ✗ Read failed: {e}")
        print("\n      Trying alternate node path...")
        
        try:
            # Try different namespace
            for ns in [2, 3, 4]:
                try:
                    node_id = f'ns={ns};s="DB_HAI"."P1_B2004"'
                    node = client.get_node(node_id)
                    value = node.get_value()
                    print(f"      ✓ Found at ns={ns}: P1_B2004 = {value}")
                    break
                except:
                    pass
        except Exception as e2:
            print(f"      ✗ Alternate also failed: {e2}")
    
    client.disconnect()
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
    return True


if __name__ == '__main__':
    test_opcua_connection()
