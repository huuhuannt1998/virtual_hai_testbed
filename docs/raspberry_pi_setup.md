# Raspberry Pi 3 Setup for HAI Testbed Attack Simulation

This guide explains how to use a Raspberry Pi 3 as an attack node to inject false data into the Siemens PLC.

## Network Architecture

```
┌─────────────────┐     Ethernet      ┌─────────────────┐
│  Raspberry Pi 3 │◄──────────────────►│  Siemens PLC    │
│  (Attacker)     │    192.168.0.x     │  192.168.0.1    │
│                 │                    │  CPU 1212FC     │
└─────────────────┘                    └─────────────────┘
        │                                      │
        └──────────────┬───────────────────────┘
                       │
              ┌────────▼────────┐
              │  Windows PC     │
              │  (TIA Portal)   │
              │  192.168.0.x    │
              └─────────────────┘
```

## Hardware Requirements

- Raspberry Pi 3 (Model B or B+)
- MicroSD card (8GB+)
- Ethernet cable
- Power supply (5V 2.5A)
- Network switch connecting Pi, PLC, and PC

## Step 1: Install Raspberry Pi OS

1. Download Raspberry Pi OS Lite (64-bit) from https://www.raspberrypi.com/software/
2. Flash to SD card using Raspberry Pi Imager
3. Enable SSH: Create empty file named `ssh` in boot partition
4. Boot the Pi and connect via SSH

```bash
ssh pi@raspberrypi.local
# Default password: raspberry
```

## Step 2: Configure Network

Set a static IP on the same subnet as the PLC:

```bash
sudo nano /etc/dhcpcd.conf
```

Add at the end:
```
interface eth0
static ip_address=192.168.0.100/24
static routers=192.168.0.1
```

Reboot:
```bash
sudo reboot
```

Verify connectivity:
```bash
ping 192.168.0.1  # Should reach PLC
```

## Step 3: Install Python and Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install -y python3 python3-pip python3-venv git

# Create virtual environment
python3 -m venv ~/hai_venv
source ~/hai_venv/bin/activate

# Install dependencies
pip install python-snap7 numpy
```

## Step 4: Install Snap7 Library

```bash
# Install Snap7 shared library
sudo apt install -y libsnap7-1 libsnap7-dev

# Or build from source if not available:
cd ~
git clone https://github.com/SCADA-LTS/Snap7.git
cd Snap7/build/unix
sudo make -f arm_v7_linux.mk install
sudo ldconfig
```

## Step 5: Clone the HAI Testbed

```bash
cd ~
git clone https://github.com/huuhuannt1998/virtual_hai_testbed.git
cd virtual_hai_testbed
```

## Step 6: Configure PLC Connection

Edit the config file:

```bash
nano hai/config.py
```

Update the PLC IP if needed:
```python
PLC_IP = "192.168.0.1"   # Your PLC IP
RACK = 0
SLOT = 0
```

## Step 7: Test PLC Connection

```bash
source ~/hai_venv/bin/activate
cd ~/virtual_hai_testbed
python test_plc_connection.py
```

Expected output:
```
✓ Connected to PLC at 192.168.0.1
✓ DB read successful
✓ All tests passed!
```

## Step 8: Run Attack Simulation

### Single Process Attack

```bash
# Attack P1 Boiler
python run_multi_attack.py p1 30

# Attack P2 Turbine
python run_multi_attack.py p2 30

# Attack P3 Water Treatment
python run_multi_attack.py p3 30

# Attack P4 HIL
python run_multi_attack.py p4 30
```

### Attack All Processes

```bash
python run_multi_attack.py all 60
```

### Monitor Attack in Real-Time

```bash
python verify_attack_on_plc.py
```

## Step 9: Automate Attacks (Optional)

Create a cron job or systemd service to run attacks automatically:

```bash
# Create attack script
nano ~/run_attack.sh
```

```bash
#!/bin/bash
source ~/hai_venv/bin/activate
cd ~/virtual_hai_testbed
python run_multi_attack.py p1 30
```

```bash
chmod +x ~/run_attack.sh
```

## Troubleshooting

### Cannot connect to PLC

1. Check network connectivity: `ping 192.168.0.1`
2. Verify PLC is in RUN mode
3. Check PUT/GET is enabled in TIA Portal:
   - Device properties → Protection & Security → Connection mechanisms
   - Enable "Permit access with PUT/GET"
4. Ensure PLC firmware is compatible (V4.2 works best with Snap7)

### Snap7 library not found

```bash
# Check if library is installed
ldconfig -p | grep snap7

# If not found, install manually
sudo apt install libsnap7-1
```

### Permission denied

```bash
# Run with sudo if needed (not recommended for production)
sudo python run_multi_attack.py p1 30
```

## Security Considerations

⚠️ **WARNING**: This setup is for **research and testing only**.

- Only use on isolated test networks
- Never deploy on production ICS systems
- Ensure physical safety interlocks are in place
- Document all testing activities

## Advanced: Headless Attack Mode

For automated/scheduled attacks without a display:

```bash
# Run in background with logging
nohup python run_multi_attack.py all 300 > attack_log.txt 2>&1 &

# Check status
tail -f attack_log.txt

# Stop attack
pkill -f run_multi_attack
```

## Network Capture (Optional)

Capture attack traffic for analysis:

```bash
sudo apt install tcpdump wireshark-common
sudo tcpdump -i eth0 -w attack_capture.pcap port 102
```

This captures S7 protocol traffic (port 102) for later analysis in Wireshark.
