# Virtual HAI Testbed for Siemens CPU 1212FC

A Python-based virtual testbed that simulates the **HAI (HIL-based Augmented ICS) Security Dataset** physical processes for use with a Siemens PLC running control logic in TIA Portal.

## Overview

This testbed simulates **86 process variables** across four interconnected processes:

| Process | Description | Controller (Original) | Tags |
|---------|-------------|----------------------|------|
| **P1** | Boiler - Heat transfer, level, pressure, temperature | Emerson Ovation DCS | 48 |
| **P2** | Turbine - Rotor speed, vibration, trips | GE Mark VIe DCS | 24 |
| **P3** | Water Treatment - Tank level, pump, flow | Siemens S7-300 | 7 |
| **P4** | HIL Simulation - Power grid coupling | dSPACE SCALEXIO | 11 |

**Reference:** [HAI Dataset (GitHub)](https://github.com/icsdataset/hai)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TIA Portal / PLC                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  DB_HAI (Data Block with all 90 REAL tags)          │   │
│  │  - Sensors: Written by Python plant simulation      │   │
│  │  - Actuators: Written by PLC control logic          │   │
│  │  - Setpoints: Written by operator/HMI               │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↕ Snap7                           │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                  Python Plant Simulation                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │P1 Boiler │ │P2 Turbine│ │ P3 Water │ │ P4 HIL   │       │
│  │ - Level  │ │ - Speed  │ │ - Level  │ │ - Power  │       │
│  │ - Temp   │ │ - Vibr.  │ │ - Flow   │ │ - Load   │       │
│  │ - Press  │ │ - Trips  │ │ - Press  │ │ - Freq   │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│                            ↓                                 │
│              CSV Logger (HAI-style dataset)                  │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
hai/
├── __init__.py     # Package exports
├── config.py       # PLC connection and DB layout configuration
├── tags.py         # Complete HAI tag definitions (86 tags)
├── io.py           # PLC communication via python-snap7
├── plant.py        # Physical process simulation (P1-P4)
└── runner.py       # Main simulation loop with CSV logging
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure TIA Portal

1. Create a new project for CPU 1212FC DC/DC/RLY
2. Create a Data Block `DB_HAI` (DB1) with **90 REAL variables**
3. **Disable "Optimized block access"** for the DB (required for Snap7)
4. Copy the tag names from `hai/tags.py` or use the offset table in `hai/config.py`
5. Implement your control logic (PID loops, safety interlocks)
6. Download to PLC or start PLCSIM Advanced

### 3. Configure Python

Edit `hai/config.py`:

```python
PLC_IP = "192.168.0.1"   # Your PLC/PLCSIM IP address
RACK = 0                  # Usually 0 for S7-1200
SLOT = 1                  # Usually 1 for S7-1200
DB_HAI_NUM = 1           # Your DB number
```

### 4. Run the Testbed

```bash
python -m hai.runner
```

This will:
1. Connect to the PLC via Snap7
2. Run the plant simulation at 1 Hz
3. Write sensor values to DB_HAI (temperatures, levels, pressures, speeds)
4. Read actuator values from DB_HAI (valve positions, pump commands)
5. Log all data to `hai_virtual_log.csv`

## Tag Categories

| Type | Description | Direction | Example |
|------|-------------|-----------|---------|
| **SENSOR** | Physical measurements | Plant → PLC | P1_PIT01 (pressure) |
| **ACTUATOR** | Control outputs | PLC → Plant | P1_FCV01D (valve demand) |
| **SETPOINT** | Reference values | Operator/HMI | P1_B2016 (pressure SP) |
| **PARAMETER** | Thresholds/limits | Configuration | P2_RTR (overspeed trip) |
| **STATUS** | Boolean states | Both | P2_Emerg (trip status) |

## Key Tags

### P1 Boiler
- `P1_PIT01`, `P1_PIT02` - Pressure transmitters
- `P1_LIT01` - Level indicator
- `P1_TIT01`, `P1_TIT02`, `P1_TIT03` - Temperature transmitters
- `P1_FCV01D`, `P1_FCV02D`, `P1_FCV03D` - Flow control valve demands
- `P1_PCV01D`, `P1_PCV02D` - Pressure control valve demands
- `P1_LCV01D` - Level control valve demand

### P2 Turbine
- `P2_SIT01` - Speed indicator (RPM)
- `P2_VIBTR01-04` - Vibration sensors
- `P2_SCO` - Speed controller output
- `P2_VT01` - Steam valve position
- `P2_Emerg` - Emergency trip status
- `P2_RTR` - Overspeed trip threshold

### P3 Water Treatment
- `P3_LIT01` - Tank level
- `P3_FIT01` - Flow rate
- `P3_LCP01D` - Pump command
- `P3_LCV01D` - Valve demand

### P4 HIL
- `P4_ST_PO` - Steam turbine power output
- `P4_HT_PO` - Hydro turbine power output
- `P4_LD` - Load demand
- `P4_ST_GOV` - Governor position

## Output: HAI-Style Dataset

The testbed generates a CSV file matching the HAI dataset format:

```csv
timestamp,P1_B2004,P1_B2016,...,P4_ST_TT01,Attack
2024-01-01 12:00:00,50.0,10.0,...,200.0,0
2024-01-01 12:00:01,50.1,10.0,...,200.1,0
...
```

## Extending the Testbed

### Adding Attacks

Create attack scenarios by modifying sensor values in the simulation:

```python
# Example: Bias attack on P1 level sensor
plant.state.p1.level += 10.0  # Add 10% bias
```

### Adjusting Process Dynamics

Modify the physics in `hai/plant.py`:
- `VALVE_TAU` - Valve response time constant
- `NOISE_LEVEL` - Sensor noise amplitude
- Individual `_step_pX_*` methods for process-specific dynamics

## License

MIT License - See LICENSE file

## References

- [HAI Dataset on GitHub](https://github.com/icsdataset/hai)
- [HAI Dataset Technical Details (PDF)](https://github.com/icsdataset/hai/blob/master/hai_dataset_technical_details.pdf)
- [python-snap7 Documentation](https://python-snap7.readthedocs.io/)
