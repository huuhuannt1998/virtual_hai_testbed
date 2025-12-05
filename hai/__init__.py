"""Virtual HAI Testbed Package.

Simulates the four processes of the HAI (HIL-based Augmented ICS) dataset:
- P1: Boiler Process (Emerson Ovation DCS)
- P2: Turbine Process (GE Mark VIe DCS)
- P3: Water Treatment Process (Siemens S7-300 PLC)
- P4: HIL Power Grid Simulation (dSPACE SCALEXIO + S7-1500)

This testbed provides simulated I/O signals for use with a real PLC
(Siemens CPU 1212FC DC/DC/RLY) running the control logic in TIA Portal.

Reference: https://github.com/icsdataset/hai

Usage:
    python -m hai.runner

Modules:
    config: PLC connection settings and DB layout
    tags: Complete HAI tag definitions (86 tags)
    io: PLC communication via Snap7
    plant: Physical process simulation
    runner: Main simulation loop with CSV logging
    attacks: Attack simulation (bias injection, replay, DoS, etc.)
"""

from .config import PLC_IP, DB_HAI_NUM, SIM_DT_SEC
from .tags import ALL_TAGS, P1_TAGS, P2_TAGS, P3_TAGS, P4_TAGS, TagType
from .io import PLCClient, ensure_connected
from .plant import VirtualHAIPlant, PlantState
from .runner import main as run
from .attacks import (
    AttackSimulator,
    AttackConfig,
    AttackType,
    AttackScenarioManager,
    AttackAwarePlant,
)

__version__ = "0.2.0"

__all__ = [
    # Config
    "PLC_IP",
    "DB_HAI_NUM", 
    "SIM_DT_SEC",
    # Tags
    "ALL_TAGS",
    "P1_TAGS",
    "P2_TAGS", 
    "P3_TAGS",
    "P4_TAGS",
    "TagType",
    # IO
    "PLCClient",
    "ensure_connected",
    # Plant
    "VirtualHAIPlant",
    "PlantState",
    # Runner
    "run",
]