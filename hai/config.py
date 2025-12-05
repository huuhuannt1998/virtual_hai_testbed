"""Configuration for the virtual HAI testbed.

This defines the PLC connection settings and the DB layout for all HAI tags.
You must update DB_HAI_NUM and verify OFFSETS match your TIA Portal DB layout.

HAI Dataset Reference: https://github.com/icsdataset/hai
"""

from typing import Dict

# =============================================================================
# PLC Connection Settings
# =============================================================================

PLC_IP = "192.168.0.1"   # TODO: Set to your PLCSIM Advanced or PLC IP
RACK = 0                  # For S7-1200/1500, usually 0
SLOT = 0                  # For S7-1200: usually 0 or 1; for S7-1500: usually 1

# =============================================================================
# Data Block Configuration
# =============================================================================

# Global data block number used for HAI tags
# Create a DB in TIA Portal with this number (e.g., DB2)
DB_HAI_NUM = 2

# =============================================================================
# Byte Offsets for HAI Tags in DB_HAI
#
# IMPORTANT: Update these offsets to match your TIA Portal DB layout!
# In TIA Portal, check the "Offset" column in the DB editor.
#
# All tags are stored as REAL (4 bytes each).
# Layout: P1 tags (48), P2 tags (24), P3 tags (7), P4 tags (11) = 90 tags
# Total: 360 bytes
# =============================================================================

OFFSETS: Dict[str, int] = {
    # === P1: Boiler Process (48 tags, offsets 0-188) ===
    "P1_B2004": 0,
    "P1_B2016": 4,
    "P1_B3004": 8,
    "P1_B3005": 12,
    "P1_B4002": 16,
    "P1_B4005": 20,
    "P1_B400B": 24,
    "P1_B4022": 28,
    "P1_FCV01D": 32,
    "P1_FCV01Z": 36,
    "P1_FCV02D": 40,
    "P1_FCV02Z": 44,
    "P1_FCV03D": 48,
    "P1_FCV03Z": 52,
    "P1_FT01": 56,
    "P1_FT01Z": 60,
    "P1_FT02": 64,
    "P1_FT02Z": 68,
    "P1_FT03": 72,
    "P1_FT03Z": 76,
    "P1_LCV01D": 80,
    "P1_LCV01Z": 84,
    "P1_LIT01": 88,
    "P1_PCV01D": 92,
    "P1_PCV01Z": 96,
    "P1_PCV02D": 100,
    "P1_PCV02Z": 104,
    "P1_PIT01": 108,
    "P1_PIT01_HH": 112,
    "P1_PIT02": 116,
    "P1_PP01AD": 120,
    "P1_PP01AR": 124,
    "P1_PP01BD": 128,
    "P1_PP01BR": 132,
    "P1_PP02D": 136,
    "P1_PP02R": 140,
    "P1_PP04": 144,
    "P1_PP04SP": 148,
    "P1_SOL01D": 152,
    "P1_SOL03D": 156,
    "P1_STSP": 160,
    "P1_TIT01": 164,
    "P1_TIT02": 168,
    "P1_TIT03": 172,
    
    # === P2: Turbine Process (24 tags, offsets 176-268) ===
    "P2_24Vdc": 176,
    "P2_ATSW_Lamp": 180,
    "P2_AutoGO": 184,
    "P2_AutoSD": 188,
    "P2_Emerg": 192,
    "P2_MASW": 196,
    "P2_MASW_Lamp": 200,
    "P2_ManualGO": 204,
    "P2_ManualSD": 208,
    "P2_OnOff": 212,
    "P2_RTR": 216,
    "P2_SCO": 220,
    "P2_SCST": 224,
    "P2_SIT01": 228,
    "P2_TripEx": 232,
    "P2_VIBTR01": 236,
    "P2_VIBTR02": 240,
    "P2_VIBTR03": 244,
    "P2_VIBTR04": 248,
    "P2_VT01": 252,
    "P2_VTR01": 256,
    "P2_VTR02": 260,
    "P2_VTR03": 264,
    "P2_VTR04": 268,
    
    # === P3: Water Treatment Process (7 tags, offsets 272-296) ===
    "P3_FIT01": 272,
    "P3_LCP01D": 276,
    "P3_LCV01D": 280,
    "P3_LH01": 284,
    "P3_LL01": 288,
    "P3_LIT01": 292,
    "P3_PIT01": 296,
    
    # === P4: HIL Simulation (11 tags, offsets 300-340) ===
    "P4_HT_FD": 300,
    "P4_HT_PO": 304,
    "P4_HT_PS": 308,
    "P4_LD": 312,
    "P4_ST_FD": 316,
    "P4_ST_GOV": 320,
    "P4_ST_LD": 324,
    "P4_ST_PO": 328,
    "P4_ST_PS": 332,
    "P4_ST_PT01": 336,
    "P4_ST_TT01": 340,
}

# Total DB size in bytes (90 tags * 4 bytes = 360)
DB_SIZE_BYTES = 344

# =============================================================================
# Simulation Parameters
# =============================================================================

SIM_DT_SEC = 1.0      # Simulation time-step (1 Hz like HAI dataset)
LOG_EVERY = 1         # Log every N steps

# =============================================================================
# Logging Configuration
# =============================================================================

LOG_ENABLED = True
LOG_FILE = "hai_virtual_log.csv"
