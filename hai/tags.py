"""HAI Dataset Tag Definitions (HAI 22.04).

Complete list of 86 tags from the HAI dataset organized by process.
Each tag is classified as:
- SENSOR: Measured value from the physical process (written by plant simulation)
- ACTUATOR: Control output from PLC (read by plant simulation)
- SETPOINT: Setpoint/parameter (can be written by operator or HIL)
- STATUS: Status/mode indicator

Reference: https://github.com/icsdataset/hai
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any


class TagType(Enum):
    SENSOR = "sensor"       # Physical measurement (plant -> PLC)
    ACTUATOR = "actuator"   # Control output (PLC -> plant)
    SETPOINT = "setpoint"   # Setpoint/reference value
    STATUS = "status"       # Status/mode indicators
    PARAMETER = "parameter" # System parameters/thresholds


@dataclass
class TagInfo:
    """Metadata for a single HAI tag."""
    name: str
    tag_type: TagType
    description: str
    unit: str = ""
    min_val: float = 0.0
    max_val: float = 100.0
    default: float = 0.0


# =============================================================================
# P1: BOILER PROCESS (Emerson Ovation DCS)
# Water-to-water heat transfer at low pressure and moderate temperature
# =============================================================================

P1_TAGS: Dict[str, TagInfo] = {
    # Setpoints
    "P1_B2004": TagInfo("P1_B2004", TagType.SETPOINT, "Setpoint (unknown)", "", 0, 100, 50),
    "P1_B2016": TagInfo("P1_B2016", TagType.SETPOINT, "Pressure setpoint", "bar", 0, 20, 10),
    "P1_B3004": TagInfo("P1_B3004", TagType.SETPOINT, "Level setpoint", "%", 0, 100, 50),
    "P1_B3005": TagInfo("P1_B3005", TagType.SETPOINT, "Flow setpoint (FT03)", "L/min", 0, 100, 30),
    "P1_B4002": TagInfo("P1_B4002", TagType.SETPOINT, "Temperature setpoint 2", "°C", 0, 100, 60),
    "P1_B4005": TagInfo("P1_B4005", TagType.SETPOINT, "Temperature setpoint 5", "°C", 0, 100, 60),
    "P1_B400B": TagInfo("P1_B400B", TagType.SETPOINT, "Temperature setpoint B", "°C", 0, 100, 60),
    "P1_B4022": TagInfo("P1_B4022", TagType.SETPOINT, "Temperature setpoint (TIT01)", "°C", 0, 100, 60),

    # Flow Control Valves - Demand (actuator output 0-100%)
    "P1_FCV01D": TagInfo("P1_FCV01D", TagType.ACTUATOR, "Flow control valve 1 demand", "%", 0, 100, 0),
    "P1_FCV02D": TagInfo("P1_FCV02D", TagType.ACTUATOR, "Flow control valve 2 demand", "%", 0, 100, 0),
    "P1_FCV03D": TagInfo("P1_FCV03D", TagType.ACTUATOR, "Flow control valve 3 demand", "%", 0, 100, 0),

    # Flow Control Valves - Position feedback (sensor)
    "P1_FCV01Z": TagInfo("P1_FCV01Z", TagType.SENSOR, "Flow control valve 1 position", "%", 0, 100, 0),
    "P1_FCV02Z": TagInfo("P1_FCV02Z", TagType.SENSOR, "Flow control valve 2 position", "%", 0, 100, 0),
    "P1_FCV03Z": TagInfo("P1_FCV03Z", TagType.SENSOR, "Flow control valve 3 position", "%", 0, 100, 0),

    # Flow Transmitters (sensors)
    "P1_FT01": TagInfo("P1_FT01", TagType.SENSOR, "Flow transmitter 1", "L/min", 0, 100, 0),
    "P1_FT01Z": TagInfo("P1_FT01Z", TagType.SENSOR, "Flow transmitter 1 (scaled)", "%", 0, 100, 0),
    "P1_FT02": TagInfo("P1_FT02", TagType.SENSOR, "Flow transmitter 2 (cascade)", "L/min", 0, 100, 0),
    "P1_FT02Z": TagInfo("P1_FT02Z", TagType.SENSOR, "Flow transmitter 2 (scaled)", "%", 0, 100, 0),
    "P1_FT03": TagInfo("P1_FT03", TagType.SENSOR, "Flow transmitter 3", "L/min", 0, 100, 0),
    "P1_FT03Z": TagInfo("P1_FT03Z", TagType.SENSOR, "Flow transmitter 3 (scaled)", "%", 0, 100, 0),

    # Level Control Valve
    "P1_LCV01D": TagInfo("P1_LCV01D", TagType.ACTUATOR, "Level control valve demand", "%", 0, 100, 0),
    "P1_LCV01Z": TagInfo("P1_LCV01Z", TagType.SENSOR, "Level control valve position", "%", 0, 100, 0),

    # Level Indicator Transmitter
    "P1_LIT01": TagInfo("P1_LIT01", TagType.SENSOR, "Tank level", "%", 0, 100, 50),

    # Pressure Control Valves
    "P1_PCV01D": TagInfo("P1_PCV01D", TagType.ACTUATOR, "Pressure control valve 1 demand", "%", 0, 100, 0),
    "P1_PCV01Z": TagInfo("P1_PCV01Z", TagType.SENSOR, "Pressure control valve 1 position", "%", 0, 100, 0),
    "P1_PCV02D": TagInfo("P1_PCV02D", TagType.ACTUATOR, "Pressure control valve 2 demand", "%", 0, 100, 0),
    "P1_PCV02Z": TagInfo("P1_PCV02Z", TagType.SENSOR, "Pressure control valve 2 position", "%", 0, 100, 0),

    # Pressure Indicator Transmitters
    "P1_PIT01": TagInfo("P1_PIT01", TagType.SENSOR, "Pressure transmitter 1", "bar", 0, 20, 5),
    "P1_PIT01_HH": TagInfo("P1_PIT01_HH", TagType.STATUS, "Pressure high-high alarm", "", 0, 1, 0),
    "P1_PIT02": TagInfo("P1_PIT02", TagType.SENSOR, "Pressure transmitter 2", "bar", 0, 20, 5),

    # Pumps
    "P1_PP01AD": TagInfo("P1_PP01AD", TagType.ACTUATOR, "Pump 1A demand", "%", 0, 100, 0),
    "P1_PP01AR": TagInfo("P1_PP01AR", TagType.SENSOR, "Pump 1A running status", "", 0, 1, 0),
    "P1_PP01BD": TagInfo("P1_PP01BD", TagType.ACTUATOR, "Pump 1B demand", "%", 0, 100, 0),
    "P1_PP01BR": TagInfo("P1_PP01BR", TagType.SENSOR, "Pump 1B running status", "", 0, 1, 0),
    "P1_PP02D": TagInfo("P1_PP02D", TagType.ACTUATOR, "Pump 2 demand", "%", 0, 100, 0),
    "P1_PP02R": TagInfo("P1_PP02R", TagType.SENSOR, "Pump 2 running status", "", 0, 1, 0),
    "P1_PP04": TagInfo("P1_PP04", TagType.ACTUATOR, "Cooling pump frequency", "%", 0, 100, 0),
    "P1_PP04SP": TagInfo("P1_PP04SP", TagType.SETPOINT, "Cooling pump frequency SP", "%", 0, 100, 50),

    # Solenoid Valves
    "P1_SOL01D": TagInfo("P1_SOL01D", TagType.ACTUATOR, "Solenoid valve 1 demand", "", 0, 1, 0),
    "P1_SOL03D": TagInfo("P1_SOL03D", TagType.ACTUATOR, "Solenoid valve 3 demand", "", 0, 1, 0),

    # Steam Setpoint
    "P1_STSP": TagInfo("P1_STSP", TagType.SETPOINT, "Steam setpoint", "", 0, 100, 50),

    # Temperature Indicator Transmitters
    "P1_TIT01": TagInfo("P1_TIT01", TagType.SENSOR, "Temperature transmitter 1", "°C", 0, 100, 60),
    "P1_TIT02": TagInfo("P1_TIT02", TagType.SENSOR, "Temperature transmitter 2", "°C", 0, 100, 60),
    "P1_TIT03": TagInfo("P1_TIT03", TagType.SENSOR, "Temperature transmitter 3 (cooling)", "°C", 0, 100, 30),
}


# =============================================================================
# P2: TURBINE PROCESS (GE Mark VIe DCS)
# Rotor kit process simulating actual rotating machine behavior
# =============================================================================

P2_TAGS: Dict[str, TagInfo] = {
    # Power and Status
    "P2_24Vdc": TagInfo("P2_24Vdc", TagType.STATUS, "24V DC power status", "V", 0, 30, 24),
    "P2_ATSW_Lamp": TagInfo("P2_ATSW_Lamp", TagType.STATUS, "Auto/Manual switch lamp", "", 0, 1, 0),
    
    # Control Commands
    "P2_AutoGO": TagInfo("P2_AutoGO", TagType.ACTUATOR, "Auto start command", "", 0, 1, 0),
    "P2_AutoSD": TagInfo("P2_AutoSD", TagType.SETPOINT, "Speed setpoint (auto)", "RPM", 0, 5000, 3000),
    "P2_Emerg": TagInfo("P2_Emerg", TagType.STATUS, "Emergency trip status", "", 0, 1, 0),
    
    # Mode Switches
    "P2_MASW": TagInfo("P2_MASW", TagType.STATUS, "Manual/Auto switch position", "", 0, 1, 1),
    "P2_MASW_Lamp": TagInfo("P2_MASW_Lamp", TagType.STATUS, "Manual/Auto switch lamp", "", 0, 1, 0),
    "P2_ManualGO": TagInfo("P2_ManualGO", TagType.ACTUATOR, "Manual start command", "", 0, 1, 0),
    "P2_ManualSD": TagInfo("P2_ManualSD", TagType.ACTUATOR, "Manual shutdown command", "", 0, 1, 0),
    "P2_OnOff": TagInfo("P2_OnOff", TagType.STATUS, "Turbine on/off status", "", 0, 1, 0),
    
    # Overspeed Trip
    "P2_RTR": TagInfo("P2_RTR", TagType.PARAMETER, "Overspeed trip threshold", "RPM", 0, 5000, 3600),
    
    # Speed Controller
    "P2_SCO": TagInfo("P2_SCO", TagType.ACTUATOR, "Speed controller output", "%", 0, 100, 0),
    "P2_SCST": TagInfo("P2_SCST", TagType.STATUS, "Speed controller status", "", 0, 1, 0),
    
    # Speed Indicator Transmitter
    "P2_SIT01": TagInfo("P2_SIT01", TagType.SENSOR, "Turbine speed", "RPM", 0, 5000, 0),
    
    # Trip Controls
    "P2_TripEx": TagInfo("P2_TripEx", TagType.ACTUATOR, "Trip reset/execute", "", 0, 1, 0),
    
    # Vibration Transmitters
    "P2_VIBTR01": TagInfo("P2_VIBTR01", TagType.SENSOR, "Vibration sensor 1", "mm/s", 0, 50, 0),
    "P2_VIBTR02": TagInfo("P2_VIBTR02", TagType.SENSOR, "Vibration sensor 2", "mm/s", 0, 50, 0),
    "P2_VIBTR03": TagInfo("P2_VIBTR03", TagType.SENSOR, "Vibration sensor 3", "mm/s", 0, 50, 0),
    "P2_VIBTR04": TagInfo("P2_VIBTR04", TagType.SENSOR, "Vibration sensor 4", "mm/s", 0, 50, 0),
    
    # Valve
    "P2_VT01": TagInfo("P2_VT01", TagType.ACTUATOR, "Steam valve position", "%", 0, 100, 0),
    
    # Vibration Trip Thresholds
    "P2_VTR01": TagInfo("P2_VTR01", TagType.PARAMETER, "Vibration trip threshold 1", "mm/s", 0, 50, 25),
    "P2_VTR02": TagInfo("P2_VTR02", TagType.PARAMETER, "Vibration trip threshold 2", "mm/s", 0, 50, 25),
    "P2_VTR03": TagInfo("P2_VTR03", TagType.PARAMETER, "Vibration trip threshold 3", "mm/s", 0, 50, 25),
    "P2_VTR04": TagInfo("P2_VTR04", TagType.PARAMETER, "Vibration trip threshold 4", "mm/s", 0, 50, 25),
}


# =============================================================================
# P3: WATER TREATMENT PROCESS (Siemens S7-300 PLC)
# Pumping water to upper reservoir and releasing to lower reservoir
# =============================================================================

P3_TAGS: Dict[str, TagInfo] = {
    # Flow Indicator Transmitter
    "P3_FIT01": TagInfo("P3_FIT01", TagType.SENSOR, "Flow rate", "L/min", 0, 100, 0),
    
    # Pump Control
    "P3_LCP01D": TagInfo("P3_LCP01D", TagType.ACTUATOR, "Pump command", "", 0, 1, 0),
    
    # Level Control Valve
    "P3_LCV01D": TagInfo("P3_LCV01D", TagType.ACTUATOR, "Level control valve demand", "%", 0, 100, 0),
    
    # Level Thresholds
    "P3_LH01": TagInfo("P3_LH01", TagType.PARAMETER, "High level threshold", "%", 0, 100, 80),
    "P3_LL01": TagInfo("P3_LL01", TagType.PARAMETER, "Low level threshold", "%", 0, 100, 20),
    
    # Level Indicator Transmitter
    "P3_LIT01": TagInfo("P3_LIT01", TagType.SENSOR, "Tank level", "%", 0, 100, 50),
    
    # Pressure Indicator Transmitter
    "P3_PIT01": TagInfo("P3_PIT01", TagType.SENSOR, "Pressure", "bar", 0, 10, 2),
}


# =============================================================================
# P4: HIL SIMULATION (dSPACE SCALEXIO + Siemens S7-1500)
# Steam-turbine and pumped-storage hydropower generation simulation
# =============================================================================

P4_TAGS: Dict[str, TagInfo] = {
    # Hydro Turbine (HT)
    "P4_HT_FD": TagInfo("P4_HT_FD", TagType.SENSOR, "Hydro turbine flow rate", "m³/s", 0, 100, 0),
    "P4_HT_PO": TagInfo("P4_HT_PO", TagType.SENSOR, "Hydro turbine power output", "MW", 0, 100, 0),
    "P4_HT_PS": TagInfo("P4_HT_PS", TagType.SETPOINT, "Hydro turbine power setpoint", "MW", 0, 100, 50),
    
    # Load Demand
    "P4_LD": TagInfo("P4_LD", TagType.SETPOINT, "Load demand", "MW", 0, 200, 100),
    
    # Steam Turbine (ST)
    "P4_ST_FD": TagInfo("P4_ST_FD", TagType.SENSOR, "Steam turbine flow rate", "kg/s", 0, 100, 0),
    "P4_ST_GOV": TagInfo("P4_ST_GOV", TagType.ACTUATOR, "Steam turbine governor", "%", 0, 100, 50),
    "P4_ST_LD": TagInfo("P4_ST_LD", TagType.SENSOR, "Steam turbine load", "MW", 0, 100, 0),
    "P4_ST_PO": TagInfo("P4_ST_PO", TagType.SENSOR, "Steam turbine power output", "MW", 0, 100, 0),
    "P4_ST_PS": TagInfo("P4_ST_PS", TagType.SETPOINT, "Steam turbine power setpoint", "MW", 0, 100, 50),
    "P4_ST_PT01": TagInfo("P4_ST_PT01", TagType.SENSOR, "Steam pressure", "bar", 0, 50, 10),
    "P4_ST_TT01": TagInfo("P4_ST_TT01", TagType.SENSOR, "Steam temperature", "°C", 0, 500, 200),
}


# =============================================================================
# Combined tag dictionary
# =============================================================================

ALL_TAGS: Dict[str, TagInfo] = {
    **P1_TAGS,
    **P2_TAGS,
    **P3_TAGS,
    **P4_TAGS,
}

# Convenience lists by process
P1_TAG_NAMES = list(P1_TAGS.keys())
P2_TAG_NAMES = list(P2_TAGS.keys())
P3_TAG_NAMES = list(P3_TAGS.keys())
P4_TAG_NAMES = list(P4_TAGS.keys())
ALL_TAG_NAMES = list(ALL_TAGS.keys())

# Tags that the plant simulation writes (sensors + valve positions)
PLANT_OUTPUT_TAGS = [
    name for name, info in ALL_TAGS.items()
    if info.tag_type == TagType.SENSOR
]

# Tags that the plant simulation reads (actuators from PLC)
PLANT_INPUT_TAGS = [
    name for name, info in ALL_TAGS.items()
    if info.tag_type == TagType.ACTUATOR
]

# Setpoints and parameters
SETPOINT_TAGS = [
    name for name, info in ALL_TAGS.items()
    if info.tag_type in (TagType.SETPOINT, TagType.PARAMETER)
]


def get_default_values() -> Dict[str, float]:
    """Return a dictionary with all tags set to their default values."""
    return {name: info.default for name, info in ALL_TAGS.items()}


def print_tag_summary():
    """Print a summary of all tags by process."""
    for process, tags in [("P1 (Boiler)", P1_TAGS), 
                          ("P2 (Turbine)", P2_TAGS),
                          ("P3 (Water)", P3_TAGS), 
                          ("P4 (HIL)", P4_TAGS)]:
        print(f"\n{process}: {len(tags)} tags")
        for name, info in tags.items():
            print(f"  {name:15s} [{info.tag_type.value:8s}] {info.description}")


if __name__ == "__main__":
    print(f"Total HAI tags: {len(ALL_TAGS)}")
    print(f"  Plant outputs (sensors): {len(PLANT_OUTPUT_TAGS)}")
    print(f"  Plant inputs (actuators): {len(PLANT_INPUT_TAGS)}")
    print(f"  Setpoints/parameters: {len(SETPOINT_TAGS)}")
    print_tag_summary()
