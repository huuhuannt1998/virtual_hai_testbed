"""Virtual HAI Plant Simulation.

This module simulates the physical dynamics of all four HAI processes:
- P1: Boiler (heat transfer, level, pressure, temperature)
- P2: Turbine (speed, vibration)
- P3: Water Treatment (tank level, flow)
- P4: HIL (power grid coupling)

The plant simulation:
1. READS actuator commands from the PLC (valve positions, pump commands, etc.)
2. Simulates physical dynamics based on these commands
3. WRITES sensor values back to the PLC (temperatures, pressures, levels, etc.)

The PLC (TIA Portal) runs the control logic (PID loops, safety interlocks).

Reference: https://github.com/icsdataset/hai (HAI 22.04 dataset)
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .io import PLCClient

from .config import SIM_DT_SEC
from .tags import ALL_TAGS, TagType


@dataclass
class P1BoilerState:
    """State variables for P1 Boiler Process."""
    # Tank level (0-100%)
    level: float = 50.0
    # Temperatures (°C)
    temp1: float = 60.0   # TIT01 - main process temp
    temp2: float = 55.0   # TIT02 - secondary temp
    temp3: float = 30.0   # TIT03 - cooling water temp
    # Pressures (bar)
    pressure1: float = 5.0  # PIT01
    pressure2: float = 4.5  # PIT02
    # Flows (L/min) - derived from valve positions
    flow1: float = 0.0   # FT01
    flow2: float = 0.0   # FT02
    flow3: float = 0.0   # FT03
    # Valve positions (0-100%) - follows demand with lag
    fcv01_pos: float = 0.0
    fcv02_pos: float = 0.0
    fcv03_pos: float = 0.0
    lcv01_pos: float = 0.0
    pcv01_pos: float = 0.0
    pcv02_pos: float = 0.0
    # Pump status
    pp01a_running: bool = False
    pp01b_running: bool = False
    pp02_running: bool = False


@dataclass
class P2TurbineState:
    """State variables for P2 Turbine Process."""
    # Turbine speed (RPM)
    speed: float = 0.0
    # Vibration levels (mm/s) - 4 sensors
    vibration: list = field(default_factory=lambda: [0.5, 0.5, 0.5, 0.5])
    # Valve position (0-100%)
    valve_pos: float = 0.0
    # Status flags
    is_running: bool = False
    is_tripped: bool = False
    # 24V DC supply
    supply_24v: float = 24.0


@dataclass
class P3WaterState:
    """State variables for P3 Water Treatment Process."""
    # Tank level (0-100%)
    level: float = 50.0
    # Flow rate (L/min)
    flow: float = 0.0
    # Pressure (bar)
    pressure: float = 2.0
    # Valve position (0-100%)
    valve_pos: float = 0.0


@dataclass
class P4HILState:
    """State variables for P4 HIL Power Grid Simulation."""
    # Steam turbine
    st_flow: float = 0.0      # Steam flow (kg/s)
    st_power: float = 0.0     # Power output (MW)
    st_load: float = 0.0      # Load (MW)
    st_pressure: float = 10.0  # Steam pressure (bar)
    st_temp: float = 200.0     # Steam temperature (°C)
    # Hydro turbine
    ht_flow: float = 0.0      # Water flow (m³/s)
    ht_power: float = 0.0     # Power output (MW)
    # Grid
    load_demand: float = 100.0  # Total load demand (MW)


@dataclass
class PlantState:
    """Complete state of the virtual HAI plant."""
    p1: P1BoilerState = field(default_factory=P1BoilerState)
    p2: P2TurbineState = field(default_factory=P2TurbineState)
    p3: P3WaterState = field(default_factory=P3WaterState)
    p4: P4HILState = field(default_factory=P4HILState)


class VirtualHAIPlant:
    """Simulates the physical dynamics of all HAI processes.
    
    This class:
    1. Reads actuator commands from the PLC via Snap7
    2. Simulates physical process dynamics
    3. Writes sensor values back to the PLC
    
    The PLC (in TIA Portal) handles all control logic.
    """
    
    # Valve dynamics time constant (seconds)
    VALVE_TAU = 2.0
    # Noise levels for sensors
    NOISE_LEVEL = 0.01
    
    def __init__(self, plc: "PLCClient", state: Optional[PlantState] = None):
        self.plc = plc
        self.state = state or PlantState()
        self.dt = SIM_DT_SEC
        self._step_count = 0
    
    def step(self):
        """Advance the plant simulation by one time step."""
        self._step_count += 1
        
        # Read all actuator commands from PLC
        actuators = self._read_actuators()
        
        # Simulate each process
        self._step_p1_boiler(actuators)
        self._step_p2_turbine(actuators)
        self._step_p3_water(actuators)
        self._step_p4_hil(actuators)
        
        # Write all sensor values to PLC
        self._write_sensors()
    
    def _read_actuators(self) -> Dict[str, float]:
        """Read all actuator commands from PLC."""
        actuators = {}
        for name, info in ALL_TAGS.items():
            if info.tag_type == TagType.ACTUATOR:
                try:
                    actuators[name] = self.plc.read_tag_real(name)
                except Exception:
                    actuators[name] = info.default
        return actuators
    
    def _read_setpoints(self) -> Dict[str, float]:
        """Read setpoints and parameters from PLC."""
        setpoints = {}
        for name, info in ALL_TAGS.items():
            if info.tag_type in (TagType.SETPOINT, TagType.PARAMETER):
                try:
                    setpoints[name] = self.plc.read_tag_real(name)
                except Exception:
                    setpoints[name] = info.default
        return setpoints
    
    def _write_sensors(self):
        """Write all sensor values to PLC."""
        p1, p2, p3, p4 = self.state.p1, self.state.p2, self.state.p3, self.state.p4
        
        # === P1 Boiler Sensors ===
        self.plc.write_tag_real("P1_LIT01", self._add_noise(p1.level))
        self.plc.write_tag_real("P1_TIT01", self._add_noise(p1.temp1))
        self.plc.write_tag_real("P1_TIT02", self._add_noise(p1.temp2))
        self.plc.write_tag_real("P1_TIT03", self._add_noise(p1.temp3))
        self.plc.write_tag_real("P1_PIT01", self._add_noise(p1.pressure1))
        self.plc.write_tag_real("P1_PIT02", self._add_noise(p1.pressure2))
        self.plc.write_tag_real("P1_PIT01_HH", 1.0 if p1.pressure1 > 15.0 else 0.0)
        
        # Flow sensors
        self.plc.write_tag_real("P1_FT01", self._add_noise(p1.flow1))
        self.plc.write_tag_real("P1_FT01Z", self._add_noise(p1.flow1))  # Scaled same
        self.plc.write_tag_real("P1_FT02", self._add_noise(p1.flow2))
        self.plc.write_tag_real("P1_FT02Z", self._add_noise(p1.flow2))
        self.plc.write_tag_real("P1_FT03", self._add_noise(p1.flow3))
        self.plc.write_tag_real("P1_FT03Z", self._add_noise(p1.flow3))
        
        # Valve position feedback
        self.plc.write_tag_real("P1_FCV01Z", p1.fcv01_pos)
        self.plc.write_tag_real("P1_FCV02Z", p1.fcv02_pos)
        self.plc.write_tag_real("P1_FCV03Z", p1.fcv03_pos)
        self.plc.write_tag_real("P1_LCV01Z", p1.lcv01_pos)
        self.plc.write_tag_real("P1_PCV01Z", p1.pcv01_pos)
        self.plc.write_tag_real("P1_PCV02Z", p1.pcv02_pos)
        
        # Pump running status
        self.plc.write_tag_real("P1_PP01AR", 1.0 if p1.pp01a_running else 0.0)
        self.plc.write_tag_real("P1_PP01BR", 1.0 if p1.pp01b_running else 0.0)
        self.plc.write_tag_real("P1_PP02R", 1.0 if p1.pp02_running else 0.0)
        
        # === P2 Turbine Sensors ===
        self.plc.write_tag_real("P2_SIT01", self._add_noise(p2.speed))
        self.plc.write_tag_real("P2_VIBTR01", self._add_noise(p2.vibration[0]))
        self.plc.write_tag_real("P2_VIBTR02", self._add_noise(p2.vibration[1]))
        self.plc.write_tag_real("P2_VIBTR03", self._add_noise(p2.vibration[2]))
        self.plc.write_tag_real("P2_VIBTR04", self._add_noise(p2.vibration[3]))
        self.plc.write_tag_real("P2_24Vdc", p2.supply_24v)
        self.plc.write_tag_real("P2_OnOff", 1.0 if p2.is_running else 0.0)
        self.plc.write_tag_real("P2_Emerg", 1.0 if p2.is_tripped else 0.0)
        
        # === P3 Water Sensors ===
        self.plc.write_tag_real("P3_LIT01", self._add_noise(p3.level))
        self.plc.write_tag_real("P3_FIT01", self._add_noise(p3.flow))
        self.plc.write_tag_real("P3_PIT01", self._add_noise(p3.pressure))
        
        # === P4 HIL Sensors ===
        self.plc.write_tag_real("P4_ST_FD", self._add_noise(p4.st_flow))
        self.plc.write_tag_real("P4_ST_PO", self._add_noise(p4.st_power))
        self.plc.write_tag_real("P4_ST_LD", self._add_noise(p4.st_load))
        self.plc.write_tag_real("P4_ST_PT01", self._add_noise(p4.st_pressure))
        self.plc.write_tag_real("P4_ST_TT01", self._add_noise(p4.st_temp))
        self.plc.write_tag_real("P4_HT_FD", self._add_noise(p4.ht_flow))
        self.plc.write_tag_real("P4_HT_PO", self._add_noise(p4.ht_power))
    
    def _step_p1_boiler(self, actuators: Dict[str, float]):
        """Simulate P1 Boiler dynamics."""
        p1 = self.state.p1
        dt = self.dt
        
        # Read actuator demands
        fcv01_demand = actuators.get("P1_FCV01D", 0.0)
        fcv02_demand = actuators.get("P1_FCV02D", 0.0)
        fcv03_demand = actuators.get("P1_FCV03D", 0.0)
        lcv01_demand = actuators.get("P1_LCV01D", 0.0)
        pcv01_demand = actuators.get("P1_PCV01D", 0.0)
        pcv02_demand = actuators.get("P1_PCV02D", 0.0)
        pp01a_demand = actuators.get("P1_PP01AD", 0.0)
        pp01b_demand = actuators.get("P1_PP01BD", 0.0)
        pp02_demand = actuators.get("P1_PP02D", 0.0)
        pp04_demand = actuators.get("P1_PP04", 0.0)
        
        # Valve dynamics (first-order lag)
        alpha = dt / (self.VALVE_TAU + dt)
        p1.fcv01_pos += alpha * (fcv01_demand - p1.fcv01_pos)
        p1.fcv02_pos += alpha * (fcv02_demand - p1.fcv02_pos)
        p1.fcv03_pos += alpha * (fcv03_demand - p1.fcv03_pos)
        p1.lcv01_pos += alpha * (lcv01_demand - p1.lcv01_pos)
        p1.pcv01_pos += alpha * (pcv01_demand - p1.pcv01_pos)
        p1.pcv02_pos += alpha * (pcv02_demand - p1.pcv02_pos)
        
        # Pump running status (on if demand > 50%)
        p1.pp01a_running = pp01a_demand > 50.0
        p1.pp01b_running = pp01b_demand > 50.0
        p1.pp02_running = pp02_demand > 50.0
        
        # Flow calculation (simplified)
        # Flow depends on valve position and pump running
        pump_factor = 1.0 if (p1.pp01a_running or p1.pp01b_running) else 0.2
        p1.flow1 = 50.0 * (p1.fcv01_pos / 100.0) * pump_factor
        p1.flow2 = 50.0 * (p1.fcv02_pos / 100.0) * pump_factor
        p1.flow3 = 50.0 * (p1.fcv03_pos / 100.0) * pump_factor
        
        # Level dynamics (mass balance)
        inflow = p1.flow1 + p1.flow2  # L/min
        outflow = 30.0 * (p1.lcv01_pos / 100.0)  # L/min
        # Tank capacity ~1000L, so change per second
        d_level = (inflow - outflow) / 1000.0 * 60.0 * dt  # % per second
        p1.level = max(0.0, min(100.0, p1.level + d_level))
        
        # Temperature dynamics (energy balance)
        # Heat input from pressure control (represents heater)
        heat_in = 5.0 * (p1.pcv01_pos + p1.pcv02_pos) / 200.0
        # Cooling from cooling loop
        cooling = 0.1 * pp04_demand / 100.0 * (p1.temp1 - 25.0)
        # Ambient losses
        ambient_loss = 0.02 * (p1.temp1 - 25.0)
        
        p1.temp1 += (heat_in - cooling - ambient_loss) * dt
        p1.temp1 = max(20.0, min(100.0, p1.temp1))
        
        # Secondary temps follow main temp
        p1.temp2 = p1.temp1 - 5.0 + random.gauss(0, 0.5)
        p1.temp3 = 25.0 + 0.2 * (p1.temp1 - 25.0) + random.gauss(0, 0.2)
        
        # Pressure dynamics (related to temperature and flow)
        target_pressure = 2.0 + 0.1 * (p1.temp1 - 40.0) + 0.05 * p1.level
        p1.pressure1 += 0.1 * (target_pressure - p1.pressure1) * dt
        p1.pressure1 = max(0.0, min(20.0, p1.pressure1))
        p1.pressure2 = p1.pressure1 * 0.9
    
    def _step_p2_turbine(self, actuators: Dict[str, float]):
        """Simulate P2 Turbine dynamics."""
        p2 = self.state.p2
        p1 = self.state.p1
        dt = self.dt
        
        # Read actuator commands
        auto_go = actuators.get("P2_AutoGO", 0.0) > 0.5
        manual_go = actuators.get("P2_ManualGO", 0.0) > 0.5
        manual_sd = actuators.get("P2_ManualSD", 0.0) > 0.5
        trip_reset = actuators.get("P2_TripEx", 0.0) > 0.5
        speed_cmd = actuators.get("P2_SCO", 0.0)  # Speed controller output
        valve_cmd = actuators.get("P2_VT01", 0.0)  # Steam valve
        speed_setpoint = actuators.get("P2_AutoSD", 3000.0)
        overspeed_trip = actuators.get("P2_RTR", 3600.0)
        vib_trips = [
            actuators.get("P2_VTR01", 25.0),
            actuators.get("P2_VTR02", 25.0),
            actuators.get("P2_VTR03", 25.0),
            actuators.get("P2_VTR04", 25.0),
        ]
        
        # Trip reset logic
        if trip_reset and p2.is_tripped:
            p2.is_tripped = False
        
        # Start/stop logic
        if (auto_go or manual_go) and not p2.is_tripped:
            p2.is_running = True
        if manual_sd:
            p2.is_running = False
        
        # Speed dynamics
        if p2.is_running and not p2.is_tripped:
            # Steam available from boiler pressure
            steam_available = p1.pressure1 / 10.0  # 0-1 factor
            
            # Valve position follows command
            alpha = dt / (self.VALVE_TAU + dt)
            p2.valve_pos += alpha * (valve_cmd - p2.valve_pos)
            
            # Torque from steam
            steam_torque = 50.0 * (p2.valve_pos / 100.0) * steam_available
            
            # Load torque (from grid coupling via P4)
            load_torque = 30.0 * (self.state.p4.st_load / 100.0)
            
            # Friction
            friction = 0.01 * p2.speed
            
            # Speed dynamics (inertia ~10 s time constant)
            inertia = 10.0
            d_speed = (steam_torque - load_torque - friction) / inertia * 60.0 * dt
            p2.speed += d_speed
            p2.speed = max(0.0, min(5000.0, p2.speed))
            
            # Vibration increases with speed and imbalance
            base_vib = 0.5 + 0.005 * p2.speed / 1000.0
            for i in range(4):
                p2.vibration[i] = base_vib + random.gauss(0, 0.2)
                p2.vibration[i] = max(0.0, p2.vibration[i])
        else:
            # Coast down
            p2.speed *= (1.0 - 0.05 * dt)
            if p2.speed < 10.0:
                p2.speed = 0.0
            p2.valve_pos *= 0.9
            for i in range(4):
                p2.vibration[i] = 0.5 + random.gauss(0, 0.1)
        
        # Check trip conditions
        if p2.speed > overspeed_trip:
            p2.is_tripped = True
            p2.is_running = False
        
        for i, vib in enumerate(p2.vibration):
            if vib > vib_trips[i]:
                p2.is_tripped = True
                p2.is_running = False
                break
    
    def _step_p3_water(self, actuators: Dict[str, float]):
        """Simulate P3 Water Treatment dynamics."""
        p3 = self.state.p3
        dt = self.dt
        
        # Read actuator commands
        pump_cmd = actuators.get("P3_LCP01D", 0.0) > 0.5
        valve_demand = actuators.get("P3_LCV01D", 0.0)
        high_level = actuators.get("P3_LH01", 80.0)
        low_level = actuators.get("P3_LL01", 20.0)
        
        # Valve dynamics
        alpha = dt / (self.VALVE_TAU + dt)
        p3.valve_pos += alpha * (valve_demand - p3.valve_pos)
        
        # Flow depends on pump and valve
        if pump_cmd:
            inflow = 20.0  # L/min when pump running
        else:
            inflow = 0.0
        
        outflow = 15.0 * (p3.valve_pos / 100.0)  # L/min
        p3.flow = outflow
        
        # Level dynamics (tank ~500L)
        d_level = (inflow - outflow) / 500.0 * 60.0 * dt
        p3.level = max(0.0, min(100.0, p3.level + d_level))
        
        # Pressure depends on level (hydrostatic)
        p3.pressure = 1.0 + 0.03 * p3.level
    
    def _step_p4_hil(self, actuators: Dict[str, float]):
        """Simulate P4 HIL Power Grid dynamics."""
        p4 = self.state.p4
        p1 = self.state.p1
        p2 = self.state.p2
        p3 = self.state.p3
        dt = self.dt
        
        # Read setpoints
        st_ps = actuators.get("P4_ST_PS", 50.0)  # Steam turbine power setpoint
        ht_ps = actuators.get("P4_HT_PS", 50.0)  # Hydro power setpoint
        governor = actuators.get("P4_ST_GOV", 50.0)  # Governor position
        
        # Steam turbine power from P2 speed
        # Normalized to MW scale
        if p2.is_running and p2.speed > 100:
            p4.st_flow = p1.pressure1 * 0.5  # Steam flow from boiler
            p4.st_power = (p2.speed / 3000.0) * 50.0 * (governor / 100.0)
            p4.st_load = p4.st_power * 0.95
        else:
            p4.st_flow = 0.0
            p4.st_power = 0.0
            p4.st_load = 0.0
        
        # Steam conditions from boiler
        p4.st_pressure = p1.pressure1 * 2.0  # Scale up for power plant
        p4.st_temp = p1.temp1 * 2.5  # Scale for steam temp
        
        # Hydro turbine power from P3 water flow
        p4.ht_flow = p3.flow * 0.01  # Convert to m³/s
        p4.ht_power = p4.ht_flow * 20.0  # Simple power calculation
        
        # Update load demand (could be time-varying)
        # For now, keep it constant or slightly varying
        p4.load_demand = 100.0 + 10.0 * math.sin(self._step_count * 0.01)
    
    def _add_noise(self, value: float) -> float:
        """Add small measurement noise to sensor value."""
        return value * (1.0 + random.gauss(0, self.NOISE_LEVEL))
    
    def get_all_values(self) -> Dict[str, float]:
        """Return current values of all plant variables for logging."""
        p1, p2, p3, p4 = self.state.p1, self.state.p2, self.state.p3, self.state.p4
        
        return {
            # P1 Boiler
            "P1_LIT01": p1.level,
            "P1_TIT01": p1.temp1,
            "P1_TIT02": p1.temp2,
            "P1_TIT03": p1.temp3,
            "P1_PIT01": p1.pressure1,
            "P1_PIT02": p1.pressure2,
            "P1_FT01": p1.flow1,
            "P1_FT02": p1.flow2,
            "P1_FT03": p1.flow3,
            "P1_FCV01Z": p1.fcv01_pos,
            "P1_FCV02Z": p1.fcv02_pos,
            "P1_FCV03Z": p1.fcv03_pos,
            "P1_LCV01Z": p1.lcv01_pos,
            "P1_PCV01Z": p1.pcv01_pos,
            "P1_PCV02Z": p1.pcv02_pos,
            # P2 Turbine
            "P2_SIT01": p2.speed,
            "P2_VIBTR01": p2.vibration[0],
            "P2_VIBTR02": p2.vibration[1],
            "P2_VIBTR03": p2.vibration[2],
            "P2_VIBTR04": p2.vibration[3],
            "P2_OnOff": 1.0 if p2.is_running else 0.0,
            "P2_Emerg": 1.0 if p2.is_tripped else 0.0,
            # P3 Water
            "P3_LIT01": p3.level,
            "P3_FIT01": p3.flow,
            "P3_PIT01": p3.pressure,
            # P4 HIL
            "P4_ST_FD": p4.st_flow,
            "P4_ST_PO": p4.st_power,
            "P4_ST_LD": p4.st_load,
            "P4_ST_PT01": p4.st_pressure,
            "P4_ST_TT01": p4.st_temp,
            "P4_HT_FD": p4.ht_flow,
            "P4_HT_PO": p4.ht_power,
            "P4_LD": p4.load_demand,
        }
