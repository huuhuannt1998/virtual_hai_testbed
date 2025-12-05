"""
Attack Simulation Module for HAI Virtual Testbed

This module implements various attack scenarios based on the HAI dataset attack types:
- Bias Injection: Add constant/variable offset to sensor readings
- Replay Attack: Record and replay old sensor values
- DoS Attack: Block or delay communications
- False Data Injection: Replace values with attacker-controlled data
- Covert Attack: Subtle manipulation to evade detection

Reference: HAI Dataset 22.04 attack scenarios
"""

import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Callable
import numpy as np
from collections import deque

from .tags import ALL_TAGS, P1_TAGS, P2_TAGS, P3_TAGS, P4_TAGS, TagType


class AttackType(Enum):
    """Types of attacks supported by the module."""
    NONE = auto()
    BIAS_INJECTION = auto()      # Add offset to sensor values
    REPLAY = auto()              # Replay previously recorded values
    DOS = auto()                 # Denial of service (freeze values)
    FALSE_DATA = auto()          # Replace with attacker-controlled values
    COVERT = auto()              # Subtle manipulation (small bias)
    RAMP = auto()                # Gradually change values
    RANDOM_NOISE = auto()        # Add random noise
    PULSE = auto()               # Periodic spike injection
    SCALING = auto()             # Multiply values by a factor


@dataclass
class AttackConfig:
    """Configuration for an attack scenario."""
    attack_type: AttackType
    target_tags: List[str]           # Tags to attack
    start_time: float = 0.0          # Attack start time (seconds from sim start)
    duration: float = float('inf')   # Attack duration (seconds)
    
    # Bias injection parameters
    bias_value: float = 0.0          # Constant bias to add
    bias_variance: float = 0.0       # Random variance in bias
    
    # Replay attack parameters
    replay_buffer_size: int = 60     # Seconds of data to buffer
    replay_delay: int = 30           # How far back to replay
    
    # Ramp attack parameters
    ramp_rate: float = 0.1           # Units per second
    ramp_target: float = 100.0       # Target value for ramp
    
    # Scaling attack parameters
    scale_factor: float = 1.0        # Multiplication factor
    
    # Pulse attack parameters
    pulse_magnitude: float = 50.0    # Spike magnitude
    pulse_period: float = 10.0       # Seconds between pulses
    pulse_duration: float = 1.0      # How long pulse lasts
    
    # Covert attack parameters
    covert_max_deviation: float = 2.0  # Max % deviation from normal
    
    # General
    enabled: bool = True


@dataclass 
class AttackState:
    """Runtime state for an active attack."""
    config: AttackConfig
    active: bool = False
    start_timestamp: float = 0.0
    replay_buffer: Dict[str, deque] = field(default_factory=dict)
    original_values: Dict[str, float] = field(default_factory=dict)
    ramp_current: Dict[str, float] = field(default_factory=dict)
    

class AttackSimulator:
    """
    Simulates cyber attacks on the HAI testbed.
    
    Integrates with VirtualHAIPlant to modify sensor/actuator values
    before they are written to the PLC or after they are read.
    """
    
    def __init__(self):
        self.attacks: List[AttackState] = []
        self.simulation_time: float = 0.0
        self.dt: float = 1.0
        self._attack_label: int = 0  # 0=normal, 1-N=attack scenarios
        
    def add_attack(self, config: AttackConfig) -> int:
        """
        Add an attack configuration.
        
        Returns:
            Attack index for reference.
        """
        state = AttackState(config=config)
        
        # Initialize replay buffers if needed
        if config.attack_type == AttackType.REPLAY:
            for tag in config.target_tags:
                state.replay_buffer[tag] = deque(maxlen=config.replay_buffer_size)
                
        # Initialize ramp current values
        if config.attack_type == AttackType.RAMP:
            for tag in config.target_tags:
                state.ramp_current[tag] = None
                
        self.attacks.append(state)
        return len(self.attacks) - 1
    
    def remove_attack(self, index: int):
        """Remove an attack by index."""
        if 0 <= index < len(self.attacks):
            self.attacks.pop(index)
            
    def clear_attacks(self):
        """Remove all attacks."""
        self.attacks.clear()
        self._attack_label = 0
        
    def step(self, dt: float = 1.0):
        """Advance simulation time."""
        self.simulation_time += dt
        self.dt = dt
        
        # Update attack active states
        for state in self.attacks:
            if state.config.enabled:
                time_since_start = self.simulation_time - state.config.start_time
                state.active = (
                    time_since_start >= 0 and 
                    time_since_start < state.config.duration
                )
                if state.active and not state.start_timestamp:
                    state.start_timestamp = self.simulation_time
                    
        # Update attack label
        self._attack_label = self._compute_attack_label()
        
    def _compute_attack_label(self) -> int:
        """Compute current attack label for logging."""
        for i, state in enumerate(self.attacks):
            if state.active:
                return i + 1
        return 0
    
    @property
    def attack_label(self) -> int:
        """Current attack label (0=normal, 1-N=attack index)."""
        return self._attack_label
    
    @property
    def is_under_attack(self) -> bool:
        """Check if any attack is currently active."""
        return any(state.active for state in self.attacks)
    
    def apply_to_sensors(self, sensor_values: Dict[str, float]) -> Dict[str, float]:
        """
        Apply attacks to sensor values (before writing to PLC).
        
        Args:
            sensor_values: Dictionary of tag_name -> value
            
        Returns:
            Modified sensor values
        """
        modified = sensor_values.copy()
        
        for state in self.attacks:
            if not state.active:
                continue
                
            for tag in state.config.target_tags:
                if tag not in modified:
                    continue
                    
                original = modified[tag]
                
                # Store original value for replay buffer
                if state.config.attack_type == AttackType.REPLAY:
                    state.replay_buffer[tag].append(original)
                    
                # Store original for reference
                state.original_values[tag] = original
                
                # Apply attack transformation
                modified[tag] = self._transform_value(
                    original, tag, state
                )
                
        return modified
    
    def apply_to_actuators(self, actuator_values: Dict[str, float]) -> Dict[str, float]:
        """
        Apply attacks to actuator values (after reading from PLC).
        
        This can simulate attacks on control commands.
        """
        modified = actuator_values.copy()
        
        for state in self.attacks:
            if not state.active:
                continue
                
            for tag in state.config.target_tags:
                if tag not in modified:
                    continue
                    
                original = modified[tag]
                modified[tag] = self._transform_value(
                    original, tag, state
                )
                
        return modified
    
    def _transform_value(self, value: float, tag: str, state: AttackState) -> float:
        """Apply attack transformation to a single value."""
        config = state.config
        attack_type = config.attack_type
        
        if attack_type == AttackType.BIAS_INJECTION:
            bias = config.bias_value
            if config.bias_variance > 0:
                bias += random.gauss(0, config.bias_variance)
            return value + bias
            
        elif attack_type == AttackType.REPLAY:
            buffer = state.replay_buffer.get(tag, deque())
            if len(buffer) >= config.replay_delay:
                return buffer[-config.replay_delay]
            return value  # Not enough history yet
            
        elif attack_type == AttackType.DOS:
            # Return last known value (freeze)
            return state.original_values.get(tag, value)
            
        elif attack_type == AttackType.FALSE_DATA:
            # Return attacker-specified value
            return config.bias_value  # Using bias_value as the false value
            
        elif attack_type == AttackType.COVERT:
            # Small random deviation within bounds
            max_dev = config.covert_max_deviation / 100.0 * abs(value)
            deviation = random.uniform(-max_dev, max_dev)
            return value + deviation
            
        elif attack_type == AttackType.RAMP:
            # Gradually ramp toward target
            current = state.ramp_current.get(tag)
            if current is None:
                current = value
            # Move toward target
            if current < config.ramp_target:
                current = min(current + config.ramp_rate * self.dt, config.ramp_target)
            else:
                current = max(current - config.ramp_rate * self.dt, config.ramp_target)
            state.ramp_current[tag] = current
            return current
            
        elif attack_type == AttackType.RANDOM_NOISE:
            noise = random.gauss(0, config.bias_variance)
            return value + noise
            
        elif attack_type == AttackType.PULSE:
            # Check if we're in a pulse period
            elapsed = self.simulation_time - state.start_timestamp
            cycle_pos = elapsed % config.pulse_period
            if cycle_pos < config.pulse_duration:
                return value + config.pulse_magnitude
            return value
            
        elif attack_type == AttackType.SCALING:
            return value * config.scale_factor
            
        return value
    

# =============================================================================
# Pre-defined Attack Scenarios (based on HAI dataset patterns)
# =============================================================================

def create_p1_pressure_bias_attack(
    bias: float = 5.0,
    start_time: float = 60.0,
    duration: float = 300.0
) -> AttackConfig:
    """
    P1 Pressure sensor bias injection.
    Adds constant offset to pressure readings.
    """
    return AttackConfig(
        attack_type=AttackType.BIAS_INJECTION,
        target_tags=['P1_PIT01'],
        bias_value=bias,
        start_time=start_time,
        duration=duration
    )


def create_p1_level_replay_attack(
    start_time: float = 120.0,
    duration: float = 180.0,
    delay: int = 30
) -> AttackConfig:
    """
    P1 Level sensor replay attack.
    Replays old level readings to mask level changes.
    """
    return AttackConfig(
        attack_type=AttackType.REPLAY,
        target_tags=['P1_LIT01'],
        start_time=start_time,
        duration=duration,
        replay_delay=delay,
        replay_buffer_size=60
    )


def create_p2_speed_ramp_attack(
    start_time: float = 60.0,
    ramp_rate: float = 10.0,
    target: float = 4000.0
) -> AttackConfig:
    """
    P2 Turbine speed sensor ramp attack.
    Gradually increases reported speed to trigger overspeed trip.
    """
    return AttackConfig(
        attack_type=AttackType.RAMP,
        target_tags=['P2_SIT01'],
        start_time=start_time,
        duration=float('inf'),  # Until target reached
        ramp_rate=ramp_rate,
        ramp_target=target
    )


def create_p2_vibration_false_data(
    false_value: float = 0.5,
    start_time: float = 30.0,
    duration: float = 600.0
) -> AttackConfig:
    """
    P2 Vibration sensor false data injection.
    Masks high vibration to prevent protective trip.
    """
    return AttackConfig(
        attack_type=AttackType.FALSE_DATA,
        target_tags=['P2_VE01'],
        bias_value=false_value,  # The false value to inject
        start_time=start_time,
        duration=duration
    )


def create_p3_dos_attack(
    start_time: float = 60.0,
    duration: float = 120.0
) -> AttackConfig:
    """
    P3 Water treatment DoS attack.
    Freezes level and flow readings.
    """
    return AttackConfig(
        attack_type=AttackType.DOS,
        target_tags=['P3_LIT01', 'P3_FT01', 'P3_FT02'],
        start_time=start_time,
        duration=duration
    )


def create_p4_covert_attack(
    max_deviation: float = 2.0,
    start_time: float = 0.0,
    duration: float = 3600.0
) -> AttackConfig:
    """
    P4 HIL covert attack.
    Small deviations in frequency/voltage to evade detection.
    """
    return AttackConfig(
        attack_type=AttackType.COVERT,
        target_tags=['P4_HT401', 'P4_HT403'],
        covert_max_deviation=max_deviation,
        start_time=start_time,
        duration=duration
    )


def create_multi_process_coordinated_attack(
    start_time: float = 60.0
) -> List[AttackConfig]:
    """
    Coordinated attack across multiple processes.
    Returns list of attack configs to add.
    """
    return [
        # Stage 1: Mask P1 pressure while manipulating
        AttackConfig(
            attack_type=AttackType.BIAS_INJECTION,
            target_tags=['P1_PIT01'],
            bias_value=-10.0,  # Under-report pressure
            start_time=start_time,
            duration=120.0
        ),
        # Stage 2: Disable P2 protection
        AttackConfig(
            attack_type=AttackType.FALSE_DATA,
            target_tags=['P2_VE01', 'P2_TIT01'],
            bias_value=1.0,  # Low safe values
            start_time=start_time + 60.0,
            duration=180.0
        ),
        # Stage 3: Manipulate P4 grid
        AttackConfig(
            attack_type=AttackType.RAMP,
            target_tags=['P4_HT401'],  # Frequency
            ramp_rate=0.1,
            ramp_target=65.0,  # Outside normal range
            start_time=start_time + 120.0,
            duration=300.0
        )
    ]


# =============================================================================
# Attack Scenario Manager
# =============================================================================

class AttackScenarioManager:
    """
    Manages predefined attack scenarios for the HAI testbed.
    """
    
    SCENARIOS = {
        'p1_pressure_bias': create_p1_pressure_bias_attack,
        'p1_level_replay': create_p1_level_replay_attack,
        'p2_speed_ramp': create_p2_speed_ramp_attack,
        'p2_vibration_mask': create_p2_vibration_false_data,
        'p3_dos': create_p3_dos_attack,
        'p4_covert': create_p4_covert_attack,
    }
    
    @classmethod
    def list_scenarios(cls) -> List[str]:
        """List available scenario names."""
        return list(cls.SCENARIOS.keys())
    
    @classmethod
    def get_scenario(cls, name: str, **kwargs) -> AttackConfig:
        """Get an attack configuration by name with optional overrides."""
        if name not in cls.SCENARIOS:
            raise ValueError(f"Unknown scenario: {name}. Available: {cls.list_scenarios()}")
        return cls.SCENARIOS[name](**kwargs)
    
    @classmethod
    def get_coordinated_attack(cls, start_time: float = 60.0) -> List[AttackConfig]:
        """Get the multi-process coordinated attack scenario."""
        return create_multi_process_coordinated_attack(start_time)


# =============================================================================
# Integration with Plant Simulation
# =============================================================================

class AttackAwarePlant:
    """
    Wrapper that adds attack simulation to VirtualHAIPlant.
    
    Usage:
        from hai.plant import VirtualHAIPlant
        from hai.attacks import AttackAwarePlant, AttackScenarioManager
        
        plant = VirtualHAIPlant()
        attack_plant = AttackAwarePlant(plant)
        
        # Add an attack
        config = AttackScenarioManager.get_scenario('p1_pressure_bias', bias=10.0)
        attack_plant.add_attack(config)
        
        # Run simulation (attacks applied automatically)
        attack_plant.step(dt=1.0)
    """
    
    def __init__(self, plant):
        """
        Initialize with a VirtualHAIPlant instance.
        
        Args:
            plant: VirtualHAIPlant instance
        """
        self.plant = plant
        self.attack_sim = AttackSimulator()
        
    def add_attack(self, config: AttackConfig) -> int:
        """Add an attack configuration."""
        return self.attack_sim.add_attack(config)
    
    def add_scenario(self, name: str, **kwargs) -> int:
        """Add a predefined attack scenario by name."""
        config = AttackScenarioManager.get_scenario(name, **kwargs)
        return self.add_attack(config)
    
    def clear_attacks(self):
        """Remove all attacks."""
        self.attack_sim.clear_attacks()
        
    @property
    def attack_label(self) -> int:
        """Current attack label for logging."""
        return self.attack_sim.attack_label
    
    @property
    def is_under_attack(self) -> bool:
        """Check if currently under attack."""
        return self.attack_sim.is_under_attack
    
    def step(self, dt: float = 1.0):
        """
        Advance simulation with attacks.
        
        This:
        1. Steps the attack simulator
        2. Reads actuators from PLC (with attack transformation)
        3. Steps the plant physics
        4. Writes sensors to PLC (with attack transformation)
        """
        # Advance attack timing
        self.attack_sim.step(dt)
        
        # The plant.step() handles PLC I/O internally
        # We need to intercept the sensor writes
        self.plant.step(dt)
        
    def get_sensor_values(self) -> Dict[str, float]:
        """Get current sensor values (with attacks applied)."""
        # Get raw sensor values from plant state
        raw_sensors = self._extract_sensors()
        
        # Apply attacks
        return self.attack_sim.apply_to_sensors(raw_sensors)
    
    def _extract_sensors(self) -> Dict[str, float]:
        """Extract sensor values from plant state."""
        sensors = {}
        state = self.plant.state
        
        # P1 sensors
        sensors['P1_PIT01'] = state.p1.pressure
        sensors['P1_LIT01'] = state.p1.level
        sensors['P1_TIT01'] = state.p1.temperature
        sensors['P1_TIT02'] = state.p1.temperature - 5.0
        sensors['P1_TIT03'] = state.p1.temperature - 10.0
        sensors['P1_FT01'] = state.p1.flow_in * 0.7
        sensors['P1_FT02'] = state.p1.flow_in * 0.3
        sensors['P1_FT03'] = state.p1.flow_out
        
        # P2 sensors
        sensors['P2_SIT01'] = state.p2.speed
        sensors['P2_PIT01'] = state.p2.inlet_pressure
        sensors['P2_TIT01'] = state.p2.temperature
        sensors['P2_TIT02'] = state.p2.temperature + 5.0
        sensors['P2_TIT03'] = state.p2.temperature - 5.0
        sensors['P2_VE01'] = state.p2.vibration
        
        # P3 sensors
        sensors['P3_LIT01'] = state.p3.level
        sensors['P3_FT01'] = state.p3.flow_in
        sensors['P3_FT02'] = state.p3.flow_out
        
        # P4 sensors
        sensors['P4_HT401'] = state.p4.frequency
        sensors['P4_HT403'] = state.p4.voltage
        
        return sensors


if __name__ == '__main__':
    # Demo: List available attack scenarios
    print("Available attack scenarios:")
    for name in AttackScenarioManager.list_scenarios():
        print(f"  - {name}")
        
    print("\nExample: Creating P1 pressure bias attack")
    config = AttackScenarioManager.get_scenario('p1_pressure_bias', bias=10.0, duration=120.0)
    print(f"  Type: {config.attack_type.name}")
    print(f"  Targets: {config.target_tags}")
    print(f"  Bias: {config.bias_value}")
    print(f"  Duration: {config.duration}s")
