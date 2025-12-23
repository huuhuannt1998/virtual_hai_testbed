#!/usr/bin/env python3
"""
CUSUM (Cumulative Sum) Detector for ICS Anomaly Detection

This module implements a CUSUM-based detector that can identify
"low-and-slow" attacks that bypass instantaneous rate limits.

The CUSUM algorithm accumulates deviations over time, making it effective
against adversaries who make small, individually-acceptable changes that
compound into dangerous states.

Reference: Page, E. S. (1954). "Continuous inspection schemes"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json


@dataclass
class CUSUMState:
    """State of a CUSUM detector for a single variable."""
    name: str
    S_high: float = 0.0      # Upper CUSUM statistic
    S_low: float = 0.0       # Lower CUSUM statistic
    mean: float = 50.0       # Expected mean (baseline)
    sigma: float = 5.0       # Expected standard deviation
    k: float = 0.5           # Slack parameter (in sigma units)
    h: float = 4.0           # Threshold (in sigma units)
    alarm_high: bool = False
    alarm_low: bool = False
    last_value: float = 0.0
    cumulative_drift: float = 0.0


@dataclass
class DetectorResult:
    """Result of detection for a single timestep."""
    timestamp: float
    cusum_score: float       # Max of S_high and S_low
    threshold: float
    alarm: bool
    alarm_type: str          # "none", "high", "low", "both"
    cumulative_drift: float
    recommended_action: str  # "allow", "warn", "block", "safe_hold"


class CUSUMDetector:
    """
    CUSUM-based anomaly detector for ICS process variables.
    
    Detects gradual drift attacks by accumulating deviations from
    expected behavior over time.
    
    Usage:
        detector = CUSUMDetector()
        detector.add_variable("Water_Level", mean=50.0, sigma=5.0)
        
        # In control loop:
        result = detector.update("Water_Level", current_value)
        if result.alarm:
            trigger_safe_hold()
    """
    
    def __init__(
        self,
        k: float = 0.5,       # Sensitivity (lower = more sensitive)
        h: float = 4.0,       # Threshold (higher = fewer false alarms)
        decay: float = 0.0,   # Optional decay factor for forgetting
        warn_threshold: float = 0.5,  # Fraction of h for warning
    ):
        self.k = k
        self.h = h
        self.decay = decay
        self.warn_threshold = warn_threshold
        self.variables: Dict[str, CUSUMState] = {}
        self.history: List[Dict] = []
        self.alarm_active = False
        self.safe_hold_engaged = False
        
    def add_variable(
        self,
        name: str,
        mean: float = 50.0,
        sigma: float = 5.0,
        k: Optional[float] = None,
        h: Optional[float] = None,
    ):
        """Add a process variable to monitor."""
        self.variables[name] = CUSUMState(
            name=name,
            mean=mean,
            sigma=sigma,
            k=k if k is not None else self.k,
            h=h if h is not None else self.h,
        )
        
    def update(
        self,
        name: str,
        value: float,
        timestamp: Optional[float] = None,
    ) -> DetectorResult:
        """
        Update the CUSUM detector with a new observation.
        
        Args:
            name: Variable name
            value: Current observed value
            timestamp: Optional timestamp (uses current time if not provided)
            
        Returns:
            DetectorResult with alarm status and recommended action
        """
        if name not in self.variables:
            raise KeyError(f"Unknown variable: {name}. Add it first with add_variable()")
        
        if timestamp is None:
            timestamp = datetime.now().timestamp()
            
        state = self.variables[name]
        
        # Normalize the observation
        z = (value - state.mean) / state.sigma
        
        # Update CUSUM statistics
        # S_high detects positive shifts (increasing values)
        # S_low detects negative shifts (decreasing values)
        state.S_high = max(0, state.S_high + z - state.k)
        state.S_low = max(0, state.S_low - z - state.k)
        
        # Apply optional decay (forgetting factor)
        if self.decay > 0:
            state.S_high *= (1 - self.decay)
            state.S_low *= (1 - self.decay)
        
        # Track cumulative drift
        state.cumulative_drift += (value - state.last_value)
        state.last_value = value
        
        # Check for alarms
        state.alarm_high = state.S_high > state.h
        state.alarm_low = state.S_low > state.h
        
        # Determine alarm type
        if state.alarm_high and state.alarm_low:
            alarm_type = "both"
        elif state.alarm_high:
            alarm_type = "high"
        elif state.alarm_low:
            alarm_type = "low"
        else:
            alarm_type = "none"
        
        alarm = state.alarm_high or state.alarm_low
        cusum_score = max(state.S_high, state.S_low)
        
        # Determine recommended action
        if alarm:
            recommended_action = "safe_hold"
            self.alarm_active = True
        elif cusum_score > state.h * self.warn_threshold:
            recommended_action = "warn"
        else:
            recommended_action = "allow"
        
        result = DetectorResult(
            timestamp=timestamp,
            cusum_score=cusum_score,
            threshold=state.h,
            alarm=alarm,
            alarm_type=alarm_type,
            cumulative_drift=state.cumulative_drift,
            recommended_action=recommended_action,
        )
        
        # Store in history
        self.history.append({
            "timestamp": timestamp,
            "variable": name,
            "value": value,
            "S_high": state.S_high,
            "S_low": state.S_low,
            "cusum_score": cusum_score,
            "alarm": alarm,
            "action": recommended_action,
        })
        
        return result
    
    def reset(self, name: Optional[str] = None):
        """Reset CUSUM statistics for one or all variables."""
        if name:
            if name in self.variables:
                self.variables[name].S_high = 0.0
                self.variables[name].S_low = 0.0
                self.variables[name].cumulative_drift = 0.0
                self.variables[name].alarm_high = False
                self.variables[name].alarm_low = False
        else:
            for var in self.variables.values():
                var.S_high = 0.0
                var.S_low = 0.0
                var.cumulative_drift = 0.0
                var.alarm_high = False
                var.alarm_low = False
        
        self.alarm_active = False
        self.safe_hold_engaged = False
    
    def get_status(self) -> Dict:
        """Get current status of all monitored variables."""
        return {
            name: {
                "S_high": state.S_high,
                "S_low": state.S_low,
                "score": max(state.S_high, state.S_low),
                "threshold": state.h,
                "alarm": state.alarm_high or state.alarm_low,
                "drift": state.cumulative_drift,
            }
            for name, state in self.variables.items()
        }
    
    def engage_safe_hold(self):
        """Engage safe hold mode - locks all outputs to safe values."""
        self.safe_hold_engaged = True
        
    def release_safe_hold(self):
        """Release safe hold mode after manual acknowledgment."""
        self.safe_hold_engaged = False
        self.reset()  # Reset CUSUM after release
        
    def export_history(self, filepath: str):
        """Export detection history to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)


class AdaptiveCUSUMDetector(CUSUMDetector):
    """
    Adaptive CUSUM that learns the baseline from normal operation.
    
    Useful when exact process parameters are unknown.
    """
    
    def __init__(
        self,
        learning_window: int = 100,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.learning_window = learning_window
        self.learning_data: Dict[str, List[float]] = {}
        self.learning_complete: Dict[str, bool] = {}
        
    def add_variable(self, name: str, **kwargs):
        """Add variable and initialize learning buffer."""
        super().add_variable(name, **kwargs)
        self.learning_data[name] = []
        self.learning_complete[name] = False
        
    def update(self, name: str, value: float, **kwargs) -> DetectorResult:
        """Update with learning phase."""
        # Learning phase
        if not self.learning_complete.get(name, False):
            self.learning_data[name].append(value)
            
            if len(self.learning_data[name]) >= self.learning_window:
                # Compute baseline statistics
                data = np.array(self.learning_data[name])
                self.variables[name].mean = float(np.mean(data))
                self.variables[name].sigma = float(np.std(data)) + 1e-6
                self.learning_complete[name] = True
                print(f"[CUSUM] Learned baseline for {name}: "
                      f"mean={self.variables[name].mean:.2f}, "
                      f"sigma={self.variables[name].sigma:.2f}")
            
            # Return neutral result during learning
            return DetectorResult(
                timestamp=kwargs.get('timestamp', datetime.now().timestamp()),
                cusum_score=0.0,
                threshold=self.variables[name].h,
                alarm=False,
                alarm_type="learning",
                cumulative_drift=0.0,
                recommended_action="allow",
            )
        
        return super().update(name, value, **kwargs)


# =============================================================================
# Standalone test
# =============================================================================

if __name__ == "__main__":
    import random
    
    print("Testing CUSUM Detector")
    print("=" * 50)
    
    # Create detector
    detector = CUSUMDetector(k=0.5, h=4.0)
    detector.add_variable("Water_Level", mean=50.0, sigma=5.0)
    
    # Simulate normal operation
    print("\nPhase 1: Normal operation (50 samples)")
    for i in range(50):
        value = 50.0 + random.gauss(0, 2)  # Normal variation
        result = detector.update("Water_Level", value)
        if i % 10 == 0:
            print(f"  t={i}: value={value:.1f}, score={result.cusum_score:.2f}, action={result.recommended_action}")
    
    # Simulate gradual attack (drift up by 0.5 per step)
    print("\nPhase 2: Gradual attack (+0.5 per step)")
    current = 50.0
    for i in range(50):
        current += 0.5  # Slow drift
        value = current + random.gauss(0, 1)
        result = detector.update("Water_Level", value)
        if result.alarm or i % 10 == 0:
            print(f"  t={50+i}: value={value:.1f}, score={result.cusum_score:.2f}, action={result.recommended_action}")
            if result.alarm:
                print(f"  *** ALARM TRIGGERED at t={50+i}! ***")
                break
    
    print("\nFinal status:", detector.get_status())
