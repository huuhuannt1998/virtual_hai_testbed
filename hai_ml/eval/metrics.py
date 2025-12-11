"""
Evaluation Metrics
==================

Metrics for control performance, safety, and timing evaluation.

Includes:
- Control metrics: ITAE, ISE, overshoot, settling time
- Safety metrics: violation rate, time-at-risk, intervention rate
- System metrics: wear proxy, energy surrogate
- Correlation utilities for OPE vs realized returns
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


# ============================================================================
# Control Performance Metrics
# ============================================================================

def itae(error: np.ndarray, dt: float = 1.0) -> float:
    """
    Integral of Time-weighted Absolute Error.
    
    ITAE = ∫ t * |e(t)| dt
    
    Lower is better. Penalizes long-lasting errors.
    
    Args:
        error: (T,) or (T, n_outputs) error signal
        dt: Time step
        
    Returns:
        ITAE value
    """
    error = np.atleast_1d(error)
    if error.ndim > 1:
        error = np.sum(np.abs(error), axis=1)
    else:
        error = np.abs(error)
    
    T = len(error)
    time = np.arange(T) * dt
    
    return float(np.trapz(time * error, dx=dt))


def ise(error: np.ndarray, dt: float = 1.0) -> float:
    """
    Integral of Squared Error.
    
    ISE = ∫ e(t)² dt
    
    Lower is better. Penalizes large errors.
    
    Args:
        error: (T,) or (T, n_outputs) error signal
        dt: Time step
        
    Returns:
        ISE value
    """
    error = np.atleast_1d(error)
    if error.ndim > 1:
        error = np.sum(error ** 2, axis=1)
    else:
        error = error ** 2
    
    return float(np.trapz(error, dx=dt))


def overshoot(y: np.ndarray, target: float) -> float:
    """
    Maximum percent overshoot.
    
    OS = max(0, (max(y) - target) / |target|) * 100
    
    Args:
        y: Response signal
        target: Target/setpoint value
        
    Returns:
        Overshoot percentage (0 if no overshoot)
    """
    if abs(target) < 1e-8:
        return 0.0
    
    max_val = np.max(y)
    if max_val > target:
        return float((max_val - target) / abs(target) * 100)
    return 0.0


def settling_time(
    y: np.ndarray,
    target: float,
    tolerance: float = 0.02,
    dt: float = 1.0,
) -> float:
    """
    Time for response to settle within tolerance band.
    
    Args:
        y: Response signal
        target: Target value
        tolerance: Percentage tolerance (0.02 = 2%)
        dt: Time step
        
    Returns:
        Settling time, or total time if never settled
    """
    tol_band = abs(target * tolerance) if abs(target) > 1e-8 else tolerance
    
    # Find last time outside tolerance
    outside = np.abs(y - target) > tol_band
    
    if not np.any(outside):
        return 0.0  # Already settled
    
    last_outside = np.where(outside)[0][-1]
    return float((last_outside + 1) * dt)


def rise_time(
    y: np.ndarray,
    target: float,
    low_pct: float = 0.1,
    high_pct: float = 0.9,
    dt: float = 1.0,
) -> float:
    """
    Time for response to rise from low_pct to high_pct of target.
    
    Args:
        y: Response signal (starting from 0 or initial value)
        target: Target value
        low_pct: Low percentage threshold
        high_pct: High percentage threshold
        dt: Time step
        
    Returns:
        Rise time
    """
    y0 = y[0] if len(y) > 0 else 0.0
    delta = target - y0
    
    low_val = y0 + low_pct * delta
    high_val = y0 + high_pct * delta
    
    # Find crossing times
    t_low = None
    t_high = None
    
    for i, val in enumerate(y):
        if t_low is None and val >= low_val:
            t_low = i * dt
        if t_high is None and val >= high_val:
            t_high = i * dt
            break
    
    if t_low is None or t_high is None:
        return float(len(y) * dt)  # Never reached
    
    return float(t_high - t_low)


# ============================================================================
# Actuator and Energy Metrics
# ============================================================================

def wear(actions: np.ndarray, dt: float = 1.0) -> float:
    """
    Actuator wear proxy (total variation).
    
    Wear = Σ |a(t+1) - a(t)|
    
    Lower is better. Penalizes frequent actuator changes.
    
    Args:
        actions: (T, n_actions) action sequence
        dt: Time step (not used, for interface consistency)
        
    Returns:
        Total wear proxy
    """
    actions = np.atleast_2d(actions)
    if actions.shape[0] < 2:
        return 0.0
    
    diffs = np.diff(actions, axis=0)
    return float(np.sum(np.abs(diffs)))


def energy(
    state: np.ndarray,
    action: np.ndarray,
    state_weight: float = 0.0,
    action_weight: float = 1.0,
) -> float:
    """
    Energy consumption surrogate.
    
    Simple quadratic form: E = w_s * ||s||² + w_a * ||a||²
    
    Args:
        state: (T, n_states) or (n_states,) state sequence
        action: (T, n_actions) or (n_actions,) action sequence
        state_weight: Weight for state contribution
        action_weight: Weight for action contribution
        
    Returns:
        Energy surrogate value
    """
    state = np.atleast_2d(state)
    action = np.atleast_2d(action)
    
    state_energy = state_weight * np.sum(state ** 2)
    action_energy = action_weight * np.sum(action ** 2)
    
    return float(state_energy + action_energy)


# ============================================================================
# Safety Metrics
# ============================================================================

def violation_rate(flags: np.ndarray) -> float:
    """
    Rate of safety violations.
    
    Args:
        flags: (T,) boolean array, True = violation
        
    Returns:
        Fraction of time in violation [0, 1]
    """
    flags = np.atleast_1d(flags).astype(bool)
    if len(flags) == 0:
        return 0.0
    return float(np.mean(flags))


def time_at_risk(
    flags: np.ndarray,
    dt: float = 1.0,
) -> float:
    """
    Total time spent in unsafe state.
    
    Args:
        flags: (T,) boolean array, True = at risk
        dt: Time step
        
    Returns:
        Total time at risk
    """
    flags = np.atleast_1d(flags).astype(bool)
    return float(np.sum(flags) * dt)


def intervention_rate(
    original_actions: np.ndarray,
    shielded_actions: np.ndarray,
    threshold: float = 1e-4,
) -> float:
    """
    Rate of shield interventions.
    
    Args:
        original_actions: (T, n_actions) actions from policy
        shielded_actions: (T, n_actions) actions after shield
        threshold: Minimum change to count as intervention
        
    Returns:
        Fraction of steps with intervention [0, 1]
    """
    original = np.atleast_2d(original_actions)
    shielded = np.atleast_2d(shielded_actions)
    
    diffs = np.sum(np.abs(original - shielded), axis=1)
    interventions = diffs > threshold
    
    return float(np.mean(interventions))


def intervention_histogram(
    rule_hits: Dict[str, int],
    total_steps: int,
) -> Dict[str, float]:
    """
    Compute normalized intervention rates per rule.
    
    Args:
        rule_hits: Dictionary mapping rule ID to hit count
        total_steps: Total number of steps
        
    Returns:
        Dictionary mapping rule ID to rate
    """
    if total_steps == 0:
        return {k: 0.0 for k in rule_hits}
    
    return {k: v / total_steps for k, v in rule_hits.items()}


# ============================================================================
# Attack Metrics
# ============================================================================

def blocked_percentage(
    n_blocked: int,
    n_total: int,
) -> float:
    """
    Percentage of attack attempts blocked.
    
    Args:
        n_blocked: Number of blocked attempts
        n_total: Total attack attempts
        
    Returns:
        Blocked percentage [0, 100]
    """
    if n_total == 0:
        return 100.0
    return float(n_blocked / n_total * 100)


def attack_detection_latency(
    attack_times: np.ndarray,
    detection_times: np.ndarray,
) -> Dict[str, float]:
    """
    Compute detection latency statistics.
    
    Args:
        attack_times: Times when attacks were injected
        detection_times: Times when attacks were detected
        
    Returns:
        Dictionary with mean, std, p50, p90, p99 latencies
    """
    if len(attack_times) == 0 or len(detection_times) == 0:
        return {'mean': 0.0, 'std': 0.0, 'p50': 0.0, 'p90': 0.0, 'p99': 0.0}
    
    # Match attack to nearest subsequent detection
    latencies = []
    for at in attack_times:
        future_detections = detection_times[detection_times >= at]
        if len(future_detections) > 0:
            latencies.append(future_detections[0] - at)
    
    if len(latencies) == 0:
        return {'mean': 0.0, 'std': 0.0, 'p50': 0.0, 'p90': 0.0, 'p99': 0.0}
    
    latencies = np.array(latencies)
    return {
        'mean': float(np.mean(latencies)),
        'std': float(np.std(latencies)),
        'p50': float(np.percentile(latencies, 50)),
        'p90': float(np.percentile(latencies, 90)),
        'p99': float(np.percentile(latencies, 99)),
    }


# ============================================================================
# Timing Metrics
# ============================================================================

def compute_latency_stats(latencies: np.ndarray) -> Dict[str, float]:
    """
    Compute latency statistics.
    
    Args:
        latencies: (N,) array of latency measurements
        
    Returns:
        Dictionary with mean, std, min, max, p50, p90, p99
    """
    if len(latencies) == 0:
        return {
            'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
            'p50': 0.0, 'p90': 0.0, 'p99': 0.0
        }
    
    return {
        'mean': float(np.mean(latencies)),
        'std': float(np.std(latencies)),
        'min': float(np.min(latencies)),
        'max': float(np.max(latencies)),
        'p50': float(np.percentile(latencies, 50)),
        'p90': float(np.percentile(latencies, 90)),
        'p99': float(np.percentile(latencies, 99)),
    }


# ============================================================================
# Episode Summary Metrics
# ============================================================================

def episode_success(
    violations: int,
    completed: bool,
    max_violations: int = 0,
) -> bool:
    """
    Determine if episode was successful.
    
    Args:
        violations: Number of safety violations
        completed: Whether episode completed without abort
        max_violations: Maximum allowed violations
        
    Returns:
        True if episode was successful
    """
    return completed and violations <= max_violations


def aggregate_episode_metrics(
    episode_metrics: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Aggregate metrics across episodes.
    
    Args:
        episode_metrics: List of per-episode metric dicts
        
    Returns:
        Aggregated statistics
    """
    if not episode_metrics:
        return {}
    
    # Get all numeric keys
    numeric_keys = []
    for key in episode_metrics[0].keys():
        if isinstance(episode_metrics[0][key], (int, float, np.number)):
            numeric_keys.append(key)
    
    aggregated = {}
    for key in numeric_keys:
        values = [m[key] for m in episode_metrics if key in m]
        if values:
            aggregated[f'{key}_mean'] = float(np.mean(values))
            aggregated[f'{key}_std'] = float(np.std(values))
            aggregated[f'{key}_min'] = float(np.min(values))
            aggregated[f'{key}_max'] = float(np.max(values))
    
    # Special handling for success rate
    if 'success' in episode_metrics[0]:
        successes = [m.get('success', False) for m in episode_metrics]
        aggregated['success_rate'] = float(np.mean(successes))
    
    return aggregated


# ============================================================================
# Correlation Utilities (OPE vs Realized)
# ============================================================================

def pearson_correlation(
    ope_estimates: np.ndarray,
    realized_returns: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute Pearson correlation between OPE and realized returns.
    
    Args:
        ope_estimates: OPE value estimates
        realized_returns: Actual episode returns
        
    Returns:
        (correlation, p_value)
    """
    if len(ope_estimates) < 3:
        return 0.0, 1.0
    
    r, p = stats.pearsonr(ope_estimates, realized_returns)
    return float(r), float(p)


def spearman_correlation(
    ope_estimates: np.ndarray,
    realized_returns: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute Spearman rank correlation between OPE and realized returns.
    
    Args:
        ope_estimates: OPE value estimates
        realized_returns: Actual episode returns
        
    Returns:
        (correlation, p_value)
    """
    if len(ope_estimates) < 3:
        return 0.0, 1.0
    
    rho, p = stats.spearmanr(ope_estimates, realized_returns)
    return float(rho), float(p)


def regret(
    ope_best_idx: int,
    realized_returns: np.ndarray,
) -> float:
    """
    Compute regret of OPE selection.
    
    Regret = max(realized) - realized[ope_best_idx]
    
    Args:
        ope_best_idx: Index of model selected by OPE
        realized_returns: Actual episode returns
        
    Returns:
        Regret value (0 if OPE selected the best)
    """
    return float(np.max(realized_returns) - realized_returns[ope_best_idx])


# ============================================================================
# Metric Collection Class
# ============================================================================

class MetricCollector:
    """
    Collector for online metric computation.
    
    Tracks running statistics during episode execution.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.errors: List[float] = []
        self.actions: List[np.ndarray] = []
        self.violations: List[bool] = []
        self.interventions: List[bool] = []
        self.latencies: List[float] = []
        self.states: List[np.ndarray] = []
        self.rewards: List[float] = []
        
        self._prev_action: Optional[np.ndarray] = None
    
    def step(
        self,
        error: float,
        action: np.ndarray,
        state: Optional[np.ndarray] = None,
        reward: float = 0.0,
        violation: bool = False,
        intervention: bool = False,
        latency: float = 0.0,
    ):
        """Record metrics for one step."""
        self.errors.append(error)
        self.actions.append(action.copy())
        self.violations.append(violation)
        self.interventions.append(intervention)
        self.latencies.append(latency)
        self.rewards.append(reward)
        
        if state is not None:
            self.states.append(state.copy())
        
        self._prev_action = action.copy()
    
    def compute_summary(self, dt: float = 1.0) -> Dict[str, float]:
        """Compute summary metrics from collected data."""
        errors = np.array(self.errors)
        actions = np.array(self.actions) if self.actions else np.zeros((0, 1))
        
        summary = {
            'itae': itae(errors, dt),
            'ise': ise(errors, dt),
            'wear': wear(actions, dt),
            'energy': energy(np.zeros_like(actions), actions),
            'violation_rate': violation_rate(np.array(self.violations)),
            'time_at_risk': time_at_risk(np.array(self.violations), dt),
            'intervention_rate': violation_rate(np.array(self.interventions)),
            'n_violations': int(sum(self.violations)),
            'n_interventions': int(sum(self.interventions)),
            'total_reward': float(sum(self.rewards)),
            'n_steps': len(self.errors),
        }
        
        # Latency stats
        if self.latencies:
            lat_stats = compute_latency_stats(np.array(self.latencies))
            for k, v in lat_stats.items():
                summary[f'latency_{k}'] = v
        
        return summary
