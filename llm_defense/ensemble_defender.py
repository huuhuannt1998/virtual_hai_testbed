#!/usr/bin/env python3
"""
Ensemble Defense Against LLM-Driven Attacks

Combines multiple detection strategies:
1. CUSUM - Cumulative sum drift detection
2. Window Rate Limiter - Sliding window rate-of-change
3. LSTM Autoencoder - Sequence anomaly detection (ML-based)
4. Trajectory Predictor - Future violation prediction

Voting mechanism with confidence weighting for robust detection.
"""

import os
import sys
import json
import logging
import numpy as np
from collections import deque
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DetectionResult:
    """Result from a single detector."""
    detector_name: str
    is_attack: bool
    confidence: float
    reason: str
    details: Dict = field(default_factory=dict)


@dataclass 
class EnsembleResult:
    """Combined result from ensemble voting."""
    is_attack: bool
    confidence: float
    votes_for_attack: int
    total_detectors: int
    individual_results: List[DetectionResult]
    recommended_action: str  # 'allow', 'block', 'alert'


# =============================================================================
# Detector 1: CUSUM Drift Detection
# =============================================================================

class CUSUMDetector:
    """
    Cumulative Sum (CUSUM) algorithm for detecting gradual drift.
    
    Accumulates deviations from baseline; triggers when sum exceeds threshold.
    Effective against low-and-slow attacks that evade instantaneous checks.
    """
    
    def __init__(self, k: float = 0.5, h: float = 4.0, baseline: float = 50.0):
        """
        Args:
            k: Allowable slack (sensitivity). Lower = more sensitive.
            h: Decision threshold. Lower = faster detection, more false positives.
            baseline: Expected normal value (e.g., 50% setpoint).
        """
        self.k = k
        self.h = h
        self.baseline = baseline
        self.S_pos = 0.0  # Cumulative sum for upward drift
        self.S_neg = 0.0  # Cumulative sum for downward drift
        self.history = []
    
    def reset(self):
        """Reset detector state."""
        self.S_pos = 0.0
        self.S_neg = 0.0
        self.history = []
    
    def update(self, value: float) -> DetectionResult:
        """
        Update CUSUM with new value and check for drift.
        
        Args:
            value: Current setpoint or sensor value.
            
        Returns:
            DetectionResult with attack status and confidence.
        """
        self.history.append(value)
        
        # Compute deviation from baseline
        deviation = value - self.baseline
        
        # Update cumulative sums
        self.S_pos = max(0, self.S_pos + deviation - self.k)
        self.S_neg = max(0, self.S_neg - deviation - self.k)
        
        # Check for alarm
        is_attack = self.S_pos > self.h or self.S_neg > self.h
        
        # Confidence based on how far above threshold
        max_S = max(self.S_pos, self.S_neg)
        confidence = min(1.0, max_S / (self.h * 2)) if is_attack else max_S / self.h
        
        direction = "upward" if self.S_pos > self.S_neg else "downward"
        reason = f"CUSUM {direction} drift: S+={self.S_pos:.2f}, S-={self.S_neg:.2f}, threshold={self.h}"
        
        return DetectionResult(
            detector_name="CUSUM",
            is_attack=is_attack,
            confidence=confidence,
            reason=reason,
            details={"S_pos": self.S_pos, "S_neg": self.S_neg, "threshold": self.h}
        )


# =============================================================================
# Detector 2: Sliding Window Rate Limiter
# =============================================================================

class WindowRateLimiter:
    """
    Sliding window rate-of-change detector.
    
    Tracks maximum drift over a rolling window of N steps.
    Triggers if cumulative change exceeds threshold.
    """
    
    def __init__(self, window_size: int = 10, max_drift: float = 15.0):
        """
        Args:
            window_size: Number of steps to track.
            max_drift: Maximum allowed drift (%) over the window.
        """
        self.window_size = window_size
        self.max_drift = max_drift
        self.history = deque(maxlen=window_size)
    
    def reset(self):
        """Reset detector state."""
        self.history.clear()
    
    def update(self, value: float) -> DetectionResult:
        """
        Update window and check for excessive drift.
        
        Args:
            value: Current setpoint value.
            
        Returns:
            DetectionResult with attack status.
        """
        self.history.append(value)
        
        if len(self.history) < 2:
            return DetectionResult(
                detector_name="WindowRate",
                is_attack=False,
                confidence=0.0,
                reason="Insufficient history",
                details={}
            )
        
        # Calculate total drift over window
        total_drift = abs(self.history[-1] - self.history[0])
        
        # Check if drift exceeds threshold
        is_attack = total_drift > self.max_drift
        
        # Confidence based on how far above threshold
        confidence = min(1.0, total_drift / self.max_drift) if is_attack else total_drift / self.max_drift
        
        reason = f"Window drift: {total_drift:.2f}% over {len(self.history)} steps (max={self.max_drift}%)"
        
        return DetectionResult(
            detector_name="WindowRate",
            is_attack=is_attack,
            confidence=confidence,
            reason=reason,
            details={"drift": total_drift, "window_size": len(self.history), "threshold": self.max_drift}
        )


# =============================================================================
# Detector 3: Trajectory Predictor
# =============================================================================

class TrajectoryPredictor:
    """
    Predicts future setpoint values and checks for impending violations.
    
    Fits linear trend to recent history and extrapolates.
    Triggers if predicted trajectory leads to safety violation.
    """
    
    def __init__(self, min_history: int = 5, horizon: int = 10, 
                 safe_min: float = 20.0, safe_max: float = 80.0):
        """
        Args:
            min_history: Minimum history length for prediction.
            horizon: Steps ahead to predict.
            safe_min: Lower safety bound.
            safe_max: Upper safety bound.
        """
        self.min_history = min_history
        self.horizon = horizon
        self.safe_min = safe_min
        self.safe_max = safe_max
        self.history = []
    
    def reset(self):
        """Reset detector state."""
        self.history = []
    
    def update(self, value: float) -> DetectionResult:
        """
        Update history and predict future trajectory.
        
        Args:
            value: Current setpoint value.
            
        Returns:
            DetectionResult with attack status if trajectory leads to violation.
        """
        self.history.append(value)
        
        if len(self.history) < self.min_history:
            return DetectionResult(
                detector_name="Trajectory",
                is_attack=False,
                confidence=0.0,
                reason=f"Insufficient history ({len(self.history)}/{self.min_history})",
                details={}
            )
        
        # Fit linear trend to recent history
        x = np.arange(len(self.history))
        coeffs = np.polyfit(x, self.history, 1)
        slope = coeffs[0]
        
        # Predict future value
        future_step = len(self.history) + self.horizon
        predicted_value = np.polyval(coeffs, future_step)
        
        # Check if prediction violates safety bounds
        will_violate_high = predicted_value > self.safe_max
        will_violate_low = predicted_value < self.safe_min
        is_attack = will_violate_high or will_violate_low
        
        # Confidence based on how far outside bounds
        if will_violate_high:
            overshoot = predicted_value - self.safe_max
            confidence = min(1.0, overshoot / 20.0)
        elif will_violate_low:
            undershoot = self.safe_min - predicted_value
            confidence = min(1.0, undershoot / 20.0)
        else:
            # How close to bounds?
            dist_to_bound = min(self.safe_max - predicted_value, predicted_value - self.safe_min)
            confidence = max(0, 1.0 - dist_to_bound / 30.0) * 0.5
        
        direction = "overflow" if will_violate_high else ("underflow" if will_violate_low else "safe")
        reason = f"Trajectory prediction: slope={slope:.3f}/step, predicted={predicted_value:.1f}% in {self.horizon} steps -> {direction}"
        
        return DetectionResult(
            detector_name="Trajectory",
            is_attack=is_attack,
            confidence=confidence,
            reason=reason,
            details={"slope": slope, "predicted": predicted_value, "horizon": self.horizon}
        )


# =============================================================================
# Detector 4: LSTM Sequence Anomaly Detector (ML-based)
# =============================================================================

class LSTMAutoencoder(nn.Module):
    """
    LSTM-based autoencoder for sequence anomaly detection.
    
    Trained on normal operator command sequences.
    High reconstruction error indicates anomalous (attack) pattern.
    """
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Output projection
        self.output = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode then decode sequence.
        
        Args:
            x: Input sequence [batch, seq_len, 1]
            
        Returns:
            Reconstructed sequence [batch, seq_len, 1]
        """
        # Encode
        _, (h, c) = self.encoder(x)
        
        # Prepare decoder input (repeat hidden state for each timestep)
        seq_len = x.size(1)
        decoder_input = h[-1].unsqueeze(1).repeat(1, seq_len, 1)
        
        # Decode
        decoded, _ = self.decoder(decoder_input)
        
        # Project to output dimension
        output = self.output(decoded)
        
        return output


class LSTMSequenceDetector:
    """
    LSTM-based sequence anomaly detector.
    
    Wraps LSTMAutoencoder with training and inference logic.
    """
    
    def __init__(self, seq_len: int = 15, hidden_dim: int = 32, 
                 threshold: float = 0.1, device: str = 'cpu'):
        """
        Args:
            seq_len: Sequence length for detection.
            hidden_dim: LSTM hidden dimension.
            threshold: Reconstruction error threshold for anomaly.
            device: 'cpu' or 'cuda'.
        """
        self.seq_len = seq_len
        self.threshold = threshold
        self.device = device
        
        self.model = LSTMAutoencoder(input_dim=1, hidden_dim=hidden_dim).to(device)
        self.history = deque(maxlen=seq_len)
        self.trained = False
        self.normal_mean = 0.0
        self.normal_std = 1.0
    
    def reset(self):
        """Reset detector state (keep trained model)."""
        self.history.clear()
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize input using training statistics."""
        return (x - self.normal_mean) / (self.normal_std + 1e-8)
    
    def train(self, normal_sequences: np.ndarray, epochs: int = 100, lr: float = 0.001):
        """
        Train the LSTM autoencoder on normal sequences.
        
        Args:
            normal_sequences: Array of shape [num_samples, seq_len]
            epochs: Training epochs.
            lr: Learning rate.
        """
        log.info(f"Training LSTM detector on {len(normal_sequences)} sequences...")
        
        # Compute normalization statistics
        self.normal_mean = np.mean(normal_sequences)
        self.normal_std = np.std(normal_sequences)
        
        # Normalize and convert to tensor
        normalized = self.normalize(normal_sequences)
        X = torch.tensor(normalized, dtype=torch.float32).unsqueeze(-1).to(self.device)
        
        # Training loop
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            recon = self.model(X)
            loss = F.mse_loss(recon, X)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                log.info(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        
        # Compute threshold from training data
        self.model.eval()
        with torch.no_grad():
            recon = self.model(X)
            errors = torch.mean((recon - X) ** 2, dim=(1, 2)).cpu().numpy()
            self.threshold = np.mean(errors) + 2 * np.std(errors)
        
        self.trained = True
        log.info(f"Training complete. Threshold: {self.threshold:.6f}")
    
    def update(self, value: float) -> DetectionResult:
        """
        Update history and check for sequence anomaly.
        
        Args:
            value: Current setpoint value.
            
        Returns:
            DetectionResult with attack status.
        """
        self.history.append(value)
        
        if len(self.history) < self.seq_len:
            return DetectionResult(
                detector_name="LSTM",
                is_attack=False,
                confidence=0.0,
                reason=f"Insufficient history ({len(self.history)}/{self.seq_len})",
                details={}
            )
        
        if not self.trained:
            return DetectionResult(
                detector_name="LSTM",
                is_attack=False,
                confidence=0.0,
                reason="Model not trained",
                details={}
            )
        
        # Prepare sequence
        sequence = np.array(list(self.history))
        normalized = self.normalize(sequence)
        X = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
        
        # Compute reconstruction error
        self.model.eval()
        with torch.no_grad():
            recon = self.model(X)
            error = torch.mean((recon - X) ** 2).item()
        
        # Check for anomaly
        is_attack = error > self.threshold
        confidence = min(1.0, error / self.threshold) if is_attack else error / self.threshold
        
        reason = f"LSTM reconstruction error: {error:.6f} (threshold={self.threshold:.6f})"
        
        return DetectionResult(
            detector_name="LSTM",
            is_attack=is_attack,
            confidence=confidence,
            reason=reason,
            details={"error": error, "threshold": self.threshold}
        )


# =============================================================================
# Ensemble Defender
# =============================================================================

class EnsembleDefender:
    """
    Ensemble defense combining multiple detectors with weighted voting.
    
    Detectors:
    1. CUSUM - Statistical drift detection
    2. WindowRate - Sliding window rate limiter
    3. Trajectory - Future violation prediction
    4. LSTM - ML-based sequence anomaly detection
    
    Voting: Weighted sum of detector outputs with confidence.
    """
    
    def __init__(self, 
                 cusum_k: float = 0.5, cusum_h: float = 4.0,
                 window_size: int = 10, max_drift: float = 15.0,
                 trajectory_horizon: int = 10,
                 lstm_seq_len: int = 15,
                 safe_min: float = 20.0, safe_max: float = 80.0,
                 baseline: float = 50.0,
                 weights: Optional[Dict[str, float]] = None,
                 vote_threshold: float = 0.5,
                 device: str = 'cpu'):
        """
        Initialize ensemble with all detectors.
        
        Args:
            cusum_k: CUSUM slack parameter.
            cusum_h: CUSUM threshold.
            window_size: Window rate limiter size.
            max_drift: Maximum allowed drift in window.
            trajectory_horizon: Steps ahead for trajectory prediction.
            lstm_seq_len: Sequence length for LSTM.
            safe_min: Lower safety bound.
            safe_max: Upper safety bound.
            baseline: Expected normal setpoint.
            weights: Detector weights for voting. Default: equal weights.
            vote_threshold: Threshold for ensemble decision (0-1).
            device: 'cpu' or 'cuda' for LSTM.
        """
        # Initialize detectors
        self.detectors = {
            "CUSUM": CUSUMDetector(k=cusum_k, h=cusum_h, baseline=baseline),
            "WindowRate": WindowRateLimiter(window_size=window_size, max_drift=max_drift),
            "Trajectory": TrajectoryPredictor(horizon=trajectory_horizon, 
                                              safe_min=safe_min, safe_max=safe_max),
            "LSTM": LSTMSequenceDetector(seq_len=lstm_seq_len, device=device)
        }
        
        # Detector weights (default: equal)
        self.weights = weights or {
            "CUSUM": 0.30,
            "WindowRate": 0.25,
            "Trajectory": 0.20,
            "LSTM": 0.25
        }
        
        self.vote_threshold = vote_threshold
        self.safe_min = safe_min
        self.safe_max = safe_max
        
        # History for analysis
        self.detection_history = []
    
    def reset(self):
        """Reset all detectors."""
        for detector in self.detectors.values():
            detector.reset()
        self.detection_history = []
    
    def train_lstm(self, normal_sequences: np.ndarray, **kwargs):
        """Train the LSTM detector on normal sequences."""
        self.detectors["LSTM"].train(normal_sequences, **kwargs)
    
    def update(self, value: float) -> EnsembleResult:
        """
        Update all detectors and compute ensemble decision.
        
        Args:
            value: Current setpoint value.
            
        Returns:
            EnsembleResult with combined decision and individual results.
        """
        # Get result from each detector
        results = []
        for name, detector in self.detectors.items():
            result = detector.update(value)
            results.append(result)
        
        # Compute weighted vote
        weighted_sum = 0.0
        total_weight = 0.0
        votes_for_attack = 0
        
        for result in results:
            weight = self.weights.get(result.detector_name, 0.25)
            if result.is_attack:
                weighted_sum += weight * result.confidence
                votes_for_attack += 1
            total_weight += weight
        
        # Normalize
        ensemble_confidence = weighted_sum / total_weight if total_weight > 0 else 0.0
        is_attack = ensemble_confidence > self.vote_threshold
        
        # Determine recommended action
        if is_attack and ensemble_confidence > 0.6:
            action = "block"
        elif is_attack and ensemble_confidence > 0.4:
            action = "alert"
        else:
            action = "allow"
        
        ensemble_result = EnsembleResult(
            is_attack=is_attack,
            confidence=ensemble_confidence,
            votes_for_attack=votes_for_attack,
            total_detectors=len(self.detectors),
            individual_results=results,
            recommended_action=action
        )
        
        # Store history
        self.detection_history.append({
            "value": value,
            "is_attack": is_attack,
            "confidence": ensemble_confidence,
            "action": action,
            "votes": votes_for_attack
        })
        
        return ensemble_result
    
    def get_safe_action(self, proposed_value: float, current_value: float) -> Tuple[float, str]:
        """
        Get safe action after ensemble check.
        
        Args:
            proposed_value: Proposed setpoint from LLM/operator.
            current_value: Current setpoint.
            
        Returns:
            (safe_value, reason) - Either proposed value or clamped/blocked value.
        """
        result = self.update(proposed_value)
        
        if result.recommended_action == "block":
            # Block: return current value (no change)
            return current_value, f"BLOCKED by ensemble (conf={result.confidence:.2f}, votes={result.votes_for_attack}/{result.total_detectors})"
        
        elif result.recommended_action == "alert":
            # Alert but allow with clamping
            clamped = max(self.safe_min, min(self.safe_max, proposed_value))
            if clamped != proposed_value:
                return clamped, f"CLAMPED to {clamped:.1f}% (alert: conf={result.confidence:.2f})"
            return proposed_value, f"ALERT (conf={result.confidence:.2f}) - allowed"
        
        else:
            # Allow
            return proposed_value, "OK"
    
    def get_summary(self) -> Dict:
        """Get summary statistics of detection history."""
        if not self.detection_history:
            return {}
        
        attacks_detected = sum(1 for h in self.detection_history if h["is_attack"])
        blocks = sum(1 for h in self.detection_history if h["action"] == "block")
        alerts = sum(1 for h in self.detection_history if h["action"] == "alert")
        
        return {
            "total_steps": len(self.detection_history),
            "attacks_detected": attacks_detected,
            "blocks": blocks,
            "alerts": alerts,
            "detection_rate": attacks_detected / len(self.detection_history) if self.detection_history else 0,
            "avg_confidence": np.mean([h["confidence"] for h in self.detection_history])
        }


# =============================================================================
# Utility Functions
# =============================================================================

def generate_normal_sequences(num_sequences: int = 1000, seq_len: int = 15, 
                               baseline: float = 50.0, noise_std: float = 2.0) -> np.ndarray:
    """
    Generate synthetic normal operator sequences for training.
    
    Normal operator behavior:
    - Small random fluctuations around baseline
    - Occasional step changes (not consistent increments like LLM)
    - Variable timing and magnitude
    """
    sequences = []
    
    for _ in range(num_sequences):
        seq = [baseline]
        for i in range(seq_len - 1):
            # Random walk with occasional jumps
            if np.random.random() < 0.1:
                # Occasional larger adjustment
                delta = np.random.uniform(-5, 5)
            else:
                # Small noise
                delta = np.random.normal(0, noise_std)
            
            new_val = seq[-1] + delta
            # Keep within reasonable bounds
            new_val = max(30, min(70, new_val))
            seq.append(new_val)
        
        sequences.append(seq)
    
    return np.array(sequences)


def generate_attack_sequences(num_sequences: int = 100, seq_len: int = 15,
                               baseline: float = 50.0, step_size: float = 2.0) -> np.ndarray:
    """
    Generate LLM-style attack sequences for testing.
    
    LLM attack behavior:
    - Consistent incremental changes (+2% per step)
    - Low variance (very predictable)
    - Monotonic increase/decrease
    """
    sequences = []
    
    for _ in range(num_sequences):
        direction = np.random.choice([-1, 1])
        seq = [baseline]
        for i in range(seq_len - 1):
            # Consistent step with small noise
            delta = direction * step_size + np.random.normal(0, 0.1)
            seq.append(seq[-1] + delta)
        
        sequences.append(seq)
    
    return np.array(sequences)


# =============================================================================
# Demo / Test
# =============================================================================

def demo():
    """Demonstrate ensemble defender against LLM attack."""
    print("=" * 70)
    print("ENSEMBLE DEFENDER DEMO")
    print("=" * 70)
    
    # Initialize defender
    defender = EnsembleDefender(
        cusum_k=0.5, cusum_h=4.0,
        window_size=10, max_drift=15.0,
        trajectory_horizon=10,
        lstm_seq_len=15,
        vote_threshold=0.4
    )
    
    # Train LSTM on normal sequences
    print("\n[1] Training LSTM on normal operator sequences...")
    normal_seqs = generate_normal_sequences(500, seq_len=15)
    defender.train_lstm(normal_seqs, epochs=50)
    
    # Simulate LLM attack (low-and-slow +2% per step)
    print("\n[2] Simulating LLM low-and-slow attack...")
    print("-" * 70)
    print(f"{'Step':>5} {'Setpoint':>10} {'Action':>10} {'Confidence':>12} {'Votes':>8} {'Detectors'}")
    print("-" * 70)
    
    setpoint = 50.0
    for step in range(25):
        # LLM proposes +2% increment
        proposed = setpoint + 2.0
        
        # Get safe action from ensemble
        safe_value, reason = defender.get_safe_action(proposed, setpoint)
        result = defender.detection_history[-1]
        
        # Get individual detector votes
        detector_votes = []
        for r in defender.update(proposed).individual_results:
            if r.is_attack:
                detector_votes.append(r.detector_name[:3])
        
        status = "BLOCKED" if result["action"] == "block" else (
                 "ALERT" if result["action"] == "alert" else "OK")
        
        print(f"{step:>5} {proposed:>10.1f}% {status:>10} {result['confidence']:>12.3f} "
              f"{result['votes']:>8} {','.join(detector_votes) if detector_votes else '-'}")
        
        # Update setpoint (blocked = no change)
        if result["action"] != "block":
            setpoint = safe_value
        
        if result["action"] == "block":
            print(f"\n[!] Attack blocked at step {step}!")
            break
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    summary = defender.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    demo()
