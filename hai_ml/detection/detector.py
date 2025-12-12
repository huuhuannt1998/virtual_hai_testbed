"""
Attack Detection Module
=======================

RL-based Intrusion Detection System (IDS) for HAI testbed.

Two detection approaches:
1. Autoencoder-based anomaly detection (unsupervised)
2. Binary classifier (supervised, uses HAI attack labels)

The detector runs in parallel with the RL controller and flags
suspicious sensor readings before they reach the policy.

Usage:
    from hai_ml.detection.detector import AttackDetector
    
    detector = AttackDetector.load("models/detector_p3.pt")
    is_attack, confidence = detector.detect(observation)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Neural Network Architectures
# =============================================================================

class Autoencoder(nn.Module):
    """
    Autoencoder for unsupervised anomaly detection.
    
    Trained on normal data only. High reconstruction error = anomaly.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 8):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
    
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        x_recon, _ = self.forward(x)
        return torch.mean((x - x_recon) ** 2, dim=-1)


class AttackClassifier(nn.Module):
    """
    Binary classifier for supervised attack detection.
    
    Trained on HAI dataset with attack labels.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits)


# =============================================================================
# Attack Detector (Main Class)
# =============================================================================

class AttackDetector:
    """
    Combined attack detection using autoencoder + classifier.
    
    Detection logic:
    - Autoencoder flags if reconstruction_error > threshold
    - Classifier provides attack probability
    - Final decision: either method flags = attack detected
    
    Attributes:
        autoencoder: Unsupervised anomaly detector
        classifier: Supervised attack classifier
        threshold: Reconstruction error threshold
        device: torch device
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        threshold: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.input_dim = input_dim
        self.threshold = threshold
        self.device = device
        
        # Create models
        self.autoencoder = Autoencoder(input_dim, hidden_dim, latent_dim).to(device)
        self.classifier = AttackClassifier(input_dim, hidden_dim).to(device)
        
        # Normalization stats
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        
        # Detection history for temporal analysis
        self.history: List[float] = []
        self.history_window = 10
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize input using training statistics."""
        if self.mean is not None and self.std is not None:
            return (x - self.mean) / (self.std + 1e-8)
        return x
    
    def detect(
        self,
        observation: np.ndarray,
        return_details: bool = False,
    ) -> Tuple[bool, float]:
        """
        Detect if the observation indicates an attack.
        
        Args:
            observation: Sensor readings (1D array)
            return_details: If True, return detailed detection info
            
        Returns:
            is_attack: Boolean indicating attack detected
            confidence: Confidence score [0, 1]
        """
        # Normalize
        obs = self.normalize(observation)
        
        # Convert to tensor
        x = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Autoencoder anomaly score
            recon_error = self.autoencoder.reconstruction_error(x).item()
            ae_anomaly = recon_error > self.threshold
            
            # Classifier probability
            attack_prob = self.classifier.predict_proba(x).item()
            cls_anomaly = attack_prob > 0.5
        
        # Combined decision (OR logic - either method flags it)
        is_attack = ae_anomaly or cls_anomaly
        
        # Confidence is max of both signals
        ae_confidence = min(recon_error / self.threshold, 1.0) if self.threshold > 0 else 0
        confidence = max(ae_confidence, attack_prob)
        
        # Update history
        self.history.append(confidence)
        if len(self.history) > self.history_window:
            self.history.pop(0)
        
        if return_details:
            return is_attack, confidence, {
                "recon_error": recon_error,
                "ae_anomaly": ae_anomaly,
                "attack_prob": attack_prob,
                "cls_anomaly": cls_anomaly,
                "history_mean": np.mean(self.history),
            }
        
        return is_attack, confidence
    
    def detect_batch(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch detection for evaluation.
        
        Args:
            observations: (N, input_dim) array
            
        Returns:
            is_attack: (N,) boolean array
            confidence: (N,) float array
        """
        obs = self.normalize(observations)
        x = torch.FloatTensor(obs).to(self.device)
        
        with torch.no_grad():
            recon_errors = self.autoencoder.reconstruction_error(x).cpu().numpy()
            attack_probs = self.classifier.predict_proba(x).squeeze(-1).cpu().numpy()
        
        ae_anomaly = recon_errors > self.threshold
        cls_anomaly = attack_probs > 0.5
        
        is_attack = ae_anomaly | cls_anomaly
        confidence = np.maximum(recon_errors / self.threshold, attack_probs)
        confidence = np.clip(confidence, 0, 1)
        
        return is_attack, confidence
    
    def fit(
        self,
        normal_data: np.ndarray,
        attack_data: np.ndarray,
        attack_labels: np.ndarray,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
    ) -> Dict[str, List[float]]:
        """
        Train both autoencoder and classifier.
        
        Args:
            normal_data: (N, input_dim) normal operation data
            attack_data: (M, input_dim) data with attack labels
            attack_labels: (M,) binary labels (1 = attack)
            epochs: Training epochs
            batch_size: Batch size
            lr: Learning rate
            
        Returns:
            Training history with losses
        """
        # Compute normalization stats from normal data
        self.mean = normal_data.mean(axis=0)
        self.std = normal_data.std(axis=0)
        
        # Normalize
        normal_norm = self.normalize(normal_data)
        attack_norm = self.normalize(attack_data)
        
        history = {"ae_loss": [], "cls_loss": [], "cls_acc": []}
        
        # =====================================================================
        # Train Autoencoder (unsupervised, normal data only)
        # =====================================================================
        print("\n[1/2] Training Autoencoder on normal data...")
        
        ae_dataset = TensorDataset(torch.FloatTensor(normal_norm))
        ae_loader = DataLoader(ae_dataset, batch_size=batch_size, shuffle=True)
        ae_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        
        self.autoencoder.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for (batch,) in ae_loader:
                batch = batch.to(self.device)
                
                ae_optimizer.zero_grad()
                recon, _ = self.autoencoder(batch)
                loss = F.mse_loss(recon, batch)
                loss.backward()
                ae_optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(ae_loader)
            history["ae_loss"].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: AE Loss = {avg_loss:.6f}")
        
        # Set threshold based on normal data reconstruction error
        self.autoencoder.eval()
        with torch.no_grad():
            normal_tensor = torch.FloatTensor(normal_norm).to(self.device)
            normal_errors = self.autoencoder.reconstruction_error(normal_tensor).cpu().numpy()
        
        # Threshold = mean + 3*std of normal errors (99.7% confidence)
        self.threshold = np.mean(normal_errors) + 3 * np.std(normal_errors)
        print(f"  Threshold set to: {self.threshold:.6f}")
        
        # =====================================================================
        # Train Classifier (supervised, attack-labeled data)
        # =====================================================================
        print("\n[2/2] Training Classifier on labeled data...")
        
        cls_dataset = TensorDataset(
            torch.FloatTensor(attack_norm),
            torch.FloatTensor(attack_labels).unsqueeze(-1)
        )
        cls_loader = DataLoader(cls_dataset, batch_size=batch_size, shuffle=True)
        cls_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        
        # Class weights for imbalanced data
        n_attacks = attack_labels.sum()
        n_normal = len(attack_labels) - n_attacks
        pos_weight = torch.tensor([n_normal / max(n_attacks, 1)]).to(self.device)
        
        self.classifier.train()
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch_x, batch_y in cls_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                cls_optimizer.zero_grad()
                logits = self.classifier(batch_x)
                loss = F.binary_cross_entropy_with_logits(logits, batch_y, pos_weight=pos_weight)
                loss.backward()
                cls_optimizer.step()
                
                epoch_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
            
            avg_loss = epoch_loss / len(cls_loader)
            accuracy = correct / total
            history["cls_loss"].append(avg_loss)
            history["cls_acc"].append(accuracy)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, Acc = {accuracy:.4f}")
        
        self.autoencoder.eval()
        self.classifier.eval()
        
        return history
    
    def evaluate(
        self,
        test_data: np.ndarray,
        test_labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate detection performance.
        
        Returns:
            Dictionary with precision, recall, F1, accuracy, FPR
        """
        predictions, confidences = self.detect_batch(test_data)
        
        # Compute metrics
        tp = np.sum(predictions & (test_labels == 1))
        fp = np.sum(predictions & (test_labels == 0))
        tn = np.sum(~predictions & (test_labels == 0))
        fn = np.sum(~predictions & (test_labels == 1))
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy = (tp + tn) / len(test_labels)
        fpr = fp / max(fp + tn, 1)  # False positive rate
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "fpr": fpr,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }
    
    def save(self, path: str):
        """Save detector to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "autoencoder": self.autoencoder.state_dict(),
            "classifier": self.classifier.state_dict(),
            "input_dim": self.input_dim,
            "threshold": self.threshold,
            "mean": self.mean,
            "std": self.std,
        }, path)
        print(f"Saved detector to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = None) -> "AttackDetector":
        """Load detector from file."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        checkpoint = torch.load(path, map_location=device)
        
        detector = cls(
            input_dim=checkpoint["input_dim"],
            threshold=checkpoint["threshold"],
            device=device,
        )
        detector.autoencoder.load_state_dict(checkpoint["autoencoder"])
        detector.classifier.load_state_dict(checkpoint["classifier"])
        detector.mean = checkpoint["mean"]
        detector.std = checkpoint["std"]
        
        detector.autoencoder.eval()
        detector.classifier.eval()
        
        print(f"Loaded detector from {path}")
        return detector


# =============================================================================
# Training Script
# =============================================================================

def train_detector(
    process: str = "p3",
    version: str = "21.03",
    data_root: str = "archive",
    epochs: int = 50,
    save_dir: str = "models",
) -> AttackDetector:
    """
    Train attack detector on HAI data.
    
    Args:
        process: HAI process (p3 for water treatment)
        version: HAI version
        data_root: Path to archive folder
        epochs: Training epochs
        save_dir: Where to save model
        
    Returns:
        Trained AttackDetector
    """
    from hai_ml.data.hai_loader import HAIDataLoader, HAIDataConfig
    
    print("=" * 60)
    print(f"Training Attack Detector for {process.upper()}")
    print("=" * 60)
    
    # Load data
    config = HAIDataConfig(
        process=process,
        version=version,
        data_root=data_root,
    )
    loader = HAIDataLoader(config)
    train_df, test_df = loader.load()
    
    # Preprocess
    train_df = loader.preprocess(train_df, fit=True)
    test_df = loader.preprocess(test_df, fit=False)
    
    # Get sensor columns
    sensors, _ = loader.get_process_columns()
    sensor_cols = [c for c in sensors if c in train_df.columns]
    
    print(f"\nUsing {len(sensor_cols)} sensor features")
    print(f"Train samples: {len(train_df):,}")
    print(f"Test samples: {len(test_df):,}")
    
    # Split by attack label
    # Train data should be normal (attack=0 for autoencoder training)
    normal_data = train_df[train_df.get('attack', 0) == 0][sensor_cols].values.astype(np.float32)
    
    # Test data has attacks
    test_features = test_df[sensor_cols].values.astype(np.float32)
    test_labels = test_df.get('attack', np.zeros(len(test_df))).values.astype(np.float32)
    
    print(f"Normal samples for AE: {len(normal_data):,}")
    print(f"Test samples with labels: {len(test_features):,} ({test_labels.sum():.0f} attacks)")
    
    # Create and train detector
    detector = AttackDetector(
        input_dim=len(sensor_cols),
        hidden_dim=64,
        latent_dim=16,
    )
    
    history = detector.fit(
        normal_data=normal_data,
        attack_data=test_features,
        attack_labels=test_labels,
        epochs=epochs,
    )
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    metrics = detector.evaluate(test_features, test_labels)
    
    print(f"\n  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  FPR:       {metrics['fpr']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP={metrics['tp']:,} FP={metrics['fp']:,}")
    print(f"    FN={metrics['fn']:,} TN={metrics['tn']:,}")
    
    # Save
    save_path = Path(save_dir) / f"detector_{process}.pt"
    detector.save(save_path)
    
    # Save metrics
    metrics_path = Path(save_dir) / f"detector_{process}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    
    return detector


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train attack detector")
    parser.add_argument("--process", default="p3")
    parser.add_argument("--version", default="21.03")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--data-root", default="archive")
    parser.add_argument("--save-dir", default="models")
    
    args = parser.parse_args()
    
    train_detector(
        process=args.process,
        version=args.version,
        epochs=args.epochs,
        data_root=args.data_root,
        save_dir=args.save_dir,
    )
