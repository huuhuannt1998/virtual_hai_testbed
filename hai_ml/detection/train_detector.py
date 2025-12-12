"""
Train Hybrid Attack Detector on HAI Data
=========================================

This script trains a hybrid attack detector combining:
1. Autoencoder (unsupervised) - detects anomalies via reconstruction error
2. Classifier (supervised) - detects known attack patterns

Usage:
    python -m hai_ml.detection.train_detector --process p3 --epochs 50
    
Output:
    models/p3_detector_v21.03.pth
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    """Autoencoder for unsupervised anomaly detection."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 16):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


class AttackClassifier(nn.Module):
    """Supervised classifier for known attack patterns."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.net(x)


class HybridDetector:
    """Combined autoencoder + classifier detector."""
    
    def __init__(self, input_dim: int, device: str = "cuda"):
        self.device = device
        self.autoencoder = Autoencoder(input_dim).to(device)
        self.classifier = AttackClassifier(input_dim).to(device)
        self.ae_threshold = None
        
    def train_autoencoder(self, normal_data: np.ndarray, epochs: int = 50, batch_size: int = 512):
        """Train autoencoder on normal data only."""
        print("\n  Training autoencoder (unsupervised)...")
        
        # Prepare data
        X = torch.FloatTensor(normal_data).to(self.device)
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Train
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        self.autoencoder.train()
        for epoch in range(epochs):
            total_loss = 0
            for (batch_x,) in loader:
                optimizer.zero_grad()
                x_recon = self.autoencoder(batch_x)
                loss = criterion(x_recon, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(loader)
                print(f"    Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")
        
        # Compute threshold (95th percentile of reconstruction error)
        self.autoencoder.eval()
        with torch.no_grad():
            recon_errors = []
            for (batch_x,) in loader:
                x_recon = self.autoencoder(batch_x)
                error = torch.mean((batch_x - x_recon) ** 2, dim=1)
                recon_errors.extend(error.cpu().numpy())
        
        self.ae_threshold = np.percentile(recon_errors, 95)
        print(f"    AE threshold (95th percentile): {self.ae_threshold:.6f}")
    
    def train_classifier(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 512):
        """Train classifier on labeled data (normal + attack)."""
        print("\n  Training classifier (supervised)...")
        
        # Prepare data
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Train
        optimizer = optim.Adam(self.classifier.parameters(), lr=1e-3)
        criterion = nn.BCELoss()
        
        self.classifier.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = self.classifier(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                predicted = (pred > 0.5).float()
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(loader)
                accuracy = 100 * correct / total
                print(f"    Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}, Acc = {accuracy:.2f}%")
    
    def predict(self, X: np.ndarray):
        """Predict using hybrid detector (OR logic)."""
        self.autoencoder.eval()
        self.classifier.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            # Autoencoder prediction
            x_recon = self.autoencoder(X_tensor)
            recon_error = torch.mean((X_tensor - x_recon) ** 2, dim=1).cpu().numpy()
            ae_pred = (recon_error > self.ae_threshold).astype(int)
            
            # Classifier prediction
            cls_pred = (self.classifier(X_tensor).cpu().numpy().squeeze() > 0.5).astype(int)
            
            # Hybrid (OR logic): flag if EITHER detects attack
            hybrid_pred = np.logical_or(ae_pred, cls_pred).astype(int)
        
        return hybrid_pred, ae_pred, cls_pred
    
    def save(self, path: str):
        """Save detector models."""
        torch.save({
            'autoencoder': self.autoencoder.state_dict(),
            'classifier': self.classifier.state_dict(),
            'ae_threshold': self.ae_threshold,
        }, path)
    
    def load(self, path: str):
        """Load detector models."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.autoencoder.load_state_dict(checkpoint['autoencoder'])
        self.classifier.load_state_dict(checkpoint['classifier'])
        self.ae_threshold = checkpoint['ae_threshold']


def train_detector(
    process: str = "p3",
    version: str = "21.03",
    epochs: int = 50,
    save_dir: str = "models",
    data_root: str = "archive",
):
    """
    Train hybrid attack detector on HAI data.
    
    Args:
        process: HAI process (p1, p2, p3, p4)
        version: HAI version
        epochs: Training epochs
        save_dir: Directory to save models
        data_root: Path to HAI data
    """
    print("=" * 60)
    print(f"Training Hybrid Detector for HAI-{version} Process {process.upper()}")
    print("=" * 60)
    
    # Load data
    from hai_ml.data.hai_loader import load_hai_for_offline_rl
    
    print("\n[1/5] Loading HAI dataset...")
    train_data, test_data = load_hai_for_offline_rl(
        process=process,
        version=version,
        data_root=data_root,
    )
    
    X_train = train_data['observations']
    y_train = train_data.get('attacks', np.zeros(len(X_train)))  # 0=normal, 1=attack
    
    X_test = test_data['observations']
    y_test = test_data.get('attacks', np.zeros(len(X_test)))
    
    n_train_normal = np.sum(y_train == 0)
    n_train_attack = np.sum(y_train == 1)
    n_test_attack = np.sum(y_test == 1)
    
    print(f"  Train: {len(X_train):,} samples ({n_train_normal:,} normal, {n_train_attack:,} attack)")
    print(f"  Test: {len(X_test):,} samples ({n_test_attack:,} attack)")
    print(f"  Input dim: {X_train.shape[1]}")
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[2/5] Creating hybrid detector on {device}...")
    
    detector = HybridDetector(input_dim=X_train.shape[1], device=device)
    
    # Train autoencoder on normal data only
    print(f"\n[3/5] Training autoencoder ({epochs} epochs)...")
    X_normal = X_train[y_train == 0]
    detector.train_autoencoder(X_normal, epochs=epochs)
    
    # Train classifier on balanced data
    print(f"\n[4/5] Training classifier ({epochs} epochs)...")
    
    # Balance dataset: equal normal and attack samples
    n_attack = n_train_attack
    if n_attack > 0:
        X_normal_sample = X_train[y_train == 0][:n_attack]
        X_attack_sample = X_train[y_train == 1]
        
        X_balanced = np.vstack([X_normal_sample, X_attack_sample])
        y_balanced = np.hstack([np.zeros(len(X_normal_sample)), np.ones(len(X_attack_sample))])
        
        # Shuffle
        idx = np.random.permutation(len(X_balanced))
        X_balanced = X_balanced[idx]
        y_balanced = y_balanced[idx]
        
        detector.train_classifier(X_balanced, y_balanced, epochs=epochs)
    else:
        print("  WARNING: No attack samples in training data, skipping classifier training")
    
    # Evaluate on test set
    if n_test_attack > 0:
        print(f"\n[5/5] Evaluating on test set...")
        y_pred, y_ae, y_cls = detector.predict(X_test)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        fpr = np.sum((y_pred == 1) & (y_test == 0)) / np.sum(y_test == 0) if np.sum(y_test == 0) > 0 else 0
        
        print(f"  Hybrid Detector:")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall: {recall:.4f}")
        print(f"    F1: {f1:.4f}")
        print(f"    FPR: {fpr:.4f}")
    else:
        print(f"\n[5/5] No attack samples in test set, skipping evaluation")
    
    # Save model
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    model_name = f"{process}_detector_v{version.replace('.', '')}"
    model_path = save_path / f"{model_name}.pth"
    
    detector.save(str(model_path))
    print(f"\nSaved detector to {model_path}")
    
    # Save metadata
    metadata = {
        "process": process,
        "version": version,
        "epochs": epochs,
        "train_samples": len(X_train),
        "train_attack_samples": int(n_train_attack),
        "input_dim": X_train.shape[1],
        "ae_threshold": float(detector.ae_threshold) if detector.ae_threshold else None,
        "trained_at": datetime.now().isoformat(),
    }
    
    if n_test_attack > 0:
        metadata.update({
            "test_precision": float(precision),
            "test_recall": float(recall),
            "test_f1": float(f1),
            "test_fpr": float(fpr),
        })
    
    with open(save_path / f"{model_name}_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\nDetector training complete!")
    return detector


def main():
    parser = argparse.ArgumentParser(description="Train hybrid attack detector on HAI data")
    parser.add_argument("--process", default="p3", choices=["p1", "p2", "p3", "p4"])
    parser.add_argument("--version", default="21.03")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save-dir", default="models")
    parser.add_argument("--data-root", default="archive")
    
    args = parser.parse_args()
    
    train_detector(
        process=args.process,
        version=args.version,
        epochs=args.epochs,
        save_dir=args.save_dir,
        data_root=args.data_root,
    )


if __name__ == "__main__":
    main()
