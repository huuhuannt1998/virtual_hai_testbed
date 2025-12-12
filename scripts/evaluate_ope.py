"""
Off-Policy Evaluation using Fitted Q-Evaluation (FQE)

Evaluates all trained RL policies using FQE to estimate their expected returns
and compares with realized performance (ITAE from actual evaluation).
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.stats import spearmanr
from tqdm import tqdm
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hai.io import load_hai_data
from hai_ml.rl.evaluate import load_trained_policy


class FQENetwork(nn.Module):
    """Q-network for Fitted Q-Evaluation"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


def train_fqe(policy, dataset, state_dim, action_dim, iterations=100, batch_size=256, lr=3e-4, device='cuda'):
    """
    Train FQE Q-network for given policy
    
    Args:
        policy: Trained policy to evaluate
        dataset: List of (state, action, reward, next_state, done) tuples
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        iterations: Number of FQE iterations
        batch_size: Batch size for training
        lr: Learning rate
        device: Device for training
    
    Returns:
        Trained FQE network
    """
    fqe_net = FQENetwork(state_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(fqe_net.parameters(), lr=lr)
    
    print(f"Training FQE for {iterations} iterations on {len(dataset)} transitions...")
    
    for iteration in tqdm(range(iterations), desc="FQE Training"):
        # Sample batch
        indices = np.random.choice(len(dataset), size=min(batch_size, len(dataset)), replace=False)
        
        states = torch.FloatTensor([dataset[i][0] for i in indices]).to(device)
        actions = torch.FloatTensor([dataset[i][1] for i in indices]).to(device)
        rewards = torch.FloatTensor([dataset[i][2] for i in indices]).to(device)
        next_states = torch.FloatTensor([dataset[i][3] for i in indices]).to(device)
        dones = torch.FloatTensor([dataset[i][4] for i in indices]).to(device)
        
        # Get next actions from policy
        with torch.no_grad():
            next_actions = policy(next_states)
            if isinstance(next_actions, tuple):
                next_actions = next_actions[0]
            
            # Compute target: r + gamma * Q(s', pi(s'))
            next_q = fqe_net(next_states, next_actions).squeeze(-1)
            targets = rewards + 0.99 * next_q * (1 - dones)
        
        # Compute current Q-values
        current_q = fqe_net(states, actions).squeeze(-1)
        
        # MSE loss
        loss = nn.MSELoss()(current_q, targets)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if iteration % 20 == 0:
            print(f"  Iteration {iteration}, Loss: {loss.item():.4f}")
    
    return fqe_net


def estimate_policy_value(fqe_net, policy, dataset, device='cuda'):
    """
    Estimate policy value using trained FQE network
    
    Args:
        fqe_net: Trained FQE network
        policy: Policy to evaluate
        dataset: Evaluation dataset
        device: Device for computation
    
    Returns:
        Estimated value (mean Q-value over dataset)
    """
    fqe_net.eval()
    
    with torch.no_grad():
        states = torch.FloatTensor([d[0] for d in dataset]).to(device)
        actions = policy(states)
        if isinstance(actions, tuple):
            actions = actions[0]
        
        q_values = fqe_net(states, actions).squeeze(-1)
        value = q_values.mean().item()
    
    return value


def bootstrap_confidence_interval(fqe_net, policy, dataset, n_bootstrap=100, device='cuda'):
    """
    Compute bootstrap confidence interval for policy value
    
    Args:
        fqe_net: Trained FQE network  
        policy: Policy to evaluate
        dataset: Evaluation dataset
        n_bootstrap: Number of bootstrap samples
        device: Device for computation
    
    Returns:
        (mean_value, lower_ci, upper_ci)
    """
    values = []
    
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap CI"):
        # Resample dataset with replacement
        indices = np.random.choice(len(dataset), size=len(dataset), replace=True)
        bootstrap_sample = [dataset[i] for i in indices]
        
        # Estimate value on bootstrap sample
        value = estimate_policy_value(fqe_net, policy, bootstrap_sample, device)
        values.append(value)
    
    values = np.array(values)
    mean_val = np.mean(values)
    lower_ci = np.percentile(values, 2.5)
    upper_ci = np.percentile(values, 97.5)
    
    return mean_val, lower_ci, upper_ci


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    processes = ['p1', 'p3', 'p4']  # P2 excluded due to data issues
    algorithms = ['bc', 'td3bc', 'cql', 'iql']
    
    # Load control performance results for comparison
    results_file = Path('results/quick_eval_20251212_115541.json')
    if results_file.exists():
        with open(results_file) as f:
            control_results = json.load(f)
    else:
        control_results = {}
    
    all_ope_results = {}
    fqe_estimates = []
    realized_values = []
    policy_names = []
    
    for process in processes:
        print(f"\n{'='*60}")
        print(f"Evaluating OPE for Process: {process.upper()}")
        print(f"{'='*60}")
        
        # Load data
        print("Loading HAI data...")
        train_data, test_data = load_hai_data(process, version='21.03')
        
        # Convert to dataset format for FQE
        # We'll use a subset for efficiency (FQE is computationally expensive)
        max_samples = 50000
        train_dataset = []
        
        for i in range(min(len(train_data) - 1, max_samples)):
            state = train_data[i]['state']
            action = train_data[i]['action'] if 'action' in train_data[i] else np.zeros(train_data[i]['state'].shape[0])
            reward = -abs(train_data[i].get('tracking_error', 1.0))  # Negative tracking error as reward
            next_state = train_data[i + 1]['state']
            done = 0.0
            
            # Skip if NaN
            if np.any(np.isnan(state)) or np.any(np.isnan(next_state)):
                continue
            
            train_dataset.append((state, action, reward, next_state, done))
        
        print(f"Prepared {len(train_dataset)} transitions for FQE training")
        
        state_dim = train_dataset[0][0].shape[0]
        action_dim = train_dataset[0][1].shape[0] if len(train_dataset[0][1].shape) > 0 else 1
        
        process_results = {}
        
        for algo in algorithms:
            print(f"\n--- Algorithm: {algo.upper()} ---")
            
            model_path = Path(f'models/{process}_{algo}_model.zip')
            if not model_path.exists():
                print(f"Model not found: {model_path}, skipping...")
                continue
            
            try:
                # Load policy
                policy = load_trained_policy(str(model_path), algo)
                
                # Train FQE network
                fqe_net = train_fqe(policy, train_dataset, state_dim, action_dim, 
                                   iterations=50, device=device)
                
                # Estimate value with confidence interval
                mean_val, lower_ci, upper_ci = bootstrap_confidence_interval(
                    fqe_net, policy, train_dataset[:5000], n_bootstrap=100, device=device
                )
                
                # Get realized ITAE value (convert to reward: negative ITAE)
                realized_itae = control_results.get(process, {}).get(algo, {}).get('itae', None)
                realized_reward = -realized_itae if realized_itae is not None else None
                
                # Admission decision (admit if lower CI > threshold)
                threshold = -500  # Corresponds to ITAE < 500
                admitted = lower_ci > threshold
                
                process_results[algo] = {
                    'fqe_estimate': float(mean_val),
                    'ci_lower': float(lower_ci),
                    'ci_upper': float(upper_ci),
                    'realized_reward': float(realized_reward) if realized_reward is not None else None,
                    'realized_itae': float(realized_itae) if realized_itae is not None else None,
                    'admitted': bool(admitted),
                    'threshold': float(threshold)
                }
                
                if realized_reward is not None:
                    fqe_estimates.append(mean_val)
                    realized_values.append(realized_reward)
                    policy_names.append(f"{process}_{algo}")
                
                print(f"  FQE Estimate: {mean_val:.2f} [{lower_ci:.2f}, {upper_ci:.2f}]")
                print(f"  Realized Reward: {realized_reward:.2f}" if realized_reward else "  Realized: N/A")
                print(f"  Admitted: {'YES' if admitted else 'NO'}")
                
            except Exception as e:
                print(f"Error evaluating {process}_{algo}: {e}")
                continue
        
        all_ope_results[process] = process_results
    
    # Compute correlation between FQE estimates and realized values
    if len(fqe_estimates) >= 3:
        correlation, p_value = spearmanr(fqe_estimates, realized_values)
        print(f"\n{'='*60}")
        print(f"OPE FIDELITY ANALYSIS")
        print(f"{'='*60}")
        print(f"Spearman correlation: ρ = {correlation:.3f}, p = {p_value:.4f}")
        print(f"Number of policies: {len(fqe_estimates)}")
        
        # Admission precision/recall
        admitted_policies = [all_ope_results[p][a]['admitted'] 
                            for p in processes for a in algorithms 
                            if p in all_ope_results and a in all_ope_results[p]]
        good_policies = [all_ope_results[p][a].get('realized_itae', 1000) < 450
                        for p in processes for a in algorithms
                        if p in all_ope_results and a in all_ope_results[p] and all_ope_results[p][a].get('realized_itae')]
        
        if len(admitted_policies) == len(good_policies) and len(admitted_policies) > 0:
            true_positives = sum([a and g for a, g in zip(admitted_policies, good_policies)])
            false_positives = sum([a and not g for a, g in zip(admitted_policies, good_policies)])
            false_negatives = sum([not a and g for a, g in zip(admitted_policies, good_policies)])
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            print(f"Admission precision: {precision:.2%}")
            print(f"Admission recall: {recall:.2%}")
        
        all_ope_results['summary'] = {
            'spearman_rho': float(correlation),
            'p_value': float(p_value),
            'n_policies': len(fqe_estimates),
            'fqe_estimates': [float(x) for x in fqe_estimates],
            'realized_values': [float(x) for x in realized_values],
            'policy_names': policy_names
        }
    
    # Save results
    output_path = Path('results/ope_fqe_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_ope_results, f, indent=2)
    
    print(f"\n✓ OPE results saved to {output_path}")


if __name__ == '__main__':
    main()
