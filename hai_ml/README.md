# HAI-ML: Machine Learning Evaluation Stack for HAI Testbed

A complete, reproducible evaluation stack for offline RL-based control of industrial processes with safety shields.

## Overview

This module provides all artifacts needed to reproduce the Evaluation section of a top-tier security conference paper, including:

- **Offline RL Training**: TD3+BC, CQL, IQL using d3rlpy
- **Off-Policy Evaluation**: FQE + WIS with bootstrap 95% CIs
- **Safety Shields**: Rule-based action projection from process schemas
- **Attack Scenarios**: Hostile takeover, packet flood, sensor bias, time delay
- **Metrics**: ITAE, ISE, overshoot, settling time, wear, energy, violations, interventions, latency
- **LaTeX Artifacts**: Auto-generated tables and publication-quality figures

## Directory Structure

```
hai_ml/
├── schemas/                 # Process definitions
│   ├── p1.yaml             # Boiler subprocess
│   ├── p3.yaml             # Water treatment subprocess
│   └── p12.yaml            # Coupled boiler-turbine
├── configs/                 # Experiment configurations
│   ├── p1_baseline.yml
│   ├── p3_baseline.yml
│   └── p12_baseline.yml
├── data/
│   └── build_dataset.py    # CSV → NPZ converter
├── envs/
│   └── hai_gym.py          # Gymnasium wrapper with attacks
├── safety/
│   ├── shield.py           # Rule-based safety shield
│   └── tests/
│       └── test_shield.py
├── il/
│   └── train_bc.py         # Behavior Cloning trainer
├── rl/
│   ├── train_offline.py    # Offline RL (TD3+BC, CQL, IQL)
│   ├── ope_gate.py         # OPE with admission gating
│   └── finetune_shielded.py # Shielded online fine-tuning
├── mpc/
│   ├── dyn_model.py        # Learned dynamics model
│   └── plan_cem.py         # CEM planner
└── eval/
    ├── metrics.py          # All evaluation metrics
    ├── run_eval.py         # Main evaluation harness
    ├── plot_ope_scatter.py # OPE vs realized return
    ├── plot_latency_cdf.py # Decision latency CDF
    └── plot_interventions.py # Shield intervention analysis

scripts/
├── setup_env.sh            # Environment setup
├── run_trackA.sh           # Track A: Behavior Cloning
├── run_trackB.sh           # Track B: Offline RL
├── run_trackC.sh           # Track C: MPC
└── make_tables.sh          # Generate LaTeX tables

results/                     # Output CSVs
paper/
├── figs/                   # Output figures (PDF/PNG)
└── tables/                 # Output LaTeX tables
```

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/yourusername/virtual_hai_testbed.git
cd virtual_hai_testbed

# Run setup script
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh

# Activate environment
source .venv/bin/activate
```

### 2. Prepare Data

Place HAI dataset CSV files in `data/hai/`:
```
data/hai/
├── train_p3.csv
├── val_p3.csv
└── test_p3.csv
```

### 3. Run Full Pipeline

```bash
# Track A: Behavior Cloning baseline
./scripts/run_trackA.sh p3 --attacks

# Track B: Offline RL with OPE gating
./scripts/run_trackB.sh p3 --attacks

# Track C: Model Predictive Control
./scripts/run_trackC.sh p3 --attacks

# Generate LaTeX tables
./scripts/make_tables.sh p3
```

## Evaluation Tracks

### Track A: Imitation Learning

Trains a Behavior Cloning policy as the baseline:

```bash
python -m hai_ml.il.train_bc \
    --dataset data/processed/p3/train.npz \
    --output models/p3/bc/policy.pt \
    --epochs 100
```

### Track B: Offline RL

1. **Train multiple algorithms**:
```bash
python -m hai_ml.rl.train_offline \
    --dataset data/processed/p3/train.npz \
    --algo td3bc \
    --n-steps 100000 \
    --output models/p3/offline_rl/td3bc.d3
```

2. **Run OPE with admission gating**:
```bash
python -m hai_ml.rl.ope_gate \
    --dataset data/processed/p3/val.npz \
    --bc-model models/p3/bc/policy.pt \
    --rl-models models/p3/offline_rl/*.d3 \
    --output results/p3_offline_leaderboard.csv
```

Admission rule: Policy passes if `FQE_lower_CI > 0` AND `FQE_lower_CI > BC_lower_CI`

### Track C: Model Predictive Control

1. **Train dynamics model**:
```bash
python -m hai_ml.mpc.dyn_model \
    --dataset data/processed/p3/train.npz \
    --output models/p3/mpc/dynamics.pt
```

2. **Run CEM-based planning**:
```bash
python -m hai_ml.mpc.plan_cem \
    --dynamics-model models/p3/mpc/dynamics.pt \
    --schema hai_ml/schemas/p3.yaml \
    --output results/p3_cem_nominal.csv
```

## Safety Shield

The shield enforces safety constraints by projecting actions:

```python
from hai_ml.safety.shield import SafetyShield

shield = SafetyShield('hai_ml/schemas/p3.yaml')

# Get safe action
safe_action, was_modified, triggered_rules = shield.project(
    state=current_state,
    action=proposed_action
)
```

## Attack Scenarios

Supported attacks for robustness testing:

| Attack | Description | Parameters |
|--------|-------------|------------|
| `hostile` | Adversary injects malicious actions | `magnitude`: 0.5 |
| `flood` | Random packet corruption | `prob`: 0.1 |
| `bias` | Persistent sensor bias | `magnitude`: 0.2 |
| `delay` | Action/observation delays | `steps`: 3 |

```python
from hai_ml.envs.hai_gym import HAIGym

env = HAIGym(
    schema_path='hai_ml/schemas/p3.yaml',
    attack_type='hostile',
    attack_params={'adversary_magnitude': 0.5}
)
```

## Metrics

All metrics computed by `hai_ml/eval/metrics.py`:

| Metric | Description |
|--------|-------------|
| ITAE | Integrated Time-weighted Absolute Error |
| ISE | Integrated Squared Error |
| Overshoot | Maximum positive deviation from setpoint |
| Settling | Time to reach ±2% of setpoint |
| Wear | Actuator wear proxy (sum of action changes) |
| Energy | Total control effort |
| Violations | Safety constraint violations |
| Interventions | Shield intervention count/rate |
| Latency | Decision time (P50, P99) |

## Generated Artifacts

### CSV Files

| File | Description |
|------|-------------|
| `results/<proc>_offline_leaderboard.csv` | OPE rankings with FQE, WIS, CIs |
| `results/<proc>_nominal.csv` | Online evaluation metrics |
| `results/<proc>_attacks.csv` | Attack scenario results |
| `results/<proc>_cross_version.csv` | Cross-version transfer |

### Figures

| Figure | Description |
|--------|-------------|
| `paper/figs/fig_<proc>_ope_scatter.pdf` | OPE vs realized return |
| `paper/figs/fig_<proc>_latency_cdf.pdf` | Decision latency CDF |
| `paper/figs/fig_<proc>_interventions.pdf` | Shield intervention analysis |

### LaTeX Tables

Generated via `./scripts/make_tables.sh`:

- `paper/tables/tab_offline_leaderboard.tex`
- `paper/tables/tab_online_performance.tex`
- `paper/tables/tab_attack_robustness.tex`
- `paper/tables/tab_cross_version.tex`

## Configuration

Edit YAML configs in `hai_ml/configs/` to customize:

```yaml
# hai_ml/configs/p3_baseline.yml
offline_rl:
  algorithms:
    - name: "td3bc"
      params:
        alpha: 2.5
    - name: "cql"
      params:
        alpha: 5.0

ope:
  bootstrap:
    n_bootstrap: 200
    confidence_level: 0.95

shield:
  enabled: true
  max_delta_per_step: 0.1
```

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.2
- d3rlpy ≥ 2.6
- Gymnasium ≥ 0.29
- NumPy, Pandas, Matplotlib, Seaborn
- ruamel.yaml ≥ 0.18

See `requirements.txt` for full dependencies.

## Citation

If you use this evaluation stack, please cite:

```bibtex
@inproceedings{hai-ml-2025,
  title={Safe Offline Reinforcement Learning for Industrial Control Systems},
  author={...},
  booktitle={Proceedings of ...},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.
