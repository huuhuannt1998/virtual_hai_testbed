#!/bin/bash
# ==============================================================================
# Track C: Model Predictive Control Pipeline
# ==============================================================================
# Trains dynamics model and runs CEM-based MPC evaluation.
#
# Usage:
#   ./scripts/run_trackC.sh [process] [--attacks] [--skip-train]
#
# Arguments:
#   process       Process to run: p1, p3, p12 (default: p3)
#   --attacks     Also run attack evaluation
#   --skip-train  Skip training, only run evaluation
#
# Examples:
#   ./scripts/run_trackC.sh p3
#   ./scripts/run_trackC.sh p1 --attacks
# ==============================================================================

set -e

# Parse arguments
PROCESS=${1:-p3}
RUN_ATTACKS=false
SKIP_TRAIN=false

shift || true
for arg in "$@"; do
    case $arg in
        --attacks)
            RUN_ATTACKS=true
            ;;
        --skip-train)
            SKIP_TRAIN=true
            ;;
    esac
    shift || true
done

echo "=============================================="
echo "Track C: Model Predictive Control"
echo "Process: $PROCESS"
echo "=============================================="

# Activate virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi

# Configuration
CONFIG="hai_ml/configs/${PROCESS}_baseline.yml"
SCHEMA="hai_ml/schemas/${PROCESS}.yaml"

# Create directories
mkdir -p models/${PROCESS}/mpc
mkdir -p logs/${PROCESS}/mpc
mkdir -p results

# Step 1: Build dataset
echo ""
echo "[Step 1/4] Building dataset..."
python -m hai_ml.data.build_dataset \
    --csv "data/hai/train_${PROCESS}.csv" \
    --schema "$SCHEMA" \
    --output "data/processed/${PROCESS}/train.npz"

# Step 2: Train dynamics model
if [ "$SKIP_TRAIN" = false ]; then
    echo ""
    echo "[Step 2/4] Training dynamics model..."
    python -m hai_ml.mpc.dyn_model \
        --dataset "data/processed/${PROCESS}/train.npz" \
        --output "models/${PROCESS}/mpc/dynamics.pt" \
        --hidden-dims 256 256 \
        --epochs 50 \
        --batch-size 256 \
        --lr 1e-3
else
    echo ""
    echo "[Step 2/4] Skipping dynamics training (--skip-train)"
fi

# Step 3: Run MPC evaluation (nominal)
echo ""
echo "[Step 3/4] Running MPC evaluation (nominal)..."

# Load schema for state/action dims
STATE_DIM=$(python3 -c "
from ruamel.yaml import YAML
yaml = YAML()
with open('$SCHEMA') as f:
    cfg = yaml.load(f)
print(len(cfg['states']))
")

ACTION_DIM=$(python3 -c "
from ruamel.yaml import YAML
yaml = YAML()
with open('$SCHEMA') as f:
    cfg = yaml.load(f)
print(len(cfg['actions']))
")

echo "State dim: $STATE_DIM, Action dim: $ACTION_DIM"

python -m hai_ml.mpc.plan_cem \
    --dynamics-model "models/${PROCESS}/mpc/dynamics.pt" \
    --schema "$SCHEMA" \
    --state-dim "$STATE_DIM" \
    --action-dim "$ACTION_DIM" \
    --n-episodes 50 \
    --horizon 10 \
    --n-candidates 500 \
    --output "results/${PROCESS}_cem_nominal.csv"

# Step 4: Attack evaluation (optional)
if [ "$RUN_ATTACKS" = true ]; then
    echo ""
    echo "[Step 4/4] Running MPC attack evaluation..."
    
    for attack in hostile flood bias delay; do
        echo "  Running attack: $attack"
        python -m hai_ml.eval.run_eval \
            --config "$CONFIG" \
            --dynamics-model "models/${PROCESS}/mpc/dynamics.pt" \
            --algo cem \
            --attack-type "$attack" \
            --n-episodes 20 \
            --output "results/${PROCESS}_cem_${attack}.csv"
    done
    
    # Merge attack results
    python3 -c "
import pandas as pd
files = [
    'results/${PROCESS}_cem_hostile.csv',
    'results/${PROCESS}_cem_flood.csv',
    'results/${PROCESS}_cem_bias.csv',
    'results/${PROCESS}_cem_delay.csv',
]
dfs = [pd.read_csv(f) for f in files]
merged = pd.concat(dfs, ignore_index=True)
merged.to_csv('results/${PROCESS}_cem_attacks.csv', index=False)
"
else
    echo ""
    echo "[Step 4/4] Skipping attack evaluation (use --attacks to enable)"
fi

echo ""
echo "=============================================="
echo "Track C Complete!"
echo "=============================================="
echo ""
echo "Results:"
echo "  - results/${PROCESS}_cem_nominal.csv"
if [ "$RUN_ATTACKS" = true ]; then
    echo "  - results/${PROCESS}_cem_attacks.csv"
fi
echo ""
echo "Model:"
echo "  - models/${PROCESS}/mpc/dynamics.pt"
echo ""
echo "Note: MPC is typically slower than learned policies."
echo "Compare latency in paper/figs/fig_${PROCESS}_latency_cdf.pdf"
echo ""
