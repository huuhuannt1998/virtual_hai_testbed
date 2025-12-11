#!/bin/bash
# ==============================================================================
# Track A: Behavior Cloning Pipeline
# ==============================================================================
# Trains and evaluates Behavior Cloning baseline.
#
# Usage:
#   ./scripts/run_trackA.sh [process] [--attacks] [--skip-train]
#
# Arguments:
#   process     Process to run: p1, p3, p12 (default: p3)
#   --attacks   Also run attack evaluation
#   --skip-train Skip training, only run evaluation
#
# Examples:
#   ./scripts/run_trackA.sh p3
#   ./scripts/run_trackA.sh p1 --attacks
#   ./scripts/run_trackA.sh p12 --skip-train
# ==============================================================================

set -e

# Parse arguments
PROCESS=${1:-p3}
RUN_ATTACKS=false
SKIP_TRAIN=false

for arg in "$@"; do
    case $arg in
        --attacks)
            RUN_ATTACKS=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
    esac
done

echo "=============================================="
echo "Track A: Behavior Cloning"
echo "Process: $PROCESS"
echo "=============================================="

# Activate virtual environment if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi

# Configuration file
CONFIG="hai_ml/configs/${PROCESS}_baseline.yml"
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Configuration file not found: $CONFIG"
    exit 1
fi

# Create output directories
mkdir -p models/${PROCESS}/bc
mkdir -p logs/${PROCESS}/bc
mkdir -p results

# Step 1: Build dataset
echo ""
echo "[Step 1/4] Building dataset..."
python -m hai_ml.data.build_dataset \
    --csv "data/hai/train_${PROCESS}.csv" \
    --schema "hai_ml/schemas/${PROCESS}.yaml" \
    --output "data/processed/${PROCESS}/train.npz"

python -m hai_ml.data.build_dataset \
    --csv "data/hai/val_${PROCESS}.csv" \
    --schema "hai_ml/schemas/${PROCESS}.yaml" \
    --output "data/processed/${PROCESS}/val.npz"

# Step 2: Train BC
if [ "$SKIP_TRAIN" = false ]; then
    echo ""
    echo "[Step 2/4] Training Behavior Cloning..."
    python -m hai_ml.il.train_bc \
        --dataset "data/processed/${PROCESS}/train.npz" \
        --val-dataset "data/processed/${PROCESS}/val.npz" \
        --output "models/${PROCESS}/bc/policy.pt" \
        --hidden-dims 256 256 256 \
        --epochs 100 \
        --batch-size 256 \
        --lr 3e-4
else
    echo ""
    echo "[Step 2/4] Skipping training (--skip-train)"
fi

# Step 3: Online evaluation (nominal)
echo ""
echo "[Step 3/4] Running online evaluation (nominal)..."
python -m hai_ml.eval.run_eval \
    --config "$CONFIG" \
    --model "models/${PROCESS}/bc/policy.pt" \
    --algo bc \
    --n-episodes 50 \
    --output "results/${PROCESS}_bc_nominal.csv"

# Step 4: Attack evaluation (optional)
if [ "$RUN_ATTACKS" = true ]; then
    echo ""
    echo "[Step 4/4] Running attack evaluation..."
    
    for attack in hostile flood bias delay; do
        echo "  Running attack: $attack"
        python -m hai_ml.eval.run_eval \
            --config "$CONFIG" \
            --model "models/${PROCESS}/bc/policy.pt" \
            --algo bc \
            --attack-type "$attack" \
            --n-episodes 20 \
            --output "results/${PROCESS}_bc_${attack}.csv"
    done
else
    echo ""
    echo "[Step 4/4] Skipping attack evaluation (use --attacks to enable)"
fi

# Generate summary
echo ""
echo "=============================================="
echo "Track A Complete!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - results/${PROCESS}_bc_nominal.csv"
if [ "$RUN_ATTACKS" = true ]; then
    echo "  - results/${PROCESS}_bc_hostile.csv"
    echo "  - results/${PROCESS}_bc_flood.csv"
    echo "  - results/${PROCESS}_bc_bias.csv"
    echo "  - results/${PROCESS}_bc_delay.csv"
fi
echo ""
echo "Model saved to:"
echo "  - models/${PROCESS}/bc/policy.pt"
echo ""
