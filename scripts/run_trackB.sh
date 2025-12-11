#!/bin/bash
# ==============================================================================
# Track B: Offline RL Pipeline
# ==============================================================================
# Trains TD3+BC, CQL, IQL, runs OPE, and evaluates admitted policies.
#
# Usage:
#   ./scripts/run_trackB.sh [process] [--attacks] [--skip-train] [--algo ALGO]
#
# Arguments:
#   process       Process to run: p1, p3, p12 (default: p3)
#   --attacks     Also run attack evaluation
#   --skip-train  Skip training, only run evaluation
#   --algo ALGO   Train/eval only specific algorithm (td3bc, cql, iql)
#
# Examples:
#   ./scripts/run_trackB.sh p3
#   ./scripts/run_trackB.sh p1 --attacks
#   ./scripts/run_trackB.sh p12 --algo cql
# ==============================================================================

set -e

# Parse arguments
PROCESS=${1:-p3}
RUN_ATTACKS=false
SKIP_TRAIN=false
SINGLE_ALGO=""

shift || true
for arg in "$@"; do
    case $arg in
        --attacks)
            RUN_ATTACKS=true
            ;;
        --skip-train)
            SKIP_TRAIN=true
            ;;
        --algo)
            shift
            SINGLE_ALGO=$1
            ;;
    esac
    shift || true
done

# Algorithms to run
if [ -n "$SINGLE_ALGO" ]; then
    ALGORITHMS=("$SINGLE_ALGO")
else
    ALGORITHMS=("td3bc" "cql" "iql")
fi

echo "=============================================="
echo "Track B: Offline Reinforcement Learning"
echo "Process: $PROCESS"
echo "Algorithms: ${ALGORITHMS[*]}"
echo "=============================================="

# Activate virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi

# Configuration
CONFIG="hai_ml/configs/${PROCESS}_baseline.yml"
SCHEMA="hai_ml/schemas/${PROCESS}.yaml"

# Create directories
mkdir -p models/${PROCESS}/offline_rl
mkdir -p logs/${PROCESS}/offline_rl
mkdir -p results

# Step 1: Build dataset
echo ""
echo "[Step 1/6] Building dataset..."
python -m hai_ml.data.build_dataset \
    --csv "data/hai/train_${PROCESS}.csv" \
    --schema "$SCHEMA" \
    --output "data/processed/${PROCESS}/train.npz"

python -m hai_ml.data.build_dataset \
    --csv "data/hai/val_${PROCESS}.csv" \
    --schema "$SCHEMA" \
    --output "data/processed/${PROCESS}/val.npz"

# Step 2: Train BC baseline (needed for OPE comparison)
if [ "$SKIP_TRAIN" = false ]; then
    echo ""
    echo "[Step 2/6] Training BC baseline for OPE comparison..."
    python -m hai_ml.il.train_bc \
        --dataset "data/processed/${PROCESS}/train.npz" \
        --val-dataset "data/processed/${PROCESS}/val.npz" \
        --output "models/${PROCESS}/bc/policy.pt" \
        --epochs 100
fi

# Step 3: Train offline RL algorithms
if [ "$SKIP_TRAIN" = false ]; then
    echo ""
    echo "[Step 3/6] Training offline RL algorithms..."
    
    for algo in "${ALGORITHMS[@]}"; do
        echo "  Training: $algo"
        python -m hai_ml.rl.train_offline \
            --dataset "data/processed/${PROCESS}/train.npz" \
            --algo "$algo" \
            --n-steps 100000 \
            --output "models/${PROCESS}/offline_rl/${algo}.d3"
    done
else
    echo ""
    echo "[Step 2-3/6] Skipping training (--skip-train)"
fi

# Step 4: Run OPE and admission gating
echo ""
echo "[Step 4/6] Running Off-Policy Evaluation..."

# Collect all model paths
MODEL_PATHS=""
for algo in "${ALGORITHMS[@]}"; do
    MODEL_PATHS="$MODEL_PATHS models/${PROCESS}/offline_rl/${algo}.d3"
done

python -m hai_ml.rl.ope_gate \
    --dataset "data/processed/${PROCESS}/val.npz" \
    --bc-model "models/${PROCESS}/bc/policy.pt" \
    --rl-models $MODEL_PATHS \
    --output "results/${PROCESS}_offline_leaderboard.csv"

# Step 5: Online evaluation of admitted policies
echo ""
echo "[Step 5/6] Running online evaluation of admitted policies..."

# Read admitted policies from leaderboard
ADMITTED=$(python3 -c "
import pandas as pd
df = pd.read_csv('results/${PROCESS}_offline_leaderboard.csv')
admitted = df[df['admit'] == True]['algo'].tolist()
print(' '.join(admitted))
")

echo "Admitted policies: $ADMITTED"

for algo in $ADMITTED; do
    echo "  Evaluating: $algo"
    python -m hai_ml.eval.run_eval \
        --config "$CONFIG" \
        --model "models/${PROCESS}/offline_rl/${algo}.d3" \
        --algo "$algo" \
        --n-episodes 50 \
        --output "results/${PROCESS}_${algo}_nominal.csv"
done

# Also evaluate BC for comparison
python -m hai_ml.eval.run_eval \
    --config "$CONFIG" \
    --model "models/${PROCESS}/bc/policy.pt" \
    --algo bc \
    --n-episodes 50 \
    --output "results/${PROCESS}_bc_nominal.csv"

# Merge results
echo "  Merging results..."
python3 -c "
import pandas as pd
import glob

files = glob.glob('results/${PROCESS}_*_nominal.csv')
dfs = [pd.read_csv(f) for f in files]
merged = pd.concat(dfs, ignore_index=True)
merged.to_csv('results/${PROCESS}_nominal.csv', index=False)
print(f'Merged {len(files)} result files')
"

# Step 6: Attack evaluation (optional)
if [ "$RUN_ATTACKS" = true ]; then
    echo ""
    echo "[Step 6/6] Running attack evaluation..."
    
    for algo in $ADMITTED bc; do
        if [ "$algo" = "bc" ]; then
            MODEL_PATH="models/${PROCESS}/bc/policy.pt"
        else
            MODEL_PATH="models/${PROCESS}/offline_rl/${algo}.d3"
        fi
        
        for attack in hostile flood bias delay; do
            echo "  $algo + $attack"
            python -m hai_ml.eval.run_eval \
                --config "$CONFIG" \
                --model "$MODEL_PATH" \
                --algo "$algo" \
                --attack-type "$attack" \
                --n-episodes 20 \
                --output "results/${PROCESS}_${algo}_${attack}.csv"
        done
    done
    
    # Merge attack results
    python3 -c "
import pandas as pd
import glob

files = glob.glob('results/${PROCESS}_*_hostile.csv') + \
        glob.glob('results/${PROCESS}_*_flood.csv') + \
        glob.glob('results/${PROCESS}_*_bias.csv') + \
        glob.glob('results/${PROCESS}_*_delay.csv')
dfs = [pd.read_csv(f) for f in files]
merged = pd.concat(dfs, ignore_index=True)
merged.to_csv('results/${PROCESS}_attacks.csv', index=False)
print(f'Merged {len(files)} attack result files')
"
else
    echo ""
    echo "[Step 6/6] Skipping attack evaluation (use --attacks to enable)"
fi

# Generate plots
echo ""
echo "=============================================="
echo "Generating plots..."
echo "=============================================="

python -m hai_ml.eval.plot_ope_scatter \
    --offline "results/${PROCESS}_offline_leaderboard.csv" \
    --online "results/${PROCESS}_nominal.csv" \
    --out "paper/figs/fig_${PROCESS}_ope_scatter.pdf"

python -m hai_ml.eval.plot_latency_cdf \
    --input "results/${PROCESS}_nominal.csv" \
    --out "paper/figs/fig_${PROCESS}_latency_cdf.pdf" \
    --synthetic

if [ "$RUN_ATTACKS" = true ]; then
    python -m hai_ml.eval.plot_interventions \
        --input "results/${PROCESS}_attacks.csv" \
        --out "paper/figs/fig_${PROCESS}_interventions.pdf"
fi

echo ""
echo "=============================================="
echo "Track B Complete!"
echo "=============================================="
echo ""
echo "Results:"
echo "  - results/${PROCESS}_offline_leaderboard.csv  (OPE rankings)"
echo "  - results/${PROCESS}_nominal.csv              (Online evaluation)"
if [ "$RUN_ATTACKS" = true ]; then
    echo "  - results/${PROCESS}_attacks.csv              (Attack evaluation)"
fi
echo ""
echo "Figures:"
echo "  - paper/figs/fig_${PROCESS}_ope_scatter.pdf"
echo "  - paper/figs/fig_${PROCESS}_latency_cdf.pdf"
if [ "$RUN_ATTACKS" = true ]; then
    echo "  - paper/figs/fig_${PROCESS}_interventions.pdf"
fi
echo ""
