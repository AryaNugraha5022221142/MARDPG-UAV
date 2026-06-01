#!/bin/bash
# run_experiment.sh
cd "$(dirname "$0")/.."

for seed in 42 100 2026; do
    echo "Starting training with seed $seed"
    PYTHONPATH=. python -m mardpg_uav.train --seed $seed --config config/default.yaml
    # Rename checkpoints folder so it isn't overwritten
    if [ -d "checkpoints" ]; then
        mv checkpoints checkpoints_seed_$seed
    fi
done
