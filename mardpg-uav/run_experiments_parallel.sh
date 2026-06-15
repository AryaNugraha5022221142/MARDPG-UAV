#!/usr/bin/env bash
# run_experiments_parallel.sh
# 
# Runs multi-seed experiments in parallel.
# WARNING: Do not run all 20 jobs at once on a single GPU. It will OOM.
# This script uses xargs or simple bash backgrounding to limit parallel jobs.

set -euo pipefail

CONFIG="config/default.yaml"
OUT_ROOT="checkpoints"
MAX_EPISODES=20000
SEEDS=(0 1 2 3 4)
VARIANTS=(mardpg maddpg ind_rdpg iddpg)

# Set max concurrent jobs (e.g., 2 or 3 depending on your GPU VRAM & Cores)
MAX_CONCURRENT=3

echo "=== Running Jobs in Parallel ($MAX_CONCURRENT max format) ==="
mkdir -p logs

job_list="jobs.txt"
rm -f $job_list

# Generate the list of commands
for v in "${VARIANTS[@]}"; do
  for s in "${SEEDS[@]}"; do
    run="cl_${v}_seed${s}"
    echo "python -m mardpg_uav.train --config $CONFIG --variant $v --seed $s --out-dir ${OUT_ROOT}/${run} --run-name $run --max-episodes $MAX_EPISODES > logs/${run}.log 2>&1" >> $job_list
  done
done

for v in mardpg maddpg; do
  for s in "${SEEDS[@]}"; do
    run="nocl_${v}_seed${s}"
    echo "python -m mardpg_uav.train --config $CONFIG --variant $v --seed $s --no-curriculum --out-dir ${OUT_ROOT}/${run} --run-name $run --max-episodes $MAX_EPISODES > logs/${run}.log 2>&1" >> $job_list
  done
done

echo "Total jobs: $(wc -l < $job_list)"
echo "Starting execution with $MAX_CONCURRENT concurrent workers..."
echo "Monitor progress: tail -f logs/*.log or watch nvidia-smi"

# Use xargs to run commands in parallel, utilizing MAX_CONCURRENT processes
xargs -max-procs=$MAX_CONCURRENT -I CMD bash -c 'CMD' < $job_list

echo "All parallel runs completed!"
