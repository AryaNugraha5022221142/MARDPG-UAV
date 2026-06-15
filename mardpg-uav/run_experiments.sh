#!/usr/bin/env bash
# run_experiments.sh — multi-seed campaign for the corrected variant comparison.
#
# Implements the experimental-validity fixes: >=5 seeds per variant, all four
# 2x2 cells (including the paper's Ind-RDPG), plus the curriculum ablation
# (--no-curriculum trains directly on the final stage at equal budget).
#
# Run from the repo root on a GPU box. Requires config.algorithm.device: cuda
# and abort_if_cpu: true so a silent CPU fallback fails loudly.
set -euo pipefail

CONFIG="config/default.yaml"
OUT_ROOT="checkpoints"
MAX_EPISODES=20000
SEEDS=(0 1 2 3 4)

# Four cells of the (recurrent x centralized) design.
#   mardpg   = R  + centralized   (proposed)
#   maddpg   = FF + centralized   (paper baseline)
#   ind_rdpg = R  + independent    (paper baseline: Ind-RDPG)   <-- was missing
#   iddpg    = FF + independent    (bonus cell, completes the 2x2)
VARIANTS=(mardpg maddpg ind_rdpg iddpg)

echo "=== Curriculum runs ==="
for v in "${VARIANTS[@]}"; do
  for s in "${SEEDS[@]}"; do
    run="cl_${v}_seed${s}"
    echo ">>> $run"
    python -m mardpg_uav.train \
      --config   "$CONFIG" \
      --variant  "$v" \
      --seed     "$s" \
      --out-dir  "${OUT_ROOT}/${run}" \
      --run-name "$run" \
      --max-episodes "$MAX_EPISODES"
  done
done

echo "=== No-curriculum ablation (train directly on final stage) ==="
# Only needed for the proposed method + the strongest baseline to make the
# "curriculum helps" claim falsifiable at equal compute.
for v in mardpg maddpg; do
  for s in "${SEEDS[@]}"; do
    run="nocl_${v}_seed${s}"
    echo ">>> $run"
    python -m mardpg_uav.train \
      --config   "$CONFIG" \
      --variant  "$v" \
      --seed     "$s" \
      --no-curriculum \
      --out-dir  "${OUT_ROOT}/${run}" \
      --run-name "$run" \
      --max-episodes "$MAX_EPISODES"
  done
done

echo "All runs submitted. Aggregate with mean +/- 95% CI across seeds per variant."
