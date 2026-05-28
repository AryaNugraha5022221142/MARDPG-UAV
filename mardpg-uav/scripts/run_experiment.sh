#!/bin/bash
# Run experiment
cd "$(dirname "$0")/.."
rm -rf checkpoints/
PYTHONPATH=. python -m mardpg_uav.train
