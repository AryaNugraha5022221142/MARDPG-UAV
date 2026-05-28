#!/bin/bash
# Run experiment
cd "$(dirname "$0")/.."
PYTHONPATH=. python -m mardpg_uav.train
