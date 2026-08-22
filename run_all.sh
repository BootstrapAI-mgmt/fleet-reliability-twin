#!/usr/bin/env bash
set -euo pipefail
python3 simulate/generate_fleet.py
python3 run.py
python3 verify.py
python3 -m pytest -q tests
octave --no-gui -q --eval "addpath('matlab'); addpath('matlab/tests'); run_tests"
