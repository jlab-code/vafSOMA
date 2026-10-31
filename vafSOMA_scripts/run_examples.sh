#!/usr/bin/env bash
set -euo pipefail
# Run both example configurations on the synthetic example tree (results in vafSOMA_output/).
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for c in example_01_per_leaf example_02_replicate_groups; do
  bash "${PROJECT_ROOT}/vafSOMA_scripts/run_config_local.sh" "${PROJECT_ROOT}/vafSOMA_configs/${c}.json"
done
