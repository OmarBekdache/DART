#!/usr/bin/env bash
# Run the main ResNet-18 comparison sequentially: 220 full training runs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATA_ROOT="${1:-data}"
RESULTS_ROOT="${2:-results}"

run() {
  local mode="$1" method="$2" alpha="$3" seed="$4"
  python main.py \
    --data_root "$DATA_ROOT" \
    --run_name "$RESULTS_ROOT/$mode/$method/alpha$alpha/seed$seed" \
    --server_method "$mode" --method "$method" \
    --non_iid_dirichlet_alpha "$alpha" --seed "$seed"
}

for alpha in 100 10 1 0.1; do
  for seed in 0 1 2 3 4; do
    for method in FedAVG FedProx FedDyn FedNova; do
      run None "$method" "$alpha" "$seed"
      run DART "$method" "$alpha" "$seed"
    done
    for method in FedAFA FedAugMix FedPrime; do
      run None "$method" "$alpha" "$seed"
    done
  done
done
