#!/usr/bin/env bash
# Full pipeline: data -> training (3 seeds) -> evaluation -> index -> report.
# ~1 hour on an Apple Silicon laptop (most of it the Sentinel-2 fetch).
set -euo pipefail
cd "$(dirname "$0")/.."
CLI=".venv/bin/python -W ignore -m satterrain.cli"
mkdir -p artifacts/results

$CLI download-eurosat
$CLI download-oscd
$CLI fetch --locations 3000 --dates 5 --workers 40
$CLI harmonize

for s in 0 1 2; do
  out=artifacts/seco; [ "$s" -gt 0 ] && out=artifacts/seco_seed$s
  $CLI train --epochs 100 --batch-size 128 --eval-every 5 --seed "$s" --out "$out" > "artifacts/train_seed$s.log" 2>&1
  $CLI evaluate --checkpoint "$out/encoder.pt" --out "artifacts/results/retrieval_seed$s.json"
done

$CLI evaluate-change --quantile 0.99 --out artifacts/results/change.json
$CLI evaluate-change --quantile 0.95 --out artifacts/results/change_q95.json
$CLI build-index --checkpoint artifacts/seco/encoder.pt
$CLI report
.venv/bin/python -m pytest -q tests
