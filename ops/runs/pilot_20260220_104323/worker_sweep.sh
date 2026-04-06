#!/usr/bin/env bash
set -euo pipefail

cd /opt/neural_pool/pilot_20260220_104323/data_generation_package
source /opt/neural_pool/pilot_20260220_104323/venv/bin/activate

python --version
nproc

for w in 32 48 64 80 96; do
  echo "BENCH_START workers=${w}"
  OUT="/dev/shm/bench_w_${w}"
  rm -rf "$OUT"
  START=$(date +%s)
  python generate_pool_shards.py \
    --episodes 1000 \
    --shard-size 100 \
    --workers "$w" \
    --frames 600 \
    --dry-run \
    --output-dir "$OUT" > "/tmp/bench_w_${w}.log" 2>&1
  END=$(date +%s)
  DUR=$((END-START))
  AVG=$(grep -E "Average speed:" "/tmp/bench_w_${w}.log" | tail -n1 | awk '{print $3}')
  TOT=$(grep -E "Total time:" "/tmp/bench_w_${w}.log" | tail -n1 | awk '{print $3}')
  echo "BENCH_RESULT workers=${w} duration_sec=${DUR} avg_eps=${AVG:-NA} total_min=${TOT:-NA}"
  rm -rf "$OUT"
done
