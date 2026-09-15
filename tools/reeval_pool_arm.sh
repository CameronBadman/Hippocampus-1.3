#!/usr/bin/env bash
# Step 12(d): the Rust engine re-evaluates the Python pool-arm checkpoints (exported
# to safetensors) under raw-v5 on the real rung-3 splits; the Python reader then
# validates the rows against the training run's (zero differing episodes, or the
# numerics note). Usage: reeval_pool_arm.sh <pin>
set -u; PIN=${1:?pin}; F=/home/cameron/entor/hippocampus-foundation; H=/home/cameron/entor/hippo-13
RUNS=$F/private/real-walk-v1/runs; BASE=stage0-ladder-r3-g0.25-prior-40k
SPL=$F/private/real-walk-v1/splits/wikidata5m-n40-d3-k2-r3-g0.25
S800=$F/private/real-walk-v1/splits/wikidata5m-n40-d3-k2-r3-greedy-screen800
VAULT=$F/private/real-walk-v1/splits/vault-quartz-docs-n40-d3-k2-r3-g0.25
EMB=$F/private/real-walk-v1/embeddings/wikidata5m/nomic-embed-text
VEMB=$F/private/real-walk-v1/embeddings/vault-quartz-docs/nomic-embed-text
for seed in 1729 2718 3141; do
  out=$RUNS/rust-reeval-pool-arm/s$seed; [ -e "$out" ] && { echo "refusing: $out exists"; continue; }
  echo "$(date +%H:%M) seed $seed"; t0=$(date +%s)
  OMP_NUM_THREADS=8 $H/target/release/hf-stage0 --config $F/experiments/real_walk_v1/training-config.stage0.wikidata5m-r3-g0.25-prior.json \
    --output "$out" --model-seed $seed --family wikidata5m --splits-dir "$SPL" --screen-episodes 400 --embeddings-dir "$EMB" \
    --heldout-family vault-quartz-docs --heldout-splits-dir "$VAULT" --heldout-embeddings-dir "$VEMB" \
    --screen2-splits-dir "$S800" --screen2-skip 400 --preregistration-commit "$PIN" --foundation-root "$F" \
    --reevaluate-checkpoint "$RUNS/$BASE-s$seed/rust-export/checkpoint.json" --deterministic
  echo "$(date +%H:%M) seed $seed exit $? in $(( $(date +%s) - t0 )) s"
  (cd $F && uv run python scripts/real_walk_stage0_report.py "$out" --departures screen --split-dir "$SPL/screen" --validate-against "$RUNS/$BASE-s$seed" 2>&1 | grep -i -E "validat|differing|MISMATCH|numerics|identical" | head -5)
done
