#!/bin/bash
# Extend the well-specified-baseline arm from 10 draws to 50.
#
# Batched ten seeds (twenty fits) per PROCESS, not one process for all forty:
# that is the workload the existing 10-seed runs are known to complete, and
# CLAUDE.md records that ~forty fits in a single kernel exhausts memory and is
# killed. Each batch writes its own per-seed CSVs, so a batch that dies loses
# only its own draws and can be re-run with the same --seeds= argument.
#
# The original ten seeds are NOT re-run -- scratch_ablation_r90_wellspec.py has
# no already-done check, so passing them would refit and overwrite them. Only
# r90_basis.SEEDS_EXTRA_40 is passed here; report() globs the directory and
# picks up all fifty.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
S=demo/synthetic/scratch_ablation_r90_wellspec.py

BATCHES=(
  "21877,61225,90033,93612,28214,99768,938,52463,96275,88982"
  "36452,33080,16847,6928,32325,2559,44329,15510,32226,71064"
  "62256,84275,89066,38578,80694,42975,41172,28767,3573,86071"
  "77175,49024,2016,93750,54351,35532,78122,55463,57719,61686"
)

for i in "${!BATCHES[@]}"; do
  echo "=== BATCH $((i + 1))/4 : ${BATCHES[$i]}"
  $PY "$S" --seeds="${BATCHES[$i]}" || echo "BATCH $((i + 1)) FAILED (exit $?)"
done

echo "=== all batches done, assembling report over every seed present"
$PY "$S" --report
