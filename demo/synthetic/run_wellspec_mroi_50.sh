#!/bin/bash
# Two-channel mROI sweep over the well-specified arm's 50 saved posteriors.
#
# No MCMC: each seed's scenario is rebuilt (~2 min) so the saved posterior can
# be reattached and the true curve recomputed. Batched ten seeds per process
# for the same reason the fitting run is -- fifty scenario builds in one kernel
# is the shape of workload that has been OOM-killed here before.
#
# --realism=wellspec is REQUIRED and is the whole point: the rebuilt scenario
# supplies the data these posteriors are reattached to. Building the realistic
# baseline instead would pair well-specified fits with realistic data and yield
# plausible, wrong numbers.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
S=demo/synthetic/scratch_mroi_both_channels_r90.py
RUN=demo/synthetic/fitted_models/scratch_ablation_r90_wellspec

BATCHES=(
  "1320,7,42,8,99,101,555,2024,12345,31337"
  "21877,61225,90033,93612,28214,99768,938,52463,96275,88982"
  "36452,33080,16847,6928,32325,2559,44329,15510,32226,71064"
  "62256,84275,89066,38578,80694,42975,41172,28767,3573,86071"
  "77175,49024,2016,93750,54351,35532,78122,55463,57719,61686"
)

for i in "${!BATCHES[@]}"; do
  echo "=== MROI BATCH $((i + 1))/5 : ${BATCHES[$i]}"
  $PY "$S" --run-dir="$RUN" --realism=wellspec --seeds="${BATCHES[$i]}" \
    || echo "MROI BATCH $((i + 1)) FAILED (exit $?)"
done

echo "=== assembling all seeds"
$PY "$S" --run-dir="$RUN" --realism=wellspec --assemble
