#!/bin/bash
# Watch training jobs (multitab) and auto-submit test jobs on completion.
# Run inside tmux: bash /home/ranema/TabPredict/watch_and_test.sh
#
# Logic per experiment:
#   1. test_output_dir contains metrics.json  → already tested, skip
#   2. job_name still in squeue               → still training, wait
#   3. best_stage_b.pt exists in ckpt_dir     → training done, submit test
#   4. none of above                          → training failed / not started, skip

set -euo pipefail

POLL=1800   # seconds between polls
LOG="/home/ranema/TabPredict/watch_and_test.log"
DONE_FILE="/home/ranema/TabPredict/.watch_tested"  # one tested experiment per line

MT_REPO="/home/ranema/TabPredict"

CKPT_BASE_MT="/deepstore/datasets/mia/TabPredict/models_fbeta"
TEST_BASE_MT="/home/ranema/TabPredict/test_results_fbeta"

# Format: "key|repo|config|job_name|ckpt_dir|test_output_dir"
EXPERIMENTS=(
    # ── multitab ─────────────────────────────────────────────────────────────
    "mt_lr|$MT_REPO|experiments/logistic_regression.yaml|logistic_regression_fbeta|$CKPT_BASE_MT/logistic_regression|$TEST_BASE_MT/logistic_regression"
    "mt_rf|$MT_REPO|experiments/random_forest.yaml|random_forest_fbeta|$CKPT_BASE_MT/random_forest|$TEST_BASE_MT/random_forest"
    "mt_gb|$MT_REPO|experiments/gradient_boosting.yaml|gradient_boosting_fbeta|$CKPT_BASE_MT/gradient_boosting|$TEST_BASE_MT/gradient_boosting"
    "mt_xgb|$MT_REPO|experiments/xgboost.yaml|xgboost_fbeta|$CKPT_BASE_MT/xgboost|$TEST_BASE_MT/xgboost"
    "mt_mlp|$MT_REPO|experiments/mlp.yaml|MLP_fbeta|$CKPT_BASE_MT/MLP|$TEST_BASE_MT/MLP"
    "mt_tabm|$MT_REPO|experiments/tabm.yaml|TabM_fbeta|$CKPT_BASE_MT/TabM|$TEST_BASE_MT/TabM"
    "mt_tabpfn|$MT_REPO|experiments/tabpfn.yaml|TabPFN_fbeta|$CKPT_BASE_MT/TabPFN|$TEST_BASE_MT/TabPFN"
    "mt_saint|$MT_REPO|experiments/saint.yaml|SAINT_fbeta|$CKPT_BASE_MT/SAINT|$TEST_BASE_MT/SAINT"
)

touch "$DONE_FILE"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

job_running() {
    local name="$1"
    squeue -u "$USER" -n "$name" --noheader 2>/dev/null | grep -q .
}

already_tested() {
    local key="$1" test_dir="$2"
    grep -qx "$key" "$DONE_FILE" 2>/dev/null && return 0
    [[ -f "$test_dir/test/metrics.json" ]] && return 0
    return 1
}

submit_test() {
    local key="$1" repo="$2" config="$3" test_dir="$4"
    local job_id
    job_id=$(cd "$repo" && sbatch --job-name="test_${key}" --parsable \
        scripts/submit_slurm.sh test \
        --config "$config" \
        --output_dir "$test_dir" \
        --split all --no-gradcam)
    log "[$key] test submitted → job $job_id  output: $test_dir"
    echo "$key" >> "$DONE_FILE"
}

log "=== watch_and_test started (${#EXPERIMENTS[@]} experiments, poll=${POLL}s) ==="

while true; do
    all_done=true

    for entry in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r key repo config job_name ckpt_dir test_dir <<< "$entry"

        if already_tested "$key" "$test_dir"; then
            continue
        fi

        all_done=false

        if job_running "$job_name"; then
            log "[$key] training running…"
            continue
        fi

        # Job gone from squeue — check checkpoint
        if [[ -f "$ckpt_dir/best_stage_b.pt" ]]; then
            submit_test "$key" "$repo" "$config" "$test_dir"
        elif [[ -f "$ckpt_dir/best_stage_a.pt" ]]; then
            # stage-A-only models (classical backbones sometimes stop at A)
            submit_test "$key" "$repo" "$config" "$test_dir"
        else
            log "[$key] no checkpoint found — training failed or not started, skipping"
            echo "$key" >> "$DONE_FILE"  # don't keep retrying a failed job
        fi
    done

    if $all_done; then
        log "=== all experiments tested — exiting ==="
        break
    fi

    log "--- sleeping ${POLL}s ---"
    sleep "$POLL"
done
