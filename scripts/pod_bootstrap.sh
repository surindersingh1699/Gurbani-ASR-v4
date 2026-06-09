#!/usr/bin/env bash
# Pod entrypoint (no inline-quoting pain): HF login, run the training runner,
# ALWAYS upload its log to runlogs/diag so we can see crashes. Stays alive after.
set +e
export HF_HUB_ENABLE_HF_TRANSFER=1
RUNLOGS=surindersinghssj/indicconformer-pa-v3-kirtan-runlogs
echo "[bootstrap] $(date -u) starting"
pip install -q "huggingface_hub<1.0" hf_transfer 2>&1 | tail -1
huggingface-cli login --token "$HF_TOKEN" 2>&1 | tail -1
huggingface-cli repo create "$(basename "$RUNLOGS")" --type dataset -y 2>/dev/null || true
TS=$(date -u +%Y%m%d-%H%M%S)
BL=/tmp/bootstrap_$TS.log
echo "[bootstrap] running run_v4_indic_train.sh (SMOKE=${SMOKE:-0})"
bash "$(dirname "$0")/run_v4_indic_train.sh" > "$BL" 2>&1
ec=$?
echo "RUNNER_EXIT=$ec" >> "$BL"
RLOG=$(ls -t /workspace/runs/*/log.txt 2>/dev/null | head -1)
[ -n "$RLOG" ] && { echo "=== runner internal log.txt ==="; cat "$RLOG"; } >> "$BL"
huggingface-cli upload --repo-type dataset "$RUNLOGS" "$BL" "diag/bootstrap_$TS.log" --create-pr=false 2>&1 | tail -2
echo "[bootstrap] uploaded diag/bootstrap_$TS.log (runner exit=$ec)"
[ "${KEEP_ALIVE:-1}" = "1" ] && { echo "[bootstrap] sleeping for inspection"; sleep infinity; }
