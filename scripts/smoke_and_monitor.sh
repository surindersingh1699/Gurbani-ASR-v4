#!/bin/bash
# Smoke test + GPU utilization monitoring for Surt v3 training.
#
# Runs a short training burst (50 steps) in foreground AND captures
# nvidia-smi samples at 1s cadence. Writes both logs to /workspace/smoke_logs/
# so they survive pod stop.
#
# Usage: bash scripts/smoke_and_monitor.sh [--steps N]
#
# After run completes, check:
#   - /workspace/smoke_logs/<timestamp>/train.log
#   - /workspace/smoke_logs/<timestamp>/nvidia-smi.log
#   - /workspace/smoke_logs/<timestamp>/summary.txt  (auto-generated)

set -euo pipefail

STEPS="${1:-50}"

# Honor existing env or fall back to workspace defaults.
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/workspace/.cache/huggingface/datasets}"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_DIR="/workspace/smoke_logs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "[smoke] Logs: $LOG_DIR"
echo "[smoke] Running ${STEPS}-step smoke test with GPU monitoring..."
echo ""

# --- GPU monitor in background (1s cadence) ---
nvidia-smi \
    --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm \
    --format=csv \
    -l 1 > "$LOG_DIR/nvidia-smi.log" &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null || true" EXIT

# Startup GPU state (before any CUDA init)
echo "[smoke] Initial GPU state:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv | tee "$LOG_DIR/gpu_initial.txt"
echo ""

# --- Run smoke test ---
START_TIME=$(date +%s)
python -m surt.train --mode smoke --smoke-steps "$STEPS" 2>&1 | tee "$LOG_DIR/train.log"
EXIT_CODE=${PIPESTATUS[0]}
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Stop monitor
kill $MONITOR_PID 2>/dev/null || true
wait $MONITOR_PID 2>/dev/null || true

# --- Analyze GPU utilization ---
SUMMARY="$LOG_DIR/summary.txt"
{
    echo "=== Surt Smoke Test Summary ==="
    echo "Timestamp: $TIMESTAMP"
    echo "Steps: $STEPS"
    echo "Elapsed: ${ELAPSED}s"
    echo "Exit code: $EXIT_CODE"
    echo ""
    echo "=== GPU Utilization (during training, excluding startup) ==="
    # Skip first 10 samples (startup) + drop header row
    tail -n +12 "$LOG_DIR/nvidia-smi.log" | awk -F',' '
        NR > 0 {
            gpu_util += $3 + 0
            mem_util += $4 + 0
            mem_used += $5 + 0
            power += $7 + 0
            count++
            if ($3 + 0 > max_gpu) max_gpu = $3 + 0
            if ($5 + 0 > max_mem) max_mem = $5 + 0
        }
        END {
            if (count > 0) {
                printf "Samples: %d\n", count
                printf "Avg GPU SM util: %.1f%%\n", gpu_util/count
                printf "Max GPU SM util: %.1f%%\n", max_gpu
                printf "Avg mem BW util: %.1f%%\n", mem_util/count
                printf "Avg VRAM used: %.0f MB\n", mem_used/count
                printf "Peak VRAM used: %.0f MB\n", max_mem
                printf "Avg power draw: %.0f W\n", power/count
            }
        }'
    echo ""
    echo "=== Verdict ==="
    AVG_GPU=$(tail -n +12 "$LOG_DIR/nvidia-smi.log" | awk -F',' 'NR>0 {s+=$3+0; n++} END {if(n>0) print int(s/n); else print 0}')
    PEAK_MEM=$(tail -n +12 "$LOG_DIR/nvidia-smi.log" | awk -F',' 'NR>0 {if($5+0>m) m=$5+0} END {print int(m)}')

    if [ "$AVG_GPU" -ge 85 ]; then
        echo "GPU SM util avg ${AVG_GPU}% — excellent, saturated."
    elif [ "$AVG_GPU" -ge 70 ]; then
        echo "GPU SM util avg ${AVG_GPU}% — good, minor headroom."
    elif [ "$AVG_GPU" -ge 50 ]; then
        echo "GPU SM util avg ${AVG_GPU}% — underutilized. Likely dataloader/CPU bottleneck."
        echo "  Fixes: raise dataloader_num_workers, pre-cache features, disable gradient_checkpointing if VRAM allows."
    else
        echo "GPU SM util avg ${AVG_GPU}% — severe bottleneck. Investigate I/O, data prep, or CPU."
    fi

    if [ "$PEAK_MEM" -lt 20000 ]; then
        echo "Peak VRAM ${PEAK_MEM} MB — plenty of headroom. Consider raising BATCH_SIZE or disabling gradient_checkpointing."
    elif [ "$PEAK_MEM" -lt 35000 ]; then
        echo "Peak VRAM ${PEAK_MEM} MB — healthy."
    elif [ "$PEAK_MEM" -lt 45000 ]; then
        echo "Peak VRAM ${PEAK_MEM} MB — near ceiling. Keep current config."
    else
        echo "Peak VRAM ${PEAK_MEM} MB — OOM risk. Enable gradient_checkpointing or reduce BATCH_SIZE."
    fi
} | tee "$SUMMARY"

echo ""
echo "[smoke] Done. Full logs in $LOG_DIR"
echo "[smoke] Key file: $SUMMARY"
exit $EXIT_CODE
