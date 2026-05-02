#!/usr/bin/env bash
# Run Stage 2 Batch-API pipeline end-to-end for kirtan 300h + sehajpath.
# Fires both pipelines in parallel, each invokes: submit → poll → submit
# reviewer → poll reviewer → resolve. Result parquets land in $OUTDIR.
#
# Usage:
#   cd /root/Gurbani_ASR_v4
#   git pull
#   set -a; source .env; set +a   # loads OPEN_API_KEY / GEMINI_API_KEY / HF_TOKEN
#   bash scripts/run_stage2_full.sh
#
# Expects the *-clean parquets to already be in HF cache (no download).
# Writes:
#   $OUTDIR/kirtan_{stage1,stage2,run.log,state.json}
#   $OUTDIR/sehaj_{stage1,stage2,run.log,state.json}

set -euo pipefail

OUTDIR="${OUTDIR:-/root/stage2_full}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$OUTDIR"

KIRTAN_REPO="surindersinghssj/gurbani-kirtan-yt-captions-300h-clean"
SEHAJ_REPO="surindersinghssj/gurbani-sehajpath-yt-captions-clean"
KIRTAN_DEST="surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical"
SEHAJ_DEST="surindersinghssj/gurbani-sehajpath-yt-captions-canonical"

KIRTAN_INPUT="$OUTDIR/kirtan_clean_full.parquet"
SEHAJ_INPUT="$OUTDIR/sehaj_clean_full.parquet"

echo "[run_stage2_full] OUTDIR=$OUTDIR"
echo "[run_stage2_full] repo root: $REPO_ROOT"

# --- 0. Materialize source metadata parquets (streaming, no audio) ---
if [ ! -f "$KIRTAN_INPUT" ]; then
  echo "[run_stage2_full] streaming $KIRTAN_REPO → $KIRTAN_INPUT (metadata only)..."
  python3 - <<PY
import time
import pandas as pd
from datasets import load_dataset
t0 = time.monotonic()
ds = load_dataset("$KIRTAN_REPO", split="train", streaming=True)
rows = []
for i, rec in enumerate(ds):
    rec.pop("audio", None)
    rows.append(rec)
    if (i + 1) % 20000 == 0:
        print(f"  {i+1} rows, elapsed {time.monotonic()-t0:.0f}s", flush=True)
pd.DataFrame(rows).to_parquet("$KIRTAN_INPUT")
print(f"wrote {len(rows)} rows → $KIRTAN_INPUT in {time.monotonic()-t0:.0f}s")
PY
fi

if [ ! -f "$SEHAJ_INPUT" ]; then
  echo "[run_stage2_full] streaming $SEHAJ_REPO → $SEHAJ_INPUT (metadata only)..."
  python3 - <<PY
import time
import pandas as pd
from datasets import load_dataset
t0 = time.monotonic()
ds = load_dataset("$SEHAJ_REPO", split="train", streaming=True)
rows = []
for i, rec in enumerate(ds):
    rec.pop("audio", None)
    rows.append(rec)
    if (i + 1) % 20000 == 0:
        print(f"  {i+1} rows, elapsed {time.monotonic()-t0:.0f}s", flush=True)
pd.DataFrame(rows).to_parquet("$SEHAJ_INPUT")
print(f"wrote {len(rows)} rows → $SEHAJ_INPUT in {time.monotonic()-t0:.0f}s")
PY
fi

# --- 1. Run kirtan + sehaj pipelines in parallel ---
cd "$REPO_ROOT"

echo "[run_stage2_full] launching KIRTAN run..."
nohup python3 scripts/stage2_batch.py run \
    --input-parquet "$KIRTAN_INPUT" \
    --dataset kirtan \
    --source-col text_cleaned \
    --stage1-parquet "$OUTDIR/kirtan_stage1.parquet" \
    --state-file    "$OUTDIR/kirtan.state.json" \
    --output-parquet "$OUTDIR/kirtan_stage2.parquet" \
    --corrector-model gpt-5-nano \
    --corrector-batch-size 15 \
    --reviewer-model gemini-3-flash-lite \
    --reviewer-batch-size 5 \
    --poll-interval 600 \
    --max-workers 10 \
  > "$OUTDIR/kirtan_run.log" 2>&1 &
KIRTAN_PID=$!
echo "[run_stage2_full] KIRTAN pid=$KIRTAN_PID → $OUTDIR/kirtan_run.log"

echo "[run_stage2_full] launching SEHAJ run..."
nohup python3 scripts/stage2_batch.py run \
    --input-parquet "$SEHAJ_INPUT" \
    --dataset sehaj \
    --source-col text_cleaned \
    --stage1-parquet "$OUTDIR/sehaj_stage1.parquet" \
    --state-file    "$OUTDIR/sehaj.state.json" \
    --output-parquet "$OUTDIR/sehaj_stage2.parquet" \
    --corrector-model gpt-5-nano \
    --corrector-batch-size 15 \
    --reviewer-model gemini-3-flash-lite \
    --reviewer-batch-size 5 \
    --poll-interval 600 \
    --max-workers 10 \
  > "$OUTDIR/sehaj_run.log" 2>&1 &
SEHAJ_PID=$!
echo "[run_stage2_full] SEHAJ pid=$SEHAJ_PID → $OUTDIR/sehaj_run.log"

echo ""
echo "[run_stage2_full] both pipelines running. Check progress with:"
echo "  tail -f $OUTDIR/kirtan_run.log"
echo "  tail -f $OUTDIR/sehaj_run.log"
echo ""
echo "[run_stage2_full] PIDs: kirtan=$KIRTAN_PID  sehaj=$SEHAJ_PID"
echo "$KIRTAN_PID" > "$OUTDIR/kirtan.pid"
echo "$SEHAJ_PID" > "$OUTDIR/sehaj.pid"

# --- 2. Wait for both to finish ---
echo "[run_stage2_full] waiting for both to complete..."
wait "$KIRTAN_PID"
KIRTAN_RC=$?
wait "$SEHAJ_PID"
SEHAJ_RC=$?

echo "[run_stage2_full] kirtan rc=$KIRTAN_RC, sehaj rc=$SEHAJ_RC"

# --- 3. Push results to HF ---
if [ $KIRTAN_RC -eq 0 ] && [ -f "$OUTDIR/kirtan_stage2.parquet" ]; then
  echo "[run_stage2_full] pushing kirtan → $KIRTAN_DEST"
  python3 scripts/stage2_batch.py push \
      --output-parquet "$OUTDIR/kirtan_stage2.parquet" \
      --repo "$KIRTAN_DEST" \
      --mode metadata-only \
    2>&1 | tee -a "$OUTDIR/kirtan_push.log"
else
  echo "[run_stage2_full] SKIPPING kirtan push (rc=$KIRTAN_RC)"
fi

if [ $SEHAJ_RC -eq 0 ] && [ -f "$OUTDIR/sehaj_stage2.parquet" ]; then
  echo "[run_stage2_full] pushing sehaj → $SEHAJ_DEST"
  python3 scripts/stage2_batch.py push \
      --output-parquet "$OUTDIR/sehaj_stage2.parquet" \
      --repo "$SEHAJ_DEST" \
      --mode metadata-only \
    2>&1 | tee -a "$OUTDIR/sehaj_push.log"
else
  echo "[run_stage2_full] SKIPPING sehaj push (rc=$SEHAJ_RC)"
fi

echo "[run_stage2_full] ✅ done"
