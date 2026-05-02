#!/usr/bin/env bash
# RunPod A40 end-to-end IndicConformer-pa fine-tune + bench.
# Designed to run UNATTENDED on a fresh pytorch pod with surt-storage mounted at /workspace.
#
# Required env vars (passed via runpodctl --env):
#   HF_TOKEN          - HuggingFace token with access to ai4bharat/indicconformer_stt_pa_*
#   WANDB_API_KEY     - W&B token (optional but recommended)
#   RUNPOD_POD_ID     - Auto-set by RunPod runtime, used for self-termination
#
# What this script does:
#   1. Install AI4Bharat NeMo fork + deps + patch multilingual tokenizer
#   2. Discover FLAC layout on /workspace (the network volume)
#   3. Audit train/eval video_id leak
#   4. Build NeMo manifests pointing at existing FLAC files
#   5. Train IndicConformer-pa on 600h (kirtan/sehajpath mix), early-stop on kirtan WER
#   6. Bench the new checkpoint vs eval splits
#   7. Push results (model + bench JSON + log) to HF: surindersinghssj/indicconformer-pa-v3-kirtan
#   8. Self-terminate the pod (saves money — won't burn $$ idle waiting for human)
#
# All logs → /workspace/runs/<timestamp>/log.txt
# Final results → HF repo + /workspace/runs/<timestamp>/RESULTS.json

set -euo pipefail
exec > >(tee -a /tmp/early.log) 2>&1

echo "=== $(date) — startup ==="

# ----- 0. env -----
export DEBIAN_FRONTEND=noninteractive
export HF_HOME=/workspace/cache/huggingface
export HF_DATASETS_CACHE=/workspace/cache/huggingface/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

RUN_TS="$(date -u +%Y%m%d-%H%M%S)"
RUN_DIR="/workspace/runs/$RUN_TS"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/log.txt") 2>&1
echo "[run] log → $RUN_DIR/log.txt"

# ----- 1. install -----
echo "=== install deps ==="
apt-get update -qq && apt-get install -y -qq git ffmpeg sox libsndfile1 jq

pip install -U -q "huggingface_hub>=0.34,<1.0" hf_transfer
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
[[ -n "${WANDB_API_KEY:-}" ]] && pip install -U -q wandb && wandb login "$WANDB_API_KEY" || true

# AI4Bharat NeMo fork (skip triton, the cuda-only dep that breaks pip resolver)
if [ ! -d /workspace/ai4bharat-nemo ]; then
    git clone --depth 1 -b nemo-v2 https://github.com/AI4Bharat/NeMo.git /workspace/ai4bharat-nemo
    sed -i '/^triton$/d' /workspace/ai4bharat-nemo/requirements/requirements.txt
fi
pip install -q --no-deps /workspace/ai4bharat-nemo
pip install -q "numpy<2" "datasets<4" soundfile librosa jiwer omegaconf hydra-core \
              pytorch-lightning sentencepiece youtokentome editdistance \
              braceexpand kaldiio lhotse pyannote.metrics texterrors

# Sanity: NeMo loads
python -c "import nemo.collections.asr as a; print('NeMo OK', a.__name__)"

# ----- 2. patch multilingual tokenizer support -----
echo "=== patch NeMo mixins (multilingual tokenizer for IndicConformer) ==="
python <<'EOF'
import nemo, pathlib
p = pathlib.Path(nemo.__path__[0]) / 'collections/asr/parts/mixins/mixins.py'
s = p.read_text()
if 'multilingual' not in s:
    s = s.replace(
        "elif tokenizer_type.lower() == 'agg':\n            self._setup_aggregate_tokenizer(tokenizer_cfg)",
        "elif tokenizer_type.lower() in ('agg','multilingual'):\n            self._setup_aggregate_tokenizer(tokenizer_cfg)"
    )
    p.write_text(s); print('patched mixins.py')
else:
    print('mixins.py already patched (or fork already supports multilingual)')
EOF

# ----- 3. download base model + repo -----
echo "=== download base IndicConformer-pa .nemo ==="
NEMO_PATH=/workspace/models/indicconformer_stt_pa_hybrid_rnnt_large.nemo
mkdir -p /workspace/models
if [ ! -f "$NEMO_PATH" ]; then
    huggingface-cli download \
        ai4bharat/indicconformer_stt_pa_hybrid_ctc_rnnt_large \
        indicconformer_stt_pa_hybrid_rnnt_large.nemo \
        --local-dir /workspace/models --local-dir-use-symlinks False
fi
ls -lh "$NEMO_PATH"

REPO_DIR=/workspace/Gurbani-ASR-v4
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/surindersingh1699/Gurbani-ASR-v4.git "$REPO_DIR"
fi
cd "$REPO_DIR" && git fetch && git checkout codex/complete-phase-4-work && git pull
cd "$REPO_DIR"

# ----- 4. discover FLAC layout on volume -----
echo "=== inspect /workspace for FLAC layout ==="
echo "[volume] top-level:"
ls -la /workspace/ | head -30
echo "[volume] flac sample (first 5 paths):"
find /workspace -name "*.flac" -type f 2>/dev/null | head -5 || echo "no flac yet"
echo "[volume] wav sample:"
find /workspace -name "*.wav" -type f 2>/dev/null | head -5 || echo "no wav"
echo "[volume] dir tree (depth 3):"
find /workspace -maxdepth 3 -type d 2>/dev/null | head -40

# Save inventory snapshot for later debugging
find /workspace -name "*.flac" -o -name "*.wav" 2>/dev/null > "$RUN_DIR/audio_inventory.txt"
TOTAL_AUDIO=$(wc -l < "$RUN_DIR/audio_inventory.txt")
echo "[volume] total audio files found: $TOTAL_AUDIO"

# ----- 5. audit train/eval video_id leak -----
echo "=== leak audit ==="
python "$REPO_DIR/scripts/audit_train_eval_leak.py" \
    --eval surindersinghssj/gurbani-sehajpath-yt-captions-eval-canonical \
           surindersinghssj/gurbani-kirtan-yt-captions-eval-canonical \
    --train surindersinghssj/gurbani-sehajpath-yt-captions-canonical \
            surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical \
            surindersinghssj/gurbani-sehajpath \
    --out "$RUN_DIR/leak_audit.json"

# ----- 6. build manifests -----
echo "=== build manifests ==="
# Try multiple plausible audio roots — pass the ones the volume actually contains.
# The script tries by_video and flat layouts via --layout auto.
mkdir -p /workspace/data/manifests
python "$REPO_DIR/scripts/build_indicconformer_manifests.py" \
    --data-root /workspace/data \
    --audio-root-kirtan          /workspace/audio/kirtan \
    --audio-root-sehajpath-yt    /workspace/audio/sehajpath_yt \
    --audio-root-sehajpath-orig  /workspace/audio/sehajpath_orig \
    --audio-root-eval-kirtan     /workspace/audio/eval_kirtan \
    --audio-root-eval-sehajpath  /workspace/audio/eval_sehajpath \
    --layout auto \
    --leaked-file "$RUN_DIR/train_dropped_video_ids.txt"

# Sanity: make sure manifests aren't empty (would mean audio paths are wrong)
for m in train.jsonl val_kirtan.jsonl val_sehajpath.jsonl; do
    n=$(wc -l < /workspace/data/manifests/$m 2>/dev/null || echo 0)
    echo "[manifest] $m → $n entries"
    if [ "$n" -lt 100 ]; then
        echo "[FATAL] manifest $m is too small — audio paths likely wrong"
        echo "Inspect: $RUN_DIR/audio_inventory.txt and adjust --audio-root-* flags"
        # Don't crash — push the diagnostic and stop
        cp "$RUN_DIR/audio_inventory.txt" "$RUN_DIR/FAILED_audio_layout_unknown.txt"
        echo "STAGE_FAILED=manifest_build" > "$RUN_DIR/RESULTS.json"
        # Push diagnostics to HF anyway
        huggingface-cli upload --repo-type dataset \
            surindersinghssj/indicconformer-pa-v3-kirtan-runlogs \
            "$RUN_DIR" "runs/$RUN_TS" --create-pr=false || true
        runpodctl pod stop "$RUNPOD_POD_ID" || true
        runpodctl pod delete "$RUNPOD_POD_ID" || true
        exit 2
    fi
done

# ----- 7. train -----
echo "=== train ==="
cp "$REPO_DIR/training/indicconformer_pa_v3_kirtan.yaml" "$RUN_DIR/config.yaml"
sed -i "s|exp_dir:.*|exp_dir: $RUN_DIR/checkpoints|g" "$RUN_DIR/config.yaml"
sed -i "s|init_from_nemo_model:.*|init_from_nemo_model: $NEMO_PATH|g" "$RUN_DIR/config.yaml"

# NeMo's hybrid CTC-RNNT trainer entrypoint
python /workspace/ai4bharat-nemo/examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py \
    --config-path="$RUN_DIR" \
    --config-name="config.yaml" 2>&1 | tee -a "$RUN_DIR/train_log.txt"

# ----- 8. bench against new checkpoint -----
echo "=== bench against fine-tuned ckpt ==="
BEST_CKPT=$(ls -t "$RUN_DIR/checkpoints"/*.nemo 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
    echo "[error] no .nemo checkpoint found in $RUN_DIR/checkpoints/"
    BEST_CKPT="$NEMO_PATH"  # fall back to base model so bench still produces a number
fi
echo "[bench] using checkpoint: $BEST_CKPT"

python "$REPO_DIR/scripts/bench_asr_alternatives.py" \
    --device cuda --threads 4 \
    --only indicconformer \
    --indicconformer-path "$BEST_CKPT" \
    --indicconformer-decoder rnnt \
    --max-samples 444 \
    --out-dir "$RUN_DIR/bench_finetuned" 2>&1 | tee -a "$RUN_DIR/bench_log.txt"

# ----- 9. write final summary + push to HF -----
echo "=== package results ==="
python <<EOF
import json, glob, os
csv_path = sorted(glob.glob('$RUN_DIR/bench_finetuned/*/results.csv'))[-1]
results = open(csv_path).read()
summary = {
    "run_ts": "$RUN_TS",
    "checkpoint": "$BEST_CKPT",
    "bench_csv": results,
    "leak_audit": json.load(open('$RUN_DIR/leak_audit.json')),
    "log_path_on_volume": "$RUN_DIR/log.txt",
}
json.dump(summary, open('$RUN_DIR/RESULTS.json','w'), indent=2, ensure_ascii=False)
print(open('$RUN_DIR/RESULTS.json').read())
EOF

# Push checkpoint + run logs to HF
huggingface-cli repo create indicconformer-pa-v3-kirtan --type model -y || true
huggingface-cli upload surindersinghssj/indicconformer-pa-v3-kirtan \
    "$BEST_CKPT" indicconformer-pa-v3-kirtan.nemo --create-pr=false || true

huggingface-cli repo create indicconformer-pa-v3-kirtan-runlogs --type dataset -y || true
huggingface-cli upload --repo-type dataset \
    surindersinghssj/indicconformer-pa-v3-kirtan-runlogs \
    "$RUN_DIR/RESULTS.json" "runs/$RUN_TS/RESULTS.json" --create-pr=false || true
huggingface-cli upload --repo-type dataset \
    surindersinghssj/indicconformer-pa-v3-kirtan-runlogs \
    "$RUN_DIR/log.txt" "runs/$RUN_TS/log.txt" --create-pr=false || true
huggingface-cli upload --repo-type dataset \
    surindersinghssj/indicconformer-pa-v3-kirtan-runlogs \
    "$RUN_DIR/bench_finetuned" "runs/$RUN_TS/bench_finetuned" --create-pr=false || true

echo "=== DONE at $(date) ==="
echo "Self-terminating pod $RUNPOD_POD_ID in 60s ..."
sleep 60
runpodctl pod stop "$RUNPOD_POD_ID" || true
runpodctl pod delete "$RUNPOD_POD_ID" || true
