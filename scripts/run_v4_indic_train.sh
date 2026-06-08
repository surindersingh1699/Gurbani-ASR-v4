#!/usr/bin/env bash
# v4-clean IndicConformer-pa fine-tune — UNATTENDED RunPod runner.
#
# Set by /goal: train on  kirtan HIGH-only (3 cleanv2 sources, x2)  +  bani-v4 FULL.
# Optimize kirtan CER (WER secondary). HARD augmentation (SpecAug + speed/pitch +
# noise + RIR). Show evals for KIRTAN ONLY. Init from base IndicConformer-pa .nemo.
#
# Self-contained: decodes cleanv2 parquet -> FLAC on the pod (no pre-staged volume).
# DISK: needs ~450-500 GB free on /workspace (250 GB parquet cache + ~90 GB FLAC +
#       base model + RIRS_NOISES). Use a big network volume or container disk.
#
# Required env (pass via runpodctl --env):
#   HF_TOKEN, [WANDB_API_KEY], RUNPOD_POD_ID (auto)
set -euo pipefail
exec > >(tee -a /tmp/early.log) 2>&1
echo "=== $(date) — v4 indic train startup ==="

export DEBIAN_FRONTEND=noninteractive
export HF_HOME=/workspace/cache/huggingface
export HF_DATASETS_CACHE=/workspace/cache/huggingface/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

RUN_TS="$(date -u +%Y%m%d-%H%M%S)"
RUN_DIR="/workspace/runs/$RUN_TS"; mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/log.txt") 2>&1
echo "[run] log -> $RUN_DIR/log.txt"
BRANCH="${REPO_BRANCH:-feat/anchor-first-letter-v1}"

# ----- 1. install -----
apt-get update -qq && apt-get install -y -qq git ffmpeg sox libsndfile1 jq unzip wget
pip install -U -q "huggingface_hub>=0.34,<1.0" hf_transfer
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
[[ -n "${WANDB_API_KEY:-}" ]] && pip install -U -q wandb && wandb login "$WANDB_API_KEY" || true

if [ ! -d /workspace/ai4bharat-nemo ]; then
    git clone --depth 1 -b nemo-v2 https://github.com/AI4Bharat/NeMo.git /workspace/ai4bharat-nemo
    sed -i '/^triton$/d' /workspace/ai4bharat-nemo/requirements/requirements.txt
fi
pip install -q --no-deps /workspace/ai4bharat-nemo
pip install -q "numpy<2" "datasets<4" soundfile librosa jiwer omegaconf hydra-core \
              pytorch-lightning sentencepiece youtokentome editdistance \
              braceexpand kaldiio lhotse pyannote.metrics texterrors
python -c "import nemo.collections.asr as a; print('NeMo OK')"

# ----- 2. patch multilingual tokenizer -----
python - <<'EOF'
import nemo, pathlib
p = pathlib.Path(nemo.__path__[0]) / 'collections/asr/parts/mixins/mixins.py'
s = p.read_text()
if 'multilingual' not in s:
    s = s.replace(
        "elif tokenizer_type.lower() == 'agg':\n            self._setup_aggregate_tokenizer(tokenizer_cfg)",
        "elif tokenizer_type.lower() in ('agg','multilingual'):\n            self._setup_aggregate_tokenizer(tokenizer_cfg)")
    p.write_text(s); print('patched mixins.py')
else:
    print('mixins.py already supports multilingual')
EOF

# ----- 3. base model + repo -----
NEMO_PATH=/workspace/models/indicconformer_stt_pa_hybrid_rnnt_large.nemo
mkdir -p /workspace/models
[ -f "$NEMO_PATH" ] || huggingface-cli download \
    ai4bharat/indicconformer_stt_pa_hybrid_ctc_rnnt_large \
    indicconformer_stt_pa_hybrid_rnnt_large.nemo \
    --local-dir /workspace/models --local-dir-use-symlinks False
ls -lh "$NEMO_PATH"

REPO_DIR=/workspace/Gurbani-ASR-v4
[ -d "$REPO_DIR" ] || git clone https://github.com/surindersingh1699/Gurbani-ASR-v4.git "$REPO_DIR"
cd "$REPO_DIR" && git fetch --all && git checkout "$BRANCH" && git pull --ff-only

# ----- 4. leak drop-list (precomputed + committed in repo) -----
DROP="$REPO_DIR/bench_results/train_dropped_video_ids.txt"
if [ -f "$DROP" ]; then
    cp "$DROP" "$RUN_DIR/train_dropped_video_ids.txt"
    echo "[leak] using committed drop-list: $(wc -l < "$DROP") video_ids"
else
    echo "[leak] no committed drop-list; running audit on cleanv2 vs eval-canonical"
    python "$REPO_DIR/scripts/audit_train_eval_leak.py" \
        --eval surindersinghssj/gurbani-kirtan-yt-captions-eval-canonical \
               surindersinghssj/gurbani-sehajpath-yt-captions-eval-canonical \
        --train surindersinghssj/gurbani-kirtan-yt-captions-300h-cleanv2 \
                surindersinghssj/gurbani-kirtan-v4-sgpc-cleanv2 \
                surindersinghssj/gurbani-kirtan-v4-1000h-cleanv2 \
                surindersinghssj/gurbani-bani-v4-cleanv2 \
        --out "$RUN_DIR/leak_audit.json"
    cp "$RUN_DIR/train_dropped_video_ids.txt" "$RUN_DIR/train_dropped_video_ids.txt" 2>/dev/null || true
fi

# ----- 5. build manifests (decode cleanv2 parquet -> FLAC; kirtan HIGH-only, bani FULL) -----
mkdir -p /workspace/data/manifests
python "$REPO_DIR/scripts/build_indicconformer_manifests_parallel.py" \
    --data-root /workspace/data \
    --audio-root /workspace/data/audio \
    --leaked-file "$RUN_DIR/train_dropped_video_ids.txt" \
    --workers "$(($(nproc)-1))"
for m in train.jsonl val_kirtan.jsonl; do
    n=$(wc -l < /workspace/data/manifests/$m 2>/dev/null || echo 0)
    echo "[manifest] $m -> $n entries"
    [ "$m" = "train.jsonl" ] && [ "$n" -lt 1000 ] && { echo "[FATAL] train manifest too small"; exit 2; }
done

# ----- 6. HARD aug data: RIRS_NOISES gives both RIR + pointsource noise -----
AUG=/workspace/data/aug; mkdir -p "$AUG"
if [ ! -f "$AUG/rir.json" ]; then
    echo "=== fetch RIRS_NOISES (OpenSLR SLR28) for RIR + noise aug ==="
    if wget -q -O "$AUG/rirs.zip" https://www.openslr.org/resources/28/rirs_noises.zip; then
        unzip -q -o "$AUG/rirs.zip" -d "$AUG" && rm -f "$AUG/rirs.zip"
        python - "$AUG" <<'EOF'
import sys, glob, json, soundfile as sf, os
aug = sys.argv[1]; base = os.path.join(aug, "RIRS_NOISES")
def manifest(globs, out):
    rows = []
    for g in globs:
        for f in glob.glob(g, recursive=True):
            try:
                info = sf.info(f); rows.append({"audio_filepath": f,
                    "duration": round(info.frames/info.samplerate, 3), "text": ""})
            except Exception: pass
    open(out, "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"  {out}: {len(rows)} files"); return len(rows)
nr = manifest([f"{base}/simulated_rirs/**/*.wav", f"{base}/real_rirs_isotropic_noises/*.wav"], f"{aug}/rir.json")
nn = manifest([f"{base}/pointsource_noises/*.wav"], f"{aug}/noise.json")
open(f"{aug}/_ok", "w").write(f"rir={nr} noise={nn}")
EOF
    else
        echo "[aug] RIRS_NOISES download failed -> RIR/noise disabled (speed/gain/white_noise stay on)"
    fi
fi

# ----- 7. assemble config: exp_dir, base ckpt, inject RIR+noise augmentor if present -----
cp "$REPO_DIR/training/indicconformer_pa_v3_kirtan.yaml" "$RUN_DIR/config.yaml"
python - "$RUN_DIR/config.yaml" "$NEMO_PATH" "$RUN_DIR/checkpoints" "$AUG" <<'EOF'
import sys, os
from omegaconf import OmegaConf
cfg_path, nemo, expdir, aug = sys.argv[1:5]
cfg = OmegaConf.load(cfg_path)
cfg.init_from_nemo_model = nemo
cfg.exp_manager.exp_dir = expdir
a = cfg.model.train_ds.augmentor
rir, noise = os.path.join(aug, "rir.json"), os.path.join(aug, "noise.json")
if os.path.exists(rir) and os.path.getsize(rir) > 5:
    a.impulse = {"prob": 0.3, "manifest_path": rir}
    print("[aug] +impulse(RIR)")
if os.path.exists(noise) and os.path.getsize(noise) > 5:
    a.noise = {"prob": 0.4, "manifest_path": noise, "min_snr_db": 5, "max_snr_db": 25}
    print("[aug] +noise(MUSAN/pointsource)")
OmegaConf.save(cfg, cfg_path)
print("[cfg] augmentor:", OmegaConf.to_yaml(cfg.model.train_ds.augmentor))
EOF

# ----- 8. train -----
echo "=== train ==="
python /workspace/ai4bharat-nemo/examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py \
    --config-path="$RUN_DIR" --config-name="config.yaml" 2>&1 | tee -a "$RUN_DIR/train_log.txt"

# ----- 9. KIRTAN-ONLY eval (best ckpt, both decoders) -----
BEST=$(ls -t "$RUN_DIR/checkpoints"/*.nemo 2>/dev/null | head -1)
[ -z "$BEST" ] && BEST="$NEMO_PATH"
echo "[eval] best ckpt: $BEST"
for manifest in val_kirtan val_kirtan_caption; do
  MF="/workspace/data/manifests/$manifest.jsonl"
  [ -s "$MF" ] || continue
  for dec in rnnt ctc; do
    python "$REPO_DIR/scripts/eval_kirtan_nemo.py" \
        --nemo "$BEST" --manifest "$MF" \
        --decoder $dec --out "$RUN_DIR/kirtan_${manifest}_${dec}.json" || true
  done
done

# ----- 10. package + push + self-terminate -----
python - "$RUN_DIR" "$BEST" "$RUN_TS" <<'EOF'
import json, glob, os, sys
rd, best, ts = sys.argv[1:4]
summary = {"run_ts": ts, "checkpoint": best, "kirtan_evals": {}}
for f in glob.glob(f"{rd}/kirtan_*.json"):
    summary["kirtan_evals"][os.path.basename(f)] = json.load(open(f))
for f in ("leak_audit.json", "train_dropped_video_ids.txt"):
    if os.path.exists(f"{rd}/{f}"): summary[f] = f"{rd}/{f}"
json.dump(summary, open(f"{rd}/RESULTS.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(summary, indent=2, ensure_ascii=False))
EOF

huggingface-cli repo create indicconformer-pa-v3-kirtan --type model -y || true
[ -f "$BEST" ] && huggingface-cli upload surindersinghssj/indicconformer-pa-v3-kirtan \
    "$BEST" indicconformer-pa-v3-kirtan.nemo --create-pr=false || true
huggingface-cli repo create indicconformer-pa-v3-kirtan-runlogs --type dataset -y || true
huggingface-cli upload --repo-type dataset surindersinghssj/indicconformer-pa-v3-kirtan-runlogs \
    "$RUN_DIR/RESULTS.json" "runs/$RUN_TS/RESULTS.json" --create-pr=false || true
huggingface-cli upload --repo-type dataset surindersinghssj/indicconformer-pa-v3-kirtan-runlogs \
    "$RUN_DIR/log.txt" "runs/$RUN_TS/log.txt" --create-pr=false || true

echo "=== DONE $(date) — self-terminating pod $RUNPOD_POD_ID in 60s ==="
# ensure runpodctl exists in-pod for self-terminate (cost guard; reads RUNPOD_API_KEY env)
if ! command -v runpodctl >/dev/null 2>&1; then
    wget -qO /usr/local/bin/runpodctl https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 \
        && chmod +x /usr/local/bin/runpodctl || true
fi
sleep 60
runpodctl remove pod "$RUNPOD_POD_ID" 2>/dev/null || runpodctl stop pod "$RUNPOD_POD_ID" 2>/dev/null || true
