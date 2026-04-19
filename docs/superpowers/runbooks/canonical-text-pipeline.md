# Canonical Text Pipeline — Operational Runbook

## Prereqs

- `database.sqlite` present at repo root (STTM ShabadOS DB).
- `GEMINI_API_KEY` and `HF_TOKEN` in `.env`.
- Gating: AKJ ingest merged + additive HF push landed (see spec §0).

## 1. Dry-run on 1000 rows

```bash
# Pull existing parquet from HF
huggingface-cli download \
  surindersinghssj/gurbani-kirtan-yt-captions-300h \
  --repo-type dataset --local-dir /tmp/kirtan-raw

python3 scripts/canonical_dry_run.py \
  --input-parquet /tmp/kirtan-raw/train-*.parquet \
  --dataset kirtan --n 1000
```

Expected: decision histogram with ~60–75% matched+replaced. If that's wildly off, investigate before the full run.

## 2. Stage 1 — full run

```bash
python3 scripts/add_canonical_column.py \
  --input-parquet /tmp/kirtan-raw/train.parquet \
  --output-parquet /tmp/kirtan-stage1.parquet \
  --audit-parquet  /tmp/kirtan-audit.parquet \
  --dataset kirtan \
  --db database.sqlite
```

Runtime: ~10 minutes on a single CPU for 60K rows. Output: two parquet files.

## 3. Stage 2 — Gemini fallback

```bash
python3 scripts/llm_canonical_pass.py \
  --stage1-parquet /tmp/kirtan-stage1.parquet \
  --audit-parquet  /tmp/kirtan-audit.parquet \
  --llm-sidecar    /tmp/kirtan-llm.parquet \
  --dataset kirtan \
  --db database.sqlite \
  --model gemini-2.5-flash-lite \
  --batch-size 30
```

Expected cost: ~$2.30 for 300h dataset. Runtime: ~10 minutes.

## 4. Merge + local verify

```bash
python3 scripts/merge_canonical_into_hf.py \
  --stage1-parquet /tmp/kirtan-stage1.parquet \
  --llm-sidecar    /tmp/kirtan-llm.parquet \
  --output-parquet /tmp/kirtan-final.parquet
```

Inspect the merged parquet in a notebook — confirm column set, spot-check
5 rows per decision, confirm no original columns were dropped.

## 5. Push to HF (ONLY after local verify)

```bash
python3 scripts/merge_canonical_into_hf.py \
  --stage1-parquet /tmp/kirtan-stage1.parquet \
  --llm-sidecar    /tmp/kirtan-llm.parquet \
  --output-parquet /tmp/kirtan-final.parquet \
  --hf-repo surindersinghssj/gurbani-kirtan-yt-captions-300h \
  --confirm-push
```

## 6. Sehaj path equivalent

Same commands, swap `--dataset kirtan` → `--dataset sehaj` and repo name
`surindersinghssj/gurbani-sehajpath`.

## Rollback

If a run pushed a bad revision:
1. On HF dataset page, revert to the previous commit.
2. Locally: no dataset state held, sidecar parquets can be regenerated.
3. Re-run after fixing config or data.
