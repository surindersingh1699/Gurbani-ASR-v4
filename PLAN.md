# CURRENT PLAN — Surt v3: 500h Kirtan Dataset + Retrain

> **READ THIS FIRST.** This file is the single source of truth for the current direction.
> Any agent picking up this repo should align with the v3 direction below, NOT the v1/v2 plans.
>
> **Historical plans (do NOT follow — superseded):**
> - [`surt_training_plan.md`](surt_training_plan.md) — v1 (sehaj path only)
> - [`surt_v2_training_plan.md`](surt_v2_training_plan.md) — v2 (HF-hosted kirtan dataset ~28h, 5k samples)
> - Old HF datasets (`gurbani-kirtan-dataset-v2`, `gurbani-sehaj-kirtan-full-v2`) are frozen artifacts, not the active data pipeline.

---

## Goal

Build a **500-hour kirtan training dataset** from the **allowlisted sources in [`kirtan.txt`](kirtan.txt)** (SikhNet ragi playlists, AKJ archive, SGPC recorded archive), labeled end-to-end with **Gemini 2.5 Flash Lite**, and retrain Surt on it.

> **Source allowlist is hard.** Only the URLs in [`kirtan.txt`](kirtan.txt) are in scope for v3. Do NOT pull from any YouTube channel, SoundCloud account, or other site — even if previous notes or earlier drafts of this plan mentioned them. If a source is not in `kirtan.txt`, it is out of scope.

## Why We Pivoted

- v2 kirtan dataset (~28h, from mixed OCR + auto-caption sources) produced best kirtan WER ≈ 55% — too noisy and too small.
- Whisper-based transcription hallucinates on sung Gurmukhi.
- Gemini 2.5 Flash Lite + pre-split audio batching (proven in `scripts/kirtan_bulk_transcribe.py`) gives clean Gurmukhi transcripts at **$0 on free tier / ~$0.035/hr paid**, 40–100× cheaper than Google Cloud STT.
- Data scale is the bottleneck, not model architecture. Going from 28h → 500h of clean labels is the highest-leverage move.
- YouTube channels are excluded in v3 because they introduced copyright, geo-block, and dedup headaches. The SikhNet / AKJ / SGPC official archives in `kirtan.txt` give clean provenance and stable URLs.

## Top-Level Architecture

```
Audio taxonomy:
  source_type: paath | kirtan

  paath_type:   sehaj | akhand | nitnem          (paath only)
  kirtan_type:  ragi  | akj    | sgpc            (kirtan only)
```

Kirtan split (target 500h total):

- **Ragi kirtan** — studio/clean, harmonium+tabla, trained vocalists (target ~170h, capped by the 12 SikhNet playlists in `kirtan.txt`)
- **AKJ** — Akhand Kirtani Jatha, fast tempo, sangat participation, crowd vocals (target ~170h, capped by what the `akj.org/keertan.php` archive contains)
- **SGPC kirtan** — live PA from Darbar Sahib and SGPC gurdwaras, reverb + ambient (floor ~160h, **absorbs any shortfall** from ragi + AKJ up to 500h total; see "Scale note + overflow policy" below)

## Hard Rules — Transcription Pipeline

These rules are non-negotiable. They come from burned experience — every deviation in the past introduced alignment bugs or cost overruns.

1. **Audio is always cut to fixed 20-second clips before it touches Gemini.**
   - Last clip of a video may be short; pad with silence or drop if < 5 s.
   - Clips are named `clip_{i:05d}.wav` where `i * 20` is the start second of that clip in the source audio. This naming IS the timestamp — no metadata file needed to recover it.

2. **Never ask Gemini for timestamps.** Gemini is bad at timestamping long audio — every approach that did this produced misaligned text (see `memory/kirtan_data_approaches.md`).
   - We own the timestamps locally because we cut the clips locally.
   - The API contract is: send N clips → receive N text strings in the same order. Nothing else.

3. **Batch 5–10 minutes of audio per API call — not more, not less.**
   - At 20 s per clip, that is **15–30 clips per call**.
   - Default: **15 clips = 5 min audio per call** (proven in `scripts/kirtan_bulk_transcribe.py`, 17.6× realtime).
   - Going higher than 30 clips risks truncated responses, malformed JSON, and higher per-call retry cost if anything fails.
   - Going lower than 15 wastes API-call overhead and inflates cost.
   - **Run many 5-min batches in parallel** to saturate throughput. Gemini 2.5 Flash Lite paid-tier RPM is **1,000–2,000 req/min**, so ~50–100 concurrent workers (each sending one 15-clip batch at a time) is well within limits. Free tier is ~15 RPM — cap workers at 4–8 there. Target: transcribe 500 h of audio in under an hour of wall-clock time on paid tier.

4. **Persist Gemini output to disk immediately — before any post-processing.**
   - Write raw responses to `data/gemini_raw/<video_id>/batch_{k:04d}.json` as each batch returns.
   - Write per-clip transcripts to `data/transcripts/<video_id>.jsonl` (append mode, one line per clip).
   - This is checkpointing: if the run crashes or we want to change normalization rules, we do NOT re-call the API. Gemini calls are the expensive, irreversible step.
   - A re-run must be idempotent: if `data/transcripts/<video_id>.jsonl` has N lines and video has M clips, only process clips N..M.

## Pipeline

```text
0. Read allowlist from kirtan.txt         →  data/manifests/sources.csv  (url, kirtan_type)
1. Enumerate tracks per allowlisted URL   →  data/manifests/<kirtan_type>_manifest.csv  (id, title, duration)
     - SikhNet playlist URL      → per-track MP3 URLs via play.sikhnet.com API
     - SikhNet single track URL  → one-row manifest
     - akj.org/keertan.php       → scrape samagam MP3 links
     - sgpc.net/recorded-kirtan* → scrape per-ragi archive MP3 links
2. Filter + dedup                         →  data/manifests/<kirtan_type>_filtered.csv
3. Audio-only download (direct MP3)       →  data/raw_audio/<kirtan_type>/<track_id>.wav  (16kHz mono)
4. Split into FIXED 20s clips             →  data/clips/<track_id>/clip_{i:05d}.wav
5. Batch 15 clips (5 min) per Gemini call →  data/gemini_raw/<track_id>/batch_{k:04d}.json   (persist raw)
6. Parse batch → per-clip JSONL           →  data/transcripts/<track_id>.jsonl              (persist parsed)
7. Normalize (strip ॥, ॥੧॥, whitespace)   →  data/transcripts_clean/<track_id>.jsonl
8. Quality filter + dedup                 →  data/transcripts_final/<track_id>.jsonl
9. Build HF dataset                       →  surindersinghssj/gurbani-kirtan-v3-500h (kirtan_type tag)
10. Retrain Surt on sehaj + v3 kirtan
```

> `track_id` replaces the old `video_id` since sources are no longer YouTube. Use the SikhNet track numeric id (e.g. `sikhnet-11220`), the AKJ samagam filename slug, or the SGPC archive filename slug. The `i * 20 sec` filename → timestamp invariant is unchanged.

Each step reads from the previous step's output directory and writes to its own — so any stage can be re-run independently without re-downloading or re-calling Gemini.

## Data Sources (ALLOWLIST — single source of truth is [`kirtan.txt`](kirtan.txt))

All kirtan audio for v3 comes from the URLs in `kirtan.txt` — nothing else. If you add a new source, edit `kirtan.txt` first and land that as its own commit so intent is explicit.

### Ragi kirtan — SikhNet playlists (play.sikhnet.com)

- Bhai Harjinder Singh (Srinagar) — `/playlist/bhai-harjinder-singh-srinagar-gurbani-jukebox`
- Bhai Ravinder Singh (Hazuri Ragi) — `/playlist/bhai-ravinder-singh-hazuri-ragi-gurbani-jukebox`
- Bhai Dalbir Singh (Hazuri Ragi) — `/playlist/bhai-dalbir-singh-hazuri-ragi-gurbani-jukebox`
- Bhai Kamaljeet Singh (Hazuri Raagi) — `/playlist/bhai-kamaljeet-singh-hazuri-raagi-gurbani-jukebox`
- Bhai Davinder Singh Ji Nirman (Amritsar) — `/playlist/bhai-davinder-singh-ji-nirman-amritsar-gurbani-jukebox`
- Bhai Randhir Singh (Patiala) — `/playlist/bhai-randhir-singh-patiala-gurbani-jukebox`
- Bhai Amarjit Singh (Patiala) — `/playlist/bhai-amarjit-singh-patiala-gurbani-jukebox`
- Bhai Jaskaran Singh (Patiala) — `/playlist/bhai-jaskaran-singh-patiala-gurbani-jukebox`
- Bhai Surinder Singh (Jodhpuri) — `/playlist/bhai-surinder-singh-jodhpuri-gurbani-jukebox`
- Bhai Anantvir Singh — `/playlist/bhai-anantvir-singh-gurbani-jukebox`
- Bibi Jaskiran Kaur — `/playlist/bibi-jaskiran-kaur-gurbani-jukebox`
- Bhai Lakhwinder Singh (Hazuri Ragi) — `/playlist/bhai-lakhwinder-singh-hazuri-ragi-gurbani-jukebox`

### AKJ

- [`play.sikhnet.com/track/jaag-ray-mann-jaganhare`](https://play.sikhnet.com/track/jaag-ray-mann-jaganhare) — single seed track
- [`akj.org/keertan.php`](https://akj.org/keertan.php) — samagam archive (Rainsbai ~8–10h, Kirtan Darbar ~6h)

### SGPC

- [`sgpc.net/recorded-kirtan-ragi-wise/`](https://sgpc.net/recorded-kirtan-ragi-wise/) — official recorded archive, grouped by ragi

### Excluded (previously considered, NOT in scope for v3)

YouTube (`@sikhnet`, `@amrittsaagar`, `@GurbaniMediaCentre`, `@NirbaanKeertan`, `@SGPCSriAmritsar`), SoundCloud (`akjdotorg`), live streams (`sgpclive.com`), third-party sites (`hukamnamasahib.com`, `gurmatsagar.com`). Do not add scripts or manifest rows for these.

### Scale note + overflow policy

The allowlist is narrower than the 500h target. Back-of-envelope: ~12 SikhNet ragi playlists × ~50 tracks × ~6 min avg ≈ ~60h ragi; AKJ samagam archive is a few hundred hours worth of long recordings; SGPC archive is by far the largest single source (the `recorded-kirtan-ragi-wise/` index aggregates decades of Harmandir Sahib recordings grouped by ragi).

**Overflow policy — SGPC is the fill source.** After enumeration (P1), if ragi + AKJ fall short of 500h, pull the remainder from `sgpc.net/recorded-kirtan-ragi-wise/`. Do NOT rebalance by adding non-allowlisted sources. Concretely:

1. Enumerate ragi (all 12 SikhNet playlists) → record actual hours.
2. Enumerate AKJ (`akj.org/keertan.php` + the one SikhNet seed track) → record actual hours.
3. Compute `sgpc_target_hours = max(160, 500 - ragi_hours - akj_hours)`. The 160h floor keeps the SGPC split meaningful even if AKJ over-delivers; the `500 - ragi - akj` formula lets SGPC absorb any shortfall.
4. Pull SGPC tracks until `sgpc_target_hours` is hit **using the diversity policy below** (not "grab the biggest ragi first").
5. Record the final split (ragi / akj / sgpc hours) in the dataset card so downstream training can weight / stratify.

If the allowlist still can't reach 500h even after maxing SGPC, surface this explicitly — either lower the target or extend `kirtan.txt` with a new explicit commit. Do not silently add hidden sources in scripts.

### SGPC ragi diversity policy (required for scraping)

The SGPC archive has **201 ragi directories** and is heavily skewed — a handful of prolific ragis (e.g. Bhai Amandeep Singh, Bhai Gurdev Singh K) each have 1,000+ MP3s, while most ragis have under 100. If we pulled SGPC top-down we would get ~150h from 3 voices, which is a training disaster for a per-speaker-rare model. Variety in timbre, tempo, vocal style, and reverb conditions matters more than raw hours.

**Rule — any SGPC download selection MUST be ragi-balanced:**

1. **Per-ragi cap first.** Pick a target of `max_hours_per_ragi` (default starts at **2 h/ragi**; raise only if the 160h floor can't otherwise be met). This caps how much one voice can dominate.
2. **Then round-robin across ragis** until the SGPC hour target is reached. Do not sort ragis by size — iterate them in a shuffled order and take one track from each before looping.
3. **Random pick within each ragi, seeded.** Within a ragi, sample tracks with a fixed RNG seed (default `seed=42`) so the selection is reproducible but not biased toward the archive's storage order (which roughly correlates with upload time and thus recording era).
4. **If a ragi has fewer tracks than its cap, take all of them** and redistribute the shortfall across the remaining ragis (the round-robin handles this automatically by just skipping exhausted ragis).
5. **Prefer more ragis over more hours per ragi.** If forced to choose between "160h from 80 ragis × 2h" vs "160h from 40 ragis × 4h", pick the 80-ragi split.

Rule of thumb for planning: at 2 h/ragi across 201 ragis the ceiling is ~400h of balanced SGPC material — comfortably above the 160h floor. Use `max_hours_per_ragi` to tune, not "go wide and hope".

**Implementation** — `scripts/v3_enumerate_sgpc.py` supports this directly:

```bash
# balanced 160h SGPC manifest, 2h/ragi cap, seeded, ~80 ragis contributing
python scripts/v3_enumerate_sgpc.py \
  --out /root/v3_data/manifests/sgpc.csv \
  --max-hours-per-ragi 2 \
  --target-total-hours 160 \
  --seed 42
```

When the downloader consumes the manifest it just downloads every row — the diversity has already been baked into row selection. Do NOT re-sort or re-filter the manifest downstream in a way that re-introduces ragi skew (e.g. don't sort by `size_bytes desc` to "prioritize long tracks").

## Why the Hard Rules Matter (~10× savings)

The four hard rules above together produce roughly a **10× reduction in cost and wall-clock time** vs. the naive approach we tried earlier:

| Lever | Naive approach | v3 approach | Effect |
|---|---|---|---|
| Clip length | Variable, Gemini-estimated | **Fixed 20 s, owned locally** | No alignment bugs, no re-transcription of misaligned segments |
| Batch size | 1 clip per call | **15 clips (5 min) per call** | ~15× fewer API calls → ~15× less per-call overhead cost |
| Timestamps | Returned by Gemini (unreliable) | **Derived from filename `clip_{i}.wav` → `i*20` sec** | Zero re-runs due to bad timestamps |
| Output persistence | Re-run regenerates from scratch | **Raw responses + parsed JSONL both persisted** | Re-runs skip expensive Gemini calls, only re-do free local steps |

End result: proven 17.6× realtime throughput, ~$0.035/hr paid (or $0 free tier) — down from the $0.30–$3.84/hr range of prior approaches.

## Build Order

| Phase | Output | Status |
|---|---|---|
| P1. Enumerate | `data/manifests/<kirtan_type>_manifest.csv` with real durations | TODO |
| P2. Filter | Dedup (chromaprint), drop <3 min / >2 hr, dedup by title+duration | TODO |
| P3. Download | `data/raw_audio/<kirtan_type>/<video_id>.wav` (16 kHz mono) | TODO |
| P4. Split | `data/clips/<video_id>/clip_{i:05d}.wav` (fixed 20 s) | TODO |
| P5a. Gemini raw | `data/gemini_raw/<video_id>/batch_{k:04d}.json` (persist first) | TODO |
| P5b. Parse | `data/transcripts/<video_id>.jsonl` (clip_id → text, start_sec = i*20) | TODO |
| P6. Normalize | `data/transcripts_clean/<video_id>.jsonl` (strip `॥`/`॥੧॥`) | TODO |
| P7. Filter + dedup | `data/transcripts_final/<video_id>.jsonl` | TODO |
| P8. Dataset build | HF dataset `surindersinghssj/gurbani-kirtan-v3-500h` | TODO |
| P9. Retrain | Surt v3 on sehaj + v3 kirtan | TODO |

## Known Gotchas

1. **Dedup by audio fingerprint**, not title — same kirtan gets re-uploaded 10+ times.
2. **Katha contamination** — many "kirtan" videos have long spoken intros. Need VAD + music/speech split, or the model learns katha cadence.
3. **SGPC 24/7 stream bias** — raw stream contains ardaas, hukamnama, announcements. Need segmentation.
4. **Copyright/channel suspensions** — stick to official channels; SGPC has already been suspended once.
5. **Gemini spelling drift** — transcripts are ~57% exact-match to canonical SGGS. Post-process against `tuks.json` / STTM where possible.
6. **Text normalization** — always strip `॥`, `॥੧॥` verse markers before tokenization (they are visual, never spoken).

## What is NOT in scope for v3

- LoRA / PEFT — full fine-tune continues to be the approach.
- Changing the Whisper-Small backbone — scale data, not model.
- Rebuilding the sehaj path dataset — reuse `surindersinghssj/gurbani-sehajpath` as-is for regularization.
- Forced alignment / OCR pipelines — Gemini replaces both.

## Key Commands (reference)

```bash
# Enumerate a channel (no download)
yt-dlp --flat-playlist --print "%(id)s,%(title)s,%(duration)s" "URL" > manifest.csv

# Audio-only download, 16kHz mono
yt-dlp -x --audio-format wav --postprocessor-args "-ar 16000 -ac 1" \
       --download-archive done.txt -a urls.txt

# Transcription (existing script, proven at scale)
python scripts/kirtan_bulk_transcribe.py <video_id>
```
