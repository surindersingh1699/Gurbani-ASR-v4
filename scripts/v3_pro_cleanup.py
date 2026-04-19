#!/usr/bin/env python3
"""Flag suspect Flash transcriptions and re-run only those through Gemini 2.5 Pro.

Flagging heuristics (local, free — no API needed):
  1. First 3 or last 3 clips of each track (intro/outro hallucination zone)
  2. Any token repeated >= REPEAT_THRESHOLD times in the clip (truncation signature)
  3. Very short text (<4 Gurmukhi chars) when neighbor clip has >20 chars
     (usually a drop-out between surrounding content)

Only flagged clips are re-transcribed. Everything else is untouched.

Input:  data/transcripts/<item_id>.jsonl   (Flash output)
Output: data/transcripts_cleaned/<item_id>.jsonl
        Each row gets:
            pro_reviewed: bool      — was it sent to Pro?
            pro_text: str           — Pro transcription ("" if not reviewed)
            flash_text: str         — preserved original Flash output
            text: str               — Pro if reviewed, else Flash (final pick)
            flag_reason: str        — why it was flagged ("" if not flagged)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPEAT_THRESHOLD = 10         # same token >=10x → likely truncation
SHORT_TEXT_CHARS = 4
NEIGHBOR_LONG_CHARS = 20
INTRO_OUTRO_WINDOW = 3
DEFAULT_FALLBACK_MODEL = "gemini-2.5-flash"
GURMUKHI_RE = re.compile(r"[\u0A00-\u0A7F]")

# Simran tokens — legitimate repetition, not truncation
SIMRAN_TOKENS = {
    "ਵਾਹਿਗੁਰੂ", "ਵਾਹਗੁਰੂ",
    "ਸਤਿਨਾਮੁ", "ਸਤਿ", "ਨਾਮੁ",
    "ਹਰਿ", "ਰਾਮ", "ਗੁਰੂ", "ਧਨੁ", "ਧੰਨ",
}


def is_simran_clip(text: str) -> bool:
    """Clip is dominated by a simran/naam token (legitimate repetition)."""
    if not text:
        return False
    tokens = [t for t in re.split(r"\s+|॥|।", text) if t]
    if not tokens:
        return False
    top_tok, top_count = Counter(tokens).most_common(1)[0]
    return top_count / len(tokens) >= 0.6 and top_tok in SIMRAN_TOKENS


def count_gurmukhi(text: str) -> int:
    return len(GURMUKHI_RE.findall(text or ""))


def flag_reason(r: dict, by_i: dict, max_i: int,
                simran_kept_so_far: int, simran_cap: int) -> str:
    ci = r["clip_i"]
    text = r.get("text", "") or ""

    # Simran clips are legitimately repetitive, but we cap them in the dataset
    # anyway — so DON'T waste a Flash call on one that will be dropped at push.
    if is_simran_clip(text):
        if simran_cap and simran_kept_so_far >= simran_cap:
            return ""  # going to be dropped — skip Flash
        return ""  # simran is not a truncation bug; no need to re-transcribe

    if ci < INTRO_OUTRO_WINDOW:
        return "intro_window"
    if ci > max_i - INTRO_OUTRO_WINDOW:
        return "outro_window"

    tokens = text.split()
    if tokens:
        top_tok, top_count = Counter(tokens).most_common(1)[0]
        # ignore repeats of simran tokens — genuine chanting, not truncation
        if top_count >= REPEAT_THRESHOLD and top_tok not in SIMRAN_TOKENS:
            return f"repeat_{top_count}x_{top_tok[:10]}"

    if 0 < count_gurmukhi(text) < SHORT_TEXT_CHARS:
        prev_t = (by_i.get(ci - 1) or {}).get("text", "")
        next_t = (by_i.get(ci + 1) or {}).get("text", "")
        if max(count_gurmukhi(prev_t), count_gurmukhi(next_t)) > NEIGHBOR_LONG_CHARS:
            return "short_between_long"

    return ""


def fallback_transcribe(client, clip_path: Path, model: str) -> str:
    from google.genai import types

    # Prompt restored to the v5 form: inline, verbose empty-list, temp=0.1.
    # In v5.1 we moved rules to system_instruction and dropped temp to 0.0,
    # which made Flash deterministically hallucinate simran on ambiguous
    # instrumental intros (collapsed to a single high-likelihood mode that
    # happened to be wrong). The small temperature reintroduces just enough
    # variance that Flash prefers "" on truly silent/instrumental clips.
    prompt = (
        "This is a Sikh kirtan audio clip of about 20 seconds. Transcribe "
        "EXACTLY what is sung, word by word, in GURMUKHI SCRIPT ONLY (with "
        "correct matras). Include every repetition. If the clip is "
        "instrumental only, pure alaap/humming without words, spoken "
        'katha/fateh/announcements, or silent — return an empty string "". '
        "Do NOT fabricate Gurbani when none is sung. Return only the "
        "transcription text, no JSON, no quotes, no commentary."
    )
    audio = clip_path.read_bytes()
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[types.Part.from_text(text=prompt),
                          types.Part.from_bytes(data=audio, mime_type="audio/wav")],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                ),
            )
            t = (resp.text or "").strip().strip('"').strip()
            return t
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                time.sleep(2 ** attempt)
                continue
            if attempt == 2:
                print(f"  [err] {clip_path.name}: {err[:120]}", file=sys.stderr)
                return ""
            time.sleep(1)
    return ""


def process_item(src: Path, dst_dir: Path, clips_root: Path,
                 client, workers: int, primary_model: str,
                 fallback_model: str, simran_cap: int) -> dict:
    rows = [json.loads(l) for l in src.open()]
    if not rows:
        return {"total": 0, "flagged": 0, "changed": 0}

    by_i = {r["clip_i"]: r for r in rows}
    max_i = max(by_i)
    item_id = rows[0]["item_id"]

    # Walk rows in clip order so simran cap accounting is sequential
    rows_sorted = sorted(rows, key=lambda r: r["clip_i"])
    simran_kept = 0
    flagged = []
    for r in rows_sorted:
        if is_simran_clip(r.get("text", "") or ""):
            if simran_cap == 0 or simran_kept < simran_cap:
                simran_kept += 1
        reason = flag_reason(r, by_i, max_i, simran_kept, simran_cap)
        r["primary_text"] = r.get("text", "")
        r["primary_model"] = primary_model
        r["fallback_text"] = ""
        r["fallback_model"] = ""
        r["fallback_reviewed"] = False
        r["flag_reason"] = reason
        if reason:
            flagged.append(r)

    def do_one(r):
        clip_path = clips_root / item_id / f"clip_{r['clip_i']:05d}.wav"
        if not clip_path.exists():
            return r, None
        out = fallback_transcribe(client, clip_path, fallback_model)
        return r, out

    # Merge rule (v5.3): on intro/outro flags, accept Flash output ONLY if
    # Flash returns empty. Flash is reliable at saying "no Gurbani here" on
    # music/silence, but sometimes hallucinates on ambiguous intros. On mid-
    # track repeat/short flags, Flash's output is usually the correction we
    # want, so behavior is unchanged for those.
    changed = 0
    if flagged:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(do_one, r) for r in flagged]
            for fut in as_completed(futs):
                r, out = fut.result()
                if out is None:
                    continue
                r["fallback_reviewed"] = True
                r["fallback_text"] = out
                r["fallback_model"] = fallback_model
                is_edge = r["flag_reason"] in ("intro_window", "outro_window")
                if is_edge and out.strip():
                    continue  # keep Lite primary
                if out != r["primary_text"]:
                    r["text"] = out
                    changed += 1

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    with dst.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {"total": len(rows), "flagged": len(flagged), "changed": changed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/root/v3_data")
    ap.add_argument("--primary-model", default="gemini-2.5-flash-lite",
                    help="Label only — what produced transcripts/")
    ap.add_argument("--fallback-model", default=DEFAULT_FALLBACK_MODEL,
                    help="Model to re-transcribe flagged clips")
    ap.add_argument("--workers", type=int, default=12,
                    help="Concurrent fallback API calls")
    ap.add_argument("--simran-cap", type=int, default=2,
                    help="Max simran clips per item (must match push-step cap). "
                    "Flash is skipped for simran clips beyond this limit.")
    args = ap.parse_args()

    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    data_root = Path(args.data_root)
    src_dir = data_root / "transcripts"
    dst_dir = data_root / "transcripts_cleaned"
    clips_root = data_root / "clips"

    files = sorted(src_dir.glob("*.jsonl"))
    totals = {"total": 0, "flagged": 0, "changed": 0}
    for i, src in enumerate(files, 1):
        s = process_item(src, dst_dir, clips_root, client, args.workers,
                         args.primary_model, args.fallback_model,
                         args.simran_cap)
        for k in totals:
            totals[k] += s[k]
        print(f"  [{i}/{len(files)}] {src.stem}: "
              f"flagged={s['flagged']}/{s['total']}, changed={s['changed']}")

    pct_flag = 100 * totals["flagged"] / max(totals["total"], 1)
    pct_chg = 100 * totals["changed"] / max(totals["flagged"], 1)
    print(f"\n[summary] {totals['total']} rows, "
          f"{totals['flagged']} flagged ({pct_flag:.1f}%), "
          f"{totals['changed']} changed by {args.fallback_model} "
          f"({pct_chg:.1f}% of flagged)")


if __name__ == "__main__":
    main()
