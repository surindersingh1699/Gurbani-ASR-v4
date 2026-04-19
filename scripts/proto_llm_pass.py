#!/usr/bin/env python3
# DEPRECATED: kept for regression reference only.
# Production code lives at scripts/canonical/ and is tested in tests/canonical/.
"""Prototype: LLM fallback pass on unchanged/review rows from v2 pipeline.

Groups rows by retrieved shabad_id, batches 30 per Gemini call, verifies each
output against the shabad's token inventory (hallucination guard).

Uses google.genai SDK with gemini-2.5-flash-lite (project default).
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from proto_canonical_v2 import (  # noqa: E402
    DB_PATH, SAMPLE, build_shabad_idx, collapse_consecutive_repeats, lev,
    load_sggs, process_row, skel, tokenize,
)

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

MODEL = "gemini-2.5-flash-lite"
BATCH_SIZE = 30

SYSTEM_PROMPT = """You correct kirtan transcriptions to match Guru Granth Sahib (SGGS) verbatim.

RULES (strict):
1. For each caption, output a corrected version that preserves word order, \
repetitions, and whitespace structure.
2. You may ONLY use words that appear in the SGGS shabad provided below, \
with correct matras. Do not invent words.
3. If the caption clearly does not correspond to this shabad, output the \
caption UNCHANGED.
4. Do not add lines, repetitions, or words not present in the caption.
5. Output strictly as JSON matching the schema. Preserve clip_ids exactly.
"""


def build_prompt(shabad_text: str, ang: int, writer: str, batch_rows: list[dict]) -> str:
    items = "\n".join(
        f'  {{"clip_id": "{r["clip_id"]}", "caption": "{r["caption"]}"}}'
        for r in batch_rows
    )
    return f"""{SYSTEM_PROMPT}

SGGS shabad (Ang {ang}, {writer}):
{shabad_text}

Captions to correct ({len(batch_rows)} items):
[
{items}
]

Return JSON: {{"corrections": [{{"clip_id": "...", "corrected": "..."}}, ...]}}
"""


def call_gemini(client, prompt: str) -> list[dict]:
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=8000,
        ),
    )
    data = json.loads(resp.text)
    return data.get("corrections", [])


def verify(llm_text: str, shabad_tokens: set[str], shabad_skels: set[str]) -> bool:
    """True if every Gurmukhi token in llm_text is verbatim in shabad OR ≤ 1-edit skeleton."""
    for tok in llm_text.split():
        if not tok or tok == ">>":
            continue
        if tok in shabad_tokens:
            continue
        ts = skel(tok)
        if not ts:
            continue
        if any(lev(ts, ss) <= 1 for ss in shabad_skels):
            continue
        return False
    return True


def fetch_shabad_text(shabad_lines) -> tuple[str, int, str]:
    # Use the canonical unicode text of each line, joined with newlines
    text = "\n".join(ln.unicode for ln in shabad_lines)
    ang = shabad_lines[0].ang if shabad_lines else 0
    return text, ang, "Guru Sahib Ji"  # writer pulled elsewhere if needed


def main():
    print("[load] SGGS index...", file=sys.stderr)
    lines, global_tok_idx = load_sggs(DB_PATH)
    idx = build_shabad_idx(lines)

    # Run v2 pipeline first to identify unchanged + review rows
    print("[v2] running DB-grounded pass to identify unchanged/review rows...", file=sys.stderr)
    rows_raw = [r.strip() for r in SAMPLE.strip().splitlines()]
    processable = [(i, r) for i, r in enumerate(rows_raw) if tokenize(r)]

    from collections import Counter
    video_hits: Counter[str] = Counter()
    llm_candidates: list[dict] = []
    all_results: list[dict] = []
    for k, (orig_i, row) in enumerate(processable):
        prev = processable[k - 1][1] if k > 0 else ""
        nxt = processable[k + 1][1] if k + 1 < len(processable) else ""
        prev2 = processable[k - 2][1] if k > 1 else ""
        nxt2 = processable[k + 2][1] if k + 2 < len(processable) else ""
        r = process_row(row, idx, (prev, nxt, prev2, nxt2), global_tok_idx, video_hits)
        dbg = r.get("_debug", {})
        entry = {
            "row_num": orig_i + 1,
            "caption": row,
            "caption_clean": " ".join(tokenize(row)),
            "clip_id": f"row_{orig_i + 1:03d}",
            "decision": r["decision"],
            "final_text": r["final_text"],
            "sggs_line": r["sggs_line"],
            "shabad_id": dbg.get("shabad_id", ""),
        }
        all_results.append(entry)
        if r["decision"] in ("unchanged", "review") and entry["shabad_id"]:
            llm_candidates.append(entry)

    print(f"[v2] {len(llm_candidates)} candidates for LLM pass", file=sys.stderr)

    # Dedup identical captions before sending (cost saver #3)
    seen: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(llm_candidates):
        seen[c["caption_clean"]].append(i)
    unique_candidates = [llm_candidates[idxs[0]] for idxs in seen.values()]
    print(f"[v2] after dedup: {len(unique_candidates)} unique captions", file=sys.stderr)

    # Group by shabad
    by_shabad: dict[str, list[dict]] = defaultdict(list)
    for c in unique_candidates:
        by_shabad[c["shabad_id"]].append(c)
    print(f"[v2] {len(by_shabad)} shabad groups: {dict((k, len(v)) for k, v in by_shabad.items())}",
          file=sys.stderr)

    # LLM pass
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[llm] no GEMINI_API_KEY — skipping LLM pass", file=sys.stderr)
        sys.exit(1)
    client = genai.Client(api_key=api_key)

    _, shabad_lines_map, _ = idx
    llm_results: dict[str, dict] = {}  # clip_id -> {corrected, verified}

    total_input_chars = 0
    total_output_chars = 0
    total_calls = 0

    for sid, group in by_shabad.items():
        shabad_lines = shabad_lines_map[sid]
        shabad_text = "\n".join(ln.unicode for ln in shabad_lines)
        ang = shabad_lines[0].ang if shabad_lines else 0
        shabad_tokens = {t for ln in shabad_lines for t in ln.tokens}
        shabad_skels = {skel(t) for t in shabad_tokens if skel(t)}

        for i in range(0, len(group), BATCH_SIZE):
            batch = group[i:i + BATCH_SIZE]
            prompt = build_prompt(shabad_text, ang, "Guru Sahib Ji", batch)
            total_input_chars += len(prompt)
            total_calls += 1
            print(f"[llm] shabad={sid} batch={i//BATCH_SIZE+1} size={len(batch)}...",
                  file=sys.stderr)
            try:
                corrections = call_gemini(client, prompt)
            except Exception as e:
                print(f"[llm] ERROR: {e!r}", file=sys.stderr)
                continue
            for c in corrections:
                cid = c.get("clip_id", "")
                corrected = c.get("corrected", "")
                total_output_chars += len(corrected)
                verified = verify(corrected, shabad_tokens, shabad_skels)
                llm_results[cid] = {"corrected": corrected, "verified": verified}

    # Broadcast dedup results back to all original candidates
    final_llm_by_clip_id: dict[str, dict] = {}
    for c in llm_candidates:
        # find unique match
        canon = next((u for u in unique_candidates if u["caption_clean"] == c["caption_clean"]), None)
        if canon and canon["clip_id"] in llm_results:
            final_llm_by_clip_id[c["clip_id"]] = llm_results[canon["clip_id"]]

    # Print comparison table
    print()
    print(f"{'#':>3}  {'decision':10}  caption  →  v2_final_text  |  llm_final_text  [verified]")
    print("-" * 180)
    for entry in all_results:
        if entry["clip_id"] in final_llm_by_clip_id:
            llm = final_llm_by_clip_id[entry["clip_id"]]
            llm_str = llm["corrected"]
            v = "✓" if llm["verified"] else "✗"
            print(f"{entry['row_num']:>3}  {entry['decision']:10}  "
                  f"{entry['caption']}  →  {entry['final_text']}  "
                  f"|  {llm_str}  [{v}]")
    print()
    # Cost estimate (Flash-Lite: $0.10/M input, $0.40/M output; ~3 chars/token rough)
    est_input_tokens = total_input_chars / 3.5
    est_output_tokens = total_output_chars / 3.5
    est_cost = est_input_tokens * 0.10 / 1e6 + est_output_tokens * 0.40 / 1e6
    print(f"[stats] calls={total_calls}  "
          f"~input_tokens={int(est_input_tokens)}  ~output_tokens={int(est_output_tokens)}  "
          f"est_cost=${est_cost:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
