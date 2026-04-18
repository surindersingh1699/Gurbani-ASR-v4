#!/usr/bin/env python3
"""Diagnostic: run the CEF batch=23 case and dump prompt size, raw response,
and Gemini usage metadata. Tells us exactly why/how truncation happened.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from proto_canonical_v2 import (  # noqa: E402
    DB_PATH, build_shabad_idx, load_sggs, process_row, tokenize,
)

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

MODEL = "gemini-3.1-pro-preview"

SYSTEM_PROMPT = """You correct kirtan transcriptions to match Guru Granth Sahib (SGGS) verbatim.

RULES (strict):
1. For each caption, output a corrected version that preserves word order, \
repetitions, and token count (± 1 token max).
2. You may ONLY use words that appear in the SGGS shabad provided below, \
with correct matras. Do not invent words.
3. If the caption clearly does not correspond to this shabad, output it UNCHANGED.
4. Do NOT add lines, repetitions, or words not present in the caption.
5. Do NOT drop trailing tokens from the caption.
6. Output strictly as JSON. Preserve clip_ids exactly.
"""


def build_prompt(shabad_text, ang, batch):
    items = "\n".join(
        f'  {{"clip_id": "{r["clip_id"]}", "caption": "{r["caption"]}"}}'
        for r in batch
    )
    return f"""{SYSTEM_PROMPT}

SGGS shabad (Ang {ang}):
{shabad_text}

Captions to correct ({len(batch)} items):
[
{items}
]

Return JSON: {{"corrections": [{{"clip_id": "...", "corrected": "..."}}, ...]}}
"""


def main():
    sample_path = Path(sys.argv[1])
    sample = sample_path.read_text()

    lines, global_tok_idx = load_sggs(DB_PATH)
    idx = build_shabad_idx(lines)
    _, shabad_lines_map, _ = idx

    rows_raw = [r.strip() for r in sample.strip().splitlines()]
    processable = [(i, r) for i, r in enumerate(rows_raw) if tokenize(r)]
    video_hits: Counter[str] = Counter()
    all_results: list[dict] = []
    for k, (orig_i, row) in enumerate(processable):
        prev = processable[k - 1][1] if k > 0 else ""
        nxt = processable[k + 1][1] if k + 1 < len(processable) else ""
        prev2 = processable[k - 2][1] if k > 1 else ""
        nxt2 = processable[k + 2][1] if k + 2 < len(processable) else ""
        r = process_row(row, idx, (prev, nxt, prev2, nxt2), global_tok_idx, video_hits)
        dbg = r.get("_debug", {})
        all_results.append({
            "row_num": orig_i + 1, "caption": row,
            "caption_clean": " ".join(tokenize(row)),
            "clip_id": f"row_{orig_i + 1:03d}",
            "decision": r["decision"], "shabad_id": dbg.get("shabad_id", ""),
        })

    llm_candidates = [e for e in all_results
                      if e["decision"] in ("unchanged", "review") and e["shabad_id"]]
    seen: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(llm_candidates):
        seen[c["caption_clean"]].append(i)
    unique = [llm_candidates[idxs[0]] for idxs in seen.values()]

    # Grab the CEF group (the big one)
    cef_group = [c for c in unique if c["shabad_id"] == "CEF"]
    shabad_lines = shabad_lines_map["CEF"]
    shabad_text = "\n".join(ln.unicode for ln in shabad_lines)
    ang = shabad_lines[0].ang if shabad_lines else 0

    # Build the same prompt that caused truncation
    prompt = build_prompt(shabad_text, ang, cef_group)

    print("=" * 80)
    print(f"CEF batch size: {len(cef_group)} rows")
    print(f"SGGS shabad 'CEF' Ang {ang} — {len(shabad_lines)} lines")
    print(f"Shabad text chars: {len(shabad_text)}")
    print(f"Total prompt chars: {len(prompt)}")
    # Gurmukhi is ~3 bytes per char in UTF-8; Gemini tokens roughly 1/4 of that
    print(f"Prompt UTF-8 bytes: {len(prompt.encode('utf-8'))}")
    print("=" * 80)
    print("--- FIRST 500 CHARS OF PROMPT ---")
    print(prompt[:500])
    print("--- LAST 500 CHARS OF PROMPT ---")
    print(prompt[-500:])
    print("=" * 80)

    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    # Call with explicit large output budget
    resp = client.models.generate_content(
        model=MODEL, contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0, max_output_tokens=32000,
        ),
    )
    print()
    print("=== GEMINI RESPONSE METADATA ===")
    try:
        um = resp.usage_metadata
        print(f"prompt_token_count:     {um.prompt_token_count}")
        print(f"candidates_token_count: {um.candidates_token_count}")
        print(f"total_token_count:      {um.total_token_count}")
    except Exception as e:
        print(f"usage_metadata unavailable: {e!r}")
    try:
        cands = resp.candidates
        for c in cands:
            fr = getattr(c, "finish_reason", None)
            print(f"finish_reason: {fr}")
    except Exception as e:
        print(f"finish_reason unavailable: {e!r}")

    print()
    print(f"=== RAW RESPONSE TEXT ({len(resp.text)} chars) ===")
    print(resp.text[:3000])
    if len(resp.text) > 3000:
        print("...")
        print(f"[truncated for display; total {len(resp.text)} chars]")
        print(resp.text[-1000:])

    print()
    print("=== PARSE ===")
    try:
        data = json.loads(resp.text)
        corrections = data.get("corrections", [])
        print(f"Parsed OK. Got {len(corrections)} corrections out of {len(cef_group)} requested.")
        sent_ids = {r["clip_id"] for r in cef_group}
        got_ids = {c.get("clip_id", "") for c in corrections}
        missing = sent_ids - got_ids
        print(f"Missing clip_ids: {sorted(missing)}")
    except Exception as e:
        print(f"JSON parse failed: {e}")


if __name__ == "__main__":
    main()
