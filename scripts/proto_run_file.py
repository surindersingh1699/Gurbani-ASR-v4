#!/usr/bin/env python3
"""Run v2 DB pipeline + Gemini 3.1 Pro LLM fallback on a sample file.

Usage: python3 scripts/proto_run_file.py /tmp/new_sample.txt
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from proto_canonical_v2 import (  # noqa: E402
    DB_PATH, build_shabad_idx, lev, load_sggs, process_row, skel, tokenize,
)

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

MODEL = "gemini-3.1-pro-preview"
BATCH_SIZE = 30

SYSTEM_PROMPT = """You correct kirtan transcriptions to match Guru Granth Sahib (SGGS) verbatim.

RULES (strict, must follow ALL):
1. You MUST return EXACTLY ONE corrections entry per input clip_id. No omissions.
2. If the caption is already correct or cannot be confidently corrected, still \
include it in the output — copy the caption verbatim as "corrected".
3. When correcting, preserve word order, repetitions, and token count (± 1 max).
4. You may ONLY use words that appear in the SGGS shabad provided below, with \
correct matras. Do not invent words.
5. Do NOT add lines, repetitions, or words not present in the caption.
6. Do NOT drop trailing tokens from the caption.
7. Output strictly as JSON. Preserve clip_ids exactly. The output's corrections \
array MUST have the same length as the input captions array.
"""


def build_prompt(shabad_text: str, ang: int, batch: list[dict]) -> str:
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


def verify(llm_text: str, shabad_tokens, shabad_skels, caption_len: int):
    toks = [t for t in llm_text.split() if t and t != ">>"]
    if abs(len(toks) - caption_len) > 3:
        return False, f"len_drift({len(toks)}vs{caption_len})"
    for t in toks:
        if t in shabad_tokens:
            continue
        ts = skel(t)
        if not ts:
            continue
        if any(lev(ts, ss) <= 1 for ss in shabad_skels):
            continue
        return False, f"invented({t})"
    return True, "ok"


def main():
    sample_path = Path(sys.argv[1])
    sample = sample_path.read_text()

    print("[load] SGGS index...", file=sys.stderr)
    lines, global_tok_idx = load_sggs(DB_PATH)
    idx = build_shabad_idx(lines)
    _, shabad_lines_map, _ = idx

    rows_raw = [r.strip() for r in sample.strip().splitlines()]
    processable = [(i, r) for i, r in enumerate(rows_raw) if tokenize(r)]
    video_hits: Counter[str] = Counter()
    all_results: list[dict] = []
    print("[v2] running DB-grounded pass...", file=sys.stderr)
    for k, (orig_i, row) in enumerate(processable):
        prev = processable[k - 1][1] if k > 0 else ""
        nxt = processable[k + 1][1] if k + 1 < len(processable) else ""
        prev2 = processable[k - 2][1] if k > 1 else ""
        nxt2 = processable[k + 2][1] if k + 2 < len(processable) else ""
        r = process_row(row, idx, (prev, nxt, prev2, nxt2), global_tok_idx, video_hits)
        dbg = r.get("_debug", {})
        all_results.append({
            "row_num": orig_i + 1,
            "caption": row,
            "caption_clean": " ".join(tokenize(row)),
            "clip_id": f"row_{orig_i + 1:03d}",
            "decision": r["decision"],
            "final_text": r["final_text"],
            "sggs_line": r["sggs_line"],
            "shabad_id": dbg.get("shabad_id", ""),
        })

    # Summary counts
    dec_counts = Counter(e["decision"] for e in all_results)
    print(f"[v2] counts: {dict(dec_counts)}", file=sys.stderr)

    # Pick LLM candidates
    llm_candidates = [
        e for e in all_results
        if e["decision"] in ("unchanged", "review") and e["shabad_id"]
    ]
    seen: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(llm_candidates):
        seen[c["caption_clean"]].append(i)
    unique = [llm_candidates[idxs[0]] for idxs in seen.values()]
    by_shabad: dict[str, list[dict]] = defaultdict(list)
    for c in unique:
        by_shabad[c["shabad_id"]].append(c)
    print(f"[v2] LLM candidates: {len(llm_candidates)} → {len(unique)} unique → "
          f"{len(by_shabad)} shabads", file=sys.stderr)

    api_key = os.environ.get("GEMINI_API_KEY") if os.environ.get("USE_LLM", "1") != "0" else None
    if not api_key:
        print("[llm] skipping (no key or USE_LLM=0)", file=sys.stderr)
        llm_results: dict[str, dict] = {}
    else:
        client = genai.Client(api_key=api_key)
        llm_results = {}
        total_cost = 0.0
        for sid, group in by_shabad.items():
            shabad_lines = shabad_lines_map[sid]
            shabad_text = "\n".join(ln.unicode for ln in shabad_lines)
            ang = shabad_lines[0].ang if shabad_lines else 0
            shabad_tokens = {t for ln in shabad_lines for t in ln.tokens}
            shabad_skels = {skel(t) for t in shabad_tokens if skel(t)}
            for i in range(0, len(group), BATCH_SIZE):
                batch = group[i:i + BATCH_SIZE]
                prompt = build_prompt(shabad_text, ang, batch)
                print(f"[llm] shabad={sid} batch={i//BATCH_SIZE+1} "
                      f"size={len(batch)}...", file=sys.stderr)
                try:
                    resp = client.models.generate_content(
                        model=MODEL, contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.0, max_output_tokens=12000,
                        ),
                    )
                    data = json.loads(resp.text)
                except Exception as e:
                    print(f"[llm] ERROR: {e!r}", file=sys.stderr)
                    continue
                cid_to_caplen = {r["clip_id"]: len(tokenize(r["caption"])) for r in batch}
                in_tok = len(prompt) / 3.5
                out_tok = len(resp.text) / 3.5
                total_cost += in_tok * 2.0 / 1e6 + out_tok * 12.0 / 1e6
                for c in data.get("corrections", []):
                    cid = c.get("clip_id", "")
                    corrected = c.get("corrected", "")
                    verified, reason = verify(corrected, shabad_tokens, shabad_skels,
                                              cid_to_caplen.get(cid, 0))
                    llm_results[cid] = {"corrected": corrected, "verified": verified,
                                        "reason": reason}
        print(f"[llm] est_cost=${total_cost:.4f}", file=sys.stderr)

    # Broadcast dedup
    def lookup_for(entry):
        canon = next((u for u in unique if u["caption_clean"] == entry["caption_clean"]),
                     None)
        if canon and canon["clip_id"] in llm_results:
            return llm_results[canon["clip_id"]]
        return None

    # Print unified table
    print()
    print(f"{'#':>3} {'dec':10} {'sh':4}  caption  →  v2_final_text  |  llm_final_text [v]")
    print("-" * 200)
    for e in all_results:
        lrow = ""
        if e["decision"] in ("unchanged", "review"):
            lr = lookup_for(e)
            if lr:
                v = "✓" if lr["verified"] else f"✗({lr['reason']})"
                lrow = f"  |  {lr['corrected']} [{v}]"
            else:
                lrow = "  |  -"
        print(f"{e['row_num']:>3} {e['decision']:10} {e['shabad_id']:4}  "
              f"{e['caption']}  →  {e['final_text']}{lrow}")


if __name__ == "__main__":
    main()
