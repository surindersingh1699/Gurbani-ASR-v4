#!/usr/bin/env python3
"""Compare Gemini 3 Flash vs Gemini 3.1 Pro on the unchanged/review rows from v2.

Shares the v2 pipeline + verify logic from proto_llm_pass.py.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from proto_canonical_v2 import (  # noqa: E402
    DB_PATH, SAMPLE, build_shabad_idx, lev, load_sggs, process_row, skel, tokenize,
)

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

MODELS = [
    # label, model_id, batch_size (3 Flash truncates JSON at 30, use 10)
    ("3-flash",  "gemini-3-flash-preview",  10),
    ("3.1-pro",  "gemini-3.1-pro-preview",  30),
]

SYSTEM_PROMPT = """You correct kirtan transcriptions to match Guru Granth Sahib (SGGS) verbatim.

RULES (strict, must follow):
1. For each caption, output a corrected version that preserves word order, \
repetitions, and token count (± 1 token max).
2. You may ONLY use words that appear in the SGGS shabad provided below, \
with correct matras. Do not invent words.
3. If the caption clearly does not correspond to this shabad, output the \
caption UNCHANGED.
4. Do NOT add lines, repetitions, or words not present in the caption.
5. Do NOT drop trailing tokens from the caption.
6. Output strictly as JSON matching the schema. Preserve clip_ids exactly.
"""


def build_prompt(shabad_text: str, ang: int, batch_rows: list[dict]) -> str:
    items = "\n".join(
        f'  {{"clip_id": "{r["clip_id"]}", "caption": "{r["caption"]}"}}'
        for r in batch_rows
    )
    return f"""{SYSTEM_PROMPT}

SGGS shabad (Ang {ang}):
{shabad_text}

Captions to correct ({len(batch_rows)} items):
[
{items}
]

Return JSON: {{"corrections": [{{"clip_id": "...", "corrected": "..."}}, ...]}}
"""


def call_gemini(client, model_id: str, prompt: str) -> list[dict]:
    resp = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=8000,
        ),
    )
    data = json.loads(resp.text)
    return data.get("corrections", [])


def verify(llm_text: str, shabad_tokens: set[str], shabad_skels: set[str],
           caption_len: int) -> tuple[bool, str]:
    llm_tokens = [t for t in llm_text.split() if t and t != ">>"]
    # Structure preservation check: token count within ±2 of caption
    if abs(len(llm_tokens) - caption_len) > 2:
        return False, f"len_drift({len(llm_tokens)}vs{caption_len})"
    # Token inventory check
    for tok in llm_tokens:
        if tok in shabad_tokens:
            continue
        ts = skel(tok)
        if not ts:
            continue
        if any(lev(ts, ss) <= 1 for ss in shabad_skels):
            continue
        return False, f"invented({tok})"
    return True, "ok"


def run_model(label: str, model_id: str, batch_size: int,
              by_shabad: dict, shabad_lines_map,
              client) -> tuple[dict[str, dict], dict]:
    """Returns (clip_id → {corrected, verified, reason}, stats)."""
    results: dict[str, dict] = {}
    stats = {"calls": 0, "input_chars": 0, "output_chars": 0,
             "latency_s": 0.0, "errors": 0}
    for sid, group in by_shabad.items():
        shabad_lines = shabad_lines_map[sid]
        shabad_text = "\n".join(ln.unicode for ln in shabad_lines)
        ang = shabad_lines[0].ang if shabad_lines else 0
        shabad_tokens = {t for ln in shabad_lines for t in ln.tokens}
        shabad_skels = {skel(t) for t in shabad_tokens if skel(t)}

        for i in range(0, len(group), batch_size):
            batch = group[i:i + batch_size]
            prompt = build_prompt(shabad_text, ang, batch)
            stats["calls"] += 1
            stats["input_chars"] += len(prompt)
            print(f"[{label}] shabad={sid} batch={i//batch_size+1} "
                  f"size={len(batch)}...", file=sys.stderr)
            t0 = time.time()
            try:
                corrections = call_gemini(client, model_id, prompt)
            except Exception as e:
                print(f"[{label}] ERROR: {e!r}", file=sys.stderr)
                stats["errors"] += 1
                continue
            stats["latency_s"] += time.time() - t0
            # Map clip_id → batch row for caption_len lookup
            cid_to_caplen = {r["clip_id"]: len(tokenize(r["caption"])) for r in batch}
            for c in corrections:
                cid = c.get("clip_id", "")
                corrected = c.get("corrected", "")
                stats["output_chars"] += len(corrected)
                verified, reason = verify(corrected, shabad_tokens, shabad_skels,
                                          cid_to_caplen.get(cid, 0))
                results[cid] = {"corrected": corrected, "verified": verified, "reason": reason}
    return results, stats


def main():
    print("[load] SGGS index...", file=sys.stderr)
    lines, global_tok_idx = load_sggs(DB_PATH)
    idx = build_shabad_idx(lines)
    _, shabad_lines_map, _ = idx

    print("[v2] running DB-grounded pass...", file=sys.stderr)
    rows_raw = [r.strip() for r in SAMPLE.strip().splitlines()]
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
            "row_num": orig_i + 1,
            "caption": row,
            "caption_clean": " ".join(tokenize(row)),
            "clip_id": f"row_{orig_i + 1:03d}",
            "decision": r["decision"],
            "final_text": r["final_text"],
            "shabad_id": dbg.get("shabad_id", ""),
        })

    llm_candidates = [
        e for e in all_results
        if e["decision"] in ("unchanged", "review") and e["shabad_id"]
    ]
    # Dedup
    seen: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(llm_candidates):
        seen[c["caption_clean"]].append(i)
    unique = [llm_candidates[idxs[0]] for idxs in seen.values()]
    by_shabad: dict[str, list[dict]] = defaultdict(list)
    for c in unique:
        by_shabad[c["shabad_id"]].append(c)
    print(f"[v2] {len(llm_candidates)} candidates → {len(unique)} unique → "
          f"{len(by_shabad)} shabads", file=sys.stderr)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[llm] no GEMINI_API_KEY", file=sys.stderr)
        sys.exit(1)
    client = genai.Client(api_key=api_key)

    all_model_results: dict[str, dict[str, dict]] = {}
    all_stats: dict[str, dict] = {}
    for label, mid, bs in MODELS:
        print(f"\n=== running {label} ({mid}) batch={bs} ===", file=sys.stderr)
        res, st = run_model(label, mid, bs, by_shabad, shabad_lines_map, client)
        all_model_results[label] = res
        all_stats[label] = st

    # Broadcast dedup
    def lookup_for(entry, model_results):
        canon = next((u for u in unique if u["caption_clean"] == entry["caption_clean"]),
                     None)
        if canon and canon["clip_id"] in model_results:
            return model_results[canon["clip_id"]]
        return None

    # Print comparison table
    print()
    hdr = f"{'#':>3} {'dec':10} {'caption':46}  {'v2_final_text':42} "
    for label, _, _ in MODELS:
        hdr += f" | {label}_final_text" + " " * 34 + "[v]"
    print(hdr)
    print("-" * 220)

    for e in all_results:
        if e["decision"] not in ("unchanged", "review"):
            continue
        row = f"{e['row_num']:>3} {e['decision']:10} {e['caption'][:44]:46}  {e['final_text'][:40]:42}"
        for label, _, _ in MODELS:
            r = lookup_for(e, all_model_results[label])
            if r:
                v = "✓" if r["verified"] else f"✗({r['reason']})"
                row += f" | {r['corrected'][:42]:42} [{v}]"
            else:
                row += " | " + "-" * 42 + " [ ]"
        print(row)

    print()
    # Cost / latency summary
    for label, mid, _ in MODELS:
        st = all_stats[label]
        # rough: 3.5 chars/token
        in_tok = st["input_chars"] / 3.5
        out_tok = st["output_chars"] / 3.5
        # Pricing (per 1M tokens): 3-flash = 0.50/3.00, 3.1-pro = 2.00/12.00
        if "flash" in label:
            cost = in_tok * 0.50 / 1e6 + out_tok * 3.00 / 1e6
        else:
            cost = in_tok * 2.00 / 1e6 + out_tok * 12.00 / 1e6
        print(f"[stats] {label}: calls={st['calls']} latency={st['latency_s']:.1f}s "
              f"in_tok~{int(in_tok)} out_tok~{int(out_tok)} errs={st['errors']} "
              f"est_cost=${cost:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
