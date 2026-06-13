#!/usr/bin/env python3
"""LLM pre-correction pass for GOLD-eval candidates.

For each clip with a caption (proposed_text), Claude judges it against the
top-K SGGS database candidates and emits a best-guess GOLD label:
  exact     — a candidate matches the caption (modulo matras/spacing) → safe bulk-accept
  corrected — caption is close to a candidate; use the canonical SGGS line
  unsure    — can't decide from text alone; human must listen
  garbage   — caption is noise (English residue, >> markers, simran-only)

Human stays the judge — the app preselects these and offers one-click bulk
accept of 'exact' rows. Clips with no caption (livestream adds) are skipped.

Auth: uses the Anthropic SDK when ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN is
set; otherwise shells out to `claude -p` (Claude Code login — no key needed).

Usage:
  python3 scripts/llm_precorrect_gold.py --data ./gold_eval [--batch 20] [--model claude-opus-4-8]

Reads  <data>/gold_candidates.jsonl
Writes <data>/llm_suggestions.jsonl   (keyed by clip_id; resumable, append-only)
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys
from pathlib import Path

MODEL = "claude-opus-4-8"

PROMPT = """You are correcting ASR/caption labels for Gurbani kirtan audio clips \
against the canonical Sri Guru Granth Sahib (SGGS) text.

For EACH clip below you get the proposed caption (from YouTube captions or ASR, \
Gurmukhi, possibly noisy) and up to 3 candidate SGGS lines retrieved by \
consonant-skeleton match (with ang number and edit distance `lev`).

Decide per clip:
- "exact": a candidate is the same line as the caption (ignore matra/spacing/vishraam \
differences, ignore repeated words from singing). gold_text = that candidate's unicode.
- "corrected": caption is clearly a noisy rendering of one candidate (word dropped, \
sung repetition, partial line). gold_text = that candidate's unicode.
- "unsure": no candidate convincingly matches, or caption could be a different line \
not retrieved. gold_text = best guess or the caption itself.
- "garbage": caption is not Gurbani text (English, '>>' residue, pure ਵਾਹਿਗੁਰੂ simran, \
empty). gold_text = "".

Reply with ONLY a JSON array, one object per clip, no markdown fences:
[{"clip_id": "...", "verdict": "exact|corrected|unsure|garbage", "cand": <index 0-2 or -1>, "gold_text": "..."}]

CLIPS:
"""


def render_batch(rows: list[dict]) -> str:
    out = []
    for r in rows:
        c = {
            "clip_id": r["clip_id"],
            "caption": r["proposed_text"],
            "candidates": [
                {"i": i, "unicode": m["unicode"], "ang": m["ang"], "lev": m["lev"]}
                for i, m in enumerate(r["candidates"][:3])
            ],
        }
        out.append(json.dumps(c, ensure_ascii=False))
    return PROMPT + "\n".join(out)


def extract_json_array(text: str):
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < 0:
        raise ValueError(f"no JSON array in reply: {text[:200]}")
    return json.loads(text[start:end + 1])


def call_llm_sdk(prompt: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    with client.messages.stream(
        model=model, max_tokens=32000, thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        msg = stream.get_final_message()
    return next(b.text for b in msg.content if b.type == "text")


def call_llm_cli(prompt: str, model: str) -> str:
    # scrub harness vars — a nested Claude Code session sets ANTHROPIC_BASE_URL
    # to a session proxy that 401s for child processes
    env = {k: v for k, v in os.environ.items()
           if k != "ANTHROPIC_BASE_URL" and not k.startswith("CLAUDE")}
    r = subprocess.run(
        ["claude", "-p", "--model", model, "--output-format", "text"],
        input=prompt, capture_output=True, text=True, timeout=600, env=env)
    reply = r.stdout
    if r.returncode != 0 or "Invalid authentication" in reply or "Not logged in" in reply:
        raise RuntimeError(f"claude -p failed: {(r.stderr or reply)[-500:]} "
                           "(run this script from your own terminal, not inside a Claude session)")
    return reply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./gold_eval")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    data = Path(args.data)
    cand_fp = data / "gold_candidates.jsonl"
    out_fp = data / "llm_suggestions.jsonl"
    if not cand_fp.exists():
        sys.exit(f"no {cand_fp} — run build_gold_eval_candidates.py first")

    rows = [json.loads(l) for l in cand_fp.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = set()
    if out_fp.exists():
        done = {json.loads(l)["clip_id"] for l in out_fp.read_text(encoding="utf-8").splitlines() if l.strip()}

    todo = [r for r in rows if r.get("proposed_text") and r["clip_id"] not in done]
    print(f"{len(todo)} clips to judge ({len(done)} already done, "
          f"{sum(1 for r in rows if not r.get('proposed_text'))} caption-less skipped)")

    use_sdk = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    call = call_llm_sdk if use_sdk else call_llm_cli
    print(f"backend: {'anthropic SDK' if use_sdk else 'claude -p (Claude Code login)'} · model {args.model}")

    by_id = {r["clip_id"]: r for r in rows}
    n_ok = 0
    for i in range(0, len(todo), args.batch):
        chunk = todo[i:i + args.batch]
        try:
            reply = call(render_batch(chunk), args.model)
            results = extract_json_array(reply)
        except Exception as e:
            print(f"  batch {i//args.batch}: FAILED ({e}) — will retry on next run", flush=True)
            continue
        with out_fp.open("a", encoding="utf-8") as f:
            for res in results:
                cid = res.get("clip_id")
                src = by_id.get(cid)
                if not src or cid in done:
                    continue
                ci = res.get("cand", -1)
                meta = src["candidates"][ci] if isinstance(ci, int) and 0 <= ci < len(src["candidates"]) else {}
                f.write(json.dumps({
                    "clip_id": cid,
                    "verdict": res.get("verdict", "unsure"),
                    "gold_text": (res.get("gold_text") or "").strip(),
                    "line_id": meta.get("line_id"), "shabad_id": meta.get("shabad_id"),
                    "ang": meta.get("ang"),
                }, ensure_ascii=False) + "\n")
                done.add(cid); n_ok += 1
        print(f"  {min(i + args.batch, len(todo))}/{len(todo)} judged", flush=True)

    print(f"[done] {n_ok} new suggestions → {out_fp}")
    if out_fp.exists():
        from collections import Counter
        verdicts = Counter(json.loads(l)["verdict"]
                           for l in out_fp.read_text(encoding="utf-8").splitlines() if l.strip())
        print("verdicts:", dict(verdicts))


if __name__ == "__main__":
    main()
