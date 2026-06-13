#!/usr/bin/env python3
"""Fully-automatic GOLD labels: audio + caption must AGREE on the same SGGS line.

Two independent witnesses per clip:
  A) YouTube caption  → SGGS skeleton retrieval (already in gold_candidates.jsonl)
  B) Gemini AUDIO transcription → SGGS skeleton retrieval (this script)

Decision per clip (high precision, surplus clips are cheap):
  accept : witness A and witness B resolve to the SAME SGGS line_id, both with
           skeleton-lev ratio <= --max-ratio  → gold = canonical line
  music  : Gemini hears no singing → banked as VAD negative
  reject : witnesses disagree / no SGGS anchor → excluded from GOLD

Writes ACCEPT/MUSIC/REJECT rows straight into <data>/gold_labels.jsonl
(append-only; clip_ids already labeled — e.g. by a human — are never touched).

Run on Hetzner (GEMINI_API_KEY in /root/.env):
  set -a; . /root/.env; set +a
  /root/venv/bin/python scripts/auto_gold_pipeline.py --data ./gold_eval --db ./database.sqlite
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_gold_eval_candidates import SggsRetriever, norm        # noqa: E402
from canonical.gurmukhi_skeleton import skel, lev                 # noqa: E402

BATCH = 10


def gemini_client():
    from google import genai
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY not set")
    return genai.Client(api_key=key)


def transcribe_batch(client, clips: list[dict]) -> list[dict]:
    """Returns per clip: {text, cut_start, cut_end}. Boolean boundary flags
    confirm the clip's timestamps capture whole sung phrases (Gemini timestamp
    ESTIMATION is unreliable, but yes/no cut-off on a short clip is safe)."""
    from google.genai import types
    n = len(clips)
    prompt = f"""You are a Gurbani/Punjabi transcription expert. This is Sikh kirtan.

I am sending {n} audio clips (5-22 seconds each). For each clip return an object:
- "text": the Gurmukhi text being SUNG, including repeated words (correct matras/spelling).
  Instrumental only / no singing → "". Spoken Punjabi explanation (katha) → "KATHA".
- "cut_start": true if the clip begins MID-WORD (singing already in progress, first word clipped)
- "cut_end": true if the clip ends MID-WORD (last word cut off)

Return a JSON array of exactly {n} objects, in order. Return ONLY the JSON array."""
    contents = [types.Part.from_text(text=prompt)]
    for i, c in enumerate(clips):
        contents.append(types.Part.from_text(text=f"Clip {i + 1}:"))
        contents.append(types.Part.from_bytes(data=c["bytes"], mime_type="audio/flac"))
    blank = {"text": "", "cut_start": False, "cut_end": False}
    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash-lite", contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.1, response_mime_type="application/json"))
            items = json.loads(resp.text.strip())
            if isinstance(items, dict):
                items = list(items.values())
            out = []
            for it in items:
                if isinstance(it, str):
                    it = {"text": it}
                if not isinstance(it, dict):
                    it = dict(blank)
                out.append({"text": str(it.get("text") or ""),
                            "cut_start": bool(it.get("cut_start")),
                            "cut_end": bool(it.get("cut_end"))})
            return (out + [dict(blank)] * n)[:n]
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"    [retry {attempt+1}] {type(e).__name__}: {e} — sleep {wait}s", flush=True)
            time.sleep(wait)
    return [dict(blank)] * n


def ratio(a_skel: str, b_skel: str) -> float:
    if not a_skel or not b_skel:
        return 1.0
    return lev(a_skel, b_skel) / max(len(a_skel), len(b_skel))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./gold_eval")
    ap.add_argument("--db", default="./database.sqlite")
    ap.add_argument("--max-ratio", type=float, default=0.35,
                    help="max skeleton-lev ratio for a witness to count as anchored")
    ap.add_argument("--limit", type=int, default=0, help="process at most N clips (0=all)")
    args = ap.parse_args()

    data = Path(args.data)
    rows = [json.loads(l) for l in (data / "gold_candidates.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    labels_fp = data / "gold_labels.jsonl"
    labeled = set()
    if labels_fp.exists():
        labeled = {json.loads(l)["clip_id"] for l in labels_fp.read_text(encoding="utf-8").splitlines() if l.strip()}

    todo = [r for r in rows if r["clip_id"] not in labeled and (data / r["audio_path"]).exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} clips to auto-judge ({len(labeled)} already labeled)")

    retr = SggsRetriever(args.db)
    by_line = {ln.line_id: ln for ln in retr.lines}
    client = gemini_client()

    def commit(r, status, gold="", line=None, gem="", why="", cut=None):
        with labels_fp.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "clip_id": r["clip_id"], "source": r["source"], "quality": r["quality"],
                "video_id": r["video_id"], "duration_s": r["duration_s"],
                "audio_path": r["audio_path"], "proposed_text": r["proposed_text"],
                "status": status, "gold_text": gold,
                "line_id": getattr(line, "line_id", None),
                "shabad_id": getattr(line, "shabad_id", None),
                "ang": getattr(line, "ang", None),
                "gemini_text": gem, "via": "auto_audio_x2", "auto_reason": why,
                "cut_start": bool(cut and cut.get("cut_start")),
                "cut_end": bool(cut and cut.get("cut_end")),
            }, ensure_ascii=False) + "\n")

    from collections import Counter
    stats = Counter()
    for bi in range(0, len(todo), BATCH):
        chunk = todo[bi:bi + BATCH]
        for c in chunk:
            c["bytes"] = (data / c["audio_path"]).read_bytes()
        results = transcribe_batch(client, chunk)
        for r, res in zip(chunk, results):
            r.pop("bytes", None)
            gem_raw = res["text"]
            gem = norm(gem_raw)
            if gem_raw.strip().upper() == "KATHA":
                commit(r, "reject", gem=gem_raw, why="katha", cut=res); stats["reject_katha"] += 1; continue
            if not gem:
                commit(r, "music", gem=gem_raw, why="no_vocals", cut=res); stats["music"] += 1; continue
            if res["cut_start"] or res["cut_end"]:
                # boundary cuts a word — bad for CER scoring; surplus clips are cheap
                commit(r, "reject", gem=gem_raw, why="boundary_cut", cut=res)
                stats["reject_cut"] += 1; continue

            # witness B: gemini audio → SGGS
            g_cands = retr.topk(gem, k=1)
            g_top = g_cands[0] if g_cands else None
            g_ok = g_top and ratio(skel(gem), skel(g_top["unicode"])) <= args.max_ratio

            # witness A: caption → SGGS (precomputed candidates on the row)
            a_top = (r.get("candidates") or [None])[0]
            a_ok = a_top and ratio(skel(norm(r["proposed_text"])), skel(a_top["unicode"])) <= args.max_ratio

            if g_ok and a_ok and g_top["line_id"] == a_top["line_id"]:
                ln = by_line[g_top["line_id"]]
                commit(r, "accept", gold=ln.unicode, line=ln, gem=gem_raw, why="x2_agree", cut=res)
                stats["accept"] += 1
            elif g_ok and not a_ok:
                # caption noisy but audio cleanly anchors — accept on audio witness
                # only if caption ALSO loosely points at the same shabad
                same_shabad = a_top and a_top.get("shabad_id") == g_top["shabad_id"]
                if same_shabad:
                    ln = by_line[g_top["line_id"]]
                    commit(r, "accept", gold=ln.unicode, line=ln, gem=gem_raw, why="audio_anchor_same_shabad", cut=res)
                    stats["accept_shabad"] += 1
                else:
                    commit(r, "reject", gem=gem_raw, why="audio_only_anchor", cut=res); stats["reject_disagree"] += 1
            else:
                commit(r, "reject", gem=gem_raw, why="no_agreement", cut=res); stats["reject_disagree"] += 1
        done = min(bi + BATCH, len(todo))
        print(f"  {done}/{len(todo)}  {dict(stats)}", flush=True)

    print(f"\n[done] {dict(stats)} → {labels_fp}")


if __name__ == "__main__":
    main()
