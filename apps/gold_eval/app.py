"""GATE 0 — GOLD eval hand-labeling app (local Streamlit).

Verify ASR labels against canonical SGGS, fast. For each clip: listen, then
1-click accept a candidate / edit it / reject. Autosaves every action, resumable.

Run:
    pip install streamlit soundfile
    cd <repo>
    HF_TOKEN=... streamlit run apps/gold_eval/app.py -- --data ./gold_eval

Reads  <data>/gold_candidates.jsonl  (+ <data>/audio/*.flac)
Writes <data>/gold_labels.jsonl       (decisions, keyed by clip_id, resumable)
Export <data>/gold_eval_verified.jsonl (accepted only → push as GOLD-eval-v1)
"""
import argparse, hashlib, json, os, sys
from pathlib import Path
import streamlit as st

# ---- args (after `--`) ----
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="./gold_eval")
ap.add_argument("--db", default="./database.sqlite",
                help="STTM database.sqlite — enables SGGS search + shabad browser")
args, _ = ap.parse_known_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
DATA = Path(args.data)
CAND = DATA / "gold_candidates.jsonl"
LABELS = DATA / "gold_labels.jsonl"
EXPORT = DATA / "gold_eval_verified.jsonl"
VAD_EXPORT = DATA / "vad_clips.jsonl"
LLM_SUGG = DATA / "llm_suggestions.jsonl"

st.set_page_config(page_title="Gurbani GOLD eval", layout="wide")

# ---- optional PIN gate (set GOLD_PIN env when exposing on a public host) ----
_PIN = os.environ.get("GOLD_PIN")
if _PIN:
    if "authed" not in st.session_state:
        st.session_state.authed = False
    if not st.session_state.authed:
        pin = st.text_input("PIN", type="password")
        if pin == _PIN:
            st.session_state.authed = True
            st.rerun()
        st.stop()

# ---- SGGS search: Latin first-letters → Gurmukhi lines (no Punjabi typing) ----
# Fuzzy map: each Latin key matches any plausible Gurmukhi word-initial letter.
LAT2GUR = {
    "a": "ਅਆਐਔ", "e": "ੲਇਈਏਐ", "i": "ੲਇਈ", "u": "ੳਉਊਓ", "o": "ੳਉਊਓਔ",
    "s": "ਸਸ਼", "h": "ਹ", "k": "ਕਖਖ਼", "g": "ਗਘਗ਼", "c": "ਚਛ",
    "j": "ਜਝਜ਼", "t": "ਤਟਥਠ", "d": "ਦਡਧਢ", "n": "ਨਣਙਞ", "p": "ਪਫ",
    "f": "ਫਫ਼", "b": "ਬਭ", "m": "ਮ", "y": "ਯ", "r": "ਰ", "l": "ਲਲ਼",
    "v": "ਵ", "w": "ਵ", "x": "ਣ", "z": "ਜ਼ਜ", "q": "ਤ",
}


@st.cache_resource
def load_sggs_lines(db_path: str):
    """(unicode, first_letters, line_id, shabad_id, ang) for all SGGS lines,
    plus shabad_id → [row, ...] for the shabad browser. None if DB missing."""
    p = Path(db_path)
    if not p.exists():
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from canonical.sttm_index import load_sggs
    lines, _ = load_sggs(p)
    rows, by_shabad = [], {}
    for ln in lines:
        fl = "".join(t[0] for t in ln.unicode.split() if t)
        row = {"unicode": ln.unicode, "fl": fl, "line_id": ln.line_id,
               "shabad_id": ln.shabad_id, "ang": ln.ang}
        rows.append(row)
        by_shabad.setdefault(ln.shabad_id, []).append(row)
    return rows, by_shabad


def fl_search(rows, query: str, limit: int = 20):
    """Match Latin first-letter query as a window over each line's Gurmukhi
    first-letters. Rank: line-start matches first, then shorter lines."""
    q = [ch for ch in query.lower() if ch.isalpha()]
    if not q:
        return []
    sets = [LAT2GUR.get(ch, "") for ch in q]
    hits = []
    for r in rows:
        fl, n = r["fl"], len(q)
        for i in range(len(fl) - n + 1):
            if all(fl[i + k] in sets[k] for k in range(n)):
                hits.append((i != 0, len(fl), r))
                break
    hits.sort(key=lambda h: (h[0], h[1]))
    return [r for _, _, r in hits[:limit]]


def use_line(key_i: int, row: dict):
    """Set the GOLD text + canonical ids from a clicked SGGS line.
    Must be called BEFORE the gold text_input is instantiated this run."""
    st.session_state[f"gold_{key_i}"] = row["unicode"]
    st.session_state[f"pickmeta_{key_i}"] = row


def append_line(key_i: int, row: dict):
    """Append a line's text to the GOLD box — for clips that span lines."""
    cur = (st.session_state.get(f"gold_{key_i}") or "").strip()
    st.session_state[f"gold_{key_i}"] = f"{cur} {row['unicode']}".strip()
    # keep canonical ids of the FIRST picked line
    st.session_state.setdefault(f"pickmeta_{key_i}", row)


@st.cache_data
def load_candidates():
    return [json.loads(l) for l in CAND.read_text(encoding="utf-8").splitlines() if l.strip()]


@st.cache_data
def load_llm_suggestions() -> dict:
    """clip_id → LLM pre-correction verdict (scripts/llm_precorrect_gold.py)."""
    if not LLM_SUGG.exists():
        return {}
    out = {}
    for l in LLM_SUGG.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l); out[r["clip_id"]] = r
    return out


def load_labels() -> dict:
    if not LABELS.exists():
        return {}
    out = {}
    for l in LABELS.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l); out[r["clip_id"]] = r
    return out


def save_label(rec: dict):
    """Append-only log; load_labels() keeps the last write per clip_id."""
    with LABELS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    st.session_state.labels[rec["clip_id"]] = rec


if not CAND.exists():
    st.error(f"No candidates at {CAND}. Run scripts/build_gold_eval_candidates.py first.")
    st.stop()

clips = load_candidates()
llm_sugg = load_llm_suggestions()
if "labels" not in st.session_state:
    st.session_state.labels = load_labels()

labels = st.session_state.labels
N = len(clips)


def spot_check(cid: str) -> bool:
    """Deterministic 10% audit sample of LLM-auto-handled clips."""
    return int(hashlib.md5(cid.encode()).hexdigest(), 16) % 10 == 0


def needs_ear(c: dict) -> bool:
    """True if a human must listen: no LLM verdict, LLM uncertain, or audit sample."""
    s = llm_sugg.get(c["clip_id"])
    if not s:
        return True
    if s["verdict"] in ("corrected", "unsure"):
        return True
    return spot_check(c["clip_id"])

# ---- sidebar: progress + export ----
with st.sidebar:
    st.header("Progress")
    n_acc = sum(1 for v in labels.values() if v.get("status") == "accept")
    n_rej = sum(1 for v in labels.values() if v.get("status") == "reject")
    n_skip = sum(1 for v in labels.values() if v.get("status") == "skip")
    n_mus = sum(1 for v in labels.values() if v.get("status") == "music")
    st.metric("Labeled", f"{len(labels)} / {N}")
    st.progress(len(labels) / N if N else 0)
    st.write(f"✅ accept **{n_acc}**  ·  🎵 music **{n_mus}**  ·  "
             f"❌ reject **{n_rej}**  ·  ⏭️ skip **{n_skip}**")
    from collections import Counter
    by_src = Counter((c["source"], "✓" if clips[k]["clip_id"] in labels else "·")
                     for k, c in enumerate(clips))
    st.caption("per-source done: " + "  ".join(
        f"{s}={sum(1 for c in clips if c['source']==s and c['clip_id'] in labels)}/"
        f"{sum(1 for c in clips if c['source']==s)}" for s in ["ragi","akj","sgpc","sehaj"]))
    _cur = st.session_state.get("i") or 0
    jump = st.number_input("Jump to #", 1, max(N, 1), _cur + 1)
    if jump - 1 != _cur and "i" in st.session_state:
        st.session_state.i = jump - 1; st.rerun()
    st.divider()
    if st.button("⬇️ Export verified (accepted)"):
        acc = [labels[c["clip_id"]] for c in clips
               if labels.get(c["clip_id"], {}).get("status") == "accept"]
        with EXPORT.open("w", encoding="utf-8") as f:
            for r in acc:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        st.success(f"Wrote {len(acc)} → {EXPORT}")
    if llm_sugg:
        def _bulk(verdict, status, label_btn):
            pend = [c for c in clips
                    if llm_sugg.get(c["clip_id"], {}).get("verdict") == verdict
                    and c["clip_id"] not in labels
                    and not spot_check(c["clip_id"])]   # spot-check 10% stays human
            if pend and st.button(f"{label_btn} {len(pend)} LLM-{verdict} clips"):
                for c in pend:
                    s = llm_sugg[c["clip_id"]]
                    save_label({
                        "clip_id": c["clip_id"], "source": c["source"], "quality": c["quality"],
                        "video_id": c["video_id"], "duration_s": c["duration_s"],
                        "audio_path": c["audio_path"], "proposed_text": c["proposed_text"],
                        "status": status, "gold_text": s.get("gold_text", ""),
                        "line_id": s.get("line_id"), "shabad_id": s.get("shabad_id"),
                        "ang": s.get("ang"), "via": f"llm_{verdict}_bulk",
                    })
                st.rerun()
        _bulk("exact", "accept", "⚡ Bulk-accept")
        _bulk("garbage", "reject", "🗑️ Bulk-reject")
        st.caption("10% of exact/garbage are held back for your ear (spot-check).")
    st.divider()
    queue_mode = st.radio("Queue", ["needs review (recommended)", "all clips"], index=0)
    if st.button("⬇️ Export VAD clips (vocal vs music)"):
        # accept = vocals present (1), music = instrumental only (0)
        out = []
        for cc in clips:
            stt = labels.get(cc["clip_id"], {}).get("status")
            if stt in ("accept", "music"):
                out.append({"clip_id": cc["clip_id"], "audio_path": cc["audio_path"],
                            "video_id": cc["video_id"], "duration_s": cc["duration_s"],
                            "vocal": 1 if stt == "accept" else 0})
        with VAD_EXPORT.open("w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        st.success(f"Wrote {len(out)} ({sum(r['vocal'] for r in out)} vocal / "
                   f"{sum(1 for r in out if not r['vocal'])} music) → {VAD_EXPORT}")

queue_idx = [k for k, cc in enumerate(clips)
             if queue_mode.startswith("all") or needs_ear(cc)]


def next_in_queue(after: int) -> int:
    for k in queue_idx:
        if k > after and clips[k]["clip_id"] not in labels:
            return k
    return N


def prev_in_queue(before: int) -> int:
    ks = [k for k in queue_idx if k < before]
    return ks[-1] if ks else before


if "i" not in st.session_state:
    st.session_state.i = next_in_queue(-1)
i = st.session_state.i
remaining = sum(1 for k in queue_idx if clips[k]["clip_id"] not in labels)
if i >= N or not remaining:
    st.balloons()
    st.success(f"Queue done ({queue_mode}). Bulk-handle the rest from the sidebar, "
               "then Export.")
    st.stop()
c = clips[i]
prev = labels.get(c["clip_id"], {})

st.subheader(f"Clip {i+1} / {N}  ·  {c['source']} / {c['quality']}"
             + (f"  ·  already: {prev.get('status')}" if prev else ""))
st.caption(f"🎧 needs-your-ear remaining: {remaining}")
st.caption(f"video {c['video_id']}  ·  {c['duration_s']}s  ·  skel_route={c.get('skel_route','')}")
audio_fp = DATA / c["audio_path"]
if audio_fp.exists():
    st.audio(str(audio_fp))
else:
    st.warning(f"missing audio {audio_fp}")

st.markdown("**Proposed (ASR/caption) label:**")
st.markdown(f"<div style='font-size:1.6rem'>{c['proposed_text']}</div>", unsafe_allow_html=True)

sugg = llm_sugg.get(c["clip_id"])
if sugg:
    icon = {"exact": "🟢", "corrected": "🟡", "unsure": "🟠", "garbage": "🔴"}.get(sugg["verdict"], "⚪")
    st.markdown(f"{icon} **LLM: {sugg['verdict']}**"
                + (f" → {sugg['gold_text']}" if sugg.get("gold_text") else ""))

# ---- candidate choice ----
opts, meta = [], []
for j, cand in enumerate(c["candidates"]):
    opts.append(f"cand{j}")
    meta.append(cand)
labels_disp = {f"cand{j}": f"{m['unicode']}   ·  ang {m['ang']} · {m['first_letters']} · lev {m['lev']}"
               for j, m in enumerate(meta)}
labels_disp["proposed"] = f"(use proposed as-is) {c['proposed_text']}"
labels_disp["custom"] = "(type my own below)"
choice = st.radio("Pick the correct SGGS line (or edit):",
                  options=list(labels_disp.keys()),
                  format_func=lambda k: labels_disp[k], index=0, key=f"choice_{i}")

# ---- no-typing edit help: SGGS search + shabad browser (sets gold below) ----
sggs = load_sggs_lines(args.db)
if sggs:
    rows, by_shabad = sggs
    with st.expander("🔎 Search SGGS — type LATIN first letters, no Punjabi keyboard "
                     "(e.g. `gms` for ਗੁਰ ਮੇਰੈ ਸੰਗਿ...)", expanded=not c["candidates"]):
        q = st.text_input("First letter of each word (a-z):", key=f"flq_{i}")
        for r in fl_search(rows, q):
            c1, c2, c3 = st.columns([8, 1, 1])
            c1.markdown(f"{r['unicode']}  · ang {r['ang']}")
            if c2.button("use", key=f"hit_{i}_{r['line_id']}"):
                use_line(i, r)
            if c3.button("➕", key=f"hitadd_{i}_{r['line_id']}"):
                append_line(i, r)
    # adjacent-line mistakes are common — browse the full shabad of any candidate
    browse_ids = {m["shabad_id"] for m in meta if m.get("shabad_id")}
    pick_prev = st.session_state.get(f"pickmeta_{i}")
    if pick_prev:
        browse_ids.add(pick_prev["shabad_id"])
    if browse_ids:
        with st.expander("📜 Browse full shabad(s) — `use` replaces, `+` appends (mix lines)"):
            for sid in browse_ids:
                shabad_rows = by_shabad.get(sid, [])
                # copyable plain-text block of the whole shabad
                st.code("\n".join(r["unicode"] for r in shabad_rows), language=None)
                for r in shabad_rows:
                    c1, c2, c3 = st.columns([8, 1, 1])
                    c1.markdown(f"{r['unicode']}  · ang {r['ang']}")
                    if c2.button("use", key=f"sh_{i}_{r['line_id']}"):
                        use_line(i, r)
                    if c3.button("➕", key=f"shadd_{i}_{r['line_id']}"):
                        append_line(i, r)
                st.divider()
else:
    st.caption(f"SGGS search off — no DB at {args.db} (pass --db to enable)")

default_text = c["proposed_text"]
if choice.startswith("cand"):
    default_text = meta[int(choice[-1])]["unicode"]
elif choice == "proposed":
    default_text = c["proposed_text"]
if sugg and sugg["verdict"] in ("exact", "corrected") and sugg.get("gold_text"):
    default_text = sugg["gold_text"]   # LLM pre-correction wins as the default
gold = st.text_input("GOLD label (edit to exactly match the audio):",
                     value=prev.get("gold_text", default_text), key=f"gold_{i}")

chosen = None
pick = st.session_state.get(f"pickmeta_{i}")
if pick and gold.strip() == pick["unicode"]:
    chosen = pick          # clicked from SGGS search / shabad browser
elif sugg and gold.strip() == (sugg.get("gold_text") or "") and sugg.get("line_id"):
    chosen = sugg          # LLM suggestion carries canonical ids
elif choice.startswith("cand"):
    chosen = meta[int(choice[-1])]

col1, col2, col3, col4, col5, col6 = st.columns(6)


def commit(status):
    save_label({
        "clip_id": c["clip_id"], "source": c["source"], "quality": c["quality"],
        "video_id": c["video_id"], "duration_s": c["duration_s"],
        "audio_path": c["audio_path"], "proposed_text": c["proposed_text"],
        "status": status, "gold_text": gold.strip(),
        "line_id": (chosen or {}).get("line_id"), "shabad_id": (chosen or {}).get("shabad_id"),
        "ang": (chosen or {}).get("ang"),
    })
    st.session_state.i = next_in_queue(i)


if col1.button("✅ Accept", use_container_width=True, type="primary"):
    commit("accept"); st.rerun()
if col2.button("🎵 Music only", use_container_width=True,
               help="No vocals / instrumental — banked as VAD training data"):
    commit("music"); st.rerun()
if col3.button("❌ Reject (garbage)", use_container_width=True):
    commit("reject"); st.rerun()
if col4.button("⏭️ Skip", use_container_width=True):
    commit("skip"); st.rerun()
if col5.button("◀ Prev", use_container_width=True):
    st.session_state.i = prev_in_queue(i); st.rerun()
if col6.button("Next ▶", use_container_width=True):
    nxt = next_in_queue(i)
    st.session_state.i = nxt if nxt < N else i
    st.rerun()

st.caption("Tip: Accept saves the GOLD label above. Music only = instrumental/no vocals "
           "(kept for VAD training). Reject = unusable audio/label. Skip = decide later. "
           "All autosave + resume.")
