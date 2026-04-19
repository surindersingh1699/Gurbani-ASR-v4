"""Gemini 3.1 Pro LLM fallback pass. Runs ONLY on rows with decision
IN (unchanged, review). Emits final_text_llm, llm_model, llm_verified.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass

from .gurmukhi_skeleton import lev, skel, tokenize


SYSTEM_PROMPT = """You correct kirtan/sehaj transcriptions to match Guru Granth Sahib (SGGS) verbatim.

RULES (strict, must follow ALL):
1. You MUST return EXACTLY ONE corrections entry per input clip_id. No omissions. \
The output's corrections array MUST have the same length as the input captions array.
2. If the caption is already correct or cannot be confidently corrected, still \
include it in the output — copy the caption verbatim as "corrected".
3. When correcting, preserve word order, repetitions, and token count (± 1 max).
4. You may ONLY use words that appear in the SGGS shabad provided below, with \
correct matras. Do not invent words.
5. Do NOT add lines, repetitions, or words not present in the caption.
6. Do NOT drop trailing tokens from the caption.
7. Output strictly as JSON. Preserve clip_ids exactly.
"""


@dataclass
class LLMConfig:
    model: str = "gemini-3.1-pro-preview"
    batch_size: int = 30
    temperature: float = 0.0
    max_output_tokens: int = 12000
    max_len_drift: int = 2
    max_skel_lev_for_valid: int = 1


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


def verify_llm_output(
    llm_text: str, shabad_tokens: set[str], shabad_skels: set[str],
    caption_len: int, max_drift: int = 2, max_skel_lev: int = 1,
) -> tuple[bool, str]:
    llm_tokens = [t for t in llm_text.split() if t and t != ">>"]
    if abs(len(llm_tokens) - caption_len) > max_drift:
        return False, f"len_drift({len(llm_tokens)}vs{caption_len})"
    for tok in llm_tokens:
        if tok in shabad_tokens:
            continue
        ts = skel(tok)
        if not ts:
            continue
        if any(lev(ts, ss) <= max_skel_lev for ss in shabad_skels):
            continue
        return False, f"invented({tok})"
    return True, "ok"


def _call_gemini(client, cfg: LLMConfig, prompt: str) -> list[dict]:
    from google.genai import types  # local import so tests can mock module
    resp = client.models.generate_content(
        model=cfg.model, contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
        ),
    )
    data = json.loads(resp.text)
    return data.get("corrections", []), resp


def run_llm_pass(
    candidates: list[dict], shabad_lines_map: dict, cfg: LLMConfig,
    client=None,
) -> dict[str, dict]:
    """Process candidate rows (dicts with clip_id, shabad_id, text/caption).
    Groups by shabad_id, batches cfg.batch_size rows per Gemini call.
    Returns {clip_id -> {corrected, verified, reason, model}}."""
    if client is None:
        import os
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # Dedup by caption (saves API cost when identical captions repeat)
    by_caption: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        key = c.get("text", "") or c.get("caption", "")
        by_caption[key].append(c)
    unique_candidates = [v[0] for v in by_caption.values()]

    by_shabad: dict[str, list[dict]] = defaultdict(list)
    for c in unique_candidates:
        if c.get("shabad_id"):
            # shape expected by build_prompt: {clip_id, caption}
            by_shabad[c["shabad_id"]].append(
                {"clip_id": c["clip_id"], "caption": c.get("text") or c.get("caption")}
            )

    results: dict[str, dict] = {}
    for sid, group in by_shabad.items():
        shabad_lines = shabad_lines_map[sid]
        shabad_text = "\n".join(ln.unicode for ln in shabad_lines)
        ang = shabad_lines[0].ang if shabad_lines else 0
        shabad_tokens = {t for ln in shabad_lines for t in ln.tokens}
        shabad_skels = {skel(t) for t in shabad_tokens if skel(t)}

        for i in range(0, len(group), cfg.batch_size):
            batch = group[i:i + cfg.batch_size]
            prompt = build_prompt(shabad_text, ang, batch)
            try:
                corrections, resp = _call_gemini(client, cfg, prompt)
            except Exception as e:
                # partial failure: record each row as not-verified
                for r in batch:
                    results[r["clip_id"]] = {
                        "corrected": r["caption"], "verified": False,
                        "reason": f"api_error({type(e).__name__})",
                        "model": cfg.model,
                    }
                continue
            cid_to_caplen = {
                r["clip_id"]: len(tokenize(r["caption"])) for r in batch
            }
            returned_ids = set()
            for corr in corrections:
                cid = corr.get("clip_id", "")
                returned_ids.add(cid)
                text = corr.get("corrected", "")
                ok, reason = verify_llm_output(
                    text, shabad_tokens, shabad_skels,
                    cid_to_caplen.get(cid, 0),
                    cfg.max_len_drift, cfg.max_skel_lev_for_valid,
                )
                results[cid] = {"corrected": text, "verified": ok,
                                "reason": reason, "model": cfg.model}
            # Fill any omitted clip_ids (shouldn't happen with correct prompt,
            # but defense in depth)
            for r in batch:
                if r["clip_id"] not in returned_ids:
                    results[r["clip_id"]] = {
                        "corrected": r["caption"], "verified": False,
                        "reason": "llm_omitted", "model": cfg.model,
                    }

    # Broadcast dedup results to all candidates
    final: dict[str, dict] = {}
    for c in candidates:
        key = c.get("text", "") or c.get("caption", "")
        canon = by_caption[key][0]
        if canon["clip_id"] in results:
            final[c["clip_id"]] = results[canon["clip_id"]]
    return final
