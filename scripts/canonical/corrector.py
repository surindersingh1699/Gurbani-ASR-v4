"""Stage 2a: LLM corrector (GPT-5 nano by default).

Takes a caption + retrieved shabad + optional STTM best-guess, returns a
corrected caption string constrained to scripture words from that shabad.

This is the 'creative' half of the two-LLM pipeline. The reviewer
(`reviewer.py`) then audits the output against a hard rubric.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass


CORRECTOR_SYSTEM_PROMPT = """You are a Gurbani expert cleaning captions for a Gurmukhi speech-recognition (ASR) training dataset.

Each caption is a transcription of an audio clip in which a kirtani (or sehaj-path reader) is reciting from the Guru Granth Sahib (SGGS). The audio is fixed — your job is to make the TEXT match what is actually being sung, using authentic SGGS spelling (correct akhar and matras), so the ASR model learns the right pangati.

The kirtani has full freedom: they may sing any line of the shabad, repeat a chorus many times, pause mid-line, or move between rahao and antras in any order. So:
  - DO correct misspelled akhar, missing/extra matras, and ASR letter-confusions to match the authentic SGGS wording of whichever line they are singing.
  - DO NOT add, remove, reorder, or de-duplicate words. Repetitions and unusual order are real performance choices, not errors.

RULES (strict, must follow ALL):
1. Preserve caption token count exactly. Do not add or remove words.
2. Preserve word order and any repetitions present in the caption.
3. Only replace words with words that appear in the SHABAD provided below.
4. Prefer matra-only fixes (same consonant skeleton, different vowel marks).
5. A 1-consonant swap is allowed only if the consonant skeletons have the
   same length (e.g. ਸਰਨ → ਸਰਣਿ is matra; ਛੇਤ → ਖੇਤ is a valid 1-cons swap).
6. Do NOT complete fragments into full scripture lines. Keep caption length.
7. If the caption is already correct or cannot be confidently corrected,
   return the caption verbatim.
8. If a caption word is garbled (orphan matras, broken letters, partial
   token), keep it verbatim — do not invent a replacement.
9. Output strict JSON only. No prose, no explanation.

EXAMPLE
SHABAD (excerpt):
ਮਿਤ੍ਰ ਪਿਆਰੇ ਨੂੰ ਹਾਲ ਮੁਰੀਦਾਂ ਦਾ ਕਹਿਣਾ ॥
ਤੁਧੁ ਬਿਨੁ ਰੋਗੁ ਰਜਾਈਆਂ ਦਾ ਓਢਣ ਨਾਗ ਨਿਵਾਸਾਂ ਦੇ ਰਹਿਣਾ ॥

CAPTION (to correct):
ਮਿਤਰ ਪਿਆਰੇ ਨੂੰ ਹਾਲ ਮੁਰੀਦਾ ਦਾ ਕਹਿਣਾ ਮਿਤਰ ਪਿਆਰੇ ਨੂੰ

CORRECTED (return):
ਮਿਤ੍ਰ ਪਿਆਰੇ ਨੂੰ ਹਾਲ ਮੁਰੀਦਾਂ ਦਾ ਕਹਿਣਾ ਮਿਤ੍ਰ ਪਿਆਰੇ ਨੂੰ

Notice: same token count, same order, repetition kept; only ਮਿਤਰ→ਮਿਤ੍ਰ (halant) and ਮੁਰੀਦਾ→ਮੁਰੀਦਾਂ (bindi) were fixed to match SGGS.

OUTPUT SHAPE: {"corrected": "<corrected caption text>"}
"""


@dataclass
class CorrectorConfig:
    model: str = "gpt-5-nano"
    # None = omit the kwarg (gpt-5-nano requires default temperature=1).
    # Set to 0.0 explicitly for gpt-4o-mini / older models that support it.
    temperature: float | None = None
    # gpt-5 nano/mini are reasoning models — they consume this budget on
    # internal thinking before producing output. For long shabads (10+
    # lines) the reasoner needs ~4-8k tokens before writing the answer.
    max_output_tokens: int = 16384
    # Captions per API call in batched mode. Shared shabad context is
    # amortized across the batch, so larger is cheaper — but at some point
    # the model runs out of attention / reasoning budget. 15 is a good
    # balance for gpt-5-nano on 20-line shabads.
    batch_size: int = 15
    # For reasoning models (gpt-5 family), sets how much internal
    # chain-of-thought the model emits before the final answer. Values:
    # "minimal" | "low" | "medium" | "high". Billed as output tokens, so
    # "minimal" is ~3-5× cheaper than default. For a caption-spelling task
    # with a structured JSON output + a reviewer safety net, "minimal"
    # is usually enough. Set to None to omit the kwarg (for non-reasoning
    # models like gpt-4o-mini).
    reasoning_effort: str | None = "minimal"


def build_corrector_prompt(
    caption: str, shabad_lines: list[str], sttm_hint: str | None = None,
) -> str:
    """Compose the user prompt. `shabad_lines` is a list of Unicode line
    strings from the retrieved shabad. `sttm_hint`, if provided, is the
    STTM pipeline's best guess — a strong signal to be adopted when it
    preserves caption structure."""
    shabad_text = "\n".join(shabad_lines)
    hint = (
        f"\n\nSTTM best-guess (apply when it preserves caption token count + "
        f"order; override only if it adds/drops words):\n{sttm_hint}"
        if sttm_hint else ""
    )
    return f"""SHABAD (retrieved for this caption):
{shabad_text}

CAPTION (to correct):
{caption}{hint}

Return JSON only: {{"corrected": "..."}}
"""


def correct(
    caption: str,
    shabad_lines: list[str],
    cfg: CorrectorConfig,
    sttm_hint: str | None = None,
    client=None,
) -> dict:
    """Call the corrector LLM. Returns {'corrected': str, 'model': str,
    'ok': bool, 'error': str | None, 'raw': str | None}."""
    if client is None:
        from openai import OpenAI
        api_key = os.environ.get("OPEN_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return {
                "corrected": caption, "model": cfg.model, "ok": False,
                "error": "missing OPEN_API_KEY / OPENAI_API_KEY", "raw": None,
            }
        client = OpenAI(api_key=api_key)

    prompt = build_corrector_prompt(caption, shabad_lines, sttm_hint)
    kwargs = dict(
        model=cfg.model,
        messages=[
            {"role": "system", "content": CORRECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=cfg.max_output_tokens,
        response_format={"type": "json_object"},
    )
    if cfg.temperature is not None:
        kwargs["temperature"] = cfg.temperature
    if cfg.reasoning_effort is not None:
        kwargs["reasoning_effort"] = cfg.reasoning_effort
    try:
        resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""
        finish = resp.choices[0].finish_reason
        if not raw:
            # Reasoning model consumed all output tokens on internal
            # thinking. Surface finish_reason so we know to bump the budget.
            return {
                "corrected": caption, "model": cfg.model, "ok": False,
                "error": f"empty_output(finish={finish})", "raw": None,
            }
        data = json.loads(raw)
        corrected = data.get("corrected", caption)
        return {
            "corrected": corrected, "model": cfg.model, "ok": True,
            "error": None, "raw": raw,
        }
    except Exception as e:
        return {
            "corrected": caption, "model": cfg.model, "ok": False,
            "error": f"{type(e).__name__}: {e}", "raw": None,
        }


def build_batch_corrector_prompt(
    items: list[dict], shabad_lines: list[str],
) -> str:
    """items: [{clip_id, caption, sttm_hint?}, ...]

    Batched corrector prompt: stresses independent per-item reasoning to
    avoid the model anchoring on earlier items' corrections (observed
    multi-script contamination at batch_size=15 without this wording).
    """
    shabad_text = "\n".join(shabad_lines)
    payload = []
    for it in items:
        entry = f'  {{"clip_id": "{it["clip_id"]}", "caption": "{it["caption"]}"'
        if it.get("sttm_hint"):
            entry += f', "sttm_hint": "{it["sttm_hint"]}"'
        entry += "}"
        payload.append(entry)
    items_json = "[\n" + ",\n".join(payload) + "\n]"
    return f"""SHABAD (retrieved for all captions):
{shabad_text}

CAPTIONS ({len(items)} items). Process EACH CAPTION INDEPENDENTLY — your
correction for one caption must not be influenced by another. Treat this
as {len(items)} separate corrections that happen to share a shabad.

For each caption:
  1. Keep all characters in Gurmukhi script only (no Devanagari ा/ै/े,
     no Kannada ಾ, no other Indic scripts leaking in).
  2. Preserve caption token count and order exactly.
  3. Apply the same correction rules from the system prompt.
  4. The sttm_hint (when present) is a STRONG signal if it preserves
     caption token count + order; override only if it adds/drops words.

Preserve each clip_id exactly. The corrections array MUST have one entry
per input clip_id — no omissions.

{items_json}

Return strict JSON: {{"corrections": [{{"clip_id": "...", "corrected": "..."}}, ...]}}
"""


def _correct_single_call(
    items: list[dict], shabad_lines: list[str], cfg: CorrectorConfig, client,
) -> tuple[bool, str | None, dict[str, dict]]:
    """One API call with `items` as a batch. Returns (ok, error, results).
    On ok=True the results dict is complete; on ok=False caller decides
    whether to retry with a smaller batch or give up."""
    prompt = build_batch_corrector_prompt(items, shabad_lines)
    kwargs = dict(
        model=cfg.model,
        messages=[
            {"role": "system", "content": CORRECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=cfg.max_output_tokens,
        response_format={"type": "json_object"},
    )
    if cfg.temperature is not None:
        kwargs["temperature"] = cfg.temperature
    if cfg.reasoning_effort is not None:
        kwargs["reasoning_effort"] = cfg.reasoning_effort
    try:
        resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""
        if not raw:
            return False, f"empty(finish={resp.choices[0].finish_reason})", {}
        data = json.loads(raw)
        corrections = data.get("corrections", [])
        by_id = {c.get("clip_id"): c.get("corrected") for c in corrections}
        results: dict[str, dict] = {}
        for it in items:
            cid = it["clip_id"]
            if cid in by_id and by_id[cid]:
                results[cid] = {
                    "corrected": by_id[cid], "model": cfg.model,
                    "ok": True, "error": None, "raw": raw,
                }
            else:
                results[cid] = {
                    "corrected": it["caption"], "model": cfg.model,
                    "ok": False, "error": "clip_id missing", "raw": raw,
                }
        return True, None, results
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", {}


def correct_batch(
    items: list[dict],
    shabad_lines: list[str],
    cfg: CorrectorConfig,
    client=None,
) -> dict[str, dict]:
    """Batched corrector with split-on-failure retry. Returns
    {clip_id: {corrected, model, ok, error}}.

    On batch failure (empty output, JSON parse, network), the batch is
    split in halves and each half is retried. At batch_size=1 and still
    failing, the row falls back to caption verbatim with ok=False so the
    driver's candidate-ranking can decide what to ship.
    """
    if client is None:
        from openai import OpenAI
        api_key = os.environ.get("OPEN_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return {
                it["clip_id"]: {
                    "corrected": it["caption"], "model": cfg.model,
                    "ok": False, "error": "missing OPEN_API_KEY", "raw": None,
                } for it in items
            }
        client = OpenAI(api_key=api_key)

    def _process(chunk: list[dict]) -> dict[str, dict]:
        if not chunk:
            return {}
        ok, err, res = _correct_single_call(chunk, shabad_lines, cfg, client)
        if ok:
            return res
        # Retry with smaller batches. Singleton = give up.
        if len(chunk) == 1:
            it = chunk[0]
            return {
                it["clip_id"]: {
                    "corrected": it["caption"], "model": cfg.model,
                    "ok": False, "error": f"singleton_failed: {err}",
                    "raw": None,
                },
            }
        mid = len(chunk) // 2
        return {**_process(chunk[:mid]), **_process(chunk[mid:])}

    results: dict[str, dict] = {}
    for chunk_start in range(0, len(items), cfg.batch_size):
        chunk = items[chunk_start:chunk_start + cfg.batch_size]
        results.update(_process(chunk))
    return results
