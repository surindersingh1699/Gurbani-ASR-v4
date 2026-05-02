"""Stage 2b: reviewer — a senior Gurbani ML expert who both audits the
corrector's output AND applies its own deeper Gurmukhi corrections as a
second layer.

Unlike a pure audit step, the reviewer ALWAYS produces `final_text`:
  - APPROVE  → final_text = corrector_output (no further changes needed)
  - IMPROVED → final_text = reviewer's version, applying fixes the
               corrector missed (halant ੍, nukta ਼, bindi/tippi, compound
               letters like ੑ, etc.)
  - REJECT   → final_text = caption (neither corrector nor reviewer can
               produce a safe correction)

A programmatic `validator.py` check runs on final_text before it ships
— the reviewer's output is trusted but verified.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass


REVIEWER_SYSTEM_PROMPT = """You are a senior Gurbani ML expert reviewing a junior LLM's spelling correction of a Gurmukhi caption from an audio recitation of the Guru Granth Sahib (SGGS).

You are given ONLY three things: the retrieved SHABAD, the original CAPTION, and the junior LLM's CORRECTOR OUTPUT. You have no access to the STTM pipeline's output — your judgement should be an independent second opinion based solely on the caption and scripture.

Your job is twofold:
  1. Judge the junior's output against strict Gurmukhi rules.
  2. When the junior missed a fix, apply your own deeper Gurmukhi knowledge
     as a second corrective layer — halant ੍, nukta ਼, bindi vs tippi,
     compound letters like ੰ/ੱ/ੑ, authentic SGGS spelling of repeated
     chorus words, and so on.

CONTEXT
The audio is FIXED. Your corrections must match what was actually sung
— do NOT add, remove, reorder, or de-duplicate words. The kirtani may
repeat a chorus many times or sing lines in unusual order; those are
real performance choices, not errors.

HARD RULES (every output must satisfy)
R1. Token count: output token count must equal caption token count ±1.
R2. Monotonic preservation: ≥ 60% of caption tokens must appear in the
    output in the same relative order (skeleton edit-distance ≤ 1).
R3. Scripture-only vocabulary: every output word must appear in the
    provided SHABAD lines (or be a garbled caption token kept verbatim).
R4. Equal-length 1-cons swaps only. Matra-only fixes preferred.
R5. No reorder-snap (2+ consecutive different-skeleton swaps at the
    start or end of the caption).
R6. No fragment completion: if the caption is a short fragment, do NOT
    complete it into a full scripture line.
R7. Garbled tokens (orphan matras, broken letters): keep verbatim.

VERDICT SEMANTICS
APPROVE  — corrector_output satisfies every rule; final_text = corrector_output.
IMPROVED — corrector_output is mostly right but you see at least one
           missed/incorrect fix that YOU can make minimally.
           final_text = your version (corrector_output with your fixes).
           You must still satisfy every hard rule in final_text.
REJECT   — neither corrector_output nor any minimal revision you can
           produce satisfies the hard rules. final_text = caption unchanged.

OUTPUT SHAPE (strict JSON, no prose outside):
{
  "verdict": "APPROVE" | "IMPROVED" | "REJECT",
  "final_text": "<gurmukhi caption you would ship>",
  "changed_tokens": [{"from": "<cap_word>", "to": "<new_word>", "reason": "halant | matra | nukta | 1cons | other"}],
  "reasons": ["<short explanation>", ...]
}

On IMPROVED, `changed_tokens` lists what you CHANGED vs corrector_output
(empty on APPROVE). On REJECT, changed_tokens = [] and final_text = caption.
"""


@dataclass
class ReviewerConfig:
    model: str = "gemini-3-flash-lite"
    temperature: float = 0.0
    max_output_tokens: int = 2048
    fallback_models: tuple[str, ...] = ("gemini-2.5-flash-lite",)
    # Items per API call in batched mode. Smaller than corrector's default
    # because each reviewer row carries both caption + corrector_output,
    # and reviewing requires per-item attention. 5 is the tested sweet
    # spot where IMPROVED verdicts are recovered vs 10.
    batch_size: int = 5
    # Per-item output cap, multiplied by batch size for batched calls.
    # Bumped to accommodate the `reasoning` CoT field.
    per_item_max_output_tokens: int = 1024


def build_reviewer_prompt(
    caption: str,
    corrector_output: str,
    shabad_lines: list[str],
) -> str:
    """Reviewer sees ONLY: shabad, caption, corrector_output. No STTM hint
    — we want the reviewer's judgement to be independent of the STTM-
    based pipeline so it acts as a true second opinion."""
    shabad_text = "\n".join(shabad_lines)
    return f"""SHABAD (retrieved):
{shabad_text}

CAPTION (original):
{caption}

CORRECTOR OUTPUT (junior LLM's attempt — audit + improve this):
{corrector_output}

Return strict JSON with fields: verdict, final_text, changed_tokens, reasons.
"""


def _call_gemini(client, model: str, cfg: ReviewerConfig, prompt: str) -> dict:
    from google.genai import types
    resp = client.models.generate_content(
        model=model,
        contents=[
            {"role": "user",
             "parts": [{"text": REVIEWER_SYSTEM_PROMPT + "\n\n" + prompt}]},
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
        ),
    )
    return json.loads(resp.text or "{}")


def review(
    caption: str,
    corrector_output: str,
    shabad_lines: list[str],
    cfg: ReviewerConfig,
    client=None,
) -> dict:
    """Call the reviewer LLM. Returns a dict:
        {verdict, final_text, changed_tokens, reasons, model, ok, error}
    On any error, returns REJECT with final_text = caption."""
    if client is None:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return _reject_fallback(caption, cfg.model, "missing GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)

    prompt = build_reviewer_prompt(caption, corrector_output, shabad_lines)
    models_to_try = (cfg.model,) + tuple(cfg.fallback_models)
    last_err: str | None = None
    for m in models_to_try:
        try:
            data = _call_gemini(client, m, cfg, prompt)
            verdict = data.get("verdict", "REJECT").upper()
            if verdict not in ("APPROVE", "IMPROVED", "REJECT"):
                verdict = "REJECT"
            final_text = data.get("final_text") or caption
            changed = data.get("changed_tokens") or []
            reasons = data.get("reasons") or []
            # Safety coercions
            if verdict == "APPROVE":
                final_text = corrector_output
                changed = []
            elif verdict == "REJECT":
                final_text = caption
                changed = []
            return {
                "verdict": verdict, "final_text": final_text,
                "changed_tokens": changed, "reasons": reasons,
                "model": m, "ok": True, "error": None,
            }
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            continue

    return _reject_fallback(caption, cfg.model, f"all_models_failed: {last_err}")


def _reject_fallback(caption: str, model: str, err: str) -> dict:
    return {
        "verdict": "REJECT", "final_text": caption,
        "changed_tokens": [], "reasons": [err],
        "model": model, "ok": False, "error": err,
    }


def build_batch_reviewer_prompt(
    items: list[dict], shabad_lines: list[str],
) -> str:
    """items: [{clip_id, caption, corrector_output}, ...]

    Prompt design notes for batched reviewer:
      - Explicit 'treat each item INDEPENDENTLY' instruction. Empirically
        batched reviewers drop the IMPROVED rate because they anchor on
        prior items; this reminds the model to reset attention per item.
      - Per-item `reasoning` chain-of-thought field forces the model to
        work through R1-R7 for each caption before picking a verdict,
        which recovers IMPROVED verdicts in the batched setting.
    """
    shabad_text = "\n".join(shabad_lines)
    payload = []
    for it in items:
        payload.append(
            f'  {{"clip_id": "{it["clip_id"]}", '
            f'"caption": "{it["caption"]}", '
            f'"corrector_output": "{it["corrector_output"]}"}}'
        )
    items_json = "[\n" + ",\n".join(payload) + "\n]"
    return f"""SHABAD (retrieved for all items):
{shabad_text}

ITEMS TO REVIEW ({len(items)}). Process EACH ITEM INDEPENDENTLY — your
judgement for one item must not be influenced by another. Treat this as
{len(items)} separate reviews that happen to share a shabad. For each
item:
  1. Walk through rules R1-R7 against the caption + corrector_output.
  2. Decide verdict: APPROVE | IMPROVED | REJECT.
  3. If IMPROVED, produce revised final_text applying your Gurmukhi
     expertise (halant, nukta, bindi, compound letters, matra fixes).

Preserve clip_ids exactly. The reviews array MUST have one entry per
input clip_id — no omissions.

{items_json}

Return strict JSON:
{{
  "reviews": [
    {{
      "clip_id": "...",
      "reasoning": "R1: token count check... R2: preservation... R3: vocab check... R4: skel-length... conclusion: <verdict> because <why>",
      "verdict": "APPROVE|IMPROVED|REJECT",
      "final_text": "...",
      "changed_tokens": [{{"from": "...", "to": "...", "reason": "matra|nukta|halant|1cons|other"}}],
      "reasons": ["short rule-cite summary", ...]
    }},
    ...
  ]
}}
"""


def _review_single_call(
    items: list[dict], shabad_lines: list[str], cfg: ReviewerConfig, client,
) -> tuple[bool, str | None, dict[str, dict]]:
    """One reviewer API call with fallback across model ids. Returns
    (ok, error, results)."""
    prompt = build_batch_reviewer_prompt(items, shabad_lines)
    max_tokens = cfg.per_item_max_output_tokens * len(items) + 256
    last_err: str | None = None
    for m in (cfg.model,) + tuple(cfg.fallback_models):
        try:
            from google.genai import types
            resp = client.models.generate_content(
                model=m,
                contents=[{"role": "user", "parts": [
                    {"text": REVIEWER_SYSTEM_PROMPT + "\n\n" + prompt},
                ]}],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=cfg.temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            data = json.loads(resp.text or "{}")
            reviews = data.get("reviews", [])
            by_id = {r.get("clip_id"): r for r in reviews}
            results: dict[str, dict] = {}
            for it in items:
                cid = it["clip_id"]
                rv = by_id.get(cid)
                if not rv:
                    results[cid] = _reject_fallback(
                        it["caption"], m, "clip_id missing from response",
                    )
                    continue
                verdict = (rv.get("verdict") or "REJECT").upper()
                if verdict not in ("APPROVE", "IMPROVED", "REJECT"):
                    verdict = "REJECT"
                final_text = rv.get("final_text") or it["caption"]
                changed = rv.get("changed_tokens") or []
                reasons = rv.get("reasons") or []
                if verdict == "APPROVE":
                    final_text = it["corrector_output"]
                    changed = []
                elif verdict == "REJECT":
                    final_text = it["caption"]
                    changed = []
                results[cid] = {
                    "verdict": verdict, "final_text": final_text,
                    "changed_tokens": changed, "reasons": reasons,
                    "model": m, "ok": True, "error": None,
                }
            return True, None, results
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            continue
    return False, last_err, {}


def review_batch(
    items: list[dict],
    shabad_lines: list[str],
    cfg: ReviewerConfig,
    client=None,
) -> dict[str, dict]:
    """Batched reviewer with split-on-failure retry (same strategy as
    `corrector.correct_batch`). Returns {clip_id: {verdict, final_text,
    ...}}."""
    if client is None:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {
                it["clip_id"]: _reject_fallback(
                    it["caption"], cfg.model, "missing GEMINI_API_KEY",
                ) for it in items
            }
        client = genai.Client(api_key=api_key)

    def _process(chunk: list[dict]) -> dict[str, dict]:
        if not chunk:
            return {}
        ok, err, res = _review_single_call(chunk, shabad_lines, cfg, client)
        if ok:
            return res
        if len(chunk) == 1:
            it = chunk[0]
            return {
                it["clip_id"]: _reject_fallback(
                    it["caption"], cfg.model, f"singleton_failed: {err}",
                ),
            }
        mid = len(chunk) // 2
        return {**_process(chunk[:mid]), **_process(chunk[mid:])}

    results: dict[str, dict] = {}
    for chunk_start in range(0, len(items), cfg.batch_size):
        chunk = items[chunk_start:chunk_start + cfg.batch_size]
        results.update(_process(chunk))
    return results
