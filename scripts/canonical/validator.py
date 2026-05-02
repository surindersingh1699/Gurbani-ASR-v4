"""Stage 2c: programmatic validator — the final gatekeeper.

This is a pure-code (no LLM) check that any proposed correction must
pass before we ship it. Same rules we ask the LLMs to follow, but
enforced deterministically here so we can never ship an output that
violates them.

A proposed correction passes validation iff ALL of:
  V1. Token count within ±1 of caption.
  V2. Every proposed word is:
        - present in the caption, OR
        - present in the retrieved shabad's vocabulary, OR
        - within skeleton-lev ≤ 1 of some shabad word.
  V3. Monotonic caption preservation: ≥ 60% of caption tokens appear
      in proposed output in the same relative order (skeleton-lev ≤ 1).
  V4. Length does not exceed caption length + 1 (redundant with V1
      but explicit — catches fragment completion).

On failure, returns the first reason(s) so we can log why it bounced.
"""
from __future__ import annotations

from .gurmukhi_skeleton import lev, skel, tokenize


def _word_is_valid(
    tok: str, cap_tokens: set[str], shabad_tokens: set[str],
    shabad_skels: set[str],
) -> bool:
    if tok in cap_tokens or tok in shabad_tokens:
        return True
    ts = skel(tok)
    if not ts:
        # Garbled orphan matra / punctuation-only token — pass through if
        # caption had it.
        return tok in cap_tokens
    if ts in shabad_skels:
        return True
    return any(lev(ts, ss) <= 1 for ss in shabad_skels)


def _skel_match(cs: str, ps: str) -> tuple[bool, bool]:
    """Return (matches, is_substring).
    Matches if cs/ps is same OR one is a substring of the other (merge/split
    case) OR same-length-skel with lev ≤ 1 (matra or equal-length 1-cons).
    Single-character skeletons require exact equality — lev=1 would match
    any other single-char skel (ਹ↔ਤ), which is spurious.
    The second return value flags substring matches so the caller can allow
    multiple cap tokens to map to one prop token (merge case)."""
    if not cs or not ps:
        return False, False
    if cs == ps:
        return True, False
    if cs in ps or ps in cs:
        return True, True
    if len(cs) == len(ps) and len(cs) >= 2 and lev(cs, ps) <= 1:
        return True, False
    return False, False


def _monotonic_preservation(
    cap_tokens: list[str], prop_tokens: list[str],
    shabad_skels: set[str], threshold: float = 0.60,
) -> tuple[bool, float]:
    """Fraction of caption tokens whose skeleton can be matched against a
    proposed token's skeleton in order. Substring matches (merge/split)
    are allowed to map multiple cap tokens to the same prop token."""
    if not cap_tokens:
        return True, 1.0
    cap_skels = [skel(t) for t in cap_tokens]
    prop_skels = [skel(t) for t in prop_tokens]
    matched = 0
    search_start = 0
    for cs in cap_skels:
        if not cs:
            matched += 1
            continue
        for j in range(search_start, len(prop_skels)):
            ok, is_substring = _skel_match(cs, prop_skels[j])
            if ok:
                matched += 1
                # For substring (merge) keep the same j — another cap
                # token might also fit here. For exact / equal-length
                # match, advance past j.
                search_start = j if is_substring else j + 1
                break
    ratio = matched / len(cap_tokens)
    return ratio >= threshold, ratio


def _positional_substitution_check(
    cap_tokens: list[str], prop_tokens: list[str],
) -> list[str]:
    """V5: when caption and proposed have equal token count, each
    substitution must be either a matra-only fix (same skeleton) or a
    1-consonant swap with skeleton-length differing by ≤ 1.

    Design note — the "≤ 1" length tolerance reflects the 'fairly usable,
    not 100% correct after API calls' bar. It lets legitimate SGGS
    substitutions through (e.g. ਪਹਰੀ→ਪੈਰੀ, ਸਲਾਹ→ਸਾਲਾਹਹ verb forms)
    while still blocking fragment completions where the skeleton grows
    by 2+ characters (ਭ→ਭਇਆ, ਲਾ→ਲਾਹਾਭ style)."""
    reasons: list[str] = []
    if len(cap_tokens) != len(prop_tokens):
        return reasons  # V1 already handles count mismatches; V5 skipped.
    for i, (c, p) in enumerate(zip(cap_tokens, prop_tokens)):
        if c == p:
            continue
        cs, ps = skel(c), skel(p)
        if cs == ps:
            continue  # matra-only fix (bindi now counts as matra too)
        if not cs or not ps:
            # one side is orphan-matra / garbled — caller should have
            # kept verbatim; flag if proposed has a non-garbled word
            # replacing a garbled one.
            if cs and not ps:
                reasons.append(f"V5 garbled_replacement@{i}({c!r}→{p!r})")
            elif not cs and ps:
                reasons.append(f"V5 filled_garbled@{i}({c!r}→{p!r})")
            continue
        length_delta = abs(len(cs) - len(ps))
        if length_delta > 1:
            reasons.append(
                f"V5 fragment_completion@{i}({c!r} skel={cs!r} → "
                f"{p!r} skel={ps!r}, len_delta={length_delta})"
            )
            continue
        if lev(cs, ps) > 1:
            reasons.append(
                f"V5 multi_cons_swap@{i}({c!r} → {p!r}, lev={lev(cs, ps)})"
            )
    return reasons


def validate(
    caption: str,
    proposed: str,
    shabad_tokens: set[str],
    max_drift: int = 1,
    monotonic_threshold: float = 0.60,
) -> tuple[bool, list[str]]:
    """Return (passes, reasons). `shabad_tokens` is the vocabulary of the
    retrieved shabad (Unicode tokens)."""
    reasons: list[str] = []

    if proposed.strip() == caption.strip():
        # Trivially safe: no-op correction.
        return True, []

    cap_tokens = tokenize(caption)
    prop_tokens = tokenize(proposed)

    # V1 + V4: token count drift
    drift = len(prop_tokens) - len(cap_tokens)
    if abs(drift) > max_drift:
        reasons.append(
            f"V1 token_count_drift({len(prop_tokens)} vs {len(cap_tokens)})"
        )

    # V2: invented words
    cap_tok_set = set(cap_tokens)
    shabad_skels = {skel(t) for t in shabad_tokens if skel(t)}
    invented: list[str] = []
    for tok in prop_tokens:
        if not _word_is_valid(tok, cap_tok_set, shabad_tokens, shabad_skels):
            invented.append(tok)
    if invented:
        reasons.append(f"V2 invented_word({', '.join(invented[:3])})")

    # V3: monotonic caption preservation
    mono_ok, ratio = _monotonic_preservation(
        cap_tokens, prop_tokens, shabad_skels, monotonic_threshold,
    )
    if not mono_ok:
        reasons.append(f"V3 monotonic_preservation({ratio:.2f})")

    # V5: positional substitution check (only when token counts match)
    reasons.extend(_positional_substitution_check(cap_tokens, prop_tokens))

    return len(reasons) == 0, reasons
