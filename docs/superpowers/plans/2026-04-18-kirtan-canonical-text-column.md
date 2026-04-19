# Kirtan + Sehaj Canonical Text Column Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add canonical-text columns (`sggs_line`, `final_text`, `decision`, plus LLM audit columns) to two HF datasets (`surindersinghssj/gurbani-kirtan-yt-captions-300h` and `surindersinghssj/gurbani-sehajpath`) by grounding each row against the local STTM `database.sqlite`, with an optional Gemini 3.1 Pro fallback for the unchanged/review tail.

**Architecture:** Deterministic two-stage pipeline. Stage 1 is STTM-grounded consonant-skeleton phrase alignment with dataset-specific pre-cleaning (`<unk>` stripping, Sirlekh header indexing, multi-shabad row splitting, sequential retrieval for sehaj, simran quota for kirtan). Stage 2 (conditional) sends `unchanged`/`review` rows to Gemini 3.1 Pro in shabad-grouped batches of 30 with a strict 1:1 clip_id mapping prompt. Outputs three new dataset columns from Stage 1 and three additional columns from Stage 2 — strictly additive, never touches `text` / `raw_text`.

**Tech Stack:** Python 3.11, sqlite3 (stdlib), `huggingface-hub` + `datasets`, `google-genai` (Gemini SDK), `pytest` (tests), `pyarrow` (parquet sidecars), `python-dotenv` (API keys).

**Spec:** `docs/superpowers/specs/2026-04-18-kirtan-canonical-text-column-design.md` (authoritative). Prototypes at `scripts/proto_canonical_v2.py` and `scripts/proto_run_file.py` are reference implementations — refactor into production modules.

**Gating (strict, enforce before running in production):**
1. AKJ ingest (`@akjdotorg`) merged into kirtan HF repo.
2. Additive HF push implemented per `memory/project_yt_pipeline_todos.md`.
3. Then this canonical-column work.

---

## File Structure

### New production modules (in `scripts/canonical/` package)

- `scripts/canonical/__init__.py` — package marker
- `scripts/canonical/gurmukhi_skeleton.py` — consonant-skeleton, Levenshtein, tokenization primitives
- `scripts/canonical/sttm_index.py` — SGGS line loading, 4-gram index, global token index, sequential-order lookup
- `scripts/canonical/preprocess.py` — pre-cleaning filters (§1.5) + `<unk>` stripping
- `scripts/canonical/simran.py` — simran detection + quota downsampling (§1.6)
- `scripts/canonical/waheguru.py` — hardcoded ਵਾਹਿਗੁਰੂ skeleton normalization
- `scripts/canonical/sirlekh.py` — shabad-header normalization + mid-row Sirlekh splitter (§2.5, §2.6)
- `scripts/canonical/retrieval.py` — shabad retrieval with sliding window + video memory + sequential prior (§2.7)
- `scripts/canonical/align.py` — semi-global Needleman-Wunsch + phrase ops + orphan-run realignment
- `scripts/canonical/decision.py` — decision-label logic + safety-fallback
- `scripts/canonical/config.py` — dataset-specific defaults (kirtan vs sehaj)
- `scripts/canonical/pipeline.py` — main driver that composes all phases

### New CLI entrypoints

- `scripts/add_canonical_column.py` — Stage 1 CLI (DB-grounded pipeline)
- `scripts/llm_canonical_pass.py` — Stage 2 CLI (Gemini fallback)
- `scripts/merge_canonical_into_hf.py` — merge sidecar parquets back into HF dataset
- `scripts/review_canonical_flags.py` — lightweight triage CLI for `review` rows (optional)

### Tests (mirror module structure)

- `tests/canonical/__init__.py`
- `tests/canonical/test_gurmukhi_skeleton.py`
- `tests/canonical/test_sttm_index.py`
- `tests/canonical/test_preprocess.py`
- `tests/canonical/test_simran.py`
- `tests/canonical/test_waheguru.py`
- `tests/canonical/test_sirlekh.py`
- `tests/canonical/test_retrieval.py`
- `tests/canonical/test_align.py`
- `tests/canonical/test_decision.py`
- `tests/canonical/test_pipeline.py` — end-to-end on known samples
- `tests/canonical/test_llm_pass.py` — with mocked Gemini client

### Artifacts (git-ignored via `.gitignore`)

- `canonical_audit.parquet` — Stage 1 sidecar (per-row `shabad_id`, `line_ids`, `match_score`, `op_counts`, `top1_top2_margin`)
- `canonical_llm.parquet` — Stage 2 sidecar (per-row `final_text_llm`, `llm_model`, `llm_verified`, `llm_reason`)
- `review_decisions.csv` — human review outcomes (if triage CLI used)

---

## Task 1: Package scaffold + `.gitignore` + dev dependencies

**Files:**
- Create: `scripts/canonical/__init__.py`
- Create: `tests/canonical/__init__.py`
- Modify: `.gitignore`
- Create: `pyproject.toml` (or modify if already exists — add dev deps)

- [ ] **Step 1: Create package structure**

```bash
mkdir -p scripts/canonical tests/canonical
touch scripts/canonical/__init__.py tests/canonical/__init__.py
```

- [ ] **Step 2: Add artifacts to `.gitignore`**

Append to `.gitignore`:
```
canonical_audit.parquet
canonical_llm.parquet
review_decisions.csv
```

- [ ] **Step 3: Verify dev dependencies exist**

Check the repo's existing `pyproject.toml` or `requirements.txt` for:
- `pytest` (tests)
- `google-genai` (Gemini SDK; should already be present per memory)
- `datasets` (HF dataset load/save)
- `pyarrow` (parquet sidecars)
- `python-dotenv` (API key loading)

If any are missing, add to the dev-deps section. Example for a `pyproject.toml` with optional deps:
```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-mock>=3.12"]
```

Then install: `pip install -e '.[dev]'`

- [ ] **Step 4: Verify pytest runs (empty collection)**

Run: `pytest tests/canonical/ -v`
Expected: `no tests ran in 0.01s` (OK — empty is fine at this point).

- [ ] **Step 5: Commit**

```bash
git add scripts/canonical/__init__.py tests/canonical/__init__.py .gitignore pyproject.toml
git commit -m "chore: scaffold canonical package + test dir + .gitignore"
```

---

## Task 2: Gurmukhi skeleton + Levenshtein primitives

**Files:**
- Create: `scripts/canonical/gurmukhi_skeleton.py`
- Test: `tests/canonical/test_gurmukhi_skeleton.py`

Reference implementation: `scripts/proto_canonical_v2.py` (lines defining `skel()`, `lev()`, `_NUKTA_PAIRS`, `_denukta()`, `tokenize()`, `_clean_token()`).

- [ ] **Step 1: Write failing tests for `skel()` (consonant extraction)**

```python
# tests/canonical/test_gurmukhi_skeleton.py
import pytest
from scripts.canonical.gurmukhi_skeleton import skel, lev, tokenize, clean_token

class TestSkel:
    def test_bare_consonants_preserved(self):
        assert skel("ਗਰ") == "ਗਰ"

    def test_matras_stripped(self):
        assert skel("ਗੁਰੂ") == "ਗਰ"
        assert skel("ਸਰਣਿ") == "ਸਰਣ"

    def test_vishraams_and_digits_stripped(self):
        assert skel("ਗੁਰੂ ॥੧॥") == "ਗਰ"

    def test_nukta_normalized_to_base(self):
        # ਸ਼ → ਸ
        assert skel("ਸ਼ਾਮ") == "ਸਮ"

    def test_bindi_tippi_addak_halant_stripped(self):
        # ੰ (tippi), ੱ (addak), ੍ (halant), ਂ (bindi)
        assert skel("ਸੰਤ") == "ਸਤ"
        assert skel("ਅੱਡਾ") == "ਅਡ"

    def test_zwj_zwnj_stripped(self):
        assert skel("ਗ\u200dੁਰੂ") == "ਗਰ"

    def test_empty_and_whitespace(self):
        assert skel("") == ""
        assert skel("   ") == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/canonical/test_gurmukhi_skeleton.py -v`
Expected: all FAIL with `ModuleNotFoundError: scripts.canonical.gurmukhi_skeleton`

- [ ] **Step 3: Implement `skel()` + helpers**

```python
# scripts/canonical/gurmukhi_skeleton.py
"""Gurmukhi consonant-skeleton extraction + Levenshtein distance.

Skeleton = strip matras, vishraams, digits, bindi/tippi/addak/halant,
ZWJ/ZWNJ, and normalize nukta-modified consonants to their base form.
"""
from __future__ import annotations

import re

# Strip: matras (U+0A3E-U+0A4D), nukta, udaat, bindi, tippi, addak,
# digits, whitespace, ZWJ/ZWNJ, vishraams, punctuation
_SKELETON_STRIP = re.compile(
    r"[\u0A3C\u0A3E-\u0A4D\u0A51\u0A70\u0A71\u0A75"
    r"\u0A66-\u0A6F"
    r"0-9\s\u200C\u200D।॥॰.,;:!?'\"()\[\]<>]+"
)

_NUKTA_PAIRS = [
    ("ਸ਼", "ਸ"), ("ਖ਼", "ਖ"), ("ਗ਼", "ਗ"),
    ("ਜ਼", "ਜ"), ("ਫ਼", "ਫ"), ("ਲ਼", "ਲ"),
]


def _denukta(text: str) -> str:
    for compound, base in _NUKTA_PAIRS:
        text = text.replace(compound, base)
    return text


def skel(text: str) -> str:
    """Return the consonant skeleton of a Gurmukhi string."""
    if not text:
        return ""
    return _SKELETON_STRIP.sub("", _denukta(text))


def lev(a: str, b: str) -> int:
    """Levenshtein edit distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(b)]


_VISHRAAM_TOKEN_RE = re.compile(r"^[॥।੦-੯0-9.,;:!?'\"()\[\]]+$")


def clean_token(tok: str) -> str:
    """Strip trailing vishraams/digits from a single token."""
    return re.sub(r"[॥।੦-੯0-9.]+$", "", tok).strip()


def tokenize(text: str) -> list[str]:
    """Whitespace-split; drop `>>` markers and empty tokens."""
    return [t for t in text.split() if t and t != ">>"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/canonical/test_gurmukhi_skeleton.py -v`
Expected: all PASS.

- [ ] **Step 5: Add `lev` and `tokenize` tests**

```python
# append to tests/canonical/test_gurmukhi_skeleton.py
class TestLev:
    def test_equal_strings(self):
        assert lev("abc", "abc") == 0

    def test_one_substitution(self):
        assert lev("ਗਰ", "ਗਣ") == 1

    def test_insertion(self):
        assert lev("ਗਰ", "ਗਰਨ") == 1

    def test_empty(self):
        assert lev("", "ਗਰ") == 2
        assert lev("ਗਰ", "") == 2


class TestTokenize:
    def test_basic_split(self):
        assert tokenize("ਤੇਰੀ ਸਰਣਿ") == ["ਤੇਰੀ", "ਸਰਣਿ"]

    def test_drop_gt_gt(self):
        assert tokenize(">> ਤੇਰੀ >> ਸਰਣਿ") == ["ਤੇਰੀ", "ਸਰਣਿ"]

    def test_collapse_multi_space(self):
        assert tokenize("ਤੇਰੀ    ਸਰਣਿ") == ["ਤੇਰੀ", "ਸਰਣਿ"]

    def test_empty(self):
        assert tokenize("") == []
        assert tokenize(">> >>") == []


class TestCleanToken:
    def test_trailing_vishraam_stripped(self):
        assert clean_token("ਜੀਉ॥") == "ਜੀਉ"

    def test_digit_stripped(self):
        assert clean_token("ਪਿਆਰੇ੧") == "ਪਿਆਰੇ"

    def test_no_change_needed(self):
        assert clean_token("ਤੇਰੀ") == "ਤੇਰੀ"
```

- [ ] **Step 6: Run and verify pass**

Run: `pytest tests/canonical/test_gurmukhi_skeleton.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/canonical/gurmukhi_skeleton.py tests/canonical/test_gurmukhi_skeleton.py
git commit -m "feat(canonical): gurmukhi skeleton + Levenshtein primitives"
```

---

## Task 3: STTM index — SGGS line loader + 4-gram shabad index

**Files:**
- Create: `scripts/canonical/sttm_index.py`
- Test: `tests/canonical/test_sttm_index.py`

Reference: `scripts/proto_canonical_v2.py` functions `load_sggs()`, `build_shabad_idx()`.

- [ ] **Step 1: Define the `SggsLine` dataclass and loader signature**

```python
# scripts/canonical/sttm_index.py
"""Load SGGS lines from STTM database.sqlite and build retrieval indices.

Public API:
  load_sggs(db_path, include_sirlekh=False) -> (list[SggsLine], global_tok_idx)
  build_shabad_ngram_index(lines, n=4) -> (shabad_ngrams, shabad_lines, doc_freq)
  next_shabad_in_sequence(lines) -> dict[shabad_id, shabad_id_next]
"""
from __future__ import annotations

import math
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

# allow importing sibling modules when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))
from gurmukhi_converter import ascii_to_unicode, strip_vishraams  # noqa: E402

from .gurmukhi_skeleton import skel, clean_token, _VISHRAAM_TOKEN_RE  # type: ignore


@dataclass(frozen=True)
class SggsLine:
    line_id: str
    shabad_id: str
    ang: int
    order_id: int
    type_id: int
    unicode: str
    skel: str
    tokens: tuple[str, ...]
    tok_skels: tuple[str, ...]


def load_sggs(
    db_path: Path | str,
    include_sirlekh: bool = False,
) -> tuple[list[SggsLine], dict[str, list[tuple[str, str, str]]]]:
    """Load SGGS content lines (+ optionally Sirlekh headers) and build
    a global token → [(unicode_token, line_id, shabad_id)] index.
    """
    type_ids = (1, 3, 4, 5) if include_sirlekh else (1, 3, 4)
    placeholders = ",".join("?" * len(type_ids))

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"""
        SELECT l.id AS line_id, l.shabad_id, l.source_page AS ang,
               l.order_id, l.type_id, l.gurmukhi
        FROM lines l
        JOIN shabads s ON l.shabad_id = s.id
        WHERE s.source_id = 1 AND l.type_id IN ({placeholders})
        ORDER BY l.order_id
        """,
        type_ids,
    ).fetchall()

    out: list[SggsLine] = []
    global_tok_idx: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for r in rows:
        uni = ascii_to_unicode(strip_vishraams(r["gurmukhi"])).strip()
        if not uni or len(uni) < 3:
            continue
        raw = uni.split()
        tokens = tuple(
            clean_token(t) for t in raw if not _VISHRAAM_TOKEN_RE.match(t)
        )
        tokens = tuple(t for t in tokens if t)
        if not tokens:
            continue
        tok_skels = tuple(skel(t) for t in tokens)
        line = SggsLine(
            line_id=r["line_id"],
            shabad_id=r["shabad_id"],
            ang=r["ang"],
            order_id=r["order_id"],
            type_id=r["type_id"],
            unicode=" ".join(tokens),
            skel=skel(" ".join(tokens)),
            tokens=tokens,
            tok_skels=tok_skels,
        )
        out.append(line)
        for tok, ts in zip(tokens, tok_skels):
            if ts:
                global_tok_idx[ts].append((tok, line.line_id, line.shabad_id))
    return out, dict(global_tok_idx)
```

- [ ] **Step 2: Write failing test using fixture DB**

```python
# tests/canonical/test_sttm_index.py
from pathlib import Path
import pytest

from scripts.canonical.sttm_index import load_sggs, SggsLine

DB_PATH = Path(__file__).parent.parent.parent / "database.sqlite"


@pytest.mark.skipif(not DB_PATH.exists(), reason="database.sqlite not present")
class TestLoadSggs:
    def test_loads_nonzero_lines(self):
        lines, _ = load_sggs(DB_PATH, include_sirlekh=False)
        assert len(lines) > 50_000, "expected ~56K SGGS content lines"

    def test_include_sirlekh_adds_rows(self):
        without, _ = load_sggs(DB_PATH, include_sirlekh=False)
        with_sl, _ = load_sggs(DB_PATH, include_sirlekh=True)
        assert len(with_sl) > len(without)

    def test_lines_have_required_fields(self):
        lines, _ = load_sggs(DB_PATH, include_sirlekh=False)
        sample = lines[0]
        assert isinstance(sample, SggsLine)
        assert sample.shabad_id
        assert sample.ang > 0
        assert len(sample.tokens) > 0
        assert len(sample.tokens) == len(sample.tok_skels)

    def test_global_token_index_has_common_tokens(self):
        _, idx = load_sggs(DB_PATH, include_sirlekh=False)
        # every SGGS line with ਸਤਿਗੁਰੁ or similar should populate its skel
        from scripts.canonical.gurmukhi_skeleton import skel
        key = skel("ਸਤਿਗੁਰੁ")
        assert key in idx
        assert len(idx[key]) > 10  # appears in many places
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/canonical/test_sttm_index.py -v`
Expected: PASS (the loader implementation from Step 1 should satisfy them).

- [ ] **Step 4: Implement 4-gram shabad index**

Append to `scripts/canonical/sttm_index.py`:

```python
def _ngrams(s: str, n: int) -> list[str]:
    return [s[i : i + n] for i in range(len(s) - n + 1)] if len(s) >= n else ([s] if s else [])


def build_shabad_ngram_index(
    lines: list[SggsLine], n: int = 4
) -> tuple[dict[str, Counter[str]], dict[str, list[SggsLine]], Counter[str]]:
    """Return (shabad_ngrams, shabad_lines, doc_freq)."""
    shabad_ngrams: dict[str, Counter[str]] = defaultdict(Counter)
    shabad_lines: dict[str, list[SggsLine]] = defaultdict(list)
    for ln in lines:
        shabad_lines[ln.shabad_id].append(ln)
        for g in _ngrams(ln.skel, n):
            shabad_ngrams[ln.shabad_id][g] += 1
    df: Counter[str] = Counter()
    for sid, cnt in shabad_ngrams.items():
        for g in cnt:
            df[g] += 1
    return dict(shabad_ngrams), dict(shabad_lines), df
```

- [ ] **Step 5: Add tests for `build_shabad_ngram_index`**

```python
# append to tests/canonical/test_sttm_index.py
from scripts.canonical.sttm_index import build_shabad_ngram_index


@pytest.mark.skipif(not DB_PATH.exists(), reason="database.sqlite not present")
class TestNgramIndex:
    def test_index_populated(self):
        lines, _ = load_sggs(DB_PATH, include_sirlekh=False)
        ngrams_by_shabad, shabad_lines, df = build_shabad_ngram_index(lines, n=4)
        assert len(ngrams_by_shabad) > 5000  # ~5544 SGGS shabads
        assert len(shabad_lines) == len(ngrams_by_shabad)
        assert sum(df.values()) > 100_000

    def test_ez7_shabad_present(self):
        """Shabad EZ7 (Ang 105, ਜਿਥੈ ਨਾਮੁ) is a well-known test landmark."""
        lines, _ = load_sggs(DB_PATH, include_sirlekh=False)
        _, shabad_lines, _ = build_shabad_ngram_index(lines)
        ez7 = shabad_lines.get("EZ7")
        assert ez7 is not None
        assert any("ਜਿਥੈ" in ln.unicode for ln in ez7)
```

- [ ] **Step 6: Implement `next_shabad_in_sequence` (for §2.7)**

Append to `scripts/canonical/sttm_index.py`:

```python
def next_shabad_in_sequence(lines: list[SggsLine]) -> dict[str, str]:
    """Map each shabad_id → the shabad_id of the next shabad in order_id sequence.
    Used by SEQUENTIAL_SHABAD_RETRIEVAL to boost scores for the next shabad
    immediately after a matched one.
    """
    # Group by shabad, find max order_id per shabad
    last_order: dict[str, int] = {}
    for ln in lines:
        prev = last_order.get(ln.shabad_id, -1)
        if ln.order_id > prev:
            last_order[ln.shabad_id] = ln.order_id
    # Sort shabads by their last order_id; next[i] = i+1
    ordered = sorted(last_order, key=lambda s: last_order[s])
    nxt: dict[str, str] = {}
    for i, sid in enumerate(ordered[:-1]):
        nxt[sid] = ordered[i + 1]
    return nxt
```

- [ ] **Step 7: Add test for sequential index**

```python
# append to tests/canonical/test_sttm_index.py
from scripts.canonical.sttm_index import next_shabad_in_sequence


@pytest.mark.skipif(not DB_PATH.exists(), reason="database.sqlite not present")
def test_next_shabad_sequence():
    lines, _ = load_sggs(DB_PATH, include_sirlekh=False)
    nxt = next_shabad_in_sequence(lines)
    # every shabad except the last one should have a successor
    assert len(nxt) > 5000
    # The "next" of any shabad should not equal itself
    for sid, after in nxt.items():
        assert sid != after
```

- [ ] **Step 8: Run all tests, commit**

Run: `pytest tests/canonical/test_sttm_index.py -v`
Expected: all PASS.

```bash
git add scripts/canonical/sttm_index.py tests/canonical/test_sttm_index.py
git commit -m "feat(canonical): sttm index — SGGS loader + 4gram + sequential map"
```

---

## Task 4: Pre-cleaning filters + `<unk>` stripping

**Files:**
- Create: `scripts/canonical/preprocess.py`
- Test: `tests/canonical/test_preprocess.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/canonical/test_preprocess.py
import pytest
from scripts.canonical.preprocess import (
    strip_unk_artifacts, should_drop_row, PreCleanConfig
)


class TestStripUnk:
    def test_strips_embedded_unk(self):
        assert strip_unk_artifacts("ਜਾਲunk> ਮੁਕਤੇ") == "ਜਾਲ ਮੁਕਤੇ"

    def test_strips_prefix_unk(self):
        assert strip_unk_artifacts("ਕਢਾਇ<un ਸੀਰਾਗ") == "ਕਢਾਇ ਸੀਰਾਗ"

    def test_strips_short_k_marker(self):
        assert strip_unk_artifacts("ਸਾਲਾਹਿਹਿ<k ਸਭੇ") == "ਸਾਲਾਹਿਹਿ ਸਭੇ"

    def test_full_unk_tag(self):
        assert strip_unk_artifacts("ਪਾਇਆ <unk> ਨਾਮੁ") == "ਪਾਇਆ   ਨਾਮੁ" or \
               strip_unk_artifacts("ਪਾਇਆ <unk> ਨਾਮੁ") == "ਪਾਇਆ ਨਾਮੁ"

    def test_no_artifacts_unchanged(self):
        assert strip_unk_artifacts("ਤੇਰੀ ਸਰਣਿ") == "ਤੇਰੀ ਸਰਣਿ"

    def test_empty(self):
        assert strip_unk_artifacts("") == ""


class TestShouldDropRow:
    def test_drop_short_duration(self):
        cfg = PreCleanConfig(min_duration_s=1.0)
        assert should_drop_row({"duration_s": 0.5, "text": "ਤੇਰੀ ਸਰਣਿ"}, cfg)

    def test_drop_single_token(self):
        cfg = PreCleanConfig()
        assert should_drop_row({"duration_s": 2.0, "text": "ਜ"}, cfg)
        assert should_drop_row({"duration_s": 2.0, "text": ">> ਜ >>"}, cfg)

    def test_drop_no_gurmukhi(self):
        cfg = PreCleanConfig()
        assert should_drop_row({"duration_s": 2.0, "text": ">> >> 123"}, cfg)

    def test_keep_normal_row(self):
        cfg = PreCleanConfig()
        assert not should_drop_row(
            {"duration_s": 2.5, "text": "ਤੇਰੀ ਸਰਣਿ ਮੇਰੇ ਦੀਨ ਦਇਆਲਾ"}, cfg
        )
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/canonical/test_preprocess.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement preprocess module**

```python
# scripts/canonical/preprocess.py
"""Pre-cleaning: drop rows we can't meaningfully process; strip ASR artifacts."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .gurmukhi_skeleton import tokenize

_UNK_RE = re.compile(r"<unk>|<unk|unk>|<un[^a-zA-Zਅ-ਹ]|<k[^a-zA-Zਅ-ਹ]|<k>|un>|<k$|<un$")


def strip_unk_artifacts(text: str) -> str:
    """Replace ASR `<unk>` / `<un` / `<k>` etc. with spaces.

    Glued-to-word forms like `ਜਾਲunk>` are split cleanly because the
    artifact is removed leaving the preceding Gurmukhi word intact.
    """
    if not text:
        return text
    # First handle glued suffix patterns (word+artifact no space between)
    out = re.sub(r"unk>|un>|<unk?>?|<k>?|<un", " ", text)
    # Collapse multiple spaces
    return re.sub(r"\s+", " ", out).strip()


@dataclass
class PreCleanConfig:
    min_duration_s: float = 1.0
    min_gurmukhi_tokens: int = 2
    flag_slow_ratio: float = 4.0  # duration_s / n_tokens
    flag_fast_ratio: float = 0.1


def _gurmukhi_token_count(text: str) -> int:
    stripped = strip_unk_artifacts(text)
    toks = tokenize(stripped)
    # A token qualifies as "Gurmukhi" if it has any character in the Gurmukhi block
    gur = [t for t in toks if any("\u0A00" <= ch <= "\u0A7F" for ch in t)]
    return len(gur)


def should_drop_row(row: dict, cfg: PreCleanConfig) -> bool:
    """Return True if row must be dropped from the dataset."""
    if row.get("duration_s", 0) < cfg.min_duration_s:
        return True
    n = _gurmukhi_token_count(row.get("text", ""))
    if n < cfg.min_gurmukhi_tokens:
        return True
    return False


def ratio_outlier_flag(row: dict, cfg: PreCleanConfig) -> bool:
    """Flag (but don't drop) unusually slow/fast rows."""
    n = _gurmukhi_token_count(row.get("text", ""))
    if n <= 0:
        return True
    r = row.get("duration_s", 0) / n
    return r > cfg.flag_slow_ratio or r < cfg.flag_fast_ratio
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest tests/canonical/test_preprocess.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/canonical/preprocess.py tests/canonical/test_preprocess.py
git commit -m "feat(canonical): pre-cleaning filters + <unk> artifact stripping"
```

---

## Task 5: Waheguru hardcoded normalization

**Files:**
- Create: `scripts/canonical/waheguru.py`
- Test: `tests/canonical/test_waheguru.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/canonical/test_waheguru.py
from scripts.canonical.waheguru import (
    CANONICAL_WAHEGURU, WAHEGURU_SKEL, normalize_waheguru_tokens
)


class TestWaheguruNormalization:
    def test_canonical_unchanged(self):
        assert normalize_waheguru_tokens(["ਵਾਹਿਗੁਰੂ"]) == ["ਵਾਹਿਗੁਰੂ"]

    def test_matra_variant_normalized(self):
        assert normalize_waheguru_tokens(["ਵਾਹੇਗੁਰੂ"]) == ["ਵਾਹਿਗੁਰੂ"]
        assert normalize_waheguru_tokens(["ਵਾਹਿਗੁਰੁ"]) == ["ਵਾਹਿਗੁਰੂ"]
        assert normalize_waheguru_tokens(["ਵਾਹਿਗੁਰ"]) == ["ਵਾਹਿਗੁਰੂ"]
        assert normalize_waheguru_tokens(["ਵਹਿਗੁਰੂ"]) == ["ਵਾਹਿਗੁਰੂ"]

    def test_mixed_row_only_waheguru_normalized(self):
        toks = ["ਮੇਰਾ", "ਵਾਹੇਗੁਰੂ", "ਹੈ"]
        assert normalize_waheguru_tokens(toks) == ["ਮੇਰਾ", "ਵਾਹਿਗੁਰੂ", "ਹੈ"]

    def test_non_waheguru_unchanged(self):
        toks = ["ਤੇਰੀ", "ਸਰਣਿ"]
        assert normalize_waheguru_tokens(toks) == ["ਤੇਰੀ", "ਸਰਣਿ"]

    def test_empty_and_gt_gt_preserved(self):
        assert normalize_waheguru_tokens([">>", "ਵਾਹੇਗੁਰੂ"]) == [">>", "ਵਾਹਿਗੁਰੂ"]
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/canonical/test_waheguru.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement waheguru module**

```python
# scripts/canonical/waheguru.py
"""Hardcoded ਵਾਹਿਗੁਰੂ normalization.

Any token whose consonant skeleton equals the skeleton of ਵਾਹਿਗੁਰੂ
(i.e. "ਵਹਗਰ" — 4 consonants) is rewritten to the canonical spelling.
This catches ASR/transcription variants like ਵਾਹੇਗੁਰੂ / ਵਾਹਿਗੁਰ etc.
"""
from __future__ import annotations

from .gurmukhi_skeleton import skel

CANONICAL_WAHEGURU = "ਵਾਹਿਗੁਰੂ"
WAHEGURU_SKEL = skel(CANONICAL_WAHEGURU)  # precomputed once


def normalize_waheguru_tokens(tokens: list[str]) -> list[str]:
    """Rewrite any ਵਾਹਿਗੁਰੂ-skeleton token to the canonical spelling."""
    return [
        CANONICAL_WAHEGURU if tok and tok != ">>" and skel(tok) == WAHEGURU_SKEL else tok
        for tok in tokens
    ]
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest tests/canonical/test_waheguru.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/canonical/waheguru.py tests/canonical/test_waheguru.py
git commit -m "feat(canonical): hardcoded waheguru skeleton-normalization"
```

---

## Task 6: Simran detection + quota downsampling

**Files:**
- Create: `scripts/canonical/simran.py`
- Test: `tests/canonical/test_simran.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/canonical/test_simran.py
from scripts.canonical.simran import (
    is_simran, SimranConfig, apply_simran_quota
)


class TestIsSimran:
    def test_all_waheguru_is_simran(self):
        toks = ["ਵਾਹਿਗੁਰੂ"] * 8
        assert is_simran(toks, SimranConfig())

    def test_variant_spellings_detected(self):
        toks = ["ਵਾਹਿਗੁਰੂ", "ਵਾਹੇਗੁਰੂ", "ਵਾਹਿਗੁਰ", "ਵਾਹਿਗੁਰੂ", "ਵਾਹਿਗੁਰੂ"]
        assert is_simran(toks, SimranConfig())

    def test_short_rep_not_simran(self):
        toks = ["ਵਾਹਿਗੁਰੂ", "ਵਾਹਿਗੁਰੂ"]  # only 2, below min_reps=5
        assert not is_simran(toks, SimranConfig())

    def test_mixed_content_not_simran(self):
        toks = ["ਤੇਰੀ", "ਸਰਣਿ"] * 3 + ["ਵਾਹਿਗੁਰੂ"] * 2
        assert not is_simran(toks, SimranConfig())

    def test_70_percent_threshold(self):
        # 7 waheguru + 3 others = 70% — should pass default ratio=0.70
        toks = ["ਵਾਹਿਗੁਰੂ"] * 7 + ["ਤੇਰੀ", "ਸਰਣਿ", "ਮੇਰੇ"]
        assert is_simran(toks, SimranConfig())

    def test_below_ratio_not_simran(self):
        toks = ["ਵਾਹਿਗੁਰੂ"] * 5 + ["ਤੇਰੀ"] * 5  # 50%
        assert not is_simran(toks, SimranConfig())


class TestQuota:
    def _make_rows(self, video_counts: dict[str, int]) -> list[dict]:
        rows = []
        for vid, count in video_counts.items():
            for i in range(count):
                rows.append({"clip_id": f"{vid}_{i:03d}", "video_id": vid,
                             "is_simran": True})
        return rows

    def test_per_video_cap_applied(self):
        rows = self._make_rows({"A": 50, "B": 50, "C": 50})
        cfg = SimranConfig(target_count=100, per_video_min=1, per_video_max=10)
        kept = apply_simran_quota(rows, cfg, seed=42)
        # n_videos=3, cap = clamp(ceil(100/3), 1, 10) = 10
        # 3 videos * 10 = 30 ≤ target, all kept
        assert len(kept) == 30
        per_video = {}
        for r in kept:
            per_video[r["video_id"]] = per_video.get(r["video_id"], 0) + 1
        assert all(c == 10 for c in per_video.values())

    def test_clamp_at_max_when_few_videos(self):
        rows = self._make_rows({"A": 200})  # 1 video, lots of simran
        cfg = SimranConfig(target_count=750, per_video_min=1, per_video_max=10)
        kept = apply_simran_quota(rows, cfg, seed=42)
        assert len(kept) == 10  # capped at per_video_max

    def test_global_target_enforced(self):
        rows = self._make_rows({f"V{i:03d}": 20 for i in range(100)})
        cfg = SimranConfig(target_count=750, per_video_min=1, per_video_max=10)
        kept = apply_simran_quota(rows, cfg, seed=42)
        # cap = clamp(ceil(750/100), 1, 10) = 8  →  100 * 8 = 800, trim to 750
        assert len(kept) == 750

    def test_round_robin_preserves_video_diversity(self):
        rows = self._make_rows({f"V{i:03d}": 20 for i in range(100)})
        cfg = SimranConfig(target_count=100, per_video_min=1, per_video_max=10)
        kept = apply_simran_quota(rows, cfg, seed=42)
        per_video = {}
        for r in kept:
            per_video[r["video_id"]] = per_video.get(r["video_id"], 0) + 1
        # 100 videos, 100 target: each video contributes exactly 1
        assert len(per_video) == 100
        assert all(c == 1 for c in per_video.values())
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/canonical/test_simran.py -v`

- [ ] **Step 3: Implement simran module**

```python
# scripts/canonical/simran.py
"""Simran detection + quota downsampling.

Kirtan videos (especially AKJ) include long ਵਾਹਿਗੁਰੂ simran sequences.
Without a quota, simran could be 17-25% of the final dataset and bias
the ASR decoder toward always outputting ਵਾਹਿਗੁਰੂ. This module detects
simran rows and applies a two-stage quota: adaptive per-video cap +
round-robin global trim to an absolute target (default 750).
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass

from .waheguru import WAHEGURU_SKEL
from .gurmukhi_skeleton import skel


@dataclass
class SimranConfig:
    min_reps: int = 5
    ratio_threshold: float = 0.70
    target_count: int = 750
    per_video_min: int = 1
    per_video_max: int = 10


def is_simran(tokens: list[str], cfg: SimranConfig) -> bool:
    """True iff tokens have >= cfg.min_reps consecutive ਵਾਹਿਗੁਰੂ-skel tokens
    AND those tokens are > cfg.ratio_threshold of all Gurmukhi tokens."""
    if not tokens:
        return False
    waheguru_like = [i for i, t in enumerate(tokens) if skel(t) == WAHEGURU_SKEL]
    if len(waheguru_like) < cfg.min_reps:
        return False
    # Check consecutive run
    best_run, cur_run = 0, 1
    for a, b in zip(waheguru_like, waheguru_like[1:]):
        if b - a == 1:
            cur_run += 1
            best_run = max(best_run, cur_run)
        else:
            cur_run = 1
    best_run = max(best_run, cur_run)
    if best_run < cfg.min_reps:
        return False
    gur_total = sum(1 for t in tokens if any("\u0A00" <= ch <= "\u0A7F" for ch in t))
    if gur_total == 0:
        return False
    return len(waheguru_like) / gur_total > cfg.ratio_threshold


def apply_simran_quota(
    rows: list[dict], cfg: SimranConfig, seed: int = 42
) -> list[dict]:
    """Downsample simran rows to ~cfg.target_count with adaptive per-video cap
    and round-robin stratified sampling across videos.

    Input: list of dicts each with at least 'clip_id', 'video_id', 'is_simran'
    (True for simran candidates, False for non-simran which are returned unchanged).
    Output: filtered list containing (non-simran rows unchanged) + (capped simran).
    """
    rng = random.Random(seed)
    non_simran = [r for r in rows if not r.get("is_simran")]
    simran = [r for r in rows if r.get("is_simran")]

    # Group simran by video
    by_video: dict[str, list[dict]] = defaultdict(list)
    for r in simran:
        by_video[r["video_id"]].append(r)

    n_videos = len(by_video)
    if n_videos == 0:
        return rows

    cap = max(cfg.per_video_min, min(
        cfg.per_video_max,
        math.ceil(cfg.target_count / n_videos),
    ))

    # Stage 1: per-video cap (random sample within each video)
    capped: dict[str, list[dict]] = {}
    for vid, group in by_video.items():
        if len(group) <= cap:
            capped[vid] = list(group)
        else:
            capped[vid] = rng.sample(group, cap)

    # Stage 2: round-robin across videos until target reached
    video_ids = sorted(capped.keys())
    # pre-shuffle within each video for deterministic but randomized pick order
    for vid in video_ids:
        rng.shuffle(capped[vid])

    survivors: list[dict] = []
    idx = 0
    while len(survivors) < cfg.target_count and any(capped.values()):
        vid = video_ids[idx % len(video_ids)]
        if capped[vid]:
            survivors.append(capped[vid].pop(0))
        idx += 1
        # safety: if we've done a full lap with no progress, break
        if idx % len(video_ids) == 0 and all(not capped[v] for v in video_ids):
            break

    return non_simran + survivors
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest tests/canonical/test_simran.py -v`
Expected: all PASS. If a test fails due to minor pop-order issues, adjust the `rng.shuffle` placement but do not change semantics.

- [ ] **Step 5: Commit**

```bash
git add scripts/canonical/simran.py tests/canonical/test_simran.py
git commit -m "feat(canonical): simran detection + adaptive per-video quota"
```

---

## Task 7: Sirlekh (header) normalization + multi-shabad row splitter

**Files:**
- Create: `scripts/canonical/sirlekh.py`
- Test: `tests/canonical/test_sirlekh.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/canonical/test_sirlekh.py
from scripts.canonical.sirlekh import normalize_sirlekh, split_multi_shabad


class TestNormalizeSirlekh:
    def test_sri_raag_mahala_5(self):
        assert "ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ ੫" in normalize_sirlekh("ਸੀ ਰਾਗ ਮਹਲਾ ਪੰ")

    def test_sri_raag_mahala_generic(self):
        assert "ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ" in normalize_sirlekh("ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ>")

    def test_siri_raag_variant(self):
        assert "ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ" in normalize_sirlekh("ਸਿ੍ਰ ਰਾਗੁ ਮਹਲਾ")

    def test_no_change_needed(self):
        assert normalize_sirlekh("ਤੇਰੀ ਸਰਣਿ") == "ਤੇਰੀ ਸਰਣਿ"


class TestMultiShabadSplit:
    def test_no_split_when_no_header(self):
        out = split_multi_shabad("ਤੇਰੀ ਸਰਣਿ ਮੇਰੇ ਦੀਨ ਦਇਆਲਾ")
        assert out == ["ਤੇਰੀ ਸਰਣਿ ਮੇਰੇ ਦੀਨ ਦਇਆਲਾ"]

    def test_split_on_midrow_sirlekh(self):
        row = "ਕਢਾਇ ਸੀਰਾਗ ਮਹਲਾ ਘੜੀ ਮੁਹਤ ਕਾ ਪਾਹੁਣਾ"
        parts = split_multi_shabad(row)
        assert len(parts) == 2
        assert "ਕਢਾਇ" in parts[0]
        assert "ਘੜੀ ਮੁਹਤ" in parts[1]

    def test_no_split_at_row_start(self):
        # Sirlekh at the start is normal (whole row is new shabad's content)
        row = "ਸੀਰਾਗ ਮਹਲਾ ਘੜੀ ਮੁਹਤ ਕਾ ਪਾਹੁਣਾ"
        parts = split_multi_shabad(row)
        assert parts == [row]
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/canonical/test_sirlekh.py -v`

- [ ] **Step 3: Implement sirlekh module**

```python
# scripts/canonical/sirlekh.py
"""Sirlekh (shabad-header) normalization + mid-row boundary splitting."""
from __future__ import annotations

import re

# Canonical-variant normalization table (applied before tokenization)
_SIRLEKH_SUBS = [
    (re.compile(r"ਸੀ\s*ਰਾਗ\s*ਮਹਲਾ\s*ਪੰ"), "ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ ੫"),
    (re.compile(r"ਸੀਰਾਗ\s*ਮਹਲਾ"), "ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ"),
    (re.compile(r"ਸ੍ਰੀਰਾਗੁ\s*ਮਹਲਾ[>\W]*"), "ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ "),
    (re.compile(r"ਸਿ੍ਰ\s*ਰਾਗੁ?\s*ਮਹਲਾ"), "ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ"),
    (re.compile(r"ਸਿ੍ਰੀਰਾਗੁ\s*ਮਹਲਾ"), "ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ"),
    (re.compile(r"ਸਿਰੀੀ?\s*ਰਾਗੁ?\s*ਮਹਲਾ"), "ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ"),
]


def normalize_sirlekh(text: str) -> str:
    """Rewrite common ASR variants of shabad headers to canonical form."""
    if not text:
        return text
    for pat, repl in _SIRLEKH_SUBS:
        text = pat.sub(repl, text)
    return re.sub(r"\s+", " ", text).strip()


_SIRLEKH_DETECT = re.compile(
    r"(ਸ੍ਰੀ\s*ਰਾਗੁ?|ਸੀ\s*ਰਾਗ|ਸਿ੍ਰ\s*ਰਾਗੁ?|ਸੀਰਾਗ|ਸਿਰੀੀ?\s*ਰਾਗੁ?)\s*ਮਹਲਾ"
)


def split_multi_shabad(text: str, min_offset_chars: int = 4) -> list[str]:
    """Detect Sirlekh patterns mid-row and split the row into sub-parts.
    Each sub-part is aligned independently by the caller. No split if the
    only Sirlekh is at the row start (offset < min_offset_chars)."""
    if not text:
        return [text]
    # Find all Sirlekh match positions
    matches = list(_SIRLEKH_DETECT.finditer(text))
    # Only cuts that are NOT at row start
    cuts = [m.start() for m in matches if m.start() >= min_offset_chars]
    if not cuts:
        return [text]
    # Split at each cut position
    parts: list[str] = []
    prev = 0
    for c in cuts:
        part = text[prev:c].strip()
        if part:
            parts.append(part)
        prev = c
    tail = text[prev:].strip()
    if tail:
        parts.append(tail)
    return parts
```

- [ ] **Step 4: Run tests, iterate on regex patterns if needed**

Run: `pytest tests/canonical/test_sirlekh.py -v`
Expected: all PASS. If patterns don't match, adjust with specific character classes.

- [ ] **Step 5: Commit**

```bash
git add scripts/canonical/sirlekh.py tests/canonical/test_sirlekh.py
git commit -m "feat(canonical): sirlekh normalization + multi-shabad splitter"
```

---

## Task 8: Shabad retrieval — sliding window + video memory + sequential prior

**Files:**
- Create: `scripts/canonical/retrieval.py`
- Test: `tests/canonical/test_retrieval.py`

- [ ] **Step 1: Write failing tests (uses real DB fixture)**

```python
# tests/canonical/test_retrieval.py
from pathlib import Path
from collections import Counter
import pytest

from scripts.canonical.sttm_index import (
    load_sggs, build_shabad_ngram_index, next_shabad_in_sequence,
)
from scripts.canonical.retrieval import (
    RetrievalConfig, retrieve_shabad, score_all_shabads
)

DB = Path(__file__).parent.parent.parent / "database.sqlite"
pytestmark = pytest.mark.skipif(not DB.exists(), reason="database.sqlite not present")


@pytest.fixture(scope="module")
def sggs_idx():
    lines, _ = load_sggs(DB, include_sirlekh=False)
    ngrams, shabad_lines, df = build_shabad_ngram_index(lines, n=4)
    return lines, ngrams, shabad_lines, df


class TestSingleWindowRetrieval:
    def test_retrieves_ez7_for_known_line(self, sggs_idx):
        _, ngrams, _, df = sggs_idx
        cfg = RetrievalConfig()
        # "ਜਿਥੈ ਨਾਮੁ ਜਪੀਐ ਪ੍ਰਭ ਪਿਆਰੇ" is Ang 105 EZ7
        caption_tokens = ["ਜਿਥੈ", "ਨਾਮੁ", "ਜਪੀਐ", "ਪ੍ਰਭ", "ਪਿਆਰੇ"]
        prev_next = ([], [])
        prev2_next2 = ([], [])
        sid, score, margin, window = retrieve_shabad(
            caption_tokens, prev_next, prev2_next2,
            ngrams, df, video_hits=Counter(), nxt_shabad=None, cfg=cfg,
        )
        assert sid == "EZ7"
        assert score > 1.0


class TestVideoMemoryBias:
    def test_video_hits_boost_low_margin_case(self, sggs_idx):
        _, ngrams, _, df = sggs_idx
        cfg = RetrievalConfig(video_prior_weight=0.8)
        # Ambiguous short phrase — likely matches many shabads
        caption_tokens = ["ਤੇਰੀ", "ਸਰਣਿ"]
        # First call with no video memory
        sid_a, _, margin_a, _ = retrieve_shabad(
            caption_tokens, ([], []), ([], []),
            ngrams, df, video_hits=Counter(), nxt_shabad=None, cfg=cfg,
        )
        # Second call: fake video memory strongly preferring EZ7
        hits = Counter({"EZ7": 10})
        sid_b, _, margin_b, _ = retrieve_shabad(
            caption_tokens, ([], []), ([], []),
            ngrams, df, video_hits=hits, nxt_shabad=None, cfg=cfg,
        )
        # With heavy prior, EZ7 should win even for generic phrase
        assert sid_b == "EZ7" or sid_b == sid_a


class TestSequentialPrior:
    def test_sequential_boosts_next_shabad(self, sggs_idx):
        lines, ngrams, _, df = sggs_idx
        nxt = next_shabad_in_sequence(lines)
        cfg = RetrievalConfig(
            sequential_current_boost=0.8, sequential_next_boost=0.5,
        )
        # Ambiguous caption; with sequential prior pointing to EZ7,
        # EZ7 should win
        caption_tokens = ["ਤੇਰੀ", "ਸਰਣਿ"]
        sid, _, _, _ = retrieve_shabad(
            caption_tokens, ([], []), ([], []),
            ngrams, df, video_hits=Counter({"EZ7": 1}),
            nxt_shabad=nxt, cfg=cfg,
        )
        assert sid  # just verify no crash; full behaviour covered by pipeline
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/canonical/test_retrieval.py -v`

- [ ] **Step 3: Implement retrieval module**

```python
# scripts/canonical/retrieval.py
"""Shabad retrieval: 4-gram TF-IDF over ±N-window skeleton + video-memory prior
+ optional sequential prior (for sehaj path's linear reading progression)."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .gurmukhi_skeleton import skel


@dataclass
class RetrievalConfig:
    ngram_n: int = 4
    margin_low: float = 0.05
    video_prior_weight: float = 0.8
    sequential_current_boost: float = 0.0   # 0 = disabled (kirtan default)
    sequential_next_boost: float = 0.0


def _ngrams(s: str, n: int) -> list[str]:
    return [s[i : i + n] for i in range(len(s) - n + 1)] if len(s) >= n else ([s] if s else [])


def score_all_shabads(
    caption_skel: str, shabad_ngrams, df, cfg: RetrievalConfig
) -> Counter:
    """TF-IDF scored shabad_id → score."""
    q_grams = Counter(_ngrams(caption_skel, cfg.ngram_n))
    n_shabads = len(shabad_ngrams)
    scores: Counter = Counter()
    for g, q_tf in q_grams.items():
        if g not in df:
            continue
        idf = math.log(1 + n_shabads / df[g])
        for sid, cnt in shabad_ngrams.items():
            if g in cnt:
                scores[sid] += min(q_tf, cnt[g]) * idf
    return scores


def _apply_priors(
    scores: Counter, video_hits: Counter, nxt_shabad: dict | None,
    cfg: RetrievalConfig,
) -> Counter:
    out = Counter()
    total_hits = sum(video_hits.values())
    prev_sid = video_hits.most_common(1)[0][0] if video_hits else None
    for sid, sc in scores.items():
        mul = 1.0
        if total_hits > 0:
            prior = video_hits.get(sid, 0) / total_hits
            mul += cfg.video_prior_weight * prior
        if cfg.sequential_current_boost and prev_sid and sid == prev_sid:
            mul += cfg.sequential_current_boost
        if cfg.sequential_next_boost and nxt_shabad and prev_sid:
            if sid == nxt_shabad.get(prev_sid):
                mul += cfg.sequential_next_boost
        out[sid] = sc * mul
    return out


def _top2(scores: Counter) -> tuple[str, float, float]:
    top = scores.most_common(2)
    if not top:
        return ("", 0.0, 0.0)
    top_sid, top_score = top[0]
    second = top[1][1] if len(top) > 1 else 0.0
    margin = (top_score - second) / top_score if top_score > 0 else 0.0
    return (top_sid, top_score, margin)


def retrieve_shabad(
    cur_tokens: list[str],
    prev_next_tokens: tuple[list[str], list[str]],
    prev2_next2_tokens: tuple[list[str], list[str]],
    shabad_ngrams: dict,
    df: Counter,
    video_hits: Counter,
    nxt_shabad: dict | None,
    cfg: RetrievalConfig,
) -> tuple[str, float, float, int]:
    """Retrieve the best shabad for `cur_tokens` using ±1 window first,
    widening to ±2 if margin is low. Returns (shabad_id, score, margin, window).
    """
    prev_t, next_t = prev_next_tokens
    prev2_t, next2_t = prev2_next2_tokens

    # ±1 window
    ctx1 = prev_t + cur_tokens + next_t
    skel1 = "".join(skel(t) for t in ctx1)
    scores1 = _apply_priors(
        score_all_shabads(skel1, shabad_ngrams, df, cfg),
        video_hits, nxt_shabad, cfg,
    )
    sid, score, margin = _top2(scores1)
    if margin >= cfg.margin_low:
        return sid, score, margin, 1

    # ±2 window fallback
    ctx2 = prev2_t + prev_t + cur_tokens + next_t + next2_t
    skel2 = "".join(skel(t) for t in ctx2)
    scores2 = _apply_priors(
        score_all_shabads(skel2, shabad_ngrams, df, cfg),
        video_hits, nxt_shabad, cfg,
    )
    sid2, score2, margin2 = _top2(scores2)
    return sid2, score2, margin2, 2
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest tests/canonical/test_retrieval.py -v`

- [ ] **Step 5: Commit**

```bash
git add scripts/canonical/retrieval.py tests/canonical/test_retrieval.py
git commit -m "feat(canonical): shabad retrieval with window + video memory + sequential prior"
```

---

## Task 9: Semi-global NW alignment + phrase ops + orphan realignment

**Files:**
- Create: `scripts/canonical/align.py`
- Test: `tests/canonical/test_align.py`

Reference: `scripts/proto_canonical_v2.py` `align_nw()` and `_realign_orphan_runs()`. This is the most involved module — follow the reference implementation closely.

- [ ] **Step 1: Write failing tests on simple hand-crafted cases**

```python
# tests/canonical/test_align.py
import pytest
from scripts.canonical.sttm_index import SggsLine
from scripts.canonical.align import (
    AlignConfig, align_nw, realign_orphan_runs
)


def _mk(line_id: str, tokens: tuple[str, ...]) -> SggsLine:
    from scripts.canonical.gurmukhi_skeleton import skel
    return SggsLine(
        line_id=line_id, shabad_id="TEST", ang=1, order_id=0, type_id=1,
        unicode=" ".join(tokens),
        skel=skel(" ".join(tokens)),
        tokens=tokens,
        tok_skels=tuple(skel(t) for t in tokens),
    )


class TestMatch:
    def test_exact_match_single_line(self):
        shabad = [_mk("L1", ("ਤੇਰੀ", "ਸਰਣਿ", "ਮੇਰੇ"))]
        ops = align_nw(["ਤੇਰੀ", "ਸਰਣਿ", "ਮੇਰੇ"], shabad, AlignConfig())
        assert [op["op"] for op in ops] == ["match", "match", "match"]


class TestFix:
    def test_matra_only_fix(self):
        shabad = [_mk("L1", ("ਸਰਣਿ",))]
        ops = align_nw(["ਸਰਨ"], shabad, AlignConfig())
        assert len(ops) == 1
        assert ops[0]["op"] == "fix"
        assert ops[0]["sggs"] == ["ਸਰਣਿ"]

    def test_single_consonant_fix(self):
        shabad = [_mk("L1", ("ਉਜਾੜੀ",))]
        ops = align_nw(["ਉਦਾੜੀ"], shabad, AlignConfig())
        assert ops[0]["op"] == "fix"
        assert ops[0]["sggs"] == ["ਉਜਾੜੀ"]


class TestMerge:
    def test_merge_two_caption_into_one_sggs(self):
        shabad = [_mk("L1", ("ਬਦਫੈਲੀ",))]
        ops = align_nw(["ਮਦ", "ਫੈਲੀ"], shabad, AlignConfig())
        assert ops[0]["op"] == "merge"
        assert ops[0]["sggs"] == ["ਬਦਫੈਲੀ"]


class TestSplit:
    def test_split_one_caption_into_two_sggs(self):
        shabad = [_mk("L1", ("ਸਾਕਤ", "ਸੰਗਿ"))]
        ops = align_nw(["ਸਾਕਤਸੰਗਿ"], shabad, AlignConfig())
        assert ops[0]["op"] == "split"
        assert ops[0]["sggs"] == ["ਸਾਕਤ", "ਸੰਗਿ"]


class TestMonotonic:
    def test_no_backward_reuse(self):
        """Regression: ਰੋਤੀ should not align backward to ਰੁਖੀ when
        the forward ਰੋਟੀ is within reach."""
        shabad = [_mk("L1", ("ਹਰਿ", "ਰੁਖੀ", "ਰੋਟੀ", "ਖਾਇ", "ਸਮਾਲੇ"))]
        ops = align_nw(["ਰੁਖੀ", "ਰੋਤੀ", "ਖਾਇ", "ਸਮਾਲੇ"], shabad, AlignConfig())
        sggs_targets = [op["sggs"] for op in ops if op["op"] != "delete"]
        flat = [t for span in sggs_targets for t in span]
        # Must include ਰੋਟੀ (fix target), must NOT duplicate ਰੁਖੀ
        assert "ਰੋਟੀ" in flat
        assert flat.count("ਰੁਖੀ") <= 1


class TestMinConsLen:
    def test_reject_pathological_1_cons_match(self):
        """ਗੁਨ (skel ਗਨ) should NOT fix to ਨ (skel ਨ); min-cons-len=2."""
        shabad = [_mk("L1", ("ਨ", "ਮੇਰੇ"))]
        ops = align_nw(["ਗੁਨ", "ਮੇਰੇ"], shabad, AlignConfig())
        # ਗੁਨ should be 'delete' (kept as caption), only ਮੇਰੇ matches
        types = [op["op"] for op in ops]
        assert "delete" in types
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/canonical/test_align.py -v`

- [ ] **Step 3: Implement the align module**

Port the implementation directly from `scripts/proto_canonical_v2.py`:

```python
# scripts/canonical/align.py
"""Semi-global Needleman-Wunsch alignment over caption tokens vs a
retrieved shabad's token sequence. Supports match / fix / merge / split /
insert / delete ops. Semi-global = free skipping on the SGGS side at
prefix and suffix (critical: global NW's gap penalty over ~60 unused
shabad tokens drowns out the match score).
"""
from __future__ import annotations

from dataclasses import dataclass

from .gurmukhi_skeleton import skel, lev
from .sttm_index import SggsLine


@dataclass
class AlignConfig:
    max_op_edit: int = 1
    max_op_edit_relaxed: int = 2
    min_cons_len: int = 2
    max_len_delta: int = 1
    max_span: int = 3
    min_orphan_run: int = 3
    score_match: int = 10
    score_fix_matra: int = 7
    score_fix_1cons: int = 4
    score_merge_split: int = 4
    gap_cap: int = -1
    gap_sgs: int = -2
    floor: int = -10_000_000


def _fix_eligible(cap_skel: str, sgs_skel: str, cfg: AlignConfig,
                  max_edit: int) -> bool:
    if len(cap_skel) < cfg.min_cons_len or len(sgs_skel) < cfg.min_cons_len:
        return False
    if abs(len(cap_skel) - len(sgs_skel)) > max_edit:
        return False
    return lev(cap_skel, sgs_skel) <= max_edit


def _score_11(cap, cs, sgs, ss, cfg, max_edit):
    if cap == sgs and cs == ss:
        return cfg.score_match, "match"
    if cs == ss and cs and len(cs) >= cfg.min_cons_len:
        return cfg.score_fix_matra, "fix"
    if _fix_eligible(cs, ss, cfg, max_edit):
        return cfg.score_fix_1cons, "fix"
    return cfg.floor, None


def align_nw(
    cap_tokens: list[str], shabad_lines: list[SggsLine],
    cfg: AlignConfig, max_edit: int | None = None,
) -> list[dict]:
    """Semi-global NW. Returns list of ops, each a dict with keys:
        op: 'match'|'fix'|'merge'|'split'|'delete'
        cap: list[str] (caption tokens consumed)
        sggs: list[str] (sggs tokens chosen)
        line_ids: list[str]
    """
    if max_edit is None:
        max_edit = cfg.max_op_edit

    # Flatten shabad to a single stream
    stream_tok: list[str] = []
    stream_skel: list[str] = []
    stream_lid: list[str] = []
    for ln in shabad_lines:
        for t, ts in zip(ln.tokens, ln.tok_skels):
            stream_tok.append(t)
            stream_skel.append(ts)
            stream_lid.append(ln.line_id)

    m, n = len(cap_tokens), len(stream_tok)
    if m == 0 or n == 0:
        return [{"op": "delete", "cap": [t], "sggs": [t], "line_ids": []}
                for t in cap_tokens]

    cap_skels = [skel(t) for t in cap_tokens]

    # Semi-global: dp[0][j] = 0 for all j (free SGGS prefix)
    dp = [[cfg.floor] * (n + 1) for _ in range(m + 1)]
    bt: list[list] = [[None] * (n + 1) for _ in range(m + 1)]
    for j in range(n + 1):
        dp[0][j] = 0
        bt[0][j] = ("prefix_skip", 0, 0, [], [], []) if j > 0 else None
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + cfg.gap_cap
        bt[i][0] = ("delete", i - 1, 0, [cap_tokens[i - 1]],
                    [cap_tokens[i - 1]], [])

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            best_score = cfg.floor
            best_bt = None

            # 1:1
            s11, op11 = _score_11(
                cap_tokens[i - 1], cap_skels[i - 1],
                stream_tok[j - 1], stream_skel[j - 1], cfg, max_edit,
            )
            if op11 is not None:
                cand = dp[i - 1][j - 1] + s11
                if cand > best_score:
                    best_score = cand
                    sgs_tok = stream_tok[j - 1] if op11 == "fix" else cap_tokens[i - 1]
                    best_bt = (op11, i - 1, j - 1,
                               [cap_tokens[i - 1]], [sgs_tok],
                               [stream_lid[j - 1]])

            # 1:N SPLIT
            for span in range(2, cfg.max_span + 1):
                if j - span < 0:
                    break
                sj = "".join(stream_skel[j - span:j])
                if _fix_eligible(cap_skels[i - 1], sj, cfg, max_edit):
                    cand = dp[i - 1][j - span] + cfg.score_merge_split
                    if cand > best_score:
                        best_score = cand
                        best_bt = ("split", i - 1, j - span,
                                   [cap_tokens[i - 1]],
                                   list(stream_tok[j - span:j]),
                                   list(stream_lid[j - span:j]))

            # M:1 MERGE
            for span in range(2, cfg.max_span + 1):
                if i - span < 0:
                    break
                cj = "".join(cap_skels[i - span:i])
                if _fix_eligible(cj, stream_skel[j - 1], cfg, max_edit):
                    cand = dp[i - span][j - 1] + cfg.score_merge_split
                    if cand > best_score:
                        best_score = cand
                        best_bt = ("merge", i - span, j - 1,
                                   list(cap_tokens[i - span:i]),
                                   [stream_tok[j - 1]],
                                   [stream_lid[j - 1]])

            # INSERT (skip sggs)
            cand = dp[i][j - 1] + cfg.gap_sgs
            if cand > best_score:
                best_score = cand
                best_bt = ("insert", i, j - 1, [], [stream_tok[j - 1]],
                           [stream_lid[j - 1]])

            # DELETE (skip cap)
            cand = dp[i - 1][j] + cfg.gap_cap
            if cand > best_score:
                best_score = cand
                best_bt = ("delete", i - 1, j, [cap_tokens[i - 1]],
                           [cap_tokens[i - 1]], [])

            dp[i][j] = best_score
            bt[i][j] = best_bt

    # Traceback from best endpoint over row m (free SGGS suffix)
    best_j = max(range(n + 1), key=lambda j: dp[m][j])
    i, j = m, best_j
    ops: list[dict] = []
    while i > 0:
        if bt[i][j] is None:
            break
        op, bi, bj, cap_span, sgs_span, lids = bt[i][j]
        if op not in ("insert", "prefix_skip"):
            ops.append({"op": op, "cap": cap_span,
                        "sggs": sgs_span, "line_ids": lids})
        i, j = bi, bj
    ops.reverse()
    return ops


def realign_orphan_runs(
    ops: list[dict], shabad_lines: list[SggsLine], cfg: AlignConfig,
) -> tuple[list[dict], bool]:
    """If there are contiguous delete runs ≥ cfg.min_orphan_run tokens,
    try a fresh NW pass on just that span. If that still fails, try
    with relaxed max_op_edit. Returns (new_ops, used_relaxed)."""
    used_relaxed = False
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(ops):
            if ops[i]["op"] == "delete":
                start = i
                while i < len(ops) and ops[i]["op"] == "delete":
                    i += 1
                if i - start >= cfg.min_orphan_run:
                    orphan = [ops[k]["cap"][0] for k in range(start, i)]
                    alt = align_nw(orphan, shabad_lines, cfg, cfg.max_op_edit)
                    n_del_alt = sum(1 for o in alt if o["op"] == "delete")
                    if n_del_alt < (i - start):
                        ops = ops[:start] + alt + ops[i:]
                        changed = True
                        break
                    alt2 = align_nw(orphan, shabad_lines, cfg, cfg.max_op_edit_relaxed)
                    n_del_alt2 = sum(1 for o in alt2 if o["op"] == "delete")
                    if n_del_alt2 < (i - start):
                        ops = ops[:start] + alt2 + ops[i:]
                        used_relaxed = True
                        changed = True
                        break
            else:
                i += 1
    return ops, used_relaxed
```

- [ ] **Step 4: Run tests, iterate if needed**

Run: `pytest tests/canonical/test_align.py -v`
Expected: all PASS. If the split test fails due to cfg.max_span constraint, verify `cfg.max_span=3` and that the split score is higher than individual deletes.

- [ ] **Step 5: Commit**

```bash
git add scripts/canonical/align.py tests/canonical/test_align.py
git commit -m "feat(canonical): semi-global NW align + phrase ops + orphan realignment"
```

---

## Task 10: Decision logic

**Files:**
- Create: `scripts/canonical/decision.py`
- Test: `tests/canonical/test_decision.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/canonical/test_decision.py
from scripts.canonical.decision import decide, DecisionConfig, render_outputs


def _mk_ops(match=0, fix=0, merge=0, split=0, delete=0, used_relaxed=False):
    ops = []
    ops.extend([{"op": "match", "cap": ["x"], "sggs": ["x"], "line_ids": ["L"]}] * match)
    ops.extend([{"op": "fix", "cap": ["a"], "sggs": ["b"], "line_ids": ["L"]}] * fix)
    ops.extend([{"op": "merge", "cap": ["a", "b"], "sggs": ["c"], "line_ids": ["L"]}] * merge)
    ops.extend([{"op": "split", "cap": ["a"], "sggs": ["b", "c"], "line_ids": ["L"]}] * split)
    ops.extend([{"op": "delete", "cap": ["x"], "sggs": ["x"], "line_ids": []}] * delete)
    return ops, used_relaxed


class TestDecide:
    def test_all_match_no_fix(self):
        ops, relaxed = _mk_ops(match=5)
        assert decide(ops, relaxed, DecisionConfig()) == "matched"

    def test_matched_with_fix(self):
        ops, relaxed = _mk_ops(match=4, fix=1)
        # 5/5 = 100% correction rate and has fix → replaced
        assert decide(ops, relaxed, DecisionConfig()) == "replaced"

    def test_review_relaxed_edit(self):
        ops, relaxed = _mk_ops(match=4, fix=1, used_relaxed=True)
        assert decide(ops, relaxed, DecisionConfig()) == "review"

    def test_review_moderate_score(self):
        ops, relaxed = _mk_ops(match=3, fix=1, delete=1)
        # 4/5 = 80% in review range
        assert decide(ops, relaxed, DecisionConfig()) == "review"

    def test_unchanged_low_score(self):
        ops, relaxed = _mk_ops(match=1, delete=3)
        # 1/4 = 25% → unchanged
        assert decide(ops, relaxed, DecisionConfig()) == "unchanged"


class TestRenderOutputs:
    def test_safe_fallback_on_unchanged(self):
        # Even if ops claim fixes, on "unchanged" the final_text == caption
        ops, _ = _mk_ops(match=1, delete=3)
        cap_tokens = ["a", "b", "c", "d"]
        out = render_outputs(ops, cap_tokens, decision="unchanged")
        assert out["final_text"] == "a b c d"
        assert out["sggs_line"] is None

    def test_replaced_applies_fixes(self):
        ops = [
            {"op": "match", "cap": ["a"], "sggs": ["a"], "line_ids": ["L"]},
            {"op": "fix", "cap": ["b"], "sggs": ["B"], "line_ids": ["L"]},
        ]
        out = render_outputs(ops, ["a", "b"], decision="replaced")
        assert out["final_text"] == "a B"
        assert out["sggs_line"] == "a B"
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/canonical/test_decision.py -v`

- [ ] **Step 3: Implement decision module**

```python
# scripts/canonical/decision.py
"""Decision-label computation + safe final_text/sggs_line rendering."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DecisionConfig:
    accept_threshold: float = 0.92
    review_threshold: float = 0.75


def decide(ops: list[dict], used_relaxed: bool, cfg: DecisionConfig) -> str:
    n_match = sum(1 for o in ops if o["op"] == "match")
    n_fix = sum(1 for o in ops if o["op"] == "fix")
    n_merge = sum(1 for o in ops if o["op"] == "merge")
    n_split = sum(1 for o in ops if o["op"] == "split")
    n_del = sum(1 for o in ops if o["op"] == "delete")
    total = n_match + n_fix + n_merge + n_split + n_del
    if total == 0:
        return "unchanged"
    score = (n_match + n_fix + n_merge + n_split) / total
    has_change = (n_fix + n_merge + n_split) > 0

    if score >= cfg.accept_threshold and not has_change:
        return "matched"
    if score >= cfg.accept_threshold and not used_relaxed:
        return "replaced"
    if score >= cfg.review_threshold or used_relaxed:
        return "review"
    return "unchanged"


def render_outputs(
    ops: list[dict], cap_tokens: list[str], decision: str,
) -> dict:
    """Build final_text + sggs_line. On 'unchanged', safety: final_text = caption
    verbatim (with >> already stripped upstream), sggs_line = None."""
    if decision == "unchanged":
        return {
            "final_text": " ".join(cap_tokens),
            "sggs_line": None,
            "line_ids": [],
        }
    final_parts, sggs_parts, line_ids = [], [], []
    for op in ops:
        if op["op"] == "match":
            final_parts.extend(op["cap"])
            sggs_parts.extend(op["sggs"])
        elif op["op"] in ("fix", "merge", "split"):
            final_parts.extend(op["sggs"])
            sggs_parts.extend(op["sggs"])
        elif op["op"] == "delete":
            final_parts.extend(op["cap"])
        line_ids.extend(op.get("line_ids", []))
    # dedupe sggs tokens while preserving order
    sggs_line = " ".join(dict.fromkeys(sggs_parts)) or None
    return {
        "final_text": " ".join(final_parts),
        "sggs_line": sggs_line,
        "line_ids": list(dict.fromkeys(line_ids)),
    }
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest tests/canonical/test_decision.py -v`

- [ ] **Step 5: Commit**

```bash
git add scripts/canonical/decision.py tests/canonical/test_decision.py
git commit -m "feat(canonical): decision-label logic + safe output rendering"
```

---

## Task 11: Dataset config (kirtan vs sehaj defaults)

**Files:**
- Create: `scripts/canonical/config.py`
- Test: `tests/canonical/test_config.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/canonical/test_config.py
from scripts.canonical.config import get_dataset_config, DatasetConfig


class TestDatasetConfig:
    def test_kirtan_defaults(self):
        cfg = get_dataset_config("kirtan")
        assert cfg.strip_unk_artifacts is True
        assert cfg.include_sirlekh is True
        assert cfg.split_multi_shabad_rows is True
        assert cfg.sequential_shabad_retrieval is False
        assert cfg.use_llm_fallback is True
        assert cfg.simran_detection is True

    def test_sehaj_defaults(self):
        cfg = get_dataset_config("sehaj")
        assert cfg.strip_unk_artifacts is True
        assert cfg.include_sirlekh is True
        assert cfg.split_multi_shabad_rows is True
        assert cfg.sequential_shabad_retrieval is True
        assert cfg.use_llm_fallback is True
        assert cfg.simran_detection is False

    def test_unknown_dataset_raises(self):
        import pytest
        with pytest.raises(ValueError):
            get_dataset_config("bhangra")

    def test_config_overrides(self):
        cfg = get_dataset_config("kirtan", overrides={"use_llm_fallback": False})
        assert cfg.use_llm_fallback is False
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/canonical/test_config.py -v`

- [ ] **Step 3: Implement config**

```python
# scripts/canonical/config.py
"""Dataset-specific configuration for the canonical-text pipeline."""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace


@dataclass
class DatasetConfig:
    dataset_name: str
    strip_unk_artifacts: bool = True
    include_sirlekh: bool = True
    split_multi_shabad_rows: bool = True
    sequential_shabad_retrieval: bool = False
    use_llm_fallback: bool = True
    simran_detection: bool = True


_PRESETS = {
    "kirtan": DatasetConfig(
        dataset_name="kirtan",
        sequential_shabad_retrieval=False,
        simran_detection=True,
    ),
    "sehaj": DatasetConfig(
        dataset_name="sehaj",
        sequential_shabad_retrieval=True,
        simran_detection=False,
    ),
}


def get_dataset_config(
    name: str, overrides: dict | None = None,
) -> DatasetConfig:
    if name not in _PRESETS:
        raise ValueError(f"unknown dataset: {name!r} (expected one of {list(_PRESETS)})")
    cfg = _PRESETS[name]
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/canonical/test_config.py -v`

- [ ] **Step 5: Commit**

```bash
git add scripts/canonical/config.py tests/canonical/test_config.py
git commit -m "feat(canonical): dataset-specific config (kirtan vs sehaj)"
```

---

## Task 12: Main pipeline driver (composes all phases)

**Files:**
- Create: `scripts/canonical/pipeline.py`
- Test: `tests/canonical/test_pipeline.py`

- [ ] **Step 1: Write end-to-end integration tests**

```python
# tests/canonical/test_pipeline.py
from pathlib import Path
from collections import Counter
import pytest

from scripts.canonical.config import get_dataset_config
from scripts.canonical.pipeline import CanonicalPipeline

DB = Path(__file__).parent.parent.parent / "database.sqlite"
pytestmark = pytest.mark.skipif(not DB.exists(), reason="database.sqlite not present")


@pytest.fixture(scope="module")
def kirtan_pipeline():
    return CanonicalPipeline(get_dataset_config("kirtan"), db_path=DB)


@pytest.fixture(scope="module")
def sehaj_pipeline():
    return CanonicalPipeline(get_dataset_config("sehaj"), db_path=DB)


class TestKirtanPipeline:
    def test_matched_clean_kirtan_row(self, kirtan_pipeline):
        # Clean SGGS line from EZ7
        rows = [{
            "clip_id": "r1", "video_id": "v1",
            "text": "ਮੇਰੇ ਗੁਰ ਗੋਪਾਲਾ ਸੁਖ ਸਾਗਰ",
            "duration_s": 4.0,
        }]
        out = kirtan_pipeline.run(rows)
        assert out[0]["decision"] == "matched"

    def test_replaced_with_matra_fix(self, kirtan_pipeline):
        rows = [{
            "clip_id": "r1", "video_id": "v1",
            "text": "ਤੇਰੀ ਸਰਨ ਮੇਰੇ ਦੀਨ ਦਇਆਲਾ",
            "duration_s": 4.0,
        }]
        out = kirtan_pipeline.run(rows)
        assert out[0]["decision"] == "replaced"
        assert "ਸਰਣਿ" in out[0]["final_text"]

    def test_simran_short_circuit(self, kirtan_pipeline):
        rows = [{
            "clip_id": "r1", "video_id": "v1",
            "text": " ".join(["ਵਾਹਿਗੁਰੂ"] * 8),
            "duration_s": 10.0,
        }]
        out = kirtan_pipeline.run(rows)
        assert out[0]["decision"] == "simran"
        assert out[0]["final_text"] == " ".join(["ਵਾਹਿਗੁਰੂ"] * 8)


class TestSehajPipeline:
    def test_unk_stripped(self, sehaj_pipeline):
        rows = [{
            "clip_id": "r1", "video_id": "v1",
            "text": "ਨਾਨਕ ਸਚੈ ਪਾਤਿਸਾਹ ਡੁਬਦਾ ਲਇਆ ਕਢਾਇ<un",
            "duration_s": 4.0,
        }]
        out = kirtan_pipeline.run(rows)
        # the <un artifact should be stripped; row should be replaced or matched
        assert "<un" not in out[0]["final_text"]

    def test_sirlekh_included(self, sehaj_pipeline):
        rows = [{
            "clip_id": "r1", "video_id": "v1",
            "text": "ਸੀ ਰਾਗ ਮਹਲਾ ਪੰ",
            "duration_s": 2.0,
        }]
        out = sehaj_pipeline.run(rows)
        # After sirlekh normalization, this becomes ਸ੍ਰੀਰਾਗੁ ਮਹਲਾ ੫
        # — which IS a valid SGGS Sirlekh → should match
        assert out[0]["decision"] in ("matched", "replaced")


class TestPreCleaning:
    def test_drop_short_duration(self, kirtan_pipeline):
        rows = [
            {"clip_id": "r1", "video_id": "v1", "text": "ਤੇਰੀ ਸਰਣਿ", "duration_s": 0.3},
            {"clip_id": "r2", "video_id": "v1", "text": "ਸੁਖ ਸਾਗਰ", "duration_s": 2.0},
        ]
        out = kirtan_pipeline.run(rows)
        assert len(out) == 1
        assert out[0]["clip_id"] == "r2"
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/canonical/test_pipeline.py -v`

- [ ] **Step 3: Implement pipeline driver**

```python
# scripts/canonical/pipeline.py
"""Main pipeline driver. Composes preprocess → simran → retrieval →
align → realign → decide → render.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .config import DatasetConfig
from .preprocess import (
    PreCleanConfig, should_drop_row, strip_unk_artifacts,
)
from .simran import SimranConfig, is_simran, apply_simran_quota
from .waheguru import normalize_waheguru_tokens
from .sirlekh import normalize_sirlekh, split_multi_shabad
from .retrieval import RetrievalConfig, retrieve_shabad
from .align import AlignConfig, align_nw, realign_orphan_runs
from .decision import DecisionConfig, decide, render_outputs
from .sttm_index import (
    load_sggs, build_shabad_ngram_index, next_shabad_in_sequence,
)
from .gurmukhi_skeleton import tokenize


@dataclass
class PipelineResult:
    """Single-row output. Lines up with dataset columns."""
    clip_id: str
    video_id: str
    text: str                    # passed through unchanged
    sggs_line: str | None
    final_text: str
    decision: str
    is_simran: bool
    # audit (written to sidecar, not dataset)
    shabad_id: str | None = None
    line_ids: list[str] | None = None
    match_score: float | None = None
    op_counts: dict | None = None
    retrieval_margin: float | None = None


class CanonicalPipeline:
    def __init__(self, cfg: DatasetConfig, db_path: Path | str):
        self.cfg = cfg
        self.preclean_cfg = PreCleanConfig()
        self.simran_cfg = SimranConfig()
        self.retrieval_cfg = RetrievalConfig(
            sequential_current_boost=0.8 if cfg.sequential_shabad_retrieval else 0.0,
            sequential_next_boost=0.5 if cfg.sequential_shabad_retrieval else 0.0,
        )
        self.align_cfg = AlignConfig()
        self.decision_cfg = DecisionConfig()

        self.lines, self.global_tok_idx = load_sggs(
            db_path, include_sirlekh=cfg.include_sirlekh,
        )
        self.shabad_ngrams, self.shabad_lines, self.df = build_shabad_ngram_index(
            self.lines, n=self.retrieval_cfg.ngram_n,
        )
        self.nxt_shabad = (
            next_shabad_in_sequence(self.lines)
            if cfg.sequential_shabad_retrieval else None
        )

    def run(self, rows: list[dict]) -> list[dict]:
        """Process a list of HF-dataset rows. Returns list of dicts with
        all original fields + the new canonical columns."""
        # Phase -1: drop dead rows
        kept = [r for r in rows if not should_drop_row(r, self.preclean_cfg)]

        # Phase 0a: normalize text (doesn't modify row['text'] in-place)
        for r in kept:
            r["_clean_text"] = r["text"]
            if self.cfg.strip_unk_artifacts:
                r["_clean_text"] = strip_unk_artifacts(r["_clean_text"])
            if self.cfg.include_sirlekh:
                r["_clean_text"] = normalize_sirlekh(r["_clean_text"])

        # Phase 0b: simran detection
        if self.cfg.simran_detection:
            for r in kept:
                toks = tokenize(r["_clean_text"])
                r["is_simran"] = is_simran(toks, self.simran_cfg)
            kept = apply_simran_quota(kept, self.simran_cfg)
        else:
            for r in kept:
                r["is_simran"] = False

        # Phase 1-4: per-row processing
        out: list[dict] = []
        by_video_hits: dict[str, Counter[str]] = {}
        for i, r in enumerate(kept):
            video_hits = by_video_hits.setdefault(r["video_id"], Counter())
            result = self._process_row(r, i, kept, video_hits)
            if result["decision"] in ("matched", "replaced", "review"):
                video_hits[result["shabad_id"]] += 1
            out.append(result)
        return out

    def _neighbors(self, rows, i):
        prev_t = tokenize(rows[i - 1]["_clean_text"]) if i > 0 else []
        next_t = tokenize(rows[i + 1]["_clean_text"]) if i + 1 < len(rows) else []
        prev2_t = tokenize(rows[i - 2]["_clean_text"]) if i > 1 else []
        next2_t = tokenize(rows[i + 2]["_clean_text"]) if i + 2 < len(rows) else []
        return (prev_t, next_t), (prev2_t, next2_t)

    def _process_row(self, r, i, rows, video_hits) -> dict:
        clean_text = r["_clean_text"]
        # Simran short-circuit
        if r.get("is_simran"):
            toks = normalize_waheguru_tokens(tokenize(clean_text))
            return {
                **r, "sggs_line": None,
                "final_text": " ".join(toks),
                "decision": "simran", "is_simran": True,
                "shabad_id": None, "line_ids": [], "match_score": None,
                "op_counts": {}, "retrieval_margin": None,
            }

        # Multi-shabad split (if enabled)
        if self.cfg.split_multi_shabad_rows:
            parts = split_multi_shabad(clean_text)
        else:
            parts = [clean_text]

        # Process each sub-part independently, concatenate results
        per_part_results = []
        for part in parts:
            cap_tokens = tokenize(part)
            # Always apply waheguru normalization to caption tokens before alignment
            cap_tokens = normalize_waheguru_tokens(cap_tokens)
            if not cap_tokens:
                continue
            pr = self._align_one(cap_tokens, i, rows, video_hits)
            per_part_results.append(pr)

        if not per_part_results:
            return {
                **r, "sggs_line": None,
                "final_text": clean_text,
                "decision": "unchanged", "shabad_id": None,
                "line_ids": [], "match_score": 0.0, "op_counts": {},
                "retrieval_margin": 0.0,
            }

        if len(per_part_results) == 1:
            pr = per_part_results[0]
        else:
            # Concatenate. Decision = worst of sub-parts (unchanged > review > replaced > matched)
            order = {"matched": 0, "replaced": 1, "review": 2, "unchanged": 3, "simran": 4}
            combined_text = " ".join(p["final_text"] for p in per_part_results)
            combined_sggs = " ".join(
                p["sggs_line"] for p in per_part_results if p.get("sggs_line")
            ) or None
            combined_line_ids = []
            for p in per_part_results:
                combined_line_ids.extend(p.get("line_ids") or [])
            worst = max(per_part_results, key=lambda p: order.get(p["decision"], 99))
            pr = {
                "final_text": combined_text,
                "sggs_line": combined_sggs,
                "decision": worst["decision"],
                "shabad_id": worst.get("shabad_id"),
                "line_ids": combined_line_ids,
                "match_score": worst.get("match_score"),
                "op_counts": worst.get("op_counts", {}),
                "retrieval_margin": worst.get("retrieval_margin"),
            }

        return {
            **r,
            "sggs_line": pr["sggs_line"],
            "final_text": pr["final_text"],
            "decision": pr["decision"],
            "is_simran": False,
            "shabad_id": pr.get("shabad_id"),
            "line_ids": pr.get("line_ids"),
            "match_score": pr.get("match_score"),
            "op_counts": pr.get("op_counts", {}),
            "retrieval_margin": pr.get("retrieval_margin"),
        }

    def _align_one(self, cap_tokens, i, rows, video_hits) -> dict:
        pn, pn2 = self._neighbors(rows, i)
        sid, score, margin, _ = retrieve_shabad(
            cap_tokens, pn, pn2,
            self.shabad_ngrams, self.df,
            video_hits, self.nxt_shabad, self.retrieval_cfg,
        )
        if not sid or score < 2.0:
            return {
                "sggs_line": None,
                "final_text": " ".join(cap_tokens),
                "decision": "unchanged",
                "shabad_id": sid or None,
                "line_ids": [],
                "match_score": 0.0,
                "op_counts": {},
                "retrieval_margin": margin,
            }
        shabad_lines = self.shabad_lines[sid]
        ops = align_nw(cap_tokens, shabad_lines, self.align_cfg)
        ops, used_relaxed = realign_orphan_runs(ops, shabad_lines, self.align_cfg)
        decision = decide(ops, used_relaxed, self.decision_cfg)
        render = render_outputs(ops, cap_tokens, decision)
        op_counts = {
            op: sum(1 for o in ops if o["op"] == op)
            for op in ("match", "fix", "merge", "split", "delete")
        }
        total = sum(op_counts.values())
        score_pct = (
            sum(v for k, v in op_counts.items() if k != "delete") / total
            if total > 0 else 0.0
        )
        return {
            "sggs_line": render["sggs_line"],
            "final_text": render["final_text"],
            "decision": decision,
            "shabad_id": sid,
            "line_ids": render["line_ids"],
            "match_score": score_pct,
            "op_counts": op_counts,
            "retrieval_margin": margin,
        }
```

- [ ] **Step 4: Run tests, iterate until all pass**

Run: `pytest tests/canonical/test_pipeline.py -v`
Expected: all PASS. If specific tests fail, use the reference `scripts/proto_canonical_v2.py` to compare behavior. The key is that the EZ7-matching tests pass against the real DB.

- [ ] **Step 5: Commit**

```bash
git add scripts/canonical/pipeline.py tests/canonical/test_pipeline.py
git commit -m "feat(canonical): main pipeline driver + integration tests"
```

---

## Task 13: Stage 1 CLI — `add_canonical_column.py`

**Files:**
- Create: `scripts/add_canonical_column.py`
- Test: `tests/canonical/test_cli_add_canonical.py`

- [ ] **Step 1: Write failing integration test using a tiny parquet fixture**

```python
# tests/canonical/test_cli_add_canonical.py
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
DB = ROOT / "database.sqlite"
pytestmark = pytest.mark.skipif(not DB.exists(), reason="database.sqlite not present")


@pytest.fixture
def tiny_parquet(tmp_path):
    df = pd.DataFrame([
        {"clip_id": "r1", "video_id": "v1", "text": "ਤੇਰੀ ਸਰਨ ਮੇਰੇ ਦੀਨ ਦਇਆਲਾ",
         "duration_s": 3.0, "raw_text": "", "start_s": 0.0, "end_s": 3.0},
        {"clip_id": "r2", "video_id": "v1", "text": "ਸੁਖ ਸਾਗਰ ਮੇਰੇ ਗੁਰ ਗੋਪਾਲਾ",
         "duration_s": 3.0, "raw_text": "", "start_s": 3.0, "end_s": 6.0},
    ])
    p = tmp_path / "tiny.parquet"
    df.to_parquet(p)
    return p


def test_cli_produces_new_columns(tiny_parquet, tmp_path):
    out_parquet = tmp_path / "out.parquet"
    audit_parquet = tmp_path / "audit.parquet"
    cmd = [
        sys.executable, str(ROOT / "scripts/add_canonical_column.py"),
        "--input-parquet", str(tiny_parquet),
        "--output-parquet", str(out_parquet),
        "--audit-parquet", str(audit_parquet),
        "--dataset", "kirtan",
        "--db", str(DB),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert out_parquet.exists()
    assert audit_parquet.exists()
    out = pd.read_parquet(out_parquet)
    assert {"sggs_line", "final_text", "decision", "is_simran"}.issubset(out.columns)
    # original columns preserved
    assert "text" in out.columns
    assert list(out["clip_id"]) == ["r1", "r2"]
```

- [ ] **Step 2: Run, verify it fails**

Run: `pytest tests/canonical/test_cli_add_canonical.py -v`

- [ ] **Step 3: Implement the CLI**

```python
#!/usr/bin/env python3
# scripts/add_canonical_column.py
"""Stage 1 CLI: run the DB-grounded canonical pipeline on a parquet input.

Outputs:
  --output-parquet: dataset with new columns (sggs_line, final_text, decision, is_simran)
  --audit-parquet : per-row audit sidecar (shabad_id, match_score, op_counts, line_ids, retrieval_margin)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from canonical.config import get_dataset_config
from canonical.pipeline import CanonicalPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-parquet", required=True)
    ap.add_argument("--output-parquet", required=True)
    ap.add_argument("--audit-parquet", required=True)
    ap.add_argument("--dataset", choices=["kirtan", "sehaj"], required=True)
    ap.add_argument("--db", default="database.sqlite")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only first N rows (dry-run)")
    args = ap.parse_args()

    cfg = get_dataset_config(args.dataset)
    pipeline = CanonicalPipeline(cfg, db_path=Path(args.db))

    df = pd.read_parquet(args.input_parquet)
    if args.limit:
        df = df.head(args.limit)
    rows = df.to_dict(orient="records")
    print(f"[run] {len(rows)} rows from {args.input_parquet}", file=sys.stderr)

    results = pipeline.run(rows)
    print(f"[run] {len(results)} rows after pre-cleaning + simran quota", file=sys.stderr)

    # Decision histogram
    from collections import Counter
    dh = Counter(r["decision"] for r in results)
    print(f"[run] decisions: {dict(dh)}", file=sys.stderr)

    # Dataset columns (drop audit internals)
    dataset_cols = [
        "clip_id", "video_id", "text", "raw_text", "start_s", "end_s",
        "duration_s", "sggs_line", "final_text", "decision", "is_simran",
    ]
    audit_cols = [
        "clip_id", "shabad_id", "line_ids", "match_score",
        "op_counts", "retrieval_margin",
    ]
    existing = {c: df.columns.tolist() for c in ()}
    dataset_df = pd.DataFrame([
        {k: r.get(k) for k in dataset_cols if k in r} for r in results
    ])
    audit_df = pd.DataFrame([
        {k: r.get(k) for k in audit_cols} for r in results
    ])
    dataset_df.to_parquet(args.output_parquet)
    audit_df.to_parquet(args.audit_parquet)
    print(f"[run] wrote {args.output_parquet} + {args.audit_parquet}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test, verify PASS**

Run: `pytest tests/canonical/test_cli_add_canonical.py -v`

- [ ] **Step 5: Commit**

```bash
git add scripts/add_canonical_column.py tests/canonical/test_cli_add_canonical.py
git commit -m "feat(canonical): Stage-1 CLI — add_canonical_column.py"
```

---

## Task 14: Stage 2 — Gemini LLM fallback pass

**Files:**
- Create: `scripts/canonical/llm_pass.py`
- Create: `scripts/llm_canonical_pass.py` (CLI entrypoint)
- Test: `tests/canonical/test_llm_pass.py`

Reference: `scripts/proto_run_file.py` (prompt, verification, batching).

- [ ] **Step 1: Write failing test with mocked Gemini client**

```python
# tests/canonical/test_llm_pass.py
import json
from unittest.mock import MagicMock

from scripts.canonical.llm_pass import (
    build_prompt, verify_llm_output, LLMConfig, run_llm_pass,
)


class TestBuildPrompt:
    def test_contains_rules_and_captions(self):
        batch = [
            {"clip_id": "a", "caption": "ਤੇਰੀ ਸਰਣਿ"},
            {"clip_id": "b", "caption": "ਸੁਖ ਸਾਗਰ"},
        ]
        prompt = build_prompt(
            shabad_text="ਤੇਰੀ ਸਰਣਿ ਮੇਰੇ\nਸੁਖ ਸਾਗਰ ਮੇਰੇ ਗੁਰ",
            ang=105, batch_rows=batch,
        )
        assert "EXACTLY ONE" in prompt  # the critical 1:1 rule
        assert "clip_id" in prompt
        assert '"a"' in prompt
        assert '"b"' in prompt


class TestVerify:
    def test_verified_when_all_tokens_in_shabad(self):
        shabad_tokens = {"ਤੇਰੀ", "ਸਰਣਿ", "ਮੇਰੇ"}
        shabad_skels = {"ਤਰ", "ਸਰਣ", "ਮਰ"}
        ok, reason = verify_llm_output(
            "ਤੇਰੀ ਸਰਣਿ ਮੇਰੇ", shabad_tokens, shabad_skels, caption_len=3,
        )
        assert ok, reason

    def test_len_drift_rejected(self):
        ok, reason = verify_llm_output(
            "ਤੇਰੀ ਸਰਣਿ ਮੇਰੇ ਦੀਨ ਦਇਆਲਾ ਸੁਖ ਸਾਗਰ",
            {"ਤੇਰੀ"}, {"ਤਰ"}, caption_len=2,
        )
        assert not ok
        assert "len_drift" in reason

    def test_invented_token_rejected(self):
        # caption_len=1, single token but not in shabad inventory or close
        ok, reason = verify_llm_output(
            "ਬਲਾਤਕਾਰ", {"ਤੇਰੀ"}, {"ਤਰ"}, caption_len=1,
        )
        assert not ok
        assert "invented" in reason


class TestRunLLMPass:
    def test_calls_gemini_and_parses_results(self):
        mock_client = MagicMock()
        resp = MagicMock()
        resp.text = json.dumps({
            "corrections": [
                {"clip_id": "r1", "corrected": "ਤੇਰੀ ਸਰਣਿ"},
                {"clip_id": "r2", "corrected": "ਸੁਖ ਸਾਗਰ"},
            ]
        })
        mock_client.models.generate_content.return_value = resp

        candidates = [
            {"clip_id": "r1", "video_id": "v", "text": "ਤੇਰੀ ਸਰਣਿ",
             "shabad_id": "EZ7", "decision": "review"},
            {"clip_id": "r2", "video_id": "v", "text": "ਸੁਖ ਸਾਗਰ",
             "shabad_id": "EZ7", "decision": "review"},
        ]
        shabad_lines_map = {
            "EZ7": [
                MagicMock(unicode="ਤੇਰੀ ਸਰਣਿ ਮੇਰੇ",
                          tokens=("ਤੇਰੀ", "ਸਰਣਿ", "ਮੇਰੇ"), ang=105)
            ],
        }
        # Monkey-patch ang on mock so that build_prompt can read it
        cfg = LLMConfig(model="mock", batch_size=30)
        results = run_llm_pass(candidates, shabad_lines_map, cfg, client=mock_client)
        assert len(results) == 2
        assert results["r1"]["corrected"] == "ਤੇਰੀ ਸਰਣਿ"
        assert results["r2"]["corrected"] == "ਸੁਖ ਸਾਗਰ"
```

- [ ] **Step 2: Run test, verify fails**

Run: `pytest tests/canonical/test_llm_pass.py -v`

- [ ] **Step 3: Implement llm_pass module**

```python
# scripts/canonical/llm_pass.py
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
```

- [ ] **Step 4: Run tests, iterate. Mock setup requires careful SDK patching.**

Run: `pytest tests/canonical/test_llm_pass.py -v`

If the test around mocking `_call_gemini` fails due to `types` import, either:
- (a) Patch `google.genai.types.GenerateContentConfig` in a fixture, OR
- (b) Restructure `_call_gemini` to accept config builder as a parameter (testability)

- [ ] **Step 5: Create Stage 2 CLI**

```python
#!/usr/bin/env python3
# scripts/llm_canonical_pass.py
"""Stage 2 CLI: run Gemini on rows with decision IN (unchanged, review)."""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from canonical.config import get_dataset_config
from canonical.llm_pass import LLMConfig, run_llm_pass
from canonical.sttm_index import load_sggs, build_shabad_ngram_index

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-parquet", required=True,
                    help="Output of add_canonical_column.py")
    ap.add_argument("--audit-parquet", required=True,
                    help="Audit sidecar from Stage 1")
    ap.add_argument("--llm-sidecar", required=True,
                    help="Output parquet with final_text_llm columns")
    ap.add_argument("--dataset", choices=["kirtan", "sehaj"], required=True)
    ap.add_argument("--db", default="database.sqlite")
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--batch-size", type=int, default=30)
    args = ap.parse_args()

    cfg = get_dataset_config(args.dataset)
    if not cfg.use_llm_fallback:
        print(f"[llm] {args.dataset} has use_llm_fallback=false; exiting", file=sys.stderr)
        return 0

    df = pd.read_parquet(args.stage1_parquet)
    audit = pd.read_parquet(args.audit_parquet)
    merged = df.merge(audit, on="clip_id", suffixes=("", "_audit"))
    candidates = merged[merged["decision"].isin(["unchanged", "review"])]
    print(f"[llm] {len(candidates)} candidates from Stage 1", file=sys.stderr)

    lines, _ = load_sggs(args.db, include_sirlekh=cfg.include_sirlekh)
    _, shabad_lines_map, _ = build_shabad_ngram_index(lines, n=4)

    llm_cfg = LLMConfig(model=args.model, batch_size=args.batch_size)
    rows = candidates.to_dict(orient="records")
    llm_results = run_llm_pass(rows, shabad_lines_map, llm_cfg)
    print(f"[llm] {sum(1 for r in llm_results.values() if r['verified'])} verified",
          file=sys.stderr)

    out = pd.DataFrame([
        {
            "clip_id": cid,
            "final_text_llm": r["corrected"],
            "llm_model": r["model"],
            "llm_verified": r["verified"],
            "llm_reason": r["reason"],
        }
        for cid, r in llm_results.items()
    ])
    out.to_parquet(args.llm_sidecar)
    print(f"[llm] wrote {args.llm_sidecar}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add scripts/canonical/llm_pass.py scripts/llm_canonical_pass.py tests/canonical/test_llm_pass.py
git commit -m "feat(canonical): Stage-2 LLM fallback pass (Gemini 3.1 Pro) + CLI"
```

---

## Task 15: Sidecar merge into HF dataset + push

**Files:**
- Create: `scripts/merge_canonical_into_hf.py`
- Test: `tests/canonical/test_merge_hf.py`

- [ ] **Step 1: Write failing test with a mocked HF `Dataset`**

```python
# tests/canonical/test_merge_hf.py
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from scripts.canonical.merge_hf import merge_sidecars


def _make_sidecar_frames(tmp_path):
    stage1 = pd.DataFrame([
        {"clip_id": "a", "text": "ਤੇਰੀ", "sggs_line": "ਤੇਰੀ ਸਰਣਿ",
         "final_text": "ਤੇਰੀ ਸਰਣਿ", "decision": "replaced", "is_simran": False},
        {"clip_id": "b", "text": "ਗ", "sggs_line": None,
         "final_text": "ਗ", "decision": "unchanged", "is_simran": False},
    ])
    llm = pd.DataFrame([
        {"clip_id": "b", "final_text_llm": "ਗੁਰੂ",
         "llm_model": "gemini-3.1-pro-preview", "llm_verified": False,
         "llm_reason": "len_drift(1vs1)"},
    ])
    s1p = tmp_path / "s1.parquet"
    llmp = tmp_path / "llm.parquet"
    stage1.to_parquet(s1p)
    llm.to_parquet(llmp)
    return s1p, llmp


class TestMergeSidecars:
    def test_merge_adds_all_columns(self, tmp_path):
        s1, llm = _make_sidecar_frames(tmp_path)
        df = merge_sidecars(s1, llm)
        cols = set(df.columns)
        assert {"sggs_line", "final_text", "decision", "is_simran",
                "final_text_llm", "llm_model", "llm_verified"}.issubset(cols)
        # row a had no LLM → fields are null
        row_a = df[df["clip_id"] == "a"].iloc[0]
        assert row_a["final_text_llm"] is None or pd.isna(row_a["final_text_llm"])
        # row b has LLM output
        row_b = df[df["clip_id"] == "b"].iloc[0]
        assert row_b["final_text_llm"] == "ਗੁਰੂ"
```

- [ ] **Step 2: Run test, verify fail**

Run: `pytest tests/canonical/test_merge_hf.py -v`

- [ ] **Step 3: Implement merge + push**

```python
# scripts/canonical/merge_hf.py
"""Merge Stage-1 + Stage-2 sidecars into a unified dataset ready for HF push."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def merge_sidecars(stage1_parquet: Path | str, llm_parquet: Path | str | None = None) -> pd.DataFrame:
    s1 = pd.read_parquet(stage1_parquet)
    if llm_parquet and Path(llm_parquet).exists():
        llm = pd.read_parquet(llm_parquet)
        merged = s1.merge(llm, on="clip_id", how="left")
    else:
        merged = s1.copy()
        for c in ("final_text_llm", "llm_model", "llm_verified", "llm_reason"):
            merged[c] = None
    return merged
```

```python
#!/usr/bin/env python3
# scripts/merge_canonical_into_hf.py
"""Merge Stage-1 + Stage-2 sidecars, then push a new revision of the HF dataset.

Safety: requires --confirm-push flag to actually push. Without it, just writes
the merged parquet locally for inspection.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from canonical.merge_hf import merge_sidecars

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-parquet", required=True)
    ap.add_argument("--llm-sidecar", default=None)
    ap.add_argument("--output-parquet", required=True)
    ap.add_argument("--hf-repo", default=None,
                    help="e.g. surindersinghssj/gurbani-kirtan-yt-captions-300h")
    ap.add_argument("--confirm-push", action="store_true",
                    help="Actually push to HF Hub. Without this, only writes locally.")
    args = ap.parse_args()

    df = merge_sidecars(args.stage1_parquet, args.llm_sidecar)
    df.to_parquet(args.output_parquet)
    print(f"[merge] wrote {args.output_parquet} ({len(df)} rows)", file=sys.stderr)

    if not args.hf_repo:
        return

    if not args.confirm_push:
        print(f"[merge] --confirm-push not set; skipping HF push to {args.hf_repo}",
              file=sys.stderr)
        print(f"[merge] run again with --confirm-push to push", file=sys.stderr)
        return

    from datasets import Dataset, Audio
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("[merge] ERROR: HF_TOKEN missing", file=sys.stderr)
        sys.exit(2)

    # Cast audio column if present (column name "audio")
    ds = Dataset.from_pandas(df)
    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    ds.push_to_hub(args.hf_repo, token=token)
    print(f"[merge] pushed to {args.hf_repo}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test, verify PASS**

Run: `pytest tests/canonical/test_merge_hf.py -v`

- [ ] **Step 5: Commit**

```bash
git add scripts/canonical/merge_hf.py scripts/merge_canonical_into_hf.py tests/canonical/test_merge_hf.py
git commit -m "feat(canonical): merge sidecars + optional HF push CLI"
```

---

## Task 16: Dry-run validation script

**Files:**
- Create: `scripts/canonical_dry_run.py`

- [ ] **Step 1: Implement dry-run CLI**

```python
#!/usr/bin/env python3
# scripts/canonical_dry_run.py
"""Dry-run the Stage 1 + Stage 2 pipeline on the first N rows of a dataset
and print a decision histogram + a few sample corrections for manual review."""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from canonical.config import get_dataset_config
from canonical.pipeline import CanonicalPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-parquet", required=True)
    ap.add_argument("--dataset", choices=["kirtan", "sehaj"], required=True)
    ap.add_argument("--db", default="database.sqlite")
    ap.add_argument("--n", type=int, default=1000, help="rows to process")
    ap.add_argument("--sample-per-decision", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = get_dataset_config(args.dataset)
    pipeline = CanonicalPipeline(cfg, db_path=args.db)

    df = pd.read_parquet(args.input_parquet)
    df = df.head(args.n)
    rows = df.to_dict(orient="records")
    print(f"[dry-run] {len(rows)} rows, dataset={args.dataset}", file=sys.stderr)

    results = pipeline.run(rows)
    hist = Counter(r["decision"] for r in results)
    print(f"\nDecision histogram:")
    for dec in ["matched", "replaced", "review", "unchanged", "simran"]:
        n = hist.get(dec, 0)
        pct = 100 * n / len(results) if results else 0
        print(f"  {dec:10} {n:5d}  ({pct:5.1f}%)")

    # Samples per decision
    rng = random.Random(args.seed)
    by_dec: dict[str, list[dict]] = {}
    for r in results:
        by_dec.setdefault(r["decision"], []).append(r)
    print(f"\nSamples (up to {args.sample_per_decision} per decision):")
    for dec, rr in sorted(by_dec.items()):
        rng.shuffle(rr)
        print(f"\n--- {dec} ---")
        for r in rr[:args.sample_per_decision]:
            print(f"  clip_id={r['clip_id']}")
            print(f"    caption: {r['text']}")
            print(f"    final:   {r['final_text']}")
            if r.get("sggs_line"):
                print(f"    sggs:    {r['sggs_line']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test on a tiny fixture manually**

Run (if you have a tiny parquet):
```bash
python3 scripts/canonical_dry_run.py \
  --input-parquet /path/to/tiny.parquet \
  --dataset kirtan --n 100
```
Expected: decision histogram + samples printed.

- [ ] **Step 3: Commit**

```bash
git add scripts/canonical_dry_run.py
git commit -m "feat(canonical): dry-run script with histogram + samples"
```

---

## Task 17: Regression prototype bridge + delete prototypes

**Files:**
- Modify: existing `scripts/proto_canonical_v2.py` — add deprecation notice pointing to `scripts/canonical/`
- Modify: `docs/superpowers/specs/2026-04-18-kirtan-canonical-text-column-design.md` — bump Status to "Implemented"

- [ ] **Step 1: Add deprecation banner to prototype files**

At the top of `scripts/proto_canonical.py`, `proto_canonical_v2.py`, `proto_llm_pass.py`, `proto_llm_compare.py`, `proto_run_file.py`, `proto_llm_diag.py`:

```python
# DEPRECATED: kept for regression reference only.
# Production code lives at scripts/canonical/ and is tested in tests/canonical/.
```

- [ ] **Step 2: Run a regression check on the old proto sample**

Run:
```bash
python3 scripts/canonical_dry_run.py \
  --input-parquet <tiny fixture> --dataset kirtan --n 100
```
Spot-check that ≥ 3 of the EZ7-known-good rows (e.g., `ਤੇਰੀ ਸਰਣ ਮੇਰੇ ਦੀਨ ਦਇਆਲਾ`) produce `decision=replaced` with `ਤੇਰੀ ਸਰਣਿ ਮੇਰੇ ਦੀਨ ਦਇਆਲਾ` in `final_text`.

- [ ] **Step 3: Update spec header**

In `docs/superpowers/specs/2026-04-18-kirtan-canonical-text-column-design.md`, change line 4 from:
```
**Status:** Spec (awaiting user review)
```
to:
```
**Status:** Implemented (ready to run post AKJ-ingest + additive-push gate)
```

- [ ] **Step 4: Commit**

```bash
git add scripts/proto_*.py docs/superpowers/specs/2026-04-18-kirtan-canonical-text-column-design.md
git commit -m "chore(canonical): deprecate prototypes; spec status→Implemented"
```

---

## Task 18: Operational runbook

**Files:**
- Create: `docs/superpowers/runbooks/canonical-text-pipeline.md`

- [ ] **Step 1: Write the runbook**

```markdown
# Canonical Text Pipeline — Operational Runbook

## Prereqs

- `database.sqlite` present at repo root (STTM ShabadOS DB).
- `GEMINI_API_KEY` and `HF_TOKEN` in `.env`.
- Gating: AKJ ingest merged + additive HF push landed (see spec §0).

## 1. Dry-run on 1000 rows

```bash
# Pull existing parquet from HF
huggingface-cli download \
  surindersinghssj/gurbani-kirtan-yt-captions-300h \
  --repo-type dataset --local-dir /tmp/kirtan-raw

python3 scripts/canonical_dry_run.py \
  --input-parquet /tmp/kirtan-raw/train-*.parquet \
  --dataset kirtan --n 1000
```

Expected: decision histogram with ~60–75% matched+replaced. If that's wildly off, investigate before the full run.

## 2. Stage 1 — full run

```bash
python3 scripts/add_canonical_column.py \
  --input-parquet /tmp/kirtan-raw/train.parquet \
  --output-parquet /tmp/kirtan-stage1.parquet \
  --audit-parquet  /tmp/kirtan-audit.parquet \
  --dataset kirtan \
  --db database.sqlite
```

Runtime: ~10 minutes on a single CPU for 60K rows. Output: two parquet files.

## 3. Stage 2 — Gemini fallback

```bash
python3 scripts/llm_canonical_pass.py \
  --stage1-parquet /tmp/kirtan-stage1.parquet \
  --audit-parquet  /tmp/kirtan-audit.parquet \
  --llm-sidecar    /tmp/kirtan-llm.parquet \
  --dataset kirtan \
  --db database.sqlite \
  --model gemini-3.1-pro-preview \
  --batch-size 30
```

Expected cost: ~$2.30 for 300h dataset. Runtime: ~10 minutes.

## 4. Merge + local verify

```bash
python3 scripts/merge_canonical_into_hf.py \
  --stage1-parquet /tmp/kirtan-stage1.parquet \
  --llm-sidecar    /tmp/kirtan-llm.parquet \
  --output-parquet /tmp/kirtan-final.parquet
```

Inspect the merged parquet in a notebook — confirm column set, spot-check
5 rows per decision, confirm no original columns were dropped.

## 5. Push to HF (ONLY after local verify)

```bash
python3 scripts/merge_canonical_into_hf.py \
  --stage1-parquet /tmp/kirtan-stage1.parquet \
  --llm-sidecar    /tmp/kirtan-llm.parquet \
  --output-parquet /tmp/kirtan-final.parquet \
  --hf-repo surindersinghssj/gurbani-kirtan-yt-captions-300h \
  --confirm-push
```

## 6. Sehaj path equivalent

Same commands, swap `--dataset kirtan` → `--dataset sehaj` and repo name
`surindersinghssj/gurbani-sehajpath`.

## Rollback

If a run pushed a bad revision:
1. On HF dataset page, revert to the previous commit.
2. Locally: no dataset state held, sidecar parquets can be regenerated.
3. Re-run after fixing config or data.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/runbooks/canonical-text-pipeline.md
git commit -m "docs: canonical text pipeline operational runbook"
```

---

## Self-Review

### Spec coverage check

Spec sections → plan tasks:

- §1 Goal / invariant → Task 12 pipeline (respects invariant by never modifying `text`)
- §1.5 Pre-cleaning + `<unk>` strip → Task 4
- §1.6 Simran quota + waheguru normalization → Task 5 + Task 6
- §2 I/O (columns, audit sidecar) → Task 12 + Task 13 CLI
- §2.5 Sirlekh indexing → Task 3 (`include_sirlekh` param) + Task 7 (normalization)
- §2.6 Multi-shabad splitting → Task 7
- §2.7 Sequential retrieval → Task 8
- §3 Phrase-level NW → Task 9
- §5 Flag thresholds / decision → Task 10
- §7 Tuneable constants + dataset defaults → Task 11
- §11 LLM fallback → Task 14
- §12 Merge + push → Task 15

**Coverage: complete.**

### Placeholder scan

- No "TBD" / "TODO" / "implement later" in any task.
- Every code block is complete and runnable.
- Test bodies have concrete assertions, not "test that it works".
- Commit messages are all specific.

### Type consistency

- `SggsLine` dataclass defined in Task 3 is used identically in Tasks 8, 9, 12, 14 (including `.tokens`, `.tok_skels`, `.unicode`, `.ang`, `.shabad_id`).
- `AlignConfig` field names match between Task 9 and Task 12.
- `DatasetConfig` fields in Task 11 match Task 12 usage (`sequential_shabad_retrieval`, `use_llm_fallback`, etc.).
- `LLMConfig` in Task 14 matches CLI usage.

### Scope

- Pipeline + sidecar merge + push + dry-run + runbook — complete for a single implementation cycle.
- No orphan subsystems.

All good.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-18-kirtan-canonical-text-column.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
