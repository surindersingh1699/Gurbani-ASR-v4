"""Offline STTM Desktop controller via Chrome DevTools Protocol (CDP).

STTM Desktop is an Electron app. When launched with
`--remote-debugging-port=9222` it exposes its renderer over CDP on
localhost:9222, which Playwright can attach to without launching a new
Chromium. This module drives the renderer directly — no cloud socket.io,
no unverified REST guesses, no sync code, no PIN. Everything is same-
machine and works with Wi-Fi off.

Previous iteration of this module POSTed to an unverified REST path on
a probe list of ports. That endpoint was a catch-all no-op on STTM's
overlay server and was silently dropped; `push_hit` would return
HTTP 200 while the highlight never moved. That code path is gone.

BaniDB helpers (`search_shabad_topn`, `search_shabad`, `_norm_hit`) are
preserved because the suggest / autolock paths in `app.py` still call
them.
"""

from __future__ import annotations

import difflib
import json
import threading
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

BANIDB_SEARCH = "https://api.banidb.com/v2/search/{q}?source=G&searchtype=0"
BANIDB_SHABAD = "https://api.banidb.com/v2/shabads/{id}"

# Default CDP port we tell users to launch STTM with.
DEFAULT_CDP_PORT = 9222


@dataclass
class STTMStatus:
    """Return shape for push operations.

    Preserved from the old REST module so call sites that read
    `res.ok` / `res.detail` don't need to change. `host` is now a
    constant-ish "cdp" marker and `port` is the CDP port (or None when
    not connected), which keeps any HTML status renderer that reads
    `.port` happy.
    """

    ok: bool
    host: str
    port: Optional[int]
    detail: str


# --- BaniDB helpers (unchanged from prior module) ---------------------------


def _get(url: str, timeout: float = 2.5) -> tuple[int, bytes]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.status, resp.read()


def _norm_hit(hit: dict, query: str) -> dict:
    """Flatten a BaniDB hit to a stable shape with a similarity score."""
    gurmukhi = (
        hit.get("verse")
        or hit.get("gurmukhi")
        or (hit.get("verse", {}) if isinstance(hit.get("verse"), dict) else {}).get("gurmukhi")
        or ""
    )
    if isinstance(gurmukhi, dict):
        gurmukhi = gurmukhi.get("gurmukhi") or gurmukhi.get("unicode") or ""
    writer = ((hit.get("writer") or {}) if isinstance(hit.get("writer"), dict) else {}).get("english") or \
             ((hit.get("writer") or {}) if isinstance(hit.get("writer"), dict) else {}).get("writerEnglish") or \
             hit.get("writerEnglish") or ""
    raag = ((hit.get("raag") or {}) if isinstance(hit.get("raag"), dict) else {}).get("english") or \
           hit.get("raagEnglish") or ""
    source = ((hit.get("source") or {}) if isinstance(hit.get("source"), dict) else {}).get("english") or \
             hit.get("sourceEnglish") or ""
    ang = hit.get("pageNo") or hit.get("ang") or hit.get("angNo") or ""
    shabad_id = hit.get("shabadId") or hit.get("shabadID") or hit.get("shabad_id")
    verse_id = hit.get("verseId") or hit.get("verseID") or hit.get("verse_id") or shabad_id

    score = 0.0
    if gurmukhi and query:
        score = difflib.SequenceMatcher(a=query.strip(), b=gurmukhi.strip()).ratio()

    return {
        "shabadId": shabad_id,
        "verseId": verse_id,
        "gurmukhi": gurmukhi,
        "writer": writer,
        "raag": raag,
        "source": source,
        "ang": ang,
        "score": round(score, 3),
    }


def search_shabad_topn(query: str, n: int = 5) -> list[dict]:
    """Return up to `n` BaniDB search hits ranked by SequenceMatcher similarity."""
    query = (query or "").strip()
    if not query:
        return []
    try:
        url = BANIDB_SEARCH.format(q=urllib.parse.quote(query))
        status, body = _get(url, timeout=4.0)
        if status != 200:
            return []
        data = json.loads(body)
        hits = data.get("verses") or data.get("shabads") or []
    except Exception:  # noqa: BLE001
        return []

    normalized = [_norm_hit(h, query) for h in hits[: max(n * 2, n)]]
    normalized = [h for h in normalized if h["shabadId"]]
    normalized.sort(key=lambda h: h["score"], reverse=True)
    return normalized[:n]


def search_shabad(query: str) -> Optional[dict]:
    hits = search_shabad_topn(query, n=1)
    return hits[0] if hits else None


# --- first-letter helper ----------------------------------------------------

# Gurmukhi block: consonants (ਕ-ਵ range) + a handful of commonly-indexed
# additions (ਸ਼, ਖ਼, ਗ਼, ਜ਼, ਫ਼, ੜ). STTM's first-letter search takes the first
# akhar of each space-separated token.
_GURMUKHI_CONSONANTS = set(
    chr(cp) for cp in range(0x0A15, 0x0A39)
) | {"\u0A36", "\u0A59", "\u0A5A", "\u0A5B", "\u0A5E", "\u0A5C"}


def _first_letters(gurmukhi: str) -> str:
    """Return STTM-style first-letter query for a Gurmukhi line.

    e.g. "ਸਤਿਗੁਰ ਪ੍ਰਸਾਦਿ" -> "ਸਪ". Non-Gurmukhi tokens are skipped.
    """
    out: list[str] = []
    for token in (gurmukhi or "").split():
        for ch in token:
            if ch in _GURMUKHI_CONSONANTS:
                out.append(ch)
                break
    return "".join(out)


# --- CDP controller ---------------------------------------------------------


class STTMController:
    """Drive STTM Desktop's renderer via Playwright's sync CDP client.

    The controller is designed to be a long-lived singleton — see
    `get_controller()`. `connect()` is idempotent and swallows errors;
    callers should check `is_connected()` (or the returned bool) before
    assuming success. `push_hit()` is the one-stop call site for the
    Gradio handlers: it picks between `open_shabad` (first push for a
    shabad) and `advance_to_verse` (subsequent within-shabad pushes) and
    always returns an `STTMStatus`.
    """

    def __init__(self, cdp_port: int = DEFAULT_CDP_PORT) -> None:
        self._cdp_port = cdp_port
        self._pw = None
        self._browser = None
        self._page = None
        self._active_shabad_id: Optional[int] = None
        self._active_line_idx: int = 0
        self._last_error: str = ""

    # ---- lifecycle ----

    def connect(self, cdp_port: Optional[int] = None) -> bool:
        """Attach to an existing STTM renderer over CDP.

        Does NOT launch a new browser — STTM must already be running
        with `--remote-debugging-port=<cdp_port>`. Returns False (and
        records `_last_error`) on any failure; never raises, so Gradio
        handlers can call this from any thread without a try/except.
        """
        if cdp_port is not None:
            self._cdp_port = cdp_port
        if self.is_connected():
            return True
        try:
            # Import lazily so the module stays importable even when
            # Playwright hasn't been installed yet (e.g. in test shells
            # or on CI that doesn't exercise the push path).
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            self._last_error = (
                f"playwright not installed: {e}. "
                "Run `pip install -r apps/transcribe/requirements.txt`."
            )
            return False
        try:
            self._pw = sync_playwright().start()
            cdp_url = f"http://127.0.0.1:{self._cdp_port}"
            self._browser = self._pw.chromium.connect_over_cdp(cdp_url)
            contexts = self._browser.contexts
            if not contexts or not contexts[0].pages:
                self._last_error = (
                    "CDP connected but no pages found — is STTM's main "
                    "window open?"
                )
                self._teardown_quietly()
                return False
            # STTM Desktop usually has a single renderer page. If there
            # are multiple we pick the first non-blank one.
            page = None
            for p in contexts[0].pages:
                try:
                    url = p.url or ""
                except Exception:  # noqa: BLE001
                    url = ""
                if url and url != "about:blank":
                    page = p
                    break
            self._page = page or contexts[0].pages[0]
            self._last_error = ""
            return True
        except Exception as e:  # noqa: BLE001
            self._last_error = (
                f"CDP not reachable on :{self._cdp_port} ({e}). "
                f"Launch STTM with "
                f"`open -a 'SikhiToTheMax' --args --remote-debugging-port={self._cdp_port}`."
            )
            self._teardown_quietly()
            return False

    def is_connected(self) -> bool:
        if self._page is None or self._browser is None:
            return False
        # Playwright raises if the underlying connection is dead; treat
        # that as "not connected" rather than propagating.
        try:
            return self._browser.is_connected()
        except Exception:  # noqa: BLE001
            return False

    def disconnect(self) -> None:
        self._teardown_quietly()

    def _teardown_quietly(self) -> None:
        for obj_name in ("_browser", "_pw"):
            obj = getattr(self, obj_name, None)
            if obj is None:
                continue
            try:
                if obj_name == "_browser":
                    obj.close()
                else:
                    obj.stop()
            except Exception:  # noqa: BLE001
                pass
            setattr(self, obj_name, None)
        self._page = None

    def last_error(self) -> str:
        return self._last_error

    # ---- navigation primitives ----

    def open_shabad(self, shabad_id: int, first_line_gurmukhi: str) -> bool:
        """Type first-letters into STTM's search box and pick the top hit.

        STTM's search UI uses first-letter matching on the Gurmukhi
        akhars. We type e.g. "ਸਪ" for "ਸਤਿਗੁਰ ਪ੍ਰਸਾਦਿ" and click the
        first result row. Selectors are deliberately forgiving — Task 3
        will pin down the real ones against the running STTM build.
        """
        if not self.is_connected():
            return False
        query = _first_letters(first_line_gurmukhi)
        if not query:
            return False
        search_selectors = [
            "input.search-field",
            "input#search-field",
            "input.search",
            "input#search",
            "input[type='search']",
            "input[placeholder*='Search' i]",
        ]
        result_selectors = [
            ".search-results .result-row",
            ".search-result",
            ".search-result-row",
            "li.result",
            "[data-shabadid]",
        ]
        try:
            search_box = None
            for sel in search_selectors:
                loc = self._page.locator(sel).first
                if loc.count() > 0:
                    search_box = loc
                    break
            if search_box is None:
                return False
            search_box.click(timeout=2000)
            search_box.fill("")
            search_box.type(query, delay=40)
            # Give STTM a moment to populate the result list.
            self._page.wait_for_timeout(350)
            for sel in result_selectors:
                results = self._page.locator(sel)
                if results.count() > 0:
                    results.first.click(timeout=2000)
                    self._active_shabad_id = int(shabad_id)
                    self._active_line_idx = 0
                    return True
        except Exception as e:  # noqa: BLE001
            self._last_error = f"open_shabad failed: {e}"
        return False

    def advance_to_verse(
        self,
        verse_id: int,
        line_idx: int,
        full_rowids: list[int],  # noqa: ARG002 (kept for call-site parity)
        gurmukhi: str,  # noqa: ARG002 (kept for call-site parity)
    ) -> bool:
        """Jump STTM's highlight to a specific pangti within the open shabad.

        Strategy: try known data-attr selectors on the verse element, then
        fall back to keyboard arrows from the last known line index.
        STTM's exact DOM attribute varies across builds, so the selector
        list is forgiving.
        """
        if not self.is_connected():
            return False
        selectors = [
            f"[data-verseid='{verse_id}']",
            f"[data-verse-id='{verse_id}']",
            f"[data-line-index='{line_idx}']",
            f"[data-line='{line_idx}']",
            f"[data-idx='{line_idx}']",
        ]
        for sel in selectors:
            try:
                loc = self._page.locator(sel).first
                if loc.count() > 0:
                    loc.click(timeout=2000)
                    self._active_line_idx = int(line_idx)
                    return True
            except Exception:  # noqa: BLE001
                continue

        # Fallback: press ArrowDown/Up the right number of times from
        # wherever we currently are. Only works if STTM already has a
        # shabad open (which it does by the time advance_to_verse is
        # called from push_hit).
        try:
            delta = int(line_idx) - int(self._active_line_idx)
            if delta == 0:
                return True
            key = "ArrowDown" if delta > 0 else "ArrowUp"
            for _ in range(abs(delta)):
                self._page.keyboard.press(key)
            self._active_line_idx = int(line_idx)
            return True
        except Exception as e:  # noqa: BLE001
            self._last_error = f"advance_to_verse fallback failed: {e}"
            return False

    # ---- high-level push ----

    def push_hit(self, hit: dict) -> STTMStatus:
        """Open the shabad (if new) or advance the highlight (if same).

        Returns an `STTMStatus` whose `.detail` starts with
        "pushed shabad=… verse=…" on success — the existing UI readouts
        in app.py show this string verbatim in the side panel.
        """
        sid_raw = hit.get("shabadId")
        if not sid_raw:
            return STTMStatus(False, "cdp", self._cdp_port, "hit missing shabadId")
        try:
            sid = int(sid_raw)
        except (TypeError, ValueError):
            return STTMStatus(False, "cdp", self._cdp_port, f"bad shabadId: {sid_raw!r}")
        vid_raw = hit.get("verseId") or sid_raw
        try:
            vid = int(vid_raw)
        except (TypeError, ValueError):
            vid = sid

        if not self.is_connected():
            # Try a lazy reconnect so the first push after launching the
            # app (before any UI-driven connect) still lands.
            if not self.connect():
                return STTMStatus(False, "cdp", None, self._last_error or "STTM not reachable")

        full_rowids = hit.get("full_rowids") or []
        try:
            highlight_idx = int(hit.get("highlight_idx", -1))
        except (TypeError, ValueError):
            highlight_idx = -1
        line_count = 1
        if full_rowids and 0 <= highlight_idx < len(full_rowids):
            line_count = highlight_idx + 1

        same_shabad = self._active_shabad_id == sid
        gurmukhi = hit.get("gurmukhi") or ""

        if not same_shabad:
            ok = self.open_shabad(sid, gurmukhi)
            if not ok:
                return STTMStatus(
                    False, "cdp", self._cdp_port,
                    self._last_error or f"open_shabad failed for sid={sid}",
                )
            # After opening, if the pointer is past line 1 we still need
            # to advance.
            if highlight_idx > 0:
                self.advance_to_verse(vid, highlight_idx, list(full_rowids), gurmukhi)
        else:
            line_idx = highlight_idx if highlight_idx >= 0 else 0
            ok = self.advance_to_verse(vid, line_idx, list(full_rowids), gurmukhi)
            if not ok:
                return STTMStatus(
                    False, "cdp", self._cdp_port,
                    self._last_error or f"advance_to_verse failed for verse={vid}",
                )

        return STTMStatus(
            True, "cdp", self._cdp_port,
            f"pushed shabad={sid} verse={vid} line={line_count}",
        )


# --- module-level singleton -------------------------------------------------

_controller_lock = threading.Lock()
_controller: Optional[STTMController] = None


def get_controller(cdp_port: int = DEFAULT_CDP_PORT) -> STTMController:
    """Return the process-wide `STTMController`, creating it on first call.

    Thread-safe — Gradio may invoke handlers from multiple workers. The
    controller's `connect()` is called lazily here so the Gradio app
    launches successfully even when STTM isn't running yet.
    """
    global _controller
    with _controller_lock:
        if _controller is None:
            _controller = STTMController(cdp_port=cdp_port)
        # Attempt (re)connect if not already attached. connect() is a
        # no-op when already connected, and never raises.
        if not _controller.is_connected():
            _controller.connect(cdp_port=cdp_port)
        return _controller
