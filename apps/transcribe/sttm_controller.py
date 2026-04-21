"""Minimal client for the SikhiToTheMax Desktop controller API + BaniDB search.

STTM Desktop exposes a local Express server (in Bani Controller mode).
Protocol: HTTP POST `/api/bani-control` with a JSON payload. Ports vary
across builds, so we probe a short list.

BaniDB is used to resolve a Gurmukhi transcript into a concrete shabad.

Reference: github.com/surindersingh1699/sttm-automate (src/controller/sttm_http.py).
"""

from __future__ import annotations

import difflib
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

CANDIDATE_PORTS = (8000, 42424, 43434, 8022, 8080)
BANIDB_SEARCH = "https://api.banidb.com/v2/search/{q}?source=G&searchtype=0"
BANIDB_SHABAD = "https://api.banidb.com/v2/shabads/{id}"


@dataclass
class STTMStatus:
    ok: bool
    host: str
    port: Optional[int]
    detail: str


def _get(url: str, timeout: float = 2.5) -> tuple[int, bytes]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.status, resp.read()


def _post_json(url: str, payload: dict, timeout: float = 2.5) -> tuple[int, bytes]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.status, resp.read()


def discover(host: str = "127.0.0.1", ports=CANDIDATE_PORTS) -> STTMStatus:
    for p in ports:
        try:
            status, _ = _get(f"http://{host}:{p}", timeout=1.0)
            if status == 200:
                return STTMStatus(True, host, p, f"STTM reachable on :{p}")
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            continue
    return STTMStatus(False, host, None, "STTM not reachable — is Bani Controller enabled?")


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


def push_shabad(
    host: str,
    port: int,
    shabad_id: int,
    verse_id: int,
    line_count: int = 1,
    pin: Optional[str] = None,
) -> STTMStatus:
    payload: dict = {
        "type": "shabad",
        "shabadId": int(shabad_id),
        "id": int(shabad_id),
        "verseId": int(verse_id),
        "lineCount": int(line_count),
        "highlight": int(verse_id),
        "homeId": int(verse_id),
    }
    if pin:
        payload["pin"] = str(pin)
    try:
        status, _ = _post_json(
            f"http://{host}:{port}/api/bani-control", payload, timeout=3.0
        )
        if 200 <= status < 300:
            return STTMStatus(True, host, port, f"pushed shabad {shabad_id}")
        return STTMStatus(False, host, port, f"http {status}")
    except Exception as e:  # noqa: BLE001
        return STTMStatus(False, host, port, f"error: {e}")


def push_hit(host: str, port: int, hit: dict, pin: Optional[str] = None) -> STTMStatus:
    sid = hit.get("shabadId")
    vid = hit.get("verseId") or sid
    if not sid:
        return STTMStatus(False, host, port, "hit missing shabadId")
    return push_shabad(host, port, int(sid), int(vid), pin=pin)


def push_transcript_as_shabad(
    host: str, port: int, text: str, pin: Optional[str] = None
) -> STTMStatus:
    hits = search_shabad_topn(text, n=1)
    if not hits:
        return STTMStatus(False, host, port, "no BaniDB match for transcript")
    return push_hit(host, port, hits[0], pin=pin)
