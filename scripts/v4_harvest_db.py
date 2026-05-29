#!/usr/bin/env python3
"""SQLite provenance layer for the v4 YouTube auto-caption harvest.

Sits alongside the existing JSONL manifests written by
`pilot_yt_caption_chunks.py`. Every video, every clip, and every Gemini
alignment-check sample is recorded so any subset of the dataset can be
re-derived later by SQL filter without re-running the pipeline.

Tables:
  videos              one row per video the harvester touches
  clips               one row per clip in each video's manifest
  alignment_checks    one row per Gemini sample-aligner call

Used by:
  scripts/bulk_yt_caption_pipeline.py   (writes during harvest)
  scripts/v4_align_check_gemini.py      (writes alignment results)
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable


SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
  video_id           TEXT PRIMARY KEY,
  channel_slug       TEXT,
  channel_url        TEXT,
  title              TEXT,
  upload_date        TEXT,
  duration_s         INTEGER,
  caption_status     TEXT,
  chunker_status     TEXT,
  alignment_matches  INTEGER,
  alignment_total    INTEGER,
  alignment_passed   INTEGER,
  status             TEXT,
  rejection_reason   TEXT,
  harvested_at       TEXT,
  pushed_at          TEXT
);

CREATE TABLE IF NOT EXISTS clips (
  clip_id           TEXT PRIMARY KEY,
  video_id          TEXT NOT NULL,
  start_s           REAL,
  end_s             REAL,
  duration_s        REAL,
  raw_text          TEXT,
  normalized_text   TEXT,
  status            TEXT,
  FOREIGN KEY (video_id) REFERENCES videos(video_id)
);

CREATE TABLE IF NOT EXISTS alignment_checks (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  video_id          TEXT NOT NULL,
  start_s           REAL,
  end_s             REAL,
  caption_text      TEXT,
  gemini_text       TEXT,
  caption_first_3w  TEXT,
  gemini_first_3w   TEXT,
  matched           INTEGER,
  checked_at        TEXT
);

CREATE INDEX IF NOT EXISTS idx_videos_status   ON videos(status);
CREATE INDEX IF NOT EXISTS idx_videos_channel  ON videos(channel_slug);
CREATE INDEX IF NOT EXISTS idx_clips_video     ON clips(video_id);
CREATE INDEX IF NOT EXISTS idx_clips_status    ON clips(status);
CREATE INDEX IF NOT EXISTS idx_align_video     ON alignment_checks(video_id);
"""


def _now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def upsert_video(conn: sqlite3.Connection, video_id: str, **fields: Any) -> None:
    fields.setdefault("harvested_at", _now())
    cur = conn.execute("SELECT video_id FROM videos WHERE video_id = ?", (video_id,))
    if cur.fetchone() is None:
        cols = ["video_id"] + list(fields.keys())
        placeholders = ",".join("?" * len(cols))
        conn.execute(
            f"INSERT INTO videos ({','.join(cols)}) VALUES ({placeholders})",
            [video_id] + list(fields.values()),
        )
    else:
        if fields:
            set_clause = ",".join(f"{k}=?" for k in fields)
            conn.execute(
                f"UPDATE videos SET {set_clause} WHERE video_id = ?",
                list(fields.values()) + [video_id],
            )
    conn.commit()


def get_video(conn: sqlite3.Connection, video_id: str) -> sqlite3.Row | None:
    cur = conn.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,))
    return cur.fetchone()


def record_clips_from_manifest(
    conn: sqlite3.Connection, video_id: str, manifest_fp: Path
) -> int:
    """Read pilot_yt_caption_chunks.py's manifest.jsonl, insert one row per clip.
    Returns the number of rows inserted. Idempotent on clip_id."""
    if not manifest_fp.exists():
        return 0
    inserted = 0
    for line in manifest_fp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        clip_id = m.get("clip_id") or f"{video_id}_{m.get('start_s', 0):.2f}"
        try:
            conn.execute(
                """INSERT OR IGNORE INTO clips
                   (clip_id, video_id, start_s, end_s, duration_s,
                    raw_text, normalized_text, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    clip_id,
                    video_id,
                    m.get("start_s"),
                    m.get("end_s"),
                    m.get("duration_s",
                          (m.get("end_s", 0) or 0) - (m.get("start_s", 0) or 0)),
                    m.get("raw_text", ""),
                    m.get("text", ""),
                    "kept",
                ),
            )
            inserted += 1
        except sqlite3.Error:
            continue
    conn.commit()
    return inserted


def record_alignment_check(
    conn: sqlite3.Connection,
    video_id: str,
    start_s: float,
    end_s: float,
    caption_text: str,
    gemini_text: str,
    caption_first_3w: str,
    gemini_first_3w: str,
    matched: bool,
) -> None:
    conn.execute(
        """INSERT INTO alignment_checks
           (video_id, start_s, end_s, caption_text, gemini_text,
            caption_first_3w, gemini_first_3w, matched, checked_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (video_id, start_s, end_s, caption_text, gemini_text,
         caption_first_3w, gemini_first_3w, int(matched), _now()),
    )
    conn.commit()


def mark_video_aligned(
    conn: sqlite3.Connection, video_id: str, matches: int, total: int,
    threshold: float
) -> bool:
    """Update alignment fields and set status. Returns True iff passed."""
    passed = total > 0 and (matches / total) >= threshold
    conn.execute(
        """UPDATE videos
              SET alignment_matches = ?,
                  alignment_total   = ?,
                  alignment_passed  = ?,
                  status            = ?,
                  rejection_reason  = ?
            WHERE video_id = ?""",
        (
            matches, total, int(passed),
            "aligned" if passed else "rejected",
            None if passed else
                f"misaligned: {matches}/{total} < {threshold:.2f}",
            video_id,
        ),
    )
    conn.commit()
    return passed


def mark_video_pushed(conn: sqlite3.Connection, video_ids: Iterable[str]) -> None:
    ts = _now()
    conn.executemany(
        "UPDATE videos SET status='pushed', pushed_at=? WHERE video_id=?",
        [(ts, vid) for vid in video_ids],
    )
    conn.commit()


def aligned_video_ids(conn: sqlite3.Connection) -> set[str]:
    cur = conn.execute(
        "SELECT video_id FROM videos WHERE status IN ('aligned', 'pushed')"
    )
    return {row["video_id"] for row in cur.fetchall()}


def summary(conn: sqlite3.Connection) -> dict[str, Any]:
    cur = conn.execute(
        """SELECT status, COUNT(*) AS n, SUM(duration_s) AS dur
             FROM videos GROUP BY status"""
    )
    by_status = {row["status"] or "queued": (row["n"], row["dur"] or 0)
                 for row in cur.fetchall()}
    cur = conn.execute("SELECT COUNT(*) AS n, SUM(duration_s) AS dur FROM clips "
                       "WHERE status='kept'")
    clip_row = cur.fetchone()
    return {
        "videos_by_status": by_status,
        "kept_clips": clip_row["n"] or 0,
        "kept_clip_hours": (clip_row["dur"] or 0) / 3600.0,
    }


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description="Inspect a v4 harvest SQLite DB.")
    ap.add_argument("db", type=Path)
    args = ap.parse_args()
    conn = connect(args.db)
    s = summary(conn)
    print(json.dumps(s, indent=2, default=str))
