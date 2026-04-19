#!/usr/bin/env python3
"""Build unified canonical text corpus from ShabadOS database.sqlite.

Exports lines from sources 1-4 (SGGS, Dasam Granth, Vaaran Bhai Gurdas,
Kabit Savaiye Bhai Gurdas) to a single JSON file, converting the ASCII
Gurmukhi encoding used in the DB to Unicode Gurmukhi.

Output schema:
    [{
        "line_id": "...",
        "source": "sggs" | "dasam" | "bhai_gurdas_vaaran" | "bhai_gurdas_kabit",
        "shabad_id": "...",
        "ang": int,
        "text": "...",         (Unicode Gurmukhi, vishraams stripped)
        "raag": "...",         (SGGS only, else "")
        "writer": "..."
    }, ...]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gurmukhi_converter import ascii_to_unicode, strip_vishraams  # noqa: E402

SOURCE_LABELS = {
    1: "sggs",
    2: "dasam",
    3: "bhai_gurdas_vaaran",
    4: "bhai_gurdas_kabit",
}
# Pankti (4), Rahao (3), Manglacharan (1) — excludes Sirlekh (headers)
CONTENT_TYPE_IDS = (1, 3, 4)


def load(db_path: str, source_ids: list[int]) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    placeholders = ",".join("?" * len(CONTENT_TYPE_IDS))
    out: list[dict] = []

    for sid in source_ids:
        rows = conn.execute(
            f"""
            SELECT l.id AS line_id, l.shabad_id, l.source_page AS ang,
                   l.gurmukhi AS gurmukhi_ascii,
                   sec.name_english AS raag,
                   w.name_english   AS writer
            FROM lines l
            JOIN shabads s    ON l.shabad_id = s.id
            JOIN sections sec ON s.section_id = sec.id
            JOIN writers w    ON s.writer_id = w.id
            WHERE s.source_id = ?
              AND l.type_id IN ({placeholders})
            ORDER BY l.order_id
            """,
            [sid, *CONTENT_TYPE_IDS],
        )
        n = 0
        for r in rows:
            clean = strip_vishraams(r["gurmukhi_ascii"])
            uni = ascii_to_unicode(clean).strip()
            if not uni or len(uni) < 4:
                continue
            out.append({
                "line_id": r["line_id"],
                "source": SOURCE_LABELS[sid],
                "shabad_id": r["shabad_id"],
                "ang": r["ang"],
                "text": uni,
                "raag": r["raag"] if sid == 1 else "",
                "writer": r["writer"],
            })
            n += 1
        print(f"[{SOURCE_LABELS[sid]}] {n} lines", file=sys.stderr)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", default="1,2,3,4",
                    help="Comma-separated source_ids from DB")
    args = ap.parse_args()

    sids = [int(x) for x in args.sources.split(",")]
    rows = load(args.db, sids)
    Path(args.out).write_text(
        json.dumps(rows, ensure_ascii=False, indent=None)
    )
    print(f"[done] {len(rows)} lines → {args.out}")


if __name__ == "__main__":
    main()
