"""
Phase 0 — Script 00: Parse SGGS from SikhiToTheMax SQLite database.

Outputs:
  data/processed/tuks.json      — all SGGS content lines (Pankti + Rahao + Manglacharan)
  data/processed/ngrams.json    — deduplicated 3-5 word n-grams from tuks
  data/processed/vishraam_segments.json — lines split at heavy vishraams

Usage:
  python scripts/00_parse_sggs.py [--db database.sqlite]
"""
import argparse
import json
import sqlite3
import unicodedata
from collections import defaultdict
from pathlib import Path

from gurmukhi_converter import ascii_to_unicode, strip_vishraams, split_by_vishraams

SGGS_SOURCE_ID = 1  # Sri Guru Granth Sahib Ji
# Include content line types: Pankti (4), Rahao (3), Manglacharan (1)
# Exclude Sirlekh (2) = headers/titles
CONTENT_TYPE_IDS = {1, 3, 4}


def load_sggs(db_path: str) -> list[dict]:
    """Load all SGGS content lines with Unicode Gurmukhi and metadata."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            l.id            AS line_id,
            l.shabad_id,
            l.source_page   AS ang,
            l.source_line,
            l.gurmukhi      AS gurmukhi_ascii,
            l.type_id,
            l.order_id,
            sec.name_english AS raag,
            w.name_english   AS writer
        FROM lines l
        JOIN shabads s      ON l.shabad_id = s.id
        JOIN sections sec   ON s.section_id = sec.id
        JOIN writers w      ON s.writer_id = w.id
        WHERE s.source_id = ?
          AND l.type_id IN ({})
        ORDER BY l.order_id
    """.format(",".join("?" * len(CONTENT_TYPE_IDS))),
        [SGGS_SOURCE_ID, *CONTENT_TYPE_IDS],
    )

    tuks = []
    for idx, row in enumerate(rows):
        ascii_text = row["gurmukhi_ascii"]

        # Strip vishraam markers before converting for the clean text
        clean_ascii = strip_vishraams(ascii_text)
        unicode_text = ascii_to_unicode(clean_ascii)
        unicode_text = unicodedata.normalize("NFC", unicode_text).strip()

        # Also convert with vishraams preserved for segment splitting
        unicode_with_vishraams = ascii_to_unicode(ascii_text)
        unicode_with_vishraams = unicodedata.normalize("NFC", unicode_with_vishraams).strip()

        # Get vishraam segments (split at ;)
        segments_ascii = split_by_vishraams(ascii_text)
        segments_unicode = []
        for seg in segments_ascii:
            seg_clean = strip_vishraams(seg)
            seg_uni = ascii_to_unicode(seg_clean)
            seg_uni = unicodedata.normalize("NFC", seg_uni).strip()
            if seg_uni:
                segments_unicode.append(seg_uni)

        tuks.append({
            "tuk_id": idx,
            "line_id": row["line_id"],
            "shabad_id": row["shabad_id"],
            "ang": row["ang"],
            "raag": row["raag"],
            "writer": row["writer"],
            "type_id": row["type_id"],
            "text": unicode_text,
            "vishraam_segments": segments_unicode,
        })

    conn.close()
    return tuks


def extract_ngrams(tuks: list[dict], n_range: tuple = (3, 5)) -> list[dict]:
    """Extract deduplicated n-grams from tuk texts."""
    ngram_map: dict[str, list[int]] = defaultdict(list)

    for tuk in tuks:
        words = tuk["text"].split()
        for n in range(n_range[0], min(n_range[1] + 1, len(words) + 1)):
            for i in range(len(words) - n + 1):
                gram = " ".join(words[i : i + n])
                # Skip if it's mostly punctuation
                alpha_chars = sum(1 for c in gram if c.isalpha())
                if alpha_chars < 3:
                    continue
                ngram_map[gram].append(tuk["tuk_id"])

    ngrams = []
    for ngram_id, (text, tuk_ids) in enumerate(ngram_map.items()):
        ngrams.append({
            "ngram_id": ngram_id,
            "text": text,
            "source_tuk_ids": list(set(tuk_ids)),
        })
    return ngrams


def extract_vishraam_segments(tuks: list[dict]) -> list[dict]:
    """Extract vishraam-split segments as additional search units."""
    segments = []
    seg_id = 0
    for tuk in tuks:
        # Only split if there are multiple segments
        if len(tuk["vishraam_segments"]) > 1:
            for seg_text in tuk["vishraam_segments"]:
                words = seg_text.split()
                # Only include segments with 3+ words (meaningful for retrieval)
                if len(words) >= 3:
                    segments.append({
                        "segment_id": seg_id,
                        "tuk_id": tuk["tuk_id"],
                        "shabad_id": tuk["shabad_id"],
                        "ang": tuk["ang"],
                        "text": seg_text,
                    })
                    seg_id += 1
    return segments


def main(db_path: str):
    print(f"Loading SGGS from {db_path}...")
    tuks = load_sggs(db_path)
    print(f"  Tuks: {len(tuks):,}")

    # Count by type
    type_counts = defaultdict(int)
    type_names = {1: "Manglacharan", 3: "Rahao", 4: "Pankti"}
    for t in tuks:
        type_counts[t["type_id"]] += 1
    for tid, count in sorted(type_counts.items()):
        print(f"    {type_names.get(tid, tid)}: {count:,}")

    # Count shabads
    shabad_ids = set(t["shabad_id"] for t in tuks)
    print(f"  Shabads: {len(shabad_ids):,}")
    print(f"  Ang range: {tuks[0]['ang']}–{tuks[-1]['ang']}")

    # Sample output
    print(f"\n  Sample tuk: {tuks[0]['text'][:80]}")
    print(f"  Sample tuk: {tuks[100]['text'][:80]}")

    print("\nExtracting n-grams...")
    ngrams = extract_ngrams(tuks)
    print(f"  Unique n-grams: {len(ngrams):,}")

    print("Extracting vishraam segments...")
    segments = extract_vishraam_segments(tuks)
    print(f"  Vishraam segments (3+ words): {len(segments):,}")

    # Save outputs
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    tuks_path = out_dir / "tuks.json"
    json.dump(tuks, open(tuks_path, "w", encoding="utf-8"), ensure_ascii=False, indent=None)
    print(f"\nSaved: {tuks_path} ({tuks_path.stat().st_size / 1024 / 1024:.1f} MB)")

    ngrams_path = out_dir / "ngrams.json"
    json.dump(ngrams, open(ngrams_path, "w", encoding="utf-8"), ensure_ascii=False, indent=None)
    print(f"Saved: {ngrams_path} ({ngrams_path.stat().st_size / 1024 / 1024:.1f} MB)")

    segments_path = out_dir / "vishraam_segments.json"
    json.dump(segments, open(segments_path, "w", encoding="utf-8"), ensure_ascii=False, indent=None)
    print(f"Saved: {segments_path} ({segments_path.stat().st_size / 1024 / 1024:.1f} MB)")

    print("\nScript 00 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="database.sqlite")
    args = parser.parse_args()
    main(args.db)
