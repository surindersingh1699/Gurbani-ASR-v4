"""
Test torchaudio MMS_FA forced alignment on SGPC live kirtan samples.

MMS_FA only supports Latin characters (a-z), so we transliterate Gurmukhi
to romanized form first using indic_transliteration, then run alignment,
and map word-level timestamps back to the original Gurmukhi words.

This tests whether MMS_FA can handle:
1. Kirtan audio with background harmonium/tabla
2. Romanized Punjabi text alignment
3. Timestamp correction accuracy

Usage:
    pip install torchaudio datasets soundfile indic-transliteration
    python scripts/test_mms_forced_align.py
    python scripts/test_mms_forced_align.py --n-samples 20
    python scripts/test_mms_forced_align.py --offset 5000  # test kirtan (not ardas)
"""

import argparse
import re
import time
import unicodedata

import numpy as np
import torch
import torchaudio
from indic_transliteration import sanscript
from torchaudio.pipelines import MMS_FA as bundle

# YouTube auto-caption markers to filter out
CAPTION_NOISE = [
    "[ਸੰਗੀਤ]", "[ਤਾੜੀਆਂ]", "[Music]", "[Applause]",
    "[ਹਾਸਾ]", "[Laughter]", "[ __ ]",
]

SRC_DATASET = "surindersinghssj/sgpc-amritsar-kirtan-live"
SAMPLE_RATE = 16000


def is_noise_caption(text: str) -> bool:
    """Check if text is a YouTube noise marker like [ਸੰਗੀਤ]."""
    text = text.strip()
    for marker in CAPTION_NOISE:
        if marker in text:
            return True
    if text.startswith("[") and text.endswith("]"):
        return True
    return False


def gurmukhi_to_roman(text: str) -> str:
    """Transliterate Gurmukhi text to plain a-z romanized form for MMS_FA.

    Uses IAST (International Alphabet of Sanskrit Transliteration) then
    strips diacritics to get plain ASCII. This preserves phoneme structure
    well enough for forced alignment.
    """
    # Transliterate to IAST first (preserves phoneme structure)
    iast = sanscript.transliterate(text, sanscript.GURMUKHI, sanscript.IAST)

    # Strip diacritics: ā→a, ī→i, ū→u, ṅ→n, etc.
    nfkd = unicodedata.normalize("NFKD", iast)
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))

    # Keep only a-z and spaces
    clean = re.sub(r"[^a-zA-Z ]", "", ascii_text).lower()
    # Collapse multiple spaces
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def load_samples(n_samples: int, offset: int = 0):
    """Load n samples from the HF dataset."""
    from datasets import Audio, load_dataset

    print(f"[data] Loading {n_samples} samples from {SRC_DATASET} (offset={offset})...")
    ds = load_dataset(SRC_DATASET, split=f"train[{offset}:{offset + n_samples}]")
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE, decode=True))
    print(f"[data] Loaded {len(ds)} samples")
    return ds


def align_sample(model, tokenizer, aligner, waveform: torch.Tensor,
                 gurmukhi_text: str, sample_rate: int):
    """Run MMS forced alignment on a single sample.

    Transliterates Gurmukhi → romanized, runs alignment, maps back to
    original Gurmukhi words.

    Returns list of {gurmukhi_word, roman_word, start, end, score} dicts.
    """
    gurmukhi_words = gurmukhi_text.split()
    roman_words = [gurmukhi_to_roman(w) for w in gurmukhi_words]

    # Filter out empty romanizations (e.g., pure punctuation)
    valid_pairs = [(g, r) for g, r in zip(gurmukhi_words, roman_words) if r]
    if not valid_pairs:
        return []

    gurmukhi_words_valid = [p[0] for p in valid_pairs]
    roman_words_valid = [p[1] for p in valid_pairs]

    # Resample if needed
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, bundle.sample_rate
        )

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Get emission probabilities
    with torch.inference_mode():
        emission, _ = model(waveform)

    # Tokenize — MMS_FA tokenizer expects a LIST of word strings
    tokens = tokenizer(roman_words_valid)
    if not tokens:
        return []

    # Run CTC forced alignment
    token_spans = aligner(emission[0], tokens)

    # Convert frame indices to seconds
    num_frames = emission.shape[1]
    duration = waveform.shape[1] / bundle.sample_rate
    ratio = duration / num_frames

    # Map token spans (character-level) back to words
    # token_spans is list of lists — one list per word
    results = []
    for word_idx, (g_word, r_word) in enumerate(zip(gurmukhi_words_valid, roman_words_valid)):
        if word_idx >= len(token_spans):
            break

        word_spans = token_spans[word_idx]
        if not word_spans:
            continue

        word_start = word_spans[0].start * ratio
        word_end = word_spans[-1].end * ratio
        word_score = np.mean([s.score for s in word_spans])

        results.append({
            "gurmukhi": g_word,
            "roman": r_word,
            "start": round(word_start, 3),
            "end": round(word_end, 3),
            "score": round(float(word_score), 3),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Test MMS FA on SGPC kirtan")
    parser.add_argument("--n-samples", type=int, default=15,
                        help="Number of samples to test")
    parser.add_argument("--offset", type=int, default=0,
                        help="Offset into dataset (use 5000+ for kirtan, 0 for ardas)")
    parser.add_argument("--device", default="cpu",
                        help="Device for model (cpu or cuda)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Test transliteration first
    print("[transliteration] Testing Gurmukhi -> Roman mapping:")
    test_phrases = [
        "ਵਾਹਿਗੁਰੂ", "ਸਤਿ ਨਾਮੁ", "ਗੋਪਾਲ ਤੇਰਾ",
        "ਖਾਲਸਾ ਜੀ ਬੋਲੋ", "ਧਿਆਈਐ ਜਿਸ ਸਭ ਦੁਖ ਜਾਇ",
    ]
    for phrase in test_phrases:
        roman = gurmukhi_to_roman(phrase)
        print(f"  {phrase:30s} -> {roman}")

    # Load MMS FA model
    print(f"\n[model] Loading MMS_FA model on {device}...")
    t0 = time.time()
    model = bundle.get_model().to(device)
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    print(f"[model] Loaded in {time.time() - t0:.1f}s")
    print(f"[model] Expected sample rate: {bundle.sample_rate}")

    # Load samples
    ds = load_samples(args.n_samples, args.offset)

    print(f"\n{'='*70}")
    print(f"FORCED ALIGNMENT RESULTS")
    print(f"{'='*70}")

    successes = 0
    failures = 0
    all_scores = []

    for i, sample in enumerate(ds):
        text = sample["gurmukhi_text"]
        audio = sample["audio"]
        orig_start = sample.get("start_time", 0)
        orig_end = sample.get("end_time", 0)
        orig_duration = sample.get("duration", 0)

        if is_noise_caption(text):
            print(f"\n[{i}] SKIPPED noise marker: {text}")
            continue

        print(f"\n--- Sample {i} (original: {orig_start:.2f}s - {orig_end:.2f}s, "
              f"dur={orig_duration:.2f}s) ---")
        print(f"  Gurmukhi: {text}")
        print(f"  Romanized: {gurmukhi_to_roman(text)}")

        waveform = torch.tensor(audio["array"], dtype=torch.float32).to(device)
        sr = audio["sampling_rate"]

        try:
            word_alignments = align_sample(
                model, tokenizer, aligner, waveform, text, sr
            )
        except Exception as e:
            print(f"  ALIGNMENT FAILED: {e}")
            failures += 1
            continue

        if not word_alignments:
            print(f"  NO ALIGNMENTS returned")
            failures += 1
            continue

        successes += 1
        scores = [w["score"] for w in word_alignments]
        all_scores.extend(scores)
        avg_score = np.mean(scores)

        # Print word-level alignment
        print(f"  Aligned {len(word_alignments)} words (avg score: {avg_score:.3f}):")
        for wa in word_alignments:
            bar = "█" * int(wa["score"] * 10)
            print(f"    {wa['start']:6.2f}s - {wa['end']:6.2f}s  "
                  f"[{wa['score']:.2f}] {bar:10s}  {wa['gurmukhi']} ({wa['roman']})")

        # Timing analysis
        first_start = word_alignments[0]["start"]
        last_end = word_alignments[-1]["end"]
        audio_dur = len(audio["array"]) / sr
        speech_pct = (last_end - first_start) / audio_dur * 100
        lead = first_start
        trail = audio_dur - last_end

        print(f"\n  Timing analysis:")
        print(f"    Audio duration:    {audio_dur:.2f}s")
        print(f"    Speech span:       {first_start:.2f}s - {last_end:.2f}s")
        print(f"    Speech coverage:   {speech_pct:.0f}%")
        print(f"    Leading silence:   {lead:.2f}s "
              f"{'<-- timestamp off?' if lead > 1.0 else 'ok'}")
        print(f"    Trailing silence:  {trail:.2f}s "
              f"{'<-- timestamp off?' if trail > 1.0 else 'ok'}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Samples tested:  {args.n_samples}")
    print(f"  Successful:      {successes}")
    print(f"  Failed:          {failures}")
    if all_scores:
        print(f"  Avg align score: {np.mean(all_scores):.3f}")
        print(f"  Min align score: {np.min(all_scores):.3f}")
        print(f"  Max align score: {np.max(all_scores):.3f}")
        low = sum(1 for s in all_scores if s < 0.5)
        print(f"  Low confidence (<0.5): {low}/{len(all_scores)} words "
              f"({low/len(all_scores)*100:.0f}%)")

    print(f"\n[next steps]")
    print(f"  If avg score > 0.6 and coverage looks good:")
    print(f"    -> MMS_FA works! Use it to re-timestamp the full dataset.")
    print(f"  If avg score < 0.4 or lots of failures:")
    print(f"    -> Try faster-whisper approach (fix_sgpc_forced_align.py)")
    print(f"    -> Or aeneas TTS+DTW with Google TTS Punjabi")


if __name__ == "__main__":
    main()
