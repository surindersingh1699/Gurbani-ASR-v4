"""
ASCII to Unicode Gurmukhi converter.
Ported from ShabadOS/gurmukhi-utils (npm) toUnicode.js + unicodeMappings.json.
"""
import re

# Multi-char mappings (must be applied before single-char)
MULTI_CHAR_MAP = {
    "ei": "ਇ",
    "au": "ਉ",
    "aU": "ਊ",
    "eI": "ਈ",
    "ey": "ਏ",
    "Aw": "ਆ",
    "AY": "ਐ",
    "AO": "ਔ",
    "AW": "ਆਂ",
    "<>": "ੴ",
    "ÅÆ": "ੴ",
}

# Single-char mappings
SINGLE_CHAR_MAP = {
    "ƒ": "ਨੂੰ",
    "†": "੍ਟ",
    "˜": "੍ਨ",
    "Î": "੍ਯ",
    "î": "੍ਯ",
    "ç": "੍ਚ",
    "œ": "੍ਤ",
    "M": "ੰ",
    "H": "੍ਹ",
    "§": "੍ਹੂ",
    "i": "ਿ",
    "I": "ੀ",
    "u": "ੁ",
    "U": "ੂ",
    "y": "ੇ",
    "Y": "ੈ",
    "N": "ਂ",
    "o": "ੋ",
    "O": "ੌ",
    "R": "੍ਰ",
    "W": "ਾਂ",
    "w": "ਾ",
    "®": "੍ਰ",
    "´": "ੵ",
    "Ï": "ੵ",
    "µ": "ੰ",
    "μ": "ੰ",
    "@": "ੑ",
    "`": "ੱ",
    "~": "ੱ",
    "Í": "੍ਵ",
    "Ú": "ਃ",
    "ü": "ੁ",
    "|": "ਙ",
    "¨": "ੂ",
    "Ø": "",
    "ˆ": "ਂ",
    "¤": "ੱ",
    "a": "ੳ",
    "A": "ਅ",
    "b": "ਬ",
    "B": "ਭ",
    "c": "ਚ",
    "C": "ਛ",
    "d": "ਦ",
    "D": "ਧ",
    "e": "ੲ",
    "E": "ਓ",
    "F": "ਢ",
    "f": "ਡ",
    "g": "ਗ",
    "G": "ਘ",
    "h": "ਹ",
    "j": "ਜ",
    "J": "ਝ",
    "k": "ਕ",
    "K": "ਖ",
    "l": "ਲ",
    "L": "ਲ਼",
    "m": "ਮ",
    "n": "ਨ",
    "p": "ਪ",
    "P": "ਫ",
    "q": "ਤ",
    "Q": "ਥ",
    "r": "ਰ",
    "s": "ਸ",
    "S": "ਸ਼",
    "t": "ਟ",
    "T": "ਠ",
    "v": "ਵ",
    "V": "ੜ",
    "x": "ਣ",
    "X": "ਯ",
    "z": "ਜ਼",
    "Z": "ਗ਼",
    "1": "੧",
    "2": "੨",
    "3": "੩",
    "4": "੪",
    "5": "੫",
    "6": "੬",
    "^": "ਖ਼",
    "7": "੭",
    "&": "ਫ਼",
    "8": "੮",
    "9": "੯",
    "0": "੦",
    "[": "।",
    "]": "॥",
    "Ò": "॥",
    "\\": "ਞ",
    "ï": "ਯ",
    "Ç": "☬",
    "¡": "ੴ",
    "æ": "਼",
    "‚": "❁",
}

# Reordering rules (applied BEFORE character mapping)
REORDER_RULES = [
    (re.compile(r"i(.)"), r"\1i"),                              # Sihari position swap
    (re.compile(r"®"), "R"),                                     # Normalize pair R
    (re.compile(r"([iMµyY])([RH§ÍÏçœ˜†])"), r"\2\1"),          # Pair sounds after matras
    (re.compile(r"([MµyY])([uU])"), r"\2\1"),                    # Tipee/lava with aunkar
    (re.compile(r"`([wWIoOyYRH§´ÍÏçœ˜†uU])"), r"\1`"),         # Adhak position
    (re.compile(r"i([´Î])"), r"\1i"),                            # Swap i with ´ or Î
    (re.compile(r"uo"), "ou"),                                   # Aunkarh+hora swap
]


def ascii_to_unicode(text: str) -> str:
    """Convert SikhiToTheMax ASCII Gurmukhi to Unicode Gurmukhi."""
    # Step 1: Apply reordering rules
    for pattern, replacement in REORDER_RULES:
        text = pattern.sub(replacement, text)

    # Step 2: Apply multi-char mappings first (longest first)
    for ascii_seq, uni_char in sorted(MULTI_CHAR_MAP.items(), key=lambda x: -len(x[0])):
        text = text.replace(ascii_seq, uni_char)

    # Step 3: Apply single-char mappings
    result = []
    for ch in text:
        result.append(SINGLE_CHAR_MAP.get(ch, ch))
    return "".join(result)


def strip_vishraams(text: str) -> str:
    """Remove vishraam markers (; , .) from Gurmukhi text."""
    return text.replace(";", "").replace(",", "").replace(".", "")


def split_by_vishraams(text: str) -> list[str]:
    """Split text at heavy vishraams (;) into segments."""
    parts = text.split(";")
    return [p.strip() for p in parts if p.strip()]


if __name__ == "__main__":
    # Quick test with Mool Mantar
    test = "<> siq nwmu krqw purKu inrBau inrvYru Akwl mUriq AjUnI sYBM gur pRswid ]"
    result = ascii_to_unicode(test)
    print(f"ASCII:   {test}")
    print(f"Unicode: {result}")
    expected = "ੴ ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ ਨਿਰਭਉ ਨਿਰਵੈਰੁ ਅਕਾਲ ਮੂਰਤਿ ਅਜੂਨੀ ਸੈਭੰ ਗੁਰ ਪ੍ਰਸਾਦਿ ॥"
    print(f"Expect:  {expected}")
    print(f"Match:   {result == expected}")
