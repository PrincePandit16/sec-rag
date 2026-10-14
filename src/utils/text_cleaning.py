import re

def fix_mojibake(text: str) -> str:
    """
    Fixes encoding artifacts (mojibake) and common SEC filing glitches.
    Uses ftfy to handle UTF-8/Latin-1 mismatches.
    """
    import ftfy
    if not text:
        return ""

    # Fixes common mojibake (e.g., âs -> 's)
    text = ftfy.fix_text(text)

    # Handle remaining SEC-specific artifacts that ftfy might miss
    # â¯ and ã are common artifacts in SEC filings when viewed/processed incorrectly
    text = text.replace("â¯", " ").replace("ã", "")

    return text.strip()

def normalize_whitespace(text: str) -> str:
    """Collapses multiple spaces and newlines into a single space or newline."""
    if not text:
        return ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def remove_boilerplate(text: str) -> str:
    """Removes common SEC filing boilerplate patterns."""
    _BOILERPLATE_PATTERNS = [
        r"Page \d+ of \d+",
        r"\(Continued on next page\)",
        r"See accompanying notes to consolidated financial statements",
        r"All amounts in millions, unless otherwise noted",
        r"All amounts in millions, unless otherwise and noted",
        r"Amounts in millions, unless otherwise noted",
        r"Total net sales",
    ]

    cleaned = text
    for pattern in _BOILERPLATE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    return cleaned.strip()

def clean_html_artifacts(text: str) -> str:
    """Removes leftover HTML entities or glyphs."""
    text = text.replace("�", "")
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()

def unified_clean(text: str) -> str:
    """
    The master cleaning pipeline for all SEC text processing.
    Applies: Mojibake fix -> Boilerplate removal -> HTML clean -> Whitespace normalization.
    """
    if not text:
        return ""
    text = fix_mojibake(text)
    text = remove_boilerplate(text)
    text = clean_html_artifacts(text)
    text = normalize_whitespace(text)
    return text
