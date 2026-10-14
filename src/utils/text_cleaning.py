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
    # Replace multiple spaces with one
    text = re.sub(r"[ \t]+", " ", text)
    # Replace multiple newlines with two (keep paragraph separation)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def remove_boilerplate(text: str) -> str:
    """Removes common SEC filing boilerplate patterns."""
    # These patterns are derived from existing chunk_documents.py logic
    _BOILERPLATE_PATTERNS = [
        r"Page \d+ of \d+",
        r"\(Continued on next page\)",
        r"See accompanying notes to consolidated financial statements",
        r"All amounts in millions, unless otherwise noted",
    ]

    cleaned = text
    for pattern in _BOILERPLATE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    return cleaned.strip()

def clean_html_artifacts(text: str) -> str:
    """Removes leftover HTML entities or glyphs."""
    # Replace common replacement characters
    text = text.replace("�", "")
    # Remove any remaining angle-bracketed noise that looks like tags
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()
