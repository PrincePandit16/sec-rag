import re

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
