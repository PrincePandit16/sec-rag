import re
from typing import List, Tuple, Optional

class FinancialNormalizer:
    """
    Normalizes financial strings into standard float values for comparison.
    Handles variants like '$1.2B', '$1,200 million', '1,200,000,000'.
    """

    SCALE_MAP = {
        'trillion': 1_000_000_000_000,
        't': 1_000_000_000_000,
        'billion': 1_000_000_000,
        'b': 1_000_000_000,
        'million': 1_000_000,
        'm': 1_000_000,
        'thousand': 1_000,
        'k': 1_000,
    }

    @classmethod
    def normalize(cls, value_str: str) -> Optional[float]:
        """
        Converts a financial string to a float.
        Returns None if no numeric value is found.
        """
        if not value_str:
            return None

        # Remove currency symbols and commas
        clean_str = value_str.lower().replace('$', '').replace(',', '').strip()

        # Regex to split numeric part and scale part (e.g., "1.2" and "billion")
        match = re.search(r"([-+]?\d*\.?\d+)\s*([a-z]+)?", clean_str)
        if not match:
            return None

        number_part = match.group(1)
        scale_part = match.group(2)

        try:
            value = float(number_part)
            if scale_part and scale_part in cls.SCALE_MAP:
                value *= cls.SCALE_MAP[scale_part]
            return value
        except ValueError:
            return None

    @classmethod
    def is_equivalent(cls, val1: str, val2: str) -> bool:
        """Checks if two financial strings represent the same numeric value."""
        n1 = cls.normalize(val1)
        n2 = cls.normalize(val2)

        if n1 is None or n2 is None:
            # Fallback to strict string match if normalization fails
            return val1.strip().lower() == val2.strip().lower()

        # Use a small epsilon for float comparison
        return abs(n1 - n2) < 1e-2

    @classmethod
    def extract_values(cls, text: str) -> List[Tuple[float, str]]:
        """
        Finds all financial-looking values in a text block.
        Returns list of (normalized_value, original_string).
        """
        # Pattern for currency/numbers with optional scales
        pattern = r"(\$\s*[\d,.]+\s*(?:billion|million|thousand|[Bmk])?)"
        matches = re.findall(pattern, text, re.IGNORECASE)

        extracted = []
        for m in matches:
            norm = cls.normalize(m)
            if norm is not None:
                extracted.append((norm, m))
        return extracted
