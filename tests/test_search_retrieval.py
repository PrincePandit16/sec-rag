import pytest
from src.utils.text_cleaning import fix_mojibake, normalize_whitespace
from src.search import verify_citations

def test_encoding_cleaning():
    # Test mojibake cleaning
    input_text = "Revenue was â¯ $10B and âs profit increased."
    cleaned = fix_mojibake(input_text)
    assert "â¯" not in cleaned
    assert "âs" not in cleaned
    assert "Revenue was   $10B" in cleaned or "Revenue was $10B" in cleaned

def test_whitespace_normalization():
    input_text = "Too    many    spaces\n\n\nToo many newlines"
    cleaned = normalize_whitespace(input_text)
    assert "    " not in cleaned
    assert "\n\n\n" not in cleaned

@pytest.mark.asyncio
async def test_citation_drifting():
    # Mock results where a unique value is ONLY in the last chunk
    results = [
        {"document": "Apple revenue was $100M", "metadata": {}},
        {"document": "Microsoft revenue was $200M", "metadata": {}},
        {"document": "Google revenue was $123.456M", "metadata": {}},
    ]

    # Synthetic answer citing the Google value as [1]
    answer = "Google revenue was $123.456M [1]."

    # We need a mock LLM client or just call the logic (passing None for client)
    is_verified, report = await verify_citations(None, answer, results)

    assert is_verified is False
    assert "Value '$123.456M' cited as [1] but found in [3]" in report

@pytest.mark.asyncio
async def test_out_of_bounds_citation():
    results = [{"document": "Some text", "metadata": {}}]
    answer = "Some value $10M [99]."

    is_verified, report = await verify_citations(None, answer, results)

    assert is_verified is False
    assert "Citation index [99] is out of range" in report

@pytest.mark.asyncio
async def test_format_agnostic_verification():
    # Source has $10M, Answer has $10 million
    results = [{"document": "Revenue was $10M", "metadata": {}}]
    answer = "Revenue was $10 million [1]."

    is_verified, report = await verify_citations(None, answer, results)

    assert is_verified is True
