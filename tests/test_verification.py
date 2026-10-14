import pytest
from src.utils.financial_normalization import FinancialNormalizer

def test_financial_normalization_equivalence():
    # Test different formats of the same value
    assert FinancialNormalizer.is_equivalent("$10 million", "$10,000,000")
    assert FinancialNormalizer.is_equivalent("$1.2B", "$1,200 million")
    assert FinancialNormalizer.is_equivalent("1,200,000,000", "$1.2 billion")
    assert FinancialNormalizer.is_equivalent("$500K", "$500,000")

    # Test different values
    assert not FinancialNormalizer.is_equivalent("$10 million", "$11 million")
    assert not FinancialNormalizer.is_equivalent("$1.2B", "$1.3B")

def test_financial_normalization_scales():
    assert FinancialNormalizer.normalize("$1.5 trillion") == 1_500_000_000_000.0
    assert FinancialNormalizer.normalize("$2.5 billion") == 2_500_000_000.0
    assert FinancialNormalizer.normalize("$100 million") == 100_000_000.0
    assert FinancialNormalizer.normalize("$50 thousand") == 50_000.0

def test_financial_normalization_edge_cases():
    assert FinancialNormalizer.normalize("") is None
    assert FinancialNormalizer.normalize("no numbers here") is None
    assert FinancialNormalizer.normalize("$abc") is None
