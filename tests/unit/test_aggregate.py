#!/usr/bin/env python3
"""Unit tests for aggregate module."""
import pytest
from tools.aggregate import frac_to_float


class TestFracToFloat:
    """Test suite for frac_to_float function."""

    def test_valid_fraction(self):
        """Test converting valid fractions."""
        assert frac_to_float("30/1") == 30.0
        assert frac_to_float("24/1") == 24.0
        assert frac_to_float("1/2") == 0.5

    def test_zero_numerator(self):
        """Test fraction with zero numerator."""
        assert frac_to_float("0/1") == 0.0

    def test_division_by_zero(self):
        """Test handling of division by zero."""
        # Should return 0.0 and log warning
        result = frac_to_float("1/0")
        assert result == 0.0

    def test_invalid_format(self):
        """Test handling of invalid fraction format."""
        assert frac_to_float("invalid") == 0.0
        assert frac_to_float("a/b/c") == 0.0
        assert frac_to_float("") == 0.0

    def test_float_string(self):
        """Test converting float string."""
        assert frac_to_float("3.14") == 3.14
        assert frac_to_float("0") == 0.0
        assert frac_to_float("100") == 100.0

    def test_negative_values(self):
        """Test handling of negative values."""
        assert frac_to_float("-1/2") == -0.5
        assert frac_to_float("1/-2") == -0.5

    def test_decimal_in_fraction(self):
        """Test handling of decimals in fractions."""
        result = frac_to_float("3.5/2")
        assert result == 1.75
