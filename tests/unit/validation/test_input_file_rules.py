"""Tests for InputFileRules — valid and invalid cases."""

from __future__ import annotations

import pytest

from syrin.capabilities import InputFileRules


class TestInputFileRulesValid:
    """Valid InputFileRules construction and behavior."""

    def test_minimal_allowed_mime_types(self) -> None:
        r = InputFileRules(allowed_mime_types=["image/png", "image/jpeg"])
        assert r.allowed_mime_types == ["image/png", "image/jpeg"]
        assert r.max_size_mb == 10.0

    def test_custom_max_size_mb(self) -> None:
        r = InputFileRules(
            allowed_mime_types=["application/pdf"],
            max_size_mb=5.0,
        )
        assert r.max_size_mb == 5.0

    def test_allows_matching_mime(self) -> None:
        r = InputFileRules(allowed_mime_types=["image/png", "image/jpeg"])
        assert r.allows("image/png") is True
        assert r.allows("image/JPEG") is True
        assert r.allows("  image/png  ") is True

    def test_allows_no_match(self) -> None:
        r = InputFileRules(allowed_mime_types=["image/png"])
        assert r.allows("application/pdf") is False
        assert r.allows("image/jpeg") is False


class TestInputFileRulesInvalid:
    """Invalid InputFileRules raises."""

    def test_negative_max_size_mb_raises(self) -> None:
        with pytest.raises(ValueError, match="max_size_mb must be > 0"):
            InputFileRules(
                allowed_mime_types=["image/png"],
                max_size_mb=-1.0,
            )

    def test_zero_max_size_mb_raises(self) -> None:
        with pytest.raises(ValueError, match="max_size_mb must be > 0"):
            InputFileRules(
                allowed_mime_types=["image/png"],
                max_size_mb=0.0,
            )

    def test_allowed_mime_types_not_list_raises(self) -> None:
        with pytest.raises(TypeError, match="must be a list"):
            InputFileRules(allowed_mime_types=("image/png",))  # type: ignore[arg-type]
