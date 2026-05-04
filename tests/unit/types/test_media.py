"""Tests for Media enum — single canonical type for content/capabilities."""

from __future__ import annotations

import pytest

from syrin.enums import Media


class TestMediaValues:
    """Valid Media enum values and string contract."""

    def test_all_values_present(self) -> None:
        expected = {"text", "image", "video", "audio", "file"}
        actual = {m.value for m in Media}
        assert actual == expected

    def test_strenum_string_value(self) -> None:
        assert Media.TEXT == "text"
        assert Media.IMAGE == "image"
        assert Media.VIDEO == "video"
        assert Media.AUDIO == "audio"
        assert Media.FILE == "file"

    def test_from_string_compatible(self) -> None:
        assert Media("text") is Media.TEXT
        assert Media("image") is Media.IMAGE
        assert Media("file") is Media.FILE

    def test_members_count(self) -> None:
        assert len(Media) == 5


class TestMediaSetUsage:
    """Media in sets (input_media / output_media style)."""

    def test_set_of_media_subset(self) -> None:
        input_media = {Media.TEXT, Media.IMAGE}
        assert Media.TEXT in input_media
        assert Media.VIDEO not in input_media

    def test_set_inclusion(self) -> None:
        supported = {Media.TEXT, Media.IMAGE, Media.VIDEO}
        required = {Media.TEXT, Media.IMAGE}
        assert required <= supported

    def test_set_missing_media(self) -> None:
        supported = {Media.TEXT}
        required = {Media.TEXT, Media.IMAGE}
        assert not (required <= supported)
        missing = required - supported
        assert missing == {Media.IMAGE}


class TestMediaInvalid:
    """Invalid string raises for Media."""

    def test_invalid_string_raises(self) -> None:
        with pytest.raises(ValueError):
            Media("invalid")
        with pytest.raises(ValueError):
            Media("")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            Media("")
