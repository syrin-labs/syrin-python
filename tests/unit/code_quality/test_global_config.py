"""Tests for GlobalConfig direct API (get/set) and ConfigurationError."""

from __future__ import annotations

import pytest

from syrin.config import GlobalConfig, get_config

# ---------------------------------------------------------------------------
# TestGlobalConfig — direct API (get/set)
# ---------------------------------------------------------------------------


class TestGlobalConfig:
    """Tests for GlobalConfig class defaults and get/set methods."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = GlobalConfig()
        assert config.trace is False
        assert config.default_model is None
        assert config.default_api_key is None

    def test_get_set_methods(self) -> None:
        """Test get/set methods."""
        config = get_config()
        config.set(trace=True)
        assert config.get("trace") is True
        assert config.get("nonexistent", "default") == "default"
        config.set(trace=False)


# ---------------------------------------------------------------------------
# TestRunFunction — run() signature only
# ---------------------------------------------------------------------------


class TestRunFunction:
    """Tests for syrin.run() function signature."""

    def test_run_signature(self) -> None:
        """Test run function has correct signature."""
        import inspect

        import syrin

        sig = inspect.signature(syrin.run)
        params = list(sig.parameters.keys())
        assert "input" in params
        assert "model" in params
        assert "system_prompt" in params
        assert "tools" in params
        assert "budget" in params


# ---------------------------------------------------------------------------
# TestGlobalConfigSecurity — set()/get() whitelist enforcement
# ---------------------------------------------------------------------------


class TestGlobalConfigSecurity:
    """Security: set()/get() must not allow private attribute access."""

    def test_set_private_attribute_raises_configuration_error(self) -> None:
        """Setting a private/unknown attribute via set() raises ConfigurationError."""
        from syrin.exceptions import ConfigurationError

        config = GlobalConfig()
        with pytest.raises(ConfigurationError, match="_lock"):
            config.set(_lock="injected")

    def test_set_unknown_key_raises_configuration_error(self) -> None:
        """Unknown keys in set() raise ConfigurationError with valid keys listed."""
        from syrin.exceptions import ConfigurationError

        config = GlobalConfig()
        with pytest.raises(ConfigurationError, match="nonexistent_key"):
            config.set(nonexistent_key="value")

    def test_get_private_attribute_returns_default(self) -> None:
        """get() on a private attribute returns the default."""
        config = GlobalConfig()
        assert config.get("_lock") is None
        assert config.get("_cloud_api_key", "SENTINEL") == "SENTINEL"

    def test_get_unknown_key_returns_default(self) -> None:
        """get() on an unknown key returns the default."""
        config = GlobalConfig()
        assert config.get("__class__") is None
        assert config.get("nonexistent", 42) == 42

    def test_set_whitelisted_keys_work(self) -> None:
        """All whitelisted keys can be set via set()."""
        config = GlobalConfig()
        config.set(trace=True, debug=True)
        assert config.trace is True
        assert config.debug is True
        config.set(trace=False, debug=False)

    def test_get_whitelisted_keys_work(self) -> None:
        """All whitelisted keys can be read via get()."""
        config = GlobalConfig()
        result = config.get("trace")
        assert result is False


# ---------------------------------------------------------------------------
# TestConfigureUnknownKeys — configure() error messaging
# ---------------------------------------------------------------------------


class TestConfigureUnknownKeys:
    def test_error_message_lists_valid_keys(self) -> None:
        """Error message must show valid keys so the user knows what to use."""
        import syrin
        from syrin.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError) as exc_info:
            syrin.configure(totally_unknown_param=True)

        error_msg = str(exc_info.value)
        assert "trace" in error_msg or "debug" in error_msg or "valid" in error_msg.lower()

    def test_multiple_unknown_keys_all_reported(self) -> None:
        """If multiple unknown keys are passed, the error must mention them."""
        import syrin
        from syrin.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError) as exc_info:
            syrin.configure(bad_key_1=1, bad_key_2=2)

        error_msg = str(exc_info.value)
        assert "bad_key_1" in error_msg or "bad_key_2" in error_msg

    def test_mixed_valid_and_invalid_raises(self) -> None:
        """When one key is valid and one is not, must raise for the invalid one."""
        import syrin
        from syrin.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="bad_key"):
            syrin.configure(trace=False, bad_key="oops")

    def test_configure_error_extends_syrin_error(self) -> None:
        """ConfigurationError must extend SyrinError for consistent catch handling."""
        from syrin.exceptions import ConfigurationError, SyrinError

        assert issubclass(ConfigurationError, SyrinError)

    def test_configure_error_importable_from_exceptions(self) -> None:
        """ConfigurationError must be importable from the syrin.exceptions package."""
        from syrin.exceptions import ConfigurationError

        assert ConfigurationError is not None
