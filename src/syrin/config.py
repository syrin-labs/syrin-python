"""Global configuration for Syrin."""

from __future__ import annotations

import os
import threading

from syrin.types import ModelConfig


class GlobalConfig:
    """Global Syrin configuration. Accessed via get_config().

    Attributes:
        trace: Whether tracing is enabled. Set via configure(trace=True).
        debug: Whether debug mode is enabled. Set via configure(debug=True).
        default_model: Default ModelConfig when none specified.
        default_api_key: Default API key (rarely used; pass per Model).
        cloud_enabled: Whether remote config is enabled (set by syrin.init()).
        cloud_api_key: API key for Syrin Cloud (env SYRIN_API_KEY or passed to init()).
        cloud_base_url: Base URL for config API (default https://api.syrin.ai/v1).
        cloud_transport: ConfigTransport when custom transport passed to init(); None when using default SSETransport.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._trace: bool = False
        self._debug: bool = False
        self._default_model: ModelConfig | None = None
        self._default_api_key: str | None = None
        self._env_prefix = "SYRIN_"
        # Remote config (cloud): set by syrin.init()
        self._cloud_api_key: str | None = None
        self._cloud_base_url: str = "https://api.syrin.ai/v1"
        self._cloud_enabled: bool = False
        self._cloud_transport: object = None
        self._load_from_env()

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        if os.environ.get(f"{self._env_prefix}TRACE", "").lower() in ("1", "true", "yes"):
            self._trace = True
        key = os.environ.get(f"{self._env_prefix}API_KEY", "").strip()
        if key:
            self._cloud_api_key = key

    @property
    def trace(self) -> bool:
        """Whether tracing is enabled."""
        return self._trace

    @trace.setter
    def trace(self, value: bool) -> None:
        with self._lock:
            self._trace = value

    @property
    def debug(self) -> bool:
        """Whether debug mode is enabled."""
        return self._debug

    @debug.setter
    def debug(self, value: bool) -> None:
        with self._lock:
            self._debug = value

    @property
    def default_model(self) -> ModelConfig | None:
        """Default model to use when none is specified."""
        return self._default_model

    @default_model.setter
    def default_model(self, value: ModelConfig | None) -> None:
        with self._lock:
            self._default_model = value

    @property
    def default_api_key(self) -> str | None:
        """Default API key to use when none is specified."""
        return self._default_api_key

    @default_api_key.setter
    def default_api_key(self, value: str | None) -> None:
        with self._lock:
            self._default_api_key = value

    @property
    def cloud_enabled(self) -> bool:
        """Whether remote config (cloud) is enabled. Set True by syrin.init()."""
        return self._cloud_enabled

    @cloud_enabled.setter
    def cloud_enabled(self, value: bool) -> None:
        with self._lock:
            self._cloud_enabled = value

    @property
    def cloud_api_key(self) -> str | None:
        """API key for Syrin Cloud. From SYRIN_API_KEY env or passed to syrin.init()."""
        return self._cloud_api_key

    @cloud_api_key.setter
    def cloud_api_key(self, value: str | None) -> None:
        with self._lock:
            self._cloud_api_key = value

    @property
    def cloud_base_url(self) -> str:
        """Base URL for config API. Default https://api.syrin.ai/v1."""
        return self._cloud_base_url

    @cloud_base_url.setter
    def cloud_base_url(self, value: str) -> None:
        with self._lock:
            self._cloud_base_url = value

    @property
    def cloud_transport(self) -> object:
        """ConfigTransport when custom transport passed to init(); None when using default SSETransport."""
        return self._cloud_transport

    @cloud_transport.setter
    def cloud_transport(self, value: object) -> None:
        with self._lock:
            self._cloud_transport = value

    #: Public attributes that can be read/written via get()/set().
    _PUBLIC_KEYS: frozenset[str] = frozenset(
        {
            "trace",
            "debug",
            "default_model",
            "default_api_key",
            "cloud_enabled",
            "cloud_api_key",
            "cloud_base_url",
            "cloud_transport",
        }
    )

    def get(self, key: str, default: object = None) -> object:
        """Get a public configuration value. Unknown keys return default."""
        if key not in self._PUBLIC_KEYS:
            return default
        return getattr(self, key, default)

    def set(self, **kwargs: object) -> None:
        """Set multiple public configuration values.

        Raises:
            ConfigurationError: If any key is not a recognised configuration key.
        """
        from syrin.exceptions import ConfigurationError  # noqa: PLC0415

        unknown = {k for k in kwargs if k not in self._PUBLIC_KEYS}
        if unknown:
            sorted_unknown = sorted(unknown)
            sorted_valid = sorted(self._PUBLIC_KEYS)
            raise ConfigurationError(
                f"Unknown configuration key(s): {sorted_unknown}. Valid keys: {sorted_valid}.",
                unknown_keys=unknown,
                valid_keys=self._PUBLIC_KEYS,
            )
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)


_config = GlobalConfig()


def get_config() -> GlobalConfig:
    """Get the global Syrin configuration instance.

    Returns:
        GlobalConfig singleton.
    """
    return _config


def configure(**kwargs: object) -> None:
    """Configure global Syrin settings.

    Args:
        trace: Enable tracing (default: False)
        debug: Enable debug mode (default: False)
        default_model: Default model to use (Model or ModelConfig)
        default_api_key: Default API key to use

    Example:
        >>> import syrin
        >>> syrin.configure(trace=True)
        >>> syrin.configure(default_model="openai/gpt-4o")
    """
    _config.set(**kwargs)


__all__ = ["GlobalConfig", "get_config", "configure"]
