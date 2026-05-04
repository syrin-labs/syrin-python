"""Deepgram Aura TTS provider. Requires deepgram-sdk package."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from syrin.generation._base_voice import BaseVoiceProvider


class DeepgramVoiceProvider(BaseVoiceProvider):
    """Deepgram Aura TTS provider. Implements VoiceGenerationProvider.

    Requires: pip install syrin[voice] or pip install deepgram-sdk.
    Set DEEPGRAM_API_KEY or pass api_key=.
    """

    def __init__(
        self,
        api_key: str | None = None,
        voice: str = "aura-asteria-en",
        model: str = "aura-asteria-en",
        **kwargs: object,
    ) -> None:
        super().__init__(api_key=api_key, model=model, **kwargs)
        self.voice = voice

    def _get_api_key_env(self) -> str:
        return "DEEPGRAM_API_KEY"

    def _get_package_name(self) -> str:
        return "syrin[voice]"

    def _synthesize(
        self,
        text: str,
        *,
        api_key: str,
        voice_id: str,
        speed: float,
        language: str,
        output_format: str,
        model_id: str,
        **kwargs: object,
    ) -> tuple[bytes, str, str]:
        from deepgram import DeepgramClient, SpeakOptions

        client = DeepgramClient(api_key=api_key)
        opts = SpeakOptions(model=model_id)
        fd, tmp_name = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            client.speak.rest.v("1").save(str(tmp), {"text": text}, opts)
            content_bytes = tmp.read_bytes()
        finally:
            tmp.unlink(missing_ok=True)
        return content_bytes, "aura", "audio/mpeg"
