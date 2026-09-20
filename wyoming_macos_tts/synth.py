"""Speech synthesis through a persistent AVSpeechSynthesizer.

Spawning `say` once per request costs roughly 820ms of process start-up
before any audio appears. Keeping one AVSpeechSynthesizer alive and asking
it for PCM buffers removes that cost entirely; the audio it returns is the
same as `say` produces.

AVSpeechSynthesizer delivers its buffer callbacks on the main queue, so the
main thread has to run a CFRunLoop while the server itself runs on another
thread (see __main__.py). Calls into synthesize() may come from any thread.
"""

import asyncio
import logging
import threading
from collections import OrderedDict
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from AVFoundation import (
    AVSpeechSynthesisVoice,
    AVSpeechSynthesizer,
    AVSpeechUtterance,
)

_LOGGER = logging.getLogger("wyoming-macos-tts")

# AVSpeechSynthesizer always hands back 32-bit float, mono, non-interleaved.
WIDTH = 2  # bytes per sample after conversion to signed 16-bit
CHANNELS = 1

_TIMEOUT = 30  # seconds


class Voice(NamedTuple):
    """A voice as advertised to Wyoming clients."""

    name: str  # unique, e.g. "Anna (de-DE)"
    short_name: str  # as macOS reports it, e.g. "Anna"
    language: str  # e.g. "de_DE"
    av_voice: object


def _to_wyoming_language(language: str) -> str:
    """Turn a BCP-47 tag (de-DE) into the form the info event uses (de_DE)."""
    return language.replace("-", "_")


def list_voices() -> List[Voice]:
    """All installed voices, sorted by name."""
    voices = [
        Voice(
            name=f"{v.name()} ({v.language()})",
            short_name=v.name(),
            language=_to_wyoming_language(v.language()),
            av_voice=v,
        )
        for v in AVSpeechSynthesisVoice.speechVoices()
    ]
    return sorted(voices, key=lambda v: v.name)


class Synthesizer:
    """Turns text into 16-bit PCM, with a cache for repeated phrases."""

    def __init__(self, cache_bytes: int = 0) -> None:
        self._synth = AVSpeechSynthesizer.alloc().init()
        self._voices = list_voices()
        self._by_name: Dict[str, Voice] = {v.name: v for v in self._voices}
        self._by_short_name: Dict[str, Voice] = {}
        for voice in self._voices:
            # First match wins, so a bare name stays usable.
            self._by_short_name.setdefault(voice.short_name, voice)

        self._lock = asyncio.Lock()
        self._cache: "OrderedDict[Tuple[str, str], Tuple[bytes, int]]" = OrderedDict()
        self._cache_bytes = cache_bytes
        self._cache_used = 0
        self._cache_lock = threading.Lock()

    @property
    def voices(self) -> List[Voice]:
        return self._voices

    def resolve_voice(self, name: Optional[str]) -> Optional[Voice]:
        """Find a voice by its advertised name, falling back to the bare name."""
        if not name:
            return None

        voice = self._by_name.get(name)
        if voice is not None:
            return voice

        voice = self._by_short_name.get(name)
        if voice is not None:
            return voice

        _LOGGER.warning("Unknown voice %r, using the system default", name)
        return None

    def _cache_get(self, key: Tuple[str, str]) -> Optional[Tuple[bytes, int]]:
        if not self._cache_bytes:
            return None

        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is not None:
                self._cache.move_to_end(key)
            return entry

    def _cache_put(self, key: Tuple[str, str], audio: bytes, rate: int) -> None:
        if not self._cache_bytes or len(audio) > self._cache_bytes:
            return

        with self._cache_lock:
            if key in self._cache:
                self._cache_used -= len(self._cache.pop(key)[0])

            self._cache[key] = (audio, rate)
            self._cache_used += len(audio)

            while self._cache_used > self._cache_bytes:
                _, evicted = self._cache.popitem(last=False)
                self._cache_used -= len(evicted[0])

    async def synthesize(
        self, text: str, voice_name: Optional[str] = None
    ) -> Tuple[bytes, int]:
        """Return (16-bit PCM audio, sample rate) for the given text."""
        voice = self.resolve_voice(voice_name)
        key = (voice.name if voice else "", text)

        cached = self._cache_get(key)
        if cached is not None:
            _LOGGER.debug("Cache hit: %r", text)
            return cached

        # AVSpeechSynthesizer handles one utterance at a time.
        async with self._lock:
            audio, rate = await asyncio.wait_for(
                self._write_utterance(text, voice), timeout=_TIMEOUT
            )

        self._cache_put(key, audio, rate)
        return audio, rate

    def _write_utterance(self, text: str, voice: Optional[Voice]) -> asyncio.Future:
        """Start synthesis; the future resolves once the last buffer arrives."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        parts: List[bytes] = []
        rate = {"value": 0}

        def finish() -> None:
            if not future.done():
                future.set_result((b"".join(parts), rate["value"]))

        def callback(buffer) -> None:
            try:
                frames = int(buffer.frameLength())
            except Exception:  # the final buffer is empty/invalid
                frames = 0

            if frames == 0:
                loop.call_soon_threadsafe(finish)
                return

            rate["value"] = int(buffer.format().sampleRate())
            raw = buffer.floatChannelData()[0].as_buffer(frames * 4)
            samples = np.frombuffer(raw, dtype=np.float32, count=frames)
            parts.append((np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes())

        utterance = AVSpeechUtterance.speechUtteranceWithString_(text)
        if voice is not None:
            utterance.setVoice_(voice.av_voice)

        self._synth.writeUtterance_toBufferCallback_(utterance, callback)
        return future
