import logging
from typing import Optional

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice

from . import __version__
from .synth import Synthesizer

_LOGGER = logging.getLogger("wyoming-macos-tts")

_WYOMING_INFO_CACHE: Optional[Info] = None


def get_wyoming_info(args, synthesizer: Synthesizer) -> Info:
    """Build the info event from the installed voices."""
    global _WYOMING_INFO_CACHE
    if _WYOMING_INFO_CACHE is not None:
        return _WYOMING_INFO_CACHE

    voices = [
        TtsVoice(
            name=voice.name,
            description=voice.short_name,
            attribution=Attribution(name="Apple", url="https://apple.com"),
            installed=True,
            version=None,
            languages=[voice.language],
        )
        for voice in synthesizer.voices
    ]
    _LOGGER.debug("Found %d voices", len(voices))

    _WYOMING_INFO_CACHE = Info(
        tts=[
            TtsProgram(
                name=args.service_name,
                description="macos-tts",
                attribution=Attribution(name="Apple", url="https://apple.com"),
                installed=True,
                voices=voices,
                version=__version__,
                supports_synthesize_streaming=args.streaming,
            )
        ],
    )
    return _WYOMING_INFO_CACHE
