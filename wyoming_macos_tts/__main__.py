#!/usr/bin/env python3
import argparse
import asyncio
import logging
import os
import threading
from functools import partial
from logging.handlers import TimedRotatingFileHandler

from CoreFoundation import CFRunLoopGetCurrent, CFRunLoopStop
from Foundation import NSDate, NSRunLoop

from wyoming.server import AsyncServer

from . import __version__
from .handler import MacosTTSEventHandler
from .synth import Synthesizer

_LOGGER = logging.getLogger("wyoming-macos-tts")


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--voice",
        help="Default voice to use (e.g., Daniel (English (UK)))",
    )
    parser.add_argument("--uri", default="stdio://", help="unix:// or tcp://")
    parser.add_argument(
        "--service-name",
        default="macos-tts",
        help="Name that will be sent in the info event",
    )
    #
    parser.add_argument(
        "--auto-punctuation", default=".?!", help="Automatically add punctuation"
    )
    parser.add_argument("--samples-per-chunk", type=int, default=1024)
    parser.add_argument(
        "--cache-mb",
        type=int,
        default=32,
        help="Megabytes of synthesized audio to keep for repeated phrases (0 disables)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable audio streaming on sentence boundaries",
    )
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format",
        default="%(asctime)s [%(levelname)s] %(message)s",
        help="Format for log messages",
    )
    parser.add_argument(
        "--log-dir",
        help="Directory to store the logs (leave empty to not save any logs)",
    )
    parser.add_argument(
        "--log-keep-days", type=int, default=7, help="Number of days to keep logs"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        handler = TimedRotatingFileHandler(
            os.path.join(args.log_dir, "app.log"),
            when="midnight",
            backupCount=args.log_keep_days,
        )
        handler.setFormatter(logging.Formatter(args.log_format))
        # Attach to the root logger so library output and unhandled
        # exceptions end up in the log file too, not just our own messages.
        logging.getLogger().addHandler(handler)
    _LOGGER.debug(f"Starting server with args: {args}")

    synthesizer = Synthesizer(cache_bytes=max(0, args.cache_mb) * 1024 * 1024)
    _LOGGER.debug("Loaded %d voices", len(synthesizer.voices))

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")

    await server.run(
        partial(
            MacosTTSEventHandler,
            args,
            synthesizer,
        )
    )


def run():
    """Serve on a worker thread; the main thread runs the CFRunLoop.

    AVSpeechSynthesizer delivers its audio buffers on the main queue, so the
    main thread has to stay available to process them.
    """
    error: list = []

    def serve() -> None:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            pass
        except BaseException as err:  # surface it after the run loop stops
            error.append(err)
        finally:
            CFRunLoopStop(CFRunLoopGetCurrent())

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    run_loop = NSRunLoop.currentRunLoop()
    while thread.is_alive():
        run_loop.runMode_beforeDate_(
            "kCFRunLoopDefaultMode", NSDate.dateWithTimeIntervalSinceNow_(0.05)
        )

    if error:
        raise error[0]


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
