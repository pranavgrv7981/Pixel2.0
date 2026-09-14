"""Pixel v2 — Entry Point.

Starts the PixelRuntime and runs a minimal console REPL.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.core.config import PixelConfig, load_config
from app.core.logging import setup_logging, get_logger
from app.runtime.runtime import PixelRuntime


async def run_repl(runtime: PixelRuntime) -> None:
    """Run a minimal console Read-Eval-Print Loop."""
    logger = get_logger("pixel.repl")

    print(f"\n{'='*50}")
    print(f"  {runtime.config.runtime.name} v{runtime.config.runtime.version}")
    print(f"  State: {runtime.state.value}")
    print(f"  Session: {runtime.session_id}")
    print(f"{'='*50}")
    print("  Type a message or 'quit' to exit.\n")

    loop = asyncio.get_event_loop()

    while runtime.running:
        try:
            # Run blocking input in executor to not block the event loop
            user_input = await loop.run_in_executor(None, lambda: input("You: "))
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break

        text = user_input.strip()
        if not text:
            continue
        if text.lower() in ("quit", "exit", "bye"):
            print("\nPixel: Goodbye! 👋")
            break

        response = await runtime.handle_input(text)
        print(f"\nPixel: {response}\n")


async def main(config_path: str | None = None) -> None:
    """Main entry point."""
    # Load configuration
    config = load_config(config_path)

    # Setup logging early
    setup_logging(
        level=config.logging.level,
        fmt=config.logging.format,
        log_file=config.logging.file,
    )

    logger = get_logger("pixel.main")

    # Create and start runtime
    runtime = PixelRuntime(config)

    try:
        await runtime.start()
        await run_repl(runtime)
    except KeyboardInterrupt:
        logger.info("keyboard_interrupt")
    except Exception as exc:
        logger.error("fatal_error", error=str(exc), error_type=type(exc).__name__)
        print(f"\nFatal error: {exc}", file=sys.stderr)
    finally:
        await runtime.stop()
        logger.info("pixel_exited")


def cli() -> None:
    """CLI argument parsing."""
    parser = argparse.ArgumentParser(
        prog="pixel",
        description="Pixel v2 — Intelligent Local AI Runtime",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to TOML configuration file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override log level",
    )
    args = parser.parse_args()

    # Apply CLI overrides via env vars (picked up by config loader)
    import os
    if args.log_level:
        os.environ["PIXEL_LOGGING__LEVEL"] = args.log_level

    asyncio.run(main(config_path=args.config))


if __name__ == "__main__":
    cli()
