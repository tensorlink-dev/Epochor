"""Minimal entry point for running the Epochor validator."""

import asyncio
import logging

from neurons.validator import Validator


async def _run_validator_loop():
    """Spin up the validator and execute its evaluation loop forever."""
    with Validator() as validator:
        logging.info("Validator services started. Entering evaluation loop.")
        while True:
            await validator.run_step()


def main() -> None:
    """Run the asynchronous validator loop and handle shutdown signals."""
    try:
        asyncio.run(_run_validator_loop())
    except KeyboardInterrupt:
        logging.info("Validator shutdown requested by user.")


if __name__ == "__main__":
    main()
