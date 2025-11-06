from __future__ import annotations

from epochor.training.validator_contract import MinerSubmissionProtocol as _BaseProtocol


class MinerSubmissionProtocol(_BaseProtocol):
    """Compatibility wrapper that mirrors the core training protocol."""

    # The protocol definition is inherited verbatim from the canonical module.
    # The explicit subclass keeps the public surface identical for miners while
    # letting template-specific utilities import from a stable path.
    pass


__all__ = ["MinerSubmissionProtocol"]
