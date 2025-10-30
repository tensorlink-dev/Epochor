"""Handles the process of setting weights on the Bittensor network in a background thread."""

import typing
import threading
import asyncio
import traceback
from datetime import timedelta

import bittensor as bt
import torch
import numpy as np  # only for typing; we convert via .numpy()
import constants


class WeightSetter:
    """
    Periodically sets weights on-chain in a dedicated background thread.
    Handles acquiring locks for metagraph and weights, and executes the
    blocking `set_weights` call in a thread pool to avoid stalling the async loop.
    """
    def __init__(
        self,
        subtensor: "bt.subtensor",
        wallet: "bt.wallet",
        netuid: int,
        metagraph: "bt.metagraph",
        weights: torch.Tensor,
        metagraph_lock: threading.RLock,
        cadence: typing.Union[float, int, timedelta] = None,
    ):
        """Initializes the WeightSetter."""
        self.subtensor = subtensor
        self.wallet = wallet
        self.netuid = netuid
        self.metagraph = metagraph
        self.weights = weights
        self.metagraph_lock = metagraph_lock
        self.weight_lock = threading.RLock()
        self.stop_event = threading.Event()
        self._thread: typing.Optional[threading.Thread] = None
        
        cadence = cadence or getattr(constants, "set_weights_cadence", 5400)
        self.cadence_s = float(cadence.total_seconds() if isinstance(cadence, timedelta) else cadence)
        self.cadence_s = max(120.0, self.cadence_s)  # Ensure a reasonable minimum cadence.

    def start(self) -> None:
        """Starts the background weight-setting thread."""
        if not (self._thread and self._thread.is_alive()):
            self.stop_event.clear()
            self._thread = threading.Thread(target=self._thread_main, name="WeightSetter", daemon=True)
            self._thread.start()
            bt.logging.info(f"WeightSetter started; cadence={self.cadence_s:.1f}s")

    def stop(self, join: bool = True, timeout: float = 5.0) -> None:
        """Signals the loop to stop and optionally joins the thread."""
        self.stop_event.set()
        if join and self._thread and self._thread.is_alive():
            self._thread.join(timeout)

    def _thread_main(self) -> None:
        """The entry point for the background thread."""
        try:
            asyncio.run(self._async_loop())
        except Exception as e:
            bt.logging.error(f"WeightSetter loop crashed: {e}\n{traceback.format_exc()}")

    async def _async_loop(self) -> None:
        """The async loop that periodically calls the weight setting logic."""
        backoff = 5.0
        while not self.stop_event.is_set():
            try:
                ok, msg = await self._set_weights(ttl=120)
                backoff = 5.0 if ok else min(backoff * 2, 300.0)
            except Exception:
                bt.logging.error(f"Error during _set_weights: {traceback.format_exc()}")
                backoff = min(backoff * 2, 300.0)

            # Sleep in small chunks to respond quickly to the stop event.
            delay = self.cadence_s if backoff <= 5.0 else backoff
            for _ in range(int(delay / 0.25)):
                if self.stop_event.is_set():
                    break
                await asyncio.sleep(0.25)

    async def _set_weights(self, ttl: int = 60) -> typing.Tuple[bool, str]:
        """Executes the blocking `set_weights` call in a separate thread."""
        loop = asyncio.get_running_loop()

        def _blocking_call() -> typing.Tuple[bool, str]:
            try:
                with self.metagraph_lock:
                    uids = self.metagraph.uids
                with self.weight_lock:
                    w = self.weights.detach().to("cpu")
                    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
                    weights_to_set = w.numpy()  # bittensor typically accepts list/np.ndarray

                bt.logging.info("Setting weights on-chain...")
                return self.subtensor.set_weights(
                    netuid=self.netuid, wallet=self.wallet, uids=uids,
                    weights=weights_to_set, wait_for_inclusion=True,
                    version_key=getattr(constants, "weights_version_key", 0),
                    max_retries=1
                )
            except Exception as e:
                return False, str(e)

        try:
            status = await asyncio.wait_for(loop.run_in_executor(None, _blocking_call), timeout=ttl)
            bt.logging.info(f"Finished setting weights with status: {status}")
            return status
        except asyncio.TimeoutError:
            bt.logging.error(f"Failed to set weights after {ttl} seconds (timeout).")
            return False, f"Timeout after {ttl} seconds"
