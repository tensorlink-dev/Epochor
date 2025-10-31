"""
Model Manager
=============

Keeps the local model store in sync with miners’ on-chain metadata and queues
models for evaluation. Also prunes stale local copies.

Key features:
- Single persistent asyncio loop for all async work (no repeated event-loop creation).
- Dynamic validator exclusion (never attempts to sync/evaluate validator UIDs).
- Compatible call to metagraph_utils.get_top_miners across forks/signatures.
- Backpressure: honors `constants.updated_models_limit`.
- Safe against metagraph size changes while iterating.
"""

from __future__ import annotations

import asyncio
import concurrent.futures as cf
import copy
import datetime as dt
import math
import threading
import time
import typing
import traceback

import bittensor as bt

import constants
from competitions import competitions
from epochor.model.model_data import EvalResult
from epochor.model.model_updater import MinerMisconfiguredError
from epochor.model.storage.disk_model_store import DiskModelStore
from epochor.utils import metagraph_utils


class ModelManager:
    """
    Coordinates model syncing, queuing for evaluation, and local cleanup.

    Lifecycle
    ---------
    - `start()` launches:
        * a persistent asyncio loop thread (used to run async coroutines),
        * the update loop thread (download + queue models),
        * the cleanup loop thread (delete unreferenced models).
    - `stop()` signals loops to stop, joins threads, and shuts down the asyncio loop.

    Exclusions
    ----------
    - Dynamic: any UID marked as a **validator** in the metagraph is skipped.
      (Looks for `metagraph.validator_permit` or `metagraph.is_validator`.)
    - Optional config overrides:
        * `constants.NON_MINER_UIDS` (set of UIDs)
        * `constants.NON_MINER_HOTKEYS` (set of hotkey strings)
    """

    def __init__(
        self,
        model_updater,
        model_tracker,
        miner_iterator,
        metagraph,
        state,
        metagraph_lock: threading.RLock,
        local_store: DiskModelStore,
        get_current_block_fn: typing.Callable[[], int],
    ):
        self.model_updater = model_updater
        self.model_tracker = model_tracker
        self.miner_iterator = miner_iterator
        self.metagraph = metagraph
        self.state = state

        self.metagraph_lock = metagraph_lock
        self.local_store = local_store
        self.get_current_block = get_current_block_fn

        # Threads + stop signal.
        self.stop_event = threading.Event()
        self.update_thread = threading.Thread(target=self._update_models_loop, name="ModelManager-Update", daemon=True)
        self.clean_thread = threading.Thread(target=self._clean_models_loop, name="ModelManager-Clean", daemon=True)

        # Exclusions: config + dynamic validator detection.
        self.excluded_uids = set(getattr(constants, "NON_MINER_UIDS", set()))
        self.excluded_hotkeys = set(getattr(constants, "NON_MINER_HOTKEYS", set()))

        # Persistent asyncio loop (so we don't recreate loops repeatedly).
        self._aio_loop: typing.Optional[asyncio.AbstractEventLoop] = None
        self._aio_thread: typing.Optional[threading.Thread] = None
        self._aio_ready = threading.Event()

    # ---------------------------------------------------------------------
    # Event loop management
    # ---------------------------------------------------------------------
    def _start_asyncio_loop(self) -> None:
        """Start a single persistent asyncio event loop in a background thread."""
        if self._aio_loop and self._aio_thread and self._aio_thread.is_alive():
            return

        self._aio_ready.clear()

        def _runner():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._aio_loop = loop
            self._aio_ready.set()
            loop.run_forever()
            # Graceful shutdown
            try:
                pending = asyncio.all_tasks(loop)
                for t in pending:
                    t.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            finally:
                loop.close()

        self._aio_thread = threading.Thread(target=_runner, name="ModelManager-AIO", daemon=True)
        self._aio_thread.start()
        self._aio_ready.wait(timeout=5.0)

    def _stop_asyncio_loop(self) -> None:
        """Stop the persistent asyncio event loop thread."""
        if not self._aio_loop:
            return
        try:
            self._aio_loop.call_soon_threadsafe(self._aio_loop.stop)
        except Exception:
            pass
        if self._aio_thread:
            self._aio_thread.join(timeout=5.0)
        self._aio_loop = None
        self._aio_thread = None

    def _run_coro(self, coro: typing.Coroutine, timeout: float = 180.0):
        """
        Run an async coroutine on the persistent loop, waiting up to `timeout` seconds.
        Returns the coroutine's result or raises on timeout/errors.
        """
        if not self._aio_loop:
            self._start_asyncio_loop()
        fut = asyncio.run_coroutine_threadsafe(coro, self._aio_loop)
        try:
            return fut.result(timeout=timeout)
        except cf.TimeoutError:
            fut.cancel()
            bt.logging.error(f"Coroutine timed out after {timeout}s")
            raise

    # ---------------------------------------------------------------------
    # Public lifecycle
    # ---------------------------------------------------------------------
    def start(self) -> None:
        """Start the async loop + update/clean worker threads."""
        self._start_asyncio_loop()
        self.update_thread.start()
        self.clean_thread.start()

    def stop(self, join: bool = True) -> None:
        """Signal workers to stop; optionally join threads; stop async loop."""
        self.stop_event.set()
        if join:
            if self.update_thread.is_alive():
                self.update_thread.join(timeout=5.0)
            if self.clean_thread.is_alive():
                self.clean_thread.join(timeout=5.0)
        self._stop_asyncio_loop()

    # ---------------------------------------------------------------------
    # Helpers: metagraph + exclusion + top miners
    # ---------------------------------------------------------------------
    def _metagraph_size(self) -> int:
        """Return current metagraph size under lock (fallbacks to hotkeys length)."""
        with self.metagraph_lock:
            try:
                return len(self.metagraph.uids)
            except Exception:
                return len(getattr(self.metagraph, "hotkeys", []))

    def _is_valid_uid(self, uid: int) -> bool:
        """True if UID is in-bounds for the current metagraph."""
        n = self._metagraph_size()
        return isinstance(uid, int) and 0 <= uid < n

    def _is_validator_uid(self, uid: int) -> bool:
        """Dynamic role check: True if metagraph marks this UID as a validator."""
        with self.metagraph_lock:
            try:
                n = len(self.metagraph.hotkeys)
                if uid < 0 or uid >= n:
                    return False
                # Common in BT forks: validator_permit is a per-uid sequence/bool
                if hasattr(self.metagraph, "validator_permit"):
                    vp = self.metagraph.validator_permit
                    try:
                        return bool(vp[uid])
                    except Exception:
                        pass
                # Some forks expose is_validator
                if hasattr(self.metagraph, "is_validator"):
                    iv = self.metagraph.is_validator
                    try:
                        return bool(iv[uid])
                    except Exception:
                        pass
            except Exception:
                return False
        return False

    def _is_excluded(self, uid: int) -> bool:
        """
        True if UID should be skipped entirely:
          - listed in NON_MINER_UIDS,
          - dynamically detected validator,
          - or hotkey present in NON_MINER_HOTKEYS (if provided).
        """
        if uid in self.excluded_uids:
            return True
        if self._is_validator_uid(uid):
            return True
        if self.excluded_hotkeys:
            with self.metagraph_lock:
                try:
                    if 0 <= uid < len(self.metagraph.hotkeys):
                        return self.metagraph.hotkeys[uid] in self.excluded_hotkeys
                except Exception:
                    pass
        return False

    def _get_top_miners_compat(self, metagraph, min_vali_stake: typing.Union[int, float], min_miner_pct: float):
        """
        Call metagraph_utils.get_top_miners() across differing signatures:
          - get_top_miners(metagraph, min_validator_stake, min_miner_percent)  # positional
          - get_top_miners(metagraph)                                         # hardcoded thresholds
        """
        fn = metagraph_utils.get_top_miners
        try:
            # Try positional first (widest compatibility).
            return fn(metagraph, min_vali_stake, min_miner_pct)
        except TypeError:
            # Fall back to bare call if fork doesn't accept thresholds.
            return fn(metagraph)

    # ---------------------------------------------------------------------
    # Core loops
    # ---------------------------------------------------------------------
    def _update_models_loop(self) -> None:
        """
        Download/refresh models for miners and queue them for evaluation.

        Behavior:
        - At a cadence, also queue "top miners" (miners weighted by top validators).
        - Respects updated_models_limit backpressure.
        - Avoids excluded/validator UIDs.
        - Uses a persistent asyncio loop to run async model syncs.
        """
        # Track recency of sequential checks.
        uid_last_checked_sequential: dict[int, dt.datetime] = {}
        last_checked_top_models_time: typing.Optional[dt.datetime] = None

        # Allow metagraph syncers to populate initially.
        time.sleep(60)

        scan_top_model_cadence = getattr(constants, "scan_top_model_cadence", dt.timedelta(minutes=10))
        chain_update_cadence = getattr(constants, "chain_update_cadence", dt.timedelta(minutes=5))

        while not self.stop_event.is_set():
            try:
                # Periodically queue "top miners" seen by larger validators.
                if (
                    not last_checked_top_models_time
                    or (dt.datetime.now() - last_checked_top_models_time) > scan_top_model_cadence
                ):
                    last_checked_top_models_time = dt.datetime.now()
                    self._queue_top_models_for_eval()

                # Ensure capacity before downloading more.
                self._wait_for_open_eval_slot()

                # Next candidate UID from miner iterator.
                next_uid = next(self.miner_iterator)

                # Skip excluded roles (validators / configured exclusions).
                if self._is_excluded(next_uid):
                    bt.logging.trace(f"[update_models] Skipping excluded UID {next_uid}.")
                    continue

                # Metagraph may have grown/shrunk.
                if not self._is_valid_uid(next_uid):
                    bt.logging.debug(
                        f"[update_models] Skipping invalid UID {next_uid}; metagraph_n={self._metagraph_size()}."
                    )
                    continue

                # Throttle revisits of the same UID within chain_update_cadence.
                now = dt.datetime.now()
                last_seen = uid_last_checked_sequential.get(next_uid)
                if last_seen and (now - last_seen) < chain_update_cadence:
                    # Too soon to re-check this UID — skip and move on to the next one.
                    # (Optional tiny backoff to avoid busy spin if many are recent.)
                    time.sleep(0.05)
                    continue

                # Only stamp when we actually proceed with this UID
                uid_last_checked_sequential[next_uid] = now

                curr_block = self.get_current_block()

                # Resolve hotkey under lock.
                with self.metagraph_lock:
                    hotkey = self.metagraph.hotkeys[next_uid]

                # Decide if we should force a refresh of the model for this UID.
                force_sync = False
                submission_snapshot = self.model_tracker.get_submission_for_miner_hotkey(hotkey)

                if submission_snapshot:
                    # Is it already queued for eval for this competition?
                    comp_id = submission_snapshot.competition_id
                    is_queued = (
                        next_uid in self.state.get_pending_uids_to_eval(comp_id)
                        or next_uid in self.state.get_uids_to_eval(comp_id)
                    )

                    if not is_queued:
                        # Simple retry heuristic without pulling epsilon deps:
                        # If it's been at least `model_retry_cadence` blocks since last eval, re-download.
                        last_eval_block = self.model_tracker.get_block_last_evaluated(hotkey) or 0
                        model_retry_cadence = getattr(constants, "model_retry_cadence", 0)
                        if curr_block - last_eval_block >= model_retry_cadence:
                            force_sync = True
                            bt.logging.debug(
                                f"[update_models] Forcing retry for UID {next_uid}; "
                                f"last_eval_block={last_eval_block}, curr_block={curr_block}"
                            )

                # Run the (possibly forced) sync.
                try:
                    updated = self._run_coro(
                        self.model_updater.sync_model(
                            uid=next_uid,
                            hotkey=hotkey,
                            curr_block=curr_block,
                            schedule=competitions.COMPETITION_SCHEDULE_BY_BLOCK,
                            force=force_sync,
                        ),
                        timeout=180.0,
                    )
                except MinerMisconfiguredError as e:
                    # Record a failure eval result to surface misconfigs in trackers.
                    bt.logging.warning(f"Failed to sync UID {next_uid}: [{hotkey}] {e}")
                    self.model_tracker.on_model_evaluated(
                        hotkey,
                        0,
                        EvalResult(
                            block=curr_block,
                            score=math.inf,
                            winning_model_block=0,
                            winning_model_score=0,
                        ),
                    )
                    updated = False

                if updated:
                    # Newly updated submission → schedule for evaluation on its competition.
                    snapshot = self.model_tracker.get_submission_for_miner_hotkey(hotkey)
                    if snapshot is not None:
                        self.state.add_pending_uid_to_eval(snapshot.competition_id, next_uid)
                        bt.logging.debug(
                            f"[update_models] New submission for UID={next_uid} (comp={snapshot.competition_id}) queued for eval."
                        )
                        # Reset EMA for this UID so it fairly re-enters leaderboards.
                        try:
                            self.state.reset_ema_uid(next_uid)
                        except Exception:
                            pass
                    else:
                        bt.logging.warning(
                            f"[update_models] Updated submission for UID {next_uid} but snapshot missing for hotkey {hotkey}."
                        )

            except StopIteration:
                # If iterator exhausts, brief nap and continue.
                time.sleep(1.0)
            except Exception as e:
                bt.logging.error(f"Error in update loop: {e}\n{traceback.format_exc()}")

        bt.logging.info("Exiting update models loop.")

    def _queue_top_models_for_eval(self) -> None:
        """
        Adds miners considered "top" by high-stake validators to the eval queue
        if they're not already pending/currently evaluated.

        This is competition-agnostic: any miner winning weight deserves a re-check.
        """
        # Work off a snapshot to avoid concurrent mutation.
        with self.metagraph_lock:
            metagraph = copy.deepcopy(self.metagraph)

        # Determine "top miners" with signature compatibility.
        min_vali_stake = getattr(constants, "WEIGHT_SYNC_VALI_MIN_STAKE", 0)
        min_miner_pct = getattr(constants, "WEIGHT_SYNC_MINER_MIN_PERCENT", 0.0)
        try:
            top_miner_uids = set(self._get_top_miners_compat(metagraph, min_vali_stake, min_miner_pct))
        except Exception as e:
            bt.logging.debug(f"[queue_top_models] get_top_miners failed: {e}")
            top_miner_uids = set()

        # Aggregate already queued/pending UIDs across all competitions.
        all_uids_to_eval = set()
        all_pending_uids_to_eval = set()
        with self.state.pending_uids_to_eval_lock:
            for uids in self.state.uids_to_eval.values():
                all_uids_to_eval.update(uids)
            for uids in self.state.pending_uids_to_eval.values():
                all_pending_uids_to_eval.update(uids)

        # Pick only those not already queued and not excluded.
        base = top_miner_uids - all_uids_to_eval - all_pending_uids_to_eval
        candidates = {u for u in base if self._is_valid_uid(u) and not self._is_excluded(u)}

        curr_block = self.get_current_block()
        model_retry_cadence = getattr(constants, "model_retry_cadence", 0)

        for uid in sorted(candidates):
            try:
                hotkey = metagraph.hotkeys[uid]
            except Exception:
                bt.logging.debug(f"[queue_top_models] UID {uid} vanished from snapshot.")
                continue

            # Avoid hammering the same miner: space retries by block cadence.
            last_eval_block = self.model_tracker.get_block_last_evaluated(hotkey) or 0
            if curr_block - last_eval_block < model_retry_cadence:
                continue

            try:
                # Force re-download/top-model check. Respects per-competition eval delays in updater.
                should_retry = self._run_coro(
                    self.model_updater.sync_model(
                        uid=uid,
                        hotkey=hotkey,
                        curr_block=curr_block,
                        schedule=competitions.COMPETITION_SCHEDULE_BY_BLOCK,
                        force=True,
                    ),
                    timeout=180.0,
                )
            except MinerMisconfiguredError as e:
                bt.logging.warning(f"Failed to sync model for UID {uid}: {e}")
                self.model_tracker.on_model_evaluated(
                    hotkey,
                    0,
                    EvalResult(
                        block=curr_block,
                        score=math.inf,
                        winning_model_block=0,
                        winning_model_score=0,
                    ),
                )
                should_retry = False
            except Exception as e:
                bt.logging.debug(f"[queue_top_models] Exception while syncing UID={uid}: {e}")
                should_retry = False

            if not should_retry:
                continue

            snapshot = self.model_tracker.get_submission_for_miner_hotkey(hotkey)
            if snapshot is None:
                bt.logging.warning(f"[queue_top_models] No submission snapshot after sync for UID {uid} ({hotkey}).")
                continue

            # Queue for evaluation on its competition (do not enforce pending/full cap here).
            self.state.add_pending_uid_to_eval(snapshot.competition_id, uid)
            bt.logging.trace(
                f"[queue_top_models] Queued UID={uid} (comp={snapshot.competition_id}) due to top-miner status."
            )

    def _wait_for_open_eval_slot(self) -> None:
        """
        Wait until there's at least one open slot to download & evaluate more models.
        Uses constants.updated_models_limit to bound pending + current load.
        """
        updated_models_limit = int(getattr(constants, "updated_models_limit", 32))
        while not self.stop_event.is_set():
            pending_uid_count, current_uid_count = self.state.get_pending_and_current_uid_counts()
            total = pending_uid_count + current_uid_count
            if total < updated_models_limit:
                return
            bt.logging.info(
                f"[update_models] {total} models pending/current (limit={updated_models_limit}). "
                f"Checking again in 300s."
            )
            time.sleep(300)

    def _clean_models_loop(self) -> None:
        """
        Periodically deletes models from local storage that are no longer referenced.

        Strategy:
        - Keep models for any hotkey currently pending/current in *any* competition.
        - Skip excluded/validator UIDs.
        - Grace period before deletion (default: 600s).
        """
        # Give the update loop time to recreate tracker state after restarts/upgrades.
        time.sleep(dt.timedelta(hours=1).total_seconds())

        grace_seconds = int(getattr(constants, "LOCAL_STORE_CLEAN_GRACE_SECONDS", 600))

        while not self.stop_event.is_set():
            try:
                bt.logging.trace("[clean_models] Starting cleanup of stale models.")

                # Map of all hotkeys → submission snapshots.
                hotkey_to_submission = self.model_tracker.get_miner_hotkey_to_submission_dict()
                hotkey_to_model_id = {
                    hk: snapshot.model_id for hk, snapshot in hotkey_to_submission.items()
                }

                # Collect UIDs we must keep (pending or current across competitions).
                uids_to_keep: set[int] = set()
                with self.state.pending_uids_to_eval_lock:
                    for pending in self.state.pending_uids_to_eval.values():
                        uids_to_keep.update(pending)
                    for current in self.state.uids_to_eval.values():
                        uids_to_keep.update(current)

                # Convert UIDs → hotkeys, filtering invalid/excluded/validators.
                hotkeys_to_keep: set[str] = set()
                with self.metagraph_lock:
                    for uid in uids_to_keep:
                        if not self._is_valid_uid(uid) or self._is_excluded(uid):
                            continue
                        try:
                            hotkeys_to_keep.add(self.metagraph.hotkeys[uid])
                        except Exception:
                            pass

                # Keep only those hotkeys' models.
                evaluated_hotkeys_to_model_id = {
                    hk: mid for hk, mid in hotkey_to_model_id.items()
                    if hk in hotkeys_to_keep
                }

                # Delete everything else (with a grace period).
                self.local_store.delete_unreferenced_models(
                    valid_models_by_hotkey=evaluated_hotkeys_to_model_id,
                    grace_period_seconds=grace_seconds,
                )

            except Exception as e:
                bt.logging.error(f"Error in clean loop: {e}\n{traceback.format_exc()}")

            # Run every 5 minutes.
            time.sleep(dt.timedelta(minutes=5).total_seconds())

        bt.logging.info("Exiting clean models loop.")
