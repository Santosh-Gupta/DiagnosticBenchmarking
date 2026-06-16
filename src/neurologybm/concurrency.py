"""Small concurrency helpers for API-backed case pipelines."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class StartRateLimiter:
    """Thread-safe limiter for spacing request starts.

    DeepSeek currently enforces concurrent-connection limits rather than token-per-minute limits.
    The spacing hook exists so future providers with RPM/TPM-style quotas can be throttled without
    changing the case pipeline shape.
    """

    def __init__(self, min_interval_seconds: float = 0.0) -> None:
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must be non-negative")
        self._min_interval_seconds = min_interval_seconds
        self._lock = threading.Lock()
        self._next_start = 0.0

    def wait(self) -> None:
        if self._min_interval_seconds <= 0:
            return
        with self._lock:
            now = time.monotonic()
            sleep_seconds = max(0.0, self._next_start - now)
            self._next_start = max(now, self._next_start) + self._min_interval_seconds
        if sleep_seconds:
            time.sleep(sleep_seconds)


def run_ordered_concurrent(
    items: Iterable[InputT],
    worker: Callable[[InputT], OutputT],
    *,
    concurrency: int = 1,
    request_spacing_seconds: float = 0.0,
) -> Iterator[OutputT]:
    """Run item workers concurrently while yielding results in input order."""

    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if request_spacing_seconds < 0:
        raise ValueError("request_spacing_seconds must be non-negative")

    materialized = list(items)
    if concurrency == 1 or len(materialized) <= 1:
        limiter = StartRateLimiter(request_spacing_seconds)
        for item in materialized:
            limiter.wait()
            yield worker(item)
        return

    limiter = StartRateLimiter(request_spacing_seconds)

    def wrapped(index: int, item: InputT) -> tuple[int, OutputT]:
        limiter.wait()
        return index, worker(item)

    next_index = 0
    pending: dict[int, OutputT] = {}
    max_workers = min(concurrency, len(materialized))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(wrapped, index, item): index
            for index, item in enumerate(materialized)
        }
        for future in as_completed(futures):
            index, result = future.result()
            pending[index] = result
            while next_index in pending:
                yield pending.pop(next_index)
                next_index += 1
