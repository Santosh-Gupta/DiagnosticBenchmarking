import time
import unittest

from neurologybm.concurrency import StartRateLimiter, run_ordered_concurrent


class ConcurrencyTests(unittest.TestCase):
    def test_run_ordered_concurrent_preserves_input_order(self) -> None:
        def worker(value: int) -> int:
            time.sleep(0.005 * (4 - value))
            return value

        output = list(run_ordered_concurrent([1, 2, 3], worker, concurrency=3))

        self.assertEqual(output, [1, 2, 3])

    def test_run_ordered_concurrent_rejects_invalid_concurrency(self) -> None:
        with self.assertRaises(ValueError):
            list(run_ordered_concurrent([1], lambda value: value, concurrency=0))

    def test_start_rate_limiter_rejects_negative_interval(self) -> None:
        with self.assertRaises(ValueError):
            StartRateLimiter(-0.1)

    def test_run_ordered_concurrent_serial_path(self) -> None:
        calls = []

        def worker(value: int) -> int:
            calls.append(value)
            return value * 2

        output = list(run_ordered_concurrent([1, 2, 3], worker, concurrency=1))

        self.assertEqual(calls, [1, 2, 3])
        self.assertEqual(output, [2, 4, 6])


if __name__ == "__main__":
    unittest.main()
