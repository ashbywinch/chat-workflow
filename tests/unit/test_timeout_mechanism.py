"""Tests for the eval harness timeout mechanism.

The timeout must interrupt a blocking network call and must not be swallowed by
the LLM client's ``except Exception`` handlers. It therefore derives from
``BaseException``, not ``Exception``.
"""

import socket
import time
import unittest

from tests.conftest import HarnessTimeout, timeout


class TestHarnessTimeout(unittest.TestCase):
    def test_timeout_is_not_an_exception_subclass(self):
        self.assertTrue(issubclass(HarnessTimeout, BaseException))
        self.assertFalse(issubclass(HarnessTimeout, Exception))

    def test_blocking_socket_is_interrupted(self):
        def block() -> None:
            left, right = socket.socketpair()
            try:
                left.settimeout(5)
                left.recv(1)
            finally:
                left.close()
                right.close()

        decorated = timeout(1)(block)
        start = time.time()

        with self.assertRaises(HarnessTimeout):
            decorated()

        self.assertLess(time.time() - start, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
