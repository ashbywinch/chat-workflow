#!/usr/bin/env python3
"""
Common test configuration and fixtures.
"""

import signal
import sys
from functools import wraps
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TimeoutError(Exception):
    """Raised when a test exceeds its timeout."""

    pass


def timeout(seconds: int):
    """
    Decorator that raises TimeoutError if the wrapped function exceeds the timeout.

    Usage:
        @timeout(10)
        def test_something(self):
            ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"Test timed out after {seconds} seconds")

            # Set the signal handler
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)

            try:
                return func(*args, **kwargs)
            finally:
                # Restore original handler and cancel alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        return wrapper

    return decorator
