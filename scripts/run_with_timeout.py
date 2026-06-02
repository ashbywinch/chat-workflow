#!/usr/bin/env python3
"""Run a command with a hard timeout."""

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a command with timeout and propagate exit code"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        required=True,
        help="Timeout in seconds before terminating the command",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run after --",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]

    if not command:
        print("No command specified", file=sys.stderr)
        return 2

    import threading

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def _stream():
        for line in iter(process.stdout.readline, ""):
            print(line, end="", flush=True)

    reader = threading.Thread(target=_stream, daemon=True)
    reader.start()

    try:
        process.wait(timeout=args.timeout)
        reader.join(timeout=5)
        return process.returncode
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        print(
            f"\nCommand timed out after {args.timeout} seconds: {' '.join(command)}",
            file=sys.stderr,
            flush=True,
        )
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
