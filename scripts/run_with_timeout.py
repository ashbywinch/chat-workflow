#!/usr/bin/env python3
"""Run a command with a hard timeout."""

import argparse
import re
import subprocess
import sys
from pathlib import Path


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

    timed_out = False
    try:
        process.wait(timeout=args.timeout)
        reader.join(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        print(
            f"\n✕ TIMED OUT after {args.timeout}s",
            file=sys.stderr,
            flush=True,
        )
        timed_out = True

    # Print eval summary if a report file exists
    report_path = Path("test-results") / "eval-report.txt"
    if report_path.exists():
        entries = []
        total_tok = 0
        total_s = 0.0
        for line in report_path.read_text().strip("\n").split("\n"):
            m = re.match(r"\s+\[(.+?)\]\s+([\d.]+)s\s+(\d+)\s+tok", line)
            if m:
                entries.append(m.group(1))
                total_s += float(m.group(2))
                total_tok += int(m.group(3))
        if entries:
            tag = "⏰ PARTIAL (timed out)" if timed_out else "✓"
            print(
                f"\n{tag}  {len(entries)} tests  {total_s:.0f}s  {total_tok:,} tok",
                flush=True,
            )

    return 124 if timed_out else (process.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
