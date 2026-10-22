"""Execution-log lines for the VOLT panel, shared by the plugin's modules.

STDOUT carries the persistent host's binary protocol, so every line goes to
STDERR. Durations read like coretoolkit's ``Volt::formatDuration``.
"""

import sys
import time


def log(message):
    """Emit one line to STDERR (STDOUT is reserved by the host)."""
    print(message, file=sys.stderr, flush=True)


def warn(message):
    log('Warning: ' + message)


def format_duration(seconds):
    """Elapsed wall time as a log line shows it: '420 ms', '3.18 s', '2 min 05 s'."""
    if seconds < 1.0:
        return '%.0f ms' % (seconds * 1000.0)
    if seconds < 60.0:
        return '%.2f s' % seconds
    whole = int(seconds)
    return '%d min %02d s' % (whole // 60, whole % 60)


class Stopwatch:
    def __init__(self):
        self._start = time.monotonic()

    def restart(self):
        self._start = time.monotonic()

    def elapsed(self):
        return format_duration(time.monotonic() - self._start)
