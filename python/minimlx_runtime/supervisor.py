"""Lifecycle hooks for Python runtimes supervised by the MiniMLX app."""

from __future__ import annotations

import os
import signal
import threading
import time


def install_parent_watchdog() -> None:
    """Exit the runtime if its owning MiniMLX process disappears.

    Rust destructors are not run when an IDE force-replaces a process. Watching
    for re-parenting closes that gap without affecting manually started Python
    runtimes, which do not receive ``MINIMLX_SUPERVISOR_PID``.
    """
    raw_parent_pid = os.environ.get("MINIMLX_SUPERVISOR_PID")
    if not raw_parent_pid:
        return
    try:
        parent_pid = int(raw_parent_pid)
    except ValueError:
        return
    if parent_pid <= 1:
        return

    shutdown_requested = threading.Event()

    def terminate_on_parent_exit(signum, _frame) -> None:
        shutdown_requested.set()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate_on_parent_exit)

    def watch() -> None:
        while os.getppid() == parent_pid:
            time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)
        if not shutdown_requested.wait(5):
            # A native MLX call can temporarily prevent Python from running
            # its signal handler. Never leave its large allocations orphaned.
            os._exit(128 + signal.SIGTERM)

    threading.Thread(target=watch, name="minimlx-parent-watchdog", daemon=True).start()
