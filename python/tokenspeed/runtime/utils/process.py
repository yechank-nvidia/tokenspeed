# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Process and signal helpers."""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time

import psutil

logger = logging.getLogger(__name__)


def register_usr_signal():
    parent_process = psutil.Process().parent()

    def signal_handler(sig, frame):
        logger.error("recv usr signal, kill usr signal to parent")
        parent_process.send_signal(signal.SIGUSR1)

    signal.signal(signal.SIGUSR1, signal_handler)


def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    """Kill the target process and all of its child processes."""
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if include_parent:
        try:
            if parent_pid == os.getpid():
                itself.kill()
                sys.exit(0)

            itself.kill()
            itself.send_signal(signal.SIGQUIT)
        except psutil.NoSuchProcess:
            pass


def stop_owned_processes(processes, *, timeout_seconds):
    """Request TERM for all owned roots, reap, and fail on any forced/nonzero exit.

    The existing hard-kill helper is reserved for deadline survivors;
    requiring it remains a cleanup failure, even if the process is reaped.
    """
    if not 0 <= timeout_seconds < float("inf"):
        raise ValueError("timeout_seconds must be nonnegative and finite")
    deadline = time.monotonic() + timeout_seconds
    errors = []
    for process in processes:
        if process.is_alive():
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            except Exception as exc:
                errors.append(f"TERM {process.pid}: {exc}")
    for process in processes:
        try:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        except Exception as exc:
            errors.append(f"join {process.pid}: {exc}")
    survivors = [process for process in processes if process.is_alive()]
    if survivors:
        errors.append(
            f"forced cleanup required: {[process.pid for process in survivors]}"
        )
        for process in survivors:
            try:
                kill_process_tree(process.pid, include_parent=True)
            except Exception as exc:
                errors.append(f"KILL {process.pid}: {exc}")
        kill_deadline = time.monotonic() + 5.0
        for process in survivors:
            try:
                process.join(timeout=max(0.0, kill_deadline - time.monotonic()))
            except Exception as exc:
                errors.append(f"forced join {process.pid}: {exc}")
    for process in processes:
        if process.is_alive() or process.exitcode != 0:
            errors.append(
                f"child {process.pid} not cleanly reaped: exit={process.exitcode}"
            )
    if errors:
        raise RuntimeError("; ".join(errors))
