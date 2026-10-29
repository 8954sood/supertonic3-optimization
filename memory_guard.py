"""Linux benchmark supervisor; stops only its own child when RAM gets tight.
This is a sampled guard, not a cgroup hard limit. No other processes are touched.
"""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
import psutil

p = argparse.ArgumentParser()
p.add_argument("--max-rss-mb", type=float, default=420)
p.add_argument("--reserve-mb", type=float, default=80)
p.add_argument("--seconds", type=float, default=600)
p.add_argument("--stats", required=True)
p.add_argument("command", nargs=argparse.REMAINDER)
a = p.parse_args()
command = a.command[1:] if a.command[:1] == ["--"] else a.command
if not command:
    p.error("provide a command after --")
if os.name != "posix":
    p.error("this supervisor requires POSIX process groups")
child = subprocess.Popen(command, start_new_session=True)
proc = psutil.Process(child.pid)
start = time.monotonic()
peak = 0.0
tree_peak = 0.0
minimum = psutil.virtual_memory().available / 1e6
reason = None
while child.poll() is None:
    try:
        rss = proc.memory_info().rss / 1e6
        tree_rss = rss
        for descendant in proc.children(recursive=True):
            try:
                tree_rss += descendant.memory_info().rss / 1e6
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        break
    available = psutil.virtual_memory().available / 1e6
    peak = max(peak, rss)
    tree_peak = max(tree_peak, tree_rss)
    minimum = min(minimum, available)
    if tree_rss > a.max_rss_mb:
        reason = "child process tree RSS above sampled budget"
    elif available < a.reserve_mb:
        reason = "system available RAM below reserve"
    elif time.monotonic() - start > a.seconds:
        reason = "timeout"
    if reason:
        print("Stopping this benchmark:", reason, flush=True)
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
        break
    time.sleep(0.05)
code = child.wait()
report = {
    "returncode": code,
    "reason": reason,
    "sampled_peak_child_rss_mb": peak,
    "sampled_peak_process_tree_rss_mb": tree_peak,
    "minimum_system_available_mb": minimum,
    "elapsed_seconds": time.monotonic() - start,
    "max_rss_budget_mb": a.max_rss_mb,
    "reserve_mb": a.reserve_mb,
    "note": "Sampled 50ms supervisor, not an OS-enforced hard memory limit. Budget applies to summed child process-tree RSS; shared pages may be counted twice.",
}
Path(a.stats).parent.mkdir(parents=True, exist_ok=True)
Path(a.stats).write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
raise SystemExit(code if code >= 0 else 128 - code)
