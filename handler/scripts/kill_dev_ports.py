"""Kill anything listening on the dev ports (8000, 8080, 5173-5176).

Windows-specific quirk: when uvicorn's `--reload` parent dies, the child
multiprocessing-fork worker keeps the listening socket but Get-NetTCPConnection
still reports the (now-dead) parent as `OwningProcess`. So the port stays
bound after one round of kill-by-port. Same shape for vite/esbuild children
under npm + concurrently.

Strategy:
  1. Kill everything Get-NetTCPConnection points at for each port.
  2. Sweep orphans: any python.exe or node.exe whose parent is no longer
     alive AND whose command line names one of our tools (uvicorn /
     handler.main / vite / esbuild / concurrently).
  3. Repeat once to catch the next layer.
  4. Exit non-zero if any port is still bound.

Invoked by `make kill` — see the Makefile comment for why this is Python
rather than inline PowerShell.
"""

from __future__ import annotations

import contextlib
import re
import subprocess
import sys
import time

PORTS = [8000, 8080, 5173, 5174, 5175, 5176]
ORPHAN_CMD_PATTERN = re.compile(
    r"uvicorn|handler\.main|multiprocessing\.spawn|vite|esbuild|concurrently",
    re.IGNORECASE,
)


def _run_powershell(script: str) -> str:
    """Run a PowerShell one-liner and return stdout (empty on any failure)."""
    cmd = ["powershell", "-NoProfile", "-Command", script]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout


def pids_on_port(port: int) -> set[int]:
    """Return PIDs that the kernel attributes to the LISTEN socket on `port`.
    On Windows this may include phantom-dead PIDs after a parent dies — those
    still need a Stop-Process call (which will no-op) so we trigger the
    follow-up orphan sweep."""
    stdout = _run_powershell(
        f"(Get-NetTCPConnection -LocalPort {port} -State Listen "
        f"-ErrorAction SilentlyContinue).OwningProcess"
    )
    return {int(line.strip()) for line in stdout.splitlines() if line.strip().isdigit()}


def orphan_pids() -> set[int]:
    """Find python.exe / node.exe processes whose parent is dead AND whose
    command line matches one of our dev tools. These are the children that
    inherit a listening socket and need to be killed explicitly."""
    stdout = _run_powershell(
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='node.exe'\" | "
        "ForEach-Object { "
        "  $parentAlive = (Get-Process -Id $_.ParentProcessId -ErrorAction SilentlyContinue) -ne $null; "
        "  if (-not $parentAlive) { '{0}|{1}' -f $_.ProcessId, $_.CommandLine } "
        "}"
    )
    pids: set[int] = set()
    for line in stdout.splitlines():
        line = line.strip()
        if "|" not in line:
            continue
        pid_str, cmdline = line.split("|", 1)
        if pid_str.isdigit() and ORPHAN_CMD_PATTERN.search(cmdline):
            pids.add(int(pid_str))
    return pids


def kill_pid(pid: int) -> bool:
    """Force-terminate `pid`. Returns True iff Stop-Process reported success."""
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        f"Stop-Process -Id {pid} -Force -ErrorAction Stop",
    ]
    with contextlib.suppress(subprocess.TimeoutExpired):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    return False


def main() -> int:
    for pass_no in range(2):
        for port in PORTS:
            for pid in pids_on_port(port):
                if kill_pid(pid):
                    print(f"killed pid {pid} on :{port}")

        for pid in orphan_pids():
            if kill_pid(pid):
                print(f"killed orphan pid {pid}")

        time.sleep(0.4)

    still_bound = [p for p in PORTS if pids_on_port(p)]
    if still_bound:
        print(f"still bound: {','.join(str(p) for p in still_bound)}")
        return 1
    print("all clear")
    return 0


if __name__ == "__main__":
    sys.exit(main())
