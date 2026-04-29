from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx


log = logging.getLogger(__name__)

HEALTH_POLL_INTERVAL_SECONDS = 0.5
HEALTH_TIMEOUT_DEFAULT_SECONDS = 600.0  # cold Q8_0 load on slow disk: 90+s; HF download can be minutes
SHUTDOWN_GRACE_SECONDS = 5.0


# ───────────────────────── Windows Job Object cleanup ─────────────────────────
#
# When this Python process dies abruptly (Ctrl-C in `make sidecar`, OOM kill,
# crash), normal stop_llama() doesn't run and llama-server orphans on :8080.
# A Job Object with KILL_ON_JOB_CLOSE makes the kernel kill llama-server when
# Python's process exits, regardless of whether shutdown ran. When this Python
# is itself a child of Godot's job, we sit inside that job AND own a nested
# job for our own child — both are belt-and-suspenders, both close the same
# loop.

def _create_kill_on_close_job() -> int | None:
    if os.name != "nt":
        return None

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    JobObjectExtendedLimitInformation = 9

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    kernel32.SetInformationJobObject.restype = ctypes.c_int
    kernel32.SetInformationJobObject.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32
    ]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]

    h = kernel32.CreateJobObjectW(None, None)
    if not h:
        log.warning("CreateJobObject failed (err=%s); llama-server cleanup falls back to stop_llama()", ctypes.get_last_error())
        return None

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
        h, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)
    ):
        log.warning("SetInformationJobObject failed (err=%s); closing job", ctypes.get_last_error())
        kernel32.CloseHandle(h)
        return None
    return h


def _assign_pid_to_job(job_handle: int, pid: int) -> None:
    if os.name != "nt" or not job_handle:
        return
    PROCESS_TERMINATE = 0x0001
    PROCESS_SET_QUOTA = 0x0100
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    h_proc = kernel32.OpenProcess(PROCESS_TERMINATE | PROCESS_SET_QUOTA, False, pid)
    if not h_proc:
        log.warning("OpenProcess(pid=%s) failed (err=%s)", pid, ctypes.get_last_error())
        return
    try:
        if not kernel32.AssignProcessToJobObject(job_handle, h_proc):
            log.warning("AssignProcessToJobObject failed (err=%s) — likely already in a non-nestable parent job", ctypes.get_last_error())
    finally:
        kernel32.CloseHandle(h_proc)


# Module-level: handle is held for the lifetime of the Python interpreter, so
# the kernel auto-closes it (and kills llama-server) on any kind of exit.
_LLAMA_JOB_HANDLE: int | None = _create_kill_on_close_job()


class LlamaStartupError(RuntimeError):
    """llama-server failed to come up healthy in time."""


@dataclass(frozen=True)
class LlamaConfig:
    bin: Path
    model_path: Path | None  # None -> use -hf flag
    hf_repo: str | None
    port: int
    host: str
    ctx: int
    extra_args: list[str]


def _exe_dir() -> Path:
    """Directory of the running executable / script entry. NEVER CWD."""
    # When running `python -m npc_service`, sys.argv[0] is "<...>/npc_service/__main__.py"
    return Path(sys.argv[0]).resolve().parent.parent


def _bin_name(name: str) -> str:
    return name + ".exe" if os.name == "nt" else name


def resolve_config() -> LlamaConfig:
    """Pull llama-server invocation params from env or sane defaults."""
    base = _exe_dir()

    bin_env = os.getenv("LLAMA_SERVER_BIN")
    bin_path = Path(bin_env) if bin_env else base / "vendor" / "llama.cpp" / _bin_name("llama-server")

    model_env = os.getenv("LLAMA_MODEL_PATH")
    hf_repo = os.getenv("LLAMA_HF_REPO", "unsloth/gemma-4-E4B-it-GGUF:Q8_0")
    if model_env:
        model_path: Path | None = Path(model_env)
        hf: str | None = None
    else:
        candidate = base / "models" / "gemma-4-E4B-it-Q8_0.gguf"
        if candidate.exists():
            model_path = candidate
            hf = None
        else:
            # fall back to HF download-on-launch
            model_path = None
            hf = hf_repo

    return LlamaConfig(
        bin=bin_path,
        model_path=model_path,
        hf_repo=hf,
        port=int(os.getenv("LLAMA_PORT", "8080")),
        host=os.getenv("LLAMA_HOST", "127.0.0.1"),
        ctx=int(os.getenv("LLAMA_CTX", "32768")),
        extra_args=os.getenv("LLAMA_EXTRA_ARGS", "").split() if os.getenv("LLAMA_EXTRA_ARGS") else [],
    )


def _build_command(cfg: LlamaConfig) -> list[str]:
    cmd = [str(cfg.bin)]
    if cfg.model_path is not None:
        cmd += ["-m", str(cfg.model_path)]
    elif cfg.hf_repo is not None:
        cmd += ["-hf", cfg.hf_repo]
    cmd += [
        "--host", cfg.host,
        "--port", str(cfg.port),
        "-c", str(cfg.ctx),
        "--jinja",  # REQUIRED for Gemma chat template
        "--reasoning", "off",  # REQUIRED for Gemma 4 E-models — default-on thinking swallows output
        "-ngl", os.getenv("LLAMA_NGL", "99"),
    ]
    cmd += cfg.extra_args
    return cmd


def _drain_pipe(pipe, log_file_handle, tag: str) -> None:
    """Background thread: read from a Popen pipe forever, tee to file."""
    try:
        for raw in iter(pipe.readline, b""):
            try:
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            except Exception:
                line = repr(raw)
            log_file_handle.write(f"[{tag}] {line}\n")
            log_file_handle.flush()
            # tee key lines to stderr so they show up in console
            lower = line.lower()
            if any(k in lower for k in ("error", "loaded model", "model loaded", "warming up", "listening on")):
                print(f"[llama:{tag}] {line}", file=sys.stderr, flush=True)
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def start_llama(cfg: LlamaConfig | None = None) -> subprocess.Popen:
    cfg = cfg or resolve_config()
    cmd = _build_command(cfg)

    log_dir = _exe_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"llama-{int(time.time())}.log"
    log_handle = open(log_path, "w", encoding="utf-8")
    log_handle.write(f"# command: {' '.join(cmd)}\n")
    log_handle.flush()

    log.info("starting llama-server: %s", " ".join(cmd))
    log.info("llama-server log: %s", log_path)

    if not cfg.bin.exists() and not cfg.bin.name == "llama-server":
        # Allow PATH lookup if user passed bare "llama-server"
        log.warning("llama-server binary not found at %s — relying on PATH", cfg.bin)

    creationflags = 0
    if os.name == "nt":
        # Avoid spawning a console window for the subprocess
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            bufsize=0,
            creationflags=creationflags,
            # Don't let llama-server inherit uvicorn's listening socket — otherwise
            # an orphaned llama keeps :8000 bound after the sidecar dies.
            close_fds=True,
        )
    except FileNotFoundError as e:
        raise LlamaStartupError(
            f"could not start llama-server: binary not found. "
            f"Set LLAMA_SERVER_BIN or place llama-server in {cfg.bin.parent}. "
            f"Underlying error: {e}"
        ) from e

    # Enroll llama-server in our KILL_ON_JOB_CLOSE job so it dies if Python
    # exits abruptly without running stop_llama(). No-op on non-Windows or
    # if job creation failed (logged above) — stop_llama() remains the
    # primary cleanup path.
    if _LLAMA_JOB_HANDLE is not None:
        _assign_pid_to_job(_LLAMA_JOB_HANDLE, proc.pid)

    threading.Thread(
        target=_drain_pipe, args=(proc.stdout, log_handle, "out"), daemon=True
    ).start()
    threading.Thread(
        target=_drain_pipe, args=(proc.stderr, log_handle, "err"), daemon=True
    ).start()

    return proc


async def wait_healthy(
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    timeout: float | None = None,
    poll_interval: float = HEALTH_POLL_INTERVAL_SECONDS,
) -> None:
    """Poll llama-server's /health endpoint until 200 + ok body, or raise.

    /health returns 200 only when the model slot is loaded. /v1/models would
    200 too early on cold start.
    """
    if timeout is None:
        timeout = float(os.getenv("LLAMA_HEALTH_TIMEOUT", str(HEALTH_TIMEOUT_DEFAULT_SECONDS)))

    url = f"http://{host}:{port}/health"
    deadline = asyncio.get_event_loop().time() + timeout
    last_err: Exception | None = None

    log.info("waiting for llama-server at %s (timeout=%ss)", url, timeout)
    async with httpx.AsyncClient(timeout=2.0) as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    body: dict = {}
                    try:
                        body = resp.json()
                    except Exception:
                        pass
                    status = body.get("status") if isinstance(body, dict) else None
                    if status == "ok" or not body:
                        log.info("llama-server healthy")
                        return
                    last_err = RuntimeError(f"unexpected health body: {body}")
                else:
                    last_err = RuntimeError(f"health returned {resp.status_code}")
            except httpx.HTTPError as e:
                last_err = e
            await asyncio.sleep(poll_interval)

    raise LlamaStartupError(
        f"llama-server at {url} did not become healthy within {timeout}s. "
        f"Last error: {last_err}"
    )


def stop_llama(proc: subprocess.Popen | None, grace_seconds: float = SHUTDOWN_GRACE_SECONDS) -> None:
    """Terminate llama-server. SIGTERM, wait grace, then kill."""
    if proc is None:
        return
    if proc.poll() is not None:
        log.info("llama-server already exited (rc=%s)", proc.returncode)
        return
    log.info("terminating llama-server pid=%s", proc.pid)
    try:
        proc.terminate()
    except Exception as e:
        log.warning("terminate failed: %s", e)
    try:
        proc.wait(timeout=grace_seconds)
        log.info("llama-server exited gracefully (rc=%s)", proc.returncode)
        return
    except subprocess.TimeoutExpired:
        log.warning("llama-server did not exit in %ss; sending SIGKILL", grace_seconds)
    try:
        proc.kill()
        proc.wait(timeout=2.0)
    except Exception as e:
        log.error("kill failed: %s", e)
