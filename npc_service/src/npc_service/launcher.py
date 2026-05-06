from __future__ import annotations

import asyncio
import ctypes
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import logfire

from .config import Settings

HEALTH_POLL_INTERVAL_SECONDS = 0.5
HEALTH_TIMEOUT_DEFAULT_SECONDS = 600.0  # cold Q8_0 load on slow disk: 90+s; HF download can be minutes
SHUTDOWN_GRACE_SECONDS = 5.0


def _warn_winapi(msg: str, **fields) -> None:
    """logfire.warn with `err=GetLastError()` already attached — every WinAPI
    failure path in this module wants the same error code."""
    logfire.warn(msg, err=ctypes.get_last_error(), **fields)


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
        _warn_winapi("CreateJobObject failed; llama-server cleanup falls back to stop_llama()")
        return None

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
        h, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)
    ):
        _warn_winapi("SetInformationJobObject failed; closing job")
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
        _warn_winapi("OpenProcess failed for pid {pid}", pid=pid)
        return
    try:
        if not kernel32.AssignProcessToJobObject(job_handle, h_proc):
            _warn_winapi("AssignProcessToJobObject failed — likely already in a non-nestable parent job")
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
    ngl: int


def _exe_dir() -> Path:
    """Workspace root — the directory containing vendor/, models/, npc_service/.
    Resolved from this module's location, NEVER from CWD.

    src layout: this file is <workspace>/npc_service/src/npc_service/launcher.py.
    parents[0]=src/npc_service, [1]=src, [2]=npc_service, [3]=<workspace>.
    """
    return Path(__file__).resolve().parents[3]


def _bin_name(name: str) -> str:
    return name + ".exe" if os.name == "nt" else name


def resolve_config(settings: Settings | None = None) -> LlamaConfig:
    """Pull llama-server invocation params from settings or sane defaults."""
    s = settings or Settings()
    base = _exe_dir()

    bin_path = s.llama_server_bin or base / "vendor" / "llama.cpp" / _bin_name("llama-server")

    if s.llama_model_path:
        model_path: Path | None = s.llama_model_path
        hf: str | None = None
    else:
        candidate = base / "models" / "ggml-org-gemma-4-E4B-it-Q8_0.gguf"
        if candidate.exists():
            model_path = candidate
            hf = None
        else:
            # fall back to HF download-on-launch
            model_path = None
            hf = s.llama_hf_repo

    return LlamaConfig(
        bin=bin_path,
        model_path=model_path,
        hf_repo=hf,
        port=s.llama_port,
        host="127.0.0.1",
        ctx=s.llama_ctx,
        extra_args=s.llama_extra_args.split() if s.llama_extra_args else [],
        ngl=s.llama_ngl,
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
        "--reasoning", "on",  # Per Gemma 4 best practices; reasoning-format keeps it out of body
        "--reasoning-format", "deepseek",  # Pulls thinking into `reasoning_content` (separate field)
        "-ngl", str(cfg.ngl),
    ]
    cmd += cfg.extra_args
    return cmd


def _drain_pipe(pipe, log_file_handle, tag: str) -> None:
    """Background thread: read from a Popen pipe forever, tee to file.

    Bulk subprocess output (every line) goes to the per-launch log file
    only — that file is the forensic record. The handful of keyword-matched
    lines (errors, model-loaded, listening, eval timings) ride into Logfire
    as structured `llama subprocess` events so they're visible alongside
    sidecar spans without staring at a tail."""
    try:
        for raw in iter(pipe.readline, b""):
            try:
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            except Exception:
                line = repr(raw)
            log_file_handle.write(f"[{tag}] {line}\n")
            log_file_handle.flush()
            lower = line.lower()
            keywords = (
                "error", "loaded model", "model loaded", "warming up", "listening on",
                "prompt eval", "eval time", "tokens per second", "slot release",
            )
            if any(k in lower for k in keywords):
                logfire.info("llama subprocess [{tag}]", tag=tag, line=line)
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def start_llama(cfg: LlamaConfig | None = None) -> subprocess.Popen:
    cfg = cfg or resolve_config()
    cmd = _build_command(cfg)

    with logfire.span(
        "start llama-server on port {port}",
        port=cfg.port,
        bin=str(cfg.bin),
        model_path=str(cfg.model_path) if cfg.model_path else None,
        hf_repo=cfg.hf_repo,
        ctx=cfg.ctx,
    ):
        log_dir = _exe_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"llama-{int(time.time())}.log"
        log_handle = open(log_path, "w", encoding="utf-8")
        log_handle.write(f"# command: {' '.join(cmd)}\n")
        log_handle.flush()

        logfire.info(
            "spawning llama-server",
            cmd=cmd, log_path=str(log_path),
        )

        if not cfg.bin.exists() and not cfg.bin.name == "llama-server":
            # Allow PATH lookup if user passed bare "llama-server"
            logfire.warn(
                "llama-server binary not found — relying on PATH",
                bin=str(cfg.bin),
            )

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
        timeout = HEALTH_TIMEOUT_DEFAULT_SECONDS

    url = f"http://{host}:{port}/health"
    last_err: Exception | None = None

    with logfire.span(
        "wait for llama at {url}",
        url=url, timeout=timeout, poll_interval=poll_interval,
    ) as span:
        start_time = asyncio.get_event_loop().time()
        deadline = start_time + timeout
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
                            span.set_attribute(
                                "wait_seconds",
                                asyncio.get_event_loop().time() - start_time,
                            )
                            logfire.info("llama-server healthy", url=url)
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
        logfire.info("llama-server already exited", returncode=proc.returncode)
        return
    logfire.info("terminating llama-server", pid=proc.pid)
    try:
        proc.terminate()
    except Exception as e:
        logfire.warn("terminate failed", error=str(e))
    try:
        proc.wait(timeout=grace_seconds)
        logfire.info("llama-server exited gracefully", returncode=proc.returncode)
        return
    except subprocess.TimeoutExpired:
        logfire.warn(
            "llama-server did not exit in time; sending SIGKILL",
            grace_seconds=grace_seconds,
        )
    try:
        proc.kill()
        proc.wait(timeout=2.0)
    except Exception:
        logfire.exception("llama kill failed")
