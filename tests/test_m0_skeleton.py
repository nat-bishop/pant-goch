"""M0 sanity checks: health endpoint responds, settings load, launcher
renders the non-negotiable flag set."""
import pytest

from handler.config import Settings
from handler.launcher import build_command, resolve_config
from handler.main import health


@pytest.fixture
def clean_settings(monkeypatch):
    """Settings without local .env / shell env contamination — tests defaults."""
    for var in ("SMALLVILLE_USE_INFERNO", "INFERNO_URL", "LLAMA_PORT", "LLAMA_HOST"):
        monkeypatch.delenv(var, raising=False)
    return Settings(_env_file=None)


def test_health_returns_ok():
    assert health() == {"status": "ok"}


def test_settings_load_with_defaults(clean_settings):
    s = clean_settings
    assert s.llama_port == 8080
    assert s.llama_host == "127.0.0.1"
    assert s.use_inferno is False
    assert s.llama_kv_unified is False
    assert s.llama_reasoning_mode == "off"


def test_build_command_includes_required_flags(clean_settings):
    """The non-negotiable flag set from SPEC + CLAUDE.md. If any of these
    drop out of the command, Gemma 4 multi-turn breaks (jinja), prefill is
    slow (flash-attn), KV pages thrash (kv-unified mismatch), or batch
    sizing oversizes our short prompts."""
    cfg = resolve_config(clean_settings)
    cmd = build_command(cfg)
    assert "--jinja" in cmd
    assert "--flash-attn" in cmd
    assert "on" in cmd  # --flash-attn on
    assert "--batch-size" in cmd
    assert "--ubatch-size" in cmd
    # v2 default is per-slot allocation; explicit flag is mandatory because
    # recent llama.cpp builds default to unified when -np is auto.
    assert "--no-kv-unified" in cmd
    # Reasoning off by default → `--reasoning off` is passed EXPLICITLY.
    # Omitting the flag would let llama-server's CLI default (`auto`) take
    # over, which on Gemma 4 with `--jinja` activates the reasoning channel
    # and leaks ~125 thinking tokens per call. `--reasoning-format` is only
    # emitted when extraction is on/auto, so it stays absent here.
    assert "--reasoning" in cmd
    assert cmd[cmd.index("--reasoning") + 1] == "off"
    assert "--reasoning-format" not in cmd
