"""Judge model factory + inferno warm-up.

Default: local Qwen on inferno's OpenAI-compatible endpoint. No API key, shares
the GPU with the Gemma-under-test (both stay resident), and inferno's continuous
batching means the whole judge batch can be fired concurrently — no client-side
semaphore or rate-limiting. `gemini:`/`claude:` prefixes switch to a cloud judge
for A/B comparison.
"""
from __future__ import annotations

import httpx
from pydantic_ai.models import Model

from handler.config import Settings


def build_judge_model(settings: Settings, *, override: str | None = None) -> Model:
    """Resolve the configured judge model.

    - `gemini:<name>` → Google model keyed by `settings.gemini_api_key`.
    - `claude:<name>` / `anthropic:<name>` → Anthropic model (key from env).
    - anything else → OpenAI-compatible model at `settings.resolved_judge_base_url`
      (inferno/Qwen by default), with a throwaway api_key.
    """
    name = override or settings.judge_model
    if name.startswith("gemini:") or name.startswith("google:"):
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        return GoogleModel(
            name.split(":", 1)[1],
            provider=GoogleProvider(api_key=settings.gemini_api_key),
        )
    if name.startswith("claude:") or name.startswith("anthropic:"):
        from pydantic_ai.models.anthropic import AnthropicModel

        return AnthropicModel(name.split(":", 1)[1])

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    return OpenAIChatModel(
        name,
        provider=OpenAIProvider(
            base_url=settings.resolved_judge_base_url, api_key="not-needed"
        ),
    )


def warm_judge(settings: Settings, *, override: str | None = None) -> None:
    """Warm the local Qwen on inferno so the first judge call doesn't eat the
    multi-minute cold start. No-op for cloud judges (gemini:/claude:)."""
    name = override or settings.judge_model
    if name.startswith(("gemini:", "google:", "claude:", "anthropic:")):
        return
    url = f"{settings.inferno_url.rstrip('/')}/warm/{name}"
    try:
        httpx.post(url, timeout=600.0)
    except httpx.HTTPError as e:  # warming is best-effort
        print(f"warm_judge: could not warm {name} ({e}); first call may be slow")
