# Agent patterns reference

A short field guide for when to reach for which multi-agent shape in Smallville. Patterns are ranked by complexity. Pick the simplest one that solves the task — the next-simplest is almost always wrong for a different reason.

## TL;DR for current code

| Concern | Pattern | Where in code |
| --- | --- | --- |
| Captor turn (one structured response, atomic state mutation) | **Single agent + structured output** | `CAPTOR_AGENT` (`main.py`) |
| Player dialogue picks (read-only over captor history) | **Specialized parallel agent (Pattern A — shared history, separate output schema)** | `OPTIONS_AGENT` (`main.py`), `/options` |
| History compaction triggered inside the captor's `history_processors` | **Programmatic delegation** | `summarize_when_large` → `SUMMARIZER_AGENT` |

Three agents already; one model. None of them are wrong. The patterns below are for *future* additions.

---

## P1 — Single agent + structured output

**Shape:** one `Agent`, `output_type=ToolOutput(SomeModel)`. Model emits one tool call per turn whose args are the structured response.

**Use when:** the task has a single coherent persona, one job, and you need bounded, validated output. Smallville's captor.

**Why it works:** modern instruct models are RLHF'd on chat-templated tool-calling. Tool-call schemas constrain output to your Pydantic model. Pydantic AI parses args back into the model with full validation. Streaming yields partial fields in declaration order — put the long text field first if you want progressive UX.

**Failure modes / mitigations:**
- Gemma 4 E-models default to thinking-on. Run llama-server with `--reasoning on --reasoning-format deepseek` so thoughts go to the separate `reasoning_content` field, AND set `OpenAIModelProfile(openai_chat_send_back_thinking_parts=False)` so pydantic-ai doesn't round-trip parsed `ThinkingPart`s back into prior-turn `content` on the next call.
- `NativeOutput` (GBNF response_format) lets unbounded text fields fill the buffer → 19K-token cascading turns. Use `ToolOutput` + `max_tokens` cap (Smallville does, see CLAUDE.md).
- Field declaration order matters for streaming. Long text fields first.

**Evidence:** Anthropic, *Building Effective Agents* — "start simple, optimize single calls with retrieval and examples before adopting multi-step systems." Pydantic AI's tool-output is the production-grade form of that single call.

---

## P2 — Specialized parallel agent over shared history (Pattern A)

**Shape:** a second `Agent` with its own `output_type`, own `instructions=`, runs against the same `HISTORY` list but **does not write back**. Endpoint is read-only.

**Use when:** you need a second LLM call with a *different role* over the *same conversation context* — coaching the player, scoring quality, extracting facts — and that call has no business mutating the primary agent's state.

**Why it works:**
- Different `instructions=` → totally different persona without polluting the primary system prompt.
- Pydantic AI's `instructions=` is per-run; you can reuse `message_history` across agents that have different system prompts.
- Read-only discipline (`/options` does NOT call `result.all_messages()`) means no chance of leaking the parallel agent's prompts into the primary's history.
- `ACTION_LOCK` shared between primary and secondary keeps reads coherent with mutations.

**Known fragility — foreign tool-call shadow:** when the primary agent uses `ToolOutput`, every prior assistant turn in `HISTORY` is a tool call to `final_result_PrimaryModel`. The secondary agent only has `final_result_SecondaryModel` registered. Models *usually* handle this fine (the foreign tool calls become inert past context), but it's documented in CLAUDE.md as a structural concern. **Escape hatch:** render history to plain text (`render_transcript`) and pass as a fenced block in the user prompt — see Pattern P3a.

**Use Pattern A when:** secondary agent is read-only, tool schemas don't conflict in practice, and you want minimum code.

**Evidence:** Pydantic AI docs — programmatic hand-off via `message_history=result.new_messages()` is documented for cross-agent reuse. The trim-trailing-tool-call (`messages[:-1]`) trick (`docs/output.md`, SQL router example) addresses the *last* turn but not earlier turns. For Smallville's options agent, empirical testing on Gemma 4 E4B showed clean OptionsResponse output despite foreign tool-call history.

---

## P3 — Specialized agent over rendered transcript

**Shape (a):** secondary agent runs with **no** `message_history`. Input is a single user prompt containing a `render_transcript(HISTORY)` plain-text block: `"Negotiator: ... / Captor: ... / [scene update: ...]"`.

**Use when:** Pattern A's foreign tool-call shadow becomes a real problem (model swap, llama.cpp upgrade, observed quality drop on the secondary), or when the secondary should impersonate a *different role* than the primary's assistant turns (e.g., suggesting what the negotiator should say — opposite role from the captor's tool calls).

**Tradeoffs vs P2:**
- Cleaner — secondary sees a clean conversation, no foreign tool noise, no role confusion.
- More code — `render_transcript` walks `HISTORY`, extracts `UserPromptPart.content` and parses `ToolCallPart.args` JSON for the captor's `response_text`.
- No KV-cache reuse on the prefix between turns — the transcript is rebuilt and re-tokenized each call. Fine for on-demand secondaries; not for hot-path.

**When to refactor from P2 → P3a:** when option/critic quality regresses or when adding a third agent that doesn't share schemas. Don't pre-build it.

**Evidence:** Pinecone "Conversational Memory for LLMs" — flattened transcripts work but lose the role-tagged training distribution; chat-templated history performs better when applicable. P3a is a deliberate trade: lose KV reuse + role-tagging in exchange for clean isolation.

---

## P4 — Agent delegation via tools (Pydantic AI joke-factory pattern)

**Shape:** primary agent has a tool `@agent.tool async def call_helper(ctx, args) -> str` whose body is `await helper_agent.run(args, usage=ctx.usage)`. The model decides when to invoke the helper.

**Use when:** the helper is a *resource the primary reasons about*, not a co-equal speaker — a SQL agent the router consults, a fact-checker the writer asks, a tool-using subagent that fetches data and returns to the parent. `usage=ctx.usage` propagates token tracking so global `UsageLimits` apply across the whole call.

**Don't use for Smallville's options agent:** options aren't something the *captor* should reason about — they're for the player UI. The captor doesn't decide when to suggest options. P4 is the wrong shape here.

**Where P4 *would* fit Smallville:** "bomb maker downstairs" character — primary captor has a `whisper_to_partner(message)` tool; partner only speaks when whispered to. Primary owns the turn, partner is a resource.

**Evidence:** Pydantic AI `docs/multi-agent-applications.md` — joke selection delegating to joke generation is the canonical example.

---

## P5 — Output functions (action handoffs)

**Shape:** `output_type=[func_a, func_b, ...]` where each function executes an action against `ctx.deps` (game state) and returns a string the model sees as the tool result. The model picks which function to call as its turn output.

**Use when:** actions need *rich feedback* the model should react to on the next turn ("I tried to release Janet and the crowd outside cheered"), and you have a strong-enough model that won't drop tool calls. Replaces enum-style `action: Literal["release_hostage", ...]` fields.

**Why it's powerful:** each action is a discrete callable with its own validator. `ModelRetry` from inside the function asks the model to retry with adjusted args. Naturally produces feedback loops between agent and game state.

**Why Smallville hasn't adopted it:** Gemma 4 E4B drops/duplicates tool calls under load. Splitting the captor into many tools risks silent turns ("model forgot to call `speak`"). The current `ToolOutput(CaptorResponse)` shape is *already* a single tool call with action as a field — same atomicity, far less brittleness on this model.

**Aspirational target** for: a model swap to Gemma 4 27B / Llama 3.3 / GPT-4-class.

**Evidence:** Pydantic AI `docs/output.md` — SQL router example demonstrates `output_type=[hand_off_to_sql_agent, RouterFailure]`.

---

## P6 — Orchestrator + speaker selection (multi-party dialogue)

**Shape:** a small "speaker selector" agent runs first (cheap, ~50 tokens out): input = recent transcript + game state, output = `{"speaker": "captor_a" | "captor_b" | "both"}`. Selected captor agent runs against shared `HISTORY` (both captors share the same `output_type` so foreign-tool-call concern is moot). Both system prompts contain awareness of the other.

**Use when:** you have multiple co-equal speakers (two captors, an NPC crowd) and the dialogue dynamics matter — players should be able to play one off the other.

**Why this over "both run every turn":** 2× latency per turn, and most turns one speaker has nothing to add. Speaker selection adds a tiny call but skips the 2× cost on turns with one speaker.

**Why this over "single agent with two personas":** voices cross-contaminate. One model emitting two personas in one tool call drifts hard on small models.

**Evidence:** Frontiers (2025), *Who speaks next? Multi-party AI discussion leveraging the systematics of turn-taking in Murder Mystery games* — adjacency-pair-based next-speaker selection produces more coherent multi-party dialogue than free-for-all generation.

---

## P7 — Evaluator-optimizer (critic loop)

**Shape:** primary agent generates a response. Critic agent reviews it ("is this in-character? is this consistent with prior commitments?"). On rejection, primary retries with critic's feedback as additional context. Loop bounded by `UsageLimits`.

**Use when:** drift is observable and the cost of a bad turn is high. For Smallville: a critic could enforce "captor never invents names not in `known_facts`" — but the current system prompt + `deflection_style` already handles that, and Gemma's drift rate is low enough that adding a critic doubles latency for marginal quality gain.

**Cheap insurance** for: small models prone to character drift, late-game when context is long, scenarios with hard-to-encode constraints.

**Evidence:** Anthropic *Building Effective Agents* — evaluator-optimizer is one of the five core patterns, recommended "with clear evaluation criteria where refinement demonstrably improves output."

---

## P8 — Memory / world-state agent

**Shape:** background agent extracts structured facts from each captor turn into a sidecar store (`{"commitments": [...], "revealed_facts": [...], "named_persons": [...]}`). Captor reads from store on each turn instead of relying on long history.

**Use when:** game length exceeds context window even after summarization, or when consistency over many turns matters more than recall of exact phrasing.

**Smallville context:** `summarize_when_large` is a primitive form of this — it folds messages into a recap. A full memory agent would extract typed facts, not free-form summaries. Worth considering if scenarios grow longer than the current 12-turn budget.

**Evidence:** Mem0 *LLM Chat History Summarization Best Practices (2025)* — typed extraction outperforms free-form summarization for long-running consistency, at the cost of an extra agent per turn.

---

## Anti-patterns to avoid

- **Cramming multiple jobs into one agent's `output_type`** (e.g., `CaptorResponseWithOptions`). LLMs perform better when each consideration gets a focused call. Anthropic's guidance is explicit on this.
- **Sharing history bidirectionally between agents with different schemas.** `OPTIONS_AGENT` writing back via `result.all_messages()` would leak its `"Generate options."` user prompt into the captor's history. CLAUDE.md L53 is the codified version of this rule.
- **Reflowing history into a flat user prompt to "save tokens."** Loses chat-template alignment, breaks KV-cache prefix reuse, hurts model performance more than it saves tokens. Only do P3a when isolation is the goal, not as an optimization.
- **Adding a critic before measuring drift.** Critic loops double latency. Earn the addition with evidence (Logfire spans showing off-character turns).

---

## Decision tree

```
Single coherent task, one persona?
├─ yes → P1 (single agent + ToolOutput)
└─ no → another agent needed
   │
   ├─ secondary is a UI helper / parallel reader?
   │  ├─ shared history works empirically → P2
   │  └─ role mismatch or quality regression → P3a
   │
   ├─ primary should *call* the secondary?
   │  └─ P4 (agent delegation via @agent.tool)
   │
   ├─ multiple co-equal speakers?
   │  └─ P6 (orchestrator + speaker selection)
   │
   ├─ actions need rich feedback the model reacts to?
   │  └─ P5 (output functions) — only on a strong model
   │
   ├─ observable drift, retry would help?
   │  └─ P7 (critic loop) — measure first
   │
   └─ context exceeds window even after summarization?
      └─ P8 (memory agent)
```

---

## Sources

- Anthropic, *Building Effective Agents* — https://www.anthropic.com/engineering/building-effective-agents
- Pydantic AI docs — https://ai.pydantic.dev/multi-agent-applications/, https://ai.pydantic.dev/output/, https://ai.pydantic.dev/message-history/
- Frontiers (2025), *Who speaks next? Multi-party AI discussion …* — https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1582287/full
- ACM CHI '24, *Exploring Presence in Interactions with LLM-Driven NPCs* — https://dl.acm.org/doi/fullHtml/10.1145/3641825.3687716
- HuggingFace LLM Course, *Chat Templates* — https://huggingface.co/learn/llm-course/en/chapter11/2
- Pinecone, *Conversational Memory for LLMs* — https://www.pinecone.io/learn/series/langchain/langchain-conversational-memory/
- Mem0, *LLM Chat History Summarization Best Practices (2025)* — https://mem0.ai/blog/llm-chat-history-summarization-guide-2025
