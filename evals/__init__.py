"""HANDLER playtest harness (eval tooling, not shipped with the game).

A cloud model (Gemini) plays the game by supplying the same callbacks the
CLI supplies from stdin, driving the existing `handler.scene_loop.play_session`.
Sessions run concurrently via pydantic-evals, each captured as a
`SessionArtifact` JSON and scored against deterministic invariants.

See EVALS.md for the strategy this implements (Bucket 3 + Bucket 1).
"""
