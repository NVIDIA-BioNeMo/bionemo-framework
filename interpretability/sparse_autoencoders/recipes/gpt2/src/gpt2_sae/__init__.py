"""GPT-2 SAE recipe: live feature-explorer backend for the Bloom GPT-2-small SAE.

The heavy lifting lives in `server.py` (the FastAPI app + the `Engine` that owns the model, the SAE,
and the feature-clamp steering hook). `cli.py` exposes `encode` / `generate` / `serve` for the terminal.
"""

from gpt2_sae.server import Engine, app, engine


__all__ = ["Engine", "app", "engine"]
