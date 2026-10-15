"""synthvae: survival-aware hybrid synthesis of clinico-pathological + expression data.

Two deliverables live in this package:

1. `synthvae.pipeline.hybrid` -- the *recommended hybrid*: a survival-supervised
   expression compressor whose latents are handed to an SDV TVAE alongside the
   clinical columns, then decoded back to full expression. Runnable end to end.

2. `synthvae.mmvae` -- scaffolding + specification stubs for the *joint multimodal
   VAE* (PoE/MoE/MoPoE, gating, three decoder heads). Intended for an agent
   (Claude Code / Codex) to implement against `docs/SPECIFICATION.md`.
"""

__version__ = "0.1.0"
