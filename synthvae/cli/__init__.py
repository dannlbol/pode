"""Unified CLI surface for the synthvae (Synthesis) theme.

Mounts the hybrid, mmvae and handoff (bundle/promote/decode) commands under a
single ``add_subparsers`` object. Consumed by ``pode.cli.main`` to build the
``pode synthesis ...`` namespace; also usable standalone via each submodule's
``main`` (e.g. ``python -m synthvae.cli.hybrid``).
"""
from __future__ import annotations


def register(subparsers) -> None:
    """Attach all synthvae subcommands (hybrid, mmvae, bundle, promote, decode)."""
    from . import handoff, hybrid, mmvae

    hybrid.register(subparsers)
    mmvae.register(subparsers)
    handoff.register(subparsers)
