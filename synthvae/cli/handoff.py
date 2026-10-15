"""synthvae <-> PODE handoff bridge subcommands: bundle / promote / decode.

Thin registration wrappers over the bundle/promote/decode implementations, mounted
flat under ``pode synthesis`` (e.g. ``pode synthesis bundle ...``).
"""
from __future__ import annotations

from . import _bundle_impl, _decode_impl, _promote_impl


def register(subparsers) -> None:
    """Mount bundle/promote/decode under a parent ``add_subparsers`` (pode synthesis)."""
    b = subparsers.add_parser(
        "bundle", help="concatenate handoff bundle(s) into a PODE training table + config.")
    _bundle_impl.add_args(b)
    b.set_defaults(func=_bundle_impl.run)

    p = subparsers.add_parser(
        "promote", help="turn a PODE tuning result into a final trainable config.")
    _promote_impl.add_args(p)
    p.set_defaults(func=_promote_impl.run)

    d = subparsers.add_parser(
        "decode", help="split a PODE-generated table and decode latent blocks to matrices.")
    _decode_impl.add_args(d)
    d.set_defaults(func=_decode_impl.run)
