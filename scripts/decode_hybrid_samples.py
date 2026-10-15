#!/usr/bin/env python
"""Deprecated shim -> `pode synthesis decode`.

The logic now lives in the `synthvae.cli._decode_impl` package module and is the primary surface via
`pode synthesis decode`. This shim forwards for one release; please migrate.
"""
import sys

from synthvae.cli._decode_impl import main

if __name__ == "__main__":
    print("[deprecated] python scripts/decode_hybrid_samples.py -> use: pode synthesis decode", file=sys.stderr)
    raise SystemExit(main())
