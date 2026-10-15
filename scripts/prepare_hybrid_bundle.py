#!/usr/bin/env python
"""Deprecated shim -> `pode synthesis bundle`.

The logic now lives in the `synthvae.cli._bundle_impl` package module and is the primary surface via
`pode synthesis bundle`. This shim forwards for one release; please migrate.
"""
import sys

from synthvae.cli._bundle_impl import main

if __name__ == "__main__":
    print("[deprecated] python scripts/prepare_hybrid_bundle.py -> use: pode synthesis bundle", file=sys.stderr)
    raise SystemExit(main())
