#!/usr/bin/env python
"""Deprecated shim -> `pode synthesis promote`.

The logic now lives in the `synthvae.cli._promote_impl` package module and is the primary surface via
`pode synthesis promote`. This shim forwards for one release; please migrate.
"""
import sys

from synthvae.cli._promote_impl import main

if __name__ == "__main__":
    print("[deprecated] python scripts/promote_best_config.py -> use: pode synthesis promote", file=sys.stderr)
    raise SystemExit(main())
