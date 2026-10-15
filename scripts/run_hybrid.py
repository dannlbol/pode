#!/usr/bin/env python
"""Deprecated shim -> `pode synthesis hybrid`.

The logic now lives in the `synthvae.cli.hybrid` package module and is the primary surface via
`pode synthesis hybrid`. This shim forwards for one release; please migrate.
"""
import sys

from synthvae.cli.hybrid import main

if __name__ == "__main__":
    print("[deprecated] python scripts/run_hybrid.py -> use: pode synthesis hybrid", file=sys.stderr)
    raise SystemExit(main())
