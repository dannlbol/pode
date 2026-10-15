#!/usr/bin/env python
"""Deprecated shim -> `pode synthesis mmvae`.

The logic now lives in the `synthvae.cli.mmvae` package module and is the primary surface via
`pode synthesis mmvae`. This shim forwards for one release; please migrate.
"""
import sys

from synthvae.cli.mmvae import main

if __name__ == "__main__":
    print("[deprecated] python scripts/run_mmvae.py -> use: pode synthesis mmvae", file=sys.stderr)
    raise SystemExit(main())
