"""pode_mantis: generic input preparation and subprocess orchestration for
MANTIS-MB, a published, frozen deep-learning survival model for
medulloblastoma (Stefan/Hovestadt, Dana-Farber).

This package does **not** reimplement or vendor the model -- per the source
project's own governing principle, the reference implementation
(``sstefan01/methylation-survival``) is never modified, only driven via
subprocess from your own checkout of it, in its own isolated environment
(see docs/mantis/README.md). pode_mantis provides everything around that:
reindexing a new cohort's beta matrix to the model's exact probe order,
dispatching CNV feature calling (R/Bioconductor conumee2 -- no Python
equivalent exists), building the survival/clinical input file, generating a
correctly-shaped reference config, and the alignment/sanity checks that
guard against the reference implementation's documented #1 failure mode
(silent row misalignment between beta/CNV/survival inputs).

Inference-only: this never retrains MANTIS-MB. Applying the released,
medulloblastoma-trained weights to a different tumour type is an exploratory,
out-of-domain use of the model -- see docs/mantis/README.md before drawing
any conclusions from its output on such a cohort.

Names are exposed lazily via module ``__getattr__`` (PEP 562), matching
``pode_foundation``'s convention, so ``import pode_mantis`` stays fast.
"""
from __future__ import annotations

import importlib

from .config import Config, load_config, validate_config

__version__ = "0.1.0"

_LAZY = {
    "run_prepare_beta": "pode_mantis.core.prepare_beta",
    "run_prepare_cnv": "pode_mantis.core.prepare_cnv",
    "run_prepare_survival": "pode_mantis.core.prepare_survival",
    "run_generate_config": "pode_mantis.core.generate_config",
    "check_pre_flight": "pode_mantis.core.assertions",
    "check_post_flight": "pode_mantis.core.assertions",
    "run_inference": "pode_mantis.core.run_reference",
    "run_evaluate": "pode_mantis.core.run_reference",
    "run_plot": "pode_mantis.core.run_reference",
}


def __getattr__(name: str):
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_path)
    return getattr(module, name)


def __dir__():
    return sorted(list(globals()) + list(_LAZY))


__all__ = ["Config", "load_config", "validate_config", *_LAZY.keys()]
