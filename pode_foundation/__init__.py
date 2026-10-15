"""pode_foundation: methylation-array masked-autoencoder (MAE) pretraining,
domain-adaptive fine-tuning, embedding extraction, and optional survival
evaluation.

Pretrain a masked autoencoder on your own methylation-array cohort (or a
broader public reference, then fine-tune onto your cohort) and use the
frozen encoder as a feature extractor for downstream analysis (see
docs/foundation/README.md).

Config loading (``Config``/``load_config``) is lightweight and imported
eagerly. Everything else here touches ``torch`` and is exposed *lazily* via
module ``__getattr__`` (PEP 562), so ``import pode_foundation`` and
``pode foundation --help`` stay fast even without torch installed -- the
same reason ``pode_foundation.cli`` defers these imports into its command
handlers rather than importing them at module scope.
"""
from __future__ import annotations

import importlib

from .config import Config, load_config, validate_config

__version__ = "0.1.0"

_LAZY = {
    "MethylationMAE": "pode_foundation.core.mae_model",
    "masked_mse_loss": "pode_foundation.core.mae_model",
    "make_mask": "pode_foundation.core.mae_model",
    "make_feature_weight": "pode_foundation.core.mae_model_weighted",
    "weighted_masked_mse_loss": "pode_foundation.core.mae_model_weighted",
    "run_preprocess": "pode_foundation.core.preprocess",
    "run_combine": "pode_foundation.core.combine",
    "run_pretrain": "pode_foundation.core.train",
    "run_finetune": "pode_foundation.core.finetune",
    "run_embed": "pode_foundation.core.embed",
    "run_survival": "pode_foundation.core.survival",
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
