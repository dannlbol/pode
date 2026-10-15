"""Generative (multiple) imputation with TVAE, plus a masking benchmark.

Engines share the :class:`~pode.imputation.base.Imputer` interface:

* ``tvae-decoder`` - conditions an existing PODE ``model.pkl`` (decoder only);
* ``tvae-masked``  - a mask-aware TVAE trained directly on incomplete data;
* baselines: ``mean_mode``, ``knn``, ``mice``, ``missforest``, ``gain``.

Unrelated to :mod:`pode.data.imputation`, which fills NaNs in *synthetic*
output. Heavy backends are imported lazily by each engine.
"""

from .base import Imputer
from .config import EngineSpec, ImputationConfig, load_imputation_config
from .engines import ENGINES, build_imputer

__all__ = [
    "ENGINES",
    "EngineSpec",
    "ImputationConfig",
    "Imputer",
    "build_imputer",
    "load_imputation_config",
]
