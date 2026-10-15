"""synthvae.mmvae subpackage -- the joint multimodal VAE (deliverable 2).

Public API::

    from synthvae.mmvae import MMVAEConfig, JointMultimodalVAE, fit
    from synthvae.mmvae import JointVAESynthesizer   # high-level orchestrator

The torch-dependent symbols are exported only when torch is importable, mirroring
the import guards used throughout the package.
"""
from __future__ import annotations

__all__ = ["MMVAEConfig"]

from .model import MMVAEConfig  # dataclass; safe without torch

try:  # torch-dependent public surface
    from .model import JointMultimodalVAE, fit
    from .generate import JointVAESynthesizer, sample_survival

    __all__ += ["JointMultimodalVAE", "fit", "JointVAESynthesizer", "sample_survival"]
except Exception:  # pragma: no cover
    pass
