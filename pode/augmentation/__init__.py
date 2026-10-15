"""Generative data augmentation for prognostic models.

A TVAE is fitted inside every training split, its samples (or cheap real-data
baselines) are added to the training rows, prognostic models are refitted, and
everything is scored on real held-out patients only. See
``docs/synthesis/AUGMENTATION.md`` and ``pode synthesis augment --help``.
"""

from .config import AugmentationConfig, ModelSpec, StrategySpec, load_augmentation_config

__all__ = ["AugmentationConfig", "ModelSpec", "StrategySpec", "load_augmentation_config"]
