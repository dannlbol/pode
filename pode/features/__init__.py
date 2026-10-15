"""Features namespace - extension hooks for the deferred biomarker phase.

Currently exposes only the :class:`FeatureExtractor` / :class:`FeatureSelector`
protocols. Concrete extractors (e.g. metagene signatures) will live here later.
"""

from .base import FeatureExtractor, FeatureSelector

__all__ = ["FeatureExtractor", "FeatureSelector"]
