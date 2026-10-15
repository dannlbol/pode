"""Privacy-preservation assessment of synthetic data (``pode synthesis privacy``).

A conventional TVAE is not private by default: it can memorise rare or
distinctive patients, which matters most in small paediatric cohorts. This
package tests a synthetic release candidate against realistic threats -
exact and near copies, membership inference, attribute inference, singling-out
and linkage, rare combinations, outlier replication and identifier leakage -
and turns the results into a PASS / REVIEW / FAIL release gate.

See ``docs/synthesis/PRIVACY.md``.
"""

from .assess import PrivacyReport, assess_privacy
from .config import AuditConfig, GateConfig, PrivacyConfig, load_privacy_config

__all__ = ["assess_privacy", "PrivacyReport", "PrivacyConfig", "GateConfig", "AuditConfig", "load_privacy_config"]
