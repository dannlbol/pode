"""Federated learning for PODE: generators and signatures without pooling patient rows.

Four capabilities, all running over :class:`~pode.federated.sites.Site`
objects that keep their rows private and answer only *site operations*
(every answer is logged in a :class:`~pode.federated.ledger.DisclosureLedger`):

* **Federated TVAE** (:mod:`.tvae`) - one generator trained by FedAvg/FedProx
  across sites, with a shared column encoder built from per-site summaries
  (:mod:`.transformer`), simulated secure aggregation (:mod:`.aggregate`) and
  optional per-site personalisation;
* **DP-SGD** (:mod:`.dp`) - per-sample clipping and Gaussian noise with Opacus
  accounting, for federated or local training;
* **Local exchange** (:mod:`.exchange`) - each site fits its own TVAE and
  releases synthetic rows only if they pass the :mod:`pode.privacy` gate;
* **Federated signatures** (:mod:`.cox`, :mod:`.signatures`) - MetaCPG's
  Cox screening, meta-analysis and leave-one-out search from per-site score,
  information and summary statistics only, reproducing the centralised run.

:mod:`.benchmark` evaluates the generators on fidelity, utility (on each
site's real held-out patients and on held-out sites) and privacy. Sites are
simulated in one process today; ``Site.run`` is the seam where a networked
client would plug in. See ``docs/federated/FEDERATED.md``.
"""

from .config import FederatedConfig, load_federated_config
from .ledger import DisclosureLedger
from .sites import Site, load_site_manifest, partition_sites, stack_sites

__all__ = [
    "DisclosureLedger",
    "FederatedConfig",
    "Site",
    "load_federated_config",
    "load_site_manifest",
    "partition_sites",
    "stack_sites",
]
