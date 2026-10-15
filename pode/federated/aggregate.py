"""Aggregation of site updates: weighting schemes and (simulated) secure aggregation.

**Weighting.** FedAvg weights each site's update by its patient count ``n``.
In childhood cancer one large centre can then dominate a federation of small
ones, so alternatives are offered: ``sqrt_n`` and ``uniform`` flatten the
weights, and ``capped`` keeps ``n``-weighting but limits any site to
``cap_fraction`` of the total (excess redistributed proportionally).

**Secure aggregation.** :class:`SecureSum` implements the pairwise-masking
protocol of Bonawitz et al. (2017) in its simplest, no-dropout form. Each site
quantises its (pre-weighted) update to fixed point and adds a mask
``sum_{j>i} m_ij - sum_{j<i} m_ji`` (mod 2^64), where ``m_ij`` is expanded from
a seed only sites i and j share. Each masked vector on its own is uniformly
random, but the masks cancel in the sum, so the coordinator learns only the
aggregate. In this in-process simulation the pair seeds are derived from one
secret, so it demonstrates the protocol's shape and numerical exactness, not
network security. A real deployment needs key agreement and dropout recovery.
"""

from __future__ import annotations

import hashlib

import numpy as np

FRAC_BITS = 24  # fixed-point resolution 2^-24 ~ 6e-8


def site_weights(ns: list[int], scheme: str = "n", cap_fraction: float = 0.5) -> np.ndarray:
    """Normalised aggregation weights for sites with ``ns`` patients."""
    n = np.asarray(ns, float)
    if scheme == "n":
        w = n
    elif scheme == "sqrt_n":
        w = np.sqrt(n)
    elif scheme == "uniform":
        w = np.ones_like(n)
    elif scheme == "capped":
        w = n / n.sum()
        cap = max(cap_fraction, 1.0 / len(n))  # a cap below 1/k is infeasible
        for _ in range(len(n)):
            over = w > cap + 1e-12
            if not over.any():
                break
            excess = float((w[over] - cap).sum())
            w[over] = cap
            under = ~over & (w < cap)
            if not under.any():
                break
            w[under] += excess * w[under] / w[under].sum()
        return w / w.sum()
    else:
        raise ValueError(f"Unknown weighting '{scheme}'.")
    return w / w.sum()


def flatten_state(state: dict) -> tuple[np.ndarray, list[tuple[str, tuple, object]]]:
    """A torch state dict as one float64 vector plus the layout to rebuild it."""
    layout, parts = [], []
    for k, v in state.items():
        a = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        layout.append((k, a.shape, v.dtype if hasattr(v, "dtype") else a.dtype))
        parts.append(a.astype(np.float64).ravel())
    return (np.concatenate(parts) if parts else np.zeros(0)), layout


def unflatten_state(vec: np.ndarray, layout, device=None) -> dict:
    import torch

    out, st = {}, 0
    for k, shape, dtype in layout:
        size = int(np.prod(shape)) if shape else 1
        t = torch.as_tensor(vec[st:st + size].reshape(shape))
        out[k] = t.to(dtype=dtype if isinstance(dtype, torch.dtype) else torch.float32,
                      device=device)
        st += size
    return out


def weighted_average(states: list[dict], weights: np.ndarray) -> dict:
    """Plain (non-secure) weighted average of torch state dicts."""
    vecs, layout = zip(*(flatten_state(s) for s in states))
    avg = np.sum([w * v for w, v in zip(weights, vecs)], axis=0)
    return unflatten_state(avg, layout[0])


class SecureSum:
    """Pairwise-masked summation of site vectors (no-dropout Bonawitz protocol, simulated)."""

    def __init__(self, n_sites: int, secret: int = 0, frac_bits: int = FRAC_BITS):
        self.n_sites = n_sites
        self.secret = int(secret)
        self.scale = float(2 ** frac_bits)

    def _pair_mask(self, i: int, j: int, round_: int, size: int) -> np.ndarray:
        digest = hashlib.sha256(f"{self.secret}:{round_}:{min(i, j)}:{max(i, j)}".encode()).digest()
        rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "little")))
        return rng.integers(0, 2 ** 63, size=size, dtype=np.uint64) * np.uint64(2) + \
            rng.integers(0, 2, size=size, dtype=np.uint64)

    def mask(self, site: int, round_: int, vector: np.ndarray) -> np.ndarray:
        """What site ``site`` sends: its fixed-point vector plus its pairwise masks (mod 2^64)."""
        q = np.round(np.asarray(vector, np.float64) * self.scale).astype(np.int64).view(np.uint64)
        with np.errstate(over="ignore"):
            for j in range(self.n_sites):
                if j == site:
                    continue
                m = self._pair_mask(site, j, round_, q.size)
                q = q + m if site < j else q - m
        return q

    def aggregate(self, masked: list[np.ndarray]) -> np.ndarray:
        """The coordinator's view: the sum of the sites' vectors (masks cancel)."""
        total = np.zeros_like(masked[0])
        with np.errstate(over="ignore"):
            for m in masked:
                total = total + m
        return total.view(np.int64).astype(np.float64) / self.scale


def secure_weighted_average(states: list[dict], weights: np.ndarray, secure: SecureSum, round_: int) -> dict:
    """Weighted average where each site pre-scales its update by its (public) weight and
    the coordinator only ever handles the masked vectors and their sum."""
    flat = [flatten_state(s) for s in states]
    layout = flat[0][1]
    masked = [secure.mask(i, round_, w * v) for i, ((v, _), w) in enumerate(zip(flat, weights))]
    return unflatten_state(secure.aggregate(masked), layout)
