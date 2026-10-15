"""Pretrain (or continue pretraining) a MethylationMAE on a beta matrix.

This is both the single-stage trainer ("just fit my own cohort") and the
Stage-1 trainer for an optional broad-reference pretrain when ``x_path``
points at a combined corpus built by :mod:`pode_foundation.core.combine` --
the training loop is identical either way, only the checkpoint gets tagged
with ``stage=1``. See :mod:`pode_foundation.core.finetune` for the
warm-started Stage-2 variant that continues from a checkpoint this module
produces.

Watch the *validation* masked MSE, not the training MSE, for early stopping /
model selection -- on a modest sample size relative to parameter count you
will overfit if you only watch training loss.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from .mae_model import MethylationMAE, make_mask, masked_mse_loss
from .mae_model_weighted import make_feature_weight, weighted_masked_mse_loss


class _MethDataset(Dataset):
    def __init__(self, x: np.ndarray):
        self.x = torch.from_numpy(x)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


@dataclass
class TrainResult:
    """Outcome of a ``run_pretrain``/``run_finetune`` call.

    ``ckpt_path`` is the best-validation-loss checkpoint (not the final
    epoch's weights -- early stopping means those can differ), the one to
    pass as ``--ckpt``/``--init-ckpt`` to ``embed``/``finetune``/``survival``.
    ``epoch`` is which epoch produced it, for sanity-checking against
    ``training.patience``.
    """

    ckpt_path: Path
    history_path: Path
    best_val_mse: float
    epoch: int


def resolve_device(device: str | None) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_loop(
    model: MethylationMAE,
    x: np.ndarray,
    *,
    device: torch.device,
    mask_ratio: float,
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    val_frac: float,
    val_mask_draws: int,
    num_workers: int,
    amp: bool,
    seed: int,
    out_dir: Path,
    ckpt_extra: dict,
    lr_min_frac: float = 0.0,
    feat_weight: torch.Tensor | None = None,
) -> TrainResult:
    """Shared masked-reconstruction training loop used by pretrain and finetune.

    ``feat_weight``, if given, is a per-feature weight vector from
    :func:`pode_foundation.core.mae_model_weighted.make_feature_weight` that
    up-weights reconstruction error on a chosen subset of features (e.g. loci
    of particular interest for your tumour type). Leave it ``None`` for plain
    flat MSE.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = _MethDataset(x)
    n_val = max(1, int(val_frac * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    print(f"[split]  train={n_train}  val={n_val}")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model]  {n_params / 1e6:.1f}M params  mask_ratio={mask_ratio}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=lr * lr_min_frac)
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict] = []
    best_val = math.inf
    patience_left = patience
    ckpt_path = out_dir / "mae_best.pt"
    epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        tr_loss, nb = 0.0, 0
        for xb in train_loader:
            xb = xb.to(device, non_blocking=True)
            mask = make_mask(xb.shape, mask_ratio, device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                x_hat, _ = model(xb, mask)
                loss = (
                    weighted_masked_mse_loss(x_hat, xb, mask, feat_weight)
                    if feat_weight is not None
                    else masked_mse_loss(x_hat, xb, mask)
                )
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tr_loss += loss.item()
            nb += 1
        sched.step()
        tr_loss /= max(nb, 1)

        model.eval()
        with torch.no_grad():
            v_total, v_nb = 0.0, 0
            for xb in val_loader:
                xb = xb.to(device, non_blocking=True)
                for _ in range(val_mask_draws):
                    mask = make_mask(xb.shape, mask_ratio, device)
                    x_hat, _ = model(xb, mask)
                    v_loss = (
                        weighted_masked_mse_loss(x_hat, xb, mask, feat_weight)
                        if feat_weight is not None
                        else masked_mse_loss(x_hat, xb, mask)
                    )
                    v_total += v_loss.item()
                    v_nb += 1
            val_loss = v_total / max(v_nb, 1)

        dt = time.time() - t0
        history.append(
            {"epoch": epoch, "train_mse": tr_loss, "val_mse": val_loss, "lr": opt.param_groups[0]["lr"], "time_s": dt}
        )
        print(
            f"[ep {epoch:03d}] train {tr_loss:.5f}  val {val_loss:.5f}  "
            f"lr {opt.param_groups[0]['lr']:.2e}  ({dt:.1f}s)"
        )

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            patience_left = patience
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "n_features": model.n_features,
                    "best_val_mse": best_val,
                    "epoch": epoch,
                    **ckpt_extra,
                },
                ckpt_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[early stop] no val improvement for {patience} epochs")
                break

    history_path = out_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[done] best val masked MSE: {best_val:.5f}  ({ckpt_path})")
    return TrainResult(ckpt_path=ckpt_path, history_path=history_path, best_val_mse=best_val, epoch=epoch)


def run_pretrain(cfg, x_path=None, out_dir=None, stage_tag: int | None = None) -> TrainResult:
    """Pretrain a MethylationMAE from scratch on a preprocessed beta matrix (``X.npy``).

    Set ``stage_tag=1`` when this is the Stage-1 pass over a broad-reference
    corpus (see :mod:`pode_foundation.core.combine`); leave it ``None`` for a
    plain single-stage fit on your own cohort.
    """
    tr = cfg["training"]
    x_path = x_path or cfg["matrix"].get("path")
    if not x_path:
        raise ValueError("pretrain: no training matrix given (config matrix.path or --x-path).")
    x_path = Path(x_path)
    out = Path(out_dir or tr["output_dir"])
    device = resolve_device(tr["device"])
    print(f"[device] {device}")

    x = np.load(x_path)
    n_samples, n_features = x.shape
    print(f"[data]   X = {x.shape}")

    model_cfg = cfg["model"]
    model = MethylationMAE(
        n_features=n_features,
        encoder_dims=tuple(model_cfg["encoder_dims"]),
        latent_dim=model_cfg["latent_dim"],
        dropout=model_cfg["dropout"],
        output_activation=model_cfg["output_activation"],
    )

    train_args = {
        "encoder_dims": list(model_cfg["encoder_dims"]),
        "latent_dim": model_cfg["latent_dim"],
        "dropout": model_cfg["dropout"],
        "output_activation": model_cfg["output_activation"],
        "mask_ratio": tr["mask_ratio"],
    }
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as f:
        json.dump(train_args, f, indent=2)

    ckpt_extra = {"args": train_args}
    if stage_tag is not None:
        ckpt_extra["stage"] = stage_tag

    feat_weight = _build_feature_weight(tr.get("prognostic_weight"), n_features, device)

    return train_loop(
        model,
        x,
        device=device,
        mask_ratio=tr["mask_ratio"],
        batch_size=tr["batch_size"],
        epochs=tr["epochs"],
        patience=tr["patience"],
        lr=tr["lr"],
        weight_decay=tr["weight_decay"],
        val_frac=tr["val_frac"],
        val_mask_draws=tr.get("val_mask_draws", 3),
        num_workers=tr.get("num_workers", 0),
        amp=tr.get("amp", False),
        seed=tr["seed"],
        out_dir=out,
        ckpt_extra=ckpt_extra,
        feat_weight=feat_weight,
    )


def _build_feature_weight(prognostic_weight_cfg, n_features: int, device) -> torch.Tensor | None:
    """Build an optional per-feature reconstruction-loss weight from config.

    ``prognostic_weight: {path: <boolean mask .npy>, weight: <float>}`` lets you
    up-weight reconstruction on a chosen subset of loci (e.g. genes of known
    relevance to your tumour type). Returns None (plain flat MSE) if unset.
    """
    if not prognostic_weight_cfg or not prognostic_weight_cfg.get("path"):
        return None
    mask = np.load(prognostic_weight_cfg["path"], allow_pickle=True).astype(bool)
    if len(mask) != n_features:
        raise ValueError(
            f"prognostic_weight.path has {len(mask)} entries but the training matrix has "
            f"{n_features} features -- must be aligned to the same feature set."
        )
    return make_feature_weight(mask, w=float(prognostic_weight_cfg.get("weight", 1.0)), device=device)
