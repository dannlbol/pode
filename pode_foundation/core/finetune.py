"""Stage-2 domain-adaptive fine-tuning (DAPT) onto a target cohort.

Loads a checkpoint produced by :func:`pode_foundation.core.train.run_pretrain`
(Stage-1, e.g. pretrained on a broad public reference corpus) and continues
training on a new beta matrix -- typically your own target cohort -- with:

  * A lower learning rate (default 10x below the Stage-1 default).
  * A freshly restarted cosine schedule over the fine-tuning horizon.
  * The same architecture and masking objective (no architectural changes).
  * Reduced early-stopping patience.

This is the seam for "pretrain broadly, then adapt to my cohort": run
``run_pretrain`` once on whatever broad reference you can source (or skip
this entirely and just call ``run_pretrain`` directly on your own cohort if
you don't have/want a broader corpus -- both are legitimate, and the
downstream ``embed``/``survival`` steps don't care which path produced the
checkpoint).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch

from .mae_model import MethylationMAE
from .train import TrainResult, resolve_device, train_loop


def run_finetune(cfg, init_ckpt, x_path=None, out_dir=None) -> TrainResult:
    """Warm-start from ``init_ckpt`` and continue training on ``x_path``.

    Architecture and ``mask_ratio`` are read from the checkpoint's own
    ``args``, not from ``cfg["model"]``/``cfg["training"]`` -- a fine-tune
    always matches the model it's warm-starting from, even if the config's
    ``model`` section has since changed for an unrelated run.
    """
    ft = cfg["finetune_stage2"]
    tr = cfg["training"]

    init_ckpt = init_ckpt or ft.get("init_ckpt")
    if not init_ckpt:
        raise ValueError("finetune: no Stage-1 checkpoint given (config finetune_stage2.init_ckpt or --init-ckpt).")
    x_path = x_path or cfg["matrix"].get("path")
    if not x_path:
        raise ValueError("finetune: no fine-tuning matrix given (config matrix.path or --x-path).")
    init_ckpt, x_path = Path(init_ckpt), Path(x_path)
    out = Path(out_dir or ft["output_dir"])
    device = resolve_device(tr["device"])
    print(f"[device] {device}")

    print(f"[init]   loading Stage-1 checkpoint: {init_ckpt}")
    ckpt = torch.load(init_ckpt, map_location="cpu", weights_only=False)
    stage1_args = ckpt.get("args", {})
    n_features = ckpt["n_features"]
    print(f"         stage-1 val MSE: {ckpt.get('best_val_mse', '?')}  epoch: {ckpt.get('epoch', '?')}")

    x = np.load(x_path)
    if x.shape[1] != n_features:
        raise ValueError(
            f"finetune: fine-tune data has {x.shape[1]} features, but the Stage-1 "
            f"checkpoint expects {n_features}. Both stages must share the same "
            f"preprocessed feature set (see core.preprocess)."
        )
    print(f"[data]   X = {x.shape}")

    encoder_dims = tuple(stage1_args.get("encoder_dims", cfg["model"]["encoder_dims"]))
    latent_dim = stage1_args.get("latent_dim", cfg["model"]["latent_dim"])
    dropout = stage1_args.get("dropout", cfg["model"]["dropout"])
    output_activation = stage1_args.get("output_activation", cfg["model"]["output_activation"])
    mask_ratio = stage1_args.get("mask_ratio", tr["mask_ratio"])

    model = MethylationMAE(
        n_features=n_features,
        encoder_dims=encoder_dims,
        latent_dim=latent_dim,
        dropout=dropout,
        output_activation=output_activation,
    )
    state_dict = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        warnings.warn(f"load_state_dict: missing={missing}  unexpected={unexpected}")
    print(f"[model]  Stage-1 weights loaded  encoder={encoder_dims}  latent={latent_dim}  mask={mask_ratio}")

    ckpt_extra = {
        "args": {
            "encoder_dims": list(encoder_dims),
            "latent_dim": latent_dim,
            "dropout": dropout,
            "output_activation": output_activation,
            "mask_ratio": mask_ratio,
        },
        "stage": 2,
        "stage1_ckpt": str(init_ckpt),
    }

    return train_loop(
        model,
        x,
        device=device,
        mask_ratio=mask_ratio,
        batch_size=ft.get("batch_size", tr["batch_size"]),
        epochs=ft["epochs"],
        patience=ft.get("patience", 20),
        lr=ft["lr"],
        weight_decay=ft.get("weight_decay", tr["weight_decay"]),
        val_frac=tr["val_frac"],
        val_mask_draws=tr.get("val_mask_draws", 3),
        num_workers=tr.get("num_workers", 0),
        amp=tr.get("amp", False),
        seed=tr["seed"],
        out_dir=out,
        ckpt_extra=ckpt_extra,
        lr_min_frac=0.01,
    )
