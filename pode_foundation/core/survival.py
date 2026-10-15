"""Optional survival evaluation of MAE embeddings.

Two modes (``survival.mode`` in config):

  frozen   - fit a regularised Cox model (``lifelines``) on a fixed
             embedding, out-of-fold across a fold column (or an
             auto-generated stratified K-fold if none is given). Fast, no
             torch training required for this step.
  finetune - fine-tune the encoder + a linear risk head directly against
             survival (Cox partial-likelihood loss), out-of-fold per fold.
             Needs a pretrain/finetune checkpoint and the original beta
             matrix (not just the embeddings CSV).

Both report Harrell's C-index on held-out folds -- this is an *internal*
number: if the same cohort's score-correlations informed feature selection or
model choice upstream, treat it as optimistically biased and prefer an
independent validation cohort for the headline number, if you have one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .embed import _load_betas, _load_feature_ids, align_betas
from .mae_model import MethylationMAE


@dataclass
class SurvivalResult:
    """Outcome of ``run_survival_frozen``/``run_survival_finetune``.

    ``c_index`` is Harrell's C computed out-of-fold (never on a sample's own
    training fold) -- still an internal number, see the module docstring's
    leakage caveat. ``method`` records which of the two modes produced it, so
    results from different runs aren't compared as if they were the same
    thing. ``risk_path`` has the per-sample OOF risk score alongside
    time/event/fold, for downstream KM plots or re-scoring.
    """

    c_index: float
    method: str
    risk_path: Path
    n_samples: int
    n_events: int


def harrell_c(risk, time, event) -> float:
    """Harrell's concordance index. NaN risks are dropped."""
    risk, time = np.asarray(risk, float), np.asarray(time, float)
    event = np.asarray(event).astype(bool)
    ok = ~np.isnan(risk)
    risk, time, event = risk[ok], time[ok], event[ok]
    ti, tj = time[:, None], time[None, :]
    comparable = event[:, None] & (ti < tj)
    ri, rj = risk[:, None], risk[None, :]
    concordant = (comparable & (ri > rj)).sum()
    discordant = (comparable & (ri < rj)).sum()
    tied = (comparable & (ri == rj)).sum()
    denom = concordant + discordant + tied
    if denom == 0:
        return float("nan")
    c = (concordant + 0.5 * tied) / denom
    # risk and hazard can point either way depending on the fitter's sign
    # convention; report the concordant orientation.
    return float(1 - c if c < 0.5 else c)


def _make_folds(n: int, event: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    from sklearn.model_selection import StratifiedKFold

    folds = np.zeros(n, dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(skf.split(np.zeros(n), event)):
        folds[te] = k
    return folds


def _load_clinical(clinical_path, id_col, time_col, event_col, fold_col, embedding_ids, n_splits, seed):
    clin = pd.read_csv(clinical_path)
    for col in (id_col, time_col, event_col):
        if col not in clin.columns:
            raise ValueError(f"survival: clinical CSV is missing required column '{col}'.")
    clin = clin.set_index(id_col)
    common = [s for s in embedding_ids if s in clin.index]
    if not common:
        raise ValueError("survival: no overlapping sample IDs between embeddings/betas and clinical CSV.")
    clin = clin.loc[common]
    time = clin[time_col].to_numpy(float)
    event = clin[event_col].to_numpy(int)
    if fold_col and fold_col in clin.columns:
        folds = clin[fold_col].to_numpy(int)
    else:
        folds = _make_folds(len(common), event, n_splits, seed)
    return common, time, event, folds


def run_survival_frozen(cfg, embeddings_path, clinical_path=None, out_dir=None) -> SurvivalResult:
    """Frozen-embedding OOF Cox benchmark (no torch training)."""
    from lifelines import CoxPHFitter
    from sklearn.preprocessing import StandardScaler

    surv = cfg["survival"]
    clinical_path = clinical_path or surv.get("clinical_path")
    if not clinical_path:
        raise ValueError("survival: no clinical CSV given (config survival.clinical_path or --clinical).")

    emb = pd.read_csv(embeddings_path, index_col=0)
    ids, time, event, folds = _load_clinical(
        clinical_path,
        surv.get("id_column", "sample_id"),
        surv["time_column"],
        surv["event_column"],
        surv.get("fold_column"),
        list(emb.index),
        surv.get("n_folds", 5),
        surv.get("seed", 42),
    )
    latent = emb.loc[ids].to_numpy(dtype=np.float64)
    cols = [f"z{i:03d}" for i in range(latent.shape[1])]
    penalizer = surv.get("penalizer", 0.1)
    l1_ratio = surv.get("l1_ratio", 0.5)

    risk = np.full(len(latent), np.nan)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        if event[tr].sum() < 1:
            continue
        scaler = StandardScaler().fit(latent[tr])
        df_tr = pd.DataFrame(scaler.transform(latent[tr]), columns=cols)
        df_tr["time"], df_tr["event"] = time[tr], event[tr]
        cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        cph.fit(df_tr, duration_col="time", event_col="event")
        df_te = pd.DataFrame(scaler.transform(latent[te]), columns=cols)
        risk[te] = cph.predict_partial_hazard(df_te).to_numpy()

    c = harrell_c(risk, time, event)
    out = Path(out_dir or surv.get("output_dir", "outputs/foundation/survival"))
    out.mkdir(parents=True, exist_ok=True)
    risk_path = out / "frozen_oof_risk.csv"
    pd.DataFrame({"risk": risk, "time": time, "event": event, "fold": folds}, index=pd.Index(ids, name="sample_id")).to_csv(
        risk_path
    )
    print(f"[survival:frozen] OOF C-index = {c:.3f}  (n={len(ids)}, events={int(event.sum())})")
    return SurvivalResult(
        c_index=c, method="frozen (lifelines CoxPHFitter, OOF)", risk_path=risk_path, n_samples=len(ids), n_events=int(event.sum())
    )


# ── finetune mode: encoder + linear head, direct Cox supervision ───────────
def _build_head(latent_dim: int, device):
    """Build the linear risk head. A factory (not a module-level class) so
    torch.nn is only imported when this function actually runs."""
    import torch.nn as nn

    class _Head(nn.Module):
        """Linear risk head on a standardised latent (no bias -- Cox has no intercept)."""

        def __init__(self):
            super().__init__()
            self.norm = nn.BatchNorm1d(latent_dim, affine=False)
            self.fc = nn.Linear(latent_dim, 1, bias=False)

        def forward(self, z):
            return self.fc(self.norm(z)).squeeze(-1)

    return _Head().to(device)


def cox_ph_loss(risk, time, event):
    import torch

    if event.sum() < 1:
        return risk.sum() * 0.0
    order = torch.argsort(time, descending=True)
    risk, event = risk[order], event[order].float()
    return -((risk - torch.logcumsumexp(risk, dim=0)) * event).sum() / event.sum().clamp_min(1.0)


def set_trainable(mae, mode: str):
    for p in mae.parameters():
        p.requires_grad = mode != "frozen"
    if mode == "partial":
        import torch.nn as nn

        first_linear = next(m for m in mae.encoder if isinstance(m, nn.Linear))
        for p in first_linear.parameters():
            p.requires_grad = False


def run_survival_finetune(cfg, ckpt, betas_path=None, clinical_path=None, out_dir=None) -> SurvivalResult:
    """Direct encoder+head Cox fine-tune, out-of-fold per fold (no leakage)."""
    import torch

    surv = cfg["survival"]
    clinical_path = clinical_path or surv.get("clinical_path")
    if not clinical_path:
        raise ValueError("survival: no clinical CSV given (config survival.clinical_path or --clinical).")
    betas_path = betas_path or cfg["matrix"].get("path")
    if not betas_path:
        raise ValueError("survival (finetune mode): no beta matrix given (config matrix.path or --betas).")
    feature_ids_path = cfg["embed"].get("feature_ids_path") or cfg["preprocess"].get("feature_ids_path")
    if not feature_ids_path:
        raise ValueError("survival (finetune mode): no feature-id list given (config embed.feature_ids_path).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_state = torch.load(ckpt, map_location="cpu", weights_only=False)
    n_features = ckpt_state["n_features"]
    args = ckpt_state.get("args", {})

    feat = _load_feature_ids(feature_ids_path)
    if len(feat) != n_features:
        raise ValueError(
            f"survival (finetune mode): feature-id list has {len(feat)} entries but the "
            f"checkpoint expects {n_features} -- must be the same feature set the checkpoint was trained on."
        )

    # First pass over the raw beta file to discover which sample IDs are
    # present, so folds/clinical can be resolved before the (potentially
    # large) aligned matrix is built.
    all_ids = list(_load_betas(betas_path, cfg["matrix"]["orientation"]).columns)
    ids, time, event, folds = _load_clinical(
        clinical_path,
        surv.get("id_column", "sample_id"),
        surv["time_column"],
        surv["event_column"],
        surv.get("fold_column"),
        all_ids,
        surv.get("n_folds", 5),
        surv.get("seed", 42),
    )
    x_all, ids = align_betas(betas_path, cfg["matrix"]["orientation"], feat, sample_ids=ids)
    latent_dim = args.get("latent_dim", cfg["model"]["latent_dim"])
    head_lr = surv.get("head_lr", 1e-3)
    encoder_lr = surv.get("encoder_lr", 1e-5)
    mode = surv.get("finetune_mode", "partial")
    epochs = surv.get("finetune_epochs", 100)

    oof = np.full(len(x_all), np.nan)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        mae = MethylationMAE(
            n_features=n_features,
            encoder_dims=tuple(args.get("encoder_dims", cfg["model"]["encoder_dims"])),
            latent_dim=latent_dim,
            dropout=args.get("dropout", 0.0),
            output_activation=args.get("output_activation", "sigmoid"),
        )
        mae.load_state_dict(ckpt_state["model_state"])
        mae.to(device)
        head = _build_head(latent_dim, device)
        set_trainable(mae, mode)

        x_tr = torch.tensor(x_all[tr], device=device)
        t_tr = torch.tensor(time[tr], device=device, dtype=torch.float32)
        e_tr = torch.tensor(event[tr], device=device, dtype=torch.float32)
        enc_params = [p for p in mae.parameters() if p.requires_grad]
        param_groups = [{"params": head.parameters(), "lr": head_lr, "weight_decay": 1e-3}]
        if enc_params:
            param_groups.append({"params": enc_params, "lr": encoder_lr, "weight_decay": 1e-4})
        opt = torch.optim.AdamW(param_groups)

        for _ in range(epochs):
            mae.train()
            head.train()
            loss = cox_ph_loss(head(mae.encode(x_tr)), t_tr, e_tr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(mae.parameters()) + list(head.parameters()), 1.0)
            opt.step()

        mae.eval()
        head.eval()
        with torch.no_grad():
            oof[te] = head(mae.encode(torch.tensor(x_all[te], device=device))).cpu().numpy()

    c = harrell_c(oof, time, event)
    out = Path(out_dir or surv.get("output_dir", "outputs/foundation/survival"))
    out.mkdir(parents=True, exist_ok=True)
    risk_path = out / "finetune_oof_risk.csv"
    pd.DataFrame({"risk": oof, "time": time, "event": event, "fold": folds}, index=pd.Index(ids, name="sample_id")).to_csv(
        risk_path
    )
    print(f"[survival:finetune] OOF C-index = {c:.3f}  (n={len(ids)}, events={int(event.sum())})")
    return SurvivalResult(
        c_index=c, method="finetune (encoder+head Cox, OOF)", risk_path=risk_path, n_samples=len(ids), n_events=int(event.sum())
    )


def run_survival(cfg, mode=None, embeddings_path=None, ckpt=None, betas_path=None, clinical_path=None, out_dir=None) -> SurvivalResult:
    mode = mode or cfg["survival"].get("mode", "frozen")
    if mode == "frozen":
        if not embeddings_path:
            raise ValueError("survival (frozen mode): --embeddings is required.")
        return run_survival_frozen(cfg, embeddings_path, clinical_path=clinical_path, out_dir=out_dir)
    if mode == "finetune":
        if not ckpt:
            raise ValueError("survival (finetune mode): --ckpt is required.")
        return run_survival_finetune(cfg, ckpt, betas_path=betas_path, clinical_path=clinical_path, out_dir=out_dir)
    raise ValueError(f"survival: unknown mode '{mode}' (expected 'frozen' or 'finetune').")
