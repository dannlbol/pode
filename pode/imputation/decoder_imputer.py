"""Engine ``tvae-decoder``: multiple imputation from an existing PODE TVAE.

Works with any ``model.pkl`` written by ``pode synthesis train``, without
retraining. Because the saved model has no encoder, each incomplete row's
latent code is conditioned on its observed cells through the decoder alone
(SIR from the prior, refined with pCN MCMC where needed; see
:mod:`pode.imputation.latent`), then missing cells are sampled from the decoder.

In benchmarks, ``fit`` trains a fresh SDV TVAE on the complete rows of the
training data, since SDV mean-fills numeric NaNs during training.
"""

from __future__ import annotations

import dataclasses
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import SynthesizerConfig
from ..schema.schema import DatasetSchema
from .base import Imputer, subset_schema
from .config import LatentConfig
from .sdv_bridge import SDVDecoderBridge
from .spans import seed_transformer
from .tvae_common import draw_completions, summarise_diagnostics


class TVAEDecoderImputer(Imputer):
    name = "tvae-decoder"

    def __init__(
        self,
        synth_config: SynthesizerConfig | None = None,
        latent: LatentConfig | None = None,
        epochs: int | None = None,
        fit_seed: int | None = None,
        min_complete_rows: int = 20,
        device: str | None = None,
    ):
        super().__init__()
        self.device = device
        cfg = synth_config or SynthesizerConfig()
        if epochs is not None:
            cfg = dataclasses.replace(cfg, epochs=epochs)
        self.synth_config = dataclasses.replace(cfg, verbose=False)
        self.latent = latent or LatentConfig()
        self.fit_seed = fit_seed
        self.min_complete_rows = min_complete_rows
        self.bridge: SDVDecoderBridge | None = None

    # ---- construction ------------------------------------------------------
    @classmethod
    def from_model(
        cls,
        model_path: str | Path,
        schema: DatasetSchema | None = None,
        latent: LatentConfig | None = None,
        device: str | None = None,
    ) -> "TVAEDecoderImputer":
        """Load a PODE ``model.pkl``.

        ``device``: ``None``/``'auto'`` tries the device the model was saved on
        and falls back to CPU if the GPU is unavailable or full (common on a
        shared server); ``'cpu'`` forces CPU; ``'cuda'`` never falls back.
        """
        gen = _load_generator(model_path, schema, device)
        obj = cls(latent=latent, device=device)
        obj._attach(gen.synthesizer, gen.schema)
        return obj

    def _attach(self, synthesizer, schema: DatasetSchema) -> None:
        self.bridge = SDVDecoderBridge(synthesizer)
        self.schema = schema
        self.columns = list(self.bridge.columns)

    def _fit(self, df: pd.DataFrame, schema: DatasetSchema) -> "TVAEDecoderImputer":
        from ..generation.synthesizer import TVAEGenerator

        cols = [c for c in schema.columns if c in df.columns]
        complete = df[cols].dropna().reset_index(drop=True)
        if len(complete) < self.min_complete_rows:
            raise ValueError(
                f"tvae-decoder needs >= {self.min_complete_rows} complete rows to train; "
                f"got {len(complete)}. Use tvae-masked for incomplete training data."
            )
        sub_schema = subset_schema(schema, cols)
        t0 = time.perf_counter()
        if self.device == "cpu":
            synthesizer = _fit_sdv_on_cpu(sub_schema, self.synth_config, complete, self.fit_seed)
        else:
            synthesizer = TVAEGenerator(sub_schema, self.synth_config).fit(complete, seed=self.fit_seed).synthesizer
        self._attach(synthesizer, sub_schema)
        self._diagnostics["fit_rows"] = len(complete)
        self._diagnostics["fit_seconds"] = time.perf_counter() - t0
        return self

    # ---- imputation --------------------------------------------------------
    def _impute(self, df: pd.DataFrame, m: int, seed: int) -> list[pd.DataFrame]:
        if self.bridge is None:
            raise RuntimeError("Imputer is not fitted. Call fit() or from_model() first.")
        df = df.reset_index(drop=True)
        seed_transformer(self.bridge.transformer, seed)
        x, obs = self.bridge.encode(df)
        todo = np.flatnonzero(~obs.all(axis=1))
        draws = [df.copy() for _ in range(m)]
        if todo.size == 0:
            self._diagnostics.update(summarise_diagnostics({}, m))
            return draws

        power = self.latent.likelihood_power
        power = self.bridge.loss_factor if power is None else power
        t0 = time.perf_counter()
        samples, diag = draw_completions(
            self.bridge.decode, x[todo], obs[todo], self.bridge.spans, self.bridge.emb_dim,
            m, self.latent, power, self.bridge.device, seed,
        )
        for i in range(m):
            filled = self.bridge.to_frame(samples[:, i, :])
            filled.index = todo
            for col in self.columns:
                draws[i][col] = draws[i][col].astype(object)
                draws[i].loc[todo, col] = filled[col].to_numpy()
        self._diagnostics.update(summarise_diagnostics(diag, m))
        self._diagnostics["impute_seconds"] = time.perf_counter() - t0
        self._row_diagnostics = pd.DataFrame(
            {"row": todo, "ess": diag["ess"], "refined": diag["refined"],
             "mcmc_acceptance": diag["acceptance"], "smc_stages": diag["smc_stages"]}
        )
        return [d.infer_objects() for d in draws]


@contextmanager
def _cpu_storage():
    """Deserialise pickled torch tensors onto the CPU (SDV pickles CUDA tensors)."""
    import io

    import torch.storage

    original = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
    try:
        yield
    finally:
        torch.storage._load_from_bytes = original


def _load_generator(model_path, schema, device: str | None):
    from ..generation.synthesizer import TVAEGenerator

    if device not in (None, "auto", "cpu", "cuda"):
        raise ValueError(f"device must be auto, cpu or cuda; got '{device}'.")
    if device != "cpu":
        try:
            return TVAEGenerator.load(model_path, schema=schema)
        except RuntimeError as exc:
            if device == "cuda" or "CUDA" not in str(exc):
                raise
            warnings.warn(f"Could not load the model onto the GPU ({str(exc).splitlines()[0]}); using CPU.",
                          stacklevel=3)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    with _cpu_storage():
        gen = TVAEGenerator.load(model_path, schema=schema)
    model = gen.synthesizer._model
    model._device = torch.device("cpu")
    model.decoder.to("cpu")
    return gen


def _fit_sdv_on_cpu(schema: DatasetSchema, cfg: SynthesizerConfig, data: pd.DataFrame, seed: int | None):
    """Same as TVAEGenerator.fit, but with SDV's ``cuda=False``."""
    from sdv.single_table import TVAESynthesizer

    from ..reproducibility.seeds import set_all_seeds

    if seed is not None:
        set_all_seeds(seed)
    synthesizer = TVAESynthesizer(
        schema.to_sdv_metadata(), epochs=cfg.epochs, verbose=cfg.verbose, cuda=False, **cfg.tvae_kwargs()
    )
    synthesizer.fit(data)
    return synthesizer
