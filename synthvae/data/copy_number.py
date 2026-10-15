"""Segment-table to fixed-matrix converter for DNA copy number.

Copy-number data arrives as *segments*: one row per (sample, genomic interval)
with a continuous log2 ratio (``Seg.CN``) and, optionally, a categorical call
(``Alteration`` in {loss, balanced, gain}). Samples have different numbers of
segments at different breakpoints, so the table cannot be fed to a VAE directly.

This module collapses that long table into an aligned ``samples x bins`` matrix
using fixed-width genomic bins, mirroring how expression/methylation arrive as a
``samples x features`` matrix. Every bin's value is the *length-weighted* summary
of the segments overlapping it, so a single large segment and many small ones are
weighted by how much of the bin they actually cover.

The transform is a one-time, offline step (unlike
:class:`~synthvae.data.preprocessing.ExpressionPreprocessor`, which runs per
epoch on an already-aligned matrix), so it lives here as a pure numpy/pandas
function with no torch dependency. Column order and sample order are frozen in
:class:`BinGrid` / :class:`CopyNumberMatrices` so train and synthesis stay
consistent, the same contract ``ExpressionPreprocessor.selected_`` provides.

Representations
---------------
* ``continuous`` -- length-weighted mean ``Seg.CN`` per bin (Gaussian-friendly).
* ``discrete``   -- a per-bin gain/balanced/loss call, encoded as either
  ``ordinal`` (one column in {-1, 0, 1}, for the Gaussian hybrid VAE) or
  ``gain_loss_binary`` (two 0/1 columns per bin, for the MMVAE Bernoulli head).
* ``both``       -- return both frames sharing the same sample order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


CHROM_ORDER = [f"chr{i}" for i in range(1, 23)]  # autosomes
ALTERATION_TO_ORDINAL = {"loss": -1, "balanced": 0, "gain": 1}
ORDINAL_TO_ALTERATION = {-1: "loss", 0: "balanced", 1: "gain"}
_CALLS = ("loss", "balanced", "gain")


@dataclass
class BinGrid:
    """A frozen set of fixed-width genomic bins.

    ``spans`` and ``columns`` are index-aligned and ordered by ``chroms``.
    """

    chrom_lengths: dict[str, int]
    bin_width: int
    columns: list[str]                    # e.g. "chr1:0-10000000"
    spans: list[tuple[str, int, int]]     # (chrom, start, end), end exclusive


@dataclass
class CopyNumberMatrices:
    """Output of :func:`segments_to_matrix`.

    ``continuous`` and/or ``discrete`` are ``samples x bins`` DataFrames whose
    row index is ``samples``. ``discrete`` has one column per bin for the
    ``ordinal`` encoding, or two per bin for ``gain_loss_binary``.
    """

    grid: BinGrid
    samples: list[str]
    continuous: pd.DataFrame | None = None
    discrete: pd.DataFrame | None = None


def infer_chrom_lengths(
    segments: pd.DataFrame,
    *,
    chroms: Sequence[str] = CHROM_ORDER,
    chrom_col: str = "Chromosome",
    end_col: str = "End_Position",
) -> dict[str, int]:
    """Self-contained chromosome lengths: max ``End_Position`` per chromosome.

    Pass an explicit genome-build table via ``chrom_lengths`` to
    :func:`segments_to_matrix` to override this data-derived default.
    """
    ends = segments.groupby(chrom_col)[end_col].max()
    lengths: dict[str, int] = {}
    for chrom in chroms:
        if chrom in ends.index:
            lengths[chrom] = int(ends.loc[chrom])
    if not lengths:
        raise ValueError(
            f"no rows matched any of chroms={list(chroms)!r} in column "
            f"{chrom_col!r}; check chromosome naming (e.g. 'chr1' vs '1')"
        )
    return lengths


def make_bin_grid(
    chrom_lengths: dict[str, int],
    *,
    bin_width: int = 10_000_000,
    chroms: Sequence[str] = CHROM_ORDER,
) -> BinGrid:
    """Tile each chromosome into fixed-width bins (last bin truncated)."""
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")
    columns: list[str] = []
    spans: list[tuple[str, int, int]] = []
    kept_lengths: dict[str, int] = {}
    for chrom in chroms:
        length = chrom_lengths.get(chrom)
        if length is None:
            continue
        kept_lengths[chrom] = int(length)
        start = 0
        while start < length:
            end = min(start + bin_width, int(length))
            columns.append(f"{chrom}:{start}-{end}")
            spans.append((chrom, start, end))
            start += bin_width
    if not columns:
        raise ValueError("no bins produced; check chrom_lengths and chroms")
    return BinGrid(
        chrom_lengths=kept_lengths,
        bin_width=int(bin_width),
        columns=columns,
        spans=spans,
    )


def _accumulate_sample(
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
    seg_values: np.ndarray,
    seg_calls: np.ndarray | None,
    bin_starts: np.ndarray,
    bin_ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Length-weighted accumulation of one chromosome's segments into bins.

    Returns ``(weighted_value_sum, weight_sum, call_weight_by_class)`` where the
    last entry is ``(n_bins, 3)`` length per {loss, balanced, gain} or ``None``
    when no calls are supplied.
    """
    n_bins = len(bin_starts)
    wv = np.zeros(n_bins)
    w = np.zeros(n_bins)
    call_w = None if seg_calls is None else np.zeros((n_bins, len(_CALLS)))
    for i in range(len(seg_starts)):
        s, e, v = seg_starts[i], seg_ends[i], seg_values[i]
        # overlap of [s, e) with every bin, vectorised across bins
        overlap = np.minimum(e, bin_ends) - np.maximum(s, bin_starts)
        np.clip(overlap, 0, None, out=overlap)
        if not overlap.any():
            continue
        wv += overlap * v
        w += overlap
        if call_w is not None:
            call_w[:, seg_calls[i]] += overlap
    return wv, w, call_w


def segments_to_matrix(
    segments: pd.DataFrame,
    *,
    bin_width: int = 10_000_000,
    representation: str = "continuous",
    chrom_lengths: dict[str, int] | None = None,
    chroms: Sequence[str] = CHROM_ORDER,
    sample_col: str = "Sample",
    chrom_col: str = "Chromosome",
    start_col: str = "Start_Position",
    end_col: str = "End_Position",
    value_col: str = "Seg.CN",
    alteration_col: str = "Alteration",
    discrete_encoding: str = "ordinal",
    discrete_from: str = "alteration",
    gain_threshold: float = 0.2,
    loss_threshold: float = -0.2,
    neutral_fill: float = 0.0,
    sample_order: Sequence[str] | None = None,
) -> CopyNumberMatrices:
    """Collapse a copy-number segment table into an aligned ``samples x bins`` matrix.

    Parameters
    ----------
    segments:
        Long-format table, one row per (sample, segment).
    bin_width:
        Fixed genomic bin width in base pairs (default 10 Mb).
    representation:
        ``"continuous"`` | ``"discrete"`` | ``"both"``.
    chrom_lengths:
        Optional {chrom: length} override; defaults to :func:`infer_chrom_lengths`.
    discrete_encoding:
        ``"ordinal"`` (one {-1,0,1} column/bin) | ``"gain_loss_binary"`` (two 0/1
        columns/bin named ``"<bin>::gain"`` / ``"<bin>::loss"``).
    discrete_from:
        ``"alteration"`` (length-weighted dominant call from ``alteration_col``)
        | ``"threshold"`` (threshold the continuous value; use for tables lacking
        a call column, e.g. a GISTIC-style export).
    neutral_fill:
        Continuous value for bins with no overlapping segment (0.0 = neutral log2).
    sample_order:
        Optional explicit row order; missing samples are filled (neutral) so the
        matrix aligns with an external clinical/survival table.
    """
    if representation not in ("continuous", "discrete", "both"):
        raise ValueError(
            "representation must be 'continuous', 'discrete', or 'both'"
        )
    if discrete_encoding not in ("ordinal", "gain_loss_binary"):
        raise ValueError(
            "discrete_encoding must be 'ordinal' or 'gain_loss_binary'"
        )
    if discrete_from not in ("alteration", "threshold"):
        raise ValueError("discrete_from must be 'alteration' or 'threshold'")

    want_continuous = representation in ("continuous", "both")
    want_discrete = representation in ("discrete", "both")
    use_calls = want_discrete and discrete_from == "alteration"

    for col in (sample_col, chrom_col, start_col, end_col, value_col):
        if col not in segments.columns:
            raise ValueError(f"segments is missing required column {col!r}")
    if use_calls and alteration_col not in segments.columns:
        raise ValueError(
            f"discrete_from='alteration' requires column {alteration_col!r}; "
            "use discrete_from='threshold' for tables without calls"
        )

    if chrom_lengths is None:
        chrom_lengths = infer_chrom_lengths(
            segments, chroms=chroms, chrom_col=chrom_col, end_col=end_col
        )
    grid = make_bin_grid(chrom_lengths, bin_width=bin_width, chroms=chroms)

    # Determine sample rows.
    present = list(pd.unique(segments[sample_col]))
    if sample_order is not None:
        samples = [str(s) for s in sample_order]
    else:
        samples = [str(s) for s in present]
    sample_pos = {s: i for i, s in enumerate(samples)}

    # Per-chromosome bin index ranges, for fast slicing.
    chrom_bins: dict[str, tuple[int, int, np.ndarray, np.ndarray]] = {}
    for j, (chrom, s, e) in enumerate(grid.spans):
        lo, _, _, _ = chrom_bins.get(chrom, (j, j, None, None))
        chrom_bins[chrom] = (lo, j + 1, None, None)
    for chrom, (lo, hi, _, _) in list(chrom_bins.items()):
        starts = np.array([grid.spans[k][1] for k in range(lo, hi)], dtype=float)
        ends = np.array([grid.spans[k][2] for k in range(lo, hi)], dtype=float)
        chrom_bins[chrom] = (lo, hi, starts, ends)

    n_bins = len(grid.columns)
    n_samples = len(samples)
    wv_sum = np.zeros((n_samples, n_bins))
    w_sum = np.zeros((n_samples, n_bins))
    call_sum = np.zeros((n_samples, n_bins, len(_CALLS))) if use_calls else None

    call_to_idx = {c: i for i, c in enumerate(_CALLS)}
    for (sample, chrom), block in segments.groupby(
        [sample_col, chrom_col], sort=False
    ):
        sample = str(sample)
        if sample not in sample_pos or chrom not in chrom_bins:
            continue
        row = sample_pos[sample]
        lo, hi, bstarts, bends = chrom_bins[chrom]
        seg_starts = block[start_col].to_numpy(dtype=float)
        seg_ends = block[end_col].to_numpy(dtype=float)
        seg_values = block[value_col].to_numpy(dtype=float)
        seg_calls = None
        if use_calls:
            raw = block[alteration_col].astype(str).str.lower().to_numpy()
            seg_calls = np.array(
                [call_to_idx.get(c, call_to_idx["balanced"]) for c in raw]
            )
        wv, w, cw = _accumulate_sample(
            seg_starts, seg_ends, seg_values, seg_calls, bstarts, bends
        )
        wv_sum[row, lo:hi] += wv
        w_sum[row, lo:hi] += w
        if call_sum is not None and cw is not None:
            call_sum[row, lo:hi, :] += cw

    covered = w_sum > 0
    continuous = None
    if want_continuous or (want_discrete and discrete_from == "threshold"):
        cont = np.full((n_samples, n_bins), float(neutral_fill))
        cont[covered] = wv_sum[covered] / w_sum[covered]
        continuous = pd.DataFrame(cont, index=samples, columns=grid.columns)

    result = CopyNumberMatrices(grid=grid, samples=samples)
    if want_continuous:
        result.continuous = continuous

    if want_discrete:
        if discrete_from == "threshold":
            base = continuous.to_numpy()
            ordinal = np.zeros_like(base, dtype=int)
            ordinal[base > gain_threshold] = 1
            ordinal[base < loss_threshold] = -1
        else:
            # length-weighted dominant call; uncovered bins default to balanced
            ordinal = np.zeros((n_samples, n_bins), dtype=int)
            dominant = call_sum.argmax(axis=2)  # index into _CALLS
            any_call = call_sum.sum(axis=2) > 0
            mapping = np.array([ALTERATION_TO_ORDINAL[c] for c in _CALLS])
            ord_vals = mapping[dominant]
            ordinal = np.where(any_call, ord_vals, 0)
        result.discrete = _encode_discrete(
            ordinal, grid.columns, samples, discrete_encoding
        )

    return result


def _encode_discrete(
    ordinal: np.ndarray,
    columns: Sequence[str],
    samples: Sequence[str],
    encoding: str,
) -> pd.DataFrame:
    if encoding == "ordinal":
        return pd.DataFrame(
            ordinal.astype(float), index=list(samples), columns=list(columns)
        )
    # gain_loss_binary: two 0/1 columns per bin, never both set
    cols: list[str] = []
    data: list[np.ndarray] = []
    for j, name in enumerate(columns):
        col = ordinal[:, j]
        cols.append(f"{name}::gain")
        data.append((col == 1).astype(float))
        cols.append(f"{name}::loss")
        data.append((col == -1).astype(float))
    return pd.DataFrame(np.column_stack(data), index=list(samples), columns=cols)


def ordinal_to_calls(discrete: pd.DataFrame) -> pd.DataFrame:
    """Map an ordinal {-1,0,1} matrix back to {loss, balanced, gain} labels."""
    snapped = np.clip(np.rint(discrete.to_numpy()), -1, 1).astype(int)
    labels = np.vectorize(ORDINAL_TO_ALTERATION.get)(snapped)
    return pd.DataFrame(labels, index=discrete.index, columns=discrete.columns)


def gain_loss_binary_to_calls(discrete: pd.DataFrame) -> pd.DataFrame:
    """Map a ``gain_loss_binary`` matrix back to per-bin {loss, balanced, gain}."""
    bins: list[str] = []
    for col in discrete.columns:
        base = col.rsplit("::", 1)[0]
        if base not in bins:
            bins.append(base)
    out = np.full((len(discrete), len(bins)), "balanced", dtype=object)
    values = discrete.to_numpy()
    col_index = {c: i for i, c in enumerate(discrete.columns)}
    for j, base in enumerate(bins):
        gain = values[:, col_index[f"{base}::gain"]] >= 0.5
        loss = values[:, col_index[f"{base}::loss"]] >= 0.5
        out[gain, j] = "gain"
        out[loss & ~gain, j] = "loss"
    return pd.DataFrame(out, index=discrete.index, columns=bins)


def convert_csv(
    in_csv: str,
    out_csv: str | None = None,
    *,
    representation: str = "continuous",
    bin_width: int = 10_000_000,
    discrete_encoding: str = "ordinal",
    discrete_from: str = "alteration",
    sample_order_csv: str | None = None,
    sample_order_col: str = "Sample",
    sample_id_col: str | None = None,
    drop_index_column: bool = True,
    **kwargs,
) -> CopyNumberMatrices:
    """Read a segment CSV, convert, and optionally write the matrix CSV(s).

    When ``representation='both'`` and ``out_csv`` is given, the discrete matrix
    is written next to it with a ``_discrete`` suffix.

    ``sample_id_col`` controls sample identity in the written CSV(s). By default
    (``None``) the output is a pure numeric matrix aligned by row position -- the
    format the synthesis pipeline (``--copy-number`` / ``--omic``) consumes,
    matching ``expression.csv``. Set it (e.g. ``"Sample"``) to prepend a sample-id
    column so a downstream step can join copy number to clinical/expression/
    methylation by id before dropping the column.
    """
    segments = pd.read_csv(in_csv)
    if drop_index_column:
        segments = segments.drop(columns=["Unnamed: 0"], errors="ignore")
    sample_order = None
    if sample_order_csv is not None:
        sample_order = pd.read_csv(sample_order_csv)[sample_order_col].tolist()
    result = segments_to_matrix(
        segments,
        representation=representation,
        bin_width=bin_width,
        discrete_encoding=discrete_encoding,
        discrete_from=discrete_from,
        sample_order=sample_order,
        **kwargs,
    )

    def _write(df, path):
        if sample_id_col is not None:
            df = df.copy()
            df.insert(0, sample_id_col, list(df.index))
        df.to_csv(path, index=False)

    if out_csv is not None:
        if result.continuous is not None:
            _write(result.continuous, out_csv)
        if result.discrete is not None:
            disc_path = (
                out_csv
                if result.continuous is None
                else out_csv.replace(".csv", "_discrete.csv")
            )
            _write(result.discrete, disc_path)
    return result


def _build_arg_parser():
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m synthvae.data.copy_number",
        description="Collapse a copy-number segment table into an aligned "
        "samples x bins matrix CSV for the hybrid/MMVAE synthesis pipelines.",
    )
    ap.add_argument("--segments", required=True, help="input segment CSV")
    ap.add_argument("--out", required=True, help="output matrix CSV")
    ap.add_argument(
        "--representation", choices=["continuous", "discrete", "both"],
        default="continuous",
    )
    ap.add_argument("--bin-width", type=int, default=10_000_000)
    ap.add_argument(
        "--discrete-encoding", choices=["ordinal", "gain_loss_binary"],
        default="ordinal",
    )
    ap.add_argument(
        "--discrete-from", choices=["alteration", "threshold"], default="alteration"
    )
    ap.add_argument("--gain-threshold", type=float, default=0.2)
    ap.add_argument("--loss-threshold", type=float, default=-0.2)
    ap.add_argument(
        "--sample-order",
        default=None,
        help="CSV whose sample column fixes the output row order (align to "
        "clinical/survival); missing samples are neutral-filled",
    )
    ap.add_argument("--sample-order-col", default="Sample")
    ap.add_argument(
        "--sample-id-col",
        default=None,
        help="prepend a sample-id column with this header (e.g. 'Sample') so the "
        "matrix can be joined to clinical/expression by id; omit for the pure "
        "numeric, position-aligned matrix the synthesis pipeline consumes",
    )
    return ap


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    result = convert_csv(
        args.segments,
        args.out,
        representation=args.representation,
        bin_width=args.bin_width,
        discrete_encoding=args.discrete_encoding,
        discrete_from=args.discrete_from,
        gain_threshold=args.gain_threshold,
        loss_threshold=args.loss_threshold,
        sample_order_csv=args.sample_order,
        sample_order_col=args.sample_order_col,
        sample_id_col=args.sample_id_col,
    )
    n_bins = len(result.grid.columns)
    print(
        f"wrote {args.out} ({len(result.samples)} samples x {n_bins} bins, "
        f"representation={args.representation})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
