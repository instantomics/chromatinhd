"""Mappings between the task's bin and query grids and ChromatinHD's parameters."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Upstream Shared-encoder resolutions. Levels that do not tile a dataset's region
# window are dropped; the finest level fixes the positional resolution.
UPSTREAM_BINWIDTHS = (5000, 1000, 500, 200, 100, 50, 25)
_BOOKKEEPING_COLUMNS = frozenset(
    {"query_index", "query_id", "reference_group_id", "unit_exposure", "weight"}
)


def regions_from_bins(bins: pd.DataFrame) -> pd.DataFrame:
    regions = bins.groupby("region_id", sort=False).agg(
        chrom=("chrom", "first"), start=("start", "min"), end=("end", "max")
    )
    regions.index = regions.index.astype(str)
    regions["width"] = regions["end"] - regions["start"]
    return regions


def window(max_width: int) -> tuple[int, tuple[int, ...]]:
    """Common region window and the upstream resolutions that tile it."""
    finest = UPSTREAM_BINWIDTHS[-1]
    width = math.ceil(max_width / finest) * finest
    return width, tuple(value for value in UPSTREAM_BINWIDTHS if width % value == 0)


def integrate_bins(
    mass: np.ndarray,
    finest: int,
    bin_region: np.ndarray,
    bin_start: np.ndarray,
    bin_end: np.ndarray,
) -> np.ndarray:
    """Cluster-by-bin expected counts from region-by-cluster-by-model-bin mass.

    Task bin coordinates are relative to their region start. Mass is uniform
    within each finest model bin, so partially overlapping bins receive their
    proportional share.
    """
    cumulative = np.concatenate(
        [np.zeros(mass.shape[:2] + (1,)), np.cumsum(mass, axis=-1)], axis=-1
    )
    lower = _cumulative_at(cumulative, bin_region, bin_start, finest)
    upper = _cumulative_at(cumulative, bin_region, bin_end, finest)
    return np.maximum(upper - lower, 0.0)


def _cumulative_at(cumulative, region, position, finest):
    n_bins = cumulative.shape[-1] - 1
    left = np.clip(position // finest, 0, n_bins - 1)
    fraction = np.clip((position - left * finest) / finest, 0.0, 1.0)
    below = cumulative[region, :, left]
    above = cumulative[region, :, left + 1]
    return (below + fraction[:, None] * (above - below)).T


def design_matrix(queries: pd.DataFrame, knots: int) -> np.ndarray:
    """Query-by-feature design within which ChromatinHD cluster deltas are fitted.

    Features are cell-type indicators, their interactions with every other
    varying categorical design column, and their interactions with a
    piecewise-linear basis of each continuous column. Columns that are fixed
    within cell types, such as zonation eligibility, add no features.
    """
    group = (
        queries["cell_type"].astype(str).to_numpy()
        if "cell_type" in queries
        else np.zeros(len(queries), dtype=str)
    )
    indicators = _one_hot(group)
    continuous = [column for column in queries.columns if column.startswith("continuous_")]
    blocks = [indicators]
    for column in queries.columns:
        if column in _BOOKKEEPING_COLUMNS or column in continuous or column == "cell_type":
            continue
        values = queries[column].astype(str)
        if values.groupby(group).nunique().max() > 1:
            blocks.append(_interact(indicators, _one_hot(values.to_numpy())))
    for column in continuous:
        blocks.append(_interact(indicators, _tent_basis(queries[column].to_numpy(float), knots)))
    return np.concatenate(blocks, axis=1)


def _one_hot(values: np.ndarray) -> np.ndarray:
    levels = list(dict.fromkeys(values.tolist()))
    return (values[:, None] == np.asarray(levels, dtype=values.dtype)[None, :]).astype(float)


def _interact(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left[:, :, None] * right[:, None, :]).reshape(len(left), -1)


def _tent_basis(values: np.ndarray, knots: int) -> np.ndarray:
    points = np.linspace(values.min(), values.max(), knots)
    spacing = max(points[1] - points[0], np.finfo(float).tiny)
    return np.clip(1.0 - np.abs(values[:, None] - points[None, :]) / spacing, 0.0, 1.0)
