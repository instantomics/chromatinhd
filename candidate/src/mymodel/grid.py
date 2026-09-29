"""Mappings between the task's bin and query grids and ChromatinHD's clusters."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Upstream Shared-encoder resolutions. Levels that do not tile a dataset's region
# window are dropped; the finest level fixes the positional resolution.
UPSTREAM_BINWIDTHS = (5000, 1000, 500, 200, 100, 50, 25)
_BOOKKEEPING_COLUMNS = frozenset({"query_index", "query_id", "weight"})


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


def nearest_represented_query(queries: pd.DataFrame, represented: np.ndarray) -> np.ndarray:
    """Map each query to itself, or to the nearest represented continuous grid point.

    Queries without fit cells borrow the fitted cluster with the same categorical
    design at the closest continuous coordinates. Without such a cluster, the
    query keeps ChromatinHD's unobserved-cluster prediction: the shared baseline.
    """
    continuous = [column for column in queries.columns if column.startswith("continuous_")]
    design = [
        column
        for column in queries.columns
        if column not in _BOOKKEEPING_COLUMNS and column not in continuous
    ]
    keys = (
        queries[design].astype(str).agg("\x1f".join, axis=1).to_numpy()
        if design
        else np.zeros(len(queries))
    )
    values = queries[continuous].to_numpy(dtype=np.float64)
    source = np.arange(len(queries))
    for index in np.flatnonzero(~represented):
        candidates = represented & (keys == keys[index])
        if not candidates.any():
            continue
        choices = np.flatnonzero(candidates)
        distance = np.abs(values[choices] - values[index]).sum(axis=1)
        source[index] = choices[np.argmin(distance)]
    return source
