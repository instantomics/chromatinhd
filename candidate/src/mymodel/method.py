"""ChromatinHD-diff fitted to one differential-accessibility dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromatinhd as chd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from chromatinhd.models.diff.model.binary import Model

from .grid import integrate_bins, nearest_represented_query, regions_from_bins, window


@dataclass(frozen=True)
class Parameters:
    n_epochs: int
    lr: float
    n_cells_step: int
    n_regions_step: int
    random_seed: int

    @classmethod
    def from_mapping(cls, value: dict) -> Parameters:
        return cls(
            n_epochs=int(value["n_epochs"]),
            lr=float(value["lr"]),
            n_cells_step=int(value["n_cells_step"]),
            n_regions_step=int(value["n_regions_step"]),
            random_seed=int(value["random_seed"]),
        )


@dataclass(frozen=True)
class Rates:
    query_ids: tuple[str, ...]
    bin_ids: tuple[str, ...]
    rates: np.ndarray


def fit_rates(
    paths: dict[str, Path],
    parameters: Parameters,
    workdir: Path,
    *,
    device: str,
    threads: int,
) -> Rates:
    bins = pq.read_table(
        paths["bins"], columns=["bin_id", "region_id", "chrom", "start", "end"]
    ).to_pandas()
    cells = pq.read_table(paths["cells"], columns=["cell_id", "query_id", "exposure"]).to_pandas()
    queries = pq.read_table(paths["queries"]).to_pandas()
    insertions = pq.read_table(
        paths["fragments"], columns=["cell_id", "bin_id", "start"]
    ).to_pandas()
    query_ids = tuple(str(value) for value in queries["query_id"])

    regions = regions_from_bins(bins)
    window_width, binwidths = window(int(regions["width"].max()))

    # Cells without positive exposure carry no rate information.
    cells = cells.loc[cells["exposure"] > 0.0].reset_index(drop=True)
    cell_index = pd.Index(cells["cell_id"].astype(str), name="cell")
    region_index = pd.Index(regions.index, name="region")
    bin_region = bins.set_index("bin_id")["region_id"].astype(str)

    region_ix = region_index.get_indexer(bin_region.loc[insertions["bin_id"]].to_numpy())
    cell_ix = cell_index.get_indexer(insertions["cell_id"].astype(str))
    keep = cell_ix >= 0
    region_ix, cell_ix = region_ix[keep], cell_ix[keep]
    coordinates = insertions["start"].to_numpy()[keep] - regions["start"].to_numpy()[region_ix]

    order = np.lexsort((coordinates, cell_ix, region_ix))
    fragments = chd.data.Fragments.create(
        path=workdir / "fragments",
        # Each task row is one Tn5 insertion: a single cut per ChromatinHD fragment.
        coordinates=coordinates[order, None].astype(np.int32),
        mapping=np.stack([cell_ix[order], region_ix[order]], axis=1).astype(np.int32),
        regions=chd.data.Regions.create(
            path=workdir / "regions",
            coordinates=regions[["chrom", "start", "end"]],
            window=np.array([0, window_width]),
        ),
        var=pd.DataFrame(index=region_index),
        obs=pd.DataFrame(index=cell_index),
    )
    fragments.create_regionxcell_indptr()

    # One ChromatinHD cluster per supplied design query, assigned to fit cells by
    # the task's nearest-query column.
    labels = pd.Series(
        pd.Categorical(cells["query_id"].astype(str), categories=query_ids),
        index=cell_index,
    )
    clustering = chd.data.Clustering.from_labels(labels, path=workdir / "clustering")

    torch.manual_seed(parameters.random_seed)
    np.random.seed(parameters.random_seed)
    torch.set_num_threads(threads)
    all_cells = np.arange(fragments.n_cells)
    model = Model.create(
        fragments=fragments,
        clustering=clustering,
        fold={"cells_train": all_cells, "cells_validation": all_cells},
        path=workdir / "model",
        encoder_params={"binwidths": binwidths},
    )
    _use_declared_exposure(model, fragments.counts, cells["exposure"].to_numpy())
    model.train_model(
        device=device,
        n_epochs=parameters.n_epochs,
        lr=parameters.lr,
        n_cells_step=parameters.n_cells_step,
        n_regions_step=parameters.n_regions_step,
        do_validation=False,
        n_workers_train=max(2, threads),
        n_workers_validation=1,
    )

    model = model.to("cpu").eval()
    with torch.no_grad():
        all_regions = torch.arange(len(region_index))
        log_density, _ = model.encoder._calculate_w(all_regions)
        log_region = model.overall_bias[:, None] + model.overall_delta(all_regions)
    finest = int(model.encoder.binwidths[-1])
    # Expected insertions per unit exposure in each finest model bin.
    mass = finest * np.exp(log_density.double().numpy() + log_region.double().numpy()[..., None])
    bin_region_ix = region_index.get_indexer(bins["region_id"].astype(str))
    offset = regions["start"].to_numpy()[bin_region_ix]
    cluster_rates = integrate_bins(
        mass,
        finest,
        bin_region_ix,
        bins["start"].to_numpy() - offset,
        bins["end"].to_numpy() - offset,
    )

    represented = np.isin(np.arange(len(query_ids)), clustering.indices)
    source = nearest_represented_query(queries, represented)
    return Rates(query_ids, tuple(bins["bin_id"].astype(str)), cluster_rates[source])


def _use_declared_exposure(model: Model, counts: np.ndarray, exposure: np.ndarray) -> None:
    """Replace upstream fragment-count library sizes by the task's declared exposure.

    Upstream fixes the region bias at pooled counts per library-size unit; the
    same formula in exposure units makes fitted rates per unit declared exposure.
    """
    model.libsize = torch.tensor(exposure, dtype=torch.float32)
    overall = counts.sum(0) / exposure.sum()
    min_overall = 1e-5
    model.overall_bias = torch.log(
        torch.tensor(min_overall + (1 - min_overall) * overall, dtype=torch.float32)
    )
