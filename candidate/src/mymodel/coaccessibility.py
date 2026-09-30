"""Co-accessibility implied by a fitted ChromatinHD-diff model."""

from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
from differential_coaccessibility.domain import evaluation_domain
from differential_coaccessibility.ratio_codec import (
    RatioPrediction,
    encode_ratio_prediction,
    read_ratio_prediction,
)


def encode(rates, bins_path) -> bytes:
    # Each query is one ChromatinHD cluster, whose cells are independent Poisson
    # draws: E[Y_a Y_b] = e^2 lambda_a lambda_b for distinct bins, so every ratio is one.
    domain = evaluation_domain(pq.read_table(bins_path).to_pylist())
    ratios = np.ones((len(rates.query_ids), len(domain.pair_bins)), dtype=np.float32)
    return encode_ratio_prediction(RatioPrediction(rates.query_ids, domain.pair_order, ratios))


def submit(outputs, dataset_handle, path) -> None:
    outputs.submit(dataset_handle, read_ratio_prediction(path))
