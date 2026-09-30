from __future__ import annotations

from differential_accessibility.rate_codec import (
    RatePrediction,
    encode_rate_prediction,
    read_rate_prediction,
)


def encode(rates) -> bytes:
    return encode_rate_prediction(RatePrediction(rates.query_ids, rates.bin_ids, rates.rates))


def submit(outputs, dataset_handle, path) -> None:
    outputs.submit(dataset_handle, read_rate_prediction(path))
