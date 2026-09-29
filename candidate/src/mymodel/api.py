from __future__ import annotations

import json
import tempfile
from pathlib import Path

from differential_accessibility.rate_codec import (
    RatePrediction,
    encode_rate_prediction,
    read_rate_prediction,
)

from .method import Parameters, fit_rates

_INPUTS = ("bins", "cells", "fragments", "queries")


def fit(inputs, outputs):
    parameterization = json.loads(inputs.parameterization_path.read_text())
    parameters = Parameters.from_mapping(parameterization["reference"]["parameters"])
    for catalog in inputs.data.list_datasets():
        dataset = inputs.data.open_dataset(catalog)
        paths = {name: Path(getattr(dataset, name).path) for name in _INPUTS}
        with tempfile.TemporaryDirectory(prefix="chromatinhd-") as workdir:
            rates = fit_rates(
                paths,
                parameters,
                Path(workdir),
                device=inputs.context.device,
                threads=inputs.context.resources.cpus,
            )
        outputs.write_bytes(
            f"{catalog.dataset_handle}.npz",
            encode_rate_prediction(RatePrediction(rates.query_ids, rates.bin_ids, rates.rates)),
        )


def posterior(inputs, outputs):
    for path in sorted(inputs.fit_state_path.glob("*.npz")):
        outputs.submit(path.stem, read_rate_prediction(path))
