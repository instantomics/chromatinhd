from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .method import Parameters, fit_rates

_INPUTS = ("bins", "cells", "fragments", "queries")


def fit(inputs, outputs):
    preset = _preset(inputs)
    parameters = Parameters.from_mapping(preset)
    coaccessibility = preset["task_id"] == "differential_coaccessibility"
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
                region_column="locus_id" if coaccessibility else "region_id",
            )
        if coaccessibility:
            from . import coaccessibility as task

            state = task.encode(rates, paths["bins"])
        else:
            from . import accessibility as task

            state = task.encode(rates)
        outputs.write_bytes(f"{catalog.dataset_handle}.npz", state)


def posterior(inputs, outputs):
    if _preset(inputs)["task_id"] == "differential_coaccessibility":
        from . import coaccessibility as task
    else:
        from . import accessibility as task
    for path in sorted(inputs.fit_state_path.glob("*.npz")):
        task.submit(outputs, path.stem, path)


def _preset(inputs) -> dict:
    return json.loads(inputs.parameterization_path.read_text())["reference"]["parameters"]
