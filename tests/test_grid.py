from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "candidate" / "src"))

from mymodel.grid import integrate_bins, nearest_represented_query  # noqa: E402


def test_bins_receive_their_share_of_model_bin_mass() -> None:
    # One region, two clusters, four 25 bp model bins.
    mass = np.array([[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]])
    region = np.zeros(3, dtype=int)
    start = np.array([0, 10, 50])
    end = np.array([25, 60, 100])

    rates = integrate_bins(mass, 25, region, start, end)

    # [10, 60) covers 15/25 of bin 0, all of bin 1, and 10/25 of bin 2.
    np.testing.assert_allclose(rates[0], [1.0, 0.6 + 2.0 + 1.2, 7.0])
    np.testing.assert_allclose(rates[1], [4.0, 2.4 + 3.0 + 0.8, 3.0])


def test_unrepresented_queries_borrow_only_from_the_same_categorical_design() -> None:
    queries = pd.DataFrame(
        {
            "query_index": range(5),
            "query_id": ["h_c0_z0", "h_c0_z1", "h_c1_z0", "h_c1_z1", "k_c0"],
            "cell_type": ["h", "h", "h", "h", "k"],
            "condition": [0, 0, 1, 1, 0],
            "continuous_zonation": [0.0, 0.1, 0.0, 0.1, 0.5],
            "weight": [0.5, 0.5, 0.5, 0.5, 1.0],
        }
    )
    represented = np.array([False, True, False, True, False])

    source = nearest_represented_query(queries, represented)

    assert source.tolist() == [1, 1, 3, 3, 4]
