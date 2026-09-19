from __future__ import annotations

import pandas as pd

from src.data.sportsdataio import _cached_feed_has_rows
from src.utils.io import write_df


def test_cached_feed_has_rows_rejects_empty_cache(tmp_path):
    path = tmp_path / "empty.parquet"
    write_df(pd.DataFrame(), path)

    assert _cached_feed_has_rows(path) is False


def test_cached_feed_has_rows_accepts_nonempty_cache(tmp_path):
    path = tmp_path / "nonempty.parquet"
    write_df(pd.DataFrame({"row": [1]}), path)

    assert _cached_feed_has_rows(path) is True
