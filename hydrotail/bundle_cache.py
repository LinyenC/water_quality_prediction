from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PARQUET_ENGINE_MESSAGE = (
    "Parquet support requires an installed engine such as `pyarrow`. "
    "Install it in the active environment before enabling bundle parquet caches."
)


def build_cache_namespace(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(serialized).hexdigest()[:16]


def downcast_numeric_frame(frame: pd.DataFrame, exclude_cols: Iterable[str] = ()) -> pd.DataFrame:
    excluded = set(exclude_cols)
    result = frame.copy()
    for column in result.columns:
        if column in excluded:
            continue
        series = result[column]
        if not (
            pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_integer_dtype(series)
            or pd.api.types.is_float_dtype(series)
        ):
            continue
        result[column] = pd.to_numeric(series, errors="coerce").astype(np.float32)
    return result


def load_parquet_frame(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError(PARQUET_ENGINE_MESSAGE) from exc


def save_parquet_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False)
    except ImportError as exc:
        raise RuntimeError(PARQUET_ENGINE_MESSAGE) from exc


def write_cache_metadata(cache_dir: Path, payload: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
