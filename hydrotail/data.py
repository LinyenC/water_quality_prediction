from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from hydrotail.bundle_cache import downcast_numeric_frame
from hydrotail.dataset_bundle import load_dataset_bundle
from hydrotail.graph import build_similarity_outputs


LOGGER = logging.getLogger(__name__)


@dataclass
class SequenceSamples:
    """Container for lookback-window samples used by sequence models."""

    frame: pd.DataFrame
    sequence_values: np.ndarray
    sequence_masks: np.ndarray
    static_values: np.ndarray
    sequence_feature_cols: list[str]
    static_feature_cols: list[str]

    def subset(self, value: str, column: str = "split") -> "SequenceSamples":
        mask = self.frame[column].fillna("").to_numpy() == value
        return SequenceSamples(
            frame=self.frame.loc[mask].reset_index(drop=True),
            sequence_values=self.sequence_values[mask],
            sequence_masks=self.sequence_masks[mask],
            static_values=self.static_values[mask],
            sequence_feature_cols=self.sequence_feature_cols,
            static_feature_cols=self.static_feature_cols,
        )

    def __len__(self) -> int:
        return len(self.frame)


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _ensure_columns(frame: pd.DataFrame, columns: Iterable[str], frame_name: str) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in {frame_name}: {missing}")


def _first_non_null(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def _downcast_enabled(config: dict[str, object]) -> bool:
    data_cfg = config.get("data", {})
    bundle_cfg = data_cfg.get("dataset_bundle", {})
    return bool(data_cfg.get("downcast_float32", bundle_cfg.get("downcast_float32", True)))


def _include_target_history_features(config: dict[str, object]) -> bool:
    feature_cfg = config.get("features", {})
    return bool(feature_cfg.get("include_target_history_features", False))


def _maybe_downcast_frame(
    frame: pd.DataFrame,
    config: dict[str, object],
    exclude_cols: Iterable[str],
) -> pd.DataFrame:
    if not _downcast_enabled(config):
        return frame
    return downcast_numeric_frame(frame, exclude_cols=exclude_cols)


def _collapse_duplicate_station_days(frame: pd.DataFrame, station_col: str, date_col: str) -> pd.DataFrame:
    """Collapse duplicate rows within the same station-day before daily alignment."""

    numeric_cols = [col for col in frame.columns if col not in {station_col, date_col} and pd.api.types.is_numeric_dtype(frame[col])]
    other_cols = [col for col in frame.columns if col not in {station_col, date_col, *numeric_cols}]
    aggregations = {col: "mean" for col in numeric_cols}
    for col in other_cols:
        aggregations[col] = _first_non_null
    return frame.groupby([station_col, date_col], as_index=False).agg(aggregations)


def _apply_dynamic_date_filter(dynamic_df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    data_cfg = config.get("data", {})
    date_filter_cfg = data_cfg.get("date_filter", {})
    if not date_filter_cfg:
        return dynamic_df

    station_col = data_cfg["station_col"]
    date_col = data_cfg["date_col"]
    start_date = date_filter_cfg.get("start_date")
    end_date = date_filter_cfg.get("end_date")
    if not start_date and not end_date:
        return dynamic_df

    result = dynamic_df.copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="coerce")
    mask = result[date_col].notna()
    if start_date:
        start_ts = pd.Timestamp(start_date)
        mask &= result[date_col] >= start_ts
    else:
        start_ts = None
    if end_date:
        end_ts = pd.Timestamp(end_date)
        mask &= result[date_col] <= end_ts
    else:
        end_ts = None

    filtered = result.loc[mask].reset_index(drop=True)
    LOGGER.info(
        "Applied dynamic date filter: start_date=%s end_date=%s rows_before=%s rows_after=%s stations_after=%s",
        start_ts,
        end_ts,
        len(dynamic_df),
        len(filtered),
        filtered[station_col].nunique() if not filtered.empty else 0,
    )
    return filtered


def _align_to_daily_grid(frame: pd.DataFrame, station_col: str, date_col: str, freq: str = "D") -> pd.DataFrame:
    """Reindex each station to a daily grid so gaps become explicit NaN rows."""

    aligned_frames: list[pd.DataFrame] = []
    for station_id, station_frame in frame.groupby(station_col, sort=False):
        station_frame = station_frame.sort_values(date_col).copy()
        full_dates = pd.date_range(station_frame[date_col].min(), station_frame[date_col].max(), freq=freq)
        base = pd.DataFrame({station_col: station_id, date_col: full_dates})
        station_frame["is_original_observation"] = 1.0
        aligned = base.merge(station_frame, on=[station_col, date_col], how="left")
        aligned["is_original_observation"] = aligned["is_original_observation"].fillna(0.0)
        aligned["is_gap_filled"] = 1.0 - aligned["is_original_observation"]
        aligned_frames.append(aligned)
    return pd.concat(aligned_frames, axis=0, ignore_index=True)


def load_datasets(config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    paths_cfg = config.get("paths", {})
    bundle_keys = ("dataset_root", "attributes_root", "time_series_root", "wq_root")
    if any(paths_cfg.get(key) for key in bundle_keys):
        dynamic_df, static_df = load_dataset_bundle(config)
        return dynamic_df, static_df

    dynamic_df = read_table(config["paths"]["dynamic_data"])
    static_path = config["paths"].get("static_data")
    static_df = None
    if static_path and Path(static_path).exists():
        static_df = read_table(static_path)

    data_cfg = config["data"]
    station_col = data_cfg["station_col"]
    date_col = data_cfg["date_col"]
    target_columns = list(data_cfg["targets"].values())

    dynamic_required = [station_col, date_col] + target_columns
    _ensure_columns(dynamic_df, dynamic_required, "dynamic data")
    if static_df is not None:
        _ensure_columns(static_df, [station_col], "static data")
        static_df = _maybe_downcast_frame(static_df, config, exclude_cols=(station_col,))
    dynamic_df = _apply_dynamic_date_filter(dynamic_df, config)
    dynamic_df = _maybe_downcast_frame(dynamic_df, config, exclude_cols=(station_col, date_col))
    return dynamic_df, static_df


def _prepare_daily_frame(dynamic_df: pd.DataFrame, config: dict[str, object]) -> tuple[pd.DataFrame, list[str]]:
    """Prepare a station-day aligned frame that keeps missingness explicit."""

    data_cfg = config["data"]
    station_col = data_cfg["station_col"]
    date_col = data_cfg["date_col"]
    target_map = data_cfg["targets"]
    dynamic_feature_cols = [col for col in data_cfg.get("dynamic_features", []) if col in dynamic_df.columns]
    target_source_cols = [col for col in dict.fromkeys(target_map.values()) if col in dynamic_df.columns]
    if _include_target_history_features(config):
        raw_series_cols = list(dict.fromkeys(dynamic_feature_cols + target_source_cols))
    else:
        raw_series_cols = dynamic_feature_cols

    frame = dynamic_df.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = _collapse_duplicate_station_days(frame, station_col=station_col, date_col=date_col)
    frame = _align_to_daily_grid(frame, station_col=station_col, date_col=date_col, freq=str(config.get("data", {}).get("time_frequency", "D")))
    frame = frame.sort_values([station_col, date_col]).reset_index(drop=True)

    for col in raw_series_cols:
        frame[f"{col}_observed"] = frame[col].notna().astype(float)
    if raw_series_cols:
        frame["any_dynamic_observed"] = frame[raw_series_cols].notna().any(axis=1).astype(float)
    else:
        frame["any_dynamic_observed"] = 0.0
    frame = _maybe_downcast_frame(frame, config, exclude_cols=(station_col, date_col))
    return frame, raw_series_cols


def _build_target_frame(
    frame: pd.DataFrame,
    grouped,
    target_map: dict[str, str],
    horizon: int,
) -> tuple[pd.DataFrame, dict[str, str], pd.Series]:
    target_data: dict[str, pd.Series] = {}
    target_cols: dict[str, str] = {}
    for target_name, source_col in target_map.items():
        target_col = f"target_{target_name}_h{horizon}"
        target_data[target_col] = grouped[source_col].shift(-horizon)
        target_cols[target_name] = target_col
    target_frame = pd.DataFrame(target_data, index=frame.index)
    target_mask = target_frame.notna().any(axis=1)
    return target_frame, target_cols, target_mask


def _build_temporal_feature_frame(
    frame: pd.DataFrame,
    grouped,
    station_col: str,
    raw_series_cols: list[str],
    lags: list[int],
    windows: list[int],
    row_mask: pd.Series,
) -> pd.DataFrame:
    selected_index = frame.index[row_mask]
    if not raw_series_cols:
        return pd.DataFrame(index=selected_index)

    feature_blocks: list[pd.DataFrame] = []
    total = len(raw_series_cols)
    for idx, col in enumerate(raw_series_cols, start=1):
        LOGGER.info("Constructing temporal features for `%s` (%s/%s)", col, idx, total)
        grouped_col = grouped[col]
        column_features: dict[str, pd.Series] = {}
        for lag in lags:
            column_features[f"{col}_lag_{lag}"] = grouped_col.shift(lag)

        shifted = grouped_col.shift(1)
        shifted_grouped = shifted.groupby(frame[station_col], sort=False)
        for window in windows:
            rolling = shifted_grouped.rolling(window, min_periods=max(1, window // 2))
            column_features[f"{col}_roll_mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
            column_features[f"{col}_roll_std_{window}"] = rolling.std().reset_index(level=0, drop=True)

        feature_blocks.append(pd.DataFrame(column_features, index=frame.index).loc[row_mask])

    return pd.concat(feature_blocks, axis=1) if feature_blocks else pd.DataFrame(index=selected_index)


def _build_seasonal_feature_frame(frame: pd.DataFrame, date_col: str, row_mask: pd.Series) -> pd.DataFrame:
    day_of_year = frame[date_col].dt.dayofyear.astype(float)
    seasonal = pd.DataFrame(
        {
            "doy_sin": np.sin(2.0 * np.pi * day_of_year / 365.25),
            "doy_cos": np.cos(2.0 * np.pi * day_of_year / 365.25),
            "month": frame[date_col].dt.month.astype(float),
        },
        index=frame.index,
    )
    return seasonal.loc[row_mask]


def build_model_frame(
    dynamic_df: pd.DataFrame,
    static_df: pd.DataFrame | None,
    config: dict[str, object],
    horizon: int,
) -> tuple[pd.DataFrame, list[str], dict[str, str], pd.DataFrame]:
    data_cfg = config["data"]
    station_col = data_cfg["station_col"]
    date_col = data_cfg["date_col"]
    target_map = data_cfg["targets"]
    feature_cfg = config["features"]
    lags = list(feature_cfg.get("lags", [1, 3, 7]))
    windows = list(feature_cfg.get("rolling_windows", [7, 30]))

    frame, raw_series_cols = _prepare_daily_frame(dynamic_df, config)
    LOGGER.info(
        "Prepared daily frame for horizon=%s: rows=%s stations=%s raw_series=%s",
        horizon,
        len(frame),
        frame[station_col].nunique(),
        raw_series_cols,
    )
    grouped = frame.groupby(station_col, group_keys=False, sort=False)

    target_frame, target_cols, target_mask = _build_target_frame(frame, grouped, target_map, horizon)
    target_row_count = int(target_mask.sum())
    if target_row_count == 0:
        raise RuntimeError(f"No target rows remain for horizon={horizon}. Check the target sources or reduce the horizon.")
    LOGGER.info("Target-eligible rows for horizon=%s: %s", horizon, target_row_count)

    base_feature_cols: list[str] = [station_col, date_col]
    for col in raw_series_cols:
        if col in frame.columns:
            base_feature_cols.append(col)
        observed_col = f"{col}_observed"
        if observed_col in frame.columns:
            base_feature_cols.append(observed_col)
    for aux_col in ("is_original_observation", "is_gap_filled", "any_dynamic_observed"):
        if aux_col in frame.columns:
            base_feature_cols.append(aux_col)
    base_feature_cols = list(dict.fromkeys(base_feature_cols))

    assembled_frames: list[pd.DataFrame] = [frame.loc[target_mask, base_feature_cols].copy(), target_frame.loc[target_mask]]
    temporal_feature_frame = _build_temporal_feature_frame(
        frame=frame,
        grouped=grouped,
        station_col=station_col,
        raw_series_cols=raw_series_cols,
        lags=lags,
        windows=windows,
        row_mask=target_mask,
    )
    if not temporal_feature_frame.empty:
        assembled_frames.append(temporal_feature_frame)

    if feature_cfg.get("add_seasonal", True):
        assembled_frames.append(_build_seasonal_feature_frame(frame, date_col, target_mask))

    frame = pd.concat(assembled_frames, axis=1)
    frame = _maybe_downcast_frame(frame, config, exclude_cols=(station_col, date_col))
    frame = frame.copy()
    LOGGER.info("Assembled model-frame core for horizon=%s: rows=%s cols=%s", horizon, len(frame), len(frame.columns))

    if static_df is not None:
        static_cols = [station_col] + [col for col in data_cfg.get("static_features", []) if col in static_df.columns]
        static_merged = static_df[static_cols].drop_duplicates(subset=[station_col]).copy()

        if config.get("graph", {}).get("enabled", False):
            graph_features = config["graph"].get("feature_columns", [])
            graph_df, edge_df = build_similarity_outputs(
                static_merged,
                station_col=station_col,
                feature_cols=graph_features,
                k_neighbors=int(config["graph"].get("k_neighbors", 5)),
            )
            static_merged = static_merged.merge(graph_df, on=station_col, how="left")
        else:
            edge_df = pd.DataFrame(columns=["source_station", "neighbor_station", "distance"])

        frame = frame.merge(static_merged, on=station_col, how="left")
        LOGGER.info("Merged static features for horizon=%s: rows=%s cols=%s", horizon, len(frame), len(frame.columns))
    else:
        edge_df = pd.DataFrame(columns=["source_station", "neighbor_station", "distance"])

    feature_cols: list[str] = []
    protected = {station_col, date_col, *target_cols.values()}
    for col in frame.columns:
        if col in protected:
            continue
        if frame[col].dtype.kind not in {"f", "i", "u", "b"}:
            continue
        feature_cols.append(col)
    LOGGER.info("Numeric feature count before filtering for horizon=%s: %s", horizon, len(feature_cols))

    non_null_counts = frame[feature_cols].notna().sum(axis=1)
    min_ratio = float(feature_cfg.get("min_non_null_feature_ratio", 0.25))
    min_features = max(1, int(len(feature_cols) * min_ratio))
    frame = frame.loc[non_null_counts >= min_features].copy()
    LOGGER.info("Rows after non-null feature filtering for horizon=%s: %s", horizon, len(frame))

    target_availability = frame[list(target_cols.values())].notna()
    frame = frame.loc[target_availability.any(axis=1)].copy()
    LOGGER.info("Rows after target-availability filtering for horizon=%s: %s", horizon, len(frame))

    frame = _maybe_downcast_frame(frame, config, exclude_cols=(station_col, date_col))
    frame = frame.reset_index(drop=True)
    return frame, feature_cols, target_cols, edge_df


def build_sequence_samples(
    frame: pd.DataFrame,
    config: dict[str, object],
    target_cols: dict[str, str],
) -> SequenceSamples:
    """Create lookback-window samples for sequence encoders such as TCN or Transformer."""

    data_cfg = config["data"]
    station_col = data_cfg["station_col"]
    date_col = data_cfg["date_col"]
    dynamic_feature_cols = [col for col in data_cfg.get("dynamic_features", []) if col in frame.columns]
    sequence_cfg = config.get("sequence", {})
    lookback_window = int(sequence_cfg.get("lookback_window", 30))
    min_history_ratio = float(sequence_cfg.get("min_history_ratio", 0.15))

    sequence_feature_cols = list(dynamic_feature_cols)
    if _include_target_history_features(config):
        target_source_cols = [col for col in dict.fromkeys(data_cfg["targets"].values()) if col in frame.columns]
        sequence_feature_cols = list(dict.fromkeys(sequence_feature_cols + target_source_cols))
    if "is_gap_filled" in frame.columns:
        sequence_feature_cols.append("is_gap_filled")
    for seasonal_col in ("doy_sin", "doy_cos", "month"):
        if seasonal_col in frame.columns:
            sequence_feature_cols.append(seasonal_col)

    static_feature_cols = [col for col in data_cfg.get("static_features", []) if col in frame.columns]
    static_feature_cols.extend([col for col in frame.columns if col.startswith("graph_neighbor_")])
    static_feature_cols = list(dict.fromkeys(static_feature_cols))

    sample_rows: list[pd.Series] = []
    sequence_values: list[np.ndarray] = []
    sequence_masks: list[np.ndarray] = []
    static_values: list[np.ndarray] = []

    for _, station_frame in frame.groupby(station_col, sort=False):
        station_frame = station_frame.sort_values(date_col).reset_index(drop=True)
        for row_idx in range(lookback_window - 1, len(station_frame)):
            row = station_frame.iloc[row_idx]
            window = station_frame.iloc[row_idx - lookback_window + 1 : row_idx + 1]
            window_values = window[sequence_feature_cols].to_numpy(dtype=float)
            window_mask = (~np.isnan(window_values)).astype(float)

            if float(window_mask.mean()) < min_history_ratio:
                continue

            sample_rows.append(row.copy())
            sequence_values.append(window_values)
            sequence_masks.append(window_mask)
            if static_feature_cols:
                static_values.append(row[static_feature_cols].to_numpy(dtype=float))
            else:
                static_values.append(np.zeros(0, dtype=float))

    if not sample_rows:
        raise RuntimeError("No valid sequence samples were created. Lower `sequence.min_history_ratio` or reduce `lookback_window`.")

    sample_frame = pd.DataFrame(sample_rows).reset_index(drop=True)
    return SequenceSamples(
        frame=sample_frame,
        sequence_values=np.stack(sequence_values).astype(np.float32),
        sequence_masks=np.stack(sequence_masks).astype(np.float32),
        static_values=np.stack(static_values).astype(np.float32),
        sequence_feature_cols=sequence_feature_cols,
        static_feature_cols=static_feature_cols,
    )
