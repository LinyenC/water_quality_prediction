from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from hydrotail.dataset_bundle import load_dataset_bundle
from hydrotail.graph import build_similarity_outputs


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


def _collapse_duplicate_station_days(frame: pd.DataFrame, station_col: str, date_col: str) -> pd.DataFrame:
    """Collapse duplicate rows within the same station-day before daily alignment."""

    numeric_cols = [col for col in frame.columns if col not in {station_col, date_col} and pd.api.types.is_numeric_dtype(frame[col])]
    other_cols = [col for col in frame.columns if col not in {station_col, date_col, *numeric_cols}]
    aggregations = {col: "mean" for col in numeric_cols}
    for col in other_cols:
        aggregations[col] = _first_non_null
    return frame.groupby([station_col, date_col], as_index=False).agg(aggregations)


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
    if config.get("paths", {}).get("dataset_root"):
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
    return dynamic_df, static_df


def _prepare_daily_frame(dynamic_df: pd.DataFrame, config: dict[str, object]) -> tuple[pd.DataFrame, list[str]]:
    """Prepare a station-day aligned frame that keeps missingness explicit."""

    data_cfg = config["data"]
    station_col = data_cfg["station_col"]
    date_col = data_cfg["date_col"]
    target_map = data_cfg["targets"]
    dynamic_feature_cols = [col for col in data_cfg.get("dynamic_features", []) if col in dynamic_df.columns]
    raw_series_cols = list(dict.fromkeys(dynamic_feature_cols + list(target_map.values())))

    frame = dynamic_df.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = _collapse_duplicate_station_days(frame, station_col=station_col, date_col=date_col)
    frame = _align_to_daily_grid(frame, station_col=station_col, date_col=date_col, freq=str(config.get("data", {}).get("time_frequency", "D")))
    frame = frame.sort_values([station_col, date_col]).reset_index(drop=True)

    # Keep explicit missingness flags so models can distinguish "not observed" from true low values.
    for col in raw_series_cols:
        frame[f"{col}_observed"] = frame[col].notna().astype(float)
    frame["any_dynamic_observed"] = frame[raw_series_cols].notna().any(axis=1).astype(float)
    return frame, raw_series_cols


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
    grouped = frame.groupby(station_col, group_keys=False)

    # Lag and rolling features are computed on the daily-aligned grid so gaps stay gaps.
    for col in raw_series_cols:
        for lag in lags:
            frame[f"{col}_lag_{lag}"] = grouped[col].shift(lag)
        shifted = grouped[col].shift(1)
        for window in windows:
            rolling = shifted.groupby(frame[station_col]).rolling(window, min_periods=max(1, window // 2))
            frame[f"{col}_roll_mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
            frame[f"{col}_roll_std_{window}"] = rolling.std().reset_index(level=0, drop=True)

    if feature_cfg.get("add_seasonal", True):
        day_of_year = frame[date_col].dt.dayofyear.astype(float)
        frame["doy_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
        frame["doy_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25)
        frame["month"] = frame[date_col].dt.month.astype(float)

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
    else:
        edge_df = pd.DataFrame(columns=["source_station", "neighbor_station", "distance"])

    target_cols: dict[str, str] = {}
    for target_name, source_col in target_map.items():
        target_col = f"target_{target_name}_h{horizon}"
        # Target shifting also happens on the aligned daily grid so future labels are only available on real observed days.
        frame[target_col] = grouped[source_col].shift(-horizon)
        target_cols[target_name] = target_col

    feature_cols: list[str] = []
    protected = {station_col, date_col, *target_cols.values()}
    for col in frame.columns:
        if col in protected:
            continue
        if frame[col].dtype.kind not in {"f", "i", "u", "b"}:
            continue
        feature_cols.append(col)

    non_null_counts = frame[feature_cols].notna().sum(axis=1)
    min_ratio = float(feature_cfg.get("min_non_null_feature_ratio", 0.25))
    min_features = max(1, int(len(feature_cols) * min_ratio))
    frame = frame.loc[non_null_counts >= min_features].copy()

    # Multi-task rows can keep partial labels; losses and metrics mask target-specific NaNs downstream.
    target_availability = frame[list(target_cols.values())].notna()
    frame = frame.loc[target_availability.any(axis=1)].copy()

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
    target_source_cols = list(data_cfg["targets"].values())
    dynamic_feature_cols = [col for col in data_cfg.get("dynamic_features", []) if col in frame.columns]
    sequence_cfg = config.get("sequence", {})
    lookback_window = int(sequence_cfg.get("lookback_window", 30))
    min_history_ratio = float(sequence_cfg.get("min_history_ratio", 0.15))

    sequence_feature_cols = list(dict.fromkeys(dynamic_feature_cols + target_source_cols))
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

            # Skip windows that are almost entirely missing; this keeps sequence training stable on sparse stations.
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
