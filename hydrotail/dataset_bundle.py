from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.io.matlab import MatlabOpaque


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


def _first_non_null(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def _normalize_station_id(value: object, width: int) -> str:
    text = str(value).strip()
    digits = "".join(char for char in text if char.isdigit())
    if not digits:
        return text
    return digits.zfill(width) if width > 0 else digits


def _pick_existing_directory(root: Path, candidates: Iterable[str], label: str) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.exists() and path.is_dir():
            return path
    raise FileNotFoundError(f"Could not find {label} directory under {root}. Tried: {list(candidates)}")


def _pick_optional_directory(root: Path, candidates: Iterable[str]) -> Path | None:
    for candidate in candidates:
        path = root / candidate
        if path.exists() and path.is_dir():
            return path
    return None


def _pick_first_file(directory: Path, patterns: Iterable[str], label: str) -> Path:
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find {label} in {directory}. Patterns tried: {list(patterns)}")


def _infer_date_column(frame: pd.DataFrame, configured: str | None = None) -> str:
    if configured and configured in frame.columns:
        return configured
    for candidate in ("date", "Date", "datetime", "Datetime", "time", "Time", "Unnamed: 0"):
        if candidate in frame.columns:
            return candidate
    return frame.columns[0]


def _aggregate_station_day(frame: pd.DataFrame, station_col: str, date_col: str) -> pd.DataFrame:
    value_cols = [col for col in frame.columns if col not in {station_col, date_col}]
    aggregations = {col: _first_non_null for col in value_cols}
    return frame.groupby([station_col, date_col], as_index=False).agg(aggregations)


def _load_attributes_bundle(dataset_root: Path, config: dict[str, object]) -> pd.DataFrame:
    data_cfg = config["data"]
    bundle_cfg = data_cfg.get("dataset_bundle", {})
    station_col = data_cfg["station_col"]
    station_width = int(data_cfg.get("station_id_width", 8))

    attr_dir = _pick_existing_directory(
        dataset_root,
        bundle_cfg.get("attributes_dir_candidates", ["attritubes", "attributes"]),
        label="attributes",
    )
    attr_file = _pick_first_file(attr_dir, bundle_cfg.get("attributes_patterns", ["*.xlsx", "*.csv"]), label="attributes file")
    frame = read_table(attr_file)

    source_station_col = str(bundle_cfg.get("attributes_station_col", "gauge_id"))
    rename_map = dict(bundle_cfg.get("attributes_rename_map", {}))
    frame = frame.rename(columns=rename_map)
    if station_col not in frame.columns:
        if source_station_col not in frame.columns:
            raise ValueError(f"Attributes file {attr_file} does not contain station column `{source_station_col}`.")
        frame = frame.rename(columns={source_station_col: station_col})

    frame[station_col] = frame[station_col].map(lambda value: _normalize_station_id(value, station_width))
    keep_cols = [station_col] + [col for col in data_cfg.get("static_features", []) if col in frame.columns]
    if station_col not in keep_cols:
        keep_cols.insert(0, station_col)
    return frame[keep_cols].drop_duplicates(subset=[station_col]).reset_index(drop=True)


def _read_station_series_file(
    path: Path,
    station_id: str,
    station_col: str,
    date_col: str,
    rename_map: dict[str, str],
    configured_date_col: str | None = None,
) -> pd.DataFrame:
    frame = read_table(path)
    raw_date_col = _infer_date_column(frame, configured=configured_date_col)
    frame = frame.rename(columns={raw_date_col: date_col, **rename_map})
    keep_cols = [date_col] + [col for col in rename_map.values() if col in frame.columns]
    frame = frame[keep_cols].copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col])
    frame[station_col] = station_id
    ordered_cols = [station_col, date_col] + [col for col in frame.columns if col not in {station_col, date_col}]
    return frame[ordered_cols]


def _load_time_series_bundle(dataset_root: Path, config: dict[str, object]) -> pd.DataFrame:
    data_cfg = config["data"]
    bundle_cfg = data_cfg.get("dataset_bundle", {})
    station_col = data_cfg["station_col"]
    date_col = data_cfg["date_col"]
    station_width = int(data_cfg.get("station_id_width", 8))

    ts_dir = _pick_existing_directory(
        dataset_root,
        bundle_cfg.get("time_series_dir_candidates", ["time series", "time_series"]),
        label="time series",
    )
    source_cfgs = list(bundle_cfg.get("time_series_sources", []))
    if not source_cfgs:
        raise ValueError("`data.dataset_bundle.time_series_sources` must be configured when using dataset_root.")

    frames: list[pd.DataFrame] = []
    for source_cfg in source_cfgs:
        source_dir = ts_dir / str(source_cfg["folder"])
        if not source_dir.exists():
            continue
        pattern = str(source_cfg.get("pattern", "*.csv"))
        rename_map = dict(source_cfg.get("rename_map", {}))
        configured_date_col = source_cfg.get("date_column")
        for file_path in sorted(source_dir.glob(pattern)):
            station_id = _normalize_station_id(file_path.stem, station_width)
            part = _read_station_series_file(
                file_path,
                station_id=station_id,
                station_col=station_col,
                date_col=date_col,
                rename_map=rename_map,
                configured_date_col=configured_date_col,
            )
            frames.append(part)

    if not frames:
        raise RuntimeError(f"No time-series files were discovered under {ts_dir}.")

    merged = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    merged = _aggregate_station_day(merged, station_col=station_col, date_col=date_col)

    if {"tmin", "tmax"}.issubset(merged.columns):
        merged["air_temp"] = merged[["tmin", "tmax"]].mean(axis=1)
    return merged.sort_values([station_col, date_col]).reset_index(drop=True)


def _read_target_tabular_file(
    path: Path,
    station_id: str,
    station_col: str,
    date_col: str,
    target_col: str,
    source_value_col: str | None = None,
    configured_date_col: str | None = None,
) -> pd.DataFrame:
    frame = read_table(path)
    raw_date_col = _infer_date_column(frame, configured=configured_date_col)
    frame = frame.rename(columns={raw_date_col: date_col})
    numeric_candidates = [
        col
        for col in frame.columns
        if col != date_col and pd.api.types.is_numeric_dtype(frame[col])
    ]
    if source_value_col and source_value_col in frame.columns:
        value_col = source_value_col
    elif target_col in frame.columns:
        value_col = target_col
    elif len(numeric_candidates) == 1:
        value_col = numeric_candidates[0]
    else:
        raise ValueError(
            f"Could not infer target column for {path}. Provide `source_value_col` or a file with a single numeric column."
        )

    result = frame[[date_col, value_col]].copy().rename(columns={value_col: target_col})
    result[date_col] = pd.to_datetime(result[date_col], errors="coerce")
    result = result.dropna(subset=[date_col])
    result[station_col] = station_id
    return result[[station_col, date_col, target_col]]


def _read_target_mat_file(path: Path, target_col: str) -> pd.DataFrame:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    objects = [value for key, value in mat.items() if not key.startswith("__")]
    if any(isinstance(value, MatlabOpaque) for value in objects):
        raise RuntimeError(
            "MATLAB timetable/table objects are stored as opaque MCOS data in this file. "
            f"Please export {path.name} to csv first, or provide a WQ cache file such as `dataset/WQ/wq_observations.csv`."
        )
    raise RuntimeError(
        f"Unsupported MAT layout for {path.name}. Only tabular exports are supported directly right now for `{target_col}`."
    )


def _resolve_target_folder(target_cfg: dict[str, object], wq_dir: Path) -> Path | None:
    folder_candidates = list(target_cfg.get("folder_candidates", []))
    if not folder_candidates and target_cfg.get("folder"):
        folder_candidates = [str(target_cfg["folder"])]
    if not folder_candidates:
        return None
    return _pick_optional_directory(wq_dir, folder_candidates)


def _load_wq_bundle(dataset_root: Path, config: dict[str, object]) -> pd.DataFrame:
    data_cfg = config["data"]
    bundle_cfg = data_cfg.get("dataset_bundle", {})
    station_col = data_cfg["station_col"]
    date_col = data_cfg["date_col"]
    station_width = int(data_cfg.get("station_id_width", 8))

    wq_dir = _pick_existing_directory(dataset_root, bundle_cfg.get("wq_dir_candidates", ["WQ", "wq"]), label="WQ")
    cache_name = str(bundle_cfg.get("wq_cache_name", "wq_observations.csv"))
    cache_path = wq_dir / cache_name
    if cache_path.exists():
        cache_frame = read_table(cache_path)
        required = [station_col, date_col] + list(data_cfg["targets"].values())
        missing = [col for col in required if col not in cache_frame.columns]
        if missing:
            raise ValueError(f"WQ cache file {cache_path} is missing columns: {missing}")
        cache_frame[station_col] = cache_frame[station_col].map(lambda value: _normalize_station_id(value, station_width))
        cache_frame[date_col] = pd.to_datetime(cache_frame[date_col], errors="coerce")
        return cache_frame.dropna(subset=[date_col]).reset_index(drop=True)

    target_frames: list[pd.DataFrame] = []
    opaque_examples: list[str] = []
    parser_errors: list[str] = []
    for target_name, target_col in data_cfg["targets"].items():
        target_cfg = dict(bundle_cfg.get("wq_sources", {}).get(target_name, {}))
        folder_path = _resolve_target_folder(target_cfg, wq_dir)
        if folder_path is None:
            continue
        pattern = str(target_cfg.get("pattern", "*"))
        source_value_col = target_cfg.get("source_value_col")
        configured_date_col = target_cfg.get("date_column")
        for file_path in sorted(folder_path.glob(pattern)):
            if file_path.suffix.lower() == ".mat":
                opaque_examples.append(file_path.name)
                continue
            station_id = _normalize_station_id(file_path.stem, station_width)
            try:
                part = _read_target_tabular_file(
                    file_path,
                    station_id=station_id,
                    station_col=station_col,
                    date_col=date_col,
                    target_col=target_col,
                    source_value_col=source_value_col,
                    configured_date_col=configured_date_col,
                )
                target_frames.append(part)
            except RuntimeError as exc:
                parser_errors.append(str(exc))

    if not target_frames:
        if opaque_examples:
            examples = ", ".join(opaque_examples[:5])
            extra = "" if len(opaque_examples) <= 5 else f", ... (+{len(opaque_examples) - 5} more)"
            raise RuntimeError(
                "WQ files are currently MATLAB timetable/table objects stored as opaque MCOS data, "
                "which scipy cannot decode directly in this environment. "
                f"Example files: {examples}{extra}. "
                "Please export them to csv first, or provide `dataset/WQ/wq_observations.csv`."
            )
        if parser_errors:
            raise RuntimeError("\n".join(sorted(set(parser_errors))))
        raise RuntimeError(f"No water-quality target files were parsed under {wq_dir}.")

    merged = pd.concat(target_frames, axis=0, ignore_index=True, sort=False)
    merged = _aggregate_station_day(merged, station_col=station_col, date_col=date_col)
    return merged.sort_values([station_col, date_col]).reset_index(drop=True)


def load_dataset_bundle(config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_root = Path(config["paths"]["dataset_root"]).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    static_df = _load_attributes_bundle(dataset_root, config)
    dynamic_covariates = _load_time_series_bundle(dataset_root, config)
    wq_targets = _load_wq_bundle(dataset_root, config)

    station_col = config["data"]["station_col"]
    date_col = config["data"]["date_col"]
    dynamic_df = dynamic_covariates.merge(wq_targets, on=[station_col, date_col], how="outer")
    dynamic_df = dynamic_df.sort_values([station_col, date_col]).reset_index(drop=True)
    return dynamic_df, static_df
