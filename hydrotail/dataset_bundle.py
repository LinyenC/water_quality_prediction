from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from hydrotail.bundle_cache import (
    build_cache_namespace,
    downcast_numeric_frame,
    load_parquet_frame,
    save_parquet_frame,
    write_cache_metadata,
)


LOGGER = logging.getLogger(__name__)


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


def _resolve_bundle_directory(
    *,
    paths_cfg: dict[str, object],
    dataset_root: Path | None,
    override_key: str,
    fallback_candidates: Iterable[str],
    label: str,
) -> Path:
    override_value = paths_cfg.get(override_key)
    if override_value:
        path = Path(str(override_value)).resolve()
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Configured {label} directory not found: {path}")
        return path
    if dataset_root is None:
        raise FileNotFoundError(
            f"No `{override_key}` was provided and dataset_root is unavailable, so the {label} directory cannot be located."
        )
    return _pick_existing_directory(dataset_root, fallback_candidates, label=label)


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


def _select_station_series_files(
    directory: Path,
    pattern: str,
    station_width: int,
    station_subset: set[str] | None,
) -> list[tuple[Path, str]]:
    selected: list[tuple[Path, str]] = []
    for file_path in sorted(directory.glob(pattern)):
        station_id = _normalize_station_id(file_path.stem, station_width)
        if station_subset is not None and station_id not in station_subset:
            continue
        selected.append((file_path, station_id))
    return selected


def _bundle_cache_enabled(config: dict[str, object]) -> bool:
    bundle_cfg = config["data"].get("dataset_bundle", {})
    return bool(bundle_cfg.get("use_parquet_cache", True))


def _bundle_refresh_cache(config: dict[str, object]) -> bool:
    bundle_cfg = config["data"].get("dataset_bundle", {})
    return bool(bundle_cfg.get("refresh_cache", False))


def _downcast_enabled(config: dict[str, object]) -> bool:
    data_cfg = config.get("data", {})
    bundle_cfg = data_cfg.get("dataset_bundle", {})
    return bool(data_cfg.get("downcast_float32", bundle_cfg.get("downcast_float32", True)))


def _drop_all_null_target_rows_enabled(config: dict[str, object]) -> bool:
    bundle_cfg = config.get("data", {}).get("dataset_bundle", {})
    return bool(bundle_cfg.get("drop_all_null_target_rows", True))


def _build_cache_payload(dataset_root: Path | None, config: dict[str, object]) -> dict[str, object]:
    data_cfg = config["data"]
    bundle_cfg = data_cfg.get("dataset_bundle", {})
    paths_cfg = config.get("paths", {})
    return {
        "schema_version": 5,
        "dataset_root": None if dataset_root is None else str(dataset_root),
        "attributes_root": paths_cfg.get("attributes_root"),
        "time_series_root": paths_cfg.get("time_series_root"),
        "wq_root": paths_cfg.get("wq_root"),
        "attributes_patterns": list(bundle_cfg.get("attributes_patterns", [])),
        "attributes_station_col": bundle_cfg.get("attributes_station_col", "gauge_id"),
        "attributes_rename_map": dict(bundle_cfg.get("attributes_rename_map", {})),
        "time_series_sources": list(bundle_cfg.get("time_series_sources", [])),
        "wq_sources": dict(bundle_cfg.get("wq_sources", {})),
        "station_limit": bundle_cfg.get("station_limit"),
        "station_selection_strategy": bundle_cfg.get("station_selection_strategy", "coverage_aware"),
        "coverage_min_stations_per_target": bundle_cfg.get("coverage_min_stations_per_target"),
        "downcast_float32": _downcast_enabled(config),
        "drop_all_null_target_rows": _drop_all_null_target_rows_enabled(config),
    }


def _resolve_cache_dir(dataset_root: Path | None, config: dict[str, object]) -> Path | None:
    if not _bundle_cache_enabled(config):
        return None

    paths_cfg = config.get("paths", {})
    cache_root_value = paths_cfg.get("cache_dir")
    if cache_root_value:
        cache_root = Path(str(cache_root_value)).resolve()
    elif dataset_root is not None:
        cache_root = dataset_root / "_hydrotail_cache"
    else:
        component_roots = [
            Path(str(paths_cfg[key])).resolve()
            for key in ("attributes_root", "time_series_root", "wq_root")
            if paths_cfg.get(key)
        ]
        if not component_roots:
            return None
        cache_root = component_roots[0].parent / "_hydrotail_cache"

    payload = _build_cache_payload(dataset_root, config)
    cache_dir = cache_root / build_cache_namespace(payload)
    write_cache_metadata(cache_dir, payload)
    return cache_dir


def _component_cache_path(cache_dir: Path | None, name: str) -> Path | None:
    if cache_dir is None:
        return None
    return cache_dir / f"{name}.parquet"


def _load_or_build_cached_frame(
    *,
    label: str,
    cache_path: Path | None,
    builder: Callable[[], pd.DataFrame],
    config: dict[str, object],
    exclude_cols: Iterable[str],
) -> pd.DataFrame:
    if cache_path is not None and cache_path.exists() and not _bundle_refresh_cache(config):
        frame = load_parquet_frame(cache_path)
        LOGGER.info("Loaded %s parquet cache: rows=%s file=%s", label, len(frame), cache_path)
    else:
        frame = builder()
        if _downcast_enabled(config):
            frame = downcast_numeric_frame(frame, exclude_cols=exclude_cols)
        if cache_path is not None:
            save_parquet_frame(frame, cache_path)
            LOGGER.info("Wrote %s parquet cache: rows=%s file=%s", label, len(frame), cache_path)

    if _downcast_enabled(config):
        frame = downcast_numeric_frame(frame, exclude_cols=exclude_cols)
    return frame


def _resolve_target_folder(target_cfg: dict[str, object], wq_dir: Path) -> Path | None:
    folder_candidates = list(target_cfg.get("folder_candidates", []))
    if not folder_candidates and target_cfg.get("folder"):
        folder_candidates = [str(target_cfg["folder"])]
    if not folder_candidates:
        return None
    return _pick_optional_directory(wq_dir, folder_candidates)


def _empty_target_counts(target_names: Iterable[str]) -> dict[str, int]:
    return {target_name: 0 for target_name in target_names}


def _build_station_target_counts_from_cache(
    cache_frame: pd.DataFrame,
    data_cfg: dict[str, object],
    station_width: int,
) -> dict[str, dict[str, int]]:
    station_col = data_cfg["station_col"]
    target_map = dict(data_cfg["targets"])
    counts: dict[str, dict[str, int]] = {}
    normalized_station_ids = cache_frame[station_col].map(lambda value: _normalize_station_id(value, station_width))
    for target_name, target_col in target_map.items():
        observed = cache_frame[target_col].notna()
        grouped = normalized_station_ids[observed].value_counts().to_dict()
        for station_id, count in grouped.items():
            counts.setdefault(station_id, _empty_target_counts(target_map.keys()))[target_name] = int(count)
    return counts


def _build_station_target_counts_from_inventory(
    wq_dir: Path,
    data_cfg: dict[str, object],
    bundle_cfg: dict[str, object],
    station_width: int,
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    target_names = list(data_cfg["targets"].keys())
    for target_name in target_names:
        target_cfg = dict(bundle_cfg.get("wq_sources", {}).get(target_name, {}))
        folder_path = _resolve_target_folder(target_cfg, wq_dir)
        if folder_path is None:
            continue
        for file_path in sorted(folder_path.glob(str(target_cfg.get("pattern", "*")))):
            if file_path.suffix.lower() == ".mat":
                continue
            station_id = _normalize_station_id(file_path.stem, station_width)
            counts.setdefault(station_id, _empty_target_counts(target_names))[target_name] += 1
    return counts


def _select_first_station_subset(station_ids: list[str], station_limit: int) -> set[str]:
    return set(station_ids[:station_limit])


def _default_min_target_coverage(station_limit: int, target_count: int) -> int:
    return max(1, station_limit // max(2, target_count * 2))


def _station_presence_count(counts: dict[str, int], target_names: Iterable[str]) -> int:
    return sum(int(counts.get(target_name, 0) > 0) for target_name in target_names)


def _station_total_count(counts: dict[str, int]) -> int:
    return int(sum(counts.values()))


def _select_coverage_aware_subset(
    station_target_counts: dict[str, dict[str, int]],
    target_names: list[str],
    station_limit: int,
    min_stations_per_target: int,
) -> set[str]:
    target_station_pool = {
        target_name: sum(int(counts.get(target_name, 0) > 0) for counts in station_target_counts.values())
        for target_name in target_names
    }
    rarity_weights = {
        target_name: 1.0 / max(1, target_station_pool[target_name])
        for target_name in target_names
    }

    def overall_rank(item: tuple[str, dict[str, int]]) -> tuple[float, float, int, str]:
        station_id, counts = item
        rarity_score = sum(rarity_weights[name] for name in target_names if counts.get(name, 0) > 0)
        return (
            -float(_station_presence_count(counts, target_names)),
            -rarity_score,
            -_station_total_count(counts),
            station_id,
        )

    def target_rank(item: tuple[str, dict[str, int]], target_name: str) -> tuple[int, float, int, str]:
        station_id, counts = item
        rarity_score = sum(rarity_weights[name] for name in target_names if counts.get(name, 0) > 0)
        return (
            -int(counts.get(target_name, 0)),
            -rarity_score,
            -_station_total_count(counts),
            station_id,
        )

    selected_station_ids: list[str] = []
    selected_station_set: set[str] = set()
    selected_target_counts = {target_name: 0 for target_name in target_names}
    priority_targets = sorted(target_names, key=lambda name: (target_station_pool[name], name))

    for target_name in priority_targets:
        required = min(min_stations_per_target, target_station_pool[target_name])
        if required <= 0:
            continue
        candidates = sorted(
            [
                item
                for item in station_target_counts.items()
                if item[0] not in selected_station_set and item[1].get(target_name, 0) > 0
            ],
            key=lambda item: target_rank(item, target_name),
        )
        for station_id, counts in candidates:
            if len(selected_station_ids) >= station_limit or selected_target_counts[target_name] >= required:
                break
            selected_station_ids.append(station_id)
            selected_station_set.add(station_id)
            for name in target_names:
                if counts.get(name, 0) > 0:
                    selected_target_counts[name] += 1

    remaining_candidates = sorted(
        [item for item in station_target_counts.items() if item[0] not in selected_station_set],
        key=overall_rank,
    )
    for station_id, _ in remaining_candidates:
        if len(selected_station_ids) >= station_limit:
            break
        selected_station_ids.append(station_id)
        selected_station_set.add(station_id)

    return selected_station_set


def _resolve_station_subset(dataset_root: Path | None, config: dict[str, object]) -> set[str] | None:
    data_cfg = config["data"]
    bundle_cfg = data_cfg.get("dataset_bundle", {})
    station_limit_value = bundle_cfg.get("station_limit")
    if station_limit_value is None:
        return None

    station_limit = int(station_limit_value)
    if station_limit <= 0:
        return None

    station_col = data_cfg["station_col"]
    station_width = int(data_cfg.get("station_id_width", 8))
    paths_cfg = config.get("paths", {})
    wq_dir = _resolve_bundle_directory(
        paths_cfg=paths_cfg,
        dataset_root=dataset_root,
        override_key="wq_root",
        fallback_candidates=bundle_cfg.get("wq_dir_candidates", ["WQ", "wq"]),
        label="WQ",
    )
    wq_cache_path = wq_dir / str(bundle_cfg.get("wq_cache_name", "wq_observations.csv"))
    if wq_cache_path.exists():
        cache_frame = read_table(wq_cache_path)
        if station_col not in cache_frame.columns:
            raise ValueError(f"WQ cache file {wq_cache_path} does not contain station column `{station_col}`.")
        station_target_counts = _build_station_target_counts_from_cache(cache_frame, data_cfg=data_cfg, station_width=station_width)
    else:
        station_target_counts = _build_station_target_counts_from_inventory(
            wq_dir,
            data_cfg=data_cfg,
            bundle_cfg=bundle_cfg,
            station_width=station_width,
        )

    if not station_target_counts:
        raise RuntimeError("`station_limit` was configured, but no candidate stations were discovered from WQ sources.")

    strategy = str(bundle_cfg.get("station_selection_strategy", "coverage_aware")).strip().lower()
    sorted_station_ids = sorted(station_target_counts)
    if strategy == "coverage_aware":
        min_stations_per_target = int(
            bundle_cfg.get(
                "coverage_min_stations_per_target",
                _default_min_target_coverage(station_limit, len(data_cfg["targets"])),
            )
        )
        subset = _select_coverage_aware_subset(
            station_target_counts,
            target_names=list(data_cfg["targets"].keys()),
            station_limit=station_limit,
            min_stations_per_target=min_stations_per_target,
        )
    else:
        subset = _select_first_station_subset(sorted_station_ids, station_limit)

    if not subset:
        raise RuntimeError("`station_limit` was configured, but no candidate stations were selected.")

    coverage_summary = {
        target_name: sum(int(station_target_counts[station_id].get(target_name, 0) > 0) for station_id in subset)
        for target_name in data_cfg["targets"]
    }
    LOGGER.info(
        "Dataset bundle debug subset enabled: station_limit=%s selected_stations=%s strategy=%s target_coverage=%s sample=%s",
        station_limit,
        len(subset),
        strategy,
        coverage_summary,
        sorted(subset)[:5],
    )
    return subset


def _load_attributes_bundle(
    dataset_root: Path | None,
    config: dict[str, object],
    station_subset: set[str] | None = None,
) -> pd.DataFrame:
    data_cfg = config["data"]
    bundle_cfg = data_cfg.get("dataset_bundle", {})
    station_col = data_cfg["station_col"]
    station_width = int(data_cfg.get("station_id_width", 8))

    paths_cfg = config.get("paths", {})
    attr_dir = _resolve_bundle_directory(
        paths_cfg=paths_cfg,
        dataset_root=dataset_root,
        override_key="attributes_root",
        fallback_candidates=bundle_cfg.get("attributes_dir_candidates", ["attritubes", "attributes"]),
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
    if station_subset is not None:
        frame = frame.loc[frame[station_col].isin(station_subset)].copy()
    keep_cols = [station_col] + [col for col in data_cfg.get("static_features", []) if col in frame.columns]
    if station_col not in keep_cols:
        keep_cols.insert(0, station_col)
    result = frame[keep_cols].drop_duplicates(subset=[station_col]).reset_index(drop=True)
    LOGGER.info("Loaded attributes: rows=%s stations=%s file=%s", len(result), result[station_col].nunique(), attr_file)
    return result


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


def _load_time_series_bundle(
    dataset_root: Path | None,
    config: dict[str, object],
    station_subset: set[str] | None = None,
) -> pd.DataFrame:
    data_cfg = config["data"]
    bundle_cfg = data_cfg.get("dataset_bundle", {})
    station_col = data_cfg["station_col"]
    date_col = data_cfg["date_col"]
    station_width = int(data_cfg.get("station_id_width", 8))

    paths_cfg = config.get("paths", {})
    ts_dir = _resolve_bundle_directory(
        paths_cfg=paths_cfg,
        dataset_root=dataset_root,
        override_key="time_series_root",
        fallback_candidates=bundle_cfg.get("time_series_dir_candidates", ["time series", "time_series"]),
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
        selected_files = _select_station_series_files(
            directory=source_dir,
            pattern=str(source_cfg.get("pattern", "*.csv")),
            station_width=station_width,
            station_subset=station_subset,
        )
        LOGGER.info("Loading time-series source `%s`: files_selected=%s", source_cfg["folder"], len(selected_files))
        rename_map = dict(source_cfg.get("rename_map", {}))
        configured_date_col = source_cfg.get("date_column")
        for file_path, station_id in selected_files:
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

    requested_dynamic = set(data_cfg.get("dynamic_features", []))
    if "air_temp" in requested_dynamic and "air_temp" not in merged.columns and {"tmin", "tmax"}.issubset(merged.columns):
        merged["air_temp"] = merged[["tmin", "tmax"]].mean(axis=1)
    result = merged.sort_values([station_col, date_col]).reset_index(drop=True)
    LOGGER.info("Loaded dynamic covariates: rows=%s stations=%s", len(result), result[station_col].nunique())
    return result


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


def _drop_rows_without_any_target(
    frame: pd.DataFrame,
    *,
    target_cols: list[str],
    date_col: str,
    context: str,
) -> pd.DataFrame:
    if frame.empty or not target_cols:
        return frame

    valid_mask = frame[target_cols].notna().any(axis=1)
    dropped = int((~valid_mask).sum())
    if dropped == 0:
        return frame

    dropped_dates = pd.to_datetime(frame.loc[~valid_mask, date_col], errors="coerce")
    LOGGER.info(
        "Dropped rows without any target values for %s: dropped=%s remaining=%s first_dropped_date=%s last_dropped_date=%s",
        context,
        dropped,
        int(valid_mask.sum()),
        dropped_dates.min(),
        dropped_dates.max(),
    )
    return frame.loc[valid_mask].copy()


def _load_wq_bundle(
    dataset_root: Path | None,
    config: dict[str, object],
    station_subset: set[str] | None = None,
) -> pd.DataFrame:
    data_cfg = config["data"]
    bundle_cfg = data_cfg.get("dataset_bundle", {})
    station_col = data_cfg["station_col"]
    date_col = data_cfg["date_col"]
    station_width = int(data_cfg.get("station_id_width", 8))

    paths_cfg = config.get("paths", {})
    wq_dir = _resolve_bundle_directory(
        paths_cfg=paths_cfg,
        dataset_root=dataset_root,
        override_key="wq_root",
        fallback_candidates=bundle_cfg.get("wq_dir_candidates", ["WQ", "wq"]),
        label="WQ",
    )
    cache_name = str(bundle_cfg.get("wq_cache_name", "wq_observations.csv"))
    cache_path = wq_dir / cache_name
    if cache_path.exists():
        cache_frame = read_table(cache_path)
        target_cols = list(data_cfg["targets"].values())
        required = [station_col, date_col] + target_cols
        missing = [col for col in required if col not in cache_frame.columns]
        if missing:
            raise ValueError(f"WQ cache file {cache_path} is missing columns: {missing}")
        cache_frame[station_col] = cache_frame[station_col].map(lambda value: _normalize_station_id(value, station_width))
        cache_frame[date_col] = pd.to_datetime(cache_frame[date_col], errors="coerce")
        if station_subset is not None:
            cache_frame = cache_frame.loc[cache_frame[station_col].isin(station_subset)].copy()
        result = cache_frame.dropna(subset=[date_col]).reset_index(drop=True)
        if _drop_all_null_target_rows_enabled(config):
            result = _drop_rows_without_any_target(
                result,
                target_cols=target_cols,
                date_col=date_col,
                context=f"WQ cache file {cache_path}",
            )
        LOGGER.info("Loaded WQ cache: rows=%s stations=%s file=%s", len(result), result[station_col].nunique(), cache_path)
        return result

    target_frames: list[pd.DataFrame] = []
    parser_errors: list[str] = []
    for target_name, target_col in data_cfg["targets"].items():
        target_cfg = dict(bundle_cfg.get("wq_sources", {}).get(target_name, {}))
        folder_path = _resolve_target_folder(target_cfg, wq_dir)
        if folder_path is None:
            continue
        selected_files: list[tuple[Path, str]] = []
        for file_path in sorted(folder_path.glob(str(target_cfg.get("pattern", "*")))):
            if file_path.suffix.lower() == ".mat":
                continue
            station_id = _normalize_station_id(file_path.stem, station_width)
            if station_subset is not None and station_id not in station_subset:
                continue
            selected_files.append((file_path, station_id))
        LOGGER.info("Loading WQ target `%s`: files_selected=%s", target_name, len(selected_files))
        source_value_col = target_cfg.get("source_value_col")
        configured_date_col = target_cfg.get("date_column")
        for file_path, station_id in selected_files:
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
        if parser_errors:
            raise RuntimeError("\n".join(sorted(set(parser_errors))))
        raise RuntimeError(f"No water-quality target files were parsed under {wq_dir}.")

    merged = pd.concat(target_frames, axis=0, ignore_index=True, sort=False)
    merged = _aggregate_station_day(merged, station_col=station_col, date_col=date_col)
    result = merged.sort_values([station_col, date_col]).reset_index(drop=True)
    if _drop_all_null_target_rows_enabled(config):
        result = _drop_rows_without_any_target(
            result,
            target_cols=list(data_cfg["targets"].values()),
            date_col=date_col,
            context="parsed WQ observations",
        )
    LOGGER.info("Loaded WQ observations: rows=%s stations=%s", len(result), result[station_col].nunique())
    return result


def load_dataset_bundle(config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths_cfg = config.get("paths", {})
    dataset_root_value = paths_cfg.get("dataset_root")
    dataset_root: Path | None = None
    if dataset_root_value:
        dataset_root = Path(str(dataset_root_value)).resolve()
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    elif not any(paths_cfg.get(key) for key in ("attributes_root", "time_series_root", "wq_root")):
        raise FileNotFoundError(
            "Provide either `paths.dataset_root` or component-specific paths: `attributes_root`, `time_series_root`, and `wq_root`."
        )

    LOGGER.info("Starting dataset bundle load.")
    station_subset = _resolve_station_subset(dataset_root, config)
    cache_dir = _resolve_cache_dir(dataset_root, config)
    station_col = config["data"]["station_col"]
    date_col = config["data"]["date_col"]

    static_df = _load_or_build_cached_frame(
        label="attributes",
        cache_path=_component_cache_path(cache_dir, "attributes"),
        builder=lambda: _load_attributes_bundle(dataset_root, config, station_subset=station_subset),
        config=config,
        exclude_cols=(station_col,),
    )
    dynamic_covariates = _load_or_build_cached_frame(
        label="time-series",
        cache_path=_component_cache_path(cache_dir, "time_series"),
        builder=lambda: _load_time_series_bundle(dataset_root, config, station_subset=station_subset),
        config=config,
        exclude_cols=(station_col, date_col),
    )
    wq_targets = _load_or_build_cached_frame(
        label="water-quality",
        cache_path=_component_cache_path(cache_dir, "wq_observations"),
        builder=lambda: _load_wq_bundle(dataset_root, config, station_subset=station_subset),
        config=config,
        exclude_cols=(station_col, date_col),
    )

    dynamic_df = dynamic_covariates.merge(wq_targets, on=[station_col, date_col], how="outer")
    dynamic_df = dynamic_df.sort_values([station_col, date_col]).reset_index(drop=True)
    if _downcast_enabled(config):
        dynamic_df = downcast_numeric_frame(dynamic_df, exclude_cols=(station_col, date_col))
        static_df = downcast_numeric_frame(static_df, exclude_cols=(station_col,))
    LOGGER.info(
        "Finished dataset bundle load: dynamic_rows=%s dynamic_stations=%s static_rows=%s static_stations=%s",
        len(dynamic_df),
        dynamic_df[station_col].nunique(),
        len(static_df),
        static_df[station_col].nunique(),
    )
    return dynamic_df, static_df

