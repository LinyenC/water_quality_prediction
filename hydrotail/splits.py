from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SplitBundle:
    frame: pd.DataFrame
    metadata: dict[str, object]


_ANALYSIS_GROUPS = [
    "train_station_early",
    "train_station_late",
    "test_station_early",
    "test_station_late",
]


def assign_splits(
    frame: pd.DataFrame,
    station_col: str,
    date_col: str,
    split_cfg: dict[str, object],
) -> SplitBundle:
    strategy = str(split_cfg.get("strategy", "unseen_station_and_future"))
    seed = int(split_cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    result = frame.copy()
    result["split"] = "drop"
    result["station_group"] = "unknown"
    result["time_group"] = "unknown"
    result["analysis_group"] = ""

    if result.empty:
        return SplitBundle(result, {"strategy": strategy, "counts": {}, "analysis_counts": {}})

    stations = np.array(sorted(result[station_col].dropna().unique().tolist()))
    rng.shuffle(stations)

    train_fraction = float(split_cfg.get("train_station_fraction", 0.7))
    valid_fraction = float(split_cfg.get("valid_station_fraction", 0.15))
    train_cut = max(1, int(len(stations) * train_fraction))
    valid_cut = min(len(stations), train_cut + max(1, int(len(stations) * valid_fraction)))

    train_stations = set(stations[:train_cut].tolist())
    valid_stations = set(stations[train_cut:valid_cut].tolist())
    test_stations = set(stations[valid_cut:].tolist())
    if not test_stations and valid_stations:
        test_stations = valid_stations
        valid_stations = set()

    dates = np.array(sorted(pd.to_datetime(result[date_col]).dropna().unique()))
    train_time_fraction = float(split_cfg.get("train_time_fraction", 0.7))
    valid_time_fraction = float(split_cfg.get("valid_time_fraction", 0.15))
    train_time_cut = max(1, int(len(dates) * train_time_fraction))
    valid_time_cut = min(len(dates), train_time_cut + max(1, int(len(dates) * valid_time_fraction)))
    train_end = dates[train_time_cut - 1]
    valid_end = dates[valid_time_cut - 1] if len(dates) else train_end

    dt = pd.to_datetime(result[date_col])
    result.loc[result[station_col].isin(train_stations), "station_group"] = "train_station"
    result.loc[result[station_col].isin(valid_stations), "station_group"] = "valid_station"
    result.loc[result[station_col].isin(test_stations), "station_group"] = "test_station"

    result.loc[dt <= train_end, "time_group"] = "early"
    result.loc[(dt > train_end) & (dt <= valid_end), "time_group"] = "middle"
    result.loc[dt > valid_end, "time_group"] = "late"

    result.loc[(result["station_group"] == "train_station") & (result["time_group"] == "early"), "analysis_group"] = "train_station_early"
    result.loc[(result["station_group"] == "train_station") & (result["time_group"] == "late"), "analysis_group"] = "train_station_late"
    result.loc[(result["station_group"] == "test_station") & (result["time_group"] == "early"), "analysis_group"] = "test_station_early"
    result.loc[(result["station_group"] == "test_station") & (result["time_group"] == "late"), "analysis_group"] = "test_station_late"

    if strategy == "unseen_station":
        result.loc[result[station_col].isin(train_stations), "split"] = "train"
        result.loc[result[station_col].isin(valid_stations), "split"] = "valid"
        result.loc[result[station_col].isin(test_stations), "split"] = "test"
    elif strategy == "time":
        result.loc[dt <= train_end, "split"] = "train"
        result.loc[(dt > train_end) & (dt <= valid_end), "split"] = "valid"
        result.loc[dt > valid_end, "split"] = "test"
    else:
        result.loc[result["analysis_group"] == "train_station_early", "split"] = "train"
        result.loc[(result[station_col].isin(valid_stations)) & (result["time_group"] == "middle"), "split"] = "valid"
        result.loc[result["analysis_group"] == "test_station_late", "split"] = "test"

    counts = result["split"].value_counts().to_dict()
    analysis_counts = {
        group_name: int((result["analysis_group"] == group_name).sum())
        for group_name in _ANALYSIS_GROUPS
    }
    metadata = {
        "strategy": strategy,
        "train_stations": sorted(train_stations),
        "valid_stations": sorted(valid_stations),
        "test_stations": sorted(test_stations),
        "train_end_date": str(pd.Timestamp(train_end).date()),
        "valid_end_date": str(pd.Timestamp(valid_end).date()),
        "counts": counts,
        "analysis_counts": analysis_counts,
    }
    return SplitBundle(result, metadata)
