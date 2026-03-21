from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def build_similarity_outputs(
    static_df: pd.DataFrame,
    station_col: str,
    feature_cols: Iterable[str],
    k_neighbors: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [col for col in feature_cols if col in static_df.columns]
    if not feature_cols:
        empty = static_df[[station_col]].drop_duplicates().copy()
        return empty, pd.DataFrame(columns=["source_station", "neighbor_station", "distance"])

    station_frame = static_df[[station_col] + feature_cols].drop_duplicates(subset=[station_col]).reset_index(drop=True)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    matrix = imputer.fit_transform(station_frame[feature_cols])
    matrix = scaler.fit_transform(matrix)

    neighbor_count = min(max(k_neighbors + 1, 2), len(station_frame))
    knn = NearestNeighbors(n_neighbors=neighbor_count)
    knn.fit(matrix)
    distances, indices = knn.kneighbors(matrix)

    edge_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    values = station_frame[feature_cols].to_numpy(dtype=float)
    stations = station_frame[station_col].tolist()

    for row_idx, station in enumerate(stations):
        neighbor_ids = []
        neighbor_distances = []
        for neighbor_idx, distance in zip(indices[row_idx], distances[row_idx]):
            if neighbor_idx == row_idx:
                continue
            neighbor_ids.append(neighbor_idx)
            neighbor_distances.append(float(distance))
            edge_rows.append(
                {
                    "source_station": station,
                    "neighbor_station": stations[neighbor_idx],
                    "distance": float(distance),
                }
            )

        row: dict[str, object] = {station_col: station}
        if neighbor_ids:
            neighbor_values = values[neighbor_ids]
            for col_idx, col_name in enumerate(feature_cols):
                row[f"graph_neighbor_mean_{col_name}"] = float(np.nanmean(neighbor_values[:, col_idx]))
            row["graph_neighbor_mean_distance"] = float(np.mean(neighbor_distances))
            row["graph_neighbor_count"] = len(neighbor_ids)
        else:
            for col_name in feature_cols:
                row[f"graph_neighbor_mean_{col_name}"] = np.nan
            row["graph_neighbor_mean_distance"] = np.nan
            row["graph_neighbor_count"] = 0
        feature_rows.append(row)

    return pd.DataFrame(feature_rows), pd.DataFrame(edge_rows)
