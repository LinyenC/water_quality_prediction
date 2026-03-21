from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass
class GraphSnapshot:
    """Per-date graph snapshot used by the optional GNN backend."""

    row_indices: np.ndarray
    features: np.ndarray
    adjacency: np.ndarray
    targets: np.ndarray | None = None
    events: np.ndarray | None = None
    row_weight: np.ndarray | None = None


class GraphConvLayer(nn.Module):
    """Light graph convolution over the station-similarity graph."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_linear = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neighbor_hidden = adjacency @ hidden
        updated = self.self_linear(hidden) + self.neighbor_linear(neighbor_hidden)
        return self.dropout(self.activation(updated))


class GraphBackbone(nn.Module):
    """Optional graph encoder that plugs into the tabular multi-task model."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([GraphConvLayer(hidden_dim, dropout) for _ in range(num_layers)])
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(self.activation(self.input_projection(x)))
        for layer in self.layers:
            hidden = layer(hidden, adjacency)
        return hidden


def build_neighbor_map(edge_df: pd.DataFrame) -> dict[str, set[str]]:
    """Convert the edge list into a fast station -> neighbors mapping."""

    neighbor_map: dict[str, set[str]] = {}
    if edge_df.empty:
        return neighbor_map

    for row in edge_df.itertuples(index=False):
        source_station = str(row.source_station)
        neighbor_station = str(row.neighbor_station)
        neighbor_map.setdefault(source_station, set()).add(neighbor_station)
        neighbor_map.setdefault(neighbor_station, set()).add(source_station)
    return neighbor_map


def build_graph_snapshots(
    frame: pd.DataFrame,
    feature_array: np.ndarray,
    neighbor_map: dict[str, set[str]],
    station_col: str,
    date_col: str,
    target_names: list[str] | None = None,
    target_cols: dict[str, str] | None = None,
    thresholds: dict[str, float] | None = None,
    sample_weight: np.ndarray | None = None,
) -> list[GraphSnapshot]:
    """Build same-day station subgraphs for graph-backend training or inference."""

    aligned_frame = frame.reset_index(drop=True)
    snapshots: list[GraphSnapshot] = []

    for _, date_frame in aligned_frame.groupby(date_col, sort=True):
        row_indices = date_frame.index.to_numpy(dtype=int)
        station_ids = date_frame[station_col].astype(str).tolist()
        local_index = {station_id: idx for idx, station_id in enumerate(station_ids)}

        adjacency = np.eye(len(station_ids), dtype=np.float32)
        for station_id, row_idx in local_index.items():
            for neighbor_id in neighbor_map.get(station_id, set()):
                if neighbor_id in local_index:
                    adjacency[row_idx, local_index[neighbor_id]] = 1.0

        degree = adjacency.sum(axis=1, keepdims=True)
        adjacency = adjacency / np.clip(degree, a_min=1.0, a_max=None)

        snapshot = GraphSnapshot(
            row_indices=row_indices,
            features=feature_array[row_indices].astype(np.float32),
            adjacency=adjacency,
        )

        if target_names is not None and target_cols is not None and thresholds is not None:
            snapshot.targets = np.column_stack(
                [date_frame[target_cols[target_name]].to_numpy(dtype=float) for target_name in target_names]
            ).astype(np.float32)
            snapshot.events = np.column_stack(
                [
                    (date_frame[target_cols[target_name]].to_numpy(dtype=float) >= thresholds[target_name]).astype(float)
                    for target_name in target_names
                ]
            ).astype(np.float32)
            if sample_weight is None:
                snapshot.row_weight = np.ones(len(date_frame), dtype=np.float32)
            else:
                snapshot.row_weight = sample_weight[row_indices].astype(np.float32)

        snapshots.append(snapshot)

    return snapshots
