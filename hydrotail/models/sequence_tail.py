from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from hydrotail.data import SequenceSamples
from hydrotail.models.graph_backends import GraphBackbone, build_neighbor_map


def _resolve_device(device_value: object) -> torch.device:
    """Resolve a user-facing device config into a concrete torch device."""

    device_text = str(device_value or "auto").strip().lower()
    if device_text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(str(device_value))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device `{device}` was requested but torch.cuda.is_available() is False.")
    return device


def _quantile_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
    quantile_weights: torch.Tensor,
    row_weight: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if valid_mask is None:
        valid_mask = torch.isfinite(target)
    if not torch.any(valid_mask):
        return prediction.sum() * 0.0

    prediction = prediction[valid_mask]
    target = target[valid_mask]
    if row_weight is not None:
        row_weight = row_weight[valid_mask]

    errors = target.unsqueeze(-1) - prediction
    loss = torch.maximum(quantiles * errors, (quantiles - 1.0) * errors)
    loss = (loss * quantile_weights).mean(dim=-1)
    if row_weight is None:
        return loss.mean()
    return (loss * row_weight).sum() / row_weight.sum().clamp_min(1e-6)


def _binary_event_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    row_weight: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if valid_mask is None:
        valid_mask = torch.ones_like(targets, dtype=torch.bool)
    if not torch.any(valid_mask):
        return logits.sum() * 0.0

    logits = logits[valid_mask]
    targets = targets[valid_mask]
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if row_weight is None:
        return loss.mean()
    row_weight = row_weight[valid_mask]
    return (loss * row_weight).sum() / row_weight.sum().clamp_min(1e-6)


def _boundary_loss(
    prediction: torch.Tensor,
    row_weight: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if valid_mask is None:
        valid_mask = torch.ones(prediction.size(0), dtype=torch.bool, device=prediction.device)
    if not torch.any(valid_mask):
        return prediction.sum() * 0.0

    prediction = prediction[valid_mask]
    loss = torch.relu(-prediction).mean(dim=-1)
    if row_weight is None:
        return loss.mean()
    row_weight = row_weight[valid_mask]
    return (loss * row_weight).sum() / row_weight.sum().clamp_min(1e-6)


@dataclass
class SequenceGraphSnapshot:
    """Per-date graph snapshot for sequence models with a same-day GNN backend."""

    row_indices: np.ndarray
    sequence_values: np.ndarray
    sequence_masks: np.ndarray
    static_values: np.ndarray
    adjacency: np.ndarray
    targets: np.ndarray | None = None
    events: np.ndarray | None = None
    row_weight: np.ndarray | None = None


class CausalConvBlock(nn.Module):
    """A small causal TCN block that preserves temporal order."""

    def __init__(self, input_dim: int, output_dim: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(input_dim, output_dim, kernel_size=1) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = self.conv(x)
        out = out[..., : x.size(-1)]
        out = self.dropout(self.activation(out))
        return self.activation(out + residual)


class SequenceEncoder(nn.Module):
    """Encode a lookback window into one fused station-day representation."""

    def __init__(
        self,
        encoder_type: str,
        lookback_window: int,
        dynamic_dim: int,
        static_dim: int,
        hidden_dim: int,
        model_cfg: dict[str, object],
    ) -> None:
        super().__init__()
        self.encoder_type = encoder_type
        seq_input_dim = dynamic_dim * 2
        dropout = float(model_cfg.get("dropout", 0.1))

        self.input_projection = nn.Linear(seq_input_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, lookback_window, hidden_dim))

        if encoder_type == "tcn":
            channels = list(model_cfg.get("tcn_channels", [hidden_dim, hidden_dim]))
            kernel_size = int(model_cfg.get("kernel_size", 3))
            blocks: list[nn.Module] = []
            current_dim = hidden_dim
            for block_idx, channel_dim in enumerate(channels):
                blocks.append(
                    CausalConvBlock(
                        input_dim=current_dim,
                        output_dim=channel_dim,
                        kernel_size=kernel_size,
                        dilation=2**block_idx,
                        dropout=dropout,
                    )
                )
                current_dim = channel_dim
            self.tcn_blocks = nn.ModuleList(blocks)
            temporal_output_dim = current_dim
            self.transformer = None
        elif encoder_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=int(model_cfg.get("num_heads", 4)),
                dim_feedforward=hidden_dim * int(model_cfg.get("ff_multiplier", 2)),
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=int(model_cfg.get("num_layers", 2)),
            )
            self.tcn_blocks = nn.ModuleList()
            temporal_output_dim = hidden_dim
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        pooled_dim = temporal_output_dim * 2
        if static_dim > 0:
            self.static_mlp = nn.Sequential(
                nn.Linear(static_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            fusion_input_dim = pooled_dim + hidden_dim
        else:
            self.static_mlp = None
            fusion_input_dim = pooled_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        sequence_values: torch.Tensor,
        sequence_masks: torch.Tensor,
        static_values: torch.Tensor,
    ) -> torch.Tensor:
        # The encoder sees both filled values and explicit masks, so missingness stays identifiable.
        seq_input = torch.cat([sequence_values, sequence_masks], dim=-1)
        hidden = self.input_projection(seq_input) + self.position_embedding[:, : seq_input.size(1), :]

        if self.encoder_type == "tcn":
            hidden = hidden.transpose(1, 2)
            for block in self.tcn_blocks:
                hidden = block(hidden)
            hidden = hidden.transpose(1, 2)
        else:
            assert self.transformer is not None
            hidden = self.transformer(hidden)

        timestep_mask = (sequence_masks.sum(dim=-1) > 0).float()
        pooled = (hidden * timestep_mask.unsqueeze(-1)).sum(dim=1) / timestep_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        last_hidden = hidden[:, -1, :]
        joint = torch.cat([pooled, last_hidden], dim=-1)

        if self.static_mlp is not None and static_values.size(-1) > 0:
            joint = torch.cat([joint, self.static_mlp(static_values)], dim=-1)

        return self.fusion(joint)

class SequenceDenseTailNet(nn.Module):
    """Sequence model with direct heads on the fused representation."""

    def __init__(
        self,
        encoder_type: str,
        lookback_window: int,
        dynamic_dim: int,
        static_dim: int,
        hidden_dim: int,
        target_names: list[str],
        quantiles: list[float],
        model_cfg: dict[str, object],
    ) -> None:
        super().__init__()
        self.encoder = SequenceEncoder(
            encoder_type=encoder_type,
            lookback_window=lookback_window,
            dynamic_dim=dynamic_dim,
            static_dim=static_dim,
            hidden_dim=hidden_dim,
            model_cfg=model_cfg,
        )
        self.quantile_heads = nn.ModuleDict({name: nn.Linear(hidden_dim, len(quantiles)) for name in target_names})
        self.event_heads = nn.ModuleDict({name: nn.Linear(hidden_dim, 1) for name in target_names})

    def forward(
        self,
        sequence_values: torch.Tensor,
        sequence_masks: torch.Tensor,
        static_values: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        hidden = self.encoder(sequence_values, sequence_masks, static_values)
        outputs: dict[str, dict[str, torch.Tensor]] = {}
        for target_name in self.quantile_heads:
            quantiles, _ = torch.sort(self.quantile_heads[target_name](hidden), dim=-1)
            outputs[target_name] = {
                "quantiles": quantiles,
                "logit": self.event_heads[target_name](hidden).squeeze(-1),
            }
        return outputs


class SequenceGraphTailNet(nn.Module):
    """Sequence model that adds same-day graph propagation after temporal encoding."""

    def __init__(
        self,
        encoder_type: str,
        lookback_window: int,
        dynamic_dim: int,
        static_dim: int,
        encoder_hidden_dim: int,
        graph_hidden_dim: int,
        num_graph_layers: int,
        dropout: float,
        target_names: list[str],
        quantiles: list[float],
        model_cfg: dict[str, object],
    ) -> None:
        super().__init__()
        self.encoder = SequenceEncoder(
            encoder_type=encoder_type,
            lookback_window=lookback_window,
            dynamic_dim=dynamic_dim,
            static_dim=static_dim,
            hidden_dim=encoder_hidden_dim,
            model_cfg=model_cfg,
        )
        self.graph = GraphBackbone(
            input_dim=encoder_hidden_dim,
            hidden_dim=graph_hidden_dim,
            num_layers=num_graph_layers,
            dropout=dropout,
        )
        self.quantile_heads = nn.ModuleDict({name: nn.Linear(graph_hidden_dim, len(quantiles)) for name in target_names})
        self.event_heads = nn.ModuleDict({name: nn.Linear(graph_hidden_dim, 1) for name in target_names})

    def forward(
        self,
        sequence_values: torch.Tensor,
        sequence_masks: torch.Tensor,
        static_values: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        encoded = self.encoder(sequence_values, sequence_masks, static_values)
        hidden = self.graph(encoded, adjacency)
        outputs: dict[str, dict[str, torch.Tensor]] = {}
        for target_name in self.quantile_heads:
            quantiles, _ = torch.sort(self.quantile_heads[target_name](hidden), dim=-1)
            outputs[target_name] = {
                "quantiles": quantiles,
                "logit": self.event_heads[target_name](hidden).squeeze(-1),
            }
        return outputs


def _build_sequence_graph_snapshots(
    frame: pd.DataFrame,
    sequence_values: np.ndarray,
    sequence_masks: np.ndarray,
    static_values: np.ndarray,
    neighbor_map: dict[str, set[str]],
    station_col: str,
    date_col: str,
    target_names: list[str] | None = None,
    target_cols: dict[str, str] | None = None,
    thresholds: dict[str, float] | None = None,
    sample_weight: np.ndarray | None = None,
) -> list[SequenceGraphSnapshot]:
    """Build same-day station graphs for sequence-based GNN training or inference."""

    aligned_frame = frame.reset_index(drop=True)
    snapshots: list[SequenceGraphSnapshot] = []

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

        snapshot = SequenceGraphSnapshot(
            row_indices=row_indices,
            sequence_values=sequence_values[row_indices].astype(np.float32),
            sequence_masks=sequence_masks[row_indices].astype(np.float32),
            static_values=static_values[row_indices].astype(np.float32),
            adjacency=adjacency,
        )

        if target_names is not None and target_cols is not None and thresholds is not None:
            snapshot.targets = np.column_stack(
                [date_frame[target_cols[target_name]].to_numpy(dtype=float) for target_name in target_names]
            ).astype(np.float32)

            event_columns: list[np.ndarray] = []
            for target_name in target_names:
                values = date_frame[target_cols[target_name]].to_numpy(dtype=float)
                threshold = thresholds.get(target_name, float("nan"))
                if np.isfinite(threshold):
                    event_columns.append(np.where(np.isfinite(values), (values >= threshold).astype(float), 0.0))
                else:
                    event_columns.append(np.zeros(len(values), dtype=float))
            snapshot.events = np.column_stack(event_columns).astype(np.float32)

            if sample_weight is None:
                snapshot.row_weight = np.ones(len(date_frame), dtype=np.float32)
            else:
                snapshot.row_weight = sample_weight[row_indices].astype(np.float32)

        snapshots.append(snapshot)

    return snapshots

class SequenceTailModel:
    input_mode = "sequence"

    def __init__(
        self,
        encoder_type: str,
        lookback_window: int,
        quantiles: list[float],
        quantile_weights: list[float],
        model_cfg: dict[str, object],
        random_state: int = 42,
    ) -> None:
        self.encoder_type = encoder_type
        self.lookback_window = lookback_window
        self.quantiles = quantiles
        self.point_quantile_index = min(range(len(quantiles)), key=lambda idx: abs(quantiles[idx] - 0.5))
        self.quantile_weights = quantile_weights
        self.model_cfg = model_cfg
        self.random_state = random_state
        self.graph_backend = str(self.model_cfg.get("graph_backend", "neighbor_stats")).lower()
        if self.graph_backend not in {"none", "neighbor_stats", "gnn"}:
            raise ValueError(f"Unsupported graph_backend: {self.graph_backend}")
        self.device = _resolve_device(self.model_cfg.get("device", "auto"))

        self.target_names: list[str] = []
        self.dynamic_medians: np.ndarray | None = None
        self.static_feature_cols_: list[str] = []
        self.static_imputer = SimpleImputer(strategy="median")
        self.model: SequenceDenseTailNet | SequenceGraphTailNet | None = None
        self.neighbor_map: dict[str, set[str]] = {}
        self.station_col = "station_id"
        self.date_col = "date"

    def _resolve_static_feature_cols(self, static_feature_cols: list[str]) -> list[str]:
        if self.graph_backend in {"none", "gnn"}:
            return [col for col in static_feature_cols if not col.startswith("graph_neighbor_")]
        return list(static_feature_cols)

    def _impute_dynamic(self, array: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            medians = np.nanmedian(array.reshape(-1, array.shape[-1]), axis=0)
            medians = np.where(np.isnan(medians), 0.0, medians)
            self.dynamic_medians = medians.astype(np.float32)
        assert self.dynamic_medians is not None
        filled = array.copy()
        nan_mask = np.isnan(filled)
        if np.any(nan_mask):
            feature_idx = np.where(nan_mask)[-1]
            filled[nan_mask] = self.dynamic_medians[feature_idx]
        return filled.astype(np.float32)

    def _prepare_bundle(self, bundle: SequenceSamples, fit: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sequence_values = self._impute_dynamic(bundle.sequence_values, fit=fit)
        sequence_masks = bundle.sequence_masks.astype(np.float32)

        if not self.static_feature_cols_ or bundle.static_values.shape[1] == 0:
            static_values = np.zeros((len(bundle.frame), 0), dtype=np.float32)
        else:
            column_indices = [bundle.static_feature_cols.index(col) for col in self.static_feature_cols_]
            selected_static = bundle.static_values[:, column_indices]
            if fit:
                static_values = self.static_imputer.fit_transform(selected_static)
            else:
                static_values = self.static_imputer.transform(selected_static)
            static_values = static_values.astype(np.float32)

        return sequence_values, sequence_masks, static_values

    def _graph_cfg(self) -> dict[str, object]:
        graph_cfg = self.model_cfg.get("gnn", {})
        if not isinstance(graph_cfg, dict):
            return {}
        return graph_cfg

    def fit(
        self,
        train_bundle: SequenceSamples,
        valid_bundle: SequenceSamples,
        target_cols: dict[str, str],
        thresholds: dict[str, float],
        sample_weight: np.ndarray | None = None,
        edge_df: pd.DataFrame | None = None,
        station_col: str = "station_id",
        date_col: str = "date",
    ) -> None:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.target_names = list(target_cols.keys())
        self.station_col = station_col
        self.date_col = date_col
        self.static_feature_cols_ = self._resolve_static_feature_cols(train_bundle.static_feature_cols)

        train_values, train_masks, train_static = self._prepare_bundle(train_bundle, fit=True)
        valid_values, valid_masks, valid_static = self._prepare_bundle(valid_bundle, fit=False)

        y_train = np.column_stack(
            [train_bundle.frame[target_cols[name]].to_numpy(dtype=float) for name in self.target_names]
        ).astype(np.float32)
        y_valid = np.column_stack(
            [valid_bundle.frame[target_cols[name]].to_numpy(dtype=float) for name in self.target_names]
        ).astype(np.float32)
        e_train = np.column_stack(
            [
                np.where(
                    np.isfinite(train_bundle.frame[target_cols[name]].to_numpy(dtype=float)),
                    (train_bundle.frame[target_cols[name]].to_numpy(dtype=float) >= thresholds[name]).astype(float),
                    0.0,
                )
                for name in self.target_names
            ]
        ).astype(np.float32)
        e_valid = np.column_stack(
            [
                np.where(
                    np.isfinite(valid_bundle.frame[target_cols[name]].to_numpy(dtype=float)),
                    (valid_bundle.frame[target_cols[name]].to_numpy(dtype=float) >= thresholds[name]).astype(float),
                    0.0,
                )
                for name in self.target_names
            ]
        ).astype(np.float32)

        quantiles_tensor = torch.tensor(self.quantiles, dtype=torch.float32, device=self.device)
        quantile_weight_tensor = torch.tensor(self.quantile_weights, dtype=torch.float32, device=self.device)

        if self.graph_backend == "gnn":
            self._fit_with_gnn_backend(
                train_bundle=train_bundle,
                valid_bundle=valid_bundle,
                train_values=train_values,
                train_masks=train_masks,
                train_static=train_static,
                valid_values=valid_values,
                valid_masks=valid_masks,
                valid_static=valid_static,
                target_cols=target_cols,
                thresholds=thresholds,
                quantiles_tensor=quantiles_tensor,
                quantile_weight_tensor=quantile_weight_tensor,
                sample_weight=sample_weight,
                edge_df=edge_df,
            )
            return

        self._fit_with_dense_backend(
            train_values=train_values,
            train_masks=train_masks,
            train_static=train_static,
            valid_values=valid_values,
            valid_masks=valid_masks,
            valid_static=valid_static,
            y_train=y_train,
            y_valid=y_valid,
            e_train=e_train,
            e_valid=e_valid,
            quantiles_tensor=quantiles_tensor,
            quantile_weight_tensor=quantile_weight_tensor,
            sample_weight=sample_weight,
        )

    def _fit_with_dense_backend(
        self,
        train_values: np.ndarray,
        train_masks: np.ndarray,
        train_static: np.ndarray,
        valid_values: np.ndarray,
        valid_masks: np.ndarray,
        valid_static: np.ndarray,
        y_train: np.ndarray,
        y_valid: np.ndarray,
        e_train: np.ndarray,
        e_valid: np.ndarray,
        quantiles_tensor: torch.Tensor,
        quantile_weight_tensor: torch.Tensor,
        sample_weight: np.ndarray | None,
    ) -> None:
        pin_memory = self.device.type == "cuda"
        train_dataset = TensorDataset(
            torch.tensor(train_values),
            torch.tensor(train_masks),
            torch.tensor(train_static),
            torch.tensor(y_train),
            torch.tensor(e_train),
        )

        if sample_weight is None:
            sample_weight = 1.0 + float(self.model_cfg.get("tail_weight_multiplier", 1.0)) * np.max(e_train, axis=1)
        sampler = WeightedRandomSampler(
            weights=torch.tensor(np.asarray(sample_weight, dtype=np.float32)),
            num_samples=len(sample_weight),
            replacement=True,
        )
        loader = DataLoader(
            train_dataset,
            batch_size=int(self.model_cfg.get("batch_size", 128)),
            sampler=sampler,
            pin_memory=pin_memory,
        )

        hidden_dim = int(self.model_cfg.get("hidden_dim", 64))
        self.model = SequenceDenseTailNet(
            encoder_type=self.encoder_type,
            lookback_window=self.lookback_window,
            dynamic_dim=train_values.shape[-1],
            static_dim=train_static.shape[-1],
            hidden_dim=hidden_dim,
            target_names=self.target_names,
            quantiles=self.quantiles,
            model_cfg=self.model_cfg,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.model_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(self.model_cfg.get("weight_decay", 1e-4)),
        )

        best_state = copy.deepcopy(self.model.state_dict())
        best_valid = float("inf")
        patience = int(self.model_cfg.get("patience", 5))
        wait = 0

        valid_tensors = (
            torch.tensor(valid_values),
            torch.tensor(valid_masks),
            torch.tensor(valid_static),
            torch.tensor(y_valid),
            torch.tensor(e_valid),
        )

        for _epoch in range(int(self.model_cfg.get("epochs", 20))):
            self.model.train()
            for batch_values, batch_masks, batch_static, batch_targets, batch_events in loader:
                batch_values = batch_values.to(self.device, non_blocking=pin_memory)
                batch_masks = batch_masks.to(self.device, non_blocking=pin_memory)
                batch_static = batch_static.to(self.device, non_blocking=pin_memory)
                batch_targets = batch_targets.to(self.device, non_blocking=pin_memory)
                batch_events = batch_events.to(self.device, non_blocking=pin_memory)
                optimizer.zero_grad()
                outputs = self.model(batch_values, batch_masks, batch_static)
                total_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                for target_idx, target_name in enumerate(self.target_names):
                    valid_mask = torch.isfinite(batch_targets[:, target_idx])
                    quantile_pred = outputs[target_name]["quantiles"]
                    event_logit = outputs[target_name]["logit"]
                    q_loss = _quantile_loss(
                        quantile_pred,
                        batch_targets[:, target_idx],
                        quantiles_tensor,
                        quantile_weight_tensor,
                        valid_mask=valid_mask,
                    )
                    event_loss = _binary_event_loss(
                        event_logit,
                        batch_events[:, target_idx],
                        valid_mask=valid_mask,
                    )
                    boundary_loss = _boundary_loss(quantile_pred, valid_mask=valid_mask)
                    total_loss = total_loss + q_loss + event_loss + 0.1 * boundary_loss
                total_loss.backward()
                optimizer.step()

            valid_loss = self._dense_validation_loss(valid_tensors, quantiles_tensor, quantile_weight_tensor)
            if valid_loss < best_valid:
                best_valid = valid_loss
                best_state = copy.deepcopy(self.model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        self.model.load_state_dict(best_state)
        self.model.eval()

    def _fit_with_gnn_backend(
        self,
        train_bundle: SequenceSamples,
        valid_bundle: SequenceSamples,
        train_values: np.ndarray,
        train_masks: np.ndarray,
        train_static: np.ndarray,
        valid_values: np.ndarray,
        valid_masks: np.ndarray,
        valid_static: np.ndarray,
        target_cols: dict[str, str],
        thresholds: dict[str, float],
        quantiles_tensor: torch.Tensor,
        quantile_weight_tensor: torch.Tensor,
        sample_weight: np.ndarray | None,
        edge_df: pd.DataFrame | None,
    ) -> None:
        if edge_df is None or edge_df.empty:
            raise RuntimeError("graph_backend='gnn' requires a non-empty edge_df. Enable the graph block in the config.")

        self.neighbor_map = build_neighbor_map(edge_df)
        if not self.neighbor_map:
            raise RuntimeError("graph_backend='gnn' requires at least one valid graph edge.")

        if sample_weight is None:
            train_targets = np.column_stack(
                [train_bundle.frame[target_cols[name]].to_numpy(dtype=float) for name in self.target_names]
            )
            train_events = []
            for target_idx, target_name in enumerate(self.target_names):
                threshold = thresholds.get(target_name, float("nan"))
                if np.isfinite(threshold):
                    target_values = train_targets[:, target_idx]
                    train_events.append(np.where(np.isfinite(target_values), (target_values >= threshold).astype(float), 0.0))
                else:
                    train_events.append(np.zeros(len(train_bundle.frame), dtype=float))
            sample_weight = 1.0 + float(self.model_cfg.get("tail_weight_multiplier", 1.0)) * np.max(
                np.column_stack(train_events),
                axis=1,
            )

        train_snapshots = _build_sequence_graph_snapshots(
            frame=train_bundle.frame,
            sequence_values=train_values,
            sequence_masks=train_masks,
            static_values=train_static,
            neighbor_map=self.neighbor_map,
            station_col=self.station_col,
            date_col=self.date_col,
            target_names=self.target_names,
            target_cols=target_cols,
            thresholds=thresholds,
            sample_weight=sample_weight,
        )
        valid_snapshots = _build_sequence_graph_snapshots(
            frame=valid_bundle.frame,
            sequence_values=valid_values,
            sequence_masks=valid_masks,
            static_values=valid_static,
            neighbor_map=self.neighbor_map,
            station_col=self.station_col,
            date_col=self.date_col,
            target_names=self.target_names,
            target_cols=target_cols,
            thresholds=thresholds,
        )

        hidden_dim = int(self.model_cfg.get("hidden_dim", 64))
        graph_cfg = self._graph_cfg()
        graph_hidden_dim = int(graph_cfg.get("hidden_dim", hidden_dim))
        num_graph_layers = int(graph_cfg.get("num_layers", 2))
        dropout = float(graph_cfg.get("dropout", self.model_cfg.get("dropout", 0.1)))

        self.model = SequenceGraphTailNet(
            encoder_type=self.encoder_type,
            lookback_window=self.lookback_window,
            dynamic_dim=train_values.shape[-1],
            static_dim=train_static.shape[-1],
            encoder_hidden_dim=hidden_dim,
            graph_hidden_dim=graph_hidden_dim,
            num_graph_layers=num_graph_layers,
            dropout=dropout,
            target_names=self.target_names,
            quantiles=self.quantiles,
            model_cfg=self.model_cfg,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.model_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(self.model_cfg.get("weight_decay", 1e-4)),
        )

        best_state = copy.deepcopy(self.model.state_dict())
        best_valid = float("inf")
        patience = int(self.model_cfg.get("patience", 5))
        wait = 0

        for _epoch in range(int(self.model_cfg.get("epochs", 20))):
            self.model.train()
            for snapshot in train_snapshots:
                optimizer.zero_grad()
                loss = self._graph_snapshot_loss(snapshot, quantiles_tensor, quantile_weight_tensor)
                loss.backward()
                optimizer.step()

            valid_loss = self._graph_validation_loss(valid_snapshots, quantiles_tensor, quantile_weight_tensor)
            if valid_loss < best_valid:
                best_valid = valid_loss
                best_state = copy.deepcopy(self.model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        self.model.load_state_dict(best_state)
        self.model.eval()

    def _dense_validation_loss(
        self,
        valid_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        quantiles: torch.Tensor,
        quantile_weights: torch.Tensor,
    ) -> float:
        assert self.model is not None
        values, masks, static_values, targets, events = valid_tensors
        values = values.to(self.device)
        masks = masks.to(self.device)
        static_values = static_values.to(self.device)
        targets = targets.to(self.device)
        events = events.to(self.device)
        with torch.no_grad():
            outputs = self.model(values, masks, static_values)
            total_loss = 0.0
            for target_idx, target_name in enumerate(self.target_names):
                valid_mask = torch.isfinite(targets[:, target_idx])
                quantile_pred = outputs[target_name]["quantiles"]
                event_logit = outputs[target_name]["logit"]
                q_loss = _quantile_loss(
                    quantile_pred,
                    targets[:, target_idx],
                    quantiles,
                    quantile_weights,
                    valid_mask=valid_mask,
                )
                event_loss = _binary_event_loss(
                    event_logit,
                    events[:, target_idx],
                    valid_mask=valid_mask,
                )
                boundary_loss = _boundary_loss(quantile_pred, valid_mask=valid_mask)
                total_loss += float(q_loss + event_loss + 0.1 * boundary_loss)
            return total_loss

    def _graph_snapshot_loss(
        self,
        snapshot: SequenceGraphSnapshot,
        quantiles: torch.Tensor,
        quantile_weights: torch.Tensor,
    ) -> torch.Tensor:
        assert self.model is not None
        assert snapshot.targets is not None
        assert snapshot.events is not None
        assert snapshot.row_weight is not None

        sequence_values = torch.tensor(snapshot.sequence_values, dtype=torch.float32, device=self.device)
        sequence_masks = torch.tensor(snapshot.sequence_masks, dtype=torch.float32, device=self.device)
        static_values = torch.tensor(snapshot.static_values, dtype=torch.float32, device=self.device)
        adjacency = torch.tensor(snapshot.adjacency, dtype=torch.float32, device=self.device)
        targets = torch.tensor(snapshot.targets, dtype=torch.float32, device=self.device)
        events = torch.tensor(snapshot.events, dtype=torch.float32, device=self.device)
        row_weight = torch.tensor(snapshot.row_weight, dtype=torch.float32, device=self.device)
        outputs = self.model(sequence_values, sequence_masks, static_values, adjacency)

        total_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        for target_idx, target_name in enumerate(self.target_names):
            valid_mask = torch.isfinite(targets[:, target_idx])
            quantile_pred = outputs[target_name]["quantiles"]
            event_logit = outputs[target_name]["logit"]
            q_loss = _quantile_loss(
                quantile_pred,
                targets[:, target_idx],
                quantiles,
                quantile_weights,
                row_weight=row_weight,
                valid_mask=valid_mask,
            )
            event_loss = _binary_event_loss(
                event_logit,
                events[:, target_idx],
                row_weight=row_weight,
                valid_mask=valid_mask,
            )
            boundary_loss = _boundary_loss(quantile_pred, row_weight=row_weight, valid_mask=valid_mask)
            total_loss = total_loss + q_loss + event_loss + 0.1 * boundary_loss
        return total_loss

    def _graph_validation_loss(
        self,
        snapshots: list[SequenceGraphSnapshot],
        quantiles: torch.Tensor,
        quantile_weights: torch.Tensor,
    ) -> float:
        assert self.model is not None
        total_loss = 0.0
        with torch.no_grad():
            for snapshot in snapshots:
                total_loss += float(self._graph_snapshot_loss(snapshot, quantiles, quantile_weights))
        return total_loss / max(len(snapshots), 1)

    def _empty_predictions(self, row_count: int) -> dict[str, dict[str, object]]:
        return {
            target_name: {
                "point": np.zeros(row_count, dtype=float),
                "quantiles": {quantile: np.zeros(row_count, dtype=float) for quantile in self.quantiles},
                "exceedance_probability": np.zeros(row_count, dtype=float),
            }
            for target_name in self.target_names
        }

    def predict(self, bundle: SequenceSamples) -> dict[str, dict[str, object]]:
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")

        sequence_values, sequence_masks, static_values = self._prepare_bundle(bundle, fit=False)
        if self.graph_backend == "gnn":
            return self._predict_with_gnn_backend(bundle.frame, sequence_values, sequence_masks, static_values)

        with torch.no_grad():
            outputs = self.model(
                torch.tensor(sequence_values, device=self.device),
                torch.tensor(sequence_masks, device=self.device),
                torch.tensor(static_values, device=self.device),
            )

        predictions = self._empty_predictions(len(bundle.frame))
        for target_name in self.target_names:
            quantile_matrix = outputs[target_name]["quantiles"].detach().cpu().numpy()
            predictions[target_name]["point"] = quantile_matrix[:, self.point_quantile_index]
            predictions[target_name]["quantiles"] = {
                quantile: quantile_matrix[:, idx] for idx, quantile in enumerate(self.quantiles)
            }
            predictions[target_name]["exceedance_probability"] = (
                torch.sigmoid(outputs[target_name]["logit"]).detach().cpu().numpy()
            )
        return predictions

    def _predict_with_gnn_backend(
        self,
        frame: pd.DataFrame,
        sequence_values: np.ndarray,
        sequence_masks: np.ndarray,
        static_values: np.ndarray,
    ) -> dict[str, dict[str, object]]:
        assert self.model is not None

        snapshots = _build_sequence_graph_snapshots(
            frame=frame,
            sequence_values=sequence_values,
            sequence_masks=sequence_masks,
            static_values=static_values,
            neighbor_map=self.neighbor_map,
            station_col=self.station_col,
            date_col=self.date_col,
        )
        predictions = self._empty_predictions(len(frame))

        with torch.no_grad():
            for snapshot in snapshots:
                outputs = self.model(
                    torch.tensor(snapshot.sequence_values, dtype=torch.float32, device=self.device),
                    torch.tensor(snapshot.sequence_masks, dtype=torch.float32, device=self.device),
                    torch.tensor(snapshot.static_values, dtype=torch.float32, device=self.device),
                    torch.tensor(snapshot.adjacency, dtype=torch.float32, device=self.device),
                )
                for target_name in self.target_names:
                    quantile_matrix = outputs[target_name]["quantiles"].detach().cpu().numpy()
                    predictions[target_name]["point"][snapshot.row_indices] = quantile_matrix[:, self.point_quantile_index]
                    for idx, quantile in enumerate(self.quantiles):
                        predictions[target_name]["quantiles"][quantile][snapshot.row_indices] = quantile_matrix[:, idx]
                    predictions[target_name]["exceedance_probability"][snapshot.row_indices] = (
                        torch.sigmoid(outputs[target_name]["logit"]).detach().cpu().numpy()
                    )

        return predictions

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            torch.save(self, handle)
