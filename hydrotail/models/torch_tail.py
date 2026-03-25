from __future__ import annotations

import copy
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from hydrotail.models.graph_backends import GraphBackbone, GraphSnapshot, build_graph_snapshots, build_neighbor_map

LOGGER = logging.getLogger(__name__)


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


class MultiTaskTailNet(nn.Module):
    """Shared MLP backbone for tabular multi-task tail prediction."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float, target_names: list[str], quantiles: list[float]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            current_dim = hidden_dim
        self.shared = nn.Sequential(*layers)
        self.quantile_heads = nn.ModuleDict({name: nn.Linear(current_dim, len(quantiles)) for name in target_names})
        self.event_heads = nn.ModuleDict({name: nn.Linear(current_dim, 1) for name in target_names})

    def forward(self, x: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        hidden = self.shared(x)
        outputs: dict[str, dict[str, torch.Tensor]] = {}
        for target_name in self.quantile_heads:
            quantile_output, _ = torch.sort(self.quantile_heads[target_name](hidden), dim=-1)
            outputs[target_name] = {
                "quantiles": quantile_output,
                "logit": self.event_heads[target_name](hidden).squeeze(-1),
            }
        return outputs


class GraphMultiTaskTailNet(nn.Module):
    """Shared graph backbone for tabular-plus-graph multi-task tail prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        target_names: list[str],
        quantiles: list[float],
    ) -> None:
        super().__init__()
        self.shared = GraphBackbone(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.quantile_heads = nn.ModuleDict({name: nn.Linear(hidden_dim, len(quantiles)) for name in target_names})
        self.event_heads = nn.ModuleDict({name: nn.Linear(hidden_dim, 1) for name in target_names})

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        hidden = self.shared(x, adjacency)
        outputs: dict[str, dict[str, torch.Tensor]] = {}
        for target_name in self.quantile_heads:
            quantile_output, _ = torch.sort(self.quantile_heads[target_name](hidden), dim=-1)
            outputs[target_name] = {
                "quantiles": quantile_output,
                "logit": self.event_heads[target_name](hidden).squeeze(-1),
            }
        return outputs


class TorchTailModel:
    input_mode = "tabular"

    def __init__(
        self,
        quantiles: list[float],
        quantile_weights: list[float],
        model_cfg: dict[str, object],
        random_state: int = 42,
    ) -> None:
        self.quantiles = quantiles
        self.point_quantile_index = min(range(len(quantiles)), key=lambda idx: abs(quantiles[idx] - 0.5))
        self.quantile_weights = quantile_weights
        self.model_cfg = model_cfg
        self.random_state = random_state
        self.graph_backend = str(self.model_cfg.get("graph_backend", "neighbor_stats")).lower()
        if self.graph_backend not in {"none", "neighbor_stats", "gnn"}:
            raise ValueError(f"Unsupported graph_backend: {self.graph_backend}")
        self.device = _resolve_device(self.model_cfg.get("device", "auto"))

        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.target_names: list[str] = []
        self.feature_cols_: list[str] = []
        self.model: MultiTaskTailNet | GraphMultiTaskTailNet | None = None
        self.neighbor_map: dict[str, set[str]] = {}
        self.station_col = "station_id"
        self.date_col = "date"

    def _resolve_feature_cols(self, feature_cols: list[str]) -> list[str]:
        if self.graph_backend in {"none", "gnn"}:
            return [col for col in feature_cols if not col.startswith("graph_neighbor_")]
        return list(feature_cols)

    def _transform_features(self, frame: pd.DataFrame, fit: bool) -> np.ndarray:
        raw = frame[self.feature_cols_]
        if fit:
            return self.scaler.fit_transform(self.imputer.fit_transform(raw))
        return self.scaler.transform(self.imputer.transform(raw))

    def _graph_cfg(self) -> dict[str, object]:
        graph_cfg = self.model_cfg.get("gnn", {})
        if not isinstance(graph_cfg, dict):
            return {}
        return graph_cfg

    def fit(
        self,
        train_frame: pd.DataFrame,
        valid_frame: pd.DataFrame,
        feature_cols: list[str],
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
        self.feature_cols_ = self._resolve_feature_cols(feature_cols)
        self.station_col = station_col
        self.date_col = date_col

        x_train = self._transform_features(train_frame, fit=True)
        x_valid = self._transform_features(valid_frame, fit=False)

        y_train = np.column_stack([train_frame[target_cols[name]].to_numpy(dtype=float) for name in self.target_names])
        y_valid = np.column_stack([valid_frame[target_cols[name]].to_numpy(dtype=float) for name in self.target_names])
        e_train = np.column_stack(
            [
                np.where(np.isfinite(train_frame[target_cols[name]].to_numpy(dtype=float)), (train_frame[target_cols[name]].to_numpy(dtype=float) >= thresholds[name]).astype(float), 0.0)
                for name in self.target_names
            ]
        )
        e_valid = np.column_stack(
            [
                np.where(np.isfinite(valid_frame[target_cols[name]].to_numpy(dtype=float)), (valid_frame[target_cols[name]].to_numpy(dtype=float) >= thresholds[name]).astype(float), 0.0)
                for name in self.target_names
            ]
        )

        quantiles_tensor = torch.tensor(self.quantiles, dtype=torch.float32, device=self.device)
        quantile_weight_tensor = torch.tensor(self.quantile_weights, dtype=torch.float32, device=self.device)

        if self.graph_backend == "gnn":
            self._fit_with_gnn_backend(
                train_frame=train_frame,
                valid_frame=valid_frame,
                x_train=x_train,
                x_valid=x_valid,
                target_cols=target_cols,
                thresholds=thresholds,
                quantiles_tensor=quantiles_tensor,
                quantile_weight_tensor=quantile_weight_tensor,
                sample_weight=sample_weight,
                edge_df=edge_df,
            )
            return

        self._fit_with_dense_backend(
            x_train=x_train,
            x_valid=x_valid,
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
        x_train: np.ndarray,
        x_valid: np.ndarray,
        y_train: np.ndarray,
        y_valid: np.ndarray,
        e_train: np.ndarray,
        e_valid: np.ndarray,
        quantiles_tensor: torch.Tensor,
        quantile_weight_tensor: torch.Tensor,
        sample_weight: np.ndarray | None,
    ) -> None:
        pin_memory = self.device.type == "cuda"
        dataset = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(e_train, dtype=torch.float32),
        )
        if sample_weight is None:
            sample_weight = 1.0 + float(self.model_cfg.get("tail_weight_multiplier", 1.0)) * np.max(e_train, axis=1)
        sampler = WeightedRandomSampler(
            weights=torch.tensor(np.asarray(sample_weight, dtype=np.float32)),
            num_samples=len(sample_weight),
            replacement=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(self.model_cfg.get("batch_size", 256)),
            sampler=sampler,
            pin_memory=pin_memory,
        )

        self.model = MultiTaskTailNet(
            input_dim=x_train.shape[1],
            hidden_dims=list(self.model_cfg.get("hidden_dims", [128, 64])),
            dropout=float(self.model_cfg.get("dropout", 0.1)),
            target_names=self.target_names,
            quantiles=self.quantiles,
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
            torch.tensor(x_valid, dtype=torch.float32),
            torch.tensor(y_valid, dtype=torch.float32),
            torch.tensor(e_valid, dtype=torch.float32),
        )

        total_epochs = int(self.model_cfg.get("epochs", 25))
        LOGGER.info(
            "Starting dense torch-tail training: epochs=%s batch_size=%s train_samples=%s valid_samples=%s",
            total_epochs,
            int(self.model_cfg.get("batch_size", 256)),
            len(x_train),
            len(x_valid),
        )
        for epoch_idx in range(total_epochs):
            epoch_number = epoch_idx + 1
            epoch_start = time.perf_counter()
            self.model.train()
            epoch_train_loss = 0.0
            epoch_batch_count = 0
            for batch_x, batch_y, batch_e in loader:
                batch_x = batch_x.to(self.device, non_blocking=pin_memory)
                batch_y = batch_y.to(self.device, non_blocking=pin_memory)
                batch_e = batch_e.to(self.device, non_blocking=pin_memory)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                total_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                for target_idx, target_name in enumerate(self.target_names):
                    valid_mask = torch.isfinite(batch_y[:, target_idx])
                    quantile_pred = outputs[target_name]["quantiles"]
                    event_logit = outputs[target_name]["logit"]
                    q_loss = _quantile_loss(
                        quantile_pred,
                        batch_y[:, target_idx],
                        quantiles_tensor,
                        quantile_weight_tensor,
                        valid_mask=valid_mask,
                    )
                    event_loss = _binary_event_loss(
                        event_logit,
                        batch_e[:, target_idx],
                        valid_mask=valid_mask,
                    )
                    boundary_loss = _boundary_loss(quantile_pred, valid_mask=valid_mask)
                    total_loss = total_loss + q_loss + event_loss + 0.1 * boundary_loss
                total_loss.backward()
                optimizer.step()
                epoch_train_loss += float(total_loss.detach().cpu())
                epoch_batch_count += 1

            valid_loss = self._dense_validation_loss(valid_tensors, quantiles_tensor, quantile_weight_tensor)
            average_train_loss = epoch_train_loss / max(epoch_batch_count, 1)
            if valid_loss < best_valid:
                best_valid = valid_loss
                best_state = copy.deepcopy(self.model.state_dict())
                wait = 0
            else:
                wait += 1
            LOGGER.info(
                "Dense torch-tail epoch %s/%s: train_loss=%.4f valid_loss=%.4f best_valid=%.4f wait=%s elapsed=%.1fs",
                epoch_number,
                total_epochs,
                average_train_loss,
                valid_loss,
                best_valid,
                wait,
                time.perf_counter() - epoch_start,
            )
            if wait >= patience:
                LOGGER.info("Dense torch-tail early stopping at epoch %s/%s", epoch_number, total_epochs)
                break

        self.model.load_state_dict(best_state)
        self.model.eval()

    def _fit_with_gnn_backend(
        self,
        train_frame: pd.DataFrame,
        valid_frame: pd.DataFrame,
        x_train: np.ndarray,
        x_valid: np.ndarray,
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

        train_snapshots = build_graph_snapshots(
            frame=train_frame,
            feature_array=x_train,
            neighbor_map=self.neighbor_map,
            station_col=self.station_col,
            date_col=self.date_col,
            target_names=self.target_names,
            target_cols=target_cols,
            thresholds=thresholds,
            sample_weight=sample_weight,
        )
        valid_snapshots = build_graph_snapshots(
            frame=valid_frame,
            feature_array=x_valid,
            neighbor_map=self.neighbor_map,
            station_col=self.station_col,
            date_col=self.date_col,
            target_names=self.target_names,
            target_cols=target_cols,
            thresholds=thresholds,
        )

        graph_cfg = self._graph_cfg()
        hidden_dims = list(self.model_cfg.get("hidden_dims", [128, 64]))
        hidden_dim = int(graph_cfg.get("hidden_dim", hidden_dims[0] if hidden_dims else 128))
        num_layers = int(graph_cfg.get("num_layers", 2))
        dropout = float(graph_cfg.get("dropout", self.model_cfg.get("dropout", 0.1)))

        self.model = GraphMultiTaskTailNet(
            input_dim=x_train.shape[1],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            target_names=self.target_names,
            quantiles=self.quantiles,
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

        total_epochs = int(self.model_cfg.get("epochs", 25))
        LOGGER.info(
            "Starting graph torch-tail training: epochs=%s train_snapshots=%s valid_snapshots=%s",
            total_epochs,
            len(train_snapshots),
            len(valid_snapshots),
        )
        for epoch_idx in range(total_epochs):
            epoch_number = epoch_idx + 1
            epoch_start = time.perf_counter()
            self.model.train()
            epoch_train_loss = 0.0
            epoch_batch_count = 0
            for snapshot in train_snapshots:
                optimizer.zero_grad()
                loss = self._graph_snapshot_loss(snapshot, quantiles_tensor, quantile_weight_tensor)
                loss.backward()
                optimizer.step()
                epoch_train_loss += float(loss.detach().cpu())
                epoch_batch_count += 1

            valid_loss = self._graph_validation_loss(valid_snapshots, quantiles_tensor, quantile_weight_tensor)
            average_train_loss = epoch_train_loss / max(epoch_batch_count, 1)
            if valid_loss < best_valid:
                best_valid = valid_loss
                best_state = copy.deepcopy(self.model.state_dict())
                wait = 0
            else:
                wait += 1
            LOGGER.info(
                "Graph torch-tail epoch %s/%s: train_loss=%.4f valid_loss=%.4f best_valid=%.4f wait=%s elapsed=%.1fs",
                epoch_number,
                total_epochs,
                average_train_loss,
                valid_loss,
                best_valid,
                wait,
                time.perf_counter() - epoch_start,
            )
            if wait >= patience:
                LOGGER.info("Graph torch-tail early stopping at epoch %s/%s", epoch_number, total_epochs)
                break

        self.model.load_state_dict(best_state)
        self.model.eval()

    def _dense_validation_loss(
        self,
        valid_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        quantiles: torch.Tensor,
        quantile_weights: torch.Tensor,
    ) -> float:
        assert self.model is not None
        features, targets, events = valid_tensors
        features = features.to(self.device)
        targets = targets.to(self.device)
        events = events.to(self.device)
        with torch.no_grad():
            outputs = self.model(features)
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
        snapshot: GraphSnapshot,
        quantiles: torch.Tensor,
        quantile_weights: torch.Tensor,
    ) -> torch.Tensor:
        assert self.model is not None
        assert snapshot.targets is not None
        assert snapshot.events is not None
        assert snapshot.row_weight is not None

        features = torch.tensor(snapshot.features, dtype=torch.float32, device=self.device)
        adjacency = torch.tensor(snapshot.adjacency, dtype=torch.float32, device=self.device)
        targets = torch.tensor(snapshot.targets, dtype=torch.float32, device=self.device)
        events = torch.tensor(snapshot.events, dtype=torch.float32, device=self.device)
        row_weight = torch.tensor(snapshot.row_weight, dtype=torch.float32, device=self.device)
        outputs = self.model(features, adjacency)

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
        snapshots: list[GraphSnapshot],
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

    def predict(self, frame: pd.DataFrame, feature_cols: list[str]) -> dict[str, dict[str, object]]:
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")

        if not self.feature_cols_:
            self.feature_cols_ = self._resolve_feature_cols(feature_cols)
        x = self._transform_features(frame, fit=False)

        if self.graph_backend == "gnn":
            return self._predict_with_gnn_backend(frame, x)

        predictions = self._empty_predictions(len(frame))
        batch_size = int(self.model_cfg.get("predict_batch_size", self.model_cfg.get("batch_size", 256)))
        with torch.no_grad():
            for start in range(0, len(frame), batch_size):
                stop = min(start + batch_size, len(frame))
                outputs = self.model(torch.tensor(x[start:stop], dtype=torch.float32, device=self.device))
                for target_name in self.target_names:
                    quantile_matrix = outputs[target_name]["quantiles"].detach().cpu().numpy()
                    predictions[target_name]["point"][start:stop] = quantile_matrix[:, self.point_quantile_index]
                    for idx, quantile in enumerate(self.quantiles):
                        predictions[target_name]["quantiles"][quantile][start:stop] = quantile_matrix[:, idx]
                    predictions[target_name]["exceedance_probability"][start:stop] = (
                        torch.sigmoid(outputs[target_name]["logit"]).detach().cpu().numpy()
                    )
        return predictions

    def _predict_with_gnn_backend(self, frame: pd.DataFrame, feature_array: np.ndarray) -> dict[str, dict[str, object]]:
        assert self.model is not None

        snapshots = build_graph_snapshots(
            frame=frame,
            feature_array=feature_array,
            neighbor_map=self.neighbor_map,
            station_col=self.station_col,
            date_col=self.date_col,
        )
        predictions = self._empty_predictions(len(frame))

        with torch.no_grad():
            for snapshot in snapshots:
                outputs = self.model(
                    torch.tensor(snapshot.features, dtype=torch.float32, device=self.device),
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

