from __future__ import annotations

import copy
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from hydrotail.data import SequenceSamples
from hydrotail.models.graph_backends import build_neighbor_map
from hydrotail.models.sequence_tail import (
    SequenceEncoder,
    _binary_event_loss,
    _boundary_loss,
    _quantile_loss,
    _resolve_device,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class StationMemoryBank:
    """Station-level donor memory built from source-basin sequence encodings."""

    station_ids: list[str]
    keys: torch.Tensor
    values: torch.Tensor


class RetrievalPrototypeTailNet(nn.Module):
    """Sequence encoder with donor retrieval, prototype attention, and tail-aware heads."""

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
        self.hidden_dim = hidden_dim
        self.top_k_donors = int(model_cfg.get("top_k_donors", 16))
        self.retrieval_temperature = float(model_cfg.get("retrieval_temperature", 0.2))
        self.use_prototypes = bool(model_cfg.get("use_prototypes", True))
        self.num_prototypes = int(model_cfg.get("num_prototypes", 8))
        dropout = float(model_cfg.get("dropout", 0.1))

        self.encoder = SequenceEncoder(
            encoder_type=encoder_type,
            lookback_window=lookback_window,
            dynamic_dim=dynamic_dim,
            static_dim=static_dim,
            hidden_dim=hidden_dim,
            model_cfg=model_cfg,
        )
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        if self.use_prototypes and self.num_prototypes > 0:
            self.prototype_tokens = nn.Parameter(torch.randn(self.num_prototypes, hidden_dim) * 0.02)
        else:
            self.register_parameter("prototype_tokens", None)

        transfer_input_dim = hidden_dim * (2 if self.use_prototypes and self.num_prototypes > 0 else 1)
        self.transfer_proj = nn.Sequential(
            nn.Linear(transfer_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gate_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.quantile_heads = nn.ModuleDict({name: nn.Linear(hidden_dim, len(quantiles)) for name in target_names})
        self.event_heads = nn.ModuleDict({name: nn.Linear(hidden_dim, 1) for name in target_names})

    def encode(
        self,
        sequence_values: torch.Tensor,
        sequence_masks: torch.Tensor,
        static_values: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(sequence_values, sequence_masks, static_values)

    def build_memory(self, hidden_bank: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        keys = nn.functional.normalize(self.key_proj(hidden_bank), dim=-1)
        values = self.value_proj(hidden_bank)
        return keys, values

    def _retrieve_from_bank(
        self,
        query: torch.Tensor,
        bank_keys: torch.Tensor,
        bank_values: torch.Tensor,
        candidate_indices: list[torch.Tensor],
    ) -> torch.Tensor:
        donor_vectors: list[torch.Tensor] = []
        for row_idx, candidate_idx in enumerate(candidate_indices):
            if candidate_idx.numel() == 0:
                candidate_key = bank_keys
                candidate_value = bank_values
            else:
                candidate_key = bank_keys.index_select(0, candidate_idx)
                candidate_value = bank_values.index_select(0, candidate_idx)

            similarity = torch.matmul(candidate_key, query[row_idx].unsqueeze(-1)).squeeze(-1)
            top_k = min(self.top_k_donors, similarity.size(0))
            if top_k <= 0:
                donor_vectors.append(torch.zeros(self.hidden_dim, dtype=query.dtype, device=query.device))
                continue

            top_scores, top_positions = torch.topk(similarity, k=top_k)
            weights = torch.softmax(top_scores / max(self.retrieval_temperature, 1e-6), dim=0)
            selected_values = candidate_value.index_select(0, top_positions)
            donor_vectors.append((weights.unsqueeze(-1) * selected_values).sum(dim=0))
        return torch.stack(donor_vectors, dim=0)

    def _prototype_context(self, query: torch.Tensor) -> torch.Tensor:
        if self.prototype_tokens is None:
            return torch.zeros_like(query)

        prototypes = nn.functional.normalize(self.prototype_tokens, dim=-1)
        similarity = torch.matmul(query, prototypes.transpose(0, 1))
        weights = torch.softmax(similarity, dim=-1)
        return torch.matmul(weights, self.prototype_tokens)

    def forward(
        self,
        sequence_values: torch.Tensor,
        sequence_masks: torch.Tensor,
        static_values: torch.Tensor,
        bank_keys: torch.Tensor,
        bank_values: torch.Tensor,
        candidate_indices: list[torch.Tensor],
    ) -> dict[str, dict[str, torch.Tensor]]:
        local_hidden = self.encode(sequence_values, sequence_masks, static_values)
        query = nn.functional.normalize(self.query_proj(local_hidden), dim=-1)
        donor_hidden = self._retrieve_from_bank(query, bank_keys, bank_values, candidate_indices)
        proto_hidden = self._prototype_context(query)

        if self.prototype_tokens is None:
            transfer_hidden = self.transfer_proj(donor_hidden)
        else:
            transfer_hidden = self.transfer_proj(torch.cat([donor_hidden, proto_hidden], dim=-1))

        gate = torch.sigmoid(self.gate_proj(torch.cat([local_hidden, donor_hidden, proto_hidden], dim=-1)))
        fused_hidden = gate * local_hidden + (1.0 - gate) * transfer_hidden
        fused_hidden = self.output_proj(fused_hidden)

        outputs: dict[str, dict[str, torch.Tensor]] = {}
        for target_name in self.quantile_heads:
            quantiles, _ = torch.sort(self.quantile_heads[target_name](fused_hidden), dim=-1)
            outputs[target_name] = {
                "quantiles": quantiles,
                "logit": self.event_heads[target_name](fused_hidden).squeeze(-1),
            }
        return outputs


class RetrievalPrototypeTailModel:
    input_mode = "sequence"

    def __init__(
        self,
        lookback_window: int,
        quantiles: list[float],
        quantile_weights: list[float],
        model_cfg: dict[str, object],
        random_state: int = 42,
    ) -> None:
        self.lookback_window = lookback_window
        self.quantiles = quantiles
        self.point_quantile_index = min(range(len(quantiles)), key=lambda idx: abs(quantiles[idx] - 0.5))
        self.quantile_weights = quantile_weights
        self.model_cfg = model_cfg
        self.random_state = random_state
        self.device = _resolve_device(self.model_cfg.get("device", "auto"))
        self.encoder_type = str(self.model_cfg.get("encoder_type", "tcn")).lower()
        if self.encoder_type not in {"tcn", "transformer"}:
            raise ValueError(f"Unsupported encoder_type: {self.encoder_type}")

        self.target_names: list[str] = []
        self.dynamic_medians: np.ndarray | None = None
        self.static_feature_cols_: list[str] = []
        self.static_imputer = SimpleImputer(strategy="median")
        self.model: RetrievalPrototypeTailNet | None = None
        self.station_col = "station_id"
        self.date_col = "date"
        self.use_graph_restriction = bool(self.model_cfg.get("use_graph_restriction", True))
        self.neighbor_map: dict[str, set[str]] = {}
        self.memory_bank_: StationMemoryBank | None = None
        self.bank_index_map_: dict[str, int] = {}
        self.candidate_cache_: dict[str, torch.Tensor] = {}
        self.all_candidate_indices_: torch.Tensor | None = None

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

    def _compute_station_memory(
        self,
        frame: pd.DataFrame,
        sequence_values: np.ndarray,
        sequence_masks: np.ndarray,
        static_values: np.ndarray,
    ) -> StationMemoryBank:
        assert self.model is not None

        batch_size = int(self.model_cfg.get("memory_batch_size", self.model_cfg.get("batch_size", 256)))
        station_ids = frame[self.station_col].astype(str).to_numpy()
        unique_station_ids, inverse_index = np.unique(station_ids, return_inverse=True)
        station_hidden = np.zeros((len(unique_station_ids), self.model.hidden_dim), dtype=np.float32)
        counts = np.zeros(len(unique_station_ids), dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(frame), batch_size):
                stop = min(start + batch_size, len(frame))
                hidden = self.model.encode(
                    torch.tensor(sequence_values[start:stop], dtype=torch.float32, device=self.device),
                    torch.tensor(sequence_masks[start:stop], dtype=torch.float32, device=self.device),
                    torch.tensor(static_values[start:stop], dtype=torch.float32, device=self.device),
                )
                hidden_numpy = hidden.detach().cpu().numpy().astype(np.float32, copy=False)
                np.add.at(station_hidden, inverse_index[start:stop], hidden_numpy)
                np.add.at(counts, inverse_index[start:stop], 1.0)
        station_hidden = station_hidden / counts[:, None].clip(min=1.0)

        hidden_tensor = torch.tensor(station_hidden, dtype=torch.float32, device=self.device)
        keys, values = self.model.build_memory(hidden_tensor)
        return StationMemoryBank(
            station_ids=unique_station_ids.tolist(),
            keys=keys.detach(),
            values=values.detach(),
        )

    def _reset_candidate_cache(self, preload_station_ids: Iterable[str] | None = None) -> None:
        if self.memory_bank_ is None:
            raise RuntimeError("Memory bank has not been built.")

        self.all_candidate_indices_ = torch.arange(
            len(self.memory_bank_.station_ids),
            dtype=torch.long,
            device=self.device,
        )
        self.candidate_cache_ = {}
        if preload_station_ids is None:
            return
        for station_id in preload_station_ids:
            self._candidate_tensor(str(station_id))

    def _candidate_tensor(self, station_id: str) -> torch.Tensor:
        cached = self.candidate_cache_.get(station_id)
        if cached is not None:
            return cached
        if self.all_candidate_indices_ is None:
            raise RuntimeError("Candidate cache has not been initialised.")

        own_index = self.bank_index_map_.get(station_id)
        if self.use_graph_restriction and self.neighbor_map:
            neighbors = self.neighbor_map.get(station_id, set())
            neighbor_indices = [
                self.bank_index_map_[neighbor_id]
                for neighbor_id in neighbors
                if neighbor_id in self.bank_index_map_
            ]
            if neighbor_indices:
                candidate = torch.tensor(neighbor_indices, dtype=torch.long, device=self.device)
            else:
                candidate = self.all_candidate_indices_
        else:
            candidate = self.all_candidate_indices_

        if own_index is not None and candidate.numel() > 0:
            filtered_candidate = candidate[candidate != own_index]
            if filtered_candidate.numel() > 0:
                candidate = filtered_candidate
        if candidate.numel() == 0:
            candidate = self.all_candidate_indices_
        self.candidate_cache_[station_id] = candidate
        return candidate

    def _candidate_indices(self, station_ids: list[str] | np.ndarray) -> list[torch.Tensor]:
        return [self._candidate_tensor(str(station_id)) for station_id in station_ids]

    def _validation_loss(
        self,
        bundle: SequenceSamples,
        sequence_values: np.ndarray,
        sequence_masks: np.ndarray,
        static_values: np.ndarray,
        station_ids: np.ndarray,
        target_cols: dict[str, str],
        thresholds: dict[str, float],
        quantiles: torch.Tensor,
        quantile_weights: torch.Tensor,
    ) -> float:
        assert self.model is not None
        if self.memory_bank_ is None:
            raise RuntimeError("Memory bank has not been built.")

        batch_size = int(self.model_cfg.get("batch_size", 128))
        total_loss = 0.0
        batch_count = 0
        with torch.no_grad():
            for start in range(0, len(bundle.frame), batch_size):
                stop = min(start + batch_size, len(bundle.frame))
                batch_frame = bundle.frame.iloc[start:stop]
                outputs = self.model(
                    torch.tensor(sequence_values[start:stop], dtype=torch.float32, device=self.device),
                    torch.tensor(sequence_masks[start:stop], dtype=torch.float32, device=self.device),
                    torch.tensor(static_values[start:stop], dtype=torch.float32, device=self.device),
                    self.memory_bank_.keys,
                    self.memory_bank_.values,
                    self._candidate_indices(station_ids[start:stop]),
                )
                targets = torch.tensor(
                    np.column_stack([batch_frame[target_cols[name]].to_numpy(dtype=float) for name in self.target_names]),
                    dtype=torch.float32,
                    device=self.device,
                )
                events = torch.tensor(
                    np.column_stack(
                        [
                            np.where(
                                np.isfinite(batch_frame[target_cols[name]].to_numpy(dtype=float)),
                                (batch_frame[target_cols[name]].to_numpy(dtype=float) >= thresholds[name]).astype(float),
                                0.0,
                            )
                            for name in self.target_names
                        ]
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
                batch_loss = 0.0
                for target_idx, target_name in enumerate(self.target_names):
                    valid_mask = torch.isfinite(targets[:, target_idx])
                    q_loss = _quantile_loss(
                        outputs[target_name]["quantiles"],
                        targets[:, target_idx],
                        quantiles,
                        quantile_weights,
                        valid_mask=valid_mask,
                    )
                    event_loss = _binary_event_loss(
                        outputs[target_name]["logit"],
                        events[:, target_idx],
                        valid_mask=valid_mask,
                    )
                    boundary_loss = _boundary_loss(outputs[target_name]["quantiles"], valid_mask=valid_mask)
                    batch_loss += float(q_loss + event_loss + 0.1 * boundary_loss)
                total_loss += batch_loss
                batch_count += 1
        return total_loss / max(batch_count, 1)

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
        self.static_feature_cols_ = list(train_bundle.static_feature_cols)
        if self.use_graph_restriction and edge_df is not None and not edge_df.empty:
            self.neighbor_map = build_neighbor_map(edge_df)
        else:
            self.neighbor_map = {}

        train_values, train_masks, train_static = self._prepare_bundle(train_bundle, fit=True)
        valid_values, valid_masks, valid_static = self._prepare_bundle(valid_bundle, fit=False)
        train_station_ids = train_bundle.frame[self.station_col].astype(str).to_numpy()
        valid_station_ids = valid_bundle.frame[self.station_col].astype(str).to_numpy()
        warm_station_ids = np.unique(np.concatenate([train_station_ids, valid_station_ids]))

        y_train = np.column_stack(
            [train_bundle.frame[target_cols[name]].to_numpy(dtype=float) for name in self.target_names]
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

        if sample_weight is None:
            sample_weight = 1.0 + float(self.model_cfg.get("tail_weight_multiplier", 1.0)) * np.max(e_train, axis=1)

        self.model = RetrievalPrototypeTailNet(
            encoder_type=self.encoder_type,
            lookback_window=self.lookback_window,
            dynamic_dim=train_values.shape[-1],
            static_dim=train_static.shape[-1],
            hidden_dim=int(self.model_cfg.get("hidden_dim", 64)),
            target_names=self.target_names,
            quantiles=self.quantiles,
            model_cfg=self.model_cfg,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.model_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(self.model_cfg.get("weight_decay", 1e-4)),
        )
        quantiles_tensor = torch.tensor(self.quantiles, dtype=torch.float32, device=self.device)
        quantile_weight_tensor = torch.tensor(self.quantile_weights, dtype=torch.float32, device=self.device)

        train_dataset = TensorDataset(
            torch.tensor(train_values),
            torch.tensor(train_masks),
            torch.tensor(train_static),
            torch.tensor(y_train),
            torch.tensor(e_train),
            torch.arange(len(train_bundle.frame), dtype=torch.long),
        )
        sampler = WeightedRandomSampler(
            weights=torch.tensor(np.asarray(sample_weight, dtype=np.float32)),
            num_samples=len(sample_weight),
            replacement=True,
        )
        loader_kwargs: dict[str, object] = {
            "batch_size": int(self.model_cfg.get("batch_size", 128)),
            "sampler": sampler,
            "pin_memory": self.device.type == "cuda",
        }
        num_workers = int(self.model_cfg.get("num_workers", 0))
        if num_workers > 0:
            loader_kwargs["num_workers"] = num_workers
            loader_kwargs["persistent_workers"] = bool(self.model_cfg.get("persistent_workers", True))
            loader_kwargs["prefetch_factor"] = int(self.model_cfg.get("prefetch_factor", 2))
        loader = DataLoader(
            train_dataset,
            **loader_kwargs,
        )

        best_state = copy.deepcopy(self.model.state_dict())
        best_valid = float("inf")
        patience = int(self.model_cfg.get("patience", 5))
        wait = 0
        total_epochs = int(self.model_cfg.get("epochs", 20))
        memory_refresh_interval = max(int(self.model_cfg.get("memory_refresh_interval", 1)), 1)
        LOGGER.info(
            "Starting retrieval training: epochs=%s batch_size=%s train_samples=%s valid_samples=%s "
            "memory_refresh_interval=%s num_workers=%s",
            total_epochs,
            int(self.model_cfg.get("batch_size", 128)),
            len(train_bundle.frame),
            len(valid_bundle.frame),
            memory_refresh_interval,
            num_workers,
        )

        for epoch_idx in range(total_epochs):
            epoch_number = epoch_idx + 1
            epoch_start = time.perf_counter()
            refreshed_memory = self.memory_bank_ is None or epoch_idx % memory_refresh_interval == 0
            if refreshed_memory:
                memory_start = time.perf_counter()
                LOGGER.info("Retrieval epoch %s/%s: refreshing donor memory bank", epoch_number, total_epochs)
                self.memory_bank_ = self._compute_station_memory(
                    train_bundle.frame,
                    train_values,
                    train_masks,
                    train_static,
                )
                self.bank_index_map_ = {
                    station_id: idx for idx, station_id in enumerate(self.memory_bank_.station_ids)
                }
                self._reset_candidate_cache(warm_station_ids)
                LOGGER.info(
                    "Retrieval epoch %s/%s: donor memory ready stations=%s elapsed=%.1fs",
                    epoch_number,
                    total_epochs,
                    len(self.memory_bank_.station_ids),
                    time.perf_counter() - memory_start,
                )

            self.model.train()
            epoch_train_loss = 0.0
            epoch_batch_count = 0
            for batch_values, batch_masks, batch_static, batch_targets, batch_events, batch_indices in loader:
                batch_station_ids = train_station_ids[batch_indices.cpu().numpy()]
                optimizer.zero_grad()
                outputs = self.model(
                    batch_values.to(self.device, non_blocking=self.device.type == "cuda"),
                    batch_masks.to(self.device, non_blocking=self.device.type == "cuda"),
                    batch_static.to(self.device, non_blocking=self.device.type == "cuda"),
                    self.memory_bank_.keys,
                    self.memory_bank_.values,
                    self._candidate_indices(batch_station_ids),
                )
                batch_targets = batch_targets.to(self.device, non_blocking=self.device.type == "cuda")
                batch_events = batch_events.to(self.device, non_blocking=self.device.type == "cuda")
                total_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                for target_idx, target_name in enumerate(self.target_names):
                    valid_mask = torch.isfinite(batch_targets[:, target_idx])
                    q_loss = _quantile_loss(
                        outputs[target_name]["quantiles"],
                        batch_targets[:, target_idx],
                        quantiles_tensor,
                        quantile_weight_tensor,
                        valid_mask=valid_mask,
                    )
                    event_loss = _binary_event_loss(
                        outputs[target_name]["logit"],
                        batch_events[:, target_idx],
                        valid_mask=valid_mask,
                    )
                    boundary_loss = _boundary_loss(outputs[target_name]["quantiles"], valid_mask=valid_mask)
                    total_loss = total_loss + q_loss + event_loss + 0.1 * boundary_loss
                total_loss.backward()
                optimizer.step()
                epoch_train_loss += float(total_loss.detach().cpu())
                epoch_batch_count += 1

            valid_loss = self._validation_loss(
                valid_bundle,
                valid_values,
                valid_masks,
                valid_static,
                valid_station_ids,
                target_cols,
                thresholds,
                quantiles_tensor,
                quantile_weight_tensor,
            )
            average_train_loss = epoch_train_loss / max(epoch_batch_count, 1)
            if valid_loss < best_valid:
                best_valid = valid_loss
                best_state = copy.deepcopy(self.model.state_dict())
                wait = 0
            else:
                wait += 1
            LOGGER.info(
                "Retrieval epoch %s/%s completed: train_loss=%.4f valid_loss=%.4f best_valid=%.4f wait=%s "
                "refreshed_memory=%s elapsed=%.1fs",
                epoch_number,
                total_epochs,
                average_train_loss,
                valid_loss,
                best_valid,
                wait,
                refreshed_memory,
                time.perf_counter() - epoch_start,
            )
            if wait >= patience:
                LOGGER.info("Retrieval early stopping at epoch %s/%s", epoch_number, total_epochs)
                break

        self.model.load_state_dict(best_state)
        self.model.eval()
        LOGGER.info("Rebuilding donor memory bank for best retrieval checkpoint")
        self.memory_bank_ = self._compute_station_memory(
            train_bundle.frame,
            train_values,
            train_masks,
            train_static,
        )
        self.bank_index_map_ = {station_id: idx for idx, station_id in enumerate(self.memory_bank_.station_ids)}
        self._reset_candidate_cache(warm_station_ids)

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
        if self.memory_bank_ is None:
            raise RuntimeError("Memory bank is missing. Fit the model before prediction.")

        sequence_values, sequence_masks, static_values = self._prepare_bundle(bundle, fit=False)
        station_ids = bundle.frame[self.station_col].astype(str).to_numpy()
        predictions = self._empty_predictions(len(bundle.frame))
        batch_size = int(self.model_cfg.get("batch_size", 128))
        with torch.no_grad():
            for start in range(0, len(bundle.frame), batch_size):
                stop = min(start + batch_size, len(bundle.frame))
                outputs = self.model(
                    torch.tensor(sequence_values[start:stop], dtype=torch.float32, device=self.device),
                    torch.tensor(sequence_masks[start:stop], dtype=torch.float32, device=self.device),
                    torch.tensor(static_values[start:stop], dtype=torch.float32, device=self.device),
                    self.memory_bank_.keys,
                    self.memory_bank_.values,
                    self._candidate_indices(station_ids[start:stop]),
                )
                for target_name in self.target_names:
                    quantile_matrix = outputs[target_name]["quantiles"].detach().cpu().numpy()
                    predictions[target_name]["point"][start:stop] = quantile_matrix[:, self.point_quantile_index]
                    for idx, quantile in enumerate(self.quantiles):
                        predictions[target_name]["quantiles"][quantile][start:stop] = quantile_matrix[:, idx]
                    predictions[target_name]["exceedance_probability"][start:stop] = (
                        torch.sigmoid(outputs[target_name]["logit"]).detach().cpu().numpy()
                    )
        return predictions

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            torch.save(self, handle)
