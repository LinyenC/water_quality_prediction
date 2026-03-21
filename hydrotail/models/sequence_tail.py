from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from hydrotail.data import SequenceSamples


def _quantile_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
    quantile_weights: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if valid_mask is None:
        valid_mask = torch.isfinite(target)
    if not torch.any(valid_mask):
        return prediction.sum() * 0.0

    prediction = prediction[valid_mask]
    target = target[valid_mask]
    errors = target.unsqueeze(-1) - prediction
    loss = torch.maximum(quantiles * errors, (quantiles - 1.0) * errors)
    return (loss * quantile_weights).mean()


def _binary_event_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if valid_mask is None:
        valid_mask = torch.ones_like(targets, dtype=torch.bool)
    if not torch.any(valid_mask):
        return logits.sum() * 0.0

    logits = logits[valid_mask]
    targets = targets[valid_mask]
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)


def _boundary_loss(prediction: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    if valid_mask is None:
        valid_mask = torch.ones(prediction.size(0), dtype=torch.bool, device=prediction.device)
    if not torch.any(valid_mask):
        return prediction.sum() * 0.0
    prediction = prediction[valid_mask]
    return torch.relu(-prediction).mean()


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


class SequenceTailNet(nn.Module):
    """Shared tail-aware network with either a TCN or Transformer temporal encoder."""

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
        elif encoder_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=int(model_cfg.get("num_heads", 4)),
                dim_feedforward=hidden_dim * int(model_cfg.get("ff_multiplier", 2)),
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(model_cfg.get("num_layers", 2)))
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
        self.quantile_heads = nn.ModuleDict({name: nn.Linear(hidden_dim, len(quantiles)) for name in target_names})
        self.event_heads = nn.ModuleDict({name: nn.Linear(hidden_dim, 1) for name in target_names})

    def forward(self, sequence_values: torch.Tensor, sequence_masks: torch.Tensor, static_values: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        # The model sees both imputed values and explicit masks, so irregular gaps stay visible.
        seq_input = torch.cat([sequence_values, sequence_masks], dim=-1)
        hidden = self.input_projection(seq_input) + self.position_embedding[:, : seq_input.size(1), :]

        if self.encoder_type == "tcn":
            hidden = hidden.transpose(1, 2)
            for block in self.tcn_blocks:
                hidden = block(hidden)
            hidden = hidden.transpose(1, 2)
        else:
            hidden = self.transformer(hidden)

        timestep_mask = (sequence_masks.sum(dim=-1) > 0).float()
        pooled = (hidden * timestep_mask.unsqueeze(-1)).sum(dim=1) / timestep_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        last_hidden = hidden[:, -1, :]
        joint = torch.cat([pooled, last_hidden], dim=-1)

        if self.static_mlp is not None and static_values.size(-1) > 0:
            joint = torch.cat([joint, self.static_mlp(static_values)], dim=-1)

        fused = self.fusion(joint)
        outputs: dict[str, dict[str, torch.Tensor]] = {}
        for target_name in self.quantile_heads:
            quantiles, _ = torch.sort(self.quantile_heads[target_name](fused), dim=-1)
            outputs[target_name] = {
                "quantiles": quantiles,
                "logit": self.event_heads[target_name](fused).squeeze(-1),
            }
        return outputs


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
        self.target_names: list[str] = []
        self.dynamic_medians: np.ndarray | None = None
        self.static_imputer = SimpleImputer(strategy="median")
        self.model: SequenceTailNet | None = None

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
        seq_values = self._impute_dynamic(bundle.sequence_values, fit=fit)
        seq_masks = bundle.sequence_masks.astype(np.float32)
        if bundle.static_values.shape[1] == 0:
            static_values = bundle.static_values.astype(np.float32)
        else:
            if fit:
                static_values = self.static_imputer.fit_transform(bundle.static_values)
            else:
                static_values = self.static_imputer.transform(bundle.static_values)
            static_values = static_values.astype(np.float32)
        return seq_values, seq_masks, static_values

    def fit(
        self,
        train_bundle: SequenceSamples,
        valid_bundle: SequenceSamples,
        target_cols: dict[str, str],
        thresholds: dict[str, float],
        sample_weight: np.ndarray | None = None,
    ) -> None:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.target_names = list(target_cols.keys())

        train_values, train_masks, train_static = self._prepare_bundle(train_bundle, fit=True)
        valid_values, valid_masks, valid_static = self._prepare_bundle(valid_bundle, fit=False)

        y_train = np.column_stack([train_bundle.frame[target_cols[name]].to_numpy(dtype=float) for name in self.target_names]).astype(np.float32)
        y_valid = np.column_stack([valid_bundle.frame[target_cols[name]].to_numpy(dtype=float) for name in self.target_names]).astype(np.float32)
        e_train = np.column_stack(
            [
                np.where(np.isfinite(train_bundle.frame[target_cols[name]].to_numpy(dtype=float)), (train_bundle.frame[target_cols[name]].to_numpy(dtype=float) >= thresholds[name]).astype(float), 0.0)
                for name in self.target_names
            ]
        ).astype(np.float32)
        e_valid = np.column_stack(
            [
                np.where(np.isfinite(valid_bundle.frame[target_cols[name]].to_numpy(dtype=float)), (valid_bundle.frame[target_cols[name]].to_numpy(dtype=float) >= thresholds[name]).astype(float), 0.0)
                for name in self.target_names
            ]
        ).astype(np.float32)

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
        loader = DataLoader(train_dataset, batch_size=int(self.model_cfg.get("batch_size", 128)), sampler=sampler)

        self.model = SequenceTailNet(
            encoder_type=self.encoder_type,
            lookback_window=self.lookback_window,
            dynamic_dim=train_values.shape[-1],
            static_dim=train_static.shape[-1],
            hidden_dim=int(self.model_cfg.get("hidden_dim", 64)),
            target_names=self.target_names,
            quantiles=self.quantiles,
            model_cfg=self.model_cfg,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.model_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(self.model_cfg.get("weight_decay", 1e-4)),
        )

        q_tensor = torch.tensor(self.quantiles, dtype=torch.float32)
        qw_tensor = torch.tensor(self.quantile_weights, dtype=torch.float32)
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
                optimizer.zero_grad()
                outputs = self.model(batch_values, batch_masks, batch_static)
                total_loss = torch.tensor(0.0, dtype=torch.float32)
                for target_idx, target_name in enumerate(self.target_names):
                    valid_mask = torch.isfinite(batch_targets[:, target_idx])
                    quantile_pred = outputs[target_name]["quantiles"]
                    event_logit = outputs[target_name]["logit"]
                    q_loss = _quantile_loss(
                        quantile_pred,
                        batch_targets[:, target_idx],
                        q_tensor,
                        qw_tensor,
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

            valid_loss = self._validation_loss(valid_tensors, q_tensor, qw_tensor)
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

    def _validation_loss(
        self,
        valid_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        quantiles: torch.Tensor,
        quantile_weights: torch.Tensor,
    ) -> float:
        assert self.model is not None
        values, masks, static_values, targets, events = valid_tensors
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

    def predict(self, bundle: SequenceSamples) -> dict[str, dict[str, object]]:
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")

        sequence_values, sequence_masks, static_values = self._prepare_bundle(bundle, fit=False)
        with torch.no_grad():
            outputs = self.model(
                torch.tensor(sequence_values),
                torch.tensor(sequence_masks),
                torch.tensor(static_values),
            )

        predictions: dict[str, dict[str, object]] = {}
        for target_name in self.target_names:
            quantile_matrix = outputs[target_name]["quantiles"].numpy()
            predictions[target_name] = {
                "point": quantile_matrix[:, self.point_quantile_index],
                "quantiles": {quantile: quantile_matrix[:, idx] for idx, quantile in enumerate(self.quantiles)},
                "exceedance_probability": torch.sigmoid(outputs[target_name]["logit"]).numpy(),
            }
        return predictions

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            torch.save(self, handle)
