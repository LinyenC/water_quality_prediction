from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hydrotail.config import load_config
from hydrotail.data import SequenceSamples, build_model_frame, build_sequence_samples, load_datasets
from hydrotail.metrics import classification_metrics, evaluate_tail_subset, quantile_metrics, regression_metrics
from hydrotail.models import GBDTTailModel, LinearTailModel, SequenceTailModel, TorchTailModel
from hydrotail.splits import assign_splits


SEQUENCE_MODEL_NAMES = {"seq_tcn_tail", "seq_transformer_tail"}
ANALYSIS_GROUP_NAMES = [
    "train_station_early",
    "train_station_late",
    "test_station_early",
    "test_station_late",
]


def _compute_thresholds(train_frame: pd.DataFrame, target_cols: dict[str, str], tail_cfg: dict[str, object]) -> dict[str, float]:
    exceedance_cfg = tail_cfg.get("exceedance_quantiles", {})
    thresholds: dict[str, float] = {}
    for target_name, target_col in target_cols.items():
        quantile = float(exceedance_cfg.get(target_name, 0.9))
        observed = train_frame[target_col].dropna()
        thresholds[target_name] = float(observed.quantile(quantile)) if not observed.empty else float("nan")
    return thresholds


def _compute_sample_weights(
    frame: pd.DataFrame,
    target_cols: dict[str, str],
    thresholds: dict[str, float],
    multiplier: float,
) -> np.ndarray:
    event_columns: list[np.ndarray] = []
    for target_name, target_col in target_cols.items():
        values = frame[target_col].to_numpy(dtype=float)
        threshold = thresholds.get(target_name, float("nan"))
        if np.isfinite(threshold):
            event_columns.append(np.where(np.isfinite(values), (values >= threshold).astype(float), 0.0))
        else:
            event_columns.append(np.zeros(len(frame), dtype=float))
    event_matrix = np.column_stack(event_columns)
    return 1.0 + multiplier * np.max(event_matrix, axis=1)


def _instantiate_model(model_name: str, config: dict[str, object], quantiles: list[float], quantile_weights: list[float], seed: int):
    model_cfg = config["models"][model_name]
    if model_name == "linear_tail":
        return LinearTailModel(quantiles=quantiles, model_cfg=model_cfg, random_state=seed)
    if model_name == "gbdt_tail":
        return GBDTTailModel(quantiles=quantiles, model_cfg=model_cfg, random_state=seed)
    if model_name == "torch_tail":
        full_cfg = dict(model_cfg)
        full_cfg["tail_weight_multiplier"] = float(config["tail"].get("tail_weight_multiplier", 4.0))
        return TorchTailModel(
            quantiles=quantiles,
            quantile_weights=quantile_weights,
            model_cfg=full_cfg,
            random_state=seed,
        )
    if model_name == "seq_tcn_tail":
        full_cfg = dict(model_cfg)
        full_cfg["tail_weight_multiplier"] = float(config["tail"].get("tail_weight_multiplier", 4.0))
        return SequenceTailModel(
            encoder_type="tcn",
            lookback_window=int(config.get("sequence", {}).get("lookback_window", 30)),
            quantiles=quantiles,
            quantile_weights=quantile_weights,
            model_cfg=full_cfg,
            random_state=seed,
        )
    if model_name == "seq_transformer_tail":
        full_cfg = dict(model_cfg)
        full_cfg["tail_weight_multiplier"] = float(config["tail"].get("tail_weight_multiplier", 4.0))
        return SequenceTailModel(
            encoder_type="transformer",
            lookback_window=int(config.get("sequence", {}).get("lookback_window", 30)),
            quantiles=quantiles,
            quantile_weights=quantile_weights,
            model_cfg=full_cfg,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _empty_target_metrics(quantiles: list[float]) -> dict[str, float]:
    metrics = {
        "observed_count": 0,
        "mae": float("nan"),
        "rmse": float("nan"),
        "r2": float("nan"),
        "interval_coverage": float("nan"),
        "interval_width": float("nan"),
        "f1": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "auc": float("nan"),
        "tail_count": 0,
    }
    for quantile in quantiles:
        metrics[f"pinball_q{quantile}"] = float("nan")
    return metrics


def _evaluate_predictions(
    frame: pd.DataFrame,
    predictions: dict[str, dict[str, object]],
    target_cols: dict[str, str],
    thresholds: dict[str, float],
    quantiles: list[float],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for target_name, target_col in target_cols.items():
        y_true = frame[target_col].to_numpy(dtype=float)
        valid_mask = np.isfinite(y_true)
        if not np.any(valid_mask):
            metrics[target_name] = _empty_target_metrics(quantiles)
            continue

        y_true_valid = y_true[valid_mask]
        point_pred = np.asarray(predictions[target_name]["point"], dtype=float)[valid_mask]
        q_pred = {
            quantile: np.asarray(predictions[target_name]["quantiles"][quantile], dtype=float)[valid_mask]
            for quantile in quantiles
        }
        event_prob = np.asarray(predictions[target_name]["exceedance_probability"], dtype=float)[valid_mask]
        threshold = thresholds.get(target_name, float("nan"))

        target_metrics = {"observed_count": int(valid_mask.sum())}
        target_metrics.update(regression_metrics(y_true_valid, point_pred))
        target_metrics.update(quantile_metrics(y_true_valid, q_pred, quantiles))

        if np.isfinite(threshold):
            event_true = (y_true_valid >= threshold).astype(int)
            target_metrics.update(classification_metrics(event_true, event_prob))
            tail_metrics = evaluate_tail_subset(y_true_valid, point_pred, threshold)
            for key, value in tail_metrics.items():
                target_metrics[f"tail_{key}"] = value
        else:
            target_metrics.update(
                {
                    "f1": float("nan"),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "auc": float("nan"),
                    "tail_count": 0,
                }
            )

        metrics[target_name] = target_metrics
    return metrics


def _save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _save_predictions(
    frame: pd.DataFrame,
    predictions: dict[str, dict[str, object]],
    target_cols: dict[str, str],
    quantiles: list[float],
    path: Path,
) -> None:
    output = frame.copy()
    for target_name, target_col in target_cols.items():
        output[f"pred_{target_name}"] = predictions[target_name]["point"]
        output[f"prob_{target_name}_exceed"] = predictions[target_name]["exceedance_probability"]
        for quantile in quantiles:
            suffix = str(quantile).replace(".", "_")
            output[f"pred_{target_name}_q{suffix}"] = predictions[target_name]["quantiles"][quantile]
        output[f"actual_{target_name}"] = output[target_col]
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)


def _collect_eval_frame_groups(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    groups = {
        "train": frame.loc[frame["split"] == "train"].copy(),
        "valid": frame.loc[frame["split"] == "valid"].copy(),
        "test": frame.loc[frame["split"] == "test"].copy(),
    }
    for group_name in ANALYSIS_GROUP_NAMES:
        group_frame = frame.loc[frame["analysis_group"] == group_name].copy()
        if not group_frame.empty:
            groups[group_name] = group_frame
    return groups


def _collect_eval_sequence_groups(sequence_all: SequenceSamples) -> dict[str, SequenceSamples]:
    groups = {
        "train": sequence_all.subset("train", column="split"),
        "valid": sequence_all.subset("valid", column="split"),
        "test": sequence_all.subset("test", column="split"),
    }
    for group_name in ANALYSIS_GROUP_NAMES:
        group_bundle = sequence_all.subset(group_name, column="analysis_group")
        if len(group_bundle) > 0:
            groups[group_name] = group_bundle
    return groups


def run_experiment(config_path: str) -> None:
    config = load_config(config_path)
    seed = int(config["splits"].get("seed", 42))
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dynamic_df, static_df = load_datasets(config)
    quantiles = [float(value) for value in config["tail"]["quantiles"]]
    quantile_weights = [float(value) for value in config["tail"]["quantile_weights"]]
    model_names = list(config["run"]["models"])
    horizons = list(config["features"].get("horizons", [1]))
    station_col = config["data"]["station_col"]
    date_col = config["data"]["date_col"]

    for horizon in horizons:
        frame, feature_cols, target_cols, edge_df = build_model_frame(dynamic_df, static_df, config, horizon=int(horizon))
        split_bundle = assign_splits(
            frame,
            station_col=station_col,
            date_col=date_col,
            split_cfg=config["splits"],
        )
        frame = split_bundle.frame
        horizon_dir = output_dir / f"horizon_{horizon}"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        _save_json(horizon_dir / "split_metadata.json", split_bundle.metadata)
        if not edge_df.empty:
            edge_df.to_csv(horizon_dir / "graph_edges.csv", index=False)

        eval_frame_groups = _collect_eval_frame_groups(frame)
        train_frame = eval_frame_groups["train"]
        valid_frame = eval_frame_groups["valid"]
        test_frame = eval_frame_groups["test"]
        if train_frame.empty or valid_frame.empty or test_frame.empty:
            raise RuntimeError("One or more tabular data splits are empty. Adjust station fractions or time fractions in the config.")

        thresholds = _compute_thresholds(train_frame, target_cols, config["tail"])
        tabular_sample_weight = _compute_sample_weights(
            train_frame,
            target_cols=target_cols,
            thresholds=thresholds,
            multiplier=float(config["tail"].get("tail_weight_multiplier", 4.0)),
        )
        _save_json(horizon_dir / "thresholds.json", thresholds)

        sequence_groups: dict[str, SequenceSamples] | None = None
        if any(model_name in SEQUENCE_MODEL_NAMES for model_name in model_names):
            sequence_all = build_sequence_samples(frame, config, target_cols)
            sequence_groups = _collect_eval_sequence_groups(sequence_all)
            for split_name in ("train", "valid", "test"):
                if len(sequence_groups[split_name]) == 0:
                    raise RuntimeError(f"No sequence samples were created for split `{split_name}`.")
            _save_json(
                horizon_dir / "sequence_metadata.json",
                {
                    "lookback_window": int(config.get("sequence", {}).get("lookback_window", 30)),
                    "sample_counts": {group_name: len(bundle) for group_name, bundle in sequence_groups.items()},
                    "sequence_feature_count": len(sequence_all.sequence_feature_cols),
                    "static_feature_count": len(sequence_all.static_feature_cols),
                },
            )

        for model_name in model_names:
            model_dir = horizon_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            model = _instantiate_model(model_name, config, quantiles, quantile_weights, seed)

            if getattr(model, "input_mode", "tabular") == "sequence":
                assert sequence_groups is not None
                sequence_train = sequence_groups["train"]
                sequence_valid = sequence_groups["valid"]
                sequence_sample_weight = _compute_sample_weights(
                    sequence_train.frame,
                    target_cols=target_cols,
                    thresholds=thresholds,
                    multiplier=float(config["tail"].get("tail_weight_multiplier", 4.0)),
                )
                model.fit(sequence_train, sequence_valid, target_cols, thresholds, sample_weight=sequence_sample_weight)
                for group_name, bundle in sequence_groups.items():
                    if len(bundle) == 0:
                        continue
                    predictions = model.predict(bundle)
                    metrics = _evaluate_predictions(bundle.frame, predictions, target_cols, thresholds, quantiles)
                    _save_json(model_dir / f"{group_name}_metrics.json", metrics)
                    _save_predictions(bundle.frame, predictions, target_cols, quantiles, model_dir / f"{group_name}_predictions.csv")
            elif model_name == "torch_tail":
                model.fit(
                    train_frame,
                    valid_frame,
                    feature_cols,
                    target_cols,
                    thresholds,
                    sample_weight=tabular_sample_weight,
                    edge_df=edge_df,
                    station_col=station_col,
                    date_col=date_col,
                )
                for group_name, group_frame in eval_frame_groups.items():
                    if group_frame.empty:
                        continue
                    predictions = model.predict(group_frame, feature_cols)
                    metrics = _evaluate_predictions(group_frame, predictions, target_cols, thresholds, quantiles)
                    _save_json(model_dir / f"{group_name}_metrics.json", metrics)
                    _save_predictions(group_frame, predictions, target_cols, quantiles, model_dir / f"{group_name}_predictions.csv")
            else:
                train_plus_valid = pd.concat([train_frame, valid_frame], axis=0, ignore_index=True)
                joined_weight = np.concatenate([tabular_sample_weight, np.ones(len(valid_frame), dtype=float)])
                model.fit(train_plus_valid, feature_cols, target_cols, thresholds, sample_weight=joined_weight)
                for group_name, group_frame in eval_frame_groups.items():
                    if group_frame.empty:
                        continue
                    predictions = model.predict(group_frame, feature_cols)
                    metrics = _evaluate_predictions(group_frame, predictions, target_cols, thresholds, quantiles)
                    _save_json(model_dir / f"{group_name}_metrics.json", metrics)
                    _save_predictions(group_frame, predictions, target_cols, quantiles, model_dir / f"{group_name}_predictions.csv")

            model.save(model_dir / "model.bin")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HydroTail experiments.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
