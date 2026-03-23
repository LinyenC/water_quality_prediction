from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from hydrotail.config import load_config
from hydrotail.data import SequenceSamples, build_model_frame, build_sequence_samples, load_datasets
from hydrotail.metrics import classification_metrics, evaluate_tail_subset, quantile_metrics, regression_metrics
from hydrotail.models import GBDTTailModel, LinearTailModel, SequenceTailModel, TorchTailModel
from hydrotail.splits import assign_splits


LOGGER = logging.getLogger(__name__)
SEQUENCE_MODEL_NAMES = {"seq_tcn_tail", "seq_transformer_tail"}
ANALYSIS_GROUP_NAMES = [
    "train_station_early",
    "train_station_late",
    "test_station_early",
    "test_station_late",
]


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def _frame_station_count(frame: pd.DataFrame, station_col: str) -> int:
    if station_col not in frame.columns:
        return 0
    return int(frame[station_col].nunique())


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

        point_all = np.asarray(predictions[target_name]["point"], dtype=float)
        q_all = {
            quantile: np.asarray(predictions[target_name]["quantiles"][quantile], dtype=float)
            for quantile in quantiles
        }
        event_all = np.asarray(predictions[target_name]["exceedance_probability"], dtype=float)
        threshold = thresholds.get(target_name, float("nan"))

        target_metrics = _empty_target_metrics(quantiles)
        target_metrics["observed_count"] = int(valid_mask.sum())

        point_mask = valid_mask & np.isfinite(point_all)
        if np.any(point_mask):
            target_metrics.update(regression_metrics(y_true[point_mask], point_all[point_mask]))
            if np.isfinite(threshold):
                tail_metrics = evaluate_tail_subset(y_true[point_mask], point_all[point_mask], threshold)
                for key, value in tail_metrics.items():
                    target_metrics[f"tail_{key}"] = value

        quantile_mask = valid_mask.copy()
        for quantile in quantiles:
            quantile_mask &= np.isfinite(q_all[quantile])
        if np.any(quantile_mask):
            q_true = y_true[quantile_mask]
            q_pred = {quantile: q_all[quantile][quantile_mask] for quantile in quantiles}
            target_metrics.update(quantile_metrics(q_true, q_pred, quantiles))

        if np.isfinite(threshold):
            event_mask = valid_mask & np.isfinite(event_all)
            if np.any(event_mask):
                event_true = (y_true[event_mask] >= threshold).astype(int)
                target_metrics.update(classification_metrics(event_true, event_all[event_mask]))

        metrics[target_name] = target_metrics
    return metrics


def _save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _save_config_snapshot(config_path: Path, destination_dir: Path) -> None:
    """Save the original YAML config alongside experiment outputs for reproducibility."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, destination_dir / "experiment_config.yaml")


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
    resolved_config_path = Path(config_path).resolve()
    LOGGER.info("Loading config from %s", resolved_config_path)
    config = load_config(resolved_config_path)
    seed = int(config["splits"].get("seed", 42))
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_config_snapshot(resolved_config_path, output_dir)
    LOGGER.info("Experiment output directory: %s", output_dir)

    dataset_start = time.perf_counter()
    LOGGER.info("Loading datasets...")
    dynamic_df, static_df = load_datasets(config)
    LOGGER.info(
        "Loaded datasets in %.1fs: dynamic_rows=%s dynamic_stations=%s static_rows=%s static_stations=%s",
        time.perf_counter() - dataset_start,
        len(dynamic_df),
        _frame_station_count(dynamic_df, config["data"]["station_col"]),
        0 if static_df is None else len(static_df),
        0 if static_df is None else _frame_station_count(static_df, config["data"]["station_col"]),
    )

    quantiles = [float(value) for value in config["tail"]["quantiles"]]
    quantile_weights = [float(value) for value in config["tail"]["quantile_weights"]]
    model_names = list(config["run"]["models"])
    horizons = list(config["features"].get("horizons", [1]))
    station_col = config["data"]["station_col"]
    date_col = config["data"]["date_col"]
    LOGGER.info("Configured horizons=%s models=%s", horizons, model_names)

    for horizon in horizons:
        horizon_start = time.perf_counter()
        LOGGER.info("Building model frame for horizon=%s", horizon)
        frame, feature_cols, target_cols, edge_df = build_model_frame(dynamic_df, static_df, config, horizon=int(horizon))
        LOGGER.info(
            "Built model frame for horizon=%s: rows=%s stations=%s features=%s targets=%s edges=%s",
            horizon,
            len(frame),
            _frame_station_count(frame, station_col),
            len(feature_cols),
            list(target_cols.keys()),
            len(edge_df),
        )

        split_bundle = assign_splits(
            frame,
            station_col=station_col,
            date_col=date_col,
            split_cfg=config["splits"],
        )
        frame = split_bundle.frame
        LOGGER.info("Split counts for horizon=%s: %s", horizon, split_bundle.metadata.get("counts", {}))
        LOGGER.info("Analysis counts for horizon=%s: %s", horizon, split_bundle.metadata.get("analysis_counts", {}))

        horizon_dir = output_dir / f"horizon_{horizon}"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        _save_config_snapshot(resolved_config_path, horizon_dir)
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
        LOGGER.info("Thresholds for horizon=%s: %s", horizon, thresholds)
        tabular_sample_weight = _compute_sample_weights(
            train_frame,
            target_cols=target_cols,
            thresholds=thresholds,
            multiplier=float(config["tail"].get("tail_weight_multiplier", 4.0)),
        )
        _save_json(horizon_dir / "thresholds.json", thresholds)

        sequence_groups: dict[str, SequenceSamples] | None = None
        if any(model_name in SEQUENCE_MODEL_NAMES for model_name in model_names):
            LOGGER.info("Building sequence samples for horizon=%s", horizon)
            sequence_all = build_sequence_samples(frame, config, target_cols)
            sequence_groups = _collect_eval_sequence_groups(sequence_all)
            for split_name in ("train", "valid", "test"):
                if len(sequence_groups[split_name]) == 0:
                    raise RuntimeError(f"No sequence samples were created for split `{split_name}`.")
            sample_counts = {group_name: len(bundle) for group_name, bundle in sequence_groups.items()}
            LOGGER.info("Sequence sample counts for horizon=%s: %s", horizon, sample_counts)
            _save_json(
                horizon_dir / "sequence_metadata.json",
                {
                    "lookback_window": int(config.get("sequence", {}).get("lookback_window", 30)),
                    "sample_counts": sample_counts,
                    "sequence_feature_count": len(sequence_all.sequence_feature_cols),
                    "static_feature_count": len(sequence_all.static_feature_cols),
                },
            )

        for model_name in model_names:
            model_start = time.perf_counter()
            LOGGER.info("Training model=%s horizon=%s", model_name, horizon)
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
                model.fit(
                    sequence_train,
                    sequence_valid,
                    target_cols,
                    thresholds,
                    sample_weight=sequence_sample_weight,
                    edge_df=edge_df,
                    station_col=station_col,
                    date_col=date_col,
                )
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
            LOGGER.info(
                "Finished model=%s horizon=%s in %.1fs; outputs=%s",
                model_name,
                horizon,
                time.perf_counter() - model_start,
                model_dir,
            )

        LOGGER.info("Finished horizon=%s in %.1fs", horizon, time.perf_counter() - horizon_start)

    LOGGER.info("Experiment completed successfully.")


def main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(description="Run HydroTail experiments.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
