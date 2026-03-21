from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from hydrotail.train import run_experiment

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def _make_synthetic_data(base_dir: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(42)
    stations = [f"S{i:03d}" for i in range(1, 9)]
    dates = pd.date_range("2019-01-01", periods=120, freq="D")

    static_rows = []
    dynamic_rows = []
    for station_id in stations:
        slope = float(rng.uniform(0.5, 8.0))
        impervious = float(rng.uniform(0.01, 0.6))
        ksat = float(rng.uniform(2.0, 25.0))
        aridity = float(rng.uniform(0.2, 1.8))
        watershed_area = float(rng.uniform(10.0, 3000.0))
        latitude = float(rng.uniform(30.0, 48.0))
        longitude = float(rng.uniform(-120.0, -75.0))

        static_rows.append(
            {
                "station_id": station_id,
                "latitude": latitude,
                "longitude": longitude,
                "watershed_area": watershed_area,
                "slope": slope,
                "impervious_ratio": impervious,
                "ksat": ksat,
                "aridity_index": aridity,
            }
        )

        base_cond = 120 + 25 * aridity + 60 * impervious - 2.5 * ksat + 0.015 * watershed_area
        base_turb = 4 + 6 * impervious + 0.3 * slope
        discharge_prev = 20.0 + rng.normal(0, 2)
        cond_prev = base_cond + rng.normal(0, 8)
        turb_prev = base_turb + rng.normal(0, 1)

        for day_idx, date in enumerate(dates):
            seasonal = np.sin(2 * np.pi * day_idx / 365.25)
            precip = max(0.0, rng.gamma(1.5, 2.0) - 1.0)
            if rng.random() < 0.05:
                precip += rng.uniform(10.0, 30.0)
            air_temp = 14.0 + 10.0 * seasonal + rng.normal(0, 2)
            discharge = max(
                0.1,
                0.55 * discharge_prev + 0.8 * precip + 3.5 * max(seasonal, 0) + rng.normal(0, 1.5),
            )
            high_event = 1.0 if precip > 12.0 else 0.0

            conductance = (
                0.82 * cond_prev
                + 0.12 * base_cond
                - 0.85 * discharge
                + 0.08 * air_temp
                + 20.0 * high_event
                + rng.normal(0, 6)
            )
            conductance = max(5.0, conductance)

            turbidity = (
                0.55 * turb_prev
                + 0.55 * base_turb
                + 0.65 * precip
                + 0.15 * discharge
                + 18.0 * high_event
                + rng.normal(0, 1.8)
            )
            turbidity = max(0.0, turbidity)

            if rng.random() < 0.12:
                conductance = np.nan
            if rng.random() < 0.18:
                turbidity = np.nan

            if rng.random() < 0.08:
                continue

            dynamic_rows.append(
                {
                    "station_id": station_id,
                    "date": date,
                    "discharge": discharge,
                    "precip": precip,
                    "air_temp": air_temp,
                    "specific_conductance": conductance,
                    "turbidity": turbidity,
                }
            )

            discharge_prev = discharge
            cond_prev = cond_prev if np.isnan(conductance) else conductance
            turb_prev = turb_prev if np.isnan(turbidity) else turbidity

    static_path = base_dir / "static_attributes.csv"
    dynamic_path = base_dir / "dynamic_daily.csv"
    pd.DataFrame(static_rows).to_csv(static_path, index=False)
    pd.DataFrame(dynamic_rows).to_csv(dynamic_path, index=False)
    return dynamic_path, static_path


def _make_base_config(dynamic_path: Path, static_path: Path, output_dir: Path) -> dict[str, object]:
    return {
        "project": {"name": "hydrotail_smoke_test"},
        "paths": {
            "dynamic_data": str(dynamic_path),
            "static_data": str(static_path),
            "output_dir": str(output_dir),
        },
        "data": {
            "station_col": "station_id",
            "date_col": "date",
            "time_frequency": "D",
            "targets": {"conductance": "specific_conductance", "turbidity": "turbidity"},
            "dynamic_features": ["discharge", "precip", "air_temp"],
            "static_features": ["latitude", "longitude", "watershed_area", "slope", "impervious_ratio", "ksat", "aridity_index"],
        },
        "features": {
            "horizons": [1],
            "lags": [1, 3, 7],
            "rolling_windows": [3, 7],
            "add_seasonal": True,
            "min_non_null_feature_ratio": 0.2,
        },
        "sequence": {"lookback_window": 14, "min_history_ratio": 0.12},
        "graph": {
            "enabled": True,
            "k_neighbors": 3,
            "feature_columns": ["slope", "impervious_ratio", "ksat", "aridity_index", "watershed_area"],
        },
        "splits": {
            "strategy": "unseen_station_and_future",
            "seed": 42,
            "train_station_fraction": 0.625,
            "valid_station_fraction": 0.125,
            "test_station_fraction": 0.25,
            "train_time_fraction": 0.7,
            "valid_time_fraction": 0.15,
        },
        "tail": {
            "exceedance_quantiles": {"conductance": 0.95, "turbidity": 0.9},
            "quantiles": [0.1, 0.5, 0.9, 0.95],
            "quantile_weights": [1.0, 1.0, 1.5, 2.0],
            "tail_weight_multiplier": 4.0,
        },
        "run": {"models": ["linear_tail", "gbdt_tail", "torch_tail", "seq_tcn_tail", "seq_transformer_tail"]},
        "models": {
            "linear_tail": {"alpha": 0.01, "max_iter": 1000},
            "gbdt_tail": {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.05, "min_samples_leaf": 6},
            "torch_tail": {
                "graph_backend": "neighbor_stats",
                "hidden_dims": [32, 16],
                "dropout": 0.1,
                "batch_size": 64,
                "epochs": 2,
                "patience": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "gnn": {"hidden_dim": 32, "num_layers": 2, "dropout": 0.1},
            },
            "seq_tcn_tail": {
                "hidden_dim": 32,
                "tcn_channels": [32, 32],
                "kernel_size": 3,
                "dropout": 0.1,
                "batch_size": 32,
                "epochs": 2,
                "patience": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
            },
            "seq_transformer_tail": {
                "hidden_dim": 32,
                "num_heads": 4,
                "num_layers": 1,
                "ff_multiplier": 2,
                "dropout": 0.1,
                "batch_size": 32,
                "epochs": 2,
                "patience": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
            },
        },
    }


def _run_scenario(config_dir: Path, config_name: str, config: dict[str, object]) -> None:
    config_path = config_dir / config_name
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    run_experiment(str(config_path))


def main() -> None:
    workspace = Path(__file__).resolve().parent.parent
    data_dir = workspace / "data"
    output_root = workspace / "outputs"
    config_dir = workspace / "configs"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    dynamic_path, static_path = _make_synthetic_data(data_dir)

    neighbor_output_dir = output_root / "smoke_test_neighbor_stats"
    neighbor_config = _make_base_config(dynamic_path, static_path, neighbor_output_dir)
    _run_scenario(config_dir, "smoke_test_neighbor_stats.yaml", neighbor_config)

    gnn_output_dir = output_root / "smoke_test_gnn"
    gnn_config = _make_base_config(dynamic_path, static_path, gnn_output_dir)
    gnn_config["run"]["models"] = ["torch_tail"]
    gnn_config["models"]["torch_tail"]["graph_backend"] = "gnn"
    _run_scenario(config_dir, "smoke_test_gnn.yaml", gnn_config)


if __name__ == "__main__":
    main()
