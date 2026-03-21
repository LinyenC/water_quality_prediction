from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _resolve_point_quantile(quantiles: list[float]) -> float:
    """Use the median quantile as the exported point prediction."""

    return min(quantiles, key=lambda value: abs(value - 0.5))


def _constant_probability(length: int, value: float) -> np.ndarray:
    return np.full(length, value, dtype=float)


class LinearTailModel:
    """A light tabular baseline with quantile and exceedance heads."""

    input_mode = "tabular"

    def __init__(self, quantiles: list[float], model_cfg: dict[str, object], random_state: int = 42) -> None:
        self.quantiles = quantiles
        self.point_quantile = _resolve_point_quantile(quantiles)
        self.model_cfg = model_cfg
        self.random_state = random_state
        self.models: dict[str, dict[str, object]] = {}

    def fit(
        self,
        frame: pd.DataFrame,
        feature_cols: list[str],
        target_cols: dict[str, str],
        thresholds: dict[str, float],
        sample_weight: np.ndarray | None = None,
    ) -> None:
        for target_name, target_col in target_cols.items():
            valid_mask = frame[target_col].notna().to_numpy()
            if not np.any(valid_mask):
                self.models[target_name] = {
                    "quantiles": {},
                    "classifier": None,
                    "constant_event_probability": float("nan"),
                }
                continue

            x = frame.loc[valid_mask, feature_cols]
            y = frame.loc[valid_mask, target_col].to_numpy(dtype=float)
            target_sample_weight = None if sample_weight is None else np.asarray(sample_weight, dtype=float)[valid_mask]

            quantile_models: dict[float, object] = {}
            for quantile in self.quantiles:
                quantile_model = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", QuantileRegressor(quantile=quantile, alpha=float(self.model_cfg.get("alpha", 0.01)))),
                    ]
                )
                quantile_model.fit(x, y, model__sample_weight=target_sample_weight)
                quantile_models[quantile] = quantile_model

            classifier = None
            constant_event_probability = float("nan")
            threshold = thresholds.get(target_name, float("nan"))
            if np.isfinite(threshold):
                event_target = (y >= threshold).astype(int)
                if len(np.unique(event_target)) > 1:
                    classifier = Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                            (
                                "model",
                                LogisticRegression(
                                    max_iter=int(self.model_cfg.get("max_iter", 2000)),
                                    random_state=self.random_state,
                                ),
                            ),
                        ]
                    )
                    classifier.fit(x, event_target, model__sample_weight=target_sample_weight)
                else:
                    constant_event_probability = float(event_target[0])

            self.models[target_name] = {
                "quantiles": quantile_models,
                "classifier": classifier,
                "constant_event_probability": constant_event_probability,
            }

    def predict(self, frame: pd.DataFrame, feature_cols: list[str]) -> dict[str, dict[str, object]]:
        x = frame[feature_cols]
        outputs: dict[str, dict[str, object]] = {}
        for target_name, bundle in self.models.items():
            if not bundle["quantiles"]:
                quantile_predictions = {quantile: np.full(len(frame), np.nan, dtype=float) for quantile in self.quantiles}
                event_probability = np.full(len(frame), np.nan, dtype=float)
            else:
                quantile_predictions = {quantile: bundle["quantiles"][quantile].predict(x) for quantile in self.quantiles}
                ordered = np.column_stack([quantile_predictions[q] for q in self.quantiles])
                ordered = np.sort(ordered, axis=1)
                for idx, quantile in enumerate(self.quantiles):
                    quantile_predictions[quantile] = ordered[:, idx]

                if bundle["classifier"] is None:
                    event_probability = _constant_probability(len(frame), float(bundle["constant_event_probability"]))
                else:
                    event_probability = bundle["classifier"].predict_proba(x)[:, 1]

            outputs[target_name] = {
                "point": quantile_predictions[self.point_quantile],
                "quantiles": quantile_predictions,
                "exceedance_probability": event_probability,
            }
        return outputs

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump(self, handle)


class GBDTTailModel:
    """Gradient boosting baseline with quantile and exceedance heads."""

    input_mode = "tabular"

    def __init__(self, quantiles: list[float], model_cfg: dict[str, object], random_state: int = 42) -> None:
        self.quantiles = quantiles
        self.point_quantile = _resolve_point_quantile(quantiles)
        self.model_cfg = model_cfg
        self.random_state = random_state
        self.imputer = SimpleImputer(strategy="median")
        self.models: dict[str, dict[str, object]] = {}

    def fit(
        self,
        frame: pd.DataFrame,
        feature_cols: list[str],
        target_cols: dict[str, str],
        thresholds: dict[str, float],
        sample_weight: np.ndarray | None = None,
    ) -> None:
        for target_name, target_col in target_cols.items():
            valid_mask = frame[target_col].notna().to_numpy()
            if not np.any(valid_mask):
                self.models[target_name] = {
                    "quantiles": {},
                    "classifier": None,
                    "constant_event_probability": float("nan"),
                    "imputer": None,
                }
                continue

            x_raw = frame.loc[valid_mask, feature_cols]
            imputer = SimpleImputer(strategy="median")
            x = imputer.fit_transform(x_raw)
            y = frame.loc[valid_mask, target_col].to_numpy(dtype=float)
            target_sample_weight = None if sample_weight is None else np.asarray(sample_weight, dtype=float)[valid_mask]

            quantile_models: dict[float, object] = {}
            for quantile in self.quantiles:
                quantile_model = GradientBoostingRegressor(
                    loss="quantile",
                    alpha=quantile,
                    n_estimators=int(self.model_cfg.get("n_estimators", 300)),
                    max_depth=int(self.model_cfg.get("max_depth", 3)),
                    learning_rate=float(self.model_cfg.get("learning_rate", 0.05)),
                    min_samples_leaf=int(self.model_cfg.get("min_samples_leaf", 20)),
                    random_state=self.random_state,
                )
                quantile_model.fit(x, y, sample_weight=target_sample_weight)
                quantile_models[quantile] = quantile_model

            classifier = None
            constant_event_probability = float("nan")
            threshold = thresholds.get(target_name, float("nan"))
            if np.isfinite(threshold):
                event_target = (y >= threshold).astype(int)
                if len(np.unique(event_target)) > 1:
                    classifier = GradientBoostingClassifier(
                        n_estimators=int(self.model_cfg.get("n_estimators", 300)),
                        learning_rate=float(self.model_cfg.get("learning_rate", 0.05)),
                        random_state=self.random_state,
                    )
                    classifier.fit(x, event_target, sample_weight=target_sample_weight)
                else:
                    constant_event_probability = float(event_target[0])

            self.models[target_name] = {
                "quantiles": quantile_models,
                "classifier": classifier,
                "constant_event_probability": constant_event_probability,
                "imputer": imputer,
            }

    def predict(self, frame: pd.DataFrame, feature_cols: list[str]) -> dict[str, dict[str, object]]:
        outputs: dict[str, dict[str, object]] = {}
        for target_name, bundle in self.models.items():
            if not bundle["quantiles"]:
                quantile_predictions = {quantile: np.full(len(frame), np.nan, dtype=float) for quantile in self.quantiles}
                event_probability = np.full(len(frame), np.nan, dtype=float)
            else:
                x = bundle["imputer"].transform(frame[feature_cols])
                quantile_predictions = {quantile: bundle["quantiles"][quantile].predict(x) for quantile in self.quantiles}
                ordered = np.column_stack([quantile_predictions[q] for q in self.quantiles])
                ordered = np.sort(ordered, axis=1)
                for idx, quantile in enumerate(self.quantiles):
                    quantile_predictions[quantile] = ordered[:, idx]

                if bundle["classifier"] is None:
                    event_probability = _constant_probability(len(frame), float(bundle["constant_event_probability"]))
                else:
                    event_probability = bundle["classifier"].predict_proba(x)[:, 1]

            outputs[target_name] = {
                "point": quantile_predictions[self.point_quantile],
                "quantiles": quantile_predictions,
                "exceedance_probability": event_probability,
            }
        return outputs

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump(self, handle)
