---
type: research-scope
date: 2026-03-24
project: hydrotail
status: active
---

# HydroTail Paper Scope Decision / Strict Ungauged / 2026-03-24

## Decision

The manuscript-facing study will use only the strict ungauged setting.

This means:

- main reported results must not use target-basin `specific_conductance` as model input
- main reported results must not use `naive_last_conductance`
- history-aware conductance runs are retained only as internal diagnostics

## Main paper line

Use the conductance-only strict ungauged configuration:

- config: `configs/dataset_bundle_linux_formal_conductance.yaml`
- target: `conductance`
- `features.include_target_history_features: false`
- dynamic drivers: `discharge`, `precip`, `tmin`, `tmax`, `pe`
- split: `unseen_station_and_future`

## Excluded from manuscript evidence

Do not use the following as paper-facing result lines:

- `configs/dataset_bundle_linux_formal_conductance_with_history_multihorizon.yaml`
- `plan/2026-03-24--conductance-history-vs-naive--summary.md`
- `plan/2026-03-24--conductance-history-vs-naive--metrics-long.csv`

These artifacts may still be kept for internal diagnosis of task difficulty and methodological upper bounds.

## Writing implication

The paper should be framed as:

- prediction in ungauged basin
- no in-situ conductance history available in the target basin
- probabilistic and tail-aware conductance forecasting under strict spatial-temporal generalization

It should not be framed as:

- monitored-basin forecasting
- warm-start conductance forecasting
- persistence-vs-history forecasting
