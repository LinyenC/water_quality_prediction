# HydroTail Experiment Setup

## 1. Available configs

Use one of these configs depending on your data source:

- `configs/default_experiment.yaml`
  - reads flat files such as `data/dynamic_daily.csv`
  - useful for smoke tests and manually prepared tabular data
- `configs/dataset_bundle_experiment.yaml`
  - reads the current folder-style dataset under `dataset/`
  - also supports separately configured roots for `attributes`, `time series`, and `WQ`
  - useful for the real multi-file station dataset

## 2. Dataset bundle layout

The folder-style loader supports two ways to locate data:

1. one shared `dataset_root`
   - `dataset/attritubes/*.xlsx` or `dataset/attributes/*.xlsx`
   - `dataset/time series/<source folder>/<station file>.csv`
   - `dataset/WQ/SEC/*.csv`, `dataset/WQ/Tur/*.csv`, or `dataset/WQ/wq_observations.csv`
2. separate component roots in config
   - `paths.attributes_root`
   - `paths.time_series_root`
   - `paths.wq_root`

If a component-specific root is configured, it takes precedence over the matching subfolder under `dataset_root`.

Current supported time-series source folders in `configs/dataset_bundle_experiment.yaml`:

- `streamflow`
- `daymet`
- `PE`
- `LAI`
- `SWE_SNODAS`

These are merged by station and date into one dynamic table.

Example config:

```yaml
paths:
  dataset_root: dataset
  attributes_root: 'F:\\jupyter_notebook\\wat_quality_pred\\dataset\\attritubes'
  time_series_root: 'F:\\jupyter_notebook\\baseflow_predict\\dataset\\time_series'
  wq_root: 'F:\\jupyter_notebook\\wat_quality_pred\\dataset\\WQ'
```

## 3. Water-quality target loading

The loader currently supports three target-input routes:

1. `dataset/WQ/wq_observations.csv`
   - recommended if you want one unified file
   - must contain:
     - `station_id`
     - `date`
     - `specific_conductance`
     - `turbidity`
2. station-level csv files under:
   - `dataset/WQ/SEC/*.csv`
   - `dataset/WQ/Tur/*.csv`
   - this is the route now configured in `configs/dataset_bundle_experiment.yaml`
3. other station-level tabular files under legacy folders such as:
   - `Paired_EC`
   - `Paired_Tur`
   - supported for `csv`, `xlsx`, and `parquet`

Current limitation:

- the old `Paired_EC/*.mat` and `Paired_Tur/*.mat` files are MATLAB `table/timetable` objects stored as opaque MCOS data
- `scipy` and `hdf5storage` still cannot convert those timetable objects directly into the tabular columns needed by the pipeline
- if you go back to MAT-only WQ inputs, the loader will still ask for either:
  - csv export of those files, or
  - a unified `dataset/WQ/wq_observations.csv`

## 4. Missing and discontinuous series

The pipeline explicitly handles:

- discontinuous station records by reindexing each station to a daily grid
- shorter `specific_conductance` or `turbidity` histories than covariate histories
- missing covariates through imputation plus explicit missingness indicators
- sequence-model gaps through explicit masks
- partial labels across targets: one row may keep only `specific_conductance` or only `turbidity`
- target-specific NaN masking during loss and metric computation

This means two observations that are several days apart are no longer treated as adjacent time steps.
It also means the multi-task pipeline no longer drops a row just because one target is missing.

## 5. Model options

Tabular options:

- `linear_tail`
- `gbdt_tail`
- `torch_tail`

Sequence options:

- `seq_tcn_tail`
- `seq_transformer_tail`
- `seq_retrieval_prototype_tail`

## 6. Graph backend options

Both the tabular and sequence routes now expose the same graph-backend switch.

Set one of these to `none`, `neighbor_stats`, or `gnn`:

- `models.torch_tail.graph_backend`
- `models.seq_tcn_tail.graph_backend`
- `models.seq_transformer_tail.graph_backend`

Meaning of each option:

- `neighbor_stats`
  - uses precomputed `graph_neighbor_*` features from `hydrotail/graph.py`
- `gnn`
  - removes those neighbor-summary features from the direct input branch
  - uses the station similarity graph directly inside the model
  - performs same-day graph propagation before the quantile and event heads
  - for sequence models, graph propagation happens after the temporal encoder on the window end date
- `none`
  - ignores the precomputed neighbor-summary features

Example:

```yaml
models:
  seq_tcn_tail:
    graph_backend: gnn
    hidden_dim: 64
    gnn:
      hidden_dim: 64
      num_layers: 2
      dropout: 0.1
```

If you want to compare `neighbor_stats` and `gnn`, run paired experiments with the same config except for the target model's `graph_backend`.

## 7. Output heads

The explicit learned `point head` has been removed.

All models now use:

- quantile outputs
- exceedance-probability outputs

Point predictions saved to metrics and csv files are derived from `q0.5`.

## 8. Split analysis

With `unseen_station_and_future`, the training pipeline saves metrics and predictions for:

- `train`
- `valid`
- `test`
- `train_station_early`
- `train_station_late`
- `test_station_early`
- `test_station_late`

This lets you separately inspect:

- familiar stations in early periods
- familiar stations in late periods
- unseen stations in early periods
- unseen stations in late periods

## 9. Run

For flat-file data:

```bash
python -m hydrotail.train --config configs/default_experiment.yaml
```

For the folder-style dataset:

```bash
python -m hydrotail.train --config configs/dataset_bundle_experiment.yaml
```

To switch a model from neighbor statistics to the internal GNN backend, change for example:

```yaml
models:
  seq_transformer_tail:
    graph_backend: gnn
```

Every experiment run now also saves a config snapshot to the matching output folders:

- `outputs/<run_name>/experiment_config.yaml`
- `outputs/<run_name>/horizon_<k>/experiment_config.yaml`

This preserves the exact YAML used for that run and makes later result tracing easier.

## 10. Smoke test

Use:

```bash
python -m hydrotail.smoke_test
```

The smoke test now runs two scenarios:

- `outputs/smoke_test_neighbor_stats`
  - verifies `torch_tail`, `seq_tcn_tail`, and `seq_transformer_tail` with `graph_backend=neighbor_stats`
  - also verifies the tabular baselines
- `outputs/smoke_test_gnn`
  - verifies `torch_tail`, `seq_tcn_tail`, and `seq_transformer_tail` with `graph_backend=gnn`

This checks:

- no learned point head
- tabular baselines
- sequence models
- both graph-backend modes in tabular and sequence routes
- extra period-slice analysis outputs



## 11. Parquet cache, downcast, and coverage-aware smoke subset

The folder-style dataset loader now supports three additional controls:

- `data.downcast_float32`
  - when enabled, numeric columns are downcast to `float32` after bundle loading and again after feature construction
  - this reduces memory pressure for sparse station-day tables with many lag and rolling features
- `data.dataset_bundle.use_parquet_cache`
  - when enabled, the bundle loader writes component-level caches for:
    - `attributes.parquet`
    - `time_series.parquet`
    - `wq_observations.parquet`
  - cache files are stored under `<dataset_root>/_hydrotail_cache/<namespace>/`
  - each namespace also writes `metadata.json`
- `data.dataset_bundle.station_selection_strategy`
  - when `station_limit` is set, the loader can now pick a coverage-aware smoke subset instead of taking the first N stations
  - current supported values:
    - `coverage_aware`
    - fallback non-coverage mode for deterministic first-N selection

Recommended smoke-specific controls:

```yaml
data:
  downcast_float32: true
  dataset_bundle:
    use_parquet_cache: true
    refresh_cache: false
    station_limit: 32
    station_selection_strategy: coverage_aware
    coverage_min_stations_per_target: 12
```

Notes:

- `refresh_cache: false` reuses existing parquet caches for repeated smoke runs on the same data/config namespace.
- If you change the source roots or the smoke-subset specification, a new cache namespace will be created automatically.
- If the raw files under the same paths are modified in place, set `refresh_cache: true` once to rebuild the cache.

## 12. Sparse-target smoke validation on Linux

The Linux smoke config `configs/dataset_bundle_linux_smoke.yaml` is now intentionally different from the main experiment config in one place:

- smoke uses `splits.strategy: unseen_station`
- the main experiment still uses `splits.strategy: unseen_station_and_future`

Reason:

- for sparse targets such as `turbidity`, the stricter spatiotemporal split can remove a target entirely from the train partition on a small smoke subset, even when the selected stations do contain that target overall
- `unseen_station` is therefore the safer smoke setting when the goal is to validate the end-to-end pipeline rather than reproduce the final research split

Observed Linux server behavior after this change:

- cold cache build still spends most of its time on first-pass file parsing and merged-table construction
- warm cache reuse reduces dataset loading from minutes to well under one second on the 32-station smoke subset
- the coverage-aware subset now loads both `conductance` and `turbidity` files for the smoke run

Update (2026-03-23, fast smoke mode):

- the Linux smoke config is now intentionally reduced to `models: [torch_tail]`
- reason: `linear_tail` and `gbdt_tail` are CPU baselines and can dominate wall-clock time on the current smoke subset even after parquet caching is enabled
- this keeps smoke focused on end-to-end validation of the main GPU-backed route
- if baseline comparison is needed, run a separate baseline config rather than mixing it into the fast smoke check

- the fast Linux smoke output directory now uses a dedicated path for torch-only validation so its artifacts do not mix with earlier baseline-heavy smoke runs

## 13. Formal static feature set (2026-03-23)

The formal experiment configs now use an expanded static feature set instead of the earlier 15-column core-only version.

Core 15 retained:

- `latitude`
- `longitude`
- `watershed_area`
- `DEMmean`
- `DROPmean`
- `DEMstd`
- `permeability`
- `porosity`
- `ksat`
- `prcp_yearly_mean`
- `PE_yearly_mean`
- `LAImean`
- `Qmean`
- `SWEmean`
- `population`

Newly added for the formal study:

- hydrologic distribution:
  - `Q50`
  - `Qstd`
  - `Q10`
- climate distribution:
  - `P50`
  - `P95`
  - `PEstd`
- vegetation / biomass:
  - `LAIstd`
  - `agBiomass`
- irrigation / managed water use:
  - `iaFmean`
  - `IWUmean`
  - `IWU_SI`
- reservoir / public supply water:
  - `RS_mean`
  - `RS_std`
  - `PSW_SW_mean`
  - `PSW_SW_si`
  - `PSW_GW_mean`
  - `PSW_GW_si`
- soil water retention / texture:
  - `Wsat`
  - `WPnt`
  - `vf_clay_s`

Selection rule used here:

- keep the original core set for comparability
- add attributes with both:
  - meaningful station-level correlation to water-quality summaries, and
  - clear physical interpretation for the paper narrative
- avoid adding obviously redundant alternatives such as:
  - `bgBiomass`
  - `Q30`
  - `P5`
  - `PE50`
  - `PSW_*_std`
  - `IWUstd`

At this stage the graph similarity feature set is intentionally left unchanged. The expanded attributes are first used as static predictors; whether they should also enter graph construction can be treated as a later ablation.


## 14. Temperature feature cleanup (2026-03-23)

To avoid redundant temperature predictors in the main experiments:

- `air_temp` has been removed from the active bundle-based dynamic feature lists
- active Linux formal and smoke configs now keep `tmin` and `tmax` only
- the bundle loader will derive `air_temp` only when a future config explicitly requests it


## 15. Model-frame performance fix (2026-03-23)

The full Linux formal run no longer fails in the raw-data ingestion stage. The new bottleneck was identified inside `build_model_frame`, where lag and rolling features were inserted column-by-column into a very large pandas DataFrame.

The current implementation now:

- assembles temporal features in batches instead of repeated `frame[col] = ...` insertion
- logs the major model-frame stages (`daily frame`, `target-eligible rows`, per-series temporal feature progress, post-merge row counts)
- keeps the existing source-level parquet cache so formal reruns reuse `attributes`, `time_series`, and `wq_observations` directly

This update is meant to reduce DataFrame fragmentation and make the next server rerun observable enough to diagnose any remaining bottleneck.


## 16. WQ cache cleanup and config split update (2026-03-23)

The bundle loader now drops any WQ row whose configured targets are all missing before writing the component parquet cache.

Why this was added:

- the data audit showed that `wq_observations.parquet` previously contained a very large number of rows where both targets were `NaN`
- those rows add storage and alignment cost but provide no learning signal
- the cache namespace schema version has therefore been bumped so future bundle caches are rebuilt with the cleaned WQ rule

At the same time, the active bundle-based configs were simplified:

- `lai` and `swe` are removed from the dynamic feature lists
- `LAImean`, `LAIstd`, and `SWEmean` are removed from the active static feature lists
- `SWEmean` is also removed from the current graph similarity feature list

Two new Linux configs were added:

- `configs/dataset_bundle_linux_formal_conductance.yaml`
  - conductance-only formal run
  - no `lai/swe`
  - torch-only
- `configs/dataset_bundle_linux_turbidity_later_era_pilot.yaml`
  - turbidity-only pilot run
  - no `lai/swe`
  - date filter starts at `2000-01-01`
  - uses a later-era split to avoid the previous turbidity train-collapse


## 17. Target-history feature removal and tail metric fix (2026-03-24)

Two follow-up changes were made after inspecting the first `conductance-only` formal run:

- bundle-based experiment configs now set `features.include_target_history_features: false`
- target source columns such as `specific_conductance` and `turbidity` are now used to create future labels, but are no longer automatically included as raw inputs, observed flags, lag features, or rolling features unless a config explicitly turns target-history features back on

This makes the default conductance workflow better aligned with the intended deployment setting where no in-situ conductance measurements are available for the prediction basin.

In addition, the tail-metric output bug has been fixed:

- the metrics writer previously stored the true tail count under `tail_tail_count` while leaving the placeholder `tail_count = 0`
- it now writes the actual count to `tail_count` and keeps `tail_mae`, `tail_rmse`, and `tail_r2` under stable names

## 18. Paper-facing scope decision (2026-03-24)

The paper-facing research line is now fixed to the strict ungauged setting.

This means:

- the main study does **not** use target-basin `specific_conductance` history as model input
- the active conductance manuscript line should use configs with:
  - `features.include_target_history_features: false`
- the main paper should not report the `naive_last_conductance` comparison tables
- the main paper should not use the `conductance-with-history` multihorizon run as a reported result line

Current recommended paper-facing config:

- `configs/dataset_bundle_linux_formal_conductance.yaml`
  - conductance-only
  - strict ungauged input setting
  - no `lai/swe`
  - no target-history features

Internal-use-only diagnostic artifacts may still remain in `plan/` or `configs/`, but they should be treated as method-diagnosis material rather than manuscript evidence.

## 19. Retrieval-prototype sequence model (2026-03-24)

The codebase now also includes a new sequence route:

- `seq_retrieval_prototype_tail`

This is the first implementable `v1` of the strict-ungauged donor-transfer idea.

Current design:

- local TCN or Transformer event encoder
- station-level donor-memory retrieval from source basins
- learnable prototype tokens
- per-epoch training logs (`epoch/train_loss/valid_loss/wait`)
- cached donor candidates plus interval-based memory-bank refresh in the current optimized `v1`
- gated fusion of:
  - local representation
  - donor representation
  - prototype representation
- existing quantile and exceedance heads

Recommended Linux config for this route:

- `configs/dataset_bundle_linux_formal_conductance_retrieval.yaml`

Recommended ablation bundle for fair comparison against the sequence baseline:

- `configs/dataset_bundle_linux_formal_conductance_retrieval_ablation.yaml`
- shared preprocessing / shared sequence-sample construction
- models:
  - `seq_tcn_tail`
  - `seq_retrieval_prototype_tail`
  - `seq_retrieval_prototype_tail_nograph`
  - `seq_retrieval_prototype_tail_noproto`

Current intended use:

- conductance-only
- `features.include_target_history_features: false`
- strict ungauged manuscript line
