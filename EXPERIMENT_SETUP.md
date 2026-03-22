# HydroTail Experiment Setup

## 1. Available configs

Use one of these configs depending on your data source:

- `configs/default_experiment.yaml`
  - reads flat files such as `data/dynamic_daily.csv`
  - useful for smoke tests and manually prepared tabular data
- `configs/dataset_bundle_experiment.yaml`
  - reads the current folder-style dataset under `dataset/`
  - useful for the real multi-file station dataset

## 2. Dataset bundle layout

The folder-style loader expects:

- `dataset/attritubes/*.xlsx` or `dataset/attributes/*.xlsx`
- `dataset/time series/<source folder>/<station file>.csv`
- water-quality targets in one of these routes:
  - `dataset/WQ/SEC/*.csv` and `dataset/WQ/Tur/*.csv`
  - `dataset/WQ/wq_observations.csv`
  - legacy folders such as `dataset/WQ/Paired_EC/*` and `dataset/WQ/Paired_Tur/*`

Current supported time-series source folders in `configs/dataset_bundle_experiment.yaml`:

- `streamflow`
- `daymet`
- `PE`
- `LAI`
- `SWE_SNODAS`

These are merged by station and date into one dynamic table.

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
