# Research Plan: HydroTail-GMT

## Topic Positioning

Suggested formal title:

`Tail-Risk Forecasting of Specific Conductance and Turbidity for Unseen Stations Across the Contiguous United States via Hydrologic Similarity Graphs and Multi-Task Quantile Learning`

Core positioning:

- Study area: monitoring stations across the contiguous United States
- Time scale: daily observations
- Target variables: `specific conductance` and `turbidity`
- Main scenario: unseen-station generalization
- Main value: not only mean prediction, but also tail-risk and exceedance prediction

## Key Concept

`Cold start` in this project means:

- the model is trained on many existing stations
- a new station appears during testing
- the new station has little or no long historical target record for separate fine-tuning
- the model still needs to generate reliable forecasts using transferable patterns and static watershed/site attributes

For formal academic writing, prefer:

- `unseen-station generalization`
- `new-station transfer prediction`

instead of relying only on the phrase `cold start`.

## Research Questions

### RQ1

Can one transferable model generalize to unseen stations for daily prediction of `specific conductance` and `turbidity` across the contiguous United States?

### RQ2

Do `specific conductance` and `turbidity` show different cross-station transferability under distinct hydrologic, climatic, and land-use settings?

### RQ3

Can tail-aware learning improve forecasting of extreme high turbidity and anomalously high conductance beyond standard mean-regression models?

### RQ4

Which watershed, meteorological, and seasonal factors are associated with prediction failures and tail-risk events?

## Main Innovations

1. The study focuses on `unseen-station generalization` rather than ordinary in-station forecasting.
2. The framework jointly predicts `specific conductance` and `turbidity`, enabling comparison of transferability between ion-dominated and particle-dominated water-quality indicators.
3. The model explicitly performs `tail-risk modeling` by producing `q0.5`-derived point estimates, quantile forecasts, and threshold exceedance probabilities.
4. A `hydrologic similarity graph` is used to connect stations based on transferable physical similarity rather than only station identity or geographic distance.
5. The evaluation highlights model robustness on tail events and failure mechanisms, not only average accuracy.

## Data Assumptions

Current assumption based on user description:

- multi-station daily observations across the contiguous United States
- observed targets include `specific conductance` and `turbidity`

Recommended dynamic features:

- lags of target variables: `1, 3, 7, 14, 30` days
- rolling mean, rolling std, rolling min/max
- day-of-year or seasonal encoding
- discharge, water temperature, pH, DO if available
- daily precipitation and air temperature if external meteorology can be joined

Recommended static features:

- latitude and longitude
- watershed area
- elevation and slope
- land use / land cover
- climate zone / aridity indicators
- impervious surface ratio
- saturated hydraulic conductivity (`Ksat`) or related infiltration proxy
- soil or geology proxies

Potential external sources to merge later:

- Daymet for daily meteorology
- EPA StreamCat for watershed attributes
- NHDPlus HR for hydrologic context
- NLCD for land cover

## Proposed Model

Model name:

`HydroTail-GMT`

Expanded name:

`Hydrologic Graph Multi-task Tail Network`

### 1. Inputs

- dynamic daily sequences for each station
- static site and watershed attributes
- optional external meteorological covariates

### 2. Temporal Encoder

Use one of the following:

- `TCN` as a strong and efficient default
- lightweight `Transformer` if longer dependencies are needed

Purpose:

- capture lag effects
- represent short- and medium-term dynamics
- encode seasonality and persistence

Current implementation note:

- both `seq_tcn_tail` and `seq_transformer_tail` now support `graph_backend = none | neighbor_stats | gnn`
- under `gnn`, the model first performs temporal encoding and then applies same-day graph propagation on the window end date

### 3. Hydrologic Similarity Graph

Construct station relationships using:

- physical distance
- watershed property similarity
- climate similarity
- optional streamflow regime similarity

Recommended `physical fingerprint` variables for graph construction:

- slope and relief
- impervious surface ratio
- infiltration or `Ksat` proxy
- land-use composition
- aridity or climate regime
- optional discharge variability statistics

This graph is intended to support transferability for unseen stations better than a pure black-box attention mechanism.

### 4. Shared Multi-Task Backbone

The model uses a shared latent representation across both tasks:

- task 1: `specific conductance`
- task 2: `turbidity`

Reason:

- both targets are water-quality indicators
- but their dynamics differ
- shared learning may transfer useful hydrologic context while task-specific heads preserve differences

### 5. Tail-Aware Prediction Heads

Each target uses a joint regression-and-event design with three outputs:

- point estimate derived from `q0.5`
- quantile prediction: `0.1 / 0.5 / 0.9`
- exceedance probability: `P(y > threshold)`

This head can be interpreted as a `regression + classification` dual-head structure:

- the regression branch estimates the target magnitude
- the exceedance branch predicts whether a high-risk event occurs
- the two branches are trained together so event signals can support tail prediction

Suggested threshold design:

- turbidity: station-wise or global `top 10%`
- conductance: station-wise high quantile, or seasonally adjusted anomaly threshold

### 6. Physical Plausibility Constraints

Recommended low-risk constraints to include:

- non-negativity for both `specific conductance` and `turbidity`
- optional upper-bound regularization when a physically justified historical range is available

Optional advanced constraint:

- a conductance-discharge monotonicity penalty can be tested only when discharge data are available and the station or region shows a stable dilution pattern

This project should treat such monotonicity as an empirical option, not a universal law, because the conductance-flow relationship can vary across regions and seasons.

### 7. Loss Function

Total loss can combine:

- quantile loss: `Pinball Loss`
- exceedance loss: `Binary Cross-Entropy`
- optional boundary penalty for physical plausibility

Recommended tail-aware extension:

- assign larger weights to upper quantiles such as `0.95` and `0.99`
- emphasize samples above the high-risk threshold during training

Simple form:

`L = beta * L_quantile + gamma * L_exceedance + delta * L_boundary`

One practical implementation is:

`L_quantile = sum_k w_k * Pinball(q_k)`

where `w_k` is larger for upper-tail quantiles.

### 8. Training Strategy for Extremes

Recommended default:

- importance sampling or sample reweighting for high-value and event-period observations

Rationale:

- extreme high turbidity and anomalously high conductance events are rare
- uniform mini-batch sampling tends to under-train the upper tail

Optional experiment:

- compare against synthetic oversampling methods such as `SMOGN`, but keep them as secondary experiments because synthetic regression samples may distort temporal structure

## Baselines

Recommended baseline groups for the strict ungauged main study:

- `Linear Regression` / `Elastic Net`
- `LightGBM` / `XGBoost`
- `LSTM`
- `TFT`
- multi-task model without graph
- graph model without tail heads

Current codebase status:

- already implemented:
  - `linear_tail`
  - `gbdt_tail`
  - `torch_tail`
  - `seq_tcn_tail`
  - `seq_transformer_tail`
- not yet implemented as dedicated baselines in the current repo:
  - `LSTM`
  - `TFT`
  - `LightGBM` / `XGBoost` as separate named routes

## Ablation Design

Required ablations:

- compare `graph_backend = none | neighbor_stats | gnn` in both tabular and sequence routes
- remove graph module
- replace hydrologic similarity graph with geographic distance graph
- remove multi-task sharing
- remove quantile head
- remove exceedance head
- remove importance sampling / tail reweighting
- remove physical boundary constraints
- remove static watershed attributes

## Evaluation Protocol

### Split Strategy

Use three evaluation settings:

1. time extrapolation on seen stations
2. unseen-station extrapolation
3. unseen-station plus future-period extrapolation

Avoid random splitting because it causes leakage and overestimates model performance.

### Metrics

For point prediction:

- `MAE`
- `RMSE`
- `R2`

For quantile prediction:

- `Pinball Loss`
- interval coverage
- interval width

For exceedance prediction:

- `AUC`
- `F1`
- `Recall`
- `Precision`

For reporting:

- overall metrics
- station-wise average metrics
- tail-event-only metrics

### Failure Analysis

Analyze where the model fails:

- humid vs arid regions
- urban vs agricultural watersheds
- low-variability vs high-variability stations
- tail-event periods vs ordinary periods

## Possible Contributions

This study can claim contribution more safely through:

- a stronger task setting
- a more meaningful generalization scenario
- explicit tail-risk prediction
- mechanism-oriented failure analysis

instead of claiming a totally novel deep-learning architecture.

## Current Implementation Snapshot (2026-03-22)

The project is no longer at the purely conceptual stage. The current repo already contains a runnable experiment framework with the following status:

- data ingestion:
  - supports both flat-table input and directory-style dataset bundles
  - supports either one unified `dataset_root` or separate roots for:
    - `attributes_root`
    - `time_series_root`
    - `wq_root`
- implemented model routes:
  - tabular baselines: `linear_tail`, `gbdt_tail`
  - main deep tabular route: `torch_tail`
  - sequence routes: `seq_tcn_tail`, `seq_transformer_tail`
- graph integration:
  - all deep routes now support `graph_backend = none | neighbor_stats | gnn`
- tail-aware outputs:
  - `q0.5`-derived point prediction
  - quantile outputs
  - exceedance probability outputs
- deployment:
  - a Linux server run path has already been validated
  - code directory: `/home/linyen3/wat_quality_pred`
  - dataset directory: `/data/data2/linyen3/wat_quality_pred/dataset`
  - output directory: `/data/data2/linyen3/wat_quality_pred/outputs`
- GPU support:
  - deep models now accept explicit `device` settings such as `cpu`, `cuda`, `cuda:0`, `cuda:1`
  - the current Linux config pins deep models to `cuda:0`
  - on the current server, `cuda:0` corresponds to `NVIDIA GeForce RTX 3090`

Current practical bottleneck:

- the first real-data server run has already started
- however, the main early bottleneck is still dataset loading and pandas-side feature preparation
- this is expected because the current dataset includes thousands of per-station files rather than one pre-merged cache

## Draft Abstract

This study targets daily forecasting of `specific conductance` and `turbidity` across multiple monitoring stations in the contiguous United States, with a particular focus on generalization to unseen stations and prediction of tail-risk events. To address strong spatial heterogeneity in climate, watershed characteristics, and hydrologic processes, we propose a hydrologic graph multi-task tail-learning framework that integrates historical time-series signals, meteorological drivers, and static watershed attributes. The framework jointly forecasts the two water-quality indicators while explicitly producing `q0.5`-derived point estimates, quantile intervals, and threshold exceedance probabilities for extreme high turbidity and anomalously high conductance conditions. A hydrologic similarity graph is introduced to connect stations using transferable physical similarity rather than station identity alone, thereby improving prediction under new-station conditions. Model performance will be evaluated under temporal extrapolation, spatial extrapolation to unseen stations, and joint spatiotemporal extrapolation. Beyond average predictive accuracy, the study will examine tail-event performance and identify the hydrologic and watershed factors associated with transfer failure. The proposed framework is expected to support more robust cross-region water-quality forecasting and risk-aware early warning.

## Writing Outline

Suggested paper structure:

1. Introduction
2. Study area and data
3. Methodology
4. Experimental design
5. Results
6. Discussion
7. Conclusion

Suggested key discussion points:

- why conductance and turbidity transfer differently
- whether static watershed attributes help unseen-station prediction
- whether tail-aware learning improves event detection
- where the model still fails and why

## Nearby References

- Zheng et al., 2025, cross-basin representation learning:
  - https://doi.org/10.1038/s41545-025-00466-2
- Bi et al., 2025, spatial-temporal graph fusion for water quality:
  - https://doi.org/10.1109/TASE.2025.3535415
- Nguyen et al., 2023, water-quality extremes via composite quantile regression neural network:
  - https://doi.org/10.1007/s10661-022-10870-7
- Smith et al., 2024, machine learning for stream specific conductance:
  - https://doi.org/10.1021/acs.est.4c05004
- Kemper et al., 2025, turbidity forecasting with National Water Model covariates:
  - https://doi.org/10.1111/1752-1688.70011

## Next Practical Steps

1. add a cached merged bundle for the current real dataset to reduce the first-run loading cost
2. add progress logging around dataset loading, feature construction, horizon start, and model start
3. complete the first real-data Linux run with `linear_tail` and `gbdt_tail` as sanity-check baselines
4. verify the first GPU-backed `torch_tail` run on `graph_backend = none`
5. compare `torch_tail` across `none | neighbor_stats | gnn`
6. only after the tabular route is stable, expand to `seq_tcn_tail` and `seq_transformer_tail`
7. report both overall performance and tail-event performance under unseen-station and joint spatiotemporal extrapolation

## Implementation Update (2026-03-23)

A new data-ingestion stabilization pass has now been completed for the real multi-file dataset.

What changed:

- coverage-aware smoke subset selection:
  - when `station_limit` is enabled, the loader no longer defaults to the first N WQ stations
  - it now prefers a subset that covers all configured targets, which is especially important for `conductance` + `turbidity` multi-task smoke runs
- parquet caching at the bundle level:
  - the directory-style loader now writes component caches for `attributes`, `time_series`, and `wq_observations`
  - cache namespaces are derived from the current dataset roots and smoke-subset specification so repeated smoke runs can reuse cached tables safely
- float32 downcast:
  - numeric columns are downcast after bundle loading and again after feature construction to reduce memory usage on sparse station-day frames
- smoke split adjustment for sparse targets:
  - the Linux smoke config now uses `unseen_station`
  - the main Linux experiment config still keeps `unseen_station_and_future`

Why this mattered:

- the previous smoke run already proved that the code path was runnable, but the dataset-loading stage still dominated wall-clock time
- sparse targets can disappear from the train partition under a strict spatiotemporal smoke split even when those targets are present in the selected stations overall
- the new smoke configuration is therefore meant to verify the end-to-end multi-target pipeline first, before returning to the stricter main split for full experiments

Current observed server status after the update:

- cold cache build on the 32-station Linux smoke subset successfully wrote:
  - `attributes.parquet`
  - `time_series.parquet`
  - `wq_observations.parquet`
- warm cache reuse reduced dataset loading to sub-second scale on the same smoke subset
- the updated smoke run now produces finite thresholds for both targets, including `turbidity`

## Revised Next Practical Steps

1. finish the cached Linux smoke run and archive its exact metrics as the new baseline sanity check
2. rerun the main Linux config with `unseen_station_and_future` to quantify how much sparse-target coverage remains under the true research split
3. if turbidity remains too sparse under the main split, decide explicitly whether the paper task should be:
   - next-day daily prediction, or
   - next-observed-value prediction
4. after the split definition is fixed, compare `torch_tail` across `none | neighbor_stats | gnn`
5. only then expand the real-data sequence experiments to `seq_tcn_tail` and `seq_transformer_tail`

## Paper-facing scope decision (2026-03-24)

The manuscript-facing study is now explicitly restricted to the strict ungauged setting.

This means:

- no target-basin `specific_conductance` history should enter the main model inputs
- the conductance main line should use `features.include_target_history_features: false`
- history-aware conductance runs can exist as internal diagnostics, but should not appear in the paper's main reported results
- `naive_last_conductance` should also be treated as internal diagnostic analysis rather than a reported manuscript baseline for the ungauged paper line

The paper-facing conductance story should therefore be framed as:

- conductance-only
- no target-history features
- unseen-station and future-period generalization
- tail-aware and probabilistic evaluation under strict ungauged assumptions

## Static attribute set update (2026-03-23)

The formal experiment configuration has now moved beyond the initial 15-feature core static set.

The current formal-study static set keeps the original core attributes and adds 20 more interpretable variables covering:

- hydrologic distribution (`Q50`, `Qstd`, `Q10`)
- climate distribution (`P50`, `P95`, `PEstd`)
- vegetation and biomass (`LAIstd`, `agBiomass`)
- irrigation and managed water use (`iaFmean`, `IWUmean`, `IWU_SI`)
- reservoir / public supply water management (`RS_mean`, `RS_std`, `PSW_SW_mean`, `PSW_SW_si`, `PSW_GW_mean`, `PSW_GW_si`)
- soil retention and texture (`Wsat`, `WPnt`, `vf_clay_s`)

This choice intentionally favors variables that are both predictive in the current station-level screening and easy to explain in the final paper. Highly redundant alternatives are still excluded for now.


## Dynamic temperature feature update (2026-03-23)

The active experiment configs now keep `tmin` and `tmax` but remove `air_temp` from the default dynamic feature set. This keeps the temperature block easier to interpret and avoids sending a direct linear combination of the same day-level temperature signal into the main model by default.


## Model-frame optimization update (2026-03-23)

After the full-data formal run successfully built source-level parquet caches, the next failure point was traced to feature construction rather than raw file ingestion. The `build_model_frame` path is now being refactored toward batched lag/rolling feature assembly with more explicit logging so that the true post-cache bottleneck can be measured on the next rerun.

## Post-audit experiment restructuring update (2026-03-23)

The project has now moved from a generic joint formal setup toward a more defensible split between:

- a conductance-first formal line, and
- a separate turbidity pilot line

Three implementation changes matter here:

1. WQ cache cleanup
- the bundle loader now drops rows whose configured targets are all missing before writing `wq_observations.parquet`
- this directly follows the audit finding that the previous WQ cache kept too many structurally useless rows

2. Active removal of `lai` / `swe` from the main bundle configs
- `lai` and `swe` have been removed from the active dynamic feature lists
- `LAImean`, `LAIstd`, and `SWEmean` were also removed from the active static lists
- the current graph similarity feature set no longer uses `SWEmean`

3. Config split for the next study round
- `configs/dataset_bundle_linux_formal_conductance.yaml`
  - the next main formal configuration
  - conductance only
  - torch only
- `configs/dataset_bundle_linux_turbidity_later_era_pilot.yaml`
  - a separate later-era turbidity pilot
  - date-filtered from `2000-01-01`

This restructuring is meant to keep the main paper line aligned with the most stable target first, while still preserving a clear path to a later sparse-target turbidity study.
