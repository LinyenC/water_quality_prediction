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

Recommended baseline groups:

- `Persistence`
- `Linear Regression` / `Elastic Net`
- `LightGBM` / `XGBoost`
- `LSTM`
- `TFT`
- multi-task model without graph
- graph model without tail heads

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

1. check data completeness for `specific conductance` and `turbidity`
2. unify station IDs, timestamps, and units
3. define unseen-station split
4. create lag and rolling features
5. collect static watershed attributes
6. build strong non-deep baseline first
7. implement `HydroTail-GMT`
8. run tail-event and generalization evaluations
