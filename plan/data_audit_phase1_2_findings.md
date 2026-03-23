# Phase 1-2 Findings: Water Quality Data Audit

## Scope Completed
This round completed the first two planned audit phases:

- raw source inventory
- station-level temporal coverage summary

Server-side artifacts were written to:
- `/data/data2/linyen3/wat_quality_pred/outputs/data_audit_phase1_2`

Main generated files:
- `source_inventory.csv`
- `station_variable_coverage.csv`
- `variable_coverage_summary.csv`
- `audit_metadata.json`
- `phase1_2_summary.md`

## Key Findings

### 1. Source sizes are large, but station support is uneven
- attributes raw table: 855 rows, 855 stations, 83 columns
- time-series cache: 13,905,291 rows, 1,089 stations
- WQ cache: 22,534,657 rows, 1,100 stations
- conductance support: 1,039 stations
- turbidity support: 394 stations

### 2. Station overlap is a major constraint
- attributes and time series overlap on 850 stations
- attributes and conductance overlap on 850 stations
- attributes and turbidity overlap on only 243 stations
- time series and conductance overlap on 1,033 stations
- time series and turbidity overlap on 383 stations
- all four blocks together overlap on only 243 stations

This means the real station pool for a fully informed multi-source, multi-target experiment is much smaller than the raw source counts suggest.

### 3. Dynamic covariates have clean within-span coverage, but their time spans differ
- streamflow: 1980-01-01 to 2021-12-31, 974 stations
- precip / tmin / tmax / pe: 1980-01-01 to 2021-12-31, 925 stations
- LAI: 2000-01-01 to 2020-12-31, 1,056 stations
- SWE: 2004-01-01 to 2022-12-31, 1,056 stations

Within each variable's own available span, median density is effectively 1.0. So the main issue for these sources is not random missingness inside coverage windows, but coverage mismatch across source families and stations.

### 4. Water-quality targets are much less aligned than the dynamic sources
- conductance: 2,915,282 valid rows across 1,039 stations, date span 1939-01-13 to 2022-04-30
- turbidity: 867,217 valid rows across 394 stations, date span 1998-10-01 to 2022-04-30

Station-level median valid counts are still substantial:
- conductance median station valid count: 1,830
- turbidity median station valid count: 1,780

So turbidity is not sparse per covered station, but sparse in station support and likely more vulnerable under strict split rules.

### 5. The very early conductance history is real, but the cache also contains clearly invalid WQ rows
- conductance has 330,787 valid rows before 1980 across 213 stations
- earliest valid conductance date is 1939-01-13
- turbidity begins at 1998-10-01

However, the WQ cache also contains a large number of rows with no valid targets at all:
- rows where both conductance and turbidity are NaN: 19,430,748
- rows at date `1900-01-01`: 24
- all `1900-01-01` rows have both targets missing

This means the current `wq_observations.parquet` contains many structurally useless rows that should not survive into later audit or modeling stages.

### 6. The time-series cache is much cleaner than the WQ cache
- total time-series rows: 13,905,291
- rows where all dynamic variables are NaN: 45,416

So the WQ cache is currently the much noisier cache layer from a row-utility perspective.

## Interpretation
The first audit round suggests that the project is limited less by raw file availability and more by cross-source overlap, target support imbalance, and cache-level row utility.

The single most important structural fact so far is:
- only 243 stations currently sit in the overlap of attributes, time series, conductance, and turbidity

This is already enough to explain why turbidity becomes fragile under strict formal splitting.

## Recommended Next Audit Steps
1. quantify missingness before and after daily-grid expansion
2. quantify how many engineered features are all-missing or near-all-missing in train / valid / test
3. audit split-aware target usability after horizon shifting and feature filtering
4. inspect whether WQ cache generation should drop rows with all targets missing
5. decide whether turbidity should keep the current next-day task definition or move to a sparse-label-aware target definition


## Split-Aware Usability Snapshot from the Completed Formal Run
Using the completed `unseen_station_and_future` formal run:

- train rows: 548,902
  - conductance observed: 548,902
  - turbidity observed: 8
- valid rows: 115,712
  - conductance observed: 110,725
  - turbidity observed: 37,049
- test rows: 199,530
  - conductance observed: 183,271
  - turbidity observed: 86,303

This is an extremely strong signal that the current formal split is effectively destroying turbidity usability in training, even though turbidity is not especially sparse within stations that actually have it.

## Feature-Utility Note from the Formal Training Log
The completed formal run also shows that the following feature families are effectively unusable during fitting:

- `lai` and many of its lag / rolling variants
- `swe` and many of its lag / rolling variants
- most longer-horizon turbidity lag / rolling variants beyond the shortest lag

So the current problem is not only target sparsity. It is also that some feature blocks survive the source-cache stage but remain functionally empty by the time the training split is formed.
