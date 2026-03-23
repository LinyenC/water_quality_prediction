# Water Quality Data Audit Plan

## 1. Audit Goal
Before continuing formal modeling, we will run a complete audit of the current dataset to answer four questions:

1. What data do we really have?
2. How much of it is valid and usable?
3. How is that usability distributed across stations and time?
4. Which data issues are currently limiting model performance?

This audit should not stop at raw file statistics. It should connect raw data quality, post-merge availability, and actual train/valid/test usability under the current HydroTail task definition.

## 2. Why This Audit Matters Now
The recent formal run tells us the pipeline can execute, but it also exposed multiple warning signs:

- source-level ingestion is no longer the main bottleneck
- many engineered features become effectively empty after alignment and splitting
- turbidity remains difficult to learn under the current formal setup
- model quality is now limited more by data usability than by code execution

So the next high-value step is not another blind model tweak. It is to measure the dataset carefully enough that we know what problem we are really solving.

## 3. Audit Scope
The audit should cover five layers.

### Layer A: Raw source inventory
We will inspect each source before heavy feature engineering.

Objects to audit:
- attributes table
- time-series sources: streamflow, daymet, PE, LAI, SWE_SNODAS
- WQ targets: conductance and turbidity

Questions:
- How many files, stations, rows, and columns exist in each source?
- What are the date ranges for each source?
- Are there duplicate station-date rows?
- Are station IDs consistent across sources?
- Are value columns numeric and parseable?
- Are there impossible or suspicious values?

### Layer B: Station-level temporal coverage
We will measure the real usable time span per station and per variable.

Metrics:
- first valid date
- last valid date
- total calendar span
- total valid observations
- valid-observation density within span
- median gap between valid observations
- longest missing streak
- longest valid streak
- overlap span between each target and its dynamic covariates

This layer is important because two stations with the same date span can have very different usable density.

### Layer C: Missingness and effective availability
We will analyze missingness in three views.

View 1: Raw-source missingness
- missing rate by variable before daily-grid expansion

View 2: Daily-grid missingness
- missing rate after station-day alignment
- proportion of rows that are structural gap rows versus original observations

View 3: Model-frame missingness
- missing rate after lag and rolling generation
- columns with no observed values in train
- columns with near-zero observed values in valid/test

This layer should directly explain warnings like skipped imputer features.

### Layer D: Target quality and task feasibility
This is the most important layer for the paper task.

Metrics for conductance and turbidity:
- station count with at least one observation
- station count with at least N observations
- date span distribution
- observation interval distribution
- overlap between conductance and turbidity at station level
- overlap between targets and dynamic covariates at station-date level
- effective train/valid/test observed_count under current split
- target availability after horizon shifting
- target availability after non-null feature filtering

We should explicitly answer:
- why turbidity is weak under the current formal split
- whether the current daily next-step task is the right target definition
- whether turbidity should be treated with a different prediction target or split rule

### Layer E: Static attribute quality
We will inspect the selected static attributes and optionally the full attribute table.

Metrics:
- missing rate per attribute
- constant or near-constant columns
- extreme outliers and suspicious ranges
- high-correlation and redundancy groups
- per-station availability of the current 35 formal static features

This layer helps confirm whether current static selections are reasonable and sufficiently complete.

## 4. Outputs We Should Produce
The audit should produce a compact but decision-useful bundle.

### Required tables
- source inventory table
- station-level coverage summary table
- variable-level missingness table
- target usability table under current split
- all-missing or nearly-all-missing engineered feature table
- static attribute quality table

### Required figures
- per-source date coverage timeline
- histograms of valid observations per station
- missingness heatmaps for representative stations
- target overlap plots for conductance and turbidity
- train/valid/test target availability comparison plots

### Required written conclusions
- current dataset strengths
- current dataset blockers
- variables to keep, drop, or redefine
- whether the task definition should stay as next-day prediction or be revised
- concrete recommendations before the next main experiment

## 5. Recommended Execution Order
The audit should be done in this order so each step explains the next one.

1. Raw inventory and schema check
2. Station-level temporal coverage
3. Raw-source missingness
4. Daily-grid missingness
5. Model-frame missingness and feature usability
6. Split-aware target usability analysis
7. Static attribute audit
8. Final diagnosis and action list

## 6. Practical Implementation Strategy
To keep runtime manageable, we should avoid repeating the expensive raw-ingestion path unless necessary.

Recommended strategy:
- use existing source-level parquet caches for the main audit tables
- compute raw-source summaries only where source-level detail is needed
- separate raw-source statistics from post-feature-engineering statistics
- save all audit outputs under a dedicated audit directory so later experiments can reference them

## 7. Specific Questions the Audit Must Answer
These are the concrete questions I think we should force the audit to answer.

1. Why do lai and swe disappear so badly in the formal run?
2. Why does turbidity remain so sparse or unstable after splitting?
3. How much of the current model frame is real observation support versus synthetic daily expansion?
4. Which dynamic variables are usable only for some stations or some years?
5. Are there stations that dominate the target labels?
6. Is the formal split too strict for turbidity under the current target definition?
7. Which features should be removed from the next experiment because they are nearly always empty?

## 8. Suggested Deliverables After the Audit
At the end of the audit, we should be able to choose one of three paths with evidence.

Path A:
- keep the current task definition and clean feature selection

Path B:
- keep the model family but revise split rules or station filtering for sparse targets

Path C:
- redefine turbidity prediction as next-observed-value or another sparse-label-aware task

## 9. My Recommendation
I think this audit is the right next move.

Not because we lack code, but because the project has reached the point where data usability is now the main uncertainty. A complete audit will give us a much stronger basis for the next experiment, and it will also make the eventual paper story more convincing.

## 10. Progress Update (2026-03-23)
The audit has now completed Phases 1-4.

Completed outputs:
- `plan/data_audit_phase1_2_findings.md`
- `plan/data_audit_phase3_4_findings.md`
- server audit directory: `/data/data2/linyen3/wat_quality_pred/outputs/data_audit_phase1_2`
- server audit directory: `/data/data2/linyen3/wat_quality_pred/outputs/data_audit_phase3_4`

The strongest new conclusion is that the current formal `unseen_station_and_future` split is structurally incompatible with the next-day turbidity task. The split places almost the entire turbidity era into valid/test, leaving train with only 8 usable `target_h1` labels from 2 stations.

This means the next step should no longer be generic model tuning. It should be a final diagnosis and an experiment-redesign decision.
