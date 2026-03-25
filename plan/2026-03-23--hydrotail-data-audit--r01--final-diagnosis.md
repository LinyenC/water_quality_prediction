---
type: results-report
date: 2026-03-23
experiment_line: hydrotail-data-audit
round: 1
purpose: final-diagnosis
status: active
source_artifacts:
  - plan/data_audit_phase1_2_findings.md
  - plan/data_audit_phase3_4_findings.md
  - /data/data2/linyen3/wat_quality_pred/outputs/data_audit_phase1_2
  - /data/data2/linyen3/wat_quality_pred/outputs/data_audit_phase3_4
linked_experiments:
  - /data/data2/linyen3/wat_quality_pred/outputs/formal_torch_static35_run
linked_results:
  - /data/data2/linyen3/wat_quality_pred/outputs/formal_torch_static35_run/horizon_1/torch_tail
---

# HydroTail Data Audit / Round 1 / Final Diagnosis / 2026-03-23

## Executive Summary
This audit changes the project diagnosis in an important way.

The main blocker is not that the model family is too weak. The dominant blocker is that the current dataset, split rule, and task definition are structurally misaligned.

The most important facts are:
- only 243 stations lie in the overlap of attributes, time series, conductance, and turbidity
- the WQ cache contains 19,430,748 rows where both targets are missing
- the current formal split uses `train_end_date = 1998-10-03`
- turbidity starts on `1998-10-01`
- under the current formal next-day setup, train turbidity survives as only 8 `target_h1` rows from 2 stations
- train contains 38 all-null input features, dominated by the full `lai` family, the full `swe` family, and most longer-context turbidity-history features

This means the current joint conductance+turbidity next-day experiment should not be treated as a faithful scientific test. It is structurally biased against turbidity and partially incompatible with later-coverage feature families.

## Experiment Identity and Decision Context
This report synthesizes the full data audit completed after the first end-to-end formal run. The purpose is not to restate raw statistics, but to decide what experiment design should be considered scientifically valid for the next round.

The audit combined:
- source inventory and station overlap analysis
- cache row-utility analysis
- split-aware target usability analysis from completed formal predictions
- train-time missing-feature analysis from the formal training log

This repository is not bound to an Obsidian project-memory registry, so this report is stored locally in `plan/` and no Obsidian write-back was attempted.

## Setup and Evaluation Protocol
Evidence used in this diagnosis comes from four sources:
- local audit summaries: `plan/data_audit_phase1_2_findings.md`, `plan/data_audit_phase3_4_findings.md`
- server audit outputs: `/data/data2/linyen3/wat_quality_pred/outputs/data_audit_phase1_2`, `/data/data2/linyen3/wat_quality_pred/outputs/data_audit_phase3_4`
- the completed formal run: `/data/data2/linyen3/wat_quality_pred/outputs/formal_torch_static35_run`
- the formal torch prediction files and training log inside that run directory

The current formal setting that matters most for interpretation is:
- split strategy: `unseen_station_and_future`
- train end date: `1998-10-03`
- valid end date: `2010-07-16`
- target definition: next-day target (`horizon = 1`)

## Main Findings

### 1. Raw data volume is large, but usable overlap is much smaller
The project has large raw tables and caches, but the scientifically usable overlap is much tighter than the source counts suggest.

Stable facts:
- attributes: 855 stations
- time series cache: 1,089 stations
- conductance: 1,039 stations
- turbidity: 394 stations
- full overlap of attributes + time series + conductance + turbidity: 243 stations

So the practical multi-source, multi-target station pool is already constrained before any splitting or feature engineering.

### 2. The WQ cache currently contains too many structurally useless rows
`wq_observations.parquet` has 22,534,657 rows, but 19,430,748 of them have both targets missing. That is 86.23 percent of the full WQ cache.

This is not a harmless detail. It inflates storage, increases row alignment overhead, and makes later analyses noisier. It also means that the current WQ cache is not a clean representation of “observed water-quality rows.”

### 3. Conductance and turbidity have very different structural support
Conductance and turbidity differ less in per-station density than in station support and start date.

Conductance:
- 2,915,282 valid rows
- 1,039 stations
- starts on 1939-01-13

Turbidity:
- 867,217 valid rows
- 394 stations
- starts on 1998-10-01

This difference becomes decisive once a strict time-based split is applied.

### 4. The current formal split is compatible with conductance but incompatible with turbidity
Under the completed formal run:
- train conductance `target_h1`: 548,902 rows
- valid conductance `target_h1`: 110,725 rows
- test conductance `target_h1`: 183,271 rows

But for turbidity:
- train `target_h1`: 8 rows from 2 stations
- valid `target_h1`: 37,049 rows from 27 stations
- test `target_h1`: 86,303 rows from 50 stations

The root cause is direct and concrete:
- turbidity begins on 1998-10-01
- the train period ends on 1998-10-03

So the training window only overlaps the first few days of the turbidity era. The resulting train split is not a realistic learning substrate for turbidity.

### 5. The surviving train turbidity signal comes from only two stations
The only stations contributing train turbidity next-day targets are:
- `14178000`: 4 rows
- `14179000`: 4 rows

Both belong to `train_station_early`.

This is too little signal to support a meaningful joint multi-task turbidity claim.

### 6. Some feature families are structurally impossible under the current split
The formal training log and the audit agree exactly on 38 train all-null features.

These are dominated by:
- the full `lai` family: raw + lag + rolling, 14 features
- the full `swe` family: raw + lag + rolling, 14 features
- most longer-context turbidity-history features: 10 features

This is not random missingness. It is a deterministic consequence of the current train window and source coverage windows.

`lai` starts in 2000 and `swe` starts in 2004, but the current train period ends in 1998. Therefore these families cannot contribute any train signal under the present formal setting.

## Failure Cases / Limitations
The previous formal run should not be over-interpreted as evidence that “the model fails on turbidity.” That conclusion is too coarse.

A more accurate reading is:
- conductance was given a feasible training substrate
- turbidity was not
- some feature families were included in configuration even though the train window made them unavailable by construction

So the prior bad formal result is partly a model outcome, but more importantly a task-design outcome.

## What Changed Our Belief
Before the audit, the main uncertainty looked like model quality and engineering stability.

After the audit, the more accurate belief is:
- the pipeline can run
- the data caches can be audited and reused
- the main research risk is now experimental validity, not execution feasibility
- the current joint next-day setup is acceptable for conductance, but not for turbidity
- the next paper-facing result should not continue with the current joint formal design unchanged

## Recommended Research Strategy
I recommend splitting the project into a stable main line and a separate sparse-target line.

### Recommended main line: strict-ungauged conductance-first
Use conductance as the main paper-quality task for the next formal round, but keep the deployment assumption strict: no target-basin conductance history in the model inputs.

Why:
- it has broad station support
- it remains learnable under strict station+time splitting
- it is not structurally destroyed by the current target era
- it gives a cleaner basis for evaluating the HydroTail modeling idea itself

Recommended conductance-first configuration:
- predict `conductance` only in the main formal run
- keep `unseen_station_and_future` if the paper goal is true spatial-temporal generalization
- if the split still starts before 2000, remove `lai` and `swe` from the main conductance feature set
- keep the dynamic families that are actually available in the train era: `discharge`, `precip`, `tmin`, `tmax`, `pe`
- do not make history-aware conductance or naive persistence runs part of the manuscript-facing evidence line

This is the most stable path to a defensible main result.

### Recommended secondary line: turbidity-specific redesign
Do not keep turbidity inside the current joint next-day formal task.

There are two realistic redesign paths.

Path T1: later-era turbidity next-day experiment
- define a later start year so train actually contains a meaningful turbidity era
- likely restrict the study to years where the desired covariates exist, especially if `lai` and `swe` are to be kept
- choose a split that still tests generalization but does not leave train nearly empty

Path T2: sparse-label-aware turbidity task
- redefine turbidity as next-observed-value prediction or another sparse-label-aware target
- stop forcing daily next-step supervision on a target whose effective train era is too short under the current split rule

Between the two, T1 is the easier short-term path. T2 is the more methodologically interesting path if the paper wants to emphasize sparse-target water-quality forecasting.

## Priority Action List

### Must do before the next formal run
1. Clean WQ cache generation so rows with both targets missing are dropped before downstream use.
2. Stop treating the current joint conductance+turbidity next-day formal setup as a valid main experiment.
3. Remove feature families that are impossible under the chosen train window, especially `lai` and `swe` if the train period remains pre-2000.
4. Decide explicitly whether the next main run is conductance-first or a redesigned turbidity study.

### Strongly recommended next
1. Add cache-level auditing metadata so future runs record valid-row fractions by source automatically.
2. Add family-aware feature gating so features with zero train support are removed before fitting instead of only surfacing in imputer warnings.
3. Record split cut dates and per-target usable counts as first-class experiment outputs.

## Concrete Experiment Redesign Proposal
I recommend the following sequence.

### Step 1: stabilize the main result
- run a conductance-only formal experiment
- keep the current strong generalization split if that is the intended claim
- drop `lai` and `swe` from that line unless the training era is moved later
- keep the current torch-based main model

### Step 2: isolate turbidity as a separate research question
- create a turbidity-only audit or pilot config
- test a later-era split first to verify that train contains real signal
- only after that decide whether a sparse-label-aware formulation is needed

### Step 3: only then revisit multi-task coupling
Once conductance-only and turbidity-specific settings are both individually valid, revisit whether joint learning actually helps. Do not use the current invalid joint setup as evidence either for or against multi-task modeling.

## Stable Conclusions vs Tentative Conclusions

Stable conclusions:
- WQ cache quality is currently poor because too many rows have no targets at all.
- The current formal split is structurally incompatible with the next-day turbidity task.
- `lai` and `swe` cannot contribute under a train period that ends before their coverage windows.
- Conductance is the more stable target for the next defensible formal run.

Tentative conclusions:
- A later-era turbidity next-day setup may be feasible, but this still needs its own pilot verification.
- A sparse-label-aware turbidity formulation may become necessary, but this is not yet proven to be the best redesign.
- Multi-task coupling may still be useful later, but the current evidence cannot answer that question fairly.

## Artifact and Reproducibility Index
- phase 1-2 findings: `plan/data_audit_phase1_2_findings.md`
- phase 3-4 findings: `plan/data_audit_phase3_4_findings.md`
- local audit scripts: `temp/data_audit_phase1_2.py`, `temp/data_audit_phase3_4.py`
- server audit outputs: `/data/data2/linyen3/wat_quality_pred/outputs/data_audit_phase1_2`, `/data/data2/linyen3/wat_quality_pred/outputs/data_audit_phase3_4`
- formal run analyzed here: `/data/data2/linyen3/wat_quality_pred/outputs/formal_torch_static35_run`
