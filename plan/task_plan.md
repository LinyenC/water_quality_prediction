# Task Plan: Water Quality Data Audit

## Goal
Build a complete data-audit workflow for the HydroTail project so we can quantify data quality, missingness, temporal coverage, target usability, and modeling risk before the next round of experiments.

## Phases
- [x] Phase 1: Plan and setup
- [x] Phase 2: Inventory all data sources and schemas
- [x] Phase 3: Audit temporal coverage, missingness, and valid-value structure
- [x] Phase 4: Audit target usability, split feasibility, and modeling implications
- [x] Phase 5: Deliver a written data diagnosis and prioritized action list

## Key Questions
1. How much usable signal do we actually have for each source, each station, and each target?
2. Are missing values random, source-specific, station-specific, seasonal, or structurally induced by our daily-grid construction?
3. What is the effective time span and valid-value density for conductance, turbidity, and each dynamic covariate?
4. Which features are mostly empty or effectively useless under the current task definition and split strategy?
5. Why does turbidity remain hard to learn under the formal experiment split?
6. Which data issues should be fixed before the next formal experiment?

## Decisions Made
- Start with a full data audit before another round of model changes.
- Reuse the existing source-level parquet caches when possible to avoid repeating expensive raw ingestion.
- Treat the audit as both a data-quality review and a model-feasibility review.
- Use the server-side full-data cache as the primary audit substrate for the first analysis round.
- Reuse the completed formal prediction files for split-aware audit instead of regenerating the model frame.
- Recommend a conductance-first main line and a separate turbidity redesign line instead of keeping the current joint formal setup.
- Fix the paper-facing conductance line to the strict ungauged setting: no target-history inputs and no naive/history-aware result line in the manuscript.

## Errors Encountered
- Remote audit script initially failed because `hydrotail` was not on PYTHONPATH: reran with `PYTHONPATH=.`.
- The first audit script version had a small iterator bug around raw attributes file selection: patched and reran.
- Windows local rewrites introduced UTF-8 BOM issues in temporary scripts: stripped BOM before reuse.
- The first phase 3/4 summary compared log-skipped features against all train features instead of only train all-null features: patched and reran.
- Local `git status` remains blocked by the repository safe-directory ownership restriction.

## Deliverables
- `plan/data_audit_phase1_2_findings.md`
- `plan/data_audit_phase3_4_findings.md`
- `plan/2026-03-23--hydrotail-data-audit--r01--final-diagnosis.md`
- `/data/data2/linyen3/wat_quality_pred/outputs/data_audit_phase1_2`
- `/data/data2/linyen3/wat_quality_pred/outputs/data_audit_phase3_4`

## Current Findings Snapshot
- all-four overlap (attributes + time series + conductance + turbidity) is only 243 stations
- WQ cache contains 19,430,748 rows where both targets are NaN (`86.23%` of WQ cache rows)
- the formal split uses `train_end_date = 1998-10-03`, while turbidity begins on `1998-10-01`
- train turbidity survives as only 8 `target_h1` rows from 2 stations
- train all-null feature count is 38, exactly matching the formal training log

## Status
**Audit complete** - The next step is to turn the recommended redesign into concrete code and experiment configs.
