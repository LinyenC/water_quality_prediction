# Phase 3-4 Findings: Water Quality Data Audit

## Scope Completed
This round completed the missingness, row-utility, and split-aware target-usability parts of the audit.

Server-side artifacts were written to:
- `/data/data2/linyen3/wat_quality_pred/outputs/data_audit_phase3_4`

Main generated files:
- `cache_row_utility_summary.csv`
- `wq_invalid_rows_by_station.csv`
- `split_target_usability.csv`
- `station_split_target_counts.csv`
- `split_group_target_usability.csv`
- `target_year_distribution.csv`
- `feature_non_null_summary.csv`
- `feature_family_non_null_summary.csv`
- `train_all_null_features.csv`
- `skipped_features_from_log.csv`
- `skipped_feature_comparison.csv`
- `phase3_4_summary.md`

## Key Findings

### 1. WQ cache row utility is extremely poor compared with the time-series cache
- `time_series.parquet`: 13,905,291 total rows, only 45,416 rows with all dynamic variables missing (`0.33%`)
- `wq_observations.parquet`: 22,534,657 total rows, 19,430,748 rows with both targets missing (`86.23%`)

This confirms that the WQ cache, not the time-series cache, is currently the much noisier substrate for downstream modeling.

### 2. The formal split almost completely destroys turbidity training usability
The formal split metadata shows:
- `strategy = unseen_station_and_future`
- `train_end_date = 1998-10-03`
- `valid_end_date = 2010-07-16`

But the earliest valid turbidity observations begin on `1998-10-01`.

Under this split:
- `train / turbidity / current_value = 6` rows from `2` stations
- `train / turbidity / target_h1 = 8` rows from `2` stations
- `valid / turbidity / target_h1 = 37,049` rows from `27` stations
- `test / turbidity / target_h1 = 86,303` rows from `50` stations

So the main turbidity problem is not generic sparsity inside covered stations. It is that the formal time split places almost the entire turbidity era into valid/test, leaving train with virtually no next-day target support.

### 3. All surviving train turbidity targets come from only two stations
The only train stations carrying `turbidity target_h1` are:
- `14178000`: 4 rows, `1998-09-30` to `1998-10-03`
- `14179000`: 4 rows, `1998-09-30` to `1998-10-03`

All of them are in `train_station_early`.

This is strong evidence that the current joint conductance+turbidity formal task is not feasible under the present split rule.

### 4. Conductance is not suffering from the same split collapse
For comparison:
- `train / conductance / target_h1 = 548,902`
- `valid / conductance / target_h1 = 110,725`
- `test / conductance / target_h1 = 183,271`

So the current formal split is compatible with conductance, but not with turbidity.

### 5. The train split has exactly 38 all-null input features, and they match the training log perfectly
The audit-derived train all-null feature set matches the model log one-for-one.

The 38 unusable train features are dominated by three families:
- the full `lai` family: raw + lag + rolling (`14` features)
- the full `swe` family: raw + lag + rolling (`14` features)
- most of the `turbidity` history family beyond the shortest context (`10` features)

This means the model log warnings were not incidental. They reflect real, systematic feature collapse under the current split and feature-construction setup.

### 6. Missingness after split is highly family-specific rather than globally random
From `feature_family_non_null_summary.csv`:
- train `discharge / precip / tmin / tmax / pe` families still have non-null ratios around `0.65`
- train `specific_conductance` features remain highly available
- train `lai` and `swe` families are completely absent
- train `turbidity` family keeps only a tiny residue at the shortest context and loses the rest

This suggests the next round of data or task redesign should be family-aware rather than based on a single global missingness threshold.

## Interpretation
The second audit round changes the diagnosis substantially.

The main blocker is no longer just “too many NaNs.” The dominant problem is a structural mismatch between:
- target start dates
- split cut dates
- feature family availability windows
- the next-day task definition

In particular, turbidity is currently being evaluated under a split that is almost guaranteed to make it unlearnable in training.

## Recommended Next Steps
1. Drop WQ rows where both targets are missing during cache generation.
2. Do not use the current `unseen_station_and_future` split for the joint next-day turbidity task.
3. Either shift the training era later, or give turbidity its own split / target definition.
4. Reconsider whether `lai` and `swe` should be kept in the formal feature set when the split starts before their coverage windows.
5. Before the next formal run, decide whether the paper should center on:
   - conductance as the stable main task, with turbidity as a secondary sparse-target study
   - or a revised turbidity formulation such as a sparse-label-aware next-observed-value task
