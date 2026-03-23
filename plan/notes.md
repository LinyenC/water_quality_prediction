# Notes: Water Quality Data Audit

## Current Known Facts
- The project uses three major data blocks: attributes, time series, and water-quality observations.
- Separate data roots and source-level parquet caching are already implemented.
- Full Linux source caches have already been built successfully for the current formal dataset.
- The latest formal experiment now runs through source-cache loading quickly.
- A previous bottleneck in build_model_frame was reduced by batching lag/rolling feature construction.
- The completed formal run showed severe generalization problems and clear target-availability issues.
- In the formal run logs, several variables such as lai, swe, and many of their lag/rolling variants were skipped by the imputer due to no observed values in the relevant training data.
- Turbidity showed very weak usability under the current formal split, which makes a dedicated data audit especially important.

## Immediate Audit Priorities
- Explain which data are truly present versus only structurally expanded as daily NaN rows.
- Quantify target coverage per station and through time.
- Quantify feature sparsity after daily alignment and feature engineering.
- Check whether current split rules destroy target usability for turbidity.

## Phase 1-2 Findings Snapshot
- all-four overlap (attributes + time series + conductance + turbidity) is only 243 stations
- WQ cache contains 19,430,748 rows where both targets are NaN
- 24 WQ rows sit at 1900-01-01 and all of them are invalid target rows
- time-series cache has only 45,416 rows with all dynamic variables missing, much cleaner than WQ cache
- conductance has real historical depth before 1980; turbidity starts in 1998

## Phase 3-4 Findings Snapshot
- `unseen_station_and_future` uses `train_end_date = 1998-10-03` and `valid_end_date = 2010-07-16`
- turbidity begins on `1998-10-01`, so the formal train window only overlaps its earliest few days
- train turbidity survives as only `8` `target_h1` rows from `2` stations: `14178000` and `14179000`
- valid turbidity has `37,049` `target_h1` rows across `27` stations; test has `86,303` across `50` stations
- WQ cache row utility is very poor: `86.23%` of rows have both targets missing
- train all-null input feature count is `38`, exactly matching the model log warnings
- the all-null train features are dominated by the full `lai` family, the full `swe` family, and most longer-context `turbidity` history features

## Working Hypothesis
The current bottleneck is structural, not only numerical.

The main failure mode is a mismatch between:
- target start dates
- split cut dates
- family-specific feature coverage windows
- the current next-day target definition

Under the present formal setup, conductance remains feasible, but turbidity is effectively unlearnable during training.

## Final Diagnosis Snapshot
- The current joint next-day conductance+turbidity formal setup should not be used as the main paper experiment.
- The most defensible immediate main line is conductance-first under a validated split and feature set.
- Turbidity should be redesigned as a separate experiment line, either with a later-era split or a sparse-label-aware target definition.
- WQ cache generation should drop rows where both targets are missing.
- `lai` and `swe` should not remain in a formal feature set whose training era ends before those source families begin.

## LAI/SWE Role Check (2026-03-23)
Using the completed formal prediction files:
- in the current formal train split, `lai` and `swe` overlap with both conductance and turbidity targets at exactly `0%`
- in valid+test combined, overlap is high because those eras are later:
  - `lai` with conductance target_h1: `88.48%`
  - `swe` with conductance target_h1: `87.65%`
  - `lai` with turbidity target_h1: `89.89%`
  - `swe` with turbidity target_h1: `91.36%`

But the marginal correlation pattern is weak once station effects are removed:
- conductance vs `lai`: pooled Pearson `-0.328`, within-station `-0.028`
- conductance vs `swe`: pooled Pearson `-0.028`, within-station `0.028`
- turbidity vs `lai`: pooled Pearson `-0.096`, within-station `0.021`
- turbidity vs `swe`: pooled Pearson `-0.021`, within-station `0.003`

Interpretation:
- under the current formal split, `lai` and `swe` have zero train-time utility
- in later eras they may carry some broad between-station or climatological context, especially `lai` for conductance
- but they do not currently show strong within-station next-day signal for either target
- this suggests they are secondary context features at best, not core drivers for the present next-day setup
