## Scope

This note summarizes five strict-ungauged conductance schemes evaluated under the same `horizon=1` setup:

1. `torch_tail_no_target_history`
2. `seq_tcn_tail`
3. `seq_retrieval_prototype_tail`
4. `seq_retrieval_prototype_tail_nograph`
5. `seq_retrieval_prototype_tail_noproto`

The full metric dump is stored in:

- `2026-03-25--strict-ungauged-five-models--metrics-wide.csv`

## Main Observations

### 1. Current best overall method remains `torch_tail_no_target_history`

On the main `test` split:

- `R2 = 0.6439`
- `MAE = 385.38`
- `RMSE = 741.28`
- `AUC = 0.8753`
- `F1 = 0.7193`
- `tail_R2 = -0.5548`

It is still the strongest all-around strict-ungauged model among the five schemes.

### 2. `seq_tcn_tail` is the strongest sequence baseline, but still below the tabular main line

On `test`:

- `R2 = 0.5443`
- `MAE = 422.13`
- `AUC = 0.8868`
- `tail_R2 = -1.4152`

Compared with `torch_tail_no_target_history`, it preserves reasonable classification quality but loses substantial regression quality and tail magnitude accuracy.

### 3. Full retrieval helps test-set regression slightly vs `seq_tcn_tail`, but is unstable

`seq_retrieval_prototype_tail` reaches:

- `test R2 = 0.5736`
- `test RMSE = 806.46`

This is slightly better than `seq_tcn_tail` in `R2/RMSE`, but:

- `valid R2 = -2.5895`
- `valid RMSE = 2220.31`

So the current retrieval design is not stable enough to serve as the main method.

### 4. Removing graph restriction improves validation stability but hurts test transfer

`seq_retrieval_prototype_tail_nograph` gives:

- `valid R2 = 0.4923`
- `test R2 = 0.3986`

This suggests graph-restricted donor retrieval may be too narrow, but simply removing that restriction does not solve the overall transfer problem.

### 5. Removing prototypes clearly hurts the retrieval model

`seq_retrieval_prototype_tail_noproto` is the weakest model overall:

- `test R2 = 0.0640`
- `test MAE = 756.82`
- `valid R2 = -4.6191`

This indicates prototype representations are not merely decorative in the current retrieval pipeline.

### 6. Tail magnitude regression remains difficult for all five schemes

All five models still have negative `test tail_R2`.

The least-bad result is from `torch_tail_no_target_history`:

- `test tail_R2 = -0.5548`

Sequence retrieval variants are materially worse:

- `seq_tcn_tail`: `-1.4152`
- `seq_retrieval_prototype_tail`: `-1.4442`
- `seq_retrieval_prototype_tail_nograph`: `-3.4259`
- `seq_retrieval_prototype_tail_noproto`: `-2.5424`

This reinforces the earlier conclusion that, under strict ungauged settings, tail event identification is currently more reliable than exact tail magnitude prediction.

## Decision-Oriented Takeaways

### Keep as main line

- `torch_tail_no_target_history`

### Keep as sequence baseline

- `seq_tcn_tail`

### Treat as experimental retrieval direction

- `seq_retrieval_prototype_tail`

### Main retrieval diagnostics learned from this ablation

- Graph-restricted donor pools may be too restrictive.
- Prototype removal is harmful.
- Current retrieval transfer is not yet reliable enough to replace the main tabular baseline.

## Recommended Next Step

Instead of expanding retrieval complexity immediately, the next round should answer one narrower question:

> Can event-level donor retrieval outperform station-level donor retrieval while preserving validation stability?

That is a better next retrieval iteration than adding more architectural components.
