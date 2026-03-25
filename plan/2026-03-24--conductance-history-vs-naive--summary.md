# Conductance History vs Naive Summary (2026-03-24)

Internal diagnostic only.
This note is excluded from the paper-facing strict-ungauged study and should not be used as a manuscript result table.

Full metrics are stored in `2026-03-24--conductance-history-vs-naive--metrics-long.csv`.

## Test Summary

| Horizon | Method | R2 | MAE | RMSE | AUC | F1 | Precision | Recall | Tail Count | Tail MAE | Tail RMSE | Tail R2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | torch_tail | 0.9572 | 44.3155 | 138.4944 | 0.9993 | 0.7301 | 0.5795 | 0.9865 | 2596 | 415.8296 | 884.0867 | 0.7294 |
| 1 | naive_last_conductance | 0.9729 | 28.8780 | 108.9640 | 0.9994 | 0.9265 | 0.9265 | 0.9265 | 2544 | 347.6336 | 639.3259 | 0.8508 |
| 2 | torch_tail | 0.9186 | 66.1369 | 190.9111 | 0.9982 | 0.7250 | 0.5816 | 0.9622 | 2596 | 581.7831 | 1141.3034 | 0.5491 |
| 2 | naive_last_conductance | 0.9444 | 45.0462 | 154.6291 | 0.9984 | 0.8853 | 0.8858 | 0.8848 | 2499 | 498.0015 | 858.8304 | 0.7203 |
| 3 | torch_tail | 0.8949 | 77.7358 | 217.0072 | 0.9962 | 0.8096 | 0.7108 | 0.9403 | 2596 | 708.1685 | 1277.1283 | 0.4354 |
| 3 | naive_last_conductance | 0.9233 | 55.1689 | 180.3450 | 0.9972 | 0.8545 | 0.8557 | 0.8533 | 2461 | 603.5962 | 995.4300 | 0.6127 |
| 7 | torch_tail | 0.8421 | 106.1282 | 266.0564 | 0.9892 | 0.7039 | 0.5687 | 0.9233 | 2596 | 909.0818 | 1489.6250 | 0.2319 |
| 7 | naive_last_conductance | 0.8756 | 74.7696 | 225.3985 | 0.9953 | 0.7942 | 0.7973 | 0.7912 | 2366 | 798.4063 | 1240.6628 | 0.3465 |
| 10 | torch_tail | 0.8262 | 115.6214 | 279.1779 | 0.9893 | 0.7157 | 0.8108 | 0.6406 | 2596 | 1006.6530 | 1535.3508 | 0.1840 |
| 10 | naive_last_conductance | 0.8571 | 82.7696 | 241.2170 | 0.9949 | 0.7736 | 0.7774 | 0.7698 | 2346 | 873.5050 | 1324.0483 | 0.2596 |
| 14 | torch_tail | 0.7990 | 132.3858 | 300.2696 | 0.9919 | 0.7320 | 0.6295 | 0.8744 | 2596 | 1071.3127 | 1617.9769 | 0.0938 |
| 14 | naive_last_conductance | 0.8367 | 90.1591 | 258.7502 | 0.9937 | 0.7508 | 0.7570 | 0.7446 | 2334 | 977.0955 | 1452.5789 | 0.1300 |
| 30 | torch_tail | 0.7115 | 163.5869 | 359.8409 | 0.9775 | 0.4905 | 0.3502 | 0.8186 | 2596 | 1338.7237 | 1895.8808 | -0.2453 |
| 30 | naive_last_conductance | 0.7732 | 110.8937 | 304.5540 | 0.9887 | 0.6655 | 0.6768 | 0.6545 | 2278 | 1234.4779 | 1709.0166 | -0.2162 |

## Note

- `naive_last_conductance` uses current-day `specific_conductance` as the point forecast for horizon `h`.
- Interval and pinball metrics are left as `NA` for the naive baseline in the long-form CSV because it does not output predictive quantiles.
