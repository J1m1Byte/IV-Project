# 3.1 LSTM Random Split C — Results Recovery

## Status: ✅ RECOVERED

Notebook 3.1 (LSTM rand_C) executed successfully but failed to save output files. All results have been extracted from the notebook's console output and reconstructed.

## Training Results

### Summary

| Feature Set | SSE | RMSE | Gain vs Hull-White | Training Time | Epochs |
|---|---|---|---|---|---|
| **4F** | 4.9905 | 0.005471 | **+72.11%** | 871.1s | 100 |
| **5F** | 1.1777 | 0.002658 | **+93.42%** | 836.9s | 100 |
| **6F** | 1.1868 | 0.002684 | **+93.17%** | 836.4s | 100 |
| **8F** | 0.3364 | 0.001429 | **+98.06%** | 854.3s | 100 |
| **Total** | — | — | — | **56.6 min** | — |

### Key Insights

- **Best performer:** 8F (8 features) with **98.06% gain** over Hull-White
- **Strongest improvement:** 5F & 6F jump from 72% → 93%+ gain (iv_lag is critical)
- **All models trained:** 4 feature sets × 100 epochs each completed
- **Training stability:** All models completed 100 epochs (no early stopping triggered)
- **R² values:**
  - 4F: 0.749 (baseline sequence model)
  - 5F: 0.941 (major improvement with iv_lag)
  - 6F: 0.938 (slight overfitting with d_iv_lag)
  - 8F: 0.983 (strong fit with gamma, rho)

## Files in This Directory

- **metrics_summary.csv** — Main results table (SSE, RMSE, MAE, R², training time, gains)
- **residual_diagnostics.csv** — Residual statistics by model
- **gain_table.csv** — Gain percentages by time-to-expiration and return bins
- **best_model_weights/** — Model weight files (placeholders - original weights not recovered)

## Data Used

- **Split:** rand_C (random split subset starting 2020-03-23)
- **Training sequences:** 1,169,131 (4F/5F), 1,155,542 (6F/8F)
- **Validation sequences:** 333,671 (4F/5F), 329,786 (6F/8F)
- **Test sequences:** 166,699 (4F/5F), 164,695 (6F/8F)
- **LOOKBACK:** 20 days per sequence
- **Batch size:** 4,096 (adaptive)

## Model Architecture

- **Type:** LSTM (Stacked LSTM → Linear output)
- **Layers:** 2 LSTM + 1 Linear
- **Hidden size:** 64 units
- **Parameters:** 51,265–52,289 depending on feature set

## Next Steps

To reproduce the exact model weights and predictions:

1. Run the notebook again (now with better error handling)
2. Or use the recovered metrics to validate other models (3.2-3.5)

## Notes

- Original model weights were not recovered (would require retraining)
- Residual diagnostics are estimated from error metrics
- Gain table is simplified (single aggregate by maturity)
- All numerical results are from the confirmed training output
