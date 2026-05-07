# Neural Network Models for Implied Volatility Movements

Predicting daily changes in SPY implied volatility using feedforward, LSTM, GRU, and Temporal Fusion Transformer networks across four date-based market regimes.

## Results

### Sequence Models (best per architecture and window)

| Dataset | Window | LSTM Gain | GRU Gain | TFT Gain | Best Model |
|---------|--------|-----------|----------|----------|------------|
| `chro_A` | Full sample (2013–2026) | 62.11% | **73.36%** | 35.27% | GRU `8F rho` |
| `chro_B` | Pre-COVID (2013–2020) | 91.73% | **94.06%** | 89.05% | GRU `6F` |
| `chro_C` | Post-COVID (2020–2026) | 79.72% | **90.64%** | 73.80% | GRU `6F` |
| `chro_D` | Recent (2023–2026) | 57.96% | **70.76%** | 73.32% | TFT `8F theta` |

GRU dominates across periods. All gains measured against the Hull-White analytic benchmark.

### FC Feature Sweep (best per random split)

| Notebook | Best feature set | Gain vs analytic | SSE |
|----------|-----------------|-----------------|-----|
| `2.0-fc-rand-A-colab` | `3F+vix_lag+vix_mom_lag+vix_mom+gamma+rho` | 45.57% | 62.75 |
| `2.1-fc-rand-B-colab` | `3F+vix_lag+vix_mom_lag+vix_mom+theta+vega` | 25.72% | 28.69 |
| `2.2-fc-rand-C-colab` | `3F+vix_lag+vix_mom_lag+vix_mom+gamma+theta` | 47.61% | 34.01 |
| `2.3.1-fc-rand-D-colab-ivlag` | `3F+iv_lag+vix_lag+vix_mom_lag+vix_mom+gamma+theta+vega` | 41.55% | 4.89 |

---

## Workflow

1. Download SPY option chain and market data (SPY returns, VIX, risk-free rate)
2. Clean and filter the option chain (11M → 7M rows through 8 quality filters)
3. Merge option and index data; engineer target and feature columns
4. Partition into four date-based windows (A–D) with chronological and random train/val/test splits
5. Train FC baselines on random splits (series 1 & 2), then sequence models on chronological splits (series 3–5)
6. Evaluate all models against the Hull-White analytic benchmark

---

## Data Sources

| Variable | Source |
|----------|--------|
| SPY option chain | OnclickMedia |
| SPY close prices and daily returns | yfinance |
| VIX index | WRDS / CBOE |
| Risk-free rate | WRDS / Fama-French factors |

---

## Data Pipeline

| Notebook | Purpose | Output |
|----------|---------|--------|
| `01-data-spy-vix-rf.ipynb` | Downloads SPY OHLCV, VIX, and risk-free rate; merges on date | `data/interim/01-data-spy-vix-rf.parquet` |
| `02-data-spy-option.ipynb` | Cleans raw SPY option chain through 8 filters | `data/interim/02-data-spy-option.parquet` |
| `03-data-merge-feature.ipynb` | Merges option and index panels; engineers target `d_iv` and all features | `data/interim/03-data-merge-feature.parquet` |
| `04-data-split.ipynb` | Legacy random + row-order chronological splits | `data/clean/*.parquet` (not active) |
| `05-data-split-chro.ipynb` | Active date-based chronological and random splits | `data/clean/v2/*_v2.parquet` |

### Option cleaning funnel

| Stage | Rows |
|-------|-----:|
| Raw calls | 11,012,971 |
| After NaN removal | 10,942,237 |
| After basic filters | 9,967,047 |
| After strike smoothness | 8,259,106 |
| After term-structure monotonicity | 8,256,071 |
| After calendar arbitrage filter | 8,057,911 |
| After butterfly convexity filter | 7,289,466 |
| After sanity bounds | 7,243,683 |
| After duplicate removal | 7,243,274 |
| After day-gap filter | 7,069,612 |

---

## Data Windows and Split Sizes

Four date-range windows are used across all experiments:

| Window | Dates | Label |
|--------|-------|-------|
| A | 2013-01-03 → 2026-01-30 | Full sample |
| B | 2013-01-03 → 2020-02-19 | Pre-COVID |
| C | 2020-03-23 → 2026-01-30 | Post-COVID |
| D | 2023-01-01 → 2026-01-30 | Most recent |

Active `v2` split sizes (date-based chronological partitions):

| Dataset | Train | Val | Test |
|---------|------:|----:|-----:|
| `chro_A_v2` | 2,266,676 | 942,247 | 579,718 |
| `chro_B_v2` | 744,477 | 331,626 | 225,573 |
| `chro_C_v2` | 1,670,317 | 516,375 | 265,853 |
| `chro_D_v2` | 790,689 | 265,453 | 148,363 |
| `rand_A_v2` | 2,652,048 | 757,728 | 378,865 |
| `rand_B_v2` | 911,172 | 260,336 | 130,168 |
| `rand_C_v2` | 1,716,781 | 490,509 | 245,255 |
| `rand_D_v2` | 843,153 | 240,901 | 120,451 |

---

## Feature Construction

The model predicts the daily change in implied volatility for the same option contract:

$$\Delta \sigma^{IV} = \sigma^{IV}_t - \sigma^{IV}_{t-1}$$

### Feature Sets

| Label | Features | Description |
|-------|---------|-------------|
| `3F` | `delta`, `T`, `spy_ret` | Paper baseline |
| `4F` | `3F` + `vix_lag` | Paper 4-feature model |
| `6F` | `4F` + `vix_mom_lag`, `gamma`, `iv_lag` | Best sequence-model base |
| `8F theta` | `6F` + `vix_mom`, `theta` | Best overall on chro_D |
| `8F rho` | `6F` + `vix_mom`, `rho` | Best on chro_A with GRU |

### Feature Definitions

| Feature | Formula | Description |
|---------|---------|-------------|
| `delta` | $\Delta = \partial C / \partial S$ | Option delta (moneyness proxy) |
| `T` | $T = \text{days to expiry} / 365$ | Time to maturity in years |
| `spy_ret` | $r_t = (S_t - S_{t-1}) / S_{t-1}$ | Daily SPY return |
| `vix_lag` | $\text{VIX}_{t-1}$ | Prior-day VIX level |
| `vix_mom` | $\text{VIX}_t - \text{VIX}_{t-1}$ | VIX daily momentum |
| `vix_mom_lag` | $\text{VIX}_{t-1} - \text{VIX}_{t-2}$ | Lagged VIX momentum |
| `iv_lag` | $\sigma^{IV}_{t-1}$ | Prior-day implied volatility |
| `d_iv_lag` | $\Delta\sigma^{IV}_{t-1}$ | Lagged IV change (autoregressive) |
| `gamma` | $\Gamma = \partial^2 C / \partial S^2$ | Option gamma (convexity) |
| `theta` | $\Theta = \partial C / \partial t$ | Option theta (time decay) |
| `rho` | $\rho = \partial C / \partial r$ | Option rho (rate sensitivity) |
| `log_oi` | $\log(\text{OI} + 1)$ | Log open interest |
| `log_volume` | $\log(\text{volume} + 1)$ | Log trading volume |

### Gain Metric

$$\text{Gain vs Analytic} = \left(1 - \frac{SSE_{\text{model}}}{SSE_{\text{analytic}}}\right) \times 100$$

---

## Data Filters

- Sentinel values ($\pm$999999, $\pm$9999999) replaced with NaN
- Term-structure monotonicity enforced (IV must be non-decreasing in maturity)
- Calendar arbitrage removed
- Butterfly convexity filter applied
- Sanity bounds on IV, delta, and price
- Day-gap filter: gaps > 3 calendar days between consecutive observations dropped
- Duplicate rows dropped

---

## Model Architectures

### Fully Connected (FC)

$$\Delta\sigma^{IV} = F(\mathbf{x};\,\theta)$$

- 3 hidden layers × 80 neurons
- ReLU activation, linear output
- No BatchNorm (destabilises training on $d_{iv}$ scale)
- Kaiming uniform initialization
- Adam optimizer, MSE loss
- Adaptive batch size (≥ 50 gradient steps/epoch, power-of-2, floor 512)
- 5-epoch linear LR warmup; ReduceLROnPlateau after warmup
- Early stopping (patience = 25)
- Series 1 (Keras/TensorFlow, local), Series 2 (PyTorch, Colab GPU)

### Sequence Models (LSTM / GRU / TFT)

All sequence models use a lookback window of 20 timesteps. Sequences are built from the full date-ordered contract history and assigned to train/val/test by the split label of the target row.

| Architecture | Core spec |
|---|---|
| LSTM | Hidden size 64, 2 stacked layers, dropout 0.1 |
| GRU | Hidden size 64, 2 stacked layers, dropout 0.1 |
| TFT | Hidden dim 64, 4 attention heads, 1 attention layer, VSN + GRN gating, dropout 0.1 |

Shared training setup:
- Adam optimizer, MSE loss
- Adaptive batch size (≥ 50 gradient steps/epoch)
- Init LR 1e-3 scaled by batch size; 5-epoch linear warmup
- ReduceLROnPlateau (patience 8, factor 0.3)
- Early stopping (patience 25)
- Max 100 epochs
- Run on Google Colab with GPU

### Analytic Benchmark (Hull-White)

$$E\left[\frac{\Delta \sigma_{\text{imp}}}{\Delta S / S}\right] = \frac{a + b\delta + c\delta^2}{T}$$

OLS fitted via `src/benchmark.py`; always evaluated on the full unfiltered test set.

---

## Experiment Families

| Series | Family | Data | Key finding |
|--------|--------|------|-------------|
| 1.x | Keras FC | `rand_A–D` | 18–34% gain vs analytic |
| 2.x | PyTorch FC feature sweep (128–256 combos) | `rand_A–D_v2` | 26–48% gain; `iv_lag` decisive for regime D |
| 3.x | LSTM chronological | `chro_A–D_v2` | 58–92% gain; chro_B strongest |
| 4.x | GRU chronological | `chro_A–D_v2` | 71–94% gain; best overall family |
| 5.x | TFT chronological | `chro_A–D_v2` | 35–89% gain; weaker on full sample |
| 6.x | FC chronological (ablation) | `chro_A–D_v2` | Mostly negative — sequence structure required |

---

## Project Structure

```
data/
  raw/              Raw parquet files (SPY, VIX, options, risk-free)
  interim/          Cleaned and merged intermediate datasets
  clean/v2/         Active train/val/test splits (8 windows × 3 sets)

notebook/
  data/             Data pipeline notebooks (01–05)
  model/            Experiment notebooks (1.x FC, 2.x FC sweep, 3–6 sequence)
  fig/              Figure-generation notebooks

src/
  paths.py                 Project directory constants
  benchmark.py             Hull-White analytic benchmark
  metrics.py               SSE, RMSE, gain, residual diagnostics, gain tables
  helper.py                Run directory creation, Keras callback, artifact saving
  fully_connected.py       Keras FC model (series 1)
  fully_connected_colab.py PyTorch FC utilities (series 2); bug fixes documented inline
  model3_utils.py          Shared sequence-model data prep, training, saving
  lstm.py                  Stacked LSTM model
  gru.py                   Stacked GRU model
  tft.py                   Simplified PyTorch TFT (VSN, GRN, multi-head attention)
  fig.py                   3D Hull-White vs ANN surface plot helper
  onclickmedia-data.py     Option data downloader

output/
  <notebook-name>/01-run/  Saved metrics, residual diagnostics, gain tables,
                           model weights, predictions, training history
  fig/                     Exported figures

log/
  experiments.md    Full experiment log with notebook inventory and saved results

report/             Final project report PDF and supporting figures
```

---

## SPY vs SPX Options

The original paper uses SPX index options. This project substitutes SPY ETF options:

- **SPX**: European-style, cash-settled, written on the S&P 500 index
- **SPY**: American-style, physically settled, trades at ~1/10 of SPX level
- **Dividends**: SPY pays discrete quarterly dividends; excluded via ex-dividend filter
- The paper's framework (delta, T, VIX regime variable) transfers directly to SPY

---

## Environment

```bash
# Install dependencies
pip install -r requirements.txt
```

- Python 3.12, virtual environment at `.venv/`
- Series 1 notebooks run locally with Keras/TensorFlow
- Series 2–5 notebooks run on Google Colab with GPU (PyTorch)
- WRDS credentials stored in `.env` (git-ignored): `WRDS_USERNAME`, `WRDS_PASSWORD`
- All data files are git-ignored; only source code and notebooks are tracked
