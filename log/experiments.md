# Experiments Log

## Notebook Inventory

### Data notebooks (`notebook/data`)

| Notebook | Role | Main Output |
|---|---|---|
| `01-data-spy-vix-rf.ipynb` | Build index panel from WRDS + yfinance | `data/interim/01-data-spy-vix-rf.parquet` |
| `02-data-spy-option.ipynb` | Clean SPY call option chain with arbitrage and quality filters | `data/interim/02-data-spy-option.parquet` |
| `03-data-merge-feature.ipynb` | Merge option + index data and engineer features/target | `data/interim/03-data-merge-feature.parquet` |
| `04-data-split.ipynb` | Create random and chronological train/val/test splits for A/B/C/D windows | `data/clean/*.parquet` |

### Model notebooks (`notebook/model`)

| Notebook | Data split used | Saved run folder |
|---|---|---|
| `1.0-fc-rand-A.ipynb` | `rand_A` | `output/1.0-fc-rand-A/01-run` |
| `1.1-fc-rand-B.ipynb` | `rand_B` | `output/1.1-fc-rand-B/01-run` |
| `1.2-fc-rand-C.ipynb` | `rand_C` | `output/1.2-fc-rand-C/01-run` |
| `1.3-fc-rand-D.ipynb` | `rand_D` | `output/1.3-fc-rand-D/01-run` |
| `3.1-lstm-rand-C-colab.ipynb` | `chro_C` | `output/3.1-lstm-rand-C-colab` |
| `3.2-gru-rand-A-colab.ipynb` | `chro_A` | `output/3.2-gru-rand-A-colab` |
| `3.3-gru-rand-C-colab.ipynb` | `chro_C` | `output/3.3-gru-rand-C-colab` |
| `3.4-tft-rand-A-colab.ipynb` | `chro_A` | `output/3.4-tft-rand-A-colab` |
| `3.5-tft-rand-C-colab-1.ipynb` | `chro_C` | `output/3.5-tft-rand-C-colab` |


## Data Pipeline

### Pipeline order

1. `notebook/data/01-data-spy-vix-rf.ipynb`
2. `notebook/data/02-data-spy-option.ipynb`
3. `notebook/data/03-data-merge-feature.ipynb`
4. `notebook/data/04-data-split.ipynb`
5. model notebooks `notebook/model/1.0-fc-rand-A.ipynb` ... `notebook/model/1.7-fc-chro-D.ipynb`

`run_pipeline.py` currently executes only the model notebooks (`1.0-fc-rand-A` through `1.7-fc-chro-D`).  
It does not execute data notebooks `01`-`04`.

### Step 01: index data build (`01-data-spy-vix-rf.ipynb`)

- Pulls `rf` from WRDS Fama-French daily factors and `vix` levels from WRDS CBOE.
- Pulls SPY OHLCV from yfinance.
- Merges on `date`, computes `spy_ret_adj` and `spy_ret`.
- Casts float64 to float32 and saves:
  - `data/raw/rf.parquet`
  - `data/raw/vix.parquet`
  - `data/raw/spy.parquet`
  - `data/interim/01-data-spy-vix-rf.parquet`

### Step 02: option chain cleaning (`02-data-spy-option.ipynb`)

- Starts from `data/raw/spy-option.parquet`, keeps calls, then applies:
  - NaN sentinel removal
  - basic quote validity filters (`bid>0`, `ask>bid`, positive strike/IV, `expiration>quote_date`)
  - strike smoothness screen (2nd-difference and minimum strike count)
  - per-slice quadratic residual filter
  - term-structure monotonicity filter
  - calendar spread no-arbitrage filter
  - butterfly convexity filter
  - hard sanity bounds on Greeks/spread
  - duplicate resolution by lowest `rel_spread`
  - day-gap filter (`day_gap <= 5` or first observation)

Cleaning audit reported in notebook output:

| Stage | Rows | Dropped vs previous | % kept vs raw calls |
|---|---:|---:|---:|
| raw calls | 11,012,971 | - | 100.0% |
| after NaN removal | 10,942,237 | 70,734 | 99.4% |
| after basic filters | 9,967,047 | 975,190 | 90.5% |
| after strike smoothness | 8,259,106 | 1,707,941 | 75.0% |
| after TS smoothness | 8,256,071 | 3,035 | 75.0% |
| after calendar arb | 8,057,911 | 198,160 | 73.2% |
| after butterfly arb | 7,289,466 | 768,445 | 66.2% |
| after sanity bounds | 7,243,683 | 45,783 | 65.8% |
| after duplicates | 7,243,274 | 409 | 65.8% |
| after day-gap filter | 7,069,612 | 173,662 | 64.2% |

Final export:

- `data/interim/02-data-spy-option.parquet`

### Step 03: merge + feature engineering (`03-data-merge-feature.ipynb`)

- Loads:
  - `data/interim/01-data-spy-vix-rf.parquet`
  - `data/interim/02-data-spy-option.parquet`
- Adds index features:
  - `vix_lag`, `vix_mom`, `vix_mom_lag`
- Merges on `date`.
- Adds option/index features:
  - `log_moneyness`, `iv_lag`, `d_iv` (target), `d_iv_lag`
  - `log_oi`, `log_volume`, `abs_spy_ret`, `ret_over_sqrtT`
- Drops rows with missing `d_iv` or `vix_lag`:
  - `6,748,340` rows remain (`321,272` dropped; `4.54%`)
- Applies paper-style filters:
  - `T >= 14/365`, `0.05 <= delta <= 0.95`, `iv <= 1.0`
  - `3,788,641` rows remain
  - reported `d_iv` std compression: `4.1x`
- Casts float64 to float32 and saves:
  - `data/interim/03-data-merge-feature.parquet`

### Step 04: split generation (`04-data-split.ipynb`)

- Reads `data/interim/03-data-merge-feature.parquet`.
- Defines windows:
  - `A`: `2013-01-03` to `2026-01-30`
  - `B`: `2013-01-03` to `2020-02-19`
  - `C`: `2020-03-23` to `2026-01-30`
  - `D`: `2023-01-01` to `2026-01-30`
- Creates both:
  - random split (`train_test_split`, 70/20/10 with `random_state=42`)
  - chronological split (first 70%, next 20%, last 10%) - **not using this date set**
- Saves `24` files into `data/clean/` (`rand_*` and `chro_*`, each with train/val/test).

Saved row counts reported by notebook:

| Dataset | Train | Val | Test |
|---|---:|---:|---:|
| `rand_A` | 2,652,048 | 757,728 | 378,865 |
| `rand_B` | 911,172 | 260,336 | 130,168 |
| `rand_C` | 1,716,781 | 490,509 | 245,255 |
| `rand_D` | 843,153 | 240,901 | 120,451 |

## 1.x FC Model Setup

Common setup in `1.0-fc-rand-A` through `1.3-fc-rand-D`:

- Framework: TensorFlow/Keras
- Architecture: `3` hidden layers x `80` units, `relu`, linear output
- Optimizer: Adam (`lr=1e-3`)
- Batch size: `4096`
- Max epochs: `100`
- Early stopping patience: `30`
- LR scheduler: patience `8`, factor `0.3`
- Features:
  - `3F`: `delta`, `T`, `spy_ret`
  - `4F`: `delta`, `T`, `spy_ret`, `vix_lag`
- Target: `d_iv`
- Baseline: Hull-White analytic benchmark from `src/benchmark.py`

### Results

| Model | Data used in notebook | Analytic SSE | 3F SSE | 3F Gain vs Analytic | 4F SSE | 4F Gain vs Analytic | 4F Gain vs 3F |
|---|---|---:|---:|---:|---:|---:|---:|
| `1.0-fc-rand-A` | `rand_A` | 115.3004 | 99.5864 | 13.63% | 83.5310 | 27.55% | 16.12% |
| `1.1-fc-rand-B` | `rand_B` | 38.6267 | 35.1879 | 8.90% | 29.7183 | 23.06% | 15.54% |
| `1.2-fc-rand-C` | `rand_C` | 64.9134 | 57.3609 | 11.63% | 43.0808 | 33.63% | 24.90% |
| `1.3-fc-rand-D` | `rand_D` | 8.3745 | 7.8642 | 6.09% | 6.8504 | 18.20% | 12.89% |

## 2.x FC Model Setup

Common setup in `2.0-fc-rand-A-colab` through `2.3-fc-rand-D-colab`:

- Framework: TensorFlow/Keras (Colab)
- Model family: fully connected feature-combination search over `3F` baseline
- Baseline features (`3F`): `delta`, `T`, `spy_ret`
- Target: `d_iv`
- Baseline for comparison: Hull-White analytic benchmark
- Selection rule in tables: top `10` by `Gain_vs_Analytic` per range; non-positive gains excluded

### Results

#### Range A (`2.0-fc-rand-A-colab`)

| Model | SSE | MSE | RMSE | MAE | MeanError | MedianAE | R2 | Training_time | Gain_vs_Analytic | Gain_Incremental |
|---|---|---|---|---|---|---|---|---|---|---|
| `3F+iv_lag+gamma+rho` | 34.133461 | 0.000090 | 0.009492 | 0.005581 | 0.000324 | 0.003418 | 0.728161 | 19.2s | 70.40% | 18.21% |
| `3F+iv_lag+gamma+theta` | 39.077766 | 0.000103 | 0.010156 | 0.005882 | -0.000260 | 0.003473 | 0.688785 | 19.0s | 66.11% | 27.57% |
| `3F+iv_lag+gamma+vega` | 41.732525 | 0.000110 | 0.010495 | 0.005958 | 0.000004 | 0.003550 | 0.667642 | 19.1s | 63.81% | -6.79% |
| `3F+vix_lag+iv_lag+gamma` | 42.518570 | 0.000112 | 0.010594 | 0.005854 | -0.000358 | 0.003303 | 0.661382 | 18.7s | 63.12% | 9.17% |
| `3F+vix_lag+iv_lag+d_iv_lag` | 42.636250 | 0.000113 | 0.010608 | 0.005699 | -0.001018 | 0.003117 | 0.660445 | 19.1s | 63.02% | 60.15% |
| `3F+vix_lag+iv_lag+theta` | 44.128506 | 0.000116 | 0.010792 | 0.005896 | 0.000211 | 0.003250 | 0.648561 | 19.0s | 61.73% | -3.79% |
| `3F+iv_lag+d_iv_lag+theta` | 44.421329 | 0.000117 | 0.010828 | 0.005923 | -0.000795 | 0.003335 | 0.646229 | 18.6s | 61.47% | 2.39% |
| `3F+iv_lag+d_iv_lag+vix_mom` | 44.988575 | 0.000119 | 0.010897 | 0.005748 | 0.001358 | 0.003227 | 0.641711 | 18.4s | 60.98% | 2.49% |
| `3F+vix_lag+iv_lag+vix_mom_lag` | 45.498112 | 0.000120 | 0.010959 | 0.005840 | -0.000542 | 0.003279 | 0.637653 | 18.7s | 60.54% | -6.71% |
| `3F+iv_lag+d_iv_lag+gamma` | 45.509666 | 0.000120 | 0.010960 | 0.006034 | -0.000709 | 0.003467 | 0.637561 | 18.6s | 60.53% | -1.16% |

#### Range B (`2.1-fc-rand-B-colab`)

| Model | SSE | MSE | RMSE | MAE | MeanError | MedianAE | R2 | Training_time | Gain_vs_Analytic | Gain_Incremental |
|---|---|---|---|---|---|---|---|---|---|---|
| `3F+iv_lag+gamma+rho` | 14.987998 | 0.000115 | 0.010730 | 0.007111 | 0.000556 | 0.004974 | 0.638160 | 13.6s | 61.20% | 16.37% |
| `3F+vix_lag+iv_lag+d_iv_lag` | 17.886305 | 0.000137 | 0.011722 | 0.007243 | 0.000177 | 0.004672 | 0.568190 | 13.6s | 53.69% | 53.94% |
| `3F+iv_lag+gamma+vega` | 17.921600 | 0.000138 | 0.011734 | 0.007443 | 0.000007 | 0.004981 | 0.567337 | 13.5s | 53.60% | 7.34% |
| `3F+vix_lag+iv_lag+gamma` | 18.972361 | 0.000146 | 0.012073 | 0.007451 | 0.002162 | 0.005066 | 0.541970 | 13.7s | 50.88% | 9.46% |
| `3F+vix_lag+iv_lag+vega` | 19.036503 | 0.000146 | 0.012093 | 0.007404 | -0.000387 | 0.004816 | 0.540422 | 13.8s | 50.72% | 1.48% |
| `3F+iv_lag+d_iv_lag+rho` | 19.300495 | 0.000148 | 0.012177 | 0.007643 | 0.001225 | 0.005035 | 0.534048 | 13.7s | 50.03% | 5.46% |
| `3F+vix_lag+iv_lag+theta` | 19.322113 | 0.000148 | 0.012184 | 0.007526 | -0.001969 | 0.004951 | 0.533526 | 13.6s | 49.98% | -1.84% |
| `3F+iv_lag+gamma+theta` | 19.342125 | 0.000149 | 0.012190 | 0.007857 | -0.000548 | 0.005402 | 0.533043 | 13.4s | 49.93% | 14.93% |
| `3F+vix_lag+iv_lag+rho` | 19.664497 | 0.000151 | 0.012291 | 0.007627 | 0.000529 | 0.005066 | 0.525261 | 13.5s | 49.09% | -3.30% |
| `3F+iv_lag+d_iv_lag+vix_mom_lag` | 19.708347 | 0.000151 | 0.012305 | 0.007624 | 0.000571 | 0.004995 | 0.524202 | 13.3s | 48.98% | 43.29% |

#### Range C (`2.2-fc-rand-C-colab`)

| Model | SSE | MSE | RMSE | MAE | MeanError | MedianAE | R2 | Training_time | Gain_vs_Analytic | Gain_Incremental |
|---|---|---|---|---|---|---|---|---|---|---|
| `3F+iv_lag+gamma+vega` | 23.147978 | 0.000094 | 0.009715 | 0.006388 | -0.001717 | 0.004539 | 0.667746 | 13.0s | 64.34% | 9.17% |
| `3F+iv_lag+gamma+theta` | 25.486212 | 0.000104 | 0.010194 | 0.006548 | 0.000818 | 0.004414 | 0.634184 | 12.8s | 60.74% | 28.38% |
| `3F+vix_lag+iv_lag+d_iv_lag` | 27.202682 | 0.000111 | 0.010532 | 0.006352 | -0.000773 | 0.004016 | 0.609547 | 12.5s | 58.09% | 55.81% |
| `3F+vix_lag+iv_lag+gamma` | 28.072662 | 0.000114 | 0.010699 | 0.006676 | 0.001704 | 0.004366 | 0.597059 | 12.7s | 56.75% | 6.45% |
| `3F+vix_lag+iv_lag+theta` | 29.027412 | 0.000118 | 0.010879 | 0.006665 | 0.000659 | 0.004320 | 0.583355 | 12.8s | 55.28% | -3.40% |
| `3F+vix_lag+iv_lag+vix_mom_lag` | 29.315367 | 0.000120 | 0.010933 | 0.006663 | -0.000764 | 0.004378 | 0.579222 | 12.7s | 54.84% | -7.77% |
| `3F+vix_lag+iv_lag+vix_mom` | 30.009705 | 0.000122 | 0.011062 | 0.006590 | -0.000393 | 0.004302 | 0.569256 | 12.9s | 53.77% | -2.37% |
| `3F+iv_lag+d_iv_lag+vix_mom` | 30.249287 | 0.000123 | 0.011106 | 0.006657 | -0.000052 | 0.004344 | 0.565817 | 12.3s | 53.40% | 7.75% |
| `3F+vix_lag+iv_lag+rho` | 30.277189 | 0.000123 | 0.011111 | 0.006668 | -0.000199 | 0.004303 | 0.565417 | 12.6s | 53.36% | 4.37% |
| `3F+d_iv_lag+vix_mom_lag+vix_mom` | 30.370039 | 0.000124 | 0.011128 | 0.006315 | -0.000271 | 0.004037 | 0.564084 | 12.7s | 53.21% | 26.03% |

#### Range D (`2.3-fc-rand-D-colab`)

| Model | SSE | MSE | RMSE | MAE | MeanError | MedianAE | R2 | Training_time | Gain_vs_Analytic | Gain_Incremental |
|---|---|---|---|---|---|---|---|---|---|---|
| `3F+vix_lag+iv_lag+gamma` | 7.817259 | 0.000065 | 0.008056 | 0.005523 | 0.000641 | 0.004022 | 0.183237 | 12.1s | 6.65% | 14.57% |
| `3F+iv_lag+gamma+vega` | 8.127499 | 0.000067 | 0.008214 | 0.005751 | 0.001089 | 0.004337 | 0.150822 | 12.2s | 2.95% | 4.69% |
| `3F+vix_lag+iv_lag+vega` | 8.249505 | 0.000068 | 0.008276 | 0.005660 | 0.001211 | 0.004139 | 0.138075 | 12.7s | 1.49% | 11.05% |



## 3.x Sequence Model Setup

Common setup in `3.1-lstm-rand-C-colab` through `3.5-tft-rand-*-colab`:

- Model families: LSTM, GRU, TFT
- Data used: chronological split sequence models on `chro_A` and `chro_C`
- Lookback window: `20`
- Max epochs: `100`
- Early stopping patience: `25`
- Warmup: `5` epochs
- Target: `d_iv`
- Feature sets:
  - `4F`: `delta`, `T`, `spy_ret`, `vix_lag`
  - `5F`: `delta`, `T`, `spy_ret`, `vix_lag`, `iv_lag`
  - `6F`: `delta`, `T`, `spy_ret`, `vix_lag`, `iv_lag`, `d_iv_lag`
  - `8F`: `delta`, `T`, `spy_ret`, `vix_lag`, `iv_lag`, `d_iv_lag`, `gamma`, `rho`
- Baseline: Hull-White analytic benchmark
- Table rule below: keep only the first analytic row as `Analytic`; do not repeat later analytic rows

### Results

#### `3.1-lstm-rand-C-colab`

| Model | SSE | MSE | RMSE | MAE | MeanError | MedianAE | R2 | Training_time | Gain_vs_Analytic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Analytic` | 17.891375 | 0.000107 | 0.010360 | 0.004405 | 0.000770 | 0.002271 | 0.100868 |  |  |
| `4F` | 4.990499 | 0.000030 | 0.005471 | 0.002198 | -0.000143 | 0.001327 | 0.749202 | 871.1s | 72.11% |
| `5F` | 1.177748 | 0.000007 | 0.002658 | 0.001572 | -0.000035 | 0.001022 | 0.940812 | 836.9s | 93.42% |
| `6F` | 1.186808 | 0.000007 | 0.002684 | 0.001558 | 0.000073 | 0.001008 | 0.938483 | 836.4s | 93.17% |
| `8F` | 0.336354 | 0.000002 | 0.001429 | 0.000785 | -0.000060 | 0.000500 | 0.982566 | 854.3s | 98.06% |

#### `3.2-gru-rand-A-colab`

| Model | SSE | MSE | RMSE | MAE | MeanError | MedianAE | R2 | Training_time | Gain_vs_Analytic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Analytic` | 34.862591 | 0.000137 | 0.011703 | 0.004686 | 0.000977 | 0.002174 | 0.097335 |  |  |
| `4F` | 10.398916 | 0.000041 | 0.006391 | 0.002754 | 0.000100 | 0.001523 | 0.730750 | 1108.9s | 70.17% |
| `5F` | 2.725460 | 0.000011 | 0.003272 | 0.001971 | 0.000060 | 0.001251 | 0.929432 | 1021.4s | 92.18% |
| `6F` | 2.439524 | 0.000010 | 0.003116 | 0.001826 | 0.000108 | 0.001143 | 0.935379 | 930.2s | 92.83% |
| `8F` | 0.315054 | 0.000001 | 0.001120 | 0.000558 | 0.000030 | 0.000297 | 0.991655 | 1025.0s | 99.07% |

#### `3.3-gru-rand-C-colab`

| Model | SSE | MSE | RMSE | MAE | MeanError | MedianAE | R2 | Training_time | Gain_vs_Analytic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Analytic` | 17.891375 | 0.000107 | 0.010360 | 0.004405 | 0.000770 | 0.002271 | 0.100868 |  |  |
| `4F` | 5.111166 | 0.000031 | 0.005537 | 0.002228 | 0.000099 | 0.001359 | 0.743138 | 855.9s | 71.43% |
| `5F` | 1.341453 | 0.000008 | 0.002837 | 0.001710 | -0.000066 | 0.001145 | 0.932585 | 793.0s | 92.50% |
| `6F` | 1.270033 | 0.000008 | 0.002777 | 0.001675 | 0.000213 | 0.001123 | 0.934170 | 830.0s | 92.69% |
| `8F` | 0.216374 | 0.000001 | 0.001146 | 0.000606 | 0.000063 | 0.000362 | 0.988785 | 815.7s | 98.75% |

#### `3.4-tft-rand-A-colab`

| Model | SSE | MSE | RMSE | MAE | MeanError | MedianAE | R2 | Training_time | Gain_vs_Analytic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Analytic` | 34.862591 | 0.000137 | 0.011703 | 0.004686 | 0.000977 | 0.002174 | 0.097335 |  |  |
| `4F` | 11.848680 | 0.000047 | 0.006822 | 0.003109 | -0.000243 | 0.001794 | 0.693213 | 1246.5s | 66.01% |
| `5F` | 1.549968 | 0.000006 | 0.002468 | 0.001472 | 0.000069 | 0.000941 | 0.959868 | 1827.3s | 95.55% |
| `6F` | 2.821856 | 0.000011 | 0.003351 | 0.002069 | -0.000243 | 0.001340 | 0.925252 | 1750.6s | 91.71% |
| `8F` | 0.325628 | 0.000001 | 0.001138 | 0.000706 | -0.000031 | 0.000438 | 0.991374 | 1909.8s | 99.04% |

#### `3.5-tft-rand-C-colab`

| Model | SSE | MSE | RMSE | MAE | MeanError | MedianAE | R2 | Training_time | Gain_vs_Analytic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Analytic` | 17.891375 | 0.000107 | 0.010360 | 0.004405 | 0.000770 | 0.002271 | 0.100868 |  |  |
| `4F` | 4.300082 | 0.000026 | 0.005079 | 0.002180 | -0.000325 | 0.001352 | 0.783899 | 1413.9s | 75.97% |
| `5F` | 0.521975 | 0.000003 | 0.001770 | 0.001045 | 0.000046 | 0.000679 | 0.973768 | 1592.1s | 97.08% |
| `6F` | 0.498316 | 0.000003 | 0.001739 | 0.000999 | 0.000056 | 0.000631 | 0.974170 | 1744.7s | 97.13% |
| `8F` | 0.170259 | 0.000001 | 0.001017 | 0.000646 | -0.000023 | 0.000392 | 0.991175 | 2081.9s | 99.02% |

# Feature Definitions

| Feature | Formula | Description |
|---------|---------|-------------|
| `delta` | $\Delta = \frac{\partial C}{\partial S}$ | Option delta (moneyness proxy) |
| `T` | $T = \frac{\text{days to expiry}}{365}$ | Time to maturity in years |
| `spy_ret` | $r_t = \frac{S_t - S_{t-1}}{S_{t-1}}$ | Daily SPY return |
| `vix_lag` | $\text{VIX}_{t-1}$ | Previous day VIX level |
| `iv_lag` | $\sigma^{\text{IV}}_{t-1}$ | Previous day implied volatility |
| `vix_mom` | $\text{VIX}_t - \text{VIX}_{t-1}$ | VIX daily momentum |
| `vix_mom_lag` | $\text{VIX}_{t-1} - \text{VIX}_{t-2}$ | Lagged VIX momentum |
| `gamma` | $\Gamma = \frac{\partial^2 C}{\partial S^2}$ | Option gamma (convexity) |
| `d_iv_lag` | $\Delta\sigma^{\text{IV}}_{t-1}$ | Lagged change in IV (autoregressive term) |
| `spread` | $\text{ask} - \text{bid}$ | Bid-ask spread |
| `abs_spy_ret` | $\lvert r_t \rvert$ | Absolute daily return (realized vol proxy) |
| `ret_over_sqrtT` | $\frac{r_t}{\sqrt{T}}$ | Return scaled by sqrt(T) (analytic interaction) |
| `log_oi` | $\log(\text{open interest} + 1)$ | Log open interest (liquidity) |
| `log_volume` | $\log(\text{volume} + 1)$ | Log volume (liquidity) |
| `theta` | $\Theta = \frac{\partial C}{\partial t}$ | Option theta (time decay) |
| `vega` | $\mathcal{V} = \frac{\partial C}{\partial \sigma}$ | Option vega (IV sensitivity) |
| `rho` | $\rho = \frac{\partial C}{\partial r}$ | Option rho (interest rate sensitivity) |
| `log_moneyness` | $\log\!\left(\frac{S}{K}\right)$ | Log spot-to-strike ratio |
| `d_iv` | $\sigma^{\text{IV}}_t - \sigma^{\text{IV}}_{t-1}$ | **Target**: daily change in IV |

## Gain Formulas

$$\text{Gain vs Analytic} = \left(1 - \frac{SSE_{model}}{SSE_{analytic}}\right) \times 100$$

$$\text{Gain vs 3F} = \left(1 - \frac{SSE_{model}}{SSE_{3F}}\right) \times 100$$

$$\text{Gain vs 4F} = \left(1 - \frac{SSE_{model}}{SSE_{4F}}\right) \times 100$$
