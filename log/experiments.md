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

## Results

| Model | Data used in notebook | Analytic SSE | 3F SSE | 3F Gain vs Analytic | 4F SSE | 4F Gain vs Analytic | 4F Gain vs 3F |
|---|---|---:|---:|---:|---:|---:|---:|
| `1.0-fc-rand-A` | `rand_A` | 115.3004 | 99.5864 | 13.63% | 83.5310 | 27.55% | 16.12% |
| `1.1-fc-rand-B` | `rand_B` | 38.6267 | 35.1879 | 8.90% | 29.7183 | 23.06% | 15.54% |
| `1.2-fc-rand-C` | `rand_C` | 64.9134 | 57.3609 | 11.63% | 43.0808 | 33.63% | 24.90% |
| `1.3-fc-rand-D` | `rand_D` | 8.3745 | 7.8642 | 6.09% | 6.8504 | 18.20% | 12.89% |




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
