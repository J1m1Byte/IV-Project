# Experiments Log

This log was rebuilt from the current repository contents after the project reset. It reflects the notebooks, source modules, and saved artifacts that are actually present in the repo today.

## Current Repo Snapshot

### What exists

- `src/` contains the active training and evaluation code.
- `notebook/data/` contains five data-prep notebooks.
- `notebook/model/` contains the active experiment notebooks plus an `archive/` subfolder.
- `notebook/fig/` contains four figure-building notebooks.
- `output/` contains saved metrics, residual diagnostics, gain tables, model histories, and figure exports.

### What is clearly stale or mismatched

- `README.md` is empty.
- `src/run_pipeline.py` points to non-existent notebooks such as `2.0-fc-rand-A-feature.ipynb` and is not the current execution entrypoint.
- `notebook/data/04-data-split.ipynb` writes the old split files in `data/clean/`, but the rebuilt model notebooks use the `*_v2.parquet` files produced by `notebook/data/05-data-split-chro.ipynb`.
- Some notebook names are narrower than the code they contain:
  - `3.0-lstm-chro-A.ipynb`
  - `4.0-gru-chro-A.ipynb`
  - `5.0-tft-chro-A.ipynb`
  These notebooks are coded as multi-dataset loops, but the committed `output/` tree only contains `chro_A` results for the `3.0`, `4.0`, and `5.0` families.

## Notebook Inventory

### Data notebooks

| Notebook | Status | Purpose | Main artifact |
|---|---|---|---|
| `01-data-spy-vix-rf.ipynb` | Active | Builds SPY, VIX, and risk-free panel | `data/interim/01-data-spy-vix-rf.parquet` |
| `02-data-spy-option.ipynb` | Active | Cleans SPY option chain | `data/interim/02-data-spy-option.parquet` |
| `03-data-merge-feature.ipynb` | Active | Merges option and index data, engineers target/features | `data/interim/03-data-merge-feature.parquet` |
| `04-data-split.ipynb` | Legacy | Writes old random + row-based chronological splits | `data/clean/*.parquet` |
| `05-data-split-chro.ipynb` | Active | Writes random + date-based chronological splits | `data/clean/v2/*_v2.parquet` |

### Model notebooks

| Notebook | Family | Data used | Saved output present in repo |
|---|---|---|---|
| `1.0-fc-rand-A.ipynb` | TensorFlow FC baseline | `rand_A` | `output/1.0-fc-rand-A/01-run` |
| `1.1-fc-rand-B.ipynb` | TensorFlow FC baseline | `rand_B` | `output/1.1-fc-rand-B/01-run` |
| `1.2-fc-rand-C.ipynb` | TensorFlow FC baseline | `rand_C` | `output/1.2-fc-rand-C/01-run` |
| `1.3-fc-rand-D.ipynb` | TensorFlow FC baseline | `rand_D` | `output/1.3-fc-rand-D/01-run` |
| `2.0-fc-rand-A-colab.ipynb` | PyTorch FC feature sweep | `rand_A_v2` | `output/2.0-fc-rand-A-colab/01-run` |
| `2.1-fc-rand-B-colab.ipynb` | PyTorch FC feature sweep | `rand_B_v2` | `output/2.1-fc-rand-B-colab/01-run` |
| `2.2-fc-rand-C-colab.ipynb` | PyTorch FC feature sweep | `rand_C_v2` | `output/2.2-fc-rand-C-colab/01-run` |
| `2.3-fc-rand-D-colab.ipynb` | PyTorch FC feature sweep | `rand_D_v2` | `output/2.3-fc-rand-D-colab/01-run` |
| `2.3.1-fc-rand-D-colab-ivlag.ipynb` | PyTorch FC feature sweep with `iv_lag` | `rand_D_v2` | `output/2.3.1-fc-rand-D-colab-ivlag/01-run` |
| `3.0-lstm-chro-A.ipynb` | LSTM | coded for all `chro_*`, committed output only for `chro_A` | `output/3.0-lstm-chro-A/01-run/chro_A` |
| `3.1-lstm-chro-B-C-D.ipynb` | LSTM | `chro_B_v2`, `chro_C_v2`, `chro_D_v2` | `output/3.1-lstm-chro-B-C-D/01-run/chro_B|C|D` |
| `4.0-gru-chro-A.ipynb` | GRU | coded for all `chro_*`, committed output only for `chro_A` | `output/4.0-gru-chro-A/01-run/chro_A` |
| `4.1-gru-chro-B-C-D.ipynb` | GRU | `chro_B_v2`, `chro_C_v2`, `chro_D_v2` | `output/4.1-gru-chro-B-C-D/01-run/chro_B|C|D` |
| `5.0-tft-chro-A.ipynb` | TFT | coded for all `chro_*`, committed output only for `chro_A` | `output/5.0-tft-chro-A/01-run/chro_A` |
| `5.0-tft-chro-all.ipynb` | TFT | coded for all `chro_*` | no committed run folder under `output/5.0-tft-chro-all/` |
| `5.1-tft-chro-B-C-D.ipynb` | TFT | `chro_B_v2`, `chro_C_v2`, `chro_D_v2` | `output/5.1-tft-chro-B-C-D/01-run/chro_B|C|D` |
| `6.0-fc-chro-all.ipynb` | PyTorch FC on chronological data | `chro_A_v2` to `chro_D_v2` | `output/6.0-fc-chro-all/01-run/chro_A|B|C|D` |

### Archived notebooks

- `notebook/model/archive/3.0.1-lstm-chro-A.ipynb`
- `notebook/model/archive/4.0.1-gru-chro-A.ipynb`
- `notebook/model/archive/5.0.1-tft-chro-A.ipynb`

These look like earlier sequence-model iterations and should not be treated as the main experiment path.

## Source Modules In Use

| File | Role |
|---|---|
| `src/paths.py` | Defines project directories, including `data/clean/v2` and `output/fig` |
| `src/helper.py` | Run directory creation, TensorFlow progress bar, result persistence |
| `src/benchmark.py` | Hull-White analytic benchmark |
| `src/metrics.py` | Error metrics, gain calculation, residual diagnostics, gain tables |
| `src/fully_connected.py` | TensorFlow 3-layer x 80-neuron FC model |
| `src/fully_connected_colab.py` | PyTorch FC utilities, NaN-safe evaluation, adaptive batch sizing, feature-sweep persistence |
| `src/model3_utils.py` | Shared sequence-model data prep, sequence construction, scaling, training, saving |
| `src/lstm.py` | Stacked LSTM model |
| `src/gru.py` | Stacked GRU model |
| `src/tft.py` | Simplified PyTorch TFT |
| `src/fig.py` | 3D analytic-vs-FC plotting helper |
| `src/onclickmedia-data.py` | OnclickMedia downloader for option data |
| `src/run_pipeline.py` | Stale pipeline script; does not match current notebook inventory |

## Active Data Pipeline

### Step 1: SPY, VIX, risk-free panel

`01-data-spy-vix-rf.ipynb` builds the index-side panel and saves `data/interim/01-data-spy-vix-rf.parquet`.

### Step 2: Option cleaning

`02-data-spy-option.ipynb` filters raw SPY option data into `data/interim/02-data-spy-option.parquet`.

The committed notebook output still shows the large clean-up funnel:

| Stage | Rows |
|---|---:|
| Raw calls | 11,012,971 |
| After NaN removal | 10,942,237 |
| After basic filters | 9,967,047 |
| After strike smoothness | 8,259,106 |
| After TS smoothness | 8,256,071 |
| After calendar arbitrage filter | 8,057,911 |
| After butterfly filter | 7,289,466 |
| After sanity bounds | 7,243,683 |
| After duplicate removal | 7,243,274 |
| After day-gap filter | 7,069,612 |

### Step 3: Merge and feature engineering

`03-data-merge-feature.ipynb` merges interim option and index data, engineers the target `d_iv`, and creates the feature table in `data/interim/03-data-merge-feature.parquet`.

The engineered feature set used throughout the rebuilt repo includes:

- base market features: `delta`, `T`, `spy_ret`
- lag and volatility features: `vix_lag`, `vix_mom`, `vix_mom_lag`, `iv_lag`, `d_iv_lag`
- Greeks and option descriptors: `gamma`, `theta`, `vega`, `rho`, `log_moneyness`, `log_oi`, `log_volume`
- target: `d_iv`

### Step 4: Legacy split notebook

`04-data-split.ipynb` writes `rand_*` and `chro_*` files into `data/clean/`, but its chronological outputs are row-order slices of the same window and are no longer the active path.

### Step 5: Active split notebook

`05-data-split-chro.ipynb` is the active split generator. It writes the `*_v2.parquet` files in `data/clean/v2/` and uses date-based chronological partitions.

The active modeling windows are:

| Window | Dates | Label |
|---|---|---|
| A | `2013-01-03` to `2026-01-30` | Full sample |
| B | `2013-01-03` to `2020-02-19` | Pre-COVID |
| C | `2020-03-23` to `2026-01-30` | Post-COVID |
| D | `2023-01-01` to `2026-01-30` | Most recent |

Saved `v2` split sizes:

| Dataset | Train | Val | Test |
|---|---:|---:|---:|
| `rand_A_v2` | 2,652,048 | 757,728 | 378,865 |
| `chro_A_v2` | 2,266,676 | 942,247 | 579,718 |
| `rand_B_v2` | 911,172 | 260,336 | 130,168 |
| `chro_B_v2` | 744,477 | 331,626 | 225,573 |
| `rand_C_v2` | 1,716,781 | 490,509 | 245,255 |
| `chro_C_v2` | 1,670,317 | 516,375 | 265,853 |
| `rand_D_v2` | 843,153 | 240,901 | 120,451 |
| `chro_D_v2` | 790,689 | 265,453 | 148,363 |

## Experiment Families

### 1.x TensorFlow FC baselines on random splits

Common setup:

- architecture: 3 hidden layers x 80 neurons, ReLU, linear output
- optimizer: Adam
- batch size: 4096
- max epochs: 100
- early stopping patience: 30
- target: `d_iv`
- models saved via `src.helper.save_run`

Best saved result in each random range:

| Notebook | Best model | Gain vs analytic | SSE | R2 |
|---|---|---:|---:|---:|
| `1.0-fc-rand-A` | `ANN-4F` | 27.55% | 83.5310 | 0.3348 |
| `1.1-fc-rand-B` | `ANN-4F` | 23.06% | 29.7183 | 0.2825 |
| `1.2-fc-rand-C` | `ANN-4F` | 33.63% | 43.0808 | 0.3816 |
| `1.3-fc-rand-D` | `ANN-4F` | 18.20% | 6.8504 | 0.2843 |

### 2.x PyTorch FC feature sweeps on random splits

Common setup:

- base features: `delta`, `T`, `spy_ret`
- extra search pool in `2.0` to `2.3`: `vix_lag`, `vix_mom_lag`, `vix_mom`, `gamma`, `theta`, `vega`, `rho`
- extra search pool in `2.3.1`: the same seven plus `iv_lag`
- architecture: 3 hidden layers x 80 neurons
- max epochs: 100
- patience: 30
- adaptive batch sizing from `src/fully_connected_colab.py`
- evaluation is NaN-safe and always keeps the full test set

Search size:

- `2.0` to `2.3`: `2^7 = 128` total feature combinations including the 3F base model
- `2.3.1`: `2^8 = 256` total feature combinations including the 3F base model

Best saved result in each run:

| Notebook | Best model | Gain vs analytic | SSE | Notes |
|---|---|---:|---:|---|
| `2.0-fc-rand-A-colab` | `3F+vix_lag+vix_mom_lag+vix_mom+gamma+rho` | 45.57% | 62.7529 | Strong improvement over 1.x |
| `2.1-fc-rand-B-colab` | `3F+vix_lag+vix_mom_lag+vix_mom+theta+vega` | 25.72% | 28.6926 | Modest but positive |
| `2.2-fc-rand-C-colab` | `3F+vix_lag+vix_mom_lag+vix_mom+gamma+theta` | 47.61% | 34.0104 | Best random-split non-`iv_lag` result |
| `2.3-fc-rand-D-colab` | `3F+vix_lag+vix_mom+vega+rho` | -0.83% | 8.4444 | No positive gain without `iv_lag` |
| `2.3.1-fc-rand-D-colab-ivlag` | `3F+iv_lag+vix_lag+vix_mom_lag+vix_mom+gamma+theta+vega` | 41.55% | 4.8949 | Adding `iv_lag` changes D materially |

Main takeaway from the saved 2.x runs:

- ranges A and C respond strongly to richer feature combinations
- range B improves, but much less
- range D is the regime where the non-`iv_lag` sweep mostly fails
- `iv_lag` is the decisive extra feature for range D in the rebuilt random-split FC search

### 3.x, 4.x, 5.x sequence models on chronological `v2` data

Shared sequence setup from notebooks plus `src/model3_utils.py`:

- lookback window: 20
- target: `d_iv`
- date-ordered sequences built by contract key `(k, expiration)`
- batch size: adaptive
- base learning rate: 1e-3, scaled by batch size
- warmup: 5 epochs
- max epochs: 100
- patience: 12
- LR patience: 5

Architecture constants:

| Family | Core architecture |
|---|---|
| LSTM | hidden size 64, 2 layers, dropout 0.1 |
| GRU | hidden size 64, 2 layers, dropout 0.1 |
| TFT | hidden dim 64, 4 heads, 1 attention layer, dropout 0.1 |

#### Best saved sequence result by architecture and dataset

| Dataset | LSTM best | Gain | GRU best | Gain | TFT best | Gain |
|---|---|---:|---|---:|---|---:|
| `chro_A` | `6F` | 62.11% | `8F rho` | 73.36% | `8F theta` | 35.27% |
| `chro_B` | `6F` | 91.73% | `6F` | 94.06% | `6F` | 89.05% |
| `chro_C` | `8F rho` | 79.72% | `6F` | 90.64% | `8F rho` | 73.80% |
| `chro_D` | `8F theta` | 57.96% | `8F theta` | 70.76% | `8F theta` | 73.32% |

Observations from the committed sequence outputs:

- GRU is the strongest overall family in the saved runs.
- `chro_B` and `chro_C` are where sequence models dominate most clearly.
- TFT improves a lot on `chro_B`, `chro_C`, and `chro_D`, but is much weaker on `chro_A`.
- The reduced three-feature-set runs (`6F`, `8F theta`, `8F rho`) are the only committed outputs for `3.1`, `4.1`, and `5.1`.
- The full 12-feature-set outputs only exist in the committed tree for `chro_A` under `3.0`, `4.0`, and `5.0`.

### 6.0 FC on chronological `v2` data

`6.0-fc-chro-all.ipynb` reuses the sequence-style higher-order feature sets on plain FC models over chronological splits.

Saved feature sets:

- `6F`
- `8F theta`
- `8F rho`

Saved best result by dataset:

| Dataset | Best model | Gain vs analytic | SSE | Notes |
|---|---|---:|---:|---|
| `chro_A` | `6F` | -239.41% | 77.6954 | Major underperformance |
| `chro_B` | `8F rho` | 22.01% | 30.8530 | Only clearly positive saved chrono-FC result |
| `chro_C` | `8F theta` | -10.06% | 11.5187 | Still worse than analytic |
| `chro_D` | `6F` | -13.56% | 8.3624 | Still worse than analytic |

Takeaway: the chronological FC rerun is not competitive with the sequence families and is mostly a negative result.

## Gain Tables and Residual Diagnostics

- Every committed run folder has `metrics_summary.csv`.
- Every committed run folder has `residual_diagnostics.csv`.
- Sequence runs also save `gain_table.csv`.

Important implementation detail from `src/model3_utils.py`:

- `gain_table.csv` is saved only for the first feature set in a sequence run, not for every feature set.
- For the full `chro_A` runs that means the gain table corresponds to `3F`.
- For the reduced `B/C/D` runs it corresponds to the first saved reduced model, typically `6F`.

Residual-diagnostic pattern across the better sequence runs:

- the analytic benchmark is consistently right-skewed with heavy tails
- the best GRU/LSTM sequence models reduce residual standard deviation sharply
- the strongest B/C results also reduce skew substantially relative to the analytic model

## Figure Outputs Present

The current repo contains these exported figures in `output/fig/`:

- `1.0-range-A.png`
- `1.1-range-B.png`
- `1.2-range-C.png`
- `1.3-range-D.png`
- `1.x-gain.png`
- `2.x-1F-splits-ABC.png`
- `2.x-2F-splits-ABC.png`
- `2.x-DvsD-iv-lag.png`
- `2.x-DvsD-iv-lag.csv`
- `2.x-avg-feature-gain-A-B-C.csv`
- `4.2-winner-map.png`
- `4.3-7-Model Performance and Stability.png`
- `model-FC.png`
- `model-GRU.png`
- `model-LSTM.png`
- `model-TFT.png`
- `model-all.png`

These correspond to the four figure notebooks under `notebook/fig/`.

## Recommended Interpretation of the Rebuilt Repo

If this project were rerun or described in the paper today, the cleanest experiment story is:

1. Build data with `01` -> `02` -> `03`.
2. Treat `05-data-split-chro.ipynb` as the active split notebook.
3. Treat `1.x` and `2.x` as random-split FC baselines and feature-search experiments.
4. Treat `3.x`, `4.x`, and `5.x` as the main chronological-sequence experiments.
5. Treat `6.0` as a chrono-FC ablation showing that high-order features alone are not enough without sequence structure.

That is the experiment lineage most consistent with the current code and saved outputs.
