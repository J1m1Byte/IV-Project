"""
Shared utilities for Tier 1 sequence models (LSTM, GRU, TFT).

Handles random-split sequence construction, dataset creation, and
common training/evaluation infrastructure shared across all sequence
model notebooks (3.x series).

Random-split sequence construction
----------------------------------
Because random splits scatter rows from the same contract across
train/val/test, sequences must be built from the FULL date-range
dataframe in true temporal order, then assigned to a partition based
on the target row's split label.

Approach:
  1. Load the full feature dataframe for the appropriate date range.
  2. Sort by (k, expiration, date) within each contract.
  3. Build sliding windows of length LOOKBACK from each contract group.
  4. Assign each sequence to train/val/test by matching the target row
     (last row of the window) to the split partition.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from src.metrics import gain, metrics, residual_diagnostics

try:
    from tqdm.notebook import tqdm
except Exception:
    from tqdm import tqdm


# ── Constants ────────────────────────────────────────────────────────────────

LOOKBACK = 20
MIN_STEPS_PER_EPOCH = 50

FEATURE_SETS = {
    # ── 3F (base) ───────────────────────────────────────────────────────────────
    '3F':               ['delta', 'T', 'spy_ret'],
    '3F+iv_lag':        ['delta', 'T', 'spy_ret', 'iv_lag'],
    
    # ── 4F (3F +vix_lag) ──────────────────────────────────────────────────────────
    '4F':               ['delta', 'T', 'spy_ret', 'vix_lag'],
    '4F+iv_lag':        ['delta', 'T', 'spy_ret', 'vix_lag', 'iv_lag'],

    # ── 6F (4F +vix_mom_lag +gamma/theta/rho) ───────────────────────────────────────
    '6F_gamma':         ['delta', 'T', 'spy_ret', 'vix_lag', 'vix_mom_lag', 'gamma'],
    '6F_gamma+iv_lag':  ['delta', 'T', 'spy_ret', 'vix_lag', 'vix_mom_lag', 'gamma', 'iv_lag'],
    
    '6F_theta':         ['delta', 'T', 'spy_ret', 'vix_lag', 'vix_mom_lag', 'theta'],
    '6F_theta+iv_lag':  ['delta', 'T', 'spy_ret', 'vix_lag', 'vix_mom_lag', 'theta', 'iv_lag'],

    # ── 8F (6F_gamma +vix_mom +theta/rho) ──────────────────────────────────
    '8F_theta':         ['delta', 'T', 'spy_ret', 'vix_lag', 'vix_mom_lag', 'vix_mom', 'gamma', 'theta'],
    '8F_theta+iv_lag':  ['delta', 'T', 'spy_ret', 'vix_lag', 'vix_mom_lag', 'vix_mom', 'gamma', 'theta', 'iv_lag'],
    
    '8F_rho':           ['delta', 'T', 'spy_ret', 'vix_lag', 'vix_mom_lag', 'vix_mom', 'gamma', 'rho'],
    '8F_rho+iv_lag':    ['delta', 'T', 'spy_ret', 'vix_lag', 'vix_mom_lag', 'vix_mom', 'gamma', 'rho', 'iv_lag'],
}

TARGET = 'd_iv'

GROUP_KEYS = ['k', 'expiration']
SORT_KEYS = ['k', 'expiration', 'date']


# ── GPU detection (reuse from fully_connected_colab) ────────────────────────

def detect_device():
    """Auto-detect compute device and return config dict."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0).lower()

        if 'h100' in name:
            cfg = dict(GPU='H100-80GB', MAX_BATCH=4_096, POLICY=torch.bfloat16)
        elif 'a100' in name:
            cfg = dict(GPU='A100-80GB', MAX_BATCH=4_096, POLICY=torch.bfloat16)
        elif 'l4' in name:
            cfg = dict(GPU='L4', MAX_BATCH=2_048, POLICY=torch.bfloat16)
        elif 't4' in name:
            cfg = dict(GPU='T4', MAX_BATCH=1_024, POLICY=torch.float16)
        else:
            cfg = dict(GPU=name[:20], MAX_BATCH=512, POLICY=torch.float16)

        cfg['DEVICE'] = torch.device('cuda')
        cfg['DEVICE_TYPE'] = 'cuda'
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

        free, total = torch.cuda.mem_get_info()
        print(f'{cfg["GPU"]}  |  VRAM: {total / 1e9:.0f} GB  |  '
              f'MAX_BATCH={cfg["MAX_BATCH"]:,}  |  dtype={cfg["POLICY"]}')

    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        cfg = dict(GPU='Apple-MPS', MAX_BATCH=512, POLICY=torch.float32,
                   DEVICE=torch.device('mps'), DEVICE_TYPE='mps')
        print(f'Apple MPS  |  MAX_BATCH={cfg["MAX_BATCH"]:,}  |  dtype={cfg["POLICY"]}')

    else:
        cfg = dict(GPU='CPU', MAX_BATCH=256, POLICY=torch.float32,
                   DEVICE=torch.device('cpu'), DEVICE_TYPE='cpu')
        print(f'CPU  |  MAX_BATCH={cfg["MAX_BATCH"]:,}  |  dtype={cfg["POLICY"]}')

    return cfg


def compute_batch_size(n_train, max_batch, min_steps=MIN_STEPS_PER_EPOCH):
    """
    Adaptive batch size ensuring at least *min_steps* gradient steps per epoch.

    Returns min(max_batch, n_train // min_steps), rounded down to the
    nearest power of 2, with a floor of 64.
    """
    target = min(max_batch, n_train // min_steps)
    if target < 64:
        return 64
    p = 1
    while p * 2 <= target:
        p *= 2
    return p


# ── Random-split sequence construction ──────────────────────────────────────

def build_split_index(df_train, df_val, df_test):
    """
    Build a mapping from (date, k, expiration) → split label.

    Returns a dict: {(date, k, expiration): 'train'|'val'|'test'}
    """
    idx = {}
    for label, df in [('train', df_train), ('val', df_val), ('test', df_test)]:
        for row in df[['date', 'k', 'expiration']].itertuples(index=False):
            idx[(row.date, row.k, row.expiration)] = label
    return idx


def build_split_index_fast(df_train, df_val, df_test):
    """
    Build split assignment using merge instead of row-by-row dict lookup.

    Returns a Series aligned to the full dataframe index with values
    'train', 'val', 'test', or NaN for unmatched rows.
    """
    keys = ['date', 'k', 'expiration']
    parts = []
    for label, df in [('train', df_train), ('val', df_val), ('test', df_test)]:
        part = df[keys].copy()
        part['_split'] = label
        parts.append(part)
    lookup = pd.concat(parts, ignore_index=True).drop_duplicates(subset=keys)
    return lookup


def assign_sequences_to_splits(
    df_full, df_train, df_val, df_test,
    feature_cols, target=TARGET, lookback=LOOKBACK,
):
    """
    Build sequences from the full ordered dataframe and assign each to
    train/val/test based on the target row's split membership.

    Parameters
    ----------
    df_full : pd.DataFrame
        Full feature dataframe (e.g., 03-data-merge-feature.parquet),
        filtered to the appropriate date range.
    df_train, df_val, df_test : pd.DataFrame
        Split dataframes with (date, k, expiration) identifying rows.
    feature_cols : list[str]
        Feature columns to include in sequences.
    target : str
        Target column name.
    lookback : int
        Sequence window length.

    Returns
    -------
    dict with keys: X_train, y_train, X_val, y_val, X_test, y_test,
                    test_indices (indices into df_full for test target rows)
    """
    keys = ['date', 'k', 'expiration']
    cols_needed = keys + feature_cols + [target]
    cols_needed = list(dict.fromkeys(cols_needed))  # deduplicate preserving order

    # Sort full data by contract and date
    df = df_full[cols_needed].copy()
    df = df.sort_values(SORT_KEYS).reset_index(drop=True)

    # Build fast split lookup via merge
    lookup = build_split_index_fast(df_train, df_val, df_test)
    df = df.merge(lookup, on=keys, how='left')

    # Extract arrays for speed
    feat_arr = df[feature_cols].values.astype(np.float32)
    target_arr = df[target].values.astype(np.float32)
    split_arr = df['_split'].values  # object array: 'train'|'val'|'test'|NaN
    group_ids = df.groupby(GROUP_KEYS, sort=False).ngroup().values

    n = len(df)
    seqs = {s: ([], []) for s in ('train', 'val', 'test')}
    test_global_indices = []

    # Build sequences within each contract group
    prev_gid = -1
    group_start = 0
    for i in range(n + 1):
        gid = group_ids[i] if i < n else -2
        if gid != prev_gid and i > 0:
            # Process group [group_start, i)
            group_len = i - group_start
            if group_len >= lookback:
                for j in range(group_start + lookback - 1, i):
                    target_split = split_arr[j]
                    if target_split not in ('train', 'val', 'test'):
                        continue
                    # Check for NaN in target
                    if np.isnan(target_arr[j]):
                        continue
                    seq_start = j - lookback + 1
                    seq_features = feat_arr[seq_start:j + 1]  # (lookback, n_feat)
                    # Skip sequences with NaN in features
                    if np.isnan(seq_features).any():
                        continue
                    seqs[target_split][0].append(seq_features)
                    seqs[target_split][1].append(target_arr[j])
                    if target_split == 'test':
                        test_global_indices.append(j)
            group_start = i
        prev_gid = gid

    result = {}
    for split_name in ('train', 'val', 'test'):
        X_list, y_list = seqs[split_name]
        if len(X_list) > 0:
            result[f'X_{split_name}'] = np.stack(X_list)
            result[f'y_{split_name}'] = np.array(y_list, dtype=np.float32)
        else:
            n_feat = len(feature_cols)
            result[f'X_{split_name}'] = np.empty((0, lookback, n_feat), dtype=np.float32)
            result[f'y_{split_name}'] = np.empty((0,), dtype=np.float32)

    result['test_indices'] = np.array(test_global_indices, dtype=np.int64)
    result['df_sorted'] = df  # keep for HW alignment

    return result


# ── Cached split structure (avoids redundant sort/merge across feature sets) ─

def precompute_split_structure(df_full, df_train, df_val, df_test,
                               target=TARGET, lookback=LOOKBACK):
    """
    Precompute sorted dataframe, split assignments, and valid sequence
    windows once per dataset.  Call this once, then pass the returned
    cache to ``build_sequences_from_cache()`` for each feature set.

    The heavy work (sort → merge → group boundary scan → target/split
    validation) is done here so it is never repeated.

    Returns
    -------
    dict  –  opaque cache consumed by ``build_sequences_from_cache``.
    """
    keys = ['date', 'k', 'expiration']

    df = df_full.copy()
    df = df.sort_values(SORT_KEYS).reset_index(drop=True)

    lookup = build_split_index_fast(df_train, df_val, df_test)
    df = df.merge(lookup, on=keys, how='left')

    target_arr = df[target].values.astype(np.float32)
    split_arr  = df['_split'].values                    # 'train'|'val'|'test'|NaN
    group_ids  = df.groupby(GROUP_KEYS, sort=False).ngroup().values

    # ── group boundaries ────────────────────────────────────────────
    n = len(df)
    boundaries = []
    if n > 0:
        prev_gid = group_ids[0]
        group_start = 0
        for i in range(1, n + 1):
            gid = group_ids[i] if i < n else -2
            if gid != prev_gid:
                boundaries.append((group_start, i))
                group_start = i
            prev_gid = gid

    # ── valid windows (target non-NaN & split assigned) ─────────────
    # Pre-filtering these avoids repeating the check for every feature set.
    valid_windows = []          # list of (seq_start, target_idx, split_label)
    for gs, ge in boundaries:
        if ge - gs < lookback:
            continue
        for j in range(gs + lookback - 1, ge):
            s = split_arr[j]
            if s not in ('train', 'val', 'test'):
                continue
            if np.isnan(target_arr[j]):
                continue
            valid_windows.append((j - lookback + 1, j, s))

    return {
        'df': df,
        'target_arr': target_arr,
        'valid_windows': valid_windows,
        'lookback': lookback,
    }


def build_sequences_from_cache(cache, feature_cols):
    """
    Build train/val/test sequence arrays for *feature_cols* using a
    pre-computed split structure (from ``precompute_split_structure``).

    Only the per-feature-set work is done here: extracting the feature
    array and checking for NaN in feature columns.

    Returns the same dict shape as ``assign_sequences_to_splits``.
    """
    df         = cache['df']
    target_arr = cache['target_arr']
    valid_windows = cache['valid_windows']
    lookback   = cache['lookback']

    feat_arr = df[feature_cols].values.astype(np.float32)

    seqs = {s: ([], []) for s in ('train', 'val', 'test')}
    test_global_indices = []

    for seq_start, j, split_label in valid_windows:
        seq_features = feat_arr[seq_start:j + 1]       # (lookback, n_feat)
        if np.isnan(seq_features).any():
            continue
        seqs[split_label][0].append(seq_features)
        seqs[split_label][1].append(target_arr[j])
        if split_label == 'test':
            test_global_indices.append(j)

    result = {}
    for split_name in ('train', 'val', 'test'):
        X_list, y_list = seqs[split_name]
        if len(X_list) > 0:
            result[f'X_{split_name}'] = np.stack(X_list)
            result[f'y_{split_name}'] = np.array(y_list, dtype=np.float32)
        else:
            n_feat = len(feature_cols)
            result[f'X_{split_name}'] = np.empty((0, lookback, n_feat), dtype=np.float32)
            result[f'y_{split_name}'] = np.empty((0,), dtype=np.float32)

    result['test_indices'] = np.array(test_global_indices, dtype=np.int64)
    result['df_sorted'] = df
    return result


# ── Scaling ─────────────────────────────────────────────────────────────────

def scale_sequences(X_train, X_val, X_test):
    """
    Fit StandardScaler on training sequences and transform all splits.

    Reshapes (N, lookback, F) → (N*lookback, F) for fitting, then back.

    Returns (X_train_sc, X_val_sc, X_test_sc, scaler).
    """
    N_tr, L, F = X_train.shape
    scaler = StandardScaler()
    X_tr_flat = X_train.reshape(-1, F)
    scaler.fit(X_tr_flat)

    X_train_sc = scaler.transform(X_tr_flat).reshape(N_tr, L, F).astype(np.float32)
    if len(X_val) > 0:
        X_val_sc = scaler.transform(X_val.reshape(-1, F)).reshape(len(X_val), L, F).astype(np.float32)
    else:
        X_val_sc = X_val.copy()
    if len(X_test) > 0:
        X_test_sc = scaler.transform(X_test.reshape(-1, F)).reshape(len(X_test), L, F).astype(np.float32)
    else:
        X_test_sc = X_test.copy()

    return X_train_sc, X_val_sc, X_test_sc, scaler


# ── PyTorch Dataset ─────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """Simple dataset for (X_seq, y) pairs."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Training loop (shared for LSTM / GRU) ──────────────────────────────────

def train_seq_model(
    model, train_loader, val_loader, *,
    device, amp_dtype, use_amp,
    max_epochs=100, patience=25, lr_patience=8, lr_factor=0.3,
    init_lr=1e-3, warmup_epochs=5, use_tqdm=False, desc='Training',
    reduce_batch_on_oom=True,
):
    """
    Train a sequence model with early stopping and LR scheduling.

    Parameters
    ----------
    use_tqdm : bool
        If True, display a tqdm progress bar over epochs.
    desc : str
        Label for the tqdm bar (e.g. model/feature-set name).
    reduce_batch_on_oom : bool
        If True, attempt to recover from OOM by reducing batch size.

    Returns dict with: model (best state loaded), history, epochs, training_time
    """
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f'  WARNING: OOM during optimizer creation. Clearing cache and retrying.')
            torch.cuda.empty_cache()
            optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
        else:
            raise
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor,
        patience=lr_patience, min_lr=1e-6)
    loss_fn = nn.MSELoss()
    device_type = device.type if device.type in ('cuda', 'cpu') else 'cpu'
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and device.type == 'cuda'))

    best_val = float('inf')
    best_state = None
    wait = 0
    hist_loss, hist_val = [], []

    pbar = None
    if use_tqdm:
        pbar = tqdm(total=max_epochs, desc=desc, unit='epoch',
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| '
                               '{n_fmt}/{total_fmt} epochs '
                               '[{elapsed}<{remaining}]  {postfix}')

    t0 = time.perf_counter()

    for epoch in range(max_epochs):
        # Warmup LR
        if epoch < warmup_epochs:
            warmup_lr = init_lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        # ── Train ────────────────────────────────────────────────────
        model.train()
        running, steps = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach().float().item())
            steps += 1

        train_loss = running / max(steps, 1)
        hist_loss.append(train_loss)

        # ── Validate ─────────────────────────────────────────────────
        model.eval()
        val_running, val_steps = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                    vloss = loss_fn(model(xb), yb)
                val_running += float(vloss.detach().float().item())
                val_steps += 1
        val_loss = val_running / max(val_steps, 1)
        hist_val.append(val_loss)

        if epoch >= warmup_epochs:
            scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if pbar is not None:
                    pbar.update(1)
                break

        if pbar is not None:
            lr_now = optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=f'{train_loss:.2e}', val=f'{val_loss:.2e}',
                             lr=f'{lr_now:.1e}', wait=wait)
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    training_time = time.perf_counter() - t0
    ep = len(hist_loss)

    # Restore best
    model.load_state_dict(best_state)
    model.eval()

    return dict(
        model=model,
        history={'loss': hist_loss, 'val_loss': hist_val},
        epochs=ep,
        training_time=training_time,
        best_val_loss=best_val,
    )


def predict_seq(model, X_tensor, device, amp_dtype, use_amp, batch_size=2048):
    """Run inference on a tensor/array and return numpy predictions."""
    model.eval()
    device_type = device.type if device.type in ('cuda', 'cpu') else 'cpu'
    if isinstance(X_tensor, np.ndarray):
        X_tensor = torch.tensor(X_tensor, dtype=torch.float32)
    X_tensor = X_tensor.to(device)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            xb = X_tensor[i:i + batch_size]
            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                yp = model(xb)
            preds.append(yp.float().cpu())
    return torch.cat(preds).numpy().reshape(-1, 1)


# ── HW benchmark alignment ─────────────────────────────────────────────────

def hw_predict_aligned(hw_coef, df_sorted, test_indices):
    """
    Compute Hull-White predictions for the exact test rows that produced
    sequences for a given feature set.

    Parameters
    ----------
    hw_coef : dict with keys 'a', 'b', 'c' from analytic_benchmark
    df_sorted : DataFrame (sorted, with all columns) from assign_sequences_to_splits
    test_indices : array of indices into df_sorted for the test target rows

    Returns
    -------
    y_hw_pred : (n_test, 1) array of HW predictions
    y_true : (n_test,) array of true targets
    hw_sse : float
    """
    rows = df_sorted.iloc[test_indices]
    delta = rows['delta'].values
    T = rows['T'].values
    ret = rows['spy_ret'].values
    y_true = rows[TARGET].values

    sqrt_T = np.sqrt(T)
    a, b, c = hw_coef['a'], hw_coef['b'], hw_coef['c']
    y_hw = (a * ret / sqrt_T +
            b * delta * ret / sqrt_T +
            c * delta**2 * ret / sqrt_T)
    y_hw_pred = y_hw.reshape(-1, 1)

    residuals = y_true - y_hw.ravel()
    hw_sse = float(np.sum(residuals**2))

    return y_hw_pred, y_true, hw_sse


# ── Results persistence ─────────────────────────────────────────────────────

def save_seq_run(run_dir, *, results_by_fs, hw_coef, df_sorted):
    """
    Save metrics_summary.csv, gain_table.csv, residual_diagnostics.csv,
    and best model weights for a sequence model notebook run.

    Parameters
    ----------
    run_dir : Path
        Output directory for this notebook.
    results_by_fs : dict
        {fs_name: {model, y_pred, y_true, test_indices, history, epochs,
                    training_time, scaler, ...}}
    hw_coef : dict
        Hull-White coefficients from analytic_benchmark.
    df_sorted : DataFrame
        Sorted full dataframe from assign_sequences_to_splits.
    """
    run_dir = Path(run_dir)
    weights_dir = run_dir / 'best_model_weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    print(f'Saving results to: {run_dir}')
    print(f'Feature sets to save: {list(results_by_fs.keys())}')

    summary_rows = []
    diag_rows = []
    total_time = 0.0
    prev_sse = None
    first_model = True

    # Single Analytic row from the first feature set
    first_fs_name = list(results_by_fs.keys())[0]
    first_res = results_by_fs[first_fs_name]
    y_hw_first, _, hw_sse_first = hw_predict_aligned(
        hw_coef, df_sorted, first_res['test_indices'])
    hw_met = metrics(first_res['y_true'], y_hw_first)
    hw_met['Model'] = 'Analytic'
    hw_met['Training_time'] = None
    hw_met['Gain_vs_Analytic'] = None
    hw_met['Gain_Incremental'] = None
    summary_rows.append(hw_met)
    prev_sse = hw_met['SSE']

    diag_rows.append(residual_diagnostics(
        first_res['y_true'], y_hw_first, label='Analytic'))

    for fs_name, res in results_by_fs.items():
        y_pred = res['y_pred']
        y_true = res['y_true']
        test_idx = res['test_indices']

        # HW benchmark aligned to this feature set's test rows
        y_hw, _, hw_sse = hw_predict_aligned(hw_coef, df_sorted, test_idx)

        # Model metrics
        met = metrics(y_true, y_pred)
        g = gain(met['SSE'], hw_sse) * 100

        # Model row
        model_row = met.copy()
        model_row['Model'] = fs_name
        model_row['Training_time'] = f"{res['training_time']:.1f}s"
        model_row['Gain_vs_Analytic'] = f"{g:.2f}%"
        model_row['Gain_Incremental'] = (
            None if first_model
            else f"{gain(met['SSE'], prev_sse) * 100:.2f}%"
        )
        summary_rows.append(model_row)
        prev_sse = met['SSE']
        first_model = False

        total_time += res['training_time']

        # Residual diagnostics
        diag_rows.append(residual_diagnostics(y_true, y_pred, label=fs_name))

        # Save weights
        torch.save(res['model'].state_dict(), weights_dir / f'{fs_name}_weights.pt')

        # Save predictions and history
        np.save(weights_dir / f'{fs_name}_predictions.npy', y_pred)
        pd.DataFrame({
            'epoch': range(1, res['epochs'] + 1),
            'loss': res['history']['loss'],
            'val_loss': res['history']['val_loss'],
        }).to_csv(weights_dir / f'{fs_name}_history.csv', index=False)

    # Total row
    total_row = {col: None for col in summary_rows[0]}
    total_row['Model'] = 'Total'
    total_row['Training_time'] = f'{total_time:.1f}s'
    summary_rows.append(total_row)

    col_order = ['Model', 'SSE', 'MSE', 'RMSE', 'MAE', 'MeanError', 'MedianAE',
                 'R2', 'Training_time', 'Gain_vs_Analytic', 'Gain_Incremental']
    summary = pd.DataFrame(summary_rows)
    for col in col_order:
        if col not in summary.columns:
            summary[col] = None
    summary = summary[col_order]
    summary.to_csv(run_dir / 'metrics_summary.csv', index=False)

    pd.DataFrame(diag_rows).to_csv(run_dir / 'residual_diagnostics.csv', index=False)

    # Gain table: use the 4F results as the representative gain table
    # (since it has the most test rows — no d_iv_lag filtering)
    first_fs = list(results_by_fs.keys())[0]
    res0 = results_by_fs[first_fs]
    _save_gain_table(run_dir, res0, hw_coef, df_sorted)

    print(f'\n✓ Saved metrics_summary.csv')
    print(f'✓ Saved residual_diagnostics.csv')
    print(f'✓ Saved gain_table.csv')
    print(f'✓ Saved {len(results_by_fs)} × 3 artifacts (weights, predictions, history)')
    print(f'✓ All results saved to {run_dir}')

    return summary


def _save_gain_table(run_dir, res, hw_coef, df_sorted):
    """Build and save gain_table.csv using binned test data."""
    from src.metrics import build_gain_table

    test_idx = res['test_indices']
    rows = df_sorted.iloc[test_idx]
    y_true = res['y_true']
    y_pred = res['y_pred'].ravel()

    # HW predictions
    y_hw, _, _ = hw_predict_aligned(hw_coef, df_sorted, test_idx)
    y_hw = y_hw.ravel()

    # Build binned dataframe
    test_df = pd.DataFrame({
        'ret': rows['spy_ret'].values,
        'T': rows['T'].values,
    })
    test_df['se_model'] = (y_true - y_pred) ** 2
    test_df['se_hw'] = (y_true - y_hw) ** 2

    # Return bins
    test_df['ret_bin'] = pd.cut(test_df['ret'],
        bins=[-np.inf, -0.01, 0, 0.01, np.inf],
        labels=['<-1%', '-1% to 0%', '0% to 1%', '>1%'])

    # Maturity bins
    test_df['T_bin'] = pd.cut(test_df['T'],
        bins=[0, 0.25, 0.5, 1.0, np.inf],
        labels=['0-3m', '3-6m', '6m-1yr', '>1yr'],
        include_lowest=True)

    long_df, pivot_df = build_gain_table(
        test_df, 'T_bin', 'ret_bin', 'se_model', 'se_hw')
    long_df.to_csv(Path(run_dir) / 'gain_table.csv', index=False)


# ── Print utilities ─────────────────────────────────────────────────────────

def print_config(cfg, batch_size, init_lr, n_train, max_epochs, patience, warmup_epochs, lookback):
    """Print training configuration summary."""
    steps = n_train // batch_size
    print(f'MAX_BATCH={cfg["MAX_BATCH"]:,}  adaptive BATCH_SIZE={batch_size:,}  '
          f'INIT_LR={init_lr:.6f}  n_train={n_train:,}  steps/epoch~{steps}')
    print(f'MAX_EPOCHS={max_epochs}  PATIENCE={patience}  '
          f'WARMUP={warmup_epochs} epochs  LOOKBACK={lookback}')


def print_feature_set_summary(fs_name, n_train, n_val, n_test, feature_cols):
    """Print per-feature-set data summary."""
    print(f'\n{"=" * 60}')
    print(f'  Feature set: {fs_name}  ({len(feature_cols)} features)')
    print(f'  Train: {n_train:,}  Val: {n_val:,}  Test: {n_test:,} sequences')
    print(f'  Features: {", ".join(feature_cols)}')
    print(f'{"=" * 60}')
