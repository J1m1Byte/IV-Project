# PyTorch fully-connected ANN utilities optimized for Colab GPUs

import gc
import shutil
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.metrics import gain, metrics, residual_diagnostics



# === GPU detection ===

def detect_device():
    """
    Auto-detect compute device and return config dict.

    Returns dict with keys: GPU, BATCH, POLICY, DEVICE
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0).lower()

        if 'h100' in name:
            cfg = dict(GPU='H100-80GB', BATCH=65_536, POLICY=torch.bfloat16)
        elif 'a100' in name:
            cfg = dict(GPU='A100-80GB', BATCH=65_536, POLICY=torch.bfloat16)
        elif 'l4' in name:
            cfg = dict(GPU='L4', BATCH=32_768, POLICY=torch.bfloat16)
        elif 't4' in name:
            cfg = dict(GPU='T4', BATCH=8_192, POLICY=torch.float16)
        else:
            cfg = dict(GPU=name[:20], BATCH=4_096, POLICY=torch.float16)

        cfg['DEVICE'] = torch.device('cuda')
        cfg['DEVICE_TYPE'] = 'cuda'
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

        free, total = torch.cuda.mem_get_info()
        print(f'{cfg["GPU"]}  |  VRAM: {total / 1e9:.0f} GB  |  '
              f'batch={cfg["BATCH"]:,}  |  dtype={cfg["POLICY"]}')

    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        cfg = dict(GPU='Apple-MPS', BATCH=4_096, POLICY=torch.float32,
                   DEVICE=torch.device('mps'), DEVICE_TYPE='mps')
        print(f'Apple MPS  |  batch={cfg["BATCH"]:,}  |  dtype={cfg["POLICY"]}')

    else:
        cfg = dict(GPU='CPU', BATCH=2_048, POLICY=torch.float32,
                   DEVICE=torch.device('cpu'), DEVICE_TYPE='cpu')
        print(f'CPU  |  batch={cfg["BATCH"]:,}  |  dtype={cfg["POLICY"]}')

    return cfg


def build_feature_combos(base_features, extra_features, max_extra=3):
    """
    Build feature combinations in the same style as 2.0-fc-rand-A-feature.

    Returns list of (name, feature_list).
    """
    combos = [('3F', list(base_features))]
    upper = min(max_extra, len(extra_features))
    for k in range(1, upper + 1):
        for combo in combinations(extra_features, k):
            name = '3F+' + '+'.join(combo)
            combos.append((name, list(base_features) + list(combo)))
    return combos


def dropna_splits(df_train, df_val, df_test, required_cols):
    """
    Drop rows with missing values in required columns for each split.

    Returns cleaned (df_train, df_val, df_test, stats_dict).
    """
    before = len(df_train) + len(df_val) + len(df_test)
    df_train = df_train.dropna(subset=required_cols).reset_index(drop=True)
    df_val = df_val.dropna(subset=required_cols).reset_index(drop=True)
    df_test = df_test.dropna(subset=required_cols).reset_index(drop=True)
    after = len(df_train) + len(df_val) + len(df_test)
    stats = dict(before=before, after=after, dropped=before - after)
    return df_train, df_val, df_test, stats


def _stage_drive_file(path, cache_dir=None):
    """
    Copy a file from a mounted Drive path into local scratch storage.
    """
    src = Path(path)
    cache_root = Path(cache_dir or '/tmp/iv_project_parquet_cache')
    cache_root.mkdir(parents=True, exist_ok=True)
    dst = cache_root / src.name

    if not dst.exists():
        shutil.copy2(src, dst)
        return dst

    try:
        src_stat = src.stat()
        dst_stat = dst.stat()
    except OSError:
        shutil.copy2(src, dst)
        return dst

    if src_stat.st_size != dst_stat.st_size or src_stat.st_mtime > dst_stat.st_mtime:
        shutil.copy2(src, dst)

    return dst


def read_parquet_safe(path, *, cache_dir=None, local_first=None, **kwargs):
    """
    Read parquet robustly on Colab, retrying from local scratch if Drive drops.
    """
    path = Path(path)
    if local_first is None:
        local_first = str(path).startswith('/content/drive/')

    def _read(target):
        return pd.read_parquet(target, **kwargs)

    if local_first:
        staged = _stage_drive_file(path, cache_dir=cache_dir)
        return _read(staged)

    try:
        return _read(path)
    except OSError as exc:
        msg = str(exc)
        if exc.errno != 107 and 'Transport endpoint is not connected' not in msg:
            raise
        staged = _stage_drive_file(path, cache_dir=cache_dir)
        return _read(staged)


def load_split_bundle(clean_data_dir, dataset, *, cache_dir=None, local_first=None, **kwargs):
    """
    Load train/val/test parquet splits for a dataset tag.
    """
    clean_data_dir = Path(clean_data_dir)
    splits = {}
    for split in ('train', 'val', 'test'):
        path = clean_data_dir / f'{dataset}_{split}.parquet'
        splits[split] = read_parquet_safe(
            path,
            cache_dir=cache_dir,
            local_first=local_first,
            **kwargs,
        )
    return splits['train'], splits['val'], splits['test']



# === Model ===

class ANN_ReLU(nn.Module):
    """3 hidden layers x neurons ReLU + BatchNorm + linear output."""

    def __init__(self, n_features, neurons=80, hidden_layers=3, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        layers = []
        in_dim = n_features
        for _ in range(hidden_layers):
            layers += [
                nn.Linear(in_dim, neurons),
                nn.BatchNorm1d(neurons),
                nn.ReLU(inplace=True),
            ]
            in_dim = neurons
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)



# === Data preparation ===

def prepare_gpu_data(df_train, df_val, df_test, all_features, target, device):
    """
    Scale features from DataFrames and move everything to GPU.

    Parameters
    ----------
    df_train, df_val, df_test : pd.DataFrame
    all_features : list[str] — all feature column names
    target : str — target column name
    device : torch.device

    Returns
    -------
    dict with: Xtr, Xva, Xte (GPU tensors), ytr, yva (GPU tensors),
               y_test (numpy), scaler, col_idx
    """
    X_train = df_train[all_features].values.astype(np.float32)
    X_val = df_val[all_features].values.astype(np.float32)
    X_test = df_test[all_features].values.astype(np.float32)

    ytr = df_train[target].values.astype(np.float32).reshape(-1, 1)
    yva = df_val[target].values.astype(np.float32).reshape(-1, 1)
    y_test = df_test[target].values.astype(np.float32).reshape(-1, 1)

    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(X_train).astype(np.float32)
    Xva_sc = scaler.transform(X_val).astype(np.float32)
    Xte_sc = scaler.transform(X_test).astype(np.float32)

    col_idx = {name: i for i, name in enumerate(all_features)}

    out = dict(
        Xtr=torch.tensor(Xtr_sc, dtype=torch.float32, device=device),
        Xva=torch.tensor(Xva_sc, dtype=torch.float32, device=device),
        Xte=torch.tensor(Xte_sc, dtype=torch.float32, device=device),
        ytr=torch.tensor(ytr, dtype=torch.float32, device=device),
        yva=torch.tensor(yva, dtype=torch.float32, device=device),
        y_test=y_test,
        scaler=scaler,
        col_idx=col_idx,
    )

    if device.type == 'cuda':
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        print(f'Data on GPU  |  '
              f'VRAM used: {(total - free) / 1e9:.2f} GB / {total / 1e9:.0f} GB  |  '
              f'Free: {free / 1e9:.1f} GB')

    print(f'Train: {Xtr_sc.shape[0]:,}  Val: {Xva_sc.shape[0]:,}  '
          f'Test: {Xte_sc.shape[0]:,}  Features: {Xtr_sc.shape[1]}')

    del X_train, X_val, X_test, Xtr_sc, Xva_sc, Xte_sc
    gc.collect()

    return out



# === Training ===

@torch.no_grad()
def _eval_loss(model, X, Y, loss_fn, amp_dtype, use_amp, device_type):
    model.eval()
    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
        return loss_fn(model(X), Y).item()


def train_one_model(name, feature_cols, *,
                    Xtr, Xva, Xte, ytr, yva, y_test,
                    hw_sse, all_feature_names,
                    device, amp_dtype, use_amp,
                    seed=42, batch_size=4096, max_epochs=100,
                    patience=25, lr_patience=8, lr_factor=0.3,
                    init_lr=1e-3, warmup_epochs=5,
                    neurons=80, hidden_layers=3):
    """
    Train a single ANN model on GPU with AMP.

    Parameters
    ----------
    name           : model label string
    feature_cols   : list of column indices into the full feature matrix
    Xtr, Xva, Xte  : full scaled tensors on GPU (all features)
    ytr, yva       : target tensors on GPU
    y_test         : numpy array of test targets
    hw_sse         : float, analytic benchmark SSE for gain calc
    all_feature_names : list of all feature name strings
    device, amp_dtype, use_amp : GPU config
    Other params   : training hyperparameters

    Returns
    -------
    dict with: name, features, n_features, SSE, RMSE, MAE,
               gain_vs_hw, epochs, training_time, y_pred, history
    """
    Xtr_sub = Xtr[:, feature_cols]
    Xva_sub = Xva[:, feature_cols]
    Xte_sub = Xte[:, feature_cols]
    n_feat = len(feature_cols)
    n_tr = Xtr_sub.shape[0]

    model = ANN_ReLU(n_feat, neurons=neurons, hidden_layers=hidden_layers,
                     seed=seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
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

    t0 = time.perf_counter()

    for epoch in range(max_epochs):
        if epoch < warmup_epochs:
            warmup_lr = init_lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        model.train()
        perm = torch.randperm(n_tr, device=device)

        running = 0.0
        steps = 0
        for start in range(0, n_tr, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = Xtr_sub[idx], ytr[idx]

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach().float().item())
            steps += 1

        val_loss = _eval_loss(
            model, Xva_sub, yva, loss_fn, amp_dtype, use_amp, device_type
        )
        train_loss = running / max(steps, 1)
        hist_loss.append(train_loss)
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
                break

    training_time = time.perf_counter() - t0
    ep = len(hist_loss)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type=device_type, dtype=amp_dtype, enabled=use_amp
    ):
        y_pred = model(Xte_sub).float().cpu().numpy().reshape(-1, 1)

    met = metrics(y_test, y_pred)
    g = gain(met['SSE'], hw_sse) * 100

    return dict(
        name=name,
        features=[all_feature_names[c] for c in feature_cols],
        n_features=n_feat,
        feature_cols=feature_cols,
        epochs=ep,
        best_val_loss=best_val,
        y_pred=y_pred,
        model=model,
        training_time=training_time,
        history={'loss': hist_loss, 'val_loss': hist_val},
        sse=met['SSE'],
        rmse=met['RMSE'],
        **met,
        gain_vs_hw=g,
    )


def train_feature_sweep(feature_combos, *, col_idx, train_kwargs, print_every=25):
    """
    Train all feature combos and return results dict + elapsed seconds.
    """
    all_results = {}
    t_global = time.time()

    for i, (name, feats) in enumerate(feature_combos, 1):
        cols = [col_idx[f] for f in feats]
        t0 = time.time()
        result = train_one_model(name, cols, **train_kwargs)
        all_results[name] = result

        if i % print_every == 0 or i == len(feature_combos) or i <= 1:
            elapsed = (time.time() - t_global) / 60
            dt = time.time() - t0
            print(
                f'  [{i:>3}/{len(feature_combos)}] {name:<30} '
                f'SSE={result["sse"]:.4f}  '
                f'Gain={result["gain_vs_hw"]:+.1f}%  '
                f'ep={result["epochs"]}  '
                f'{dt:.1f}s  '
                f'elapsed={elapsed:.1f}min'
            )

    elapsed_s = time.time() - t_global
    return all_results, elapsed_s


def build_results_frame(all_results):
    """
    Convert model results dict into ranked DataFrame.
    """
    rows = []
    for name, result in all_results.items():
        rows.append({
            'combo_name': name,
            'n_features': result['n_features'],
            'features': ', '.join(result['features']),
            'SSE': result['sse'],
            'RMSE': result['rmse'],
            'Gain_vs_HW_%': result['gain_vs_hw'],
            'training_time_s': result['training_time'],
            'epochs_run': result['epochs'],
        })
    return pd.DataFrame(rows).sort_values('SSE').reset_index(drop=True)


def save_colab_run(run_dir, *, y_test, hw, models):
    """
    Save Colab/PyTorch run artifacts with output format close to save_run().
    """
    run_dir = Path(run_dir)
    history_dir = run_dir / 'train-history'
    history_dir.mkdir(parents=True, exist_ok=True)
    yt = np.asarray(y_test).ravel()

    # metrics summary
    summary_rows = []
    hw_met = metrics(yt, hw['y_pred'])
    hw_met['Model'] = 'Analytic'
    hw_met['Training_time'] = None
    hw_met['Gain_vs_Analytic'] = None
    hw_met['Gain_Incremental'] = None
    summary_rows.append(hw_met)

    prev_sse = hw_met['SSE']
    first_model = True
    total_time = 0.0

    for name, result in models.items():
        met = metrics(yt, result['y_pred'])
        met['Model'] = name
        met['Training_time'] = f"{result['training_time']:.1f}s"
        met['Gain_vs_Analytic'] = f"{gain(met['SSE'], hw_met['SSE']) * 100:.2f}%"
        met['Gain_Incremental'] = None if first_model else f"{gain(met['SSE'], prev_sse) * 100:.2f}%"
        summary_rows.append(met)
        prev_sse = met['SSE']
        first_model = False
        total_time += result['training_time']

        safe_name = name.replace('+', '_')
        np.save(history_dir / f'{safe_name}_predictions.npy', result['y_pred'])
        pd.DataFrame({
            'epoch': range(1, result['epochs'] + 1),
            'loss': result['history']['loss'],
            'val_loss': result['history']['val_loss'],
        }).to_csv(history_dir / f'{safe_name}_history.csv', index=False)
        torch.save(result['model'].state_dict(), history_dir / f'{safe_name}_weights.pt')

    total_row = {col: None for col in summary_rows[0]}
    total_row['Model'] = 'Total'
    total_row['Training_time'] = f'{total_time:.1f}s'
    summary_rows.append(total_row)

    col_order = ['Model', 'SSE', 'MSE', 'RMSE', 'MAE', 'MeanError', 'MedianAE',
                 'R2', 'Training_time', 'Gain_vs_Analytic', 'Gain_Incremental']
    summary = pd.DataFrame(summary_rows)[col_order]
    summary.to_csv(run_dir / 'metrics_summary.csv', index=False)

    # residual diagnostics
    diag_rows = [residual_diagnostics(yt, hw['y_pred'], label='Analytic')]
    for name, result in models.items():
        diag_rows.append(residual_diagnostics(yt, result['y_pred'], label=name))
    pd.DataFrame(diag_rows).to_csv(run_dir / 'residual_diagnostics.csv', index=False)

    # exploration ranking table
    df_results = build_results_frame(models)
    df_results.to_csv(history_dir / 'feature_exploration_results.csv', index=False)

    return summary, df_results
