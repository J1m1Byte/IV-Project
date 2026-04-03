"""
TensorFlow/Keras utilities for sequence models (LSTM, GRU, TFT).

Local-machine counterpart of model3_utils.py (which requires PyTorch).
Sequence construction and HW alignment are pure numpy; models use Keras.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from src.metrics import gain, metrics, residual_diagnostics

try:
    from tqdm.notebook import tqdm
except Exception:
    from tqdm import tqdm


# ── Constants (mirrored from model3_utils) ────────────────────────────────────

LOOKBACK = 20
MIN_STEPS_PER_EPOCH = 50

FEATURE_SETS = {
    # ── 3F base ───────────────────────────────────────────────────────────────
    '3F':    ['delta', 'T', 'spy_ret'],

    # ── 3F + 2 (vix_lag, vix_mom_lag) ────────────────────────────────────────
    '5F':    ['delta', 'T', 'spy_ret', 'vix_lag', 'vix_mom_lag'],

    # ── 3F + 3 variants ───────────────────────────────────────────────────────
    '6F_G':  ['delta', 'T', 'spy_ret', 'vix_lag', 'vix_mom_lag', 'gamma'],
    '6F_R':  ['delta', 'T', 'spy_ret', 'vix_lag', 'vix_mom_lag', 'rho'],
    '6F_T':  ['delta', 'T', 'spy_ret', 'vix_lag', 'vix_mom_lag', 'theta'],

    # ── 3F + 5 variants ───────────────────────────────────────────────────────
    '8F_GT': ['delta', 'T', 'spy_ret', 'vix_lag', 'vix_mom', 'vix_mom_lag', 'gamma', 'theta'],
    '8F_GR': ['delta', 'T', 'spy_ret', 'vix_lag', 'vix_mom', 'vix_mom_lag', 'gamma', 'rho'],
}
TARGET = 'd_iv'

GROUP_KEYS = ['k', 'expiration']
SORT_KEYS = ['k', 'expiration', 'date']


# ── Batch size ────────────────────────────────────────────────────────────────

def compute_batch_size(n_train, max_batch=512, min_steps=MIN_STEPS_PER_EPOCH):
    target = min(max_batch, n_train // min_steps)
    if target < 64:
        return 64
    p = 1
    while p * 2 <= target:
        p *= 2
    return p


# ── Sequence construction (numpy-only, from model3_utils) ─────────────────────

def build_split_index_fast(df_train, df_val, df_test):
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
    keys = ['date', 'k', 'expiration']
    cols_needed = keys + feature_cols + [target]
    cols_needed = list(dict.fromkeys(cols_needed))

    df = df_full[cols_needed].copy()
    df = df.sort_values(SORT_KEYS).reset_index(drop=True)

    lookup = build_split_index_fast(df_train, df_val, df_test)
    df = df.merge(lookup, on=keys, how='left')

    feat_arr = df[feature_cols].values.astype(np.float32)
    target_arr = df[target].values.astype(np.float32)
    split_arr = df['_split'].values
    group_ids = df.groupby(GROUP_KEYS, sort=False).ngroup().values

    n = len(df)
    seqs = {s: ([], []) for s in ('train', 'val', 'test')}
    test_global_indices = []

    prev_gid = -1
    group_start = 0
    for i in range(n + 1):
        gid = group_ids[i] if i < n else -2
        if gid != prev_gid and i > 0:
            group_len = i - group_start
            if group_len >= lookback:
                for j in range(group_start + lookback - 1, i):
                    target_split = split_arr[j]
                    if target_split not in ('train', 'val', 'test'):
                        continue
                    if np.isnan(target_arr[j]):
                        continue
                    seq_start = j - lookback + 1
                    seq_features = feat_arr[seq_start:j + 1]
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
    result['df_sorted'] = df
    return result


def scale_sequences(X_train, X_val, X_test):
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


# ── HW benchmark alignment (numpy-only, from model3_utils) ────────────────────

def hw_predict_aligned(hw_coef, df_sorted, test_indices):
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


# ── Keras model builders ──────────────────────────────────────────────────────

def build_lstm(n_features, lookback=LOOKBACK, hidden_size=64, num_layers=2,
               dropout=0.1, seed=42):
    tf.random.set_seed(seed)
    inputs = tf.keras.Input(shape=(lookback, n_features))
    x = inputs
    for i in range(num_layers):
        return_seq = (i < num_layers - 1)
        x = tf.keras.layers.LSTM(
            hidden_size,
            return_sequences=return_seq,
            dropout=dropout if i < num_layers - 1 else 0.0,
        )(x)
    x = tf.keras.layers.Dense(1, kernel_initializer='he_uniform')(x)
    return tf.keras.Model(inputs, x, name='LSTM')


def build_gru(n_features, lookback=LOOKBACK, hidden_size=64, num_layers=2,
              dropout=0.1, seed=42):
    tf.random.set_seed(seed)
    inputs = tf.keras.Input(shape=(lookback, n_features))
    x = inputs
    for i in range(num_layers):
        return_seq = (i < num_layers - 1)
        x = tf.keras.layers.GRU(
            hidden_size,
            return_sequences=return_seq,
            dropout=dropout if i < num_layers - 1 else 0.0,
        )(x)
    x = tf.keras.layers.Dense(1, kernel_initializer='he_uniform')(x)
    return tf.keras.Model(inputs, x, name='GRU')


# ── TFT building blocks (Keras layers) ───────────────────────────────────────

class GatedLinearUnit(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.fc = tf.keras.layers.Dense(units)
        self.gate = tf.keras.layers.Dense(units, activation='sigmoid')

    def call(self, x):
        return self.gate(x) * self.fc(x)


class GatedResidualNetwork(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, output_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = tf.keras.layers.Dense(hidden_dim, activation='elu')
        self.fc2 = tf.keras.layers.Dense(output_dim)
        self.glu = GatedLinearUnit(output_dim)
        self.layernorm = tf.keras.layers.LayerNorm()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.skip = None  # built lazily

    def build(self, input_shape):
        in_dim = input_shape[-1]
        if in_dim != self.output_dim:
            self.skip = tf.keras.layers.Dense(self.output_dim, use_bias=False)
        super().build(input_shape)

    def call(self, x, training=False):
        residual = self.skip(x) if self.skip else x
        h = self.fc1(x)
        h = self.fc2(h)
        h = self.dropout(h, training=training)
        h = self.glu(h)
        return self.layernorm(residual + h)


class VariableSelectionNetwork(tf.keras.layers.Layer):
    def __init__(self, n_features, hidden_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.feature_grns = [
            GatedResidualNetwork(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(n_features)
        ]
        self.selection_grn = GatedResidualNetwork(
            hidden_dim, n_features, dropout=dropout)

    def call(self, x, training=False):
        # x: (B, T, n_features)
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        processed = []
        for i in range(self.n_features):
            feat_i = x[:, :, i:i+1]                          # (B, T, 1)
            feat_i = tf.reshape(feat_i, [B * T, 1])
            out_i = self.feature_grns[i](feat_i, training=training)
            processed.append(tf.reshape(out_i, [B, T, self.hidden_dim]))

        stacked = tf.stack(processed, axis=2)  # (B, T, n_features, hidden_dim)
        flat = tf.reshape(stacked, [B * T, self.n_features * self.hidden_dim])

        weights = self.selection_grn(flat, training=training)  # (B*T, n_features)
        weights = tf.nn.softmax(weights, axis=-1)
        weights = tf.reshape(weights, [B, T, self.n_features, 1])

        selected = tf.reduce_sum(stacked * weights, axis=2)  # (B, T, hidden_dim)
        return selected


def build_tft(n_features, lookback=LOOKBACK, hidden_dim=64, n_heads=4,
              num_layers=1, dropout=0.1, seed=42):
    tf.random.set_seed(seed)
    inputs = tf.keras.Input(shape=(lookback, n_features))

    # Variable selection
    vsn = VariableSelectionNetwork(n_features, hidden_dim, dropout=dropout)
    selected = vsn(inputs)  # (B, T, hidden_dim)

    # LSTM temporal processing (locality-enhancing layer)
    lstm_out = tf.keras.layers.LSTM(hidden_dim, return_sequences=True)(selected)
    gated = GatedLinearUnit(hidden_dim)(lstm_out)
    temporal = tf.keras.layers.LayerNormalization()(selected + gated)

    # Self-attention blocks
    h = temporal
    for _ in range(num_layers):
        attn_out = tf.keras.layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=hidden_dim // n_heads,
            dropout=dropout,
        )(h, h)
        grn_out = GatedResidualNetwork(hidden_dim, hidden_dim, dropout=dropout)(attn_out)
        h = tf.keras.layers.LayerNormalization()(h + grn_out)

    # Last timestep -> output
    last = h[:, -1, :]
    out = GatedResidualNetwork(hidden_dim, hidden_dim, dropout=dropout)(last)
    out = tf.keras.layers.Dense(1, kernel_initializer='he_uniform')(out)

    return tf.keras.Model(inputs, out, name='TFT')


# ── Training wrapper ──────────────────────────────────────────────────────────

def train_seq_model(model, X_tr, y_tr, X_va, y_va, *,
                    epochs=100, batch_size=512, lr=1e-3,
                    patience=25, lr_patience=8, lr_factor=0.3,
                    desc="Model"):
    """
    Train a Keras sequence model. Returns dict compatible with save_seq_run.
    """
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')

    from src.helper import TQDMEpochBar

    t0 = time.perf_counter()
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=patience, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', patience=lr_patience, factor=lr_factor, min_lr=1e-6),
            TQDMEpochBar(total_epochs=epochs, desc=desc),
        ],
        verbose=0,
    )
    training_time = time.perf_counter() - t0

    return dict(
        model=model,
        history={
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
        },
        epochs=len(history.history['loss']),
        training_time=training_time,
    )


# ── Results persistence ───────────────────────────────────────────────────────

def save_seq_run(run_dir, *, results_by_fs, hw_coef, df_sorted):
    """Save metrics, model weights, predictions, and history for all feature sets."""
    from src.metrics import build_gain_table

    run_dir = Path(run_dir)
    weights_dir = run_dir / 'best_model_weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    diag_rows = []
    total_time = 0.0

    for fs_name, res in results_by_fs.items():
        y_pred = res['y_pred']
        y_true = res['y_true']
        test_idx = res['test_indices']

        y_hw, _, hw_sse = hw_predict_aligned(hw_coef, df_sorted, test_idx)

        met = metrics(y_true, y_pred)
        hw_met = metrics(y_true, y_hw)
        g = gain(met['SSE'], hw_sse) * 100

        hw_row = hw_met.copy()
        hw_row['Model'] = f'Analytic ({fs_name})'
        hw_row['Training_time'] = None
        hw_row['Gain_vs_Analytic'] = None
        summary_rows.append(hw_row)

        model_row = met.copy()
        model_row['Model'] = fs_name
        model_row['Training_time'] = f"{res['training_time']:.1f}s"
        model_row['Gain_vs_Analytic'] = f"{g:.2f}%"
        summary_rows.append(model_row)

        total_time += res['training_time']

        diag_rows.append(residual_diagnostics(y_true, y_hw, label=f'Analytic ({fs_name})'))
        diag_rows.append(residual_diagnostics(y_true, y_pred, label=fs_name))

        # Save Keras model
        res['model'].save(str(weights_dir / f'{fs_name}_model.keras'))
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
                 'R2', 'Training_time', 'Gain_vs_Analytic']
    summary = pd.DataFrame(summary_rows)
    for col in col_order:
        if col not in summary.columns:
            summary[col] = None
    summary = summary[col_order]
    summary.to_csv(run_dir / 'metrics_summary.csv', index=False)
    pd.DataFrame(diag_rows).to_csv(run_dir / 'residual_diagnostics.csv', index=False)

    # Gain table from first feature set
    first_fs = list(results_by_fs.keys())[0]
    res0 = results_by_fs[first_fs]
    _save_gain_table(run_dir, res0, hw_coef, df_sorted)

    print(f'\nSaved metrics_summary.csv')
    print(f'Saved residual_diagnostics.csv')
    print(f'Saved gain_table.csv')
    print(f'Saved {len(results_by_fs)} x 3 artifacts (model, predictions, history)')
    print(f'All results saved to {run_dir}')

    return summary


def _save_gain_table(run_dir, res, hw_coef, df_sorted):
    from src.metrics import build_gain_table

    test_idx = res['test_indices']
    rows = df_sorted.iloc[test_idx]
    y_true = res['y_true']
    y_pred = res['y_pred'].ravel()

    y_hw, _, _ = hw_predict_aligned(hw_coef, df_sorted, test_idx)
    y_hw = y_hw.ravel()

    test_df = pd.DataFrame({
        'ret': rows['spy_ret'].values,
        'T': rows['T'].values,
    })
    test_df['se_model'] = (y_true - y_pred) ** 2
    test_df['se_hw'] = (y_true - y_hw) ** 2

    test_df['ret_bin'] = pd.cut(test_df['ret'],
        bins=[-np.inf, -0.01, 0, 0.01, np.inf],
        labels=['<-1%', '-1% to 0%', '0% to 1%', '>1%'])
    test_df['T_bin'] = pd.cut(test_df['T'],
        bins=[0, 0.25, 0.5, 1.0, np.inf],
        labels=['0-3m', '3-6m', '6m-1yr', '>1yr'],
        include_lowest=True)

    long_df, pivot_df = build_gain_table(
        test_df, 'T_bin', 'ret_bin', 'se_model', 'se_hw')
    long_df.to_csv(Path(run_dir) / 'gain_table.csv', index=False)
