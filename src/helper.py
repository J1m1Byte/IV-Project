"""
Training helpers:
  - TQDMEpochBar  : clean epoch progress bar for Keras
  - make_run_dir  : create a numbered output directory  (output/<name>/run-01/)
  - save_results  : persist model, scaler, predictions, history, and metrics
"""

import os
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from src.paths import OUTPUT

try:
    from tqdm.notebook import tqdm
except Exception:
    from tqdm import tqdm


class TQDMEpochBar(tf.keras.callbacks.Callback):
    """
    Single-bar epoch progress for Keras training.

    Parameters
    ----------
    total_epochs : int
        Total number of epochs passed to model.fit.
    desc : str
        Label shown on the left of the bar (e.g. model name).
    """

    def __init__(self, total_epochs: int, desc: str = "Training"):
        super().__init__()
        self._total = total_epochs
        self._desc = desc
        self._bar = None
        self._epoch_start: float = 0.0

    # ------------------------------------------------------------------
    def on_train_begin(self, logs=None):
        self._bar = tqdm(
            total=self._total,
            desc=self._desc,
            unit="epoch",
            dynamic_ncols=True,
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| "
                "{n_fmt}/{total_fmt} epochs "
                "[{elapsed}<{remaining}, {rate_fmt}]  {postfix}"
            ),
        )

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        elapsed = time.perf_counter() - self._epoch_start

        postfix = {
            "loss": f"{logs.get('loss', float('nan')):.2e}",
            "val_loss": f"{logs.get('val_loss', float('nan')):.2e}",
            "s/ep": f"{elapsed:.1f}",
        }
        lr = logs.get("learning_rate") or logs.get("lr")
        if lr is not None:
            postfix["lr"] = f"{lr:.0e}"

        self._bar.set_postfix(postfix)
        self._bar.update(1)

    def on_train_end(self, logs=None):
        if self._bar is not None:
            self._bar.close()


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def _notebook_stem() -> str:
    """Get the filename stem of the calling notebook (or script)."""
    # 1. Set explicitly by run_pipeline.py via env var
    stem = os.environ.get("NOTEBOOK_STEM")
    if stem:
        return stem
    # 2. VS Code / Jupyter extension sets this variable
    try:
        from IPython import get_ipython
        return Path(get_ipython().user_ns['__vsc_ipynb_file__']).stem
    except Exception:
        pass
    # 3. JupyterLab session API
    try:
        import json, urllib.request
        from IPython import get_ipython
        kernel_id = get_ipython().config['IPKernelApp']['connection_file'].split('kernel-')[1].split('.')[0]
        sessions = json.loads(urllib.request.urlopen('http://localhost:8888/api/sessions').read())
        for s in sessions:
            if s['kernel']['id'] == kernel_id:
                return Path(s['notebook']['path']).stem
    except Exception:
        pass
    return "unknown"


def make_run_dir(name: str = None) -> Path:
    """
    Create and return output/<name>/run-01/, run-02/, … (auto-incremented).

    If name is omitted, auto-detects from the calling notebook filename.

    Example
    -------
        out = make_run_dir()          # auto: "1.0-ann" from 1.0-ann.ipynb
        out = make_run_dir("custom")  # explicit name
    """
    if name is None:
        name = _notebook_stem()
    base = OUTPUT / name
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(base.glob("*-run"))
    n = len(existing) + 1
    run_dir = base / f"{n:02d}-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run(run_dir, y_test, hw, models: dict) -> pd.DataFrame:
    """
    Save all results for a single experiment run.

    Parameters
    ----------
    run_dir : Path from make_run_dir()
    y_test  : array — true test targets (e.g. df_test['d_iv'].values)
    hw      : dict from analytic_benchmark (needs y_pred)
    models  : dict like {"ANN-3F": result_3f, "ANN-4F": result_4f}
              Each value is a dict from train_model.

    Files written
    -------------
    metrics_summary.csv         full metrics table + gains + training time
    residual_diagnostics.csv    per-model residual stats
    <name>.keras                Keras model weights
    <name>_scaler.pkl           fitted scaler
    <name>_predictions.npy      test-set predictions
    <name>_history.csv          training loss per epoch

    Returns
    -------
    metrics_summary DataFrame (for display in notebook)
    """
    from src.metrics import metrics, gain, residual_diagnostics

    run_dir = Path(run_dir)
    yt = np.asarray(y_test).ravel()

    # --- metrics summary ---
    rows = []

    # analytic row
    hw_met = metrics(yt, hw["y_pred"])
    hw_met["Model"] = "Analytic"
    hw_met["Training_time"] = None
    hw_met["Gain_vs_Analytic"] = None
    hw_met["Gain_Incremental"] = None
    rows.append(hw_met)

    prev_sse = hw_met["SSE"]
    total_time = 0.0
    first_model = True

    for name, result in models.items():
        met = metrics(yt, result["y_pred"])
        met["Model"] = name
        met["Training_time"] = f"{result['training_time']:.1f}s"
        met["Gain_vs_Analytic"] = f"{gain(met['SSE'], hw_met['SSE']) * 100:.2f}%"
        met["Gain_Incremental"] = None if first_model else f"{gain(met['SSE'], prev_sse) * 100:.2f}%"
        rows.append(met)
        prev_sse = met["SSE"]
        first_model = False
        total_time += result["training_time"]

        # per-model artifacts
        result["model"].save(str(run_dir / f"{name}.keras"))

        with open(run_dir / f"{name}_scaler.pkl", "wb") as f:
            pickle.dump(result["scaler"], f)

        np.save(run_dir / f"{name}_predictions.npy", result["y_pred"])

        pd.DataFrame({
            "epoch":    range(1, len(result["history"]["loss"]) + 1),
            "loss":     result["history"]["loss"],
            "val_loss": result["history"]["val_loss"],
        }).to_csv(run_dir / f"{name}_history.csv", index=False)

    # total training time row
    total_row = {col: None for col in rows[0]}
    total_row["Model"] = "Total"
    total_row["Training_time"] = f"{total_time:.1f}s"
    rows.append(total_row)

    col_order = ["Model", "SSE", "MSE", "RMSE", "MAE", "MeanError", "MedianAE",
                 "R2", "Training_time", "Gain_vs_Analytic", "Gain_Incremental"]
    summary = pd.DataFrame(rows)[col_order]
    summary.to_csv(run_dir / "metrics_summary.csv", index=False)

    # --- residual diagnostics ---
    diag_rows = [residual_diagnostics(yt, hw["y_pred"], label="Analytic")]
    for name, result in models.items():
        diag_rows.append(residual_diagnostics(yt, result["y_pred"], label=name))
    pd.DataFrame(diag_rows).to_csv(run_dir / "residual_diagnostics.csv", index=False)

    return summary
