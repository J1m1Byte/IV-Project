import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from pathlib import Path

from src.paths import OUTPUT, CLEAN_DATA_V2, FIG_OUTPUT
from src.benchmark import analytic_benchmark


def fig_3d(
    set_letter: str,
    model_number: str = '1',
    returns: tuple = (-0.0125, +0.0125),
    output_path: Path = None,
    show: bool = True,
):
    """
    Create a 2x2 3D surface plot comparing Hull-White analytic model with ANN-3F
    for a given data set (Range A-D).

    Parameters
    ----------
    set_letter : str
        Data set letter ('A', 'B', 'C', or 'D').
    model_number : str, default '1'
        Model version number.
    returns : tuple of float, default (-0.0125, +0.0125)
        Negative and positive SPY return scenarios for prediction.
    output_path : Path, optional
        Path to save the figure. If None, figure is not saved.
    show : bool, default True
        Whether to call plt.show().

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    """
    # ── Load data and model artifacts ──
    set_name = f'rand_{set_letter}'
    run_path = OUTPUT / f'{model_number}.{ord(set_letter) - ord("A")}-fc-rand-{set_letter}' / '01-run'

    df_train = pd.read_parquet(CLEAN_DATA_V2 / (set_name + '_train_v2.parquet'))
    df_val = pd.read_parquet(CLEAN_DATA_V2 / (set_name + '_val_v2.parquet'))
    df_test = pd.read_parquet(CLEAN_DATA_V2 / (set_name + '_test_v2.parquet'))

    scaler_3f = pd.read_pickle(run_path / 'train-history/ANN-3F_scaler.pkl')
    model_3f = tf.keras.models.load_model(run_path / 'train-history/ANN-3F.keras')

    # ── Fit Hull-White benchmark ──
    hw = analytic_benchmark(df_train, df_val, df_test, target='d_iv')
    a = hw['coef']['a']
    b = hw['coef']['b']
    c = hw['coef']['c']

    # ── Setup grids ──
    TRADING_YEAR = 252

    # Coarse grid (paper values)
    deltas_coarse = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    T_days_coarse = np.array([63, 126, 252, 378])
    T_years_coarse = T_days_coarse / TRADING_YEAR

    # Fine grid for smooth surface display
    N_fine = 60
    deltas_fine = np.linspace(0.1, 0.9, N_fine)
    T_days_fine = np.linspace(63, 378, N_fine)
    T_years_fine = T_days_fine / TRADING_YEAR

    # ── Predict and smooth ──
    def predict_and_smooth(ret_val):
        """Evaluate at coarse grid, bicubic-spline to fine grid."""
        ret_arr = np.full(deltas_coarse.size * T_days_coarse.size, ret_val)
        D_c, T_c = np.meshgrid(deltas_coarse, T_years_coarse, indexing='ij')
        D_flat = D_c.ravel()
        T_flat = T_c.ravel()

        # Hull-White (use fine grid directly)
        D_fw, T_fw = np.meshgrid(deltas_fine, T_years_fine, indexing='ij')
        sqrt_T = np.sqrt(T_fw)
        hw_fine = (a + b * D_fw + c * D_fw**2) * ret_val / sqrt_T * 1e4

        # ANN-3F: evaluate on coarse grid → spline
        X3 = scaler_3f.transform(np.column_stack([D_flat, T_flat, ret_arr]))
        z3 = model_3f.predict(X3, verbose=0).ravel().reshape(5, 4) * 1e4
        spl3 = RectBivariateSpline(deltas_coarse, T_days_coarse, z3, kx=3, ky=3)
        ann3_fine = spl3(deltas_fine, T_days_fine)

        return hw_fine, ann3_fine

    ret_neg, ret_pos = returns
    hw_neg, ann3_neg = predict_and_smooth(ret_neg)
    hw_pos, ann3_pos = predict_and_smooth(ret_pos)

    # ── Create figure ──
    D_mesh, T_mesh = np.meshgrid(deltas_fine, T_days_fine, indexing='ij')

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 9),
        subplot_kw={'projection': '3d'},
    )
    fig.suptitle(
        'Expected Change in Implied Volatility\n'
        f'Analytical and FC ANN 3-feature model | Range {set_letter}',
        fontsize=11,
    )

    panels = [
        (hw_neg, f'Analytic  | SPY Return = −{abs(ret_neg)*100:.2f}%', 'Blues_r'),
        (hw_pos, f'Analytic  | SPY Return = +{ret_pos*100:.2f}%', 'Reds'),
        (ann3_neg, f'3F ANN    | SPY Return = −{abs(ret_neg)*100:.2f}%', 'Blues_r'),
        (ann3_pos, f'3F ANN    | SPY Return = +{ret_pos*100:.2f}%', 'Reds'),
    ]

    for ax, (Z, title, cmap) in zip(axes.flat, panels):
        surf = ax.plot_surface(
            D_mesh, T_mesh, Z,
            cmap=cmap, edgecolor='none', alpha=0.9,
            rcount=60, ccount=60
        )
        ax.set_title(title, fontsize=10, pad=8)
        ax.set_xlabel('Delta', labelpad=6)
        ax.set_ylabel('T (days)', labelpad=6)
        ax.set_zlabel('Δ IV (bps)', labelpad=6)
        ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
        ax.set_yticks([63, 126, 252, 378])
        ax.tick_params(labelsize=7)
        ax.view_init(elev=25, azim=-50)

    plt.tight_layout()

    # ── Save and show ──
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig
