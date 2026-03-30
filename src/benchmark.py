"""
Hull-White Analytic Benchmark for Implied Volatility Prediction.

Implements Equation (2) from Cao, Chen & Hull (2019):

    E[Δσ_imp] = (ΔS/S) * (a + b·δ + c·δ²) / √T

where a, b, c are estimated via no-intercept linear regression
on the transformed features:
    x1 = ret / √T
    x2 = δ · ret / √T
    x3 = δ² · ret / √T

Usage
-----
    hw = analytic_benchmark(df_train, df_val, df_test, target='d_iv')
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def _build_hw_features(delta, T, ret):
    """Transform raw features into Hull-White regression inputs."""
    sqrt_T = np.sqrt(T)
    return np.column_stack([
        ret / sqrt_T,
        delta * ret / sqrt_T,
        delta ** 2 * ret / sqrt_T,
    ]).astype(np.float32)


def analytic_benchmark(df_train, df_val, df_test, target='d_iv'):
    """
    Fit Hull-White analytic benchmark on train+val, evaluate on test.

    Parameters
    ----------
    df_train, df_val, df_test : pd.DataFrame
        Must contain columns: delta, T, spy_ret, and <target>.
    target : str
        Name of the target column (default 'd_iv').

    Returns
    -------
    dict with keys: y_pred, coef, sse, mse, rmse
    """
    # Combine train + val for fitting
    df_fit = pd.concat([df_train, df_val], ignore_index=True)

    X_fit = _build_hw_features(
        df_fit['delta'].values, df_fit['T'].values, df_fit['spy_ret'].values,
    )
    y_fit = df_fit[target].values.ravel()

    X_te = _build_hw_features(
        df_test['delta'].values, df_test['T'].values, df_test['spy_ret'].values,
    )
    y_true = df_test[target].values.ravel()

    # Fit no-intercept linear regression
    hw = LinearRegression(fit_intercept=False)
    hw.fit(X_fit, y_fit)

    # Predict
    y_pred = hw.predict(X_te).reshape(-1, 1)

    # Metrics
    residuals = y_true - y_pred.ravel()
    sse = float(np.sum(residuals ** 2))
    n = len(y_true)
    mse = sse / n
    rmse = float(np.sqrt(mse))

    coef = dict(zip(["a", "b", "c"], hw.coef_.tolist()))

    print(f"Analytic Benchmark\nSSE = {sse:.4f}  RMSE = {rmse:.6f}")
    print(f"Coefficients: a = {coef['a']:.6f}, b = {coef['b']:.6f}, c = {coef['c']:.6f}")

    return dict(y_pred=y_pred, coef=coef, sse=sse, mse=mse, rmse=rmse)
