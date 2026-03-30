import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


# 1. Core scalar metrics

def metrics(y_true, y_pred):
    """
    Compute core error metrics between true and predicted values.

    Returns dict with: SSE, MSE, RMSE, MAE, MeanError (bias), MedianAE, R2
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    residuals = yt - yp
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    n = len(yt)

    return dict(
        SSE=ss_res,
        MSE=ss_res / n,
        RMSE=float(np.sqrt(ss_res / n)),
        MAE=float(mean_absolute_error(yt, yp)),
        MeanError=float(np.mean(residuals)),          # bias: 0 = unbiased
        MedianAE=float(np.median(np.abs(residuals))),  # robust to outliers
        R2=1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
    )


#  2. Gain computation 

def gain(sse_model, sse_baseline):
    """
    Out-of-sample gain: 1 - SSE_model / SSE_baseline.

    Positive = model beats baseline. Multiply by 100 for percentage.
    """
    if sse_baseline == 0:
        return np.nan
    return 1.0 - sse_model / sse_baseline


#  3. Model comparison summary 

def compare_models(results, baseline_name="Analytic"):
    """
    Build a comparison DataFrame from a dict of {name: metrics_dict}.

    Parameters
    ----------
    results : dict
        e.g. {"Analytic": hw_met, "ANN-3F": met3, "ANN-4F": met4, "ANN-5F": met5}
        Each value is a dict returned by metrics().
    baseline_name : str
        Which entry to use as the SSE baseline for gain calculations.

    Returns
    -------
    pd.DataFrame with one row per model, all metrics + Gain_vs_Baseline
    """
    rows = []
    baseline_sse = results[baseline_name]["SSE"]

    for name, met in results.items():
        row = {"Model": name, **met}
        if name == baseline_name:
            row["Gain_vs_Baseline"] = None
        else:
            row["Gain_vs_Baseline"] = gain(met["SSE"], baseline_sse) * 100
        rows.append(row)

    df = pd.DataFrame(rows)

    # pairwise incremental gain: each model vs the one above it
    df["Gain_Incremental"] = None
    for i in range(1, len(df)):
        prev_sse = df.iloc[i - 1]["SSE"]
        curr_sse = df.iloc[i]["SSE"]
        df.loc[df.index[i], "Gain_Incremental"] = gain(curr_sse, prev_sse) * 100

    return df


#  4. Gain table (pivot by bins) 

def build_gain_table(df, row_col, col_col, se_model, se_base):
    """
    Build a gain-% pivot table over categorical bins.

    Parameters
    ----------
    df       : DataFrame with categorical bin columns and SE columns.
    row_col  : column name for row categories (e.g. "T_bin", "vix_bin").
    col_col  : column name for column categories (e.g. "ret_bin").
    se_model : column with squared errors of the model.
    se_base  : column with squared errors of the baseline.

    Returns
    -------
    long_df  : tidy DataFrame with row_col, col_col, gain%, n
    pivot_df : pivot table for display
    """
    rows = []
    row_cats = df[row_col].cat.categories.tolist() + ["All"]
    col_cats = df[col_col].cat.categories.tolist() + ["All"]

    for rv in row_cats:
        for cv in col_cats:
            sub = df
            if rv != "All":
                sub = sub[sub[row_col] == rv]
            if cv != "All":
                sub = sub[sub[col_col] == cv]

            n = len(sub)
            base_sum = sub[se_base].sum()

            if n == 0 or base_sum == 0:
                g = np.nan
            else:
                g = (1.0 - sub[se_model].sum() / base_sum) * 100

            rows.append({row_col: rv, col_col: cv, "gain%": round(g, 2), "n": n})

    long_df = pd.DataFrame(rows)
    pivot_df = long_df.pivot(index=row_col, columns=col_col, values="gain%")
    return long_df, pivot_df


#  5. Residual diagnostics 

def residual_diagnostics(y_true, y_pred, label="model"):
    """
    Compute diagnostic stats on residuals for a single model.

    Returns a dict with: mean, std, skew, kurtosis, pct_within_1std,
    and quantiles [5%, 25%, 50%, 75%, 95%].
    """
    residuals = np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()
    std = float(np.std(residuals))

    return {
        "model": label,
        "mean": float(np.mean(residuals)),
        "std": std,
        "skew": float(pd.Series(residuals).skew()),
        "kurtosis": float(pd.Series(residuals).kurtosis()),  # excess kurtosis
        "pct_within_1std": float(np.mean(np.abs(residuals) <= std) * 100),
        "q05": float(np.percentile(residuals, 5)),
        "q25": float(np.percentile(residuals, 25)),
        "q50": float(np.percentile(residuals, 50)),
        "q75": float(np.percentile(residuals, 75)),
        "q95": float(np.percentile(residuals, 95)),
    }


#  6. Bin builder helper 

def build_test_bins(y_true, y_preds, raw_ret, raw_T, raw_vix, idx_te):
    """
    Build a binned test DataFrame ready for gain tables.

    Parameters
    ----------
    y_true  : full target array (indexed by idx_te)
    y_preds : dict of {"model_name": y_pred_array} — each is test-length
    raw_ret, raw_T, raw_vix : full arrays for return, maturity, VIX
    idx_te  : test indices

    Returns
    -------
    pd.DataFrame with ret_bin, T_bin, vix_bin, and se_<name> columns
    """
    yt = y_true[idx_te].ravel()

    test = pd.DataFrame({
        "ret": raw_ret[idx_te],
        "T":   raw_T[idx_te],
        "vix": raw_vix[idx_te],
    })

    for name, yp in y_preds.items():
        col = f"se_{name}"
        test[col] = (yt - np.asarray(yp).ravel()) ** 2

    # return bins (paper-consistent)
    test["ret_bin"] = pd.cut(test["ret"],
        bins=[-np.inf, -0.01, 0, 0.01, np.inf],
        labels=["<-1%", "-1% to 0%", "0% to 1%", ">1%"])

    # maturity bins
    test["T_bin"] = pd.cut(test["T"],
        bins=[0, 0.25, 0.5, 1.0, np.inf],
        labels=["0-3m", "3-6m", "6m-1yr", ">1yr"],
        include_lowest=True)

    # VIX bins (auto-detect scale)
    vix_med = test["vix"].median()
    if vix_med < 1:
        test["vix_bin"] = pd.cut(test["vix"],
            bins=[-np.inf, 0.13, 0.19, np.inf],
            labels=["≤13%", "13-19%", "≥19%"])
    else:
        test["vix_bin"] = pd.cut(test["vix"],
            bins=[-np.inf, 13, 19, np.inf],
            labels=["≤13%", "13-19%", "≥19%"])

    return test
