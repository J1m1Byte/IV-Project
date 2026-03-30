"""
Simple fully-connected ANN for IV prediction (TensorFlow/Keras).

Architecture from Cao, Chen & Hull (2019):
  3 hidden layers × 80 neurons, ReLU, linear output, MSE loss.

Usage
-----
    result = train_model(df_train, df_val, df_test,
                         features=['delta', 'T', 'spy_ret'],
                         target='d_iv')
"""

import time
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from src.helper import TQDMEpochBar


def train_model(df_train, df_val, df_test, features, target='d_iv',
                epochs=80, batch_size=4096, lr=1e-3, patience=25,
                lr_patience=8, lr_factor=0.3,
                hidden_layers=3, neurons=80, activation='relu',
                seed=42, desc="ANN"):
    """
    Train a fully-connected ANN on pre-split DataFrames.

    Parameters
    ----------
    df_train, df_val, df_test : pd.DataFrame
    features : list[str]   — column names used as input
    target   : str         — column name for the target
    epochs, batch_size, lr, patience — training config
    lr_patience, lr_factor           — ReduceLROnPlateau config
    hidden_layers, neurons, activation — architecture config
    seed     : int
    desc     : str — progress bar label

    Returns
    -------
    dict: model, scaler, y_test, y_pred, sse, mse, rmse,
          training_time, history
    """
    tf.random.set_seed(seed)

    X_train = df_train[features].values
    X_val   = df_val[features].values
    X_test  = df_test[features].values

    ytr = df_train[target].values.ravel()
    yva = df_val[target].values.ravel()
    yte = df_test[target].values.ravel()

    # --- scale ---
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xva = scaler.transform(X_val)
    Xte = scaler.transform(X_test)

    # --- build model ---
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(Xtr.shape[1],)))
    for _ in range(hidden_layers):
        model.add(tf.keras.layers.Dense(neurons, activation=activation))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")

    # --- train ---
    t0 = time.perf_counter()
    history = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", patience=lr_patience, factor=lr_factor, min_lr=1e-6),
            TQDMEpochBar(total_epochs=epochs, desc=desc),
        ],
        verbose=0,
    )

    training_time = time.perf_counter() - t0

    # --- evaluate ---
    y_pred = model.predict(Xte, batch_size=batch_size, verbose=0).ravel()
    residuals = yte - y_pred
    sse  = float(np.sum(residuals ** 2))
    mse  = sse / len(yte)
    rmse = float(np.sqrt(mse))

    print(f"\nTest:\nSSE = {sse:.4f}  RMSE = {rmse:.6f}  Time = {training_time:.1f}s")

    return dict(model=model, scaler=scaler, y_test=yte, y_pred=y_pred,
                sse=sse, mse=mse, rmse=rmse,
                training_time=training_time, history=history.history)
