# %%
# src/__init__.py
"""
Punto de entrada del proyecto Airfoil Self-Noise.

Reúne las implementaciones existentes de los módulos del proyecto:
    - Random Forest: R3RandomForest_p3.TunedRandomForest
    - MLP:           R4NeuralNetwork_p4.build_mlp_model / build_mlp_search
    - Métricas:      R5Metrics_p5.compute_metrics / compute_metrics_by_fold
    - Figuras:       R5Metrics_p5.plot_rmse_by_model / plot_pred_vs_real

El bloque main() carga los datos, ajusta ambos modelos sobre una partición
agrupada, calcula las métricas y guarda predicciones, tablas y figuras.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from R0config import (
    CV_SEEDS,
    FEATURE_COLS,
    FIGURES_DIR,
    HIGH_NOISE_QUANTILE,
    MASTER_SEED,
    N_OUTER_SPLITS,
    TABLES_DIR,
    TARGET_COL,
)
from R0io_utils import (
    build_fold_predictions_df,
    load_or_download_airfoil_data,
    validate_and_save_predictions,
)
from R3RandomForest_p3 import TunedRandomForest
from R4NeuralNetwork_p4 import (
    build_mlp_model,
    build_mlp_search,
    grouped_kfold_splits,
    make_group_ids,
)
from R5Metrics_p5 import (
    METRIC_COLS,
    compute_metrics,
    compute_metrics_by_fold,
    plot_pred_vs_real,
    plot_rmse_by_model,
)

MODEL_NAMES = ("RandomForest", "MLP")


def build_model(name: str, seed: int = MASTER_SEED):
    if name == "RandomForest":
        return TunedRandomForest(random_state=seed)
    if name == "MLP":
        return build_mlp_model(seed)
    raise ValueError(f"Modelo desconocido: {name!r}. Opciones válidas: {list(MODEL_NAMES)}.")


def build_models(seed: int = MASTER_SEED) -> dict:
    return {name: build_model(name, seed) for name in MODEL_NAMES}


def main(seed: int = MASTER_SEED) -> pd.DataFrame:
    """
    - Carga el dataset.
    - Separa train/test con una partición agrupada por configuración física
      (evita fuga de información entre frecuencias de una misma configuración).
    - Ajusta Random Forest y MLP.
    - Calcula RMSE, MAE, R² y pendiente de calibración.
    - Guarda predicciones, tabla de métricas y figuras en results/.
    """
    print("AIRFOIL SELF-NOISE | Modelos + metricas (P5)")

    df = load_or_download_airfoil_data()
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    groups = make_group_ids(df)
    train_idx, test_idx = grouped_kfold_splits(groups, N_OUTER_SPLITS, seed)[0]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(
        f"Semilla: {seed} | "
        f"Train: {len(train_idx)} filas ({len(np.unique(groups[train_idx]))} grupos) | "
        f"Test: {len(test_idx)} filas ({len(np.unique(groups[test_idx]))} grupos)"
    )

    q90_threshold = float(np.quantile(y, HIGH_NOISE_QUANTILE))
    print(f"Umbral P90: {q90_threshold:.2f} dB")

    modelos = build_models(seed=seed)
    predicciones = []

    for nombre, modelo in modelos.items():
        print(f"\n Ajustando {nombre}")
        t0 = time.perf_counter()
        modelo.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        y_pred = modelo.predict(X_test)
        print(f"{nombre} ajustado en {elapsed:.1f} s")

        predicciones.append(build_fold_predictions_df(
            familia="no_lineal",
            modelo=nombre,
            repeticion=0,
            pliegue=0,
            semilla=seed,
            sample_ids=df.index[test_idx].to_numpy(),
            y_true=y_test.to_numpy(),
            y_pred=y_pred,
            q90_threshold=q90_threshold,
            tiempo_entrenamiento_s=elapsed,
        ))

    df_preds = pd.concat(predicciones, ignore_index=True)
    df_metricas = compute_metrics_by_fold(df_preds)

    columnas = ["modelo", "rmse_db", "mae_db", "r2", "pendiente"]
    print("METRICAS EN EL CONJUNTO DE PRUEBA")
    print(df_metricas[columnas].round(4).to_string(index=False))

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ruta_metricas = TABLES_DIR / "metricas_inicializador.csv"
    df_metricas.to_csv(ruta_metricas, index=False)
    print(f"\nMetricas guardadas en: {ruta_metricas}")

    for nombre in modelos:
        validate_and_save_predictions(df_preds[df_preds["modelo"] == nombre], nombre)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_rmse_by_model(df_metricas, save_path=FIGURES_DIR / "rmse_modelos.png")
    for nombre in modelos:
        plot_pred_vs_real(
            df_preds,
            modelo=nombre,
            save_path=FIGURES_DIR / f"pred_vs_real_{nombre.lower()}.png",
        )

    print("\nProceso terminado.")
    return df_metricas


__all__ = [
    "CV_SEEDS",
    "FEATURE_COLS",
    "FIGURES_DIR",
    "MASTER_SEED",
    "METRIC_COLS",
    "MODEL_NAMES",
    "TARGET_COL",
    "TunedRandomForest",
    "build_mlp_model",
    "build_mlp_search",
    "build_model",
    "build_models",
    "compute_metrics",
    "compute_metrics_by_fold",
    "load_or_download_airfoil_data",
    "main",
    "plot_pred_vs_real",
    "plot_rmse_by_model",
]


if __name__ == "__main__":
    main()

