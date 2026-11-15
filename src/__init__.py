"""
Punto de entrada del proyecto Airfoil Self-Noise.

Reúne las implementaciones existentes de los módulos del proyecto:
    - Random Forest: R3RandomForest_p3.TunedRandomForest
    - MLP:           R4NeuralNetwork_p4.build_mlp_search 
    - Métricas:      R5Metrics_p5.compute_metrics / compute_metrics_by_fold
    - Figuras:       R5Metrics_p5.plot_rmse_by_model / plot_pred_vs_real

El bloque main() carga los datos, ajusta ambos modelos sobre una partición
agrupada, calcula las métricas y guarda predicciones, tablas y figuras.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, QuantileTransformer, MaxAbsScaler

from R0config import (
    CV_SEEDS,
    FEATURE_COLS,
    FIGURES_DIR,
    HIGH_NOISE_QUANTILE,
    MASTER_SEED,
    N_INNER_SPLITS,
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


def build_model(name: str, seed: int = MASTER_SEED, inner_cv=None):
    """
    Crea una instancia nueva del modelo ya implementado en el proyecto.

    El MLP se construye con `build_mlp_search`, que busca hiperparámetros
    con la CV interna `inner_cv`y reentrena la mejor combinación.
    """
    if name == "RandomForest":
        return TunedRandomForest(random_state=seed)
    if name == "MLP":
        if inner_cv is None:
            raise ValueError(
                "El MLP se ajusta con build_mlp_search y requiere inner_cv "
                "(particiones agrupadas dentro del train)."
            )
        return build_mlp_search(seed, inner_cv)
    raise ValueError(f"Modelo desconocido: {name!r}. Opciones válidas: {list(MODEL_NAMES)}.")


def build_models(seed: int = MASTER_SEED, inner_cv=None) -> dict:
    """Crea una instancia nueva de cada modelo del proyecto."""
    return {name: build_model(name, seed, inner_cv) for name in MODEL_NAMES}


def main(seed: int = MASTER_SEED) -> pd.DataFrame:
    """
    - Carga el dataset.
    - Separa train/test con una partición agrupada por configuración física
      (evita fuga de información entre frecuencias de una misma configuración).
    - Ajusta Random Forest (búsqueda interna de hiperparámetros) y MLP
      (búsqueda aleatoria con CV interna).
    - Calcula RMSE, MAE, R² , MSE y pendiente de calibración.
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

    # CV interna para el MLP: pliegues agrupados DENTRO del train, en índices
    # posicionales relativos a X_train (así los usa RandomizedSearchCV).
    inner_cv = grouped_kfold_splits(groups[train_idx], N_INNER_SPLITS, seed)
    print(f"CV interna del MLP: {len(inner_cv)} pliegues agrupados")

    modelos = build_models(seed=seed, inner_cv=inner_cv)
    predicciones = []

    for nombre, modelo in modelos.items():
        print(f"\n Ajustando {nombre}")
        t0 = time.perf_counter()
        modelo.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        y_pred = modelo.predict(X_test)
        print(f"{nombre} ajustado en {elapsed:.1f} s")
        if getattr(modelo, "best_params_", None) is not None:
            print(f"  Mejores hiperparámetros: {modelo.best_params_}")

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

    columnas = ["modelo", "rmse_db", "mae_db", "r2", "pendiente",'mse']
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

