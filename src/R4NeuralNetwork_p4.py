# %%
# src/nonlinear_models_p4.py
"""
Familia no lineal (P4): Perceptron multicapa (MLP) para predecir
el nivel de presion sonora (SPL, dB) del dataset Airfoil Self-Noise.

Contenido:
  1. Particiones agrupadas por configuracion fisica (cuerda, velocidad, angulo).

  2. Pipeline de la MLP:  log10 + estandarizacion de entradas
                          -> MLPRegressor
                          -> el objetivo (SPL) tambien se estandariza internamente.
  3. Busqueda de hiperparametros con CV interna, optimizando RMSE en dB.


"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from R0config import CONFIG_GROUP_COLS, FEATURE_COLS

# PARTICIONES AGRUPADAS
def make_group_ids(df: pd.DataFrame) -> np.ndarray:
    """
    Asigna un id entero a cada configuracion fisica unica
    (longitud_cuerda_m, velocidad_flujo_ms, angulo_ataque_deg).
    Todas las filas de una misma configuracion (distintas frecuencias)
    comparten id. En el dataset real hay 106 configuraciones.
    """
    return df.groupby(CONFIG_GROUP_COLS, sort=False).ngroup().to_numpy()


def grouped_kfold_splits(groups: np.ndarray, n_splits: int, seed: int) -> list:
    """
    K-fold agrupado y reproducible: se barajan los GRUPOS (no las filas) y se
    reparten entre los pliegues. Una configuracion completa queda en prueba o en
    entrenamiento, nunca dividida entre ambos.

    Devuelve una lista de (indices_entrenamiento, indices_prueba) POSICIONALES
    respecto al arreglo `groups` recibido.
    """
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    group_folds = np.array_split(unique_groups, n_splits)

    splits = []
    for test_groups in group_folds:
        test_mask = np.isin(groups, test_groups)
        splits.append((np.flatnonzero(~test_mask), np.flatnonzero(test_mask)))
    return splits



# 2. PIPELINE DE LA MLP


# Variables positivas de "escala" -> log10. Con SPL en decibeles (escala
# logaritmica) las leyes de potencia (p. ej. SPL ~ 50*log10(velocidad))
# se vuelven aproximadamente lineales en el logaritmo de la entrada.
LOG_COLS = [
    "frecuencia_hz",
    "longitud_cuerda_m",
    "velocidad_flujo_ms",
    "espesor_desplazamiento_m",
]
# El angulo contiene 0 grados (log10(0) no existe) y no es una magnitud de
# escala, asi que solo se estandariza.
LIN_COLS = ["angulo_ataque_deg"]

assert set(LOG_COLS + LIN_COLS) == set(FEATURE_COLS), "Columnas desalineadas con config.py"

MAX_ITER = 2000


class _QuietMLPRegressor(MLPRegressor):
    """
    MLPRegressor que no imprime ConvergenceWarning (con cientos de ajustes en
    paralelo inunda la consola). Si una red no converge NO se pierde la
    informacion: run_p4_mlp.py registra `n_iter_` y `convergio` por pliegue.
    """

    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            return super().fit(X, y)


def _build_preprocessor() -> ColumnTransformer:
    log_branch = Pipeline([
        ("log10", FunctionTransformer(np.log10, feature_names_out="one-to-one")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[
            ("log", log_branch, LOG_COLS),
            ("lin", StandardScaler(), LIN_COLS),
        ],
        verbose_feature_names_out=False,
    )


def build_mlp_model(seed: int) -> TransformedTargetRegressor:
    """
    MLP lista para entrenar. Predice en dB (el objetivo se estandariza
    internamente para entrenar y se des-estandariza al predecir).

    Solver 'lbfgs': para ~1,200 observaciones por pliegue suele converger
    mas rapido y mas estable que SGD/Adam, y no requiere ajustar tasa de
    aprendizaje ni tamano de lote.
    """
    mlp = Pipeline([
        ("prep", _build_preprocessor()),
        ("mlp", _QuietMLPRegressor(
            hidden_layer_sizes=(32,),
            activation="tanh",
            solver="lbfgs",
            alpha=1e-3,
            max_iter=MAX_ITER,
            random_state=seed,
        )),
    ])
    return TransformedTargetRegressor(regressor=mlp, transformer=StandardScaler())




# Prefijo 'regressor__mlp__' = TransformedTargetRegressor -> Pipeline -> MLP
PARAM_DISTRIBUTIONS = {
    "regressor__mlp__hidden_layer_sizes": [
        (16,), (32,), (64,), (16, 16), (32, 16), (32, 32), (64, 32),
    ],
    "regressor__mlp__activation": ["tanh", "relu"],
    "regressor__mlp__alpha": loguniform(1e-5, 1e0),  # penalizacion L2
}


def build_mlp_search(seed: int, inner_cv: list, n_iter: int = 20, n_jobs: int = -1):
    """
    Busqueda aleatoria de hiperparametros. Cada combinacion se evalua con la CV
    interna (`inner_cv`, lista de particiones) y se elige la de menor RMSE en dB.
    Despues re-entrena la mejor con todo el conjunto de entrenamiento (refit).
    """
    return RandomizedSearchCV(
        estimator=build_mlp_model(seed),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",  # RMSE en dB (metrica principal)
        cv=inner_cv,
        random_state=seed,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
# %%