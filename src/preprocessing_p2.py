"""
Módulo de preprocesamiento, particiones agrupadas y optimización (Responsabilidad P2).
Este archivo será importado por P1, P3 y P4.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import CONFIG_GROUP_COLS


def assign_configuration_groups(df: pd.DataFrame) -> np.ndarray:
    """
    Genera un ID único (grupo) para cada combinación de cuerda, velocidad y ángulo.
    Crucial para evitar fuga de información: las mismas configuraciones físicas 
    (medidas a distintas frecuencias) no deben separarse entre train y test.
    """
    return df.groupby(CONFIG_GROUP_COLS).ngroup().to_numpy()

def get_physical_preprocessor() -> ColumnTransformer:
    """
    Construye el pipeline de estandarización y transformaciones motivadas físicamente.
    Aplica log10 a predictores con rangos amplios y dependencias relativas.
    La estandarización aprende sus parámetros exclusivamente en train.
    """
    log_cols = ["frecuencia_hz", "longitud_cuerda_m", "velocidad_flujo_ms", "espesor_desplazamiento_m"]
    pass_cols = ["angulo_ataque_deg"]

    # Transformación logarítmica segura (evita log(0) clipando a 1e-9)
    log_transformer = FunctionTransformer(
        lambda x: np.log10(np.clip(x, 1e-9, None)),
        feature_names_out="one-to-one"
    )

    phys_preprocessor = ColumnTransformer(
        transformers=[
            ("log_trans", log_transformer, log_cols),
            ("pass", "passthrough", pass_cols),
        ]
    )

    # Se encadena con StandardScaler para que todas las variables tengan media 0 y varianza 1
    # Esto ayuda a mitigar la colinealidad detectada en el EDA.
    return phys_preprocessor

def run_nested_cv_search(
    pipeline, 
    param_grid: dict, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    groups_train: np.ndarray, 
    n_splits: int = 5
):
    """
    Utilidad compartida para que P3 y P4 busquen hiperparámetros sin implementarlo ellos.
    Usa GroupKFold internamente para respetar la organización de los datos.
    """
    inner_cv = GroupKFold(n_splits=n_splits)
    
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True
    )
    
    # Se pasa 'groups' a fit para que GroupKFold sepa cómo hacer los cortes internos
    search.fit(X_train, y_train, groups=groups_train)
    
    return search.best_estimator_, search.best_params_