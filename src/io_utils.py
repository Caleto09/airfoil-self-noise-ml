# %%
# src/io_utils.py
"""
Funciones de ingesta, validación de contrato de datos y registro de entorno (P1).
"""
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from src.config import (
    ALL_DATASET_COLS,
    DATA_RAW_PATH,
    FEATURE_COLS,
    HIGH_NOISE_QUANTILE,
    METADATA_PATH,
    PREDICTIONS_DIR,
    REQUIRED_PREDICTION_COLS,
    TARGET_COL,
    UCI_DATA_URL,
)


def load_or_download_airfoil_data() -> pd.DataFrame:
    """
    Carga el dataset Airfoil Self-Noise local o lo descarga de UCI si no existe.
    Verifica dimensiones (1503x6) y ausencia de valores nulos.
    """
    DATA_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not DATA_RAW_PATH.exists():
        print(f"[INFO] Descargando dataset desde UCI: {UCI_DATA_URL}")
        df = pd.read_csv(UCI_DATA_URL, sep=r"\s+", header=None, names=ALL_DATASET_COLS)
        df.to_csv(DATA_RAW_PATH, index=False)
        print(f"[OK] Dataset guardado en: {DATA_RAW_PATH}")
    else:
        df = pd.read_csv(DATA_RAW_PATH)

    # Validaciones de integridad del protocolo
    assert df.shape == (1503, 6), f"Dimensión inválida: se esperaba (1503, 6), se obtuvo {df.shape}"
    assert list(df.columns) == ALL_DATASET_COLS, f"Columnas incorrectas: {list(df.columns)}"
    assert df.isna().sum().sum() == 0, "Se detectaron valores faltantes en el dataset."

    return df


def build_fold_predictions_df(
    familia: str,
    modelo: str,
    repeticion: int,
    pliegue: int,
    semilla: int,
    sample_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    q90_threshold: float,
    tiempo_entrenamiento_s: float,
) -> pd.DataFrame:
    """
    Empaqueta las predicciones de un pliegue en el formato oficial de P1.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    residuo = y_pred_arr - y_true_arr  # Negativo = subestimación (el error más costoso en alto ruido)

    df_fold = pd.DataFrame({
        "familia": familia,
        "modelo": modelo,
        "repeticion": int(repeticion),
        "pliegue": int(pliegue),
        "semilla": int(semilla),
        "sample_id": np.asarray(sample_ids, dtype=int),
        "y_true_db": y_true_arr,
        "y_pred_db": y_pred_arr,
        "residuo_db": residuo,
        "es_cuantil_alto": y_true_arr >= q90_threshold,
        "tiempo_entrenamiento_s": float(tiempo_entrenamiento_s),
    })
    return df_fold


def validate_and_save_predictions(df_preds: pd.DataFrame, model_name: str) -> Path:
    """
    Valida que el DataFrame de predicciones de P3 o P4 cumpla con las convenciones
    de P1 y lo guarda en disco.
    """
    missing_cols = set(REQUIRED_PREDICTION_COLS) - set(df_preds.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas obligatorias en {model_name}: {missing_cols}")

    if df_preds[REQUIRED_PREDICTION_COLS].isna().any().any():
        raise ValueError(f"El archivo de predicciones de {model_name} contiene valores NaN.")

    valid_families = {"lineal", "no_lineal"}
    if not set(df_preds["familia"].unique()).issubset(valid_families):
        raise ValueError(f"Columna 'familia' debe ser 'lineal' o 'no_lineal'.")

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"predicciones_{model_name.lower().replace(' ', '_')}.csv"
    df_preds[REQUIRED_PREDICTION_COLS].to_csv(out_path, index=False)
    print(f"[OK] Predicciones validadas y guardadas: {out_path}")
    return out_path


def log_software_environment():
    """
    Registra versiones de software, sistema operativo y semillas para reproducibilidad.
    """
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    env_info = {
        "fecha_ejecucion": datetime.now().isoformat(),
        "python_version": sys.version,
        "plataforma": platform.platform(),
        "arquitectura": platform.machine(),
        "librerias": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
    }
    METADATA_PATH.write_text(json.dumps(env_info, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Metadatos de reproducibilidad guardados en: {METADATA_PATH}")
# %%
