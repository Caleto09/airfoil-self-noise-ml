# %%
# src/config.py
"""
Protocolo experimental, semillas y convenciones de datos (Responsabilidad P1).
Dataset: Airfoil Self-Noise (UCI / NASA RP-1218) - 1,503 observaciones.
"""
from pathlib import Path


# 1. RUTAS DEL PROYECTO

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_PATH = ROOT_DIR / "data" / "raw" / "airfoil_self_noise.csv"
PREDICTIONS_DIR = ROOT_DIR / "results" / "predictions"
TABLES_DIR = ROOT_DIR / "results" / "tables"
FIGURES_DIR = ROOT_DIR / "results" / "figures"
METADATA_PATH = ROOT_DIR / "results" / "reproducibility_env.json"

# URL oficial del repositorio UCI (Airfoil Self-Noise)
UCI_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"


# 2. SEMILLAS Y PROTOCOLO DE PARTICIÓN

MASTER_SEED = 42
CV_SEEDS = [42, 100, 2026]  # Para validación cruzada repetida
N_OUTER_SPLITS = 5          # Pliegues de evaluación externa

# Cuantil superior para evaluar condiciones de mayor emisión sonora (top 10% más ruidoso)
HIGH_NOISE_QUANTILE = 0.90


# 3. NOMBRES CANÓNICOS DE COLUMNAS

FEATURE_COLS = [
    "frecuencia_hz",                 # Frecuencia [Hz]
    "angulo_ataque_deg",             # Ángulo de ataque [grados]
    "longitud_cuerda_m",             # Longitud de cuerda [m]
    "velocidad_flujo_ms",            # Velocidad del flujo libre [m/s]
    "espesor_desplazamiento_m",      # Espesor de desplazamiento lado de succión [m]
]

# Columnas que definen una configuración física única (para agrupamiento en P2)
CONFIG_GROUP_COLS = [
    "longitud_cuerda_m",
    "velocidad_flujo_ms",
    "angulo_ataque_deg",
]

TARGET_COL = "spl_db"                # Nivel de presión sonora escalado [dB]

ALL_DATASET_COLS = FEATURE_COLS + [TARGET_COL]


# 4. CONTRATO DE INTERFAZ: FORMATO DE PREDICCIONES POR PLIEGUE (P3 y P4)

REQUIRED_PREDICTION_COLS = [
    "familia",          # 'lineal' o 'no_lineal'
    "modelo",           # Ej: 'Ridge_Polinomial', 'RandomForest', 'GradientBoosting'
    "repeticion",       # Entero (0, 1, 2...)
    "pliegue",          # Entero (0 a N_OUTER_SPLITS - 1)
    "semilla",          # Semilla utilizada en esa partición
    "sample_id",        # Índice original de la observación (0 a 1502)
    "y_true_db",        # Valor real de SPL en dB
    "y_pred_db",        # Predicción del modelo en dB
    "residuo_db",       # y_pred_db - y_true_db (positivo = sobreestimación, negativo = subestimación)
    "es_cuantil_alto",  # Booleano: True si y_true_db >= P90 del conjunto
    "tiempo_entrenamiento_s",  # Tiempo de ajuste en segundos
]


# 5. MÉTRICAS OFICIALES DEL PROYECTO

PRIMARY_METRIC = "rmse_db"
COMPLEMENTARY_METRICS = [
    "mae_db",
    "r2",
    "sesgo_db",
    "pendiente"
]

