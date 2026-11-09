# run_experiment.py
"""
Orquestador principal del experimento (Responsabilidad P1).
- Verifica que el código corra desde cero.
- Ejecuta la validación cruzada repetida sin fuga de información (Pipeline).
- Calcula métricas globales y en el cuantil superior (P90).
- Exporta los archivos oficiales de resultados experimentales.
"""
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    PolynomialFeatures,
    StandardScaler,
)
from src.config import (
    CV_SEEDS,
    FEATURE_COLS,
    HIGH_NOISE_QUANTILE,
    N_OUTER_SPLITS,
    TABLES_DIR,
    TARGET_COL,
)
from src.io_utils import (
    build_fold_predictions_df,
    load_or_download_airfoil_data,
    log_software_environment,
    validate_and_save_predictions,
)


# ==============================================================================
# 1. CÁLCULO DE MÉTRICAS
# ==============================================================================
def compute_fold_metrics(df_fold: pd.DataFrame) -> dict:
    """
    Calcula la métrica principal (RMSE en dB) y las complementarias
    tanto de forma global como en el 10% más ruidoso (cuantil >= 0.90).
    """
    y_true = df_fold["y_true_db"].to_numpy()
    y_pred = df_fold["y_pred_db"].to_numpy()
    residuo = df_fold["residuo_db"].to_numpy()

    # Métricas globales
    rmse_global = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae_global = float(mean_absolute_error(y_true, y_pred))
    r2_global = float(r2_score(y_true, y_pred))
    sesgo_global = float(np.mean(residuo))
    error_max = float(np.max(np.abs(residuo)))

    # Métricas en el cuantil superior (condiciones de mayor emisión sonora)
    mask_q90 = df_fold["es_cuantil_alto"].to_numpy()
    if np.any(mask_q90):
        y_true_q90 = y_true[mask_q90]
        y_pred_q90 = y_pred[mask_q90]
        res_q90 = residuo[mask_q90]
        rmse_q90 = float(np.sqrt(mean_squared_error(y_true_q90, y_pred_q90)))
        mae_q90 = float(mean_absolute_error(y_true_q90, y_pred_q90))
        sesgo_q90 = float(np.mean(res_q90))
    else:
        rmse_q90, mae_q90, sesgo_q90 = np.nan, np.nan, np.nan

    return {
        "familia": df_fold["familia"].iloc[0],
        "modelo": df_fold["modelo"].iloc[0],
        "repeticion": int(df_fold["repeticion"].iloc[0]),
        "pliegue": int(df_fold["pliegue"].iloc[0]),
        "semilla": int(df_fold["semilla"].iloc[0]),
        "rmse_db": rmse_global,
        "mae_db": mae_global,
        "r2": r2_global,
        "sesgo_db": sesgo_global,
        "error_max_db": error_max,
        "rmse_q90_db": rmse_q90,
        "mae_q90_db": mae_q90,
        "sesgo_q90_db": sesgo_q90,
        "tiempo_entrenamiento_s": float(df_fold["tiempo_entrenamiento_s"].iloc[0]),
    }


# ==============================================================================
# 2. DEFINICIÓN DE PIPELINES (Fase 0 / Reemplazable por módulos de P2, P3 y P4)
# ==============================================================================
def get_models_catalog(seed: int) -> dict:
    """
    Retorna los pipelines de cada familia asegurando que las transformaciones
    y la estandarización se ajusten exclusivamente con datos de entrenamiento.
    """
    # Transformación física para familia lineal: log10 en frecuencia y espesor (con offset seguro)
    log_cols = ["frecuencia_hz", "longitud_cuerda_m", "velocidad_flujo_ms", "espesor_desplazamiento_m"]
    pass_cols = ["angulo_ataque_deg"]

    phys_preprocessor = ColumnTransformer(
        transformers=[
            ("log_trans", FunctionTransformer(lambda x: np.log10(np.clip(x, 1e-9, None))), log_cols),
            ("pass", "passthrough", pass_cols),
        ]
    )

    linear_pipeline = Pipeline([
        ("phys_features", phys_preprocessor),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1.0, random_state=seed)),
    ])

    nonlinear_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1
        )),
    ])

    return {
        "Ridge_Polinomial_Log": ("lineal", linear_pipeline),
        "Random_Forest": ("no_lineal", nonlinear_pipeline),
    }


# ==============================================================================
# 3. EJECUCIÓN DEL DISEÑO EXPERIMENTAL
# ==============================================================================
def run_full_experiment():
    print("=" * 70)
    print("INICIANDO EJECUCIÓN REPRODUCIBLE - AIRFOIL SELF-NOISE (P1)")
    print("=" * 70)

    log_software_environment()
    df = load_or_download_airfoil_data()

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Umbral global del cuantil 90 (condiciones más ruidosas)
    q90_threshold = float(np.quantile(y, HIGH_NOISE_QUANTILE))
    print(f"[INFO] Umbral P90 de emisión sonora: {q90_threshold:.2f} dB")

    all_predictions = []
    all_fold_metrics = []

    # Esquema de Validación Cruzada Repetida
    for rep_idx, seed in enumerate(CV_SEEDS):
        kf = KFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=seed)
        models_dict = get_models_catalog(seed=seed)

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            for model_name, (familia, pipeline) in models_dict.items():
                t0 = time.perf_counter()
                pipeline.fit(X_train, y_train)
                elapsed = time.perf_counter() - t0

                y_pred = pipeline.predict(X_test)

                df_fold_pred = build_fold_predictions_df(
                    familia=familia,
                    modelo=model_name,
                    repeticion=rep_idx,
                    pliegue=fold_idx,
                    semilla=seed,
                    sample_ids=df.index[test_idx].to_numpy(),
                    y_true=y_test.to_numpy(),
                    y_pred=y_pred,
                    q90_threshold=q90_threshold,
                    tiempo_entrenamiento_s=elapsed,
                )

                all_predictions.append(df_fold_pred)
                all_fold_metrics.append(compute_fold_metrics(df_fold_pred))

    # Consolidación y validación de predicciones por modelo
    df_all_preds = pd.concat(all_predictions, ignore_index=True)
    for model_name in df_all_preds["modelo"].unique():
        sub_df = df_all_preds[df_all_preds["modelo"] == model_name]
        validate_and_save_predictions(sub_df, model_name)

    # Exportación de archivos finales de resultados experimentales
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df_metrics = pd.DataFrame(all_fold_metrics)

    folds_csv_path = TABLES_DIR / "resultados_metricas_por_pliegue.csv"
    df_metrics.to_csv(folds_csv_path, index=False)

    # Tabla resumen (Media y Desviación Estándar entre todos los pliegues y repeticiones)
    metric_cols = [
        "rmse_db", "mae_db", "r2", "sesgo_db",
        "error_max_db", "rmse_q90_db", "mae_q90_db", "sesgo_q90_db",
        "tiempo_entrenamiento_s"
    ]
    summary_df = (
        df_metrics.groupby(["familia", "modelo"])[metric_cols]
        .agg(["mean", "std"])
        .round(4)
    )
    summary_csv_path = TABLES_DIR / "resultados_resumen_final.csv"
    summary_df.to_csv(summary_csv_path)

    print("\n" + "=" * 70)
    print("RESUMEN EXPERIMENTAL PRELIMINAR (MEDIA ± STD EN DB)")
    print("=" * 70)
    for (familia, modelo), row in summary_df.iterrows():
        rmse_m, rmse_s = row[("rmse_db", "mean")], row[("rmse_db", "std")]
        q90_m, q90_s = row[("rmse_q90_db", "mean")], row[("rmse_q90_db", "std")]
        r2_m = row[("r2", "mean")]
        print(
            f"- [{familia.upper()}] {modelo:22s} | "
            f"RMSE Global: {rmse_m:.3f} ± {rmse_s:.3f} dB | "
            f"RMSE P90 (Alto Ruido): {q90_m:.3f} ± {q90_s:.3f} dB | "
            f"R²: {r2_m:.4f}"
        )
    print("=" * 70)
    print(f"[OK] Archivo por pliegue exportado en: {folds_csv_path}")
    print(f"[OK] Archivo resumen exportado en:     {summary_csv_path}")


if __name__ == "__main__":
    run_full_experiment()