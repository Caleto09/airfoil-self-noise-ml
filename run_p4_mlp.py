# run_p4_mlp.py
"""
Ejecucion del experimento de la MLP (Responsabilidad P4) con el mismo protocolo
y el mismo contrato de salida que run_experiment.py de P1:
  - 3 repeticiones (CV_SEEDS) x 5 pliegues externos.
  - Predicciones por pliegue en el formato oficial (11 columnas).
  - Metricas por pliegue con compute_fold_metrics de P1.

Uso (desde la raiz del proyecto):
    python run_p4_mlp.py                       # protocolo completo, particion agrupada
    python run_p4_mlp.py --quick               # prueba rapida (1 semilla, busqueda corta)
    python run_p4_mlp.py --particion simple    # KFold simple (como run_experiment.py de P1)
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from run_experiment import compute_fold_metrics
from src.config import (
    CV_SEEDS,
    FEATURE_COLS,
    HIGH_NOISE_QUANTILE,
    N_INNER_SPLITS,
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
from src.nonlinear_models_p4 import (
    MAX_ITER,
    build_mlp_search,
    grouped_kfold_splits,
    make_group_ids,
)


def run_mlp_experiment(partition: str = "agrupada", quick: bool = False,
                       n_iter: int = 20, n_jobs: int = -1) -> None:
    seeds = CV_SEEDS[:1] if quick else CV_SEEDS
    n_iter = min(n_iter, 6) if quick else n_iter
    model_name = "MLP" if partition == "agrupada" else "MLP_KFold_simple"

    print("=" * 70)
    print(f"MLP (P4) | particion: {partition} | semillas: {seeds} | n_iter: {n_iter}")
    print("=" * 70)

    log_software_environment()
    df = load_or_download_airfoil_data()
    X, y = df[FEATURE_COLS], df[TARGET_COL]
    groups = make_group_ids(df)
    print(f"[INFO] Configuraciones fisicas unicas: {len(np.unique(groups))}")

    q90_threshold = float(np.quantile(y, HIGH_NOISE_QUANTILE))
    print(f"[INFO] Umbral P90: {q90_threshold:.2f} dB")

    all_predictions, all_metrics, all_params = [], [], []

    for rep_idx, seed in enumerate(seeds):
        if partition == "agrupada":
            outer_splits = grouped_kfold_splits(groups, N_OUTER_SPLITS, seed)
        else:
            outer_splits = list(KFold(N_OUTER_SPLITS, shuffle=True, random_state=seed).split(X))

        for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # CV interna: SOLO con datos de entrenamiento (nunca ve el pliegue de prueba).
            if partition == "agrupada":
                inner_cv = grouped_kfold_splits(groups[train_idx], N_INNER_SPLITS, seed)
            else:
                inner_cv = list(KFold(N_INNER_SPLITS, shuffle=True, random_state=seed).split(X_train))

            search = build_mlp_search(seed, inner_cv, n_iter=n_iter, n_jobs=n_jobs)

            t0 = time.perf_counter()
            search.fit(X_train, y_train)          # busqueda + re-entrenamiento final
            elapsed = time.perf_counter() - t0    # incluye la busqueda de hiperparametros

            y_pred = search.predict(X_test)

            df_fold = build_fold_predictions_df(
                familia="no_lineal",
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
            m = compute_fold_metrics(df_fold)
            all_predictions.append(df_fold)
            all_metrics.append(m)

            best_mlp = search.best_estimator_.regressor_.named_steps["mlp"]
            all_params.append({
                "repeticion": rep_idx, "pliegue": fold_idx, "semilla": seed,
                "rmse_cv_interna_db": -search.best_score_,
                "capas": str(search.best_params_["regressor__mlp__hidden_layer_sizes"]),
                "activacion": search.best_params_["regressor__mlp__activation"],
                "alpha": search.best_params_["regressor__mlp__alpha"],
                "iteraciones": int(best_mlp.n_iter_),
                "convergio": bool(best_mlp.n_iter_ < MAX_ITER),
            })
            print(f"[rep {rep_idx} | pliegue {fold_idx}] RMSE={m['rmse_db']:.3f} dB | "
                  f"RMSE P90={m['rmse_q90_db']:.3f} dB | {elapsed:.0f}s | "
                  f"{all_params[-1]['capas']} {all_params[-1]['activacion']} "
                  f"alpha={all_params[-1]['alpha']:.1e}")

    # ---- Exportacion --------------------------------------------------------
    df_all = pd.concat(all_predictions, ignore_index=True)
    validate_and_save_predictions(df_all, model_name)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tag = model_name.lower()
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(TABLES_DIR / f"p4_{tag}_metricas_por_pliegue.csv", index=False)
    pd.DataFrame(all_params).to_csv(TABLES_DIR / f"p4_{tag}_hiperparametros.csv", index=False)

    print("\n" + "=" * 70)
    print(f"RESUMEN {model_name} (media ± std entre pliegues y repeticiones)")
    print("=" * 70)
    for col in ["rmse_db", "mae_db", "r2", "sesgo_db", "rmse_q90_db", "sesgo_q90_db",
                "tiempo_entrenamiento_s"]:
        print(f"{col:24s} {df_metrics[col].mean():9.4f} ± {df_metrics[col].std():.4f}")
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--particion", choices=["agrupada", "simple"], default="agrupada")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n_iter", type=int, default=20)
    ap.add_argument("--n_jobs", type=int, default=-1)
    a = ap.parse_args()
    run_mlp_experiment(a.particion, a.quick, a.n_iter, a.n_jobs)
