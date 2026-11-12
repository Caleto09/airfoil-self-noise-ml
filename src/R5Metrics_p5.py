"""
Métricas oficiales por pliegue:
    - RMSE [dB]
    - MAE [dB]
    - R²
    - Pendiente de calibración (ideal = 1.0)

Figuras:
    - RMSE por modelo (media ± desviación entre pliegues).
    - Valores predichos vs reales, con recta ideal 1:1 y ajuste de calibración.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

# Columnas que devuelve compute_metrics (mismo orden en todas las tablas).
METRIC_COLS = ["rmse_db", "mae_db", "r2", "pendiente"]

# Columnas de identificación que se conservan al resumir predicciones por pliegue.
_PREDICTION_ID_COLS = ["familia", "modelo", "repeticion", "pliegue"]


# 1. MÉTRICAS

def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """
    Calcula RMSE, MAE, R² y la pendiente de calibración.

    La pendiente de calibración es el coeficiente de la regresión lineal
    y_pred = a * y_true + b. Vale 1.0 si el modelo no tiene sesgo de escala;
    valores menores que 1 indican predicciones comprimidas (el modelo suaviza
    los extremos) y mayores que 1, predicciones exageradas.

    Parámetros:

    y_true, y_pred : array-like
        Valores reales y predichos, en dB, con la misma longitud.

    Retorna:
    rmse_db, mae_db, r2 y pendiente.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true y y_pred deben tener la misma forma; "
        )
    if y_true.size < 2:
        raise ValueError("Se requieren al menos 2 observaciones para calcular las métricas.")
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("y_true y y_pred no pueden contener NaN ni infinitos.")

    error = y_pred - y_true
    varianza_real = float(np.var(y_true, ddof=1))

    # Pendiente OLS: cov(y_true, y_pred) / var(y_true)
    pendiente = (
        float(np.cov(y_true, y_pred, ddof=1)[0, 1] / varianza_real)
        if varianza_real > 0
        else float("nan")
    )

    return {
        "rmse_db": float(np.sqrt(np.mean(error**2))),
        "mae_db": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "pendiente": pendiente,
    }


def compute_metrics_by_fold(df_preds: pd.DataFrame, by=None) -> pd.DataFrame:
    """
    Calcula las métricas de cada pliegue a partir del DataFrame oficial de
    predicciones (columnas 'y_true_db' y 'y_pred_db').

    Parámetros:
    df_preds : DataFrame
        Predicciones en el formato de src.config.REQUIRED_PREDICTION_COLS.
    by : str | secuencia de str, opcional
        Columnas que identifican cada pliegue. Por defecto usa las columnas de
        identificación presentes: familia, modelo, repeticion y pliegue.

    Retorna:
    DataFrame
        Una fila por pliegue con las columnas de identificación seguidas de
        METRIC_COLS.
    """
    faltantes = {"y_true_db", "y_pred_db"} - set(df_preds.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas de predicciones: {sorted(faltantes)}")

    if by is None:
        by = [col for col in _PREDICTION_ID_COLS if col in df_preds.columns]
    elif isinstance(by, str):
        by = [by]
    else:
        by = list(by)
    if not by:
        raise ValueError("Indique al menos una columna en 'by' para agrupar las predicciones.")
    faltantes_by = [col for col in by if col not in df_preds.columns]
    if faltantes_by:
        raise ValueError(f"Columnas de agrupación inexistentes: {faltantes_by}.")

    filas = []
    for claves, grupo in df_preds.groupby(by, sort=False):
        claves = claves if isinstance(claves, tuple) else (claves,)
        fila = dict(zip(by, claves))
        fila.update(compute_metrics(grupo["y_true_db"], grupo["y_pred_db"]))
        filas.append(fila)

    return pd.DataFrame(filas, columns=by + METRIC_COLS)


# 2. FIGURAS

def plot_rmse_by_model(
    df,
    ax=None,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """
    Grafica el RMSE de cada modelo.

    Acepta la tabla por pliegue devuelta por compute_metrics_by_fold o, por
    comodidad, el DataFrame de predicciones (en ese caso calcula las métricas
    primero). Las barras muestran la media y los bigotes ±1 desviación estándar
    entre pliegues; los puntos negros son el RMSE de cada pliegue individual.

    Parámetros
    df : DataFrame
        Tabla con columnas 'modelo' y 'rmse_db', o predicciones con 'y_true_db'
        y 'y_pred_db'.
    ax : matplotlib.axes.Axes, opcional
        Eje donde dibujar. Si se omite, se crea una figura nueva.
    save_path : str | Path, opcional
        Ruta donde guardar la figura (p. ej. results/figures/rmse_modelos.png).
    show : bool, por defecto False
        Si es True, muestra la figura al terminar.

    """
    if {"y_true_db", "y_pred_db"} <= set(df.columns):
        df = compute_metrics_by_fold(df)

    faltantes = {"modelo", "rmse_db"} - set(df.columns)
    if faltantes:
        raise ValueError(
            "Se requieren las columnas 'modelo' y 'rmse_db' (o el DataFrame de "
            f"predicciones). Faltan: {sorted(faltantes)}."
        )

    resumen = (
        df.groupby("modelo", sort=False)["rmse_db"]
        .agg(media="mean", desviacion="std")
        .fillna(0.0)
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    else:
        fig = ax.figure

    posiciones = np.arange(len(resumen))
    colores = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#4C72B0"])
    ax.bar(
        posiciones,
        resumen["media"],
        width=0.6,
        yerr=resumen["desviacion"],
        capsize=5,
        color=[colores[i % len(colores)] for i in range(len(resumen))],
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
        label="Media ± desv. est.",
        zorder=2,
    )

    # RMSE de cada pliegue, ligeramente desplazado para evitar solapamientos.
    rng = np.random.default_rng(0)
    for i, (_, grupo) in enumerate(df.groupby("modelo", sort=False)):
        ax.scatter(
            i + rng.uniform(-0.12, 0.12, len(grupo)),
            grupo["rmse_db"],
            s=16,
            color="black",
            alpha=0.5,
            zorder=3,
            label="Pliegues" if i == 0 else None,
        )

    for i, (media, desviacion) in enumerate(
        zip(resumen["media"], resumen["desviacion"])
    ):
        ax.annotate(
            f"{media:.3f}",
            (i, media + desviacion),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=9,
        )

    ax.set_xticks(posiciones, resumen.index)
    ax.set_xlim(-0.75, len(resumen) - 0.25)
    ax.set_xlabel("Modelo")
    ax.set_ylabel("RMSE [dB]")
    ax.set_title("RMSE por modelo (media ± desviación entre pliegues)")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)
    ax.margins(y=0.15)
    ax.legend(frameon=False, loc="upper left")

    return _save_figure(fig, save_path, show)


def plot_pred_vs_real(
    df_preds: pd.DataFrame,
    modelo: str | None = None,
    ax=None,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """
    Grafica los valores predichos contra los reales para un modelo.

    Incluye la recta ideal 1:1, el ajuste de calibración y un cuadro con RMSE,
    MAE, R² y pendiente de calibración.
    """
    faltantes = {"y_true_db", "y_pred_db"} - set(df_preds.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas de predicciones: {sorted(faltantes)}")

    if modelo is not None:
        if "modelo" not in df_preds.columns:
            raise ValueError("No se puede filtrar por 'modelo': la columna no existe.")
        df = df_preds[df_preds["modelo"] == modelo]
        if df.empty:
            disponibles = sorted(df_preds["modelo"].unique())
            raise ValueError(f"No hay predicciones para {modelo!r}. Disponibles: {disponibles}.")
    else:
        if "modelo" in df_preds.columns and df_preds["modelo"].nunique() > 1:
            disponibles = sorted(df_preds["modelo"].unique())
            raise ValueError(
                "El DataFrame contiene varios modelos; indique 'modelo'. "
                f"Disponibles: {disponibles}."
            )
        df = df_preds

    y_true = df["y_true_db"].to_numpy(dtype=float)
    y_pred = df["y_pred_db"].to_numpy(dtype=float)
    metricas = compute_metrics(y_true, y_pred)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    else:
        fig = ax.figure

    ax.scatter(
        y_true,
        y_pred,
        s=14,
        alpha=0.35,
        edgecolors="none",
        color="#4C72B0",
        zorder=2,
        label="Observaciones",
    )

    minimo = min(y_true.min(), y_pred.min())
    maximo = max(y_true.max(), y_pred.max())
    margen = 0.03 * (maximo - minimo) if maximo > minimo else 1.0
    limite = (minimo - margen, maximo + margen)

    ax.plot(limite, limite, "--", color="black", linewidth=1.2, label="Ideal (1:1)")

    pendiente = metricas["pendiente"]
    if np.isfinite(pendiente):
        intercepto = float(y_pred.mean() - pendiente * y_true.mean())
        ax.plot(
            np.asarray(limite),
            pendiente * np.asarray(limite) + intercepto,
            ":",
            color="#C44E52",
            linewidth=1.5,
            label=f"Ajuste (pendiente = {pendiente:.3f})",
        )

    ax.set_xlim(limite)
    ax.set_ylim(limite)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("SPL real [dB]")
    ax.set_ylabel("SPL predicho [dB]")
    ax.set_title(
        f"Predicho vs real — {modelo}" if modelo is not None else "Predicho vs real"
    )
    ax.grid(linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)

    texto = (
        f"RMSE = {metricas['rmse_db']:.3f} dB\n"
        f"MAE = {metricas['mae_db']:.3f} dB\n"
        f"R² = {metricas['r2']:.4f}\n"
        f"Pendiente = {metricas['pendiente']:.3f}\n"
        f"n = {len(y_true)}"
    )
    ax.text(
        0.03,
        0.97,
        texto,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )
    ax.legend(frameon=False, loc="lower right")

    return _save_figure(fig, save_path, show)


def _save_figure(fig, save_path: str | Path | None, show: bool):
    """Guarda y/o muestra la figura; siempre devuelve el objeto Figure."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Figura guardada en: {save_path}")
    if show:
        plt.show()
    return fig
# %%
