# %%
"""
Análisis Exploratorio de Datos (EDA) - Responsabilidad P2.
Ejecutar en VS Code Interactive Window.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.append(str(Path.cwd().parent)) # Para importar src si estás en la carpeta notebooks

from src.config import CONFIG_GROUP_COLS, FEATURE_COLS, TARGET_COL
from src.io_utils import load_or_download_airfoil_data

# Configuración visual
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Cargar datos validados por P1
df = load_or_download_airfoil_data()

# %% [1] AGRUPACIÓN Y CONFIGURACIONES ÚNICAS
# ¿Cuántas configuraciones físicas únicas existen en el túnel de viento?
df["grupo_configuracion"] = df.groupby(CONFIG_GROUP_COLS).ngroup()
n_configs = df["grupo_configuracion"].nunique()

print(f"Total de observaciones: {len(df)}")
print(f"Configuraciones físicas únicas (Cuerda, Velocidad, Ángulo): {n_configs}")
print("Ejemplo de frecuencias por configuración:")
print(df.groupby(CONFIG_GROUP_COLS).size().head())

# %% [2] RELACIÓN FRECUENCIA VS RUIDO (Curvatura)
# Muestra la forma de campana y justifica por qué la familia no lineal podría ganar
plt.figure()
sns.scatterplot(data=df, x="frecuencia_hz", y=TARGET_COL, alpha=0.5, hue="velocidad_flujo_ms")
plt.xscale("log")
plt.title("SPL vs Frecuencia (Escala Logarítmica)")
plt.xlabel("Frecuencia (Hz) [Log Scale]")
plt.ylabel("SPL (dB)")
plt.show()

# %% [3] ANÁLISIS DE MULTICOLINEALIDAD (VIF)
# Crítico para justificar la regularización (Ridge) en P3
X_vif = df[FEATURE_COLS].copy()
# Añadir constante para calcular VIF correctamente
X_vif.insert(0, "Constante", 1.0)

vif_data = pd.DataFrame()
vif_data["Variable"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print("\nFactores de Inflación de Varianza (VIF):")
print(vif_data[vif_data["Variable"] != "Constante"].sort_values("VIF", ascending=False))

# %% [4] RESIDUOS DE UN MODELO LINEAL SIMPLE
# Evaluar si una regresión sin transformaciones capta la señal adecuadamente
X_simple = df[FEATURE_COLS]
y_simple = df[TARGET_COL]

modelo_base = LinearRegression().fit(X_simple, y_simple)
residuos = y_simple - modelo_base.predict(X_simple)

plt.figure()
sns.scatterplot(x=y_simple, y=residuos, alpha=0.4)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuos del Modelo Lineal Simple vs Valor Real")
plt.xlabel("SPL Real (dB)")
plt.ylabel("Residuo (dB)")
plt.show()