# %%
# init_project.py
"""
Script de inicialización de estructura de directorios para P1.
Crea las carpetas y archivos base del proyecto Airfoil Self-Noise.
"""
from pathlib import Path

DIRS = [
    "data/raw",
    "data/processed",
    "src",
    "results/predictions",
    "results/tables",
    "results/figures",
    "notebooks",
]

FILES = {
    "src/__init__.py": "",
    "src/preprocessing_p2.py": "# Módulo reservado para P2: Pipeline y particiones agrupadas/repetidas\n",
    "src/linear_models_p3.py": "# Módulo reservado para P3: Modelos de la familia lineal\n",
    "src/nonlinear_models_p4.py": "# Módulo reservado para P4: Modelos de la familia no lineal\n",
    "src/metrics_stats_p5.py": "# Módulo reservado para P5: Métricas (RMSE, MAE, R2, Sesgo, Cuantil 90) y estadística\n",
    "requirements.txt": "numpy>=1.24.0\npandas>=2.0.0\nscikit-learn>=1.3.0\nscipy>=1.10.0\nmatplotlib>=3.7.0\n",
}

def main():
    for d in DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"[OK] Directorio creado/verificado: {d}/")

    for filepath, content in FILES.items():
        p = Path(filepath)
        if not p.exists():
            p.write_text(content, encoding="utf-8")
            print(f"[OK] Archivo base creado: {filepath}")
        else:
            print(f"[SKIP] El archivo ya existe: {filepath}")

if __name__ == "__main__":
    main()
# %%
