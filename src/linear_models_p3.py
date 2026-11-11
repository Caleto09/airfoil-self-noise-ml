# Módulo reservado para P3: Modelos de la familia lineal

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

# Importamos las funciones que construyó tu compañero P2
from src.preprocessing_p2 import assign_configuration_groups, run_nested_cv_search

class TunedRandomForest(BaseEstimator, RegressorMixin):
    """
    Wrapper orientado a objetos que encapsula la búsqueda de hiperparámetros.
    Permite que P1 llame a .fit(X, y) de forma estándar, mientras internamente
    calculamos los grupos y hacemos la validación cruzada anidada usando la 
    herramienta de P2.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_estimator_ = None
        self.best_params_ = None
        
    def fit(self, X, y):
        groups = assign_configuration_groups(X)
        
        # 1. Definimos el pipeline base (Datos crudos directos al Random Forest)
        base_pipeline = Pipeline([
            ('rf', RandomForestRegressor(random_state=self.random_state))
        ])
        
        # 2. Definimos la malla de búsqueda. 
        param_grid = {
            'rf__n_estimators': [100, 200, 300],
            'rf__max_depth': [10, 15, None],
            'rf__max_features': [2, 3, 4], 
            'rf__min_samples_split': [2, 5],
            'rf__min_samples_leaf': [1, 2]
        }
        
        # 3. Invocamos la utilidad de P2 para la CV interna con agrupamiento
        self.best_estimator_, self.best_params_ = run_nested_cv_search(
            pipeline=base_pipeline,
            param_grid=param_grid,
            X_train=X,
            y_train=y,
            groups_train=groups,
            n_splits=5
        )
        
        # Imprimir qué hiperparámetros ganaron en este pliegue externo
        print(f"   [RF Tuning] Mejores params para este pliegue: {self.best_params_}")
        
        return self
        
    def predict(self, X):
        if self.best_estimator_ is None:
            raise ValueError("El modelo debe ser entrenado con .fit() antes de predecir.")
        # Usamos el modelo ganador encontrado en la CV interna
        return self.best_estimator_.predict(X)